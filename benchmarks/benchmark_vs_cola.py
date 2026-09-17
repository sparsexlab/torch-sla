#!/usr/bin/env python3
"""Head-to-head benchmark: torch-sla vs CoLA on a differentiable SPD solve.

Addresses the reviewer request for a like-for-like comparison against
CoLA (Potapczynski et al., NeurIPS 2023) on a common SPD system, measuring
*forward + backward time and peak memory at a fixed accuracy*, with the
gradient taken w.r.t. the matrix values.

Both paths solve the same 2D Poisson SPD system A x = b with CG, form the
scalar loss L = ||x||^2, and backpropagate to obtain dL/d(val). We report,
per problem size: forward solve time, backward time, peak GPU memory, the
final relative residual ||Ax-b||/||b||, and the gradient agreement
||g_sla - g_cola|| / ||g_cola|| between the two libraries (a cross-check
that both compute the same adjoint gradient).

IMPORTANT — which CoLA operator we use, and why:
    CoLA 0.0.7 ships a ``cola.ops.Sparse`` (COO) operator, but on the PyTorch
    backend with current PyTorch it (a) builds an incorrect matrix for our
    systems -- ``A @ x`` is wrong for non-constant ``x`` -- so CG diverges,
    and (b) is not autograd-compatible: the backward pass raises
    "Sparse CSR tensors do not have strides" inside functorch's ``vjp``.
    We verified both on CPU at tiny sizes. We therefore benchmark against
    CoLA's *dense* differentiable solve (``cola.ops.Dense``), which works
    correctly and matches torch-sla's gradient to machine precision. The
    consequence is the expected one: CoLA's dense O(N^2) memory caps the
    problem size it can reach, whereas torch-sla's sparse adjoint scales to
    N where a dense operator is infeasible. Set ``--cola-op sparse`` to
    reproduce the Sparse-path failure.

Install CoLA first (benchmark-only dependency):
    pip install cola-ml

Run:
    python benchmarks/benchmark_vs_cola.py --sizes 64 128 256 512 1024
    # grid side n -> DOF = n^2; CoLA(dense) runs at the small end and is
    # skipped once a dense N x N matrix would exceed --max-cola-dense-dof.
"""

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from torch_sla import SparseTensor

warnings.filterwarnings("ignore", message="PCG did not converge")
warnings.filterwarnings("ignore", message="Sparse CSR tensor support")


# ----------------------------------------------------------------------------
# Problem assembly (identical SPD 2D Poisson as the other benchmarks)
# ----------------------------------------------------------------------------
def make_poisson_2d(n_grid, device, dtype):
    """2D Poisson 5-point stencil. Returns COO (val, row, col), shape, RHS b
    (chosen so x_true = 1), and a coalesced torch sparse matrix for residual
    checks."""
    N = n_grid * n_grid
    diag = 4.0 * np.ones(N)
    off = -np.ones(N - 1)
    off[np.arange(1, N) % n_grid == 0] = 0.0
    A = sp.diags(
        [diag, off, off, -np.ones(N - n_grid), -np.ones(N - n_grid)],
        [0, -1, 1, -n_grid, n_grid],
        format="coo",
    )
    row = torch.tensor(A.row, dtype=torch.long, device=device)
    col = torch.tensor(A.col, dtype=torch.long, device=device)
    val0 = torch.tensor(A.data, dtype=dtype, device=device)
    x_true = torch.ones(N, dtype=dtype, device=device)
    A_torch = torch.sparse_coo_tensor(
        torch.stack([row, col]), val0, (N, N)).coalesce()
    b = torch.sparse.mm(A_torch, x_true.unsqueeze(1)).squeeze(1)
    return dict(N=N, row=row, col=col, val0=val0, b=b, A_torch=A_torch)


def residual_norm(A_torch, x, b):
    r = b - torch.sparse.mm(A_torch, x.detach().unsqueeze(1)).squeeze(1)
    return float(r.norm() / b.norm())


def reset_cuda(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _time_fwd_bwd(forward, device):
    """Run forward(), then backward on L=||x||^2; return (x, t_fwd, t_bwd)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    x = forward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_fwd = (time.perf_counter() - t0) * 1000

    loss = x.pow(2).sum()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_bwd = (time.perf_counter() - t0) * 1000
    return x, t_fwd, t_bwd


# ----------------------------------------------------------------------------
# torch-sla forward+backward (sparse)
# ----------------------------------------------------------------------------
def run_torch_sla(P, device, atol, maxiter, warmup=1):
    def make_fwd(v):
        def fwd():
            A = SparseTensor(v, P["row"], P["col"], shape=(P["N"], P["N"]))
            return A.solve(P["b"], backend="pytorch", method="cg",
                           atol=atol, maxiter=maxiter)
        return fwd

    for _ in range(warmup):  # untimed: amortize first-call / JIT overhead
        vw = P["val0"].clone().requires_grad_(True)
        make_fwd(vw)().pow(2).sum().backward()

    val = P["val0"].clone().requires_grad_(True)
    reset_cuda(device)
    x, t_fwd, t_bwd = _time_fwd_bwd(make_fwd(val), device)
    peak = (torch.cuda.max_memory_allocated(device) / 1024 / 1024
            if device.type == "cuda" else 0.0)
    return dict(t_fwd=t_fwd, t_bwd=t_bwd, peak_mb=peak,
                residual=residual_norm(P["A_torch"], x, P["b"]),
                grad=val.grad.detach().clone())


# ----------------------------------------------------------------------------
# CoLA forward+backward (dense operator built differentiably from `val`)
# ----------------------------------------------------------------------------
def run_cola(P, device, atol, maxiter, op="dense", warmup=1):
    import cola

    N = P["N"]

    def make_fwd(v):  # v is the shared leaf -> gradient comparable to torch-sla
        if op == "dense":
            def fwd():
                M = torch.zeros(N, N, dtype=v.dtype, device=device)
                M = M.index_put((P["row"], P["col"]), v)  # differentiable in v
                A = cola.PSD(cola.ops.Dense(M))
                return cola.linalg.solve(
                    A, P["b"], alg=cola.linalg.CG(tol=atol, max_iters=maxiter))
        else:  # "sparse" -- reproduces the CoLA Sparse-path failure (see header)
            def fwd():
                A = cola.PSD(cola.ops.Sparse(v, P["row"], P["col"], (N, N)))
                return cola.linalg.solve(
                    A, P["b"], alg=cola.linalg.CG(tol=atol, max_iters=maxiter))
        return fwd

    for _ in range(warmup):  # untimed
        vw = P["val0"].clone().requires_grad_(True)
        make_fwd(vw)().pow(2).sum().backward()

    val = P["val0"].clone().requires_grad_(True)
    reset_cuda(device)
    x, t_fwd, t_bwd = _time_fwd_bwd(make_fwd(val), device)
    peak = (torch.cuda.max_memory_allocated(device) / 1024 / 1024
            if device.type == "cuda" else 0.0)
    return dict(t_fwd=t_fwd, t_bwd=t_bwd, peak_mb=peak,
                residual=residual_norm(P["A_torch"], x, P["b"]),
                grad=val.grad.detach().clone())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024],
                        help="Grid side n; DOF = n^2.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--maxiter", type=int, default=50000)
    parser.add_argument("--cola-op", type=str, default="dense",
                        choices=["dense", "sparse"],
                        help="CoLA operator type. 'dense' is the working path; "
                             "'sparse' reproduces the CoLA Sparse failure.")
    parser.add_argument("--max-cola-dense-dof", type=int, default=90_000,
                        help="Skip CoLA dense above this DOF (an N x N float64 "
                             "matrix needs 8*N^2 bytes; 90k -> ~65 GB).")
    parser.add_argument("--out", type=str, default="results/benchmark_vs_cola")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    have_cola = False
    try:
        import cola
        have_cola = True
        print(f"[setup] cola {getattr(cola, '__version__', '?')}  "
              f"(CoLA operator: {args.cola_op})", flush=True)
    except ImportError:
        print("[setup] CoLA not installed. Install with: pip install cola-ml\n"
              "        (torch-sla numbers will still be recorded.)", flush=True)

    if device.type == "cuda":
        print(f"[setup] device: {torch.cuda.get_device_name(device)}", flush=True)

    rows = []
    for n in args.sizes:
        P = make_poisson_2d(n, device, dtype)
        N = P["N"]
        print(f"\n=== n={n}  DOF={N:,}  nnz={P['val0'].numel():,} ===", flush=True)
        rec = {"n_grid": n, "dof": N, "nnz": int(P["val0"].numel())}

        # ---- torch-sla (sparse) ----
        try:
            s = run_torch_sla(P, device, args.atol, args.maxiter)
            rec["torch_sla"] = {k: v for k, v in s.items() if k != "grad"}
            print(f"  torch-sla    : fwd={s['t_fwd']:.1f} ms  bwd={s['t_bwd']:.1f} ms"
                  f"  mem={s['peak_mb']:.1f} MB  res={s['residual']:.2e}", flush=True)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            rec["torch_sla"] = {"error": str(e)[:120]}
            s = None
            print(f"  torch-sla    : FAILED ({str(e)[:80]})", flush=True)
        reset_cuda(device)

        # ---- CoLA (dense) ----
        if have_cola and N <= args.max_cola_dense_dof:
            try:
                c = run_cola(P, device, args.atol, args.maxiter, op=args.cola_op)
                rec["cola"] = {k: v for k, v in c.items() if k != "grad"}
                print(f"  cola ({args.cola_op}) : fwd={c['t_fwd']:.1f} ms  "
                      f"bwd={c['t_bwd']:.1f} ms  mem={c['peak_mb']:.1f} MB  "
                      f"res={c['residual']:.2e}", flush=True)
                if s is not None and "error" not in rec["torch_sla"]:
                    g_rel = float((s["grad"] - c["grad"]).norm() /
                                  (c["grad"].norm() + 1e-30))
                    rec["grad_rel_diff"] = g_rel
                    print(f"  grad agreement ||g_sla - g_cola||/||g_cola|| = "
                          f"{g_rel:.2e}", flush=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                rec["cola"] = {"error": str(e)[:160]}
                print(f"  cola ({args.cola_op}) : FAILED "
                      f"({type(e).__name__}: {str(e)[:100]})", flush=True)
        elif have_cola:
            rec["cola"] = {"error": f"skipped (dense > {args.max_cola_dense_dof} DOF)"}
            print(f"  cola ({args.cola_op}) : skipped (dense matrix too large)",
                  flush=True)

        rows.append(rec)
        with open(out_dir / "results.json", "w") as fh:
            json.dump({"rows": rows, "atol": args.atol,
                       "cola_op": args.cola_op}, fh, indent=2)
        del P
        reset_cuda(device)

    print(f"\n[done] -> {out_dir/'results.json'}", flush=True)

    # ------------------------------------------------------------------ plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def series(lib, field, field2=None):
            xs, ys = [], []
            for r in rows:
                d = r.get(lib, {})
                if "error" not in d and field in d:
                    xs.append(r["dof"])
                    ys.append(d[field] + (d[field2] if field2 else 0.0))
            o = np.argsort(xs)
            return [xs[i] for i in o], [ys[i] for i in o]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        for lib, color, mk, lab in [("torch_sla", "#d62728", "^", "torch-sla (sparse)"),
                                    ("cola", "#1f77b4", "o", "CoLA (dense)")]:
            xs, yt = series(lib, "t_fwd", "t_bwd")
            if xs:
                axes[0].loglog(xs, yt, marker=mk, color=color, label=lab,
                               markersize=7, linewidth=1.7)
            xm, ym = series(lib, "peak_mb")
            if xm:
                axes[1].loglog(xm, ym, marker=mk, color=color, label=lab,
                               markersize=7, linewidth=1.7)
        axes[0].set(xlabel="Degrees of freedom $N$",
                    ylabel="Forward + backward time (ms)",
                    title="Differentiable SPD solve: time")
        axes[1].set(xlabel="Degrees of freedom $N$",
                    ylabel="Peak GPU memory (MB)",
                    title="Differentiable SPD solve: memory")
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "vs_cola.png", dpi=150)
        fig.savefig(out_dir / "vs_cola.pdf")
        print(f"[plot] -> {out_dir/'vs_cola.png'}", flush=True)
    except ImportError:
        print("[plot] matplotlib not available", flush=True)


if __name__ == "__main__":
    main()
