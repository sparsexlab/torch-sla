#!/usr/bin/env python3
"""Single-GPU scalability benchmark.

For each problem size, runs three solver backends through the same
torch-sla entry point ``spsolve``:
  - scipy  (CPU, SuperLU direct)
  - cudss  (CUDA, Cholesky direct)
  - pytorch (CUDA, CG with Jacobi preconditioning)

Records wall time, peak GPU memory (where applicable), and final
residual. OOM / time-out failures are recorded so the table reflects
the regime crossover (small: direct wins; large: only CG fits).
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from torch_sla import spsolve

warnings.filterwarnings("ignore", message="PCG did not converge")
warnings.filterwarnings("ignore", message="Sparse CSR tensor support")


def make_poisson_2d(n_grid, device, dtype):
    """2D Poisson 5-point stencil; returns (val, row, col, shape, b)."""
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
    val = torch.tensor(A.data, dtype=dtype, device=device)
    # Choose b so the solution is the all-ones vector (lets us verify residual)
    x_true = torch.ones(N, dtype=dtype, device=device)
    A_torch = torch.sparse_coo_tensor(
        torch.stack([row, col]), val, (N, N)
    ).coalesce()
    b = torch.sparse.mm(A_torch, x_true.unsqueeze(1)).squeeze(1)
    return val, row, col, (N, N), b, A_torch


def residual_norm(A_torch, x, b):
    r = b - torch.sparse.mm(A_torch, x.unsqueeze(1)).squeeze(1)
    return float(r.norm() / b.norm())


def reset_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_one(backend, method, val, row, col, shape, b, A_torch,
                atol, rtol, maxiter, num_runs, device):
    """Returns dict with time_ms, memory_mb, residual. Raises on OOM."""
    # Warmup
    _ = spsolve(val, row, col, shape, b,
                backend=backend, method=method,
                atol=atol, tol=rtol, maxiter=maxiter)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    x = None
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        x = spsolve(val, row, col, shape, b,
                    backend=backend, method=method,
                    atol=atol, tol=rtol, maxiter=maxiter)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000)

    peak_mem = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda" else 0.0
    )

    # Move to device for residual computation if needed
    if x.device != A_torch.device:
        x_eval = x.to(A_torch.device)
    else:
        x_eval = x
    r = residual_norm(A_torch, x_eval, b)

    return {
        "time_ms_mean": float(np.mean(times)),
        "time_ms_min": float(np.min(times)),
        "time_ms_max": float(np.max(times)),
        "memory_mb": peak_mem,
        "residual": r,
    }


def main():
    parser = argparse.ArgumentParser()
    # Sizes match the paper's Table 4: 10K, 100K, 1M, 2M, 16M, 169M
    # plus 100M for a smoother CG curve.
    parser.add_argument(
        "--sizes", type=int, nargs="+",
        default=[100, 316, 1000, 1414, 4000, 10000, 13000],
        help="Grid side lengths (DOF = n^2). Defaults give "
             "10K, 100K, 1M, 2M, 16M, 100M, 169M.",
    )
    parser.add_argument("--cuda-device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--cg-atol", type=float, default=0.0)
    parser.add_argument("--cg-rtol", type=float, default=1e-6)
    parser.add_argument("--cg-maxiter", type=int, default=50000)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--max-direct-dof", type=int, default=2_500_000,
                        help="Skip cudss/scipy on problems above this DOF "
                             "(direct solvers OOM well before this).")
    parser.add_argument("--out", type=str,
                        default="results/benchmark_single_gpu_scaling")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu = torch.device("cpu")
    cuda = torch.device(args.cuda_device)
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]

    print(f"[setup] CUDA: {torch.cuda.get_device_name(cuda)}  "
          f"sizes (DOF) = {[n*n for n in args.sizes]}", flush=True)

    results_path = out_dir / "results.json"
    if results_path.exists():
        with open(results_path) as fh:
            all_results = json.load(fh)
        print(f"[setup] Resuming with {len(all_results['rows'])} prior rows",
              flush=True)
    else:
        all_results = {"rows": []}

    completed = {(r["dof"], r["backend"]) for r in all_results["rows"]}

    for n in args.sizes:
        N = n * n
        print(f"\n=== n_grid={n}  DOF={N:,} ===", flush=True)

        # Build CUDA copy once, plus CPU copy for scipy
        reset_cuda()
        try:
            val_g, row_g, col_g, shape, b_g, A_g = make_poisson_2d(n, cuda, dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"  [build] OOM allocating CUDA tensors at DOF={N:,}; skip",
                  flush=True)
            continue
        nnz = val_g.numel()

        # ---- pytorch CG (CUDA) ----
        if (N, "pytorch") not in completed:
            try:
                reset_cuda()
                m = measure_one(
                    "pytorch", "cg",
                    val_g, row_g, col_g, shape, b_g, A_g,
                    atol=args.cg_atol, rtol=args.cg_rtol,
                    maxiter=args.cg_maxiter, num_runs=args.num_runs,
                    device=cuda,
                )
                row = {"backend": "pytorch", "n_grid": n, "dof": N,
                       "nnz": nnz, **m}
                all_results["rows"].append(row)
                print(f"  pytorch CG : time={m['time_ms_mean']:.1f} ms  "
                      f"mem={m['memory_mb']:.1f} MB  res={m['residual']:.2e}",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                row = {"backend": "pytorch", "n_grid": n, "dof": N,
                       "nnz": nnz, "error": "OOM"}
                all_results["rows"].append(row)
                print(f"  pytorch CG : OOM", flush=True)

        # ---- cuDSS Cholesky (CUDA) ----
        if N <= args.max_direct_dof and (N, "cudss") not in completed:
            try:
                reset_cuda()
                m = measure_one(
                    "cudss", "cholesky",
                    val_g, row_g, col_g, shape, b_g, A_g,
                    atol=args.cg_atol, rtol=args.cg_rtol,
                    maxiter=args.cg_maxiter, num_runs=args.num_runs,
                    device=cuda,
                )
                row = {"backend": "cudss", "n_grid": n, "dof": N,
                       "nnz": nnz, **m}
                all_results["rows"].append(row)
                print(f"  cuDSS Chol : time={m['time_ms_mean']:.1f} ms  "
                      f"mem={m['memory_mb']:.1f} MB  res={m['residual']:.2e}",
                      flush=True)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                row = {"backend": "cudss", "n_grid": n, "dof": N,
                       "nnz": nnz, "error": str(e)[:120]}
                all_results["rows"].append(row)
                print(f"  cuDSS Chol : FAILED ({type(e).__name__}: "
                      f"{str(e)[:80]})", flush=True)
        elif N > args.max_direct_dof:
            row = {"backend": "cudss", "n_grid": n, "dof": N,
                   "nnz": nnz, "error": "skipped (size > max_direct_dof)"}
            all_results["rows"].append(row)
            print(f"  cuDSS Chol : skipped (DOF > {args.max_direct_dof:,})",
                  flush=True)

        # ---- SciPy SuperLU (CPU) ----
        if N <= args.max_direct_dof and (N, "scipy") not in completed:
            try:
                val_c = val_g.detach().cpu()
                row_c = row_g.detach().cpu()
                col_c = col_g.detach().cpu()
                b_c = b_g.detach().cpu()
                A_c = A_g.cpu()
                m = measure_one(
                    "scipy", "lu",
                    val_c, row_c, col_c, shape, b_c, A_c,
                    atol=args.cg_atol, rtol=args.cg_rtol,
                    maxiter=args.cg_maxiter, num_runs=max(1, args.num_runs - 1),
                    device=cpu,
                )
                row = {"backend": "scipy", "n_grid": n, "dof": N,
                       "nnz": nnz, **m}
                all_results["rows"].append(row)
                print(f"  scipy LU   : time={m['time_ms_mean']:.1f} ms  "
                      f"mem={m['memory_mb']:.1f} MB (host)  "
                      f"res={m['residual']:.2e}", flush=True)
                del val_c, row_c, col_c, b_c, A_c
            except MemoryError as e:
                row = {"backend": "scipy", "n_grid": n, "dof": N,
                       "nnz": nnz, "error": "MemoryError"}
                all_results["rows"].append(row)
                print(f"  scipy LU   : MemoryError", flush=True)
        elif N > args.max_direct_dof:
            row = {"backend": "scipy", "n_grid": n, "dof": N,
                   "nnz": nnz, "error": "skipped (size > max_direct_dof)"}
            all_results["rows"].append(row)
            print(f"  scipy LU   : skipped (DOF > {args.max_direct_dof:,})",
                  flush=True)

        # Save after each size
        with open(results_path, "w") as fh:
            json.dump(all_results, fh, indent=2)
        del val_g, row_g, col_g, b_g, A_g
        reset_cuda()

    print(f"\n[done] -> {results_path}", flush=True)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        backends = ["scipy", "cudss", "pytorch"]
        labels = {"scipy": "SciPy SuperLU (CPU)",
                  "cudss": "cuDSS Cholesky (GPU)",
                  "pytorch": "torch-sla CG (GPU)"}
        colors = {"scipy": "#1f77b4", "cudss": "#2ca02c",
                  "pytorch": "#d62728"}
        markers = {"scipy": "o", "cudss": "s", "pytorch": "^"}

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        # Time
        ax = axes[0]
        for bk in backends:
            xs, ys = [], []
            for r in all_results["rows"]:
                if r["backend"] == bk and "error" not in r:
                    xs.append(r["dof"])
                    ys.append(r["time_ms_mean"])
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            if xs:
                ax.loglog(xs, ys, marker=markers[bk], color=colors[bk],
                          label=labels[bk], markersize=7, linewidth=1.7)
        ax.set_xlabel("Degrees of freedom $N$")
        ax.set_ylabel("Solve time (ms)")
        ax.set_title("Solve time")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

        # Memory (only GPU backends)
        ax = axes[1]
        for bk in ("cudss", "pytorch"):
            xs, ys = [], []
            for r in all_results["rows"]:
                if r["backend"] == bk and "error" not in r and r["memory_mb"] > 0:
                    xs.append(r["dof"]); ys.append(r["memory_mb"])
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            if xs:
                ax.loglog(xs, ys, marker=markers[bk], color=colors[bk],
                          label=labels[bk], markersize=7, linewidth=1.7)
        ax.set_xlabel("Degrees of freedom $N$")
        ax.set_ylabel("Peak GPU memory (MB)")
        ax.set_title("GPU memory")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

        # Residual
        ax = axes[2]
        for bk in backends:
            xs, ys = [], []
            for r in all_results["rows"]:
                if r["backend"] == bk and "error" not in r:
                    xs.append(r["dof"])
                    ys.append(max(r["residual"], 1e-16))
            order = np.argsort(xs)
            xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
            if xs:
                ax.loglog(xs, ys, marker=markers[bk], color=colors[bk],
                          label=labels[bk], markersize=7, linewidth=1.7)
        ax.set_xlabel("Degrees of freedom $N$")
        ax.set_ylabel("Final residual $\\|Ax-b\\|/\\|b\\|$")
        ax.set_title("Residual")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(out_dir / "single_gpu_scaling.png", dpi=150)
        fig.savefig(out_dir / "single_gpu_scaling.pdf")
        print(f"[plot] -> {out_dir/'single_gpu_scaling.png'}", flush=True)
    except ImportError:
        print("[plot] matplotlib not available", flush=True)


if __name__ == "__main__":
    main()
