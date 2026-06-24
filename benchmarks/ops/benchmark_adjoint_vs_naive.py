#!/usr/bin/env python3
"""Adjoint vs naive backpropagation through CG iterations.

Compares two paths through the same PyTorch-native CG forward kernel:
  - naive : run k CG iterations under autograd; backward traces through all of them
  - adjoint: run k CG iterations under torch.no_grad inside a custom
            autograd.Function; backward executes one explicit adjoint solve

Records, for each k:
  peak GPU memory, forward time, backward time, gradient relative error
  (between the two methods at the largest converged k).

Outputs JSON + PNG figure under results/benchmark_adjoint_vs_naive/.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torch_sla import spsolve

warnings.filterwarnings("ignore", message="PCG did not converge")
warnings.filterwarnings("ignore", message="Sparse CSR tensor support")


def make_poisson_2d(n_grid, device, dtype):
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
    return val, row, col, (N, N)


def spmv_scatter(val, row, col, x, n):
    """SpMV via scatter-add: y[i] = sum_k val[k] * x[col[k]] where row[k] = i.

    Unlike ``torch.sparse.mm``, the autograd gradient w.r.t. ``val`` here
    stays sparse (one entry per non-zero), which is essential for the
    naive baseline to even fit in memory.
    """
    contrib = val * x[col]
    y = torch.zeros(n, dtype=x.dtype, device=x.device)
    y.index_add_(0, row, contrib)
    return y


def naive_cg(val, row, col, shape, b, k, atol=0.0):
    """Vanilla CG (no preconditioner), fully autograd-tracked.

    Runs at most ``k`` iterations. Stops early if the residual norm
    drops below ``atol`` (or becomes non-finite, to avoid 0/0 NaN once
    fully converged in exact arithmetic).
    """
    n = shape[0]
    x = torch.zeros_like(b)
    r = b - spmv_scatter(val, row, col, x, n)
    p = r.clone()
    rs = torch.dot(r, r)
    eps = torch.finfo(r.dtype).tiny
    for _ in range(k):
        rs_val = float(rs.detach())
        if not (rs_val > eps) or rs_val ** 0.5 < atol:
            break
        Ap = spmv_scatter(val, row, col, p, n)
        alpha = rs / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        beta = rs_new / rs
        p = r + beta * p
        rs = rs_new
    return x


def reset_mem(device):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def run_naive(val0, row, col, shape, b0, k, device, atol=0.0):
    reset_mem(device)
    val = val0.detach().clone().requires_grad_(True)
    b = b0.detach().clone().requires_grad_(True)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    x = naive_cg(val, row, col, shape, b, k, atol=atol)
    loss = x.pow(2).sum()
    torch.cuda.synchronize(device)
    t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    t_bwd = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return {
        "peak_mem_MB": peak,
        "fwd_ms": t_fwd * 1000,
        "bwd_ms": t_bwd * 1000,
        "loss": float(loss.detach()),
        "grad_val": val.grad.detach().cpu(),
        "grad_b": b.grad.detach().cpu(),
    }


def run_adjoint(val0, row, col, shape, b0, k, device, atol=0.0):
    reset_mem(device)
    val = val0.detach().clone().requires_grad_(True)
    b = b0.detach().clone().requires_grad_(True)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    x = spsolve(
        val, row, col, shape, b,
        backend="pytorch", method="cg",
        preconditioner="none",
        atol=atol, maxiter=k,
    )
    loss = x.pow(2).sum()
    torch.cuda.synchronize(device)
    t_fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    t_bwd = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return {
        "peak_mem_MB": peak,
        "fwd_ms": t_fwd * 1000,
        "bwd_ms": t_bwd * 1000,
        "loss": float(loss.detach()),
        "grad_val": val.grad.detach().cpu(),
        "grad_b": b.grad.detach().cpu(),
    }


def rel_err(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-grid", type=int, default=800,
                        help="Poisson grid side length; n = n_grid^2")
    parser.add_argument("--ks", type=int, nargs="+",
                        default=[10, 50, 100, 500, 1000, 2000, 5000])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--out", type=str,
                        default="results/benchmark_adjoint_vs_naive")
    parser.add_argument("--correctness-k", type=int, default=0,
                        help="If >0, also run a separate well-converged "
                             "correctness check at this iteration count "
                             "on a smaller grid (n_grid_check x n_grid_check).")
    parser.add_argument("--n-grid-check", type=int, default=64,
                        help="Grid for the correctness check.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] device={device} dtype={dtype} n_grid={args.n_grid} "
          f"(N={args.n_grid**2}) ks={args.ks}", flush=True)

    val, row, col, shape = make_poisson_2d(args.n_grid, device, dtype)
    nnz = val.numel()
    N = shape[0]
    print(f"[setup] N={N} nnz={nnz}", flush=True)

    # Use a fixed RHS (consistent across runs)
    torch.manual_seed(0)
    b = torch.randn(N, device=device, dtype=dtype)

    results = []
    naive_alive = True

    for k in args.ks:
        record = {"k": k}
        print(f"\n[k={k}] ----------------------------------------", flush=True)

        # Adjoint always
        try:
            r = run_adjoint(val, row, col, shape, b, k, device)
            record["adjoint"] = {
                "peak_mem_MB": r["peak_mem_MB"],
                "fwd_ms": r["fwd_ms"],
                "bwd_ms": r["bwd_ms"],
                "loss": r["loss"],
            }
            print(f"  adjoint: mem={r['peak_mem_MB']:.1f} MB  "
                  f"fwd={r['fwd_ms']:.1f} ms  bwd={r['bwd_ms']:.1f} ms  "
                  f"loss={r['loss']:.4e}", flush=True)
            grad_val_adj = r["grad_val"]
            grad_b_adj = r["grad_b"]
        except torch.cuda.OutOfMemoryError as e:
            record["adjoint"] = {"error": "OOM"}
            print(f"  adjoint: OOM ({e})", flush=True)
            grad_val_adj = grad_b_adj = None

        # Naive only while it still fits
        if naive_alive:
            try:
                r = run_naive(val, row, col, shape, b, k, device)
                record["naive"] = {
                    "peak_mem_MB": r["peak_mem_MB"],
                    "fwd_ms": r["fwd_ms"],
                    "bwd_ms": r["bwd_ms"],
                    "loss": r["loss"],
                }
                print(f"  naive  : mem={r['peak_mem_MB']:.1f} MB  "
                      f"fwd={r['fwd_ms']:.1f} ms  bwd={r['bwd_ms']:.1f} ms  "
                      f"loss={r['loss']:.4e}", flush=True)
                if grad_val_adj is not None:
                    record["grad_val_rel_err"] = rel_err(r["grad_val"], grad_val_adj)
                    record["grad_b_rel_err"] = rel_err(r["grad_b"], grad_b_adj)
                    print(f"  grad rel_err: val={record['grad_val_rel_err']:.2e}  "
                          f"b={record['grad_b_rel_err']:.2e}", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                record["naive"] = {"error": "OOM"}
                print(f"  naive  : OOM at k={k} ({e})", flush=True)
                naive_alive = False
                reset_mem(device)
        else:
            record["naive"] = {"error": "skipped (prior OOM)"}
            print("  naive  : skipped (prior OOM)", flush=True)

        results.append(record)
        with open(out_dir / "results.json", "w") as f:
            json.dump(
                {
                    "config": {
                        "n_grid": args.n_grid,
                        "N": N,
                        "nnz": nnz,
                        "dtype": args.dtype,
                        "device": str(device),
                    },
                    "results": results,
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Optional: separate well-converged gradient correctness check
    # ------------------------------------------------------------------
    correctness = None
    if args.correctness_k > 0:
        ng = args.n_grid_check
        print(f"\n[correctness] n_grid={ng} (N={ng*ng}) k={args.correctness_k}",
              flush=True)
        v2, r2, c2, sh2 = make_poisson_2d(ng, device, dtype)
        torch.manual_seed(1)
        b2 = torch.randn(sh2[0], device=device, dtype=dtype)

        adj = run_adjoint(v2, r2, c2, sh2, b2, args.correctness_k, device,
                          atol=1e-12)
        nai = run_naive(v2, r2, c2, sh2, b2, args.correctness_k, device,
                        atol=1e-12)

        correctness = {
            "n_grid": ng,
            "N": sh2[0],
            "k": args.correctness_k,
            "adjoint_loss": adj["loss"],
            "naive_loss": nai["loss"],
            "loss_rel_err": abs(adj["loss"] - nai["loss"]) / abs(nai["loss"]),
            "grad_val_rel_err": rel_err(nai["grad_val"], adj["grad_val"]),
            "grad_b_rel_err": rel_err(nai["grad_b"], adj["grad_b"]),
        }
        print(f"  loss adj={adj['loss']:.6e}  naive={nai['loss']:.6e}  "
              f"rel_err={correctness['loss_rel_err']:.2e}", flush=True)
        print(f"  grad_val rel_err = {correctness['grad_val_rel_err']:.2e}",
              flush=True)
        print(f"  grad_b   rel_err = {correctness['grad_b_rel_err']:.2e}",
              flush=True)

        with open(out_dir / "results.json", "w") as f:
            json.dump(
                {
                    "config": {
                        "n_grid": args.n_grid,
                        "N": N,
                        "nnz": nnz,
                        "dtype": args.dtype,
                        "device": str(device),
                    },
                    "results": results,
                    "correctness": correctness,
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ks_a, mem_a, fwd_a, bwd_a = [], [], [], []
        ks_n, mem_n, fwd_n, bwd_n = [], [], [], []
        for r in results:
            if "error" not in r["adjoint"]:
                ks_a.append(r["k"])
                mem_a.append(r["adjoint"]["peak_mem_MB"])
                fwd_a.append(r["adjoint"]["fwd_ms"])
                bwd_a.append(r["adjoint"]["bwd_ms"])
            if "error" not in r["naive"]:
                ks_n.append(r["k"])
                mem_n.append(r["naive"]["peak_mem_MB"])
                fwd_n.append(r["naive"]["fwd_ms"])
                bwd_n.append(r["naive"]["bwd_ms"])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        ax = axes[0]
        ax.loglog(ks_n, mem_n, "o-", color="#d62728", label="naive (autograd through CG)")
        ax.loglog(ks_a, mem_a, "s-", color="#1f77b4", label="adjoint (torch-sla)")
        if ks_n and len(ks_n) < len(ks_a):
            ax.axvline(ks_n[-1], ls=":", color="gray", alpha=0.7)
            ax.text(ks_n[-1] * 1.1, max(mem_a) * 1.5,
                    f"naive OOM\n@ k>{ks_n[-1]}",
                    fontsize=9, color="gray")
        ax.set_xlabel("CG iterations $k$")
        ax.set_ylabel("Peak GPU memory (MB)")
        ax.set_title(f"Memory vs iterations  (N = {N:,}, nnz = {nnz:,})")
        ax.legend(loc="best")
        ax.grid(True, which="both", alpha=0.3)

        ax = axes[1]
        ax.loglog(ks_n, bwd_n, "o-", color="#d62728", label="naive backward")
        ax.loglog(ks_a, bwd_a, "s-", color="#1f77b4", label="adjoint backward")
        ax.set_xlabel("CG iterations $k$")
        ax.set_ylabel("Backward time (ms)")
        ax.set_title("Backward time vs iterations")
        ax.legend(loc="best")
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_dir / "adjoint_vs_naive.png", dpi=150)
        fig.savefig(out_dir / "adjoint_vs_naive.pdf")
        print(f"[plot] -> {out_dir/'adjoint_vs_naive.png'}", flush=True)
    except ImportError:
        print("[plot] matplotlib not available; skipping figure", flush=True)

    print(f"\n[done] results -> {out_dir/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
