#!/usr/bin/env python
"""Time + peak-memory scaling of every public SparseTensor op vs nnz.

Produces a table with one row per (op, n) over a sweep of 5-point
Poisson 2D matrices (M = n*n; nnz ~ 5M). Numbers are *single-process,
single-thread CPU* unless ``--cuda`` is passed.
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch_sla import SparseTensor, DetConfig  # noqa: E402


def poisson_2d_coo(side: int, dtype=torch.float64):
    """5-point 2D Poisson Laplacian on a side x side grid."""
    n = side * side
    rows, cols, vals = [], [], []
    for i in range(side):
        for j in range(side):
            k = i * side + j
            rows.append(k); cols.append(k); vals.append(4.0)
            if j + 1 < side:
                rows.append(k); cols.append(k + 1); vals.append(-1.0)
                rows.append(k + 1); cols.append(k); vals.append(-1.0)
            if i + 1 < side:
                rows.append(k); cols.append(k + side); vals.append(-1.0)
                rows.append(k + side); cols.append(k); vals.append(-1.0)
    return (
        torch.tensor(vals, dtype=dtype),
        torch.tensor(rows, dtype=torch.int64),
        torch.tensor(cols, dtype=torch.int64),
        (n, n),
    )


def time_call(fn, *, reps: int = 3) -> float:
    """Median wall-clock time of fn() over ``reps`` repeats."""
    samples = []
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(sorted(samples)[len(samples) // 2])


def peak_mem_call(fn) -> float:
    """Peak Python-level allocation during fn() in MB (tracemalloc)."""
    gc.collect()
    tracemalloc.start()
    fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def benchmark_size(side: int, ops: list[str]) -> list[dict]:
    val, row, col, shape = poisson_2d_coo(side)
    A = SparseTensor(val, row, col, shape)
    n = side * side
    nnz = int(val.numel())
    print(f"\n=== n={n} (side={side}), nnz={nnz:,} ===", flush=True)

    rows = []

    if "matvec" in ops:
        x = torch.randn(n, dtype=torch.float64)
        # warm up the SparseTensor's local CSR cache (first matvec builds it)
        _ = A @ x
        t = time_call(lambda: (A @ x), reps=5)
        pm = peak_mem_call(lambda: (A @ x))
        rows.append({"op": "A @ x", "n": n, "nnz": nnz, "time_s": t, "peak_mb": pm})

    if "solve" in ops:
        b = torch.ones(n, dtype=torch.float64)
        t = time_call(lambda: A.solve(b))
        pm = peak_mem_call(lambda: A.solve(b))
        rows.append({"op": "A.solve(b)", "n": n, "nnz": nnz, "time_s": t, "peak_mb": pm})

    if "det" in ops:
        t = time_call(lambda: A.det(), reps=2)
        pm = peak_mem_call(lambda: A.det())
        rows.append({"op": "A.det()", "n": n, "nnz": nnz, "time_s": t, "peak_mb": pm})

    if "det_grad" in ops:
        v = A.values.detach().clone().requires_grad_(True)
        B = SparseTensor(v, A.row_indices, A.col_indices, shape=shape)
        def grad():
            B.values.grad = None
            B.det().backward()
        t = time_call(grad, reps=2)
        pm = peak_mem_call(grad)
        rows.append({"op": "A.det() backward", "n": n, "nnz": nnz, "time_s": t, "peak_mb": pm})

    if "logdet_hutch" in ops:
        with DetConfig(method="hutchinson", num_probes=20, lanczos_iter=30):
            t = time_call(lambda: A.logdet(), reps=2)
            pm = peak_mem_call(lambda: A.logdet())
        rows.append({"op": "A.logdet() Hutchinson", "n": n, "nnz": nnz, "time_s": t, "peak_mb": pm})

    if "eigsh" in ops:
        # Now backed by ARPACK Lanczos for n > 1024 (CPU) -- truly O(nnz*k*iter).
        t = time_call(lambda: A.eigsh(k=4, which="LM"), reps=2)
        pm = peak_mem_call(lambda: A.eigsh(k=4, which="LM"))
        rows.append({"op": "A.eigsh(k=4) LM", "n": n, "nnz": nnz, "time_s": t, "peak_mb": pm})

    if "norm" in ops:
        t = time_call(lambda: A.norm("fro"), reps=5)
        pm = peak_mem_call(lambda: A.norm("fro"))
        rows.append({"op": "A.norm('fro')", "n": n, "nnz": nnz, "time_s": t, "peak_mb": pm})

    for r in rows:
        print(f"  {r['op']:<28s}  time {r['time_s']*1e3:>8.2f} ms   peak {r['peak_mb']:>7.1f} MB", flush=True)

    return rows


def fit_loglog(xs, ys) -> float:
    """Return the slope of log(y) vs log(x) — i.e. the empirical exponent.

    O(nnz)   -> slope ~ 1
    O(nnz^1.5)-> slope ~ 1.5
    O(n^2)   -> slope ~ 2  (for Poisson 5-pt: nnz~5n so slopes are 1:1)
    """
    if len(xs) < 2:
        return float('nan')
    lx = np.log(np.array(xs, dtype=float))
    ly = np.log(np.array(ys, dtype=float))
    return float(np.polyfit(lx, ly, 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sides", type=str, default="20,40,80,160",
                        help="comma-separated grid sides (n = side^2)")
    parser.add_argument("--ops", type=str,
                        default="matvec,solve,det,det_grad,logdet_hutch,eigsh,norm")
    args = parser.parse_args()

    sides = [int(s) for s in args.sides.split(",")]
    ops = args.ops.split(",")

    torch.manual_seed(0)
    torch.set_num_threads(1)

    all_rows = []
    for side in sides:
        all_rows.extend(benchmark_size(side, ops))

    # Group by op and fit log-log slope vs nnz.
    print("\n=== Scaling (log-log slope of time vs nnz) ===")
    print(f"{'op':<28s}  {'slope_time':>10s}  {'slope_mem':>10s}  notes")
    print("-" * 80)
    by_op: dict[str, list[dict]] = {}
    for r in all_rows:
        by_op.setdefault(r["op"], []).append(r)
    for op, runs in by_op.items():
        if len(runs) < 2:
            continue
        nnzs = [r["nnz"] for r in runs]
        st = fit_loglog(nnzs, [r["time_s"] for r in runs])
        sm = fit_loglog(nnzs, [r["peak_mb"] for r in runs])
        # Classify.
        if st < 1.2:
            note = "O(nnz) ✓"
        elif st < 1.7:
            note = "~ O(nnz^1.5) (sparse LU fill)"
        elif st < 2.3:
            note = "~ O(n^2)"
        else:
            note = "super-quadratic"
        print(f"{op:<28s}  {st:>10.2f}  {sm:>10.2f}  {note}")


if __name__ == "__main__":
    main()
