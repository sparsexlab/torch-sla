"""Wall-time benchmark for complex sparse solve + Wirtinger adjoint.

Iterates the complex entries of ``torch_sla.datasets.SuiteSparse``,
measures forward-solve and backward-adjoint wall time on each, and
dumps a JSON summary + bar chart. Forward correctness is validated by
``Benchmark.evaluate`` (round-trip against the catalogue's stored
``x_ref``); the script does no separate scipy comparison and no
separate download / cache management -- both live in
``torch_sla.datasets``.

  cache:        $TORCH_SLA_DATASET (default ~/.cache/torch_sla/datasets)
  results:      results/benchmark_complex_solve/
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import torch

from torch_sla import SparseTensor
from torch_sla.datasets import SuiteSparse, DatasetUnavailable

warnings.filterwarnings("ignore")

RESULT_DIR = "results/benchmark_complex_solve"
os.makedirs(RESULT_DIR, exist_ok=True)

# Complex catalogue entries -- everything else is dtype-real.
_COMPLEX_KEYS = [k for k in SuiteSparse if "complex" in k]


def _solver(val, row, col, shape, rhs):
    return SparseTensor(val, row, col, shape).solve(rhs)


def benchmark_one(key: str) -> dict | None:
    try:
        b = SuiteSparse[key]
    except DatasetUnavailable as e:
        print(f"  [skip] {key}: {e}", flush=True)
        return None

    # Forward correctness via the round-trip stored in the catalogue.
    fwd_err = max(b.evaluate(_solver, metric="rel_l2"))

    A = SparseTensor(b.val, b.row, b.col, b.shape)
    rhs = b[0]["b"]

    # Forward timing (warm + measured).
    for _ in range(2):
        _ = A.solve(rhs)
    t0 = time.perf_counter()
    _ = A.solve(rhs)
    t_fwd = time.perf_counter() - t0

    # Backward timing -- gradient w.r.t. ``val``.
    v = b.val.clone().contiguous().requires_grad_(True)
    x_g = SparseTensor(v, b.row, b.col, b.shape).solve(rhs)
    loss = (x_g.conj() * x_g).real.sum()  # ||x||^2
    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0
    bwd_finite = bool(torch.isfinite(v.grad.abs()).all().item())

    result = {
        "matrix":   b.name,
        "key":      key,
        "math_kind":     b.math_kind,
        "detected_kind": b.detected_kind,
        "n":              b.shape[0],
        "nnz":            b.val.numel(),
        "fwd_ms":         t_fwd * 1e3,
        "fwd_err":        fwd_err,
        "bwd_ms":         t_bwd * 1e3,
        "bwd_finite":     bwd_finite,
    }
    print(
        f"  {key:25s} n={b.shape[0]:>5d} nnz={b.val.numel():>8d} "
        f"| {b.math_kind:>10s} "
        f"| fwd {t_fwd*1e3:7.2f}ms err {fwd_err:.0e} "
        f"| bwd {t_bwd*1e3:7.2f}ms finite={bwd_finite}",
        flush=True,
    )
    return result


def gradcheck_validation() -> bool:
    """Gold-standard numerical-FD check of the Wirtinger adjoint on the
    smallest complex catalogue entry (qc324, n=324). ``fast_mode=True``
    makes it O(1) full solves instead of O(nnz)."""
    try:
        b = SuiteSparse["complex_sym"]
    except DatasetUnavailable as e:
        print(f"  [skip gradcheck] {e}")
        return False
    val = b.val.clone().contiguous().requires_grad_(True)
    rhs = b[0]["b"]

    def fn(v):
        return SparseTensor(v, b.row, b.col, b.shape).solve(rhs)

    return torch.autograd.gradcheck(
        fn, (val,), eps=1e-6, atol=1e-4, rtol=1e-3,
        check_grad_dtypes=True, fast_mode=True,
    )


def plot(results):
    if not results:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [r["matrix"].split("/")[-1] for r in results]
    fwd = [r["fwd_ms"] for r in results]
    bwd = [r["bwd_ms"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.20, fwd, 0.40, label="forward solve")
    ax.bar(x + 0.20, bwd, 0.40, label="backward (adjoint)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("wall time (ms)"); ax.set_yscale("log")
    ax.set_title("Complex sparse solve on SuiteSparse "
                 "(scipy backend, complex128)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "timings.png")
    plt.savefig(out, dpi=150)
    print(f"\n  plot saved -> {out}")


def main():
    print("=== Benchmark: complex sparse solve on SuiteSparse matrices ===\n")
    print(f"  output: {RESULT_DIR}\n")

    results = [r for r in (benchmark_one(k) for k in _COMPLEX_KEYS) if r]

    with open(os.path.join(RESULT_DIR, "timings.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Wirtinger gradient validation (autograd.gradcheck) ===")
    try:
        ok = gradcheck_validation()
        print(f"  gradcheck: {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"  gradcheck FAIL: {type(e).__name__}: {e}")

    plot(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
