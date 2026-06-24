#!/usr/bin/env python
"""Per-op device/backend comparison plots.

Different ops are NOT comparable to each other (different work units), so each op
gets its own figure. Within an op, comparing **devices** (CPU vs CUDA) and
**backends** (e.g. solve: scipy-lu vs pytorch-cg vs strumpack) IS meaningful, so
those are overlaid on the same axes.

Reads the JSON dumps emitted by ``benchmark_all_ops_scaling.py`` (one per device,
e.g. ``allops_results.json`` for CPU and ``cuda_allops_results.json`` for CUDA) and
writes ``cmp_<op>.png`` with one line per (device, backend).

    python benchmarks/plot_device_compare.py \
        --cpu benchmarks/results/allops_results.json \
        --cuda benchmarks/results/cuda_allops_results.json \
        --out benchmarks/results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Full, human-readable op names (no cryptic abbreviations).
NAMES = {
    "spmv": "sparse matvec  (A @ x)",
    "matmat": "sparse matmat  (A @ A)",
    "norm": "Frobenius norm  (||A||_F)",
    "transpose": "transpose  (Aᵀ)",
    "cc": "connected_components",
    "solve_cg": "linear solve  (CG)",
    "solve_lu": "linear solve  (LU)",
    "solve_strumpack": "linear solve  (STRUMPACK)",
    "det": "determinant",
    "det_backward": "determinant — backward (adjoint)",
    "logdet": "log-determinant  (Hutchinson)",
    "eigsh": "eigsh  (smallest-k eigenpairs)",
}
BACKENDS = {
    "spmv": "torch", "matmat": "torch", "norm": "torch", "transpose": "torch",
    "cc": "torch", "solve_cg": "pytorch/cg", "solve_lu": "scipy/lu",
    "solve_strumpack": "strumpack", "det": "scipy", "det_backward": "adjoint",
    "logdet": "hutchinson", "eigsh": "lobpcg",
}


def _load(path):
    if not path or not Path(path).exists():
        return {}
    data = json.load(open(path))
    return data.get("results", data) if isinstance(data, dict) else {}


def _slope(dofs, times):
    if len(dofs) < 2:
        return float("nan")
    lx, ly = np.log(np.array(dofs)), np.log(np.array(times))
    return float(np.polyfit(lx, ly, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", default="benchmarks/results/allops_results.json")
    ap.add_argument("--cuda", default="benchmarks/results/cuda_allops_results.json")
    ap.add_argument("--rocm", default="benchmarks/results/rocm_allops_results.json")
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()

    series = {"CPU": _load(args.cpu), "CUDA": _load(args.cuda), "ROCm": _load(args.rocm)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ops = sorted({op for s in series.values() for op in s})
    written = []
    for op in ops:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        plotted = False
        for dev, res in series.items():
            rows = res.get(op)
            if not rows or len(rows) < 2:
                continue
            dofs = [r["dof"] for r in rows]
            times = [r["time_s"] * 1e3 for r in rows]  # ms
            ax.loglog(dofs, times, "o-",
                      label=f"{dev} [{BACKENDS.get(op, '?')}]  (slope={_slope([r['dof'] for r in rows], [r['time_s'] for r in rows]):.2f})")
            plotted = True
        if not plotted:
            plt.close(fig); continue
        # O(N) reference anchored at the first CPU point
        ref = series["CPU"].get(op) or next(iter([r for r in series.values() if r.get(op)]), None)
        ax.set_xlabel("DOF (N)"); ax.set_ylabel("time [ms]")
        ax.set_title(f"{NAMES.get(op, op)} — CPU vs CUDA")
        ax.grid(True, which="both", alpha=0.3); ax.legend()
        fig.tight_layout()
        p = out / f"cmp_{op}.png"
        fig.savefig(p, dpi=110); plt.close(fig)
        written.append(str(p))
    print("\n".join(written) if written else "no overlapping ops found")


if __name__ == "__main__":
    main()
