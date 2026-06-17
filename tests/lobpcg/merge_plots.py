"""Merge per-device JSON dumps from multiple machines into one figure.

The multi-device bench script saves one JSON per machine -- e.g.
Mac M4 produces ``lobpcg_fix_comparison_data_cpu_mps.json`` (CPU + MPS),
a Linux/Windows GPU box produces ``..._cpu_cuda.json``. This script
loads them all, dedupes by device name (later files win), and
generates the combined CPU / MPS / CUDA plot for the PR.

Usage::

    python examples/lobpcg_fix_comparison_merge.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ASSETS = os.path.join(HERE, "assets")


def main():
    files = sorted(glob.glob(os.path.join(DATA, "*.json")))
    if not files:
        raise SystemExit(f"no data files found in {DATA}")

    merged = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for dev, payload in data.items():
            # If a device shows up in two files (e.g. CPU on both Mac and
            # 4070ti), the later file wins -- the 4070ti CPU run is
            # arguably more relevant for the CUDA-companion narrative.
            merged[dev] = payload
        print(f"loaded {path}: devices = {list(data.keys())}")

    # Skip MPS in the published plot. Even with our CPU-roundtrip
    # workarounds for the MPS-missing eigh / O(n^2) qr kernels, MPS
    # forces float32 which caps the achievable Ritz residual at
    # ~1e-4..1e-3 on PDE-like operators -- the precision panel
    # would compress against the y-axis ceiling and bias readers
    # against the algorithm. ``_lobpcg_core`` now emits a
    # ``RuntimeWarning`` on MPS to point users at CPU/CUDA. Raw MPS
    # data still lives in the JSON if anyone wants to plot it.
    include_mps = os.environ.get("INCLUDE_MPS") == "1"
    devs = ("cpu", "mps", "cuda") if include_mps else ("cpu", "cuda")
    order = [d for d in devs if d in merged]
    n_devs = len(order)

    fig, axes = plt.subplots(2, n_devs, figsize=(6 * n_devs, 9),
                             squeeze=False)
    colors = {"v1 (pre-fix)": "#d62728",
              "v1.5 (conv. fix only)": "#ff7f0e",
              "v2 (this PR)": "#2ca02c",
              "torch.lobpcg": "#1f77b4"}
    for col, dev_name in enumerate(order):
        data = merged[dev_name]
        ax_t = axes[0, col]
        ax_e = axes[1, col]
        for label, ts in data["times"].items():
            style = "o-" if label != "torch.lobpcg" else "s--"
            ax_t.plot(data["sizes"], ts, style, label=label,
                      color=colors[label], markersize=7, linewidth=2)
            ax_e.semilogy(data["sizes"], data["errs"][label], style,
                          label=label, color=colors[label],
                          markersize=7, linewidth=2)
        ax_t.set_xlabel("matrix size n")
        ax_t.set_ylabel("wall-clock (ms)")
        ax_t.set_title(f"{dev_name.upper()}: speed")
        ax_t.legend(loc="upper left", fontsize=9)
        ax_t.grid(True, alpha=0.3)
        ax_e.set_xlabel("matrix size n")
        ax_e.set_ylabel(r"$\max_i\, |\lambda_i - \lambda_i^{\rm true}|$")
        ax_e.set_title(f"{dev_name.upper()}: precision")
        ax_e.axhline(1e-8, color="gray", linestyle=":", label="tol = 1e-8")
        ax_e.legend(loc="upper left", fontsize=9)
        ax_e.grid(True, alpha=0.3, which="both")

    os.makedirs(ASSETS, exist_ok=True)
    out = os.path.join(ASSETS, "comparison_all.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\ncombined plot ({n_devs} devices): {out}")


if __name__ == "__main__":
    main()
