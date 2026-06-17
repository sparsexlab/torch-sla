"""Multi-device sweep: CPU / MPS / CUDA.

Runs the same v1 / v1.5 / v2 / torch.lobpcg comparison from
``lobpcg_fix_comparison.py`` on whatever devices are available.
Useful for asking: does the LAPACK-QR-vs-Python-CGS2 win hold on
GPU backends (cuSOLVER on CUDA, MPS Metal kernels)?

Save: ``assets/examples/lobpcg_fix_comparison_<device>.png`` per
device, plus a combined ``assets/examples/lobpcg_fix_comparison_all.png``.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lobpcg_fix_comparison import (  # noqa: E402
    _cgs2_python_loop,
    _qr_orthonormalize,
    lobpcg_param,
    make_sparse_spd,
    time_call,
)


def available_devices():
    devs = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devs.append(torch.device("mps"))
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


def can_use_torch_lobpcg(device):
    """torch.lobpcg on MPS in current torch versions has gaps -- skip if
    it errors. Returns True if a single trial call works."""
    try:
        A = torch.eye(20, dtype=torch.float64, device=device)
        torch.lobpcg(A, k=2, largest=True, niter=10, tol=1e-6)
        return True
    except Exception:
        return False


def to_device(A_coo, A_dense, device):
    """Move both representations to ``device``. MPS doesn't yet have
    full sparse_coo support; fall back to dense for the matvec there."""
    if device.type == "mps":
        # MPS: keep dense (sparse_coo unsupported for many ops on MPS).
        return None, A_dense.to(device).to(torch.float32)
    return A_coo.to(device), A_dense.to(device)


def make_matvec(A_coo, A_dense, device):
    """Return a callable for matvec given whichever representation we
    chose to move to ``device``."""
    if A_coo is not None:
        return lambda B, _A=A_coo: torch.sparse.mm(_A, B)
    return lambda B, _A=A_dense: _A @ B


def main():
    sizes = [200, 400, 700, 1000, 1500, 2000]
    k = 6
    variants = {
        "v1 (pre-fix)":          dict(convergence="eigvals_diff", orthonormalize="cgs2_loop"),
        "v1.5 (conv. fix only)": dict(convergence="residual",      orthonormalize="cgs2_loop"),
        "v2 (this PR)":          dict(convergence="residual",      orthonormalize="qr"),
    }

    per_device = {}  # device_name -> {label: [times]}, {label: [errs]}

    for device in available_devices():
        dev_name = device.type
        print(f"\n{'='*70}\nDevice: {dev_name.upper()}")
        if dev_name == "cuda":
            print(f"  ({torch.cuda.get_device_name(0)})")
        has_torch_lobpcg = can_use_torch_lobpcg(device)
        print(f"  torch.lobpcg works on this device: {has_torch_lobpcg}")

        times = {label: [] for label in variants}
        errs = {label: [] for label in variants}
        if has_torch_lobpcg:
            times["torch.lobpcg"] = []
            errs["torch.lobpcg"] = []

        # MPS forces float32; CPU/CUDA stay float64
        use_float32 = (dev_name == "mps")
        dtype = torch.float32 if use_float32 else torch.float64

        for n in sizes:
            A_coo_cpu, A_dense_cpu = make_sparse_spd(n)
            if use_float32:
                A_coo_cpu = torch.sparse_coo_tensor(
                    A_coo_cpu.indices(), A_coo_cpu.values().float(),
                    size=A_coo_cpu.shape).coalesce()
                A_dense_cpu = A_dense_cpu.float()
            A_coo, A_dense = to_device(A_coo_cpu, A_dense_cpu, device)
            gt = sorted(np.linalg.eigvalsh(A_dense_cpu.double().numpy()),
                        reverse=True)[:k]
            matvec = make_matvec(A_coo, A_dense, device)

            if has_torch_lobpcg and A_coo is not None:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                vals_t, _ = torch.lobpcg(A_coo, k=k, largest=True,
                                          niter=300, tol=1e-8)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter() - t0
                err = max(abs(g - e) for g, e in zip(gt, vals_t.cpu().double().tolist()))
                times["torch.lobpcg"].append(t * 1000)
                errs["torch.lobpcg"].append(err)
                print(f"  n={n:>5d}  torch.lobpcg            {t*1000:>9.2f}ms  err={err:.2e}")

            for label, opts in variants.items():
                if device.type == "cuda":
                    torch.cuda.synchronize()
                torch.manual_seed(0)
                t0 = time.perf_counter()
                try:
                    vals, _ = lobpcg_param(
                        matvec, n, k,
                        dtype=dtype, device=device,
                        largest=True, maxiter=300, tol=1e-8, seed=0,
                        **opts,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t = time.perf_counter() - t0
                    err = max(abs(g - e) for g, e in zip(gt, vals.cpu().double().tolist()))
                except Exception as exc:
                    print(f"  n={n:>5d}  {label:22s}  ERROR: {type(exc).__name__}: {str(exc)[:80]}")
                    t = float("nan")
                    err = float("nan")
                times[label].append(t * 1000)
                errs[label].append(err)
                print(f"  n={n:>5d}  {label:22s}  {t*1000:>9.2f}ms  err={err:.2e}")

        per_device[dev_name] = {"times": times, "errs": errs, "sizes": sizes}

    # Persist raw data so multiple runs (different machines) can be
    # combined into one figure later.
    import json
    out_dir = os.path.normpath(os.path.join(HERE, "..", "assets", "examples"))
    os.makedirs(out_dir, exist_ok=True)
    devs_tag = "_".join(per_device.keys())
    json_path = os.path.join(out_dir, f"lobpcg_fix_comparison_data_{devs_tag}.json")
    with open(json_path, "w") as f:
        json.dump(per_device, f, indent=2)
    print(f"raw data: {json_path}")

    # Plot a combined figure: one column per device
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_devs = len(per_device)
        fig, axes = plt.subplots(2, n_devs, figsize=(6 * n_devs, 9),
                                  squeeze=False)
        colors = {"v1 (pre-fix)": "#d62728",
                  "v1.5 (conv. fix only)": "#ff7f0e",
                  "v2 (this PR)": "#2ca02c",
                  "torch.lobpcg": "#1f77b4"}
        for col, (dev_name, data) in enumerate(per_device.items()):
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

        out_dir = os.path.normpath(os.path.join(HERE, "..", "assets", "examples"))
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "lobpcg_fix_comparison_all.png")
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\ncombined plot: {out}")
    except ImportError:
        print("\n(matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
