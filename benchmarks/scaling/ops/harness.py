#!/usr/bin/env python
"""Shared harness for the per-op scaling benchmarks under ``ops/``.

Every headline torch-sla operation gets its OWN runnable benchmark file in this
directory (``spmv.py``, ``solve_cg.py``, ``eigsh.py``, ...). Each of those files
is thin: it declares an :class:`OpSpec` (name, setup, reps, availability, the
DOF sweep) and calls :func:`run_and_plot`. All the real machinery -- the DOF
sweep with a per-point TIME CAP, median timing after warmup, slope fit, and the
log-log scaling-curve plot -- lives here so it is written once.

Problems come from ``torch_sla.datasets.poisson_2d(side)`` (DOF = side**2,
nnz ~ 5*DOF), reusing :func:`build`. The setup functions are imported by the
per-op files from ``benchmark_all_ops_scaling`` (the original monolith) wherever
they already exist, so we never rewrite a working op.

Output: each op emits a single scaling curve to
``assets/benchmarks/<png_name>_scaling.png`` (DOF vs time, log-log, with the op
name, device and backend in the title). The five historical names the docs
reference -- ``cg_scaling``, ``lu_scaling``, ``spmv_scaling``, ``eigsh_scaling``,
``connected_components_scaling`` -- are preserved via each op's ``png_name``.

The TIME CAP keeps any single op sweep well under a couple of minutes: once a
DOF point's median time exceeds ``time_cap`` seconds, the remaining (larger)
points are skipped. A representative curve over a few orders of magnitude of DOF
is the goal, not the full tail.
"""
from __future__ import annotations

import gc
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

warnings.filterwarnings("ignore")

# repo root on sys.path so ``import torch_sla`` works when run directly
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch_sla.datasets as d  # noqa: E402
from torch_sla import SparseTensor  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ASSETS = _REPO / "assets" / "benchmarks"


# ---------------------------------------------------------------------------
# Problem builder (shared with the monolith)
# ---------------------------------------------------------------------------
def build(side: int, device: str):
    """poisson_2d(side) -> (SparseTensor, dof, nnz)."""
    p = d.poisson_2d(side)
    val, row, col, shape = p.coo()
    A = SparseTensor(val.to(device), row.to(device), col.to(device), shape)
    return A, shape[0], A.nnz


# ---------------------------------------------------------------------------
# Default DOF (side) sweeps -- a per-op file picks one of these or its own.
# poisson_2d(side): DOF = side**2, nnz ~ 5*DOF.
# ---------------------------------------------------------------------------
SWEEP_CHEAP = [32, 64, 128, 256, 448, 640, 832, 1024]   # spmv/matmat/norm/cc -> ~1e6 DOF
SWEEP_MID = [32, 64, 128, 256, 448, 640]                 # CG / logdet / strumpack
SWEEP_DIRECT = [16, 32, 48, 64, 96, 128, 192, 256, 320]  # LU / det fill-in heavy
SWEEP_EIG = [32, 64, 96, 128, 192, 256, 350]             # eigsh / svd / condition_number

SWEEP_CHEAP_QUICK = [32, 64, 128, 200]
SWEEP_MID_QUICK = [32, 64, 128]
SWEEP_DIRECT_QUICK = [16, 32, 48, 64]
SWEEP_EIG_QUICK = [32, 64, 96]


@dataclass
class OpSpec:
    """One headline op's scaling benchmark.

    Attributes
    ----------
    name : human-readable op name (no abbreviations) used in the plot title.
    setup : ``setup(A, dof, device) -> callable()`` performing the op once.
    backend : backend/method string shown in the plot title and legend.
    png_name : basename for ``assets/benchmarks/<png_name>_scaling.png``.
    reps : timed repetitions (median reported) after one warmup.
    avail : ``avail(device) -> bool`` predicate; op is skipped if False.
    sweep / sweep_quick : list of poisson_2d sides to sweep.
    verify : optional ``verify(A, dof, device) -> float`` correctness probe.
    verify_ok : predicate on the verify value -> True if the result is correct.
    """
    name: str
    setup: Callable
    backend: str
    png_name: str
    reps: int = 3
    avail: Callable[[str], bool] = field(default=lambda dev: True)
    sweep: Sequence[int] = field(default_factory=lambda: SWEEP_CHEAP)
    sweep_quick: Sequence[int] = field(default_factory=lambda: SWEEP_CHEAP_QUICK)
    verify: Optional[Callable] = None
    verify_ok: Callable[[float], bool] = field(default=lambda v: v < 1e-3)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def _time_median(run, *, reps: int, device: str, warmup: int = 1) -> float:
    for _ in range(warmup):
        run()
        _sync(device)
    samples = []
    for _ in range(reps):
        gc.collect()
        _sync(device)
        t0 = time.perf_counter()
        run()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def fit_slope(dofs: Sequence[float], vals: Sequence[float]) -> float:
    if len(dofs) < 2:
        return float("nan")
    x = np.log(np.asarray(dofs, float))
    y = np.log(np.asarray(vals, float))
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# Sweep (with a per-point TIME CAP)
# ---------------------------------------------------------------------------
def sweep_op(spec: OpSpec, device: str, *, time_cap: float, quick: bool = False):
    """Run ``spec`` over its DOF sweep; return list of row dicts.

    Stops growing DOF once a point's median time exceeds ``time_cap`` (the
    larger tail points are skipped) or on the first OOM / error. This keeps a
    single op's sweep well under a couple of minutes so the server is not
    overloaded.
    """
    sides = list(spec.sweep_quick if quick else spec.sweep)
    rows = []
    print(f"\n--- {spec.name}  [{spec.backend}]  device={device}  sides={sides} ---",
          flush=True)
    for side in sides:
        try:
            A, dof, nnz = build(side, device)
            run = spec.setup(A, dof, device)
            t = _time_median(run, reps=spec.reps, device=device)
        except (RuntimeError, MemoryError) as e:
            print(f"    side={side} dof={side*side} FAILED: "
                  f"{type(e).__name__}: {e}", flush=True)
            break
        row = dict(op=spec.name, dof=dof, nnz=nnz, time_s=t, tput=dof / t)
        chk_tag = ""
        if spec.verify is not None:
            chk = spec.verify(A, dof, device)
            row["check"] = chk
            ok = spec.verify_ok(chk)
            chk_tag = f"  check={chk:.2e}"
            if not ok:
                row["correctness_fail"] = True
                chk_tag += "  *** WRONG RESULT ***"
                print(f"    !!! CORRECTNESS FAIL dof={dof}:{chk_tag}", flush=True)
        rows.append(row)
        print(f"    dof={dof:>8d} nnz={nnz:>9d}  t={t*1e3:>9.2f}ms "
              f"tput={row['tput']:.2e}{chk_tag}", flush=True)
        del A, run
        gc.collect()
        if t > time_cap:
            print(f"    [time-cap] {t:.1f}s > {time_cap:.0f}s -- skipping larger DOF",
                  flush=True)
            break
    return rows


# ---------------------------------------------------------------------------
# Plot: a single scaling curve per op -> assets/benchmarks/<png_name>_scaling.png
# ---------------------------------------------------------------------------
def plot_scaling(spec: OpSpec, rows, device: str, out_dir: Path) -> Optional[Path]:
    if len(rows) < 2:
        print(f"  [plot] {spec.name}: <2 points, no curve", flush=True)
        return None
    dofs = [r["dof"] for r in rows]
    times = [r["time_s"] for r in rows]
    slope = fit_slope(dofs, times)

    plt.figure(figsize=(7, 5))
    plt.loglog(dofs, times, "o-", label=f"{spec.name} (slope={slope:.2f})")
    d0 = np.asarray(dofs, float)
    anchor = times[0] / d0[0]
    plt.loglog(d0, anchor * d0, "k--", alpha=0.3, label="O(N) ref")
    plt.loglog(d0, anchor * d0[0] * (d0 / d0[0]) ** 2, "k:", alpha=0.3, label="O(N^2) ref")
    plt.xlabel("DOF (N)")
    plt.ylabel("time [s]")
    plt.title(f"{spec.name}: time vs DOF  [{spec.backend}, {device}]")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.png_name}_scaling.png"
    plt.savefig(path, dpi=110)
    plt.close()
    print(f"  [plot] wrote {path}  (slope={slope:.2f})", flush=True)
    return path


# ---------------------------------------------------------------------------
# Entry point a per-op file calls
# ---------------------------------------------------------------------------
def add_common_args(ap):
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--out", default=str(ASSETS),
                    help="output dir for the <op>_scaling.png")
    ap.add_argument("--quick", action="store_true", help="fast smoke sweep")
    ap.add_argument("--time-cap", type=float, default=20.0,
                    help="skip larger DOF once a point exceeds this many seconds")
    return ap


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to cpu")
        return "cpu"
    return device


def run_and_plot(spec: OpSpec, args) -> Optional[Path]:
    """Standard per-op ``main`` body: skip-if-unavailable, sweep, plot."""
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    if not spec.avail(device):
        print(f"[skip] {spec.name}: backend/dep unavailable on device={device}")
        return None
    rows = sweep_op(spec, device, time_cap=args.time_cap, quick=args.quick)
    return plot_scaling(spec, rows, device, out_dir)


def main_for(spec: OpSpec):
    """Build an argparse main for a single op file."""
    import argparse

    def main():
        ap = argparse.ArgumentParser(description=f"Scaling benchmark: {spec.name}")
        add_common_args(ap)
        args = ap.parse_args()
        run_and_plot(spec, args)

    return main
