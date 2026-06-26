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
import json
import platform
import re
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
# Device / environment label  (answers the user's "在什么设备上测的?")
#
# Captured once and stamped into every plot caption + the JSON sidecar so the
# numbers are never device-ambiguous: CPU model (from /proc/cpuinfo or
# platform), GPU name (torch.cuda.get_device_name), dtype, torch version, and
# whether torch.compile was used (we run eager, no compile).
# ---------------------------------------------------------------------------
def _cpu_model() -> str:
    try:
        txt = Path("/proc/cpuinfo").read_text()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def device_info(device: str, *, dtype: str = "float64", compiled: bool = False) -> dict:
    """Structured device/environment label for plots and JSON."""
    if device == "cuda" and torch.cuda.is_available():
        proc = torch.cuda.get_device_name(0)
        backend_dev = "CUDA"
    elif device == "cuda":
        proc = "CUDA (unavailable)"
        backend_dev = "CUDA"
    else:
        proc = _cpu_model()
        backend_dev = "CPU"
    return {
        "device": device,
        "device_kind": backend_dev,
        "processor": proc,
        "dtype": dtype,
        "torch": torch.__version__,
        "compile": "eager, no compile" if not compiled else "torch.compile",
        "platform": platform.platform(),
    }


def device_caption(info: dict) -> str:
    """One-line human caption: 'CUDA: NVIDIA GeForce RTX 4070 Ti | float64 |
    torch 2.x | eager, no compile'."""
    return (f"{info['device_kind']}: {info['processor']} | {info['dtype']} | "
            f"torch {info['torch']} | {info['compile']}")


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
    backward_setup : optional ``backward_setup(A, dof, device) -> callable()``
        running the FULL forward+backward (gradient) pass once. When set, the op
        is differentiable: the harness times this too and plots a BACKWARD curve
        alongside the forward one. Leave None for non-differentiable ops
        (connected_components) so they stay forward-only.
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
    backward_setup: Optional[Callable] = None


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
        # --- backward (gradient) pass timing for differentiable ops ---
        bwd_tag = ""
        if spec.backward_setup is not None:
            try:
                run_b = spec.backward_setup(A, dof, device)
                t_bwd = _time_median(run_b, reps=spec.reps, device=device)
                row["time_bwd_s"] = t_bwd
                bwd_tag = f"  t_bwd={t_bwd*1e3:>9.2f}ms (x{t_bwd/t:.2f})"
                del run_b
            except (RuntimeError, MemoryError) as e:
                print(f"    side={side} dof={dof} backward FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
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
              f"tput={row['tput']:.2e}{bwd_tag}{chk_tag}", flush=True)
        del A, run
        gc.collect()
        # Cap on the SLOWER of forward / backward: ops like det have a cheap
        # forward but an O(N^2) dense adjoint backward, so capping on forward
        # alone would let the backward tail run for minutes.
        t_worst = max(t, row.get("time_bwd_s", 0.0))
        if t_worst > time_cap:
            which = "backward" if row.get("time_bwd_s", 0.0) > t else "forward"
            print(f"    [time-cap] {which} {t_worst:.1f}s > {time_cap:.0f}s -- "
                  f"skipping larger DOF", flush=True)
            break
    return rows


# ---------------------------------------------------------------------------
# Plot: a single scaling curve per op -> assets/benchmarks/<png_name>_scaling.png
# ---------------------------------------------------------------------------
def plot_scaling(spec: OpSpec, rows, device: str, out_dir: Path,
                 info: Optional[dict] = None) -> Optional[Path]:
    """Per-op scaling curve. Plots the FORWARD curve and, for differentiable
    ops (``time_bwd_s`` present), a BACKWARD (gradient) curve on the same axes so
    the O(1)-adjoint cost is visible. The device/dtype/torch/compile label goes
    in the caption (the user asked "在什么设备上测的")."""
    if len(rows) < 2:
        print(f"  [plot] {spec.name}: <2 points, no curve", flush=True)
        return None
    if info is None:
        info = device_info(device)
    dofs = [r["dof"] for r in rows]
    times = [r["time_s"] for r in rows]
    slope = fit_slope(dofs, times)

    fig = plt.figure(figsize=(7, 5.4))
    plt.loglog(dofs, times, "o-", color="C0",
               label=f"forward (slope={slope:.2f})")

    bwd_rows = [r for r in rows if r.get("time_bwd_s") is not None]
    if len(bwd_rows) >= 2:
        bdofs = [r["dof"] for r in bwd_rows]
        btimes = [r["time_bwd_s"] for r in bwd_rows]
        bslope = fit_slope(bdofs, btimes)
        plt.loglog(bdofs, btimes, "s--", color="C3",
                   label=f"backward / gradient (slope={bslope:.2f})")

    d0 = np.asarray(dofs, float)
    anchor = times[0] / d0[0]
    plt.loglog(d0, anchor * d0, "k--", alpha=0.3, label="O(N) ref")
    plt.loglog(d0, anchor * d0[0] * (d0 / d0[0]) ** 2, "k:", alpha=0.3, label="O(N^2) ref")
    plt.xlabel("DOF (N)")
    plt.ylabel("time [s]")
    plt.title(f"{spec.name}: time vs DOF  [{spec.backend}]")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    fig.text(0.5, 0.005, device_caption(info), ha="center", va="bottom",
             fontsize=8, color="0.35")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec.png_name}_scaling.png"
    plt.savefig(path, dpi=110)
    plt.close()
    extra = "" if not bwd_rows else f", bwd slope={fit_slope([r['dof'] for r in bwd_rows], [r['time_bwd_s'] for r in bwd_rows]):.2f}"
    print(f"  [plot] wrote {path}  (fwd slope={slope:.2f}{extra})", flush=True)
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
    """Standard per-op ``main`` body: skip-if-unavailable, sweep, plot.

    Also writes a JSON sidecar ``<png_name>_scaling.json`` next to the PNG
    recording the device label and every (dof, forward, backward) row, so the
    numbers behind the plot are auditable and never device-ambiguous.
    """
    device = resolve_device(args.device)
    info = device_info(device)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    if not spec.avail(device):
        print(f"[skip] {spec.name}: backend/dep unavailable on device={device}")
        return None
    rows = sweep_op(spec, device, time_cap=args.time_cap, quick=args.quick)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "op": spec.name,
        "backend": spec.backend,
        "differentiable": spec.backward_setup is not None,
        "device_label": info,
        "rows": rows,
    }
    jpath = out_dir / f"{spec.png_name}_scaling.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  [json] wrote {jpath}", flush=True)
    return plot_scaling(spec, rows, device, out_dir, info)


def main_for(spec: OpSpec):
    """Build an argparse main for a single op file."""
    import argparse

    def main():
        ap = argparse.ArgumentParser(description=f"Scaling benchmark: {spec.name}")
        add_common_args(ap)
        args = ap.parse_args()
        run_and_plot(spec, args)

    return main


# ---------------------------------------------------------------------------
# Multi-backend comparison + precision (accuracy) plots for the linear solve.
#
# Answers two of the user's complaints at once: (1) "为啥没有不同的backend" -- one
# time-vs-DOF figure overlays every solver backend available on the device; and
# (2) "只有速度的图,没有精度的图" -- a second figure plots the relative residual
# ||Ax-b|| / ||b|| vs DOF for the same backends, so the direct (~1e-14) vs
# iterative (~1e-6) accuracy gap is VISIBLE, not just prose.
#
# A BackendCase pairs a setup (timed run) with a residual probe. Each is gated by
# an availability predicate so backends absent on the current device are skipped
# (and recorded as not-run rather than faked).
# ---------------------------------------------------------------------------
@dataclass
class BackendCase:
    label: str          # legend label, e.g. "scipy/lu (direct)"
    setup: Callable     # setup(A, dof, device) -> run() (timed)
    residual: Callable  # residual(A, dof, device) -> float  ||Ax-b||/||b||
    avail: Callable[[str], bool] = field(default=lambda dev: True)
    reps: int = 2


def sweep_backends(cases: Sequence[BackendCase], device: str, sides: Sequence[int],
                   *, time_cap: float):
    """Run every available backend over a SHARED DOF sweep; return
    ``{label: [rows]}`` where each row has dof, time_s, residual."""
    results = {}
    for case in cases:
        if not case.avail(device):
            print(f"[skip backend] {case.label}: unavailable on device={device}",
                  flush=True)
            continue
        rows = []
        print(f"\n--- backend {case.label}  device={device} ---", flush=True)
        for side in sides:
            try:
                A, dof, nnz = build(side, device)
                run = case.setup(A, dof, device)
                t = _time_median(run, reps=case.reps, device=device)
                res = case.residual(A, dof, device)
            except (RuntimeError, MemoryError, Exception) as e:  # noqa: BLE001
                print(f"    side={side} dof={side*side} FAILED: "
                      f"{type(e).__name__}: {str(e)[:80]}", flush=True)
                break
            rows.append(dict(dof=dof, time_s=t, residual=res))
            print(f"    dof={dof:>8d}  t={t*1e3:>9.2f}ms  residual={res:.2e}",
                  flush=True)
            del A, run
            gc.collect()
            if t > time_cap:
                print(f"    [time-cap] {t:.1f}s > {time_cap:.0f}s -- stop", flush=True)
                break
        if rows:
            results[case.label] = rows
    return results


def plot_backend_time(results: dict, info: dict, out_dir: Path,
                      png_name: str = "solve_backends") -> Optional[Path]:
    """One figure: time vs DOF, one line per backend (log-log)."""
    if not results:
        return None
    plt.figure(figsize=(7.4, 5.4))
    for i, (label, rows) in enumerate(results.items()):
        if len(rows) < 1:
            continue
        dofs = [r["dof"] for r in rows]
        times = [r["time_s"] for r in rows]
        slope = fit_slope(dofs, times) if len(rows) >= 2 else float("nan")
        lab = f"{label} (slope={slope:.2f})" if len(rows) >= 2 else label
        plt.loglog(dofs, times, "o-", color=f"C{i}", label=lab)
    plt.xlabel("DOF (N)")
    plt.ylabel("time [s]")
    plt.title("linear solve: time vs DOF across backends")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.gcf().text(0.5, 0.005, device_caption(info), ha="center", va="bottom",
                   fontsize=8, color="0.35")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{png_name}_scaling.png"
    plt.savefig(path, dpi=110)
    plt.close()
    print(f"  [plot] wrote {path}", flush=True)
    return path


def plot_backend_precision(results: dict, info: dict, out_dir: Path,
                           png_name: str = "solve_precision") -> Optional[Path]:
    """One figure: relative residual ||Ax-b||/||b|| vs DOF, one line per backend
    (semilog-y). Makes the direct vs iterative accuracy gap visible."""
    if not results:
        return None
    plt.figure(figsize=(7.4, 5.4))
    for i, (label, rows) in enumerate(results.items()):
        dofs = [r["dof"] for r in rows]
        res = [max(r["residual"], 1e-18) for r in rows]
        plt.semilogy(dofs, res, "o-", color=f"C{i}", label=label)
    plt.axhline(1e-12, color="green", ls="--", alpha=0.4,
                label="direct ~1e-12..1e-14")
    plt.axhline(1e-6, color="orange", ls="--", alpha=0.4,
                label="iterative tol ~1e-6")
    plt.xscale("log")
    plt.xlabel("DOF (N)")
    plt.ylabel(r"relative residual  $\|Ax-b\| / \|b\|$")
    plt.title("linear solve: accuracy (residual) vs DOF across backends")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.gcf().text(0.5, 0.005, device_caption(info), ha="center", va="bottom",
                   fontsize=8, color="0.35")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{png_name}.png"
    plt.savefig(path, dpi=110)
    plt.close()
    print(f"  [plot] wrote {path}", flush=True)
    return path
