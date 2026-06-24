#!/usr/bin/env python
"""Comprehensive scaling + capacity benchmark for EVERY public torch-sla op.

For each op, over a poisson_2d size sweep (DOF = side**2, nnz ~ 5*DOF), measure
  - time (median of reps, after warmup)
  - throughput (DOF / s)
  - peak memory: CPU -> tracemalloc peak AND process RSS delta (psutil);
                 CUDA -> torch.cuda.max_memory_allocated
  - CPU utilization sampled over the op (psutil.cpu_percent)

A ``--max-probe`` pass finds the largest DOF each op can sustain before it
OOMs / errors / exceeds a per-op time cap. Each probe step runs in an isolated
SUBPROCESS so an OOM or hard crash cannot take down the whole benchmark, and a
RAM-headroom guard refuses sizes that would exhaust the box.

Plots (--out): per-op time-vs-DOF log-log with fitted slope, a combined
memory-vs-DOF plot, a combined throughput plot, and a max-capacity bar chart.

Ops covered (skipped gracefully if a backend/dep is missing):
  spmv (A@x), matmat (A@A), solve_cg (pytorch CG), solve_lu (scipy LU),
  solve_strumpack, det, det_backward, logdet (Hutchinson), eigsh (k=6 SA),
  norm (fro), transpose, connected_components.

Usage::

    ~/.venvs/torchsla/bin/python benchmarks/benchmark_all_ops_scaling.py
    ~/.venvs/torchsla/bin/python benchmarks/benchmark_all_ops_scaling.py --quick
    ~/.venvs/torchsla/bin/python benchmarks/benchmark_all_ops_scaling.py --max-probe

GPU: pass ``--device cuda``. NOTE: produce real GPU numbers on the 4070ti;
macor7's 780M iGPU OOMs on sparse kernels.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_sla.datasets as d  # noqa: E402
from torch_sla import SparseTensor, spsolve, DetConfig  # noqa: E402
from torch_sla.backends import is_strumpack_available  # noqa: E402

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover
    _HAVE_PSUTIL = False
    _PROC = None

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import connected_components as scipy_cc  # noqa: E402


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------
def build(side, device):
    """poisson_2d(side) -> (SparseTensor, dof, nnz)."""
    p = d.poisson_2d(side)
    val, row, col, shape = p.coo()
    A = SparseTensor(val.to(device), row.to(device), col.to(device), shape)
    return A, shape[0], A.nnz


def to_scipy_csr(A):
    val = A.values.detach().cpu().numpy()
    row = A.row_indices.detach().cpu().numpy()
    col = A.col_indices.detach().cpu().numpy()
    M, N = A.sparse_shape
    return sp.coo_matrix((val, (row, col)), shape=(M, N)).tocsr()


# ---------------------------------------------------------------------------
# Op registry: name -> setup(A, dof, device) -> callable run()
# Each setup returns a zero-arg callable that performs the op once.
# 'avail' is a predicate; ops that are unavailable are skipped.
# ---------------------------------------------------------------------------
def _setup_spmv(A, dof, device):
    x = torch.randn(dof, dtype=A.dtype, device=device)
    return lambda: (A @ x)


def _setup_matmat(A, dof, device):
    return lambda: (A @ A)


def _setup_solve_cg(A, dof, device):
    val, row, col = A.values, A.row_indices, A.col_indices
    shape = A.sparse_shape
    b = torch.ones(dof, dtype=A.dtype, device=device)
    return lambda: spsolve(val, row, col, shape, b, backend="pytorch",
                           method="cg", is_spd=True, tol=1e-8, maxiter=20000)


def _setup_solve_lu(A, dof, device):
    val, row, col = A.values.cpu(), A.row_indices.cpu(), A.col_indices.cpu()
    shape = A.sparse_shape
    b = torch.ones(dof, dtype=A.dtype)
    return lambda: spsolve(val, row, col, shape, b, backend="scipy", method="lu")


def _setup_solve_strumpack(A, dof, device):
    val, row, col = A.values, A.row_indices, A.col_indices
    shape = A.sparse_shape
    b = torch.ones(dof, dtype=A.dtype, device=device)
    return lambda: spsolve(val, row, col, shape, b, backend="strumpack")


def _setup_det(A, dof, device):
    return lambda: A.det()


def _setup_det_backward(A, dof, device):
    v = A.values.detach().clone().requires_grad_(True)
    B = SparseTensor(v, A.row_indices, A.col_indices, shape=A.sparse_shape)

    def run():
        v.grad = None
        B.det().backward()
    return run


def _setup_logdet(A, dof, device):
    def run():
        with DetConfig(method="hutchinson", num_probes=20, lanczos_iter=30):
            return A.logdet()
    return run


def _setup_eigsh(A, dof, device):
    return lambda: A.eigsh(k=6, which="SA", return_eigenvectors=False)


def _setup_norm(A, dof, device):
    return lambda: A.norm("fro")


def _setup_transpose(A, dof, device):
    return lambda: A.T()


def _setup_cc(A, dof, device):
    return lambda: A.connected_components()


OPS = {
    "spmv":            dict(setup=_setup_spmv,            reps=5, avail=lambda dev: True),
    "matmat":          dict(setup=_setup_matmat,         reps=3, avail=lambda dev: True),
    "solve_cg":        dict(setup=_setup_solve_cg,       reps=2, avail=lambda dev: True),
    "solve_lu":        dict(setup=_setup_solve_lu,       reps=2, avail=lambda dev: True),
    "solve_strumpack": dict(setup=_setup_solve_strumpack, reps=2,
                            avail=lambda dev: is_strumpack_available()),
    "det":             dict(setup=_setup_det,            reps=2, avail=lambda dev: True),
    "det_backward":    dict(setup=_setup_det_backward,   reps=2, avail=lambda dev: True),
    "logdet":          dict(setup=_setup_logdet,         reps=2, avail=lambda dev: True),
    "eigsh":           dict(setup=_setup_eigsh,          reps=2, avail=lambda dev: True),
    "norm":            dict(setup=_setup_norm,           reps=5, avail=lambda dev: True),
    "transpose":       dict(setup=_setup_transpose,      reps=5, avail=lambda dev: True),
    "cc":              dict(setup=_setup_cc,             reps=3, avail=lambda dev: True),
}


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def measure(run, *, reps, device, warmup=1):
    """Return dict(time_s, peak_tracemalloc_mb, rss_delta_mb, cpu_util,
    cuda_peak_mb) for `run`."""
    for _ in range(warmup):
        run()
        _sync(device)

    # --- time (median) ---
    samples = []
    for _ in range(reps):
        gc.collect()
        _sync(device)
        t0 = time.perf_counter()
        run()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    time_s = float(np.median(samples))

    # --- CPU util + RSS delta over one timed call ---
    cpu_util = float("nan")
    rss_delta_mb = float("nan")
    if _HAVE_PSUTIL:
        gc.collect()
        rss0 = _PROC.memory_info().rss
        _PROC.cpu_percent(None)  # reset counter
        run()
        _sync(device)
        cpu_util = float(_PROC.cpu_percent(None))
        rss1 = _PROC.memory_info().rss
        rss_delta_mb = max(0.0, (rss1 - rss0) / 1024 / 1024)

    # --- peak python alloc (tracemalloc) ---
    gc.collect()
    tracemalloc.start()
    run()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_tm_mb = peak / 1024 / 1024

    # --- cuda peak ---
    cuda_peak_mb = float("nan")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        run()
        torch.cuda.synchronize()
        cuda_peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return dict(time_s=time_s, peak_tm_mb=peak_tm_mb, rss_delta_mb=rss_delta_mb,
                cpu_util=cpu_util, cuda_peak_mb=cuda_peak_mb)


def peak_mb(rec, device):
    """Best single peak-memory number for plotting/reporting."""
    if device == "cuda" and not np.isnan(rec["cuda_peak_mb"]):
        return rec["cuda_peak_mb"]
    # CPU: take the max of tracemalloc peak and RSS delta
    vals = [rec["peak_tm_mb"]]
    if not np.isnan(rec["rss_delta_mb"]):
        vals.append(rec["rss_delta_mb"])
    return max(vals)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(op_names, sides_map, device):
    """Returns results[op] = list of row dicts."""
    results = {op: [] for op in op_names}
    for op in op_names:
        spec = OPS[op]
        if not spec["avail"](device):
            print(f"  [skip] {op}: backend/dep unavailable", flush=True)
            continue
        sides = sides_map[op]
        print(f"\n--- {op} (sides {sides}) ---", flush=True)
        for side in sides:
            try:
                A, dof, nnz = build(side, device)
                run = spec["setup"](A, dof, device)
                rec = measure(run, reps=spec["reps"], device=device)
            except (RuntimeError, MemoryError) as e:
                print(f"    side={side} dof={side*side} FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                break
            pm = peak_mb(rec, device)
            row = dict(rec)  # time_s, peak_tm_mb, rss_delta_mb, cpu_util, cuda_peak_mb
            row.update(op=op, dof=dof, nnz=nnz,
                       tput=dof / rec["time_s"], peak_mb=pm)
            results[op].append(row)
            print(f"    dof={dof:>8d} nnz={nnz:>9d}  t={rec['time_s']*1e3:>9.2f}ms "
                  f"tput={row['tput']:.2e}  peak={pm:>8.1f}MB  cpu={rec['cpu_util']:>5.0f}%",
                  flush=True)
            del A, run
            gc.collect()
    return results


def fit_slope(dofs, vals):
    if len(dofs) < 2:
        return float("nan")
    x = np.log(np.asarray(dofs, float))
    y = np.log(np.asarray(vals, float))
    return float(np.polyfit(x, y, 1)[0])


# ---------------------------------------------------------------------------
# Max-capacity probe (subprocess-isolated)
# ---------------------------------------------------------------------------
PROBE_WORKER = r"""
import sys, json, time, warnings, gc, importlib.util
warnings.filterwarnings("ignore")
sys.path.insert(0, {repo!r})
import torch
import torch_sla.datasets as d
from torch_sla import SparseTensor, spsolve, DetConfig
# load the benchmark module by path (benchmarks/ is not a package)
_spec = importlib.util.spec_from_file_location("_bench", {modpath!r})
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

op = sys.argv[1]; side = int(sys.argv[2]); device = sys.argv[3]
try:
    A, dof, nnz = B.build(side, device)
    run = B.OPS[op]["setup"](A, dof, device)
    t0 = time.perf_counter()
    run()
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter() - t0
    try:
        import psutil, os
        peak = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        peak = float("nan")
    print("OK " + json.dumps(dict(dof=dof, nnz=nnz, time_s=t, peak_mb=peak)))
except Exception as e:
    print("ERR " + type(e).__name__ + ": " + str(e)[:200])
"""


def avail_ram_mb():
    if _HAVE_PSUTIL:
        return psutil.virtual_memory().available / 1024 / 1024
    return float("inf")


def max_probe(op_names, device, time_cap=60.0, headroom_mb=12000, start_side=64):
    """For each op, increase side until OOM / error / time-cap / RAM guard.

    Each candidate runs in a fresh subprocess so a hard OOM can't kill us.
    Returns probe[op] = dict(max_dof, max_nnz, peak_mb, time_s, stop_reason).
    """
    repo = str(Path(__file__).resolve().parents[1])
    modpath = str(Path(__file__).resolve())
    worker = PROBE_WORKER.format(repo=repo, modpath=modpath)
    worker_path = Path(repo) / "benchmarks" / "_probe_worker.py"
    worker_path.write_text(worker)

    probe = {}
    try:
        for op in op_names:
            if not OPS[op]["avail"](device):
                continue
            print(f"\n[probe] {op} ...", flush=True)
            side = start_side
            best = None
            stop_reason = "?"
            while True:
                dof = side * side
                # RAM headroom guard (rough): refuse if free RAM small
                free = avail_ram_mb()
                if device == "cpu" and free < headroom_mb:
                    stop_reason = f"ram-guard (free {free:.0f}MB < {headroom_mb}MB)"
                    break
                t0 = time.perf_counter()
                try:
                    out = subprocess.run(
                        [sys.executable, str(worker_path), op, str(side), device],
                        cwd=repo, capture_output=True, text=True,
                        timeout=time_cap + 60,
                    )
                except subprocess.TimeoutExpired:
                    stop_reason = f"timeout (>{time_cap+60:.0f}s wall) at dof={dof}"
                    break
                wall = time.perf_counter() - t0
                stdout = out.stdout.strip().splitlines()
                line = stdout[-1] if stdout else ""
                if line.startswith("OK "):
                    rec = json.loads(line[3:])
                    best = rec
                    print(f"    dof={rec['dof']:>9d} nnz={rec['nnz']:>10d} "
                          f"t={rec['time_s']:.2f}s peak={rec['peak_mb']:.0f}MB",
                          flush=True)
                    if rec["time_s"] > time_cap:
                        stop_reason = f"time-cap ({rec['time_s']:.1f}s > {time_cap:.0f}s)"
                        break
                    # grow geometrically (~1.6x in dof => ~1.26x in side)
                    side = int(side * 1.26) + 1
                else:
                    # error or crash
                    msg = line[4:] if line.startswith("ERR ") else (line or f"rc={out.returncode}")
                    if out.returncode != 0 and not line.startswith("ERR"):
                        msg = f"subprocess died rc={out.returncode} (likely OOM-kill)"
                    stop_reason = f"fail at dof={dof}: {msg}"
                    print(f"    dof={dof:>9d}  STOP: {msg}", flush=True)
                    break
            if best is not None:
                probe[op] = dict(max_dof=best["dof"], max_nnz=best["nnz"],
                                 peak_mb=best["peak_mb"], time_s=best["time_s"],
                                 stop_reason=stop_reason)
            else:
                probe[op] = dict(max_dof=0, max_nnz=0, peak_mb=float("nan"),
                                 time_s=float("nan"), stop_reason=stop_reason)
            print(f"  -> {op}: max_dof={probe[op]['max_dof']}  ({stop_reason})", flush=True)
    finally:
        try:
            worker_path.unlink()
        except OSError:
            pass
    return probe


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_time_per_op(out, results, slopes, device):
    paths = []
    for op, rows in results.items():
        if len(rows) < 2:
            continue
        dofs = [r["dof"] for r in rows]
        times = [r["time_s"] for r in rows]
        plt.figure(figsize=(7, 5))
        plt.loglog(dofs, times, "o-", label=f"{_lbl(op)} (slope={slopes[op]:.2f})")
        d0 = np.array(dofs, float)
        anchor = times[0] / d0[0]
        plt.loglog(d0, anchor * d0, "k--", alpha=0.3, label="O(N) ref")
        plt.loglog(d0, anchor * d0[0] * (d0 / d0[0]) ** 2, "k:", alpha=0.3, label="O(N^2) ref")
        plt.xlabel("DOF (N)"); plt.ylabel("time [s]")
        plt.title(f"{op}: time vs DOF ({device})")
        plt.grid(True, which="both", alpha=0.3); plt.legend(); plt.tight_layout()
        p = out / f"allops_time_{op}.png"
        plt.savefig(p, dpi=110); plt.close()
        paths.append(p)
    return paths


def plot_combined_mem(out, results, device):
    plt.figure(figsize=(8, 6))
    for op, rows in results.items():
        if len(rows) < 2:
            continue
        plt.loglog([r["dof"] for r in rows], [r["peak_mb"] for r in rows], "o-", label=op)
    plt.xlabel("DOF (N)"); plt.ylabel("peak memory [MB]")
    plt.title(f"Peak memory vs DOF (all ops, {device})")
    plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    p = out / "allops_memory.png"
    plt.savefig(p, dpi=110); plt.close()
    return p


# Backend / method actually exercised by each op (shown in plot legends).
BACKENDS = {
    "spmv": "torch", "matmat": "torch", "norm": "torch", "transpose": "torch",
    "cc": "torch (pure)", "solve_cg": "pytorch/cg", "solve_lu": "scipy/lu",
    "solve_strumpack": "strumpack", "det": "scipy", "det_backward": "adjoint",
    "logdet": "hutchinson", "eigsh": "lobpcg",
}


def _lbl(op):
    return f"{op} [{BACKENDS.get(op, '?')}]"


def plot_combined_latency(out, results, device):
    """Latency (wall time) vs DOF -- the primary metric. Backend in the legend."""
    plt.figure(figsize=(8, 6))
    for op, rows in results.items():
        if len(rows) < 2:
            continue
        plt.loglog([r["dof"] for r in rows],
                   [r["time_s"] * 1e3 for r in rows], "o-", label=_lbl(op))
    plt.xlabel("DOF (N)"); plt.ylabel("latency [ms]")
    plt.title(f"Latency vs DOF (all ops, {device})")
    plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    p = out / "allops_latency.png"
    plt.savefig(p, dpi=110); plt.close()
    return p


def plot_combined_tput(out, results, device):
    plt.figure(figsize=(8, 6))
    for op, rows in results.items():
        if len(rows) < 2:
            continue
        plt.loglog([r["dof"] for r in rows], [r["tput"] for r in rows], "o-", label=_lbl(op))
    plt.xlabel("DOF (N)"); plt.ylabel("throughput [DOF / s]")
    plt.title(f"Throughput vs DOF (all ops, {device})")
    plt.grid(True, which="both", alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    p = out / "allops_throughput.png"
    plt.savefig(p, dpi=110); plt.close()
    return p


def plot_capacity(out, probe, device):
    ops = [o for o in probe if probe[o]["max_dof"] > 0]
    if not ops:
        return None
    dofs = [probe[o]["max_dof"] for o in ops]
    order = np.argsort(dofs)
    ops = [ops[i] for i in order]; dofs = [dofs[i] for i in order]
    plt.figure(figsize=(9, 5))
    bars = plt.barh(ops, dofs, color="steelblue")
    plt.xscale("log")
    plt.xlabel("max DOF before OOM / error / time-cap")
    plt.title(f"Max capacity per op ({device})")
    for b, dof in zip(bars, dofs):
        plt.text(dof, b.get_y() + b.get_height() / 2, f" {dof:.0f}",
                 va="center", fontsize=8)
    plt.grid(True, axis="x", which="both", alpha=0.3); plt.tight_layout()
    p = out / "allops_max_capacity.png"
    plt.savefig(p, dpi=110); plt.close()
    return p


# ---------------------------------------------------------------------------
# Tables / reporting
# ---------------------------------------------------------------------------
def print_results(results, slopes, device):
    print("\n" + "=" * 92)
    print(f"PER-OP SCALING TABLE ({device})")
    print("=" * 92)
    hdr = f"{'op':<16}{'dof':>9}{'nnz':>11}{'time_s':>11}{'tput':>11}{'peak_MB':>10}{'cpu%':>7}"
    for op, rows in results.items():
        if not rows:
            continue
        print(f"\n[{op}]  slope(time vs DOF) = {slopes.get(op, float('nan')):.3f}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            print(f"{op:<16}{r['dof']:>9}{r['nnz']:>11}{r['time_s']:>11.4g}"
                  f"{r['tput']:>11.3g}{r['peak_mb']:>10.1f}{r['cpu_util']:>7.0f}")


def print_capacity(probe):
    print("\n" + "=" * 78)
    print("MAX-CAPACITY TABLE")
    print("=" * 78)
    print(f"{'op':<16}{'max_dof':>11}{'max_nnz':>13}{'peak_MB':>10}{'time_s':>9}  stop_reason")
    print("-" * 78)
    for op, p in probe.items():
        print(f"{op:<16}{p['max_dof']:>11}{p['max_nnz']:>13}{p['peak_mb']:>10.0f}"
              f"{p['time_s']:>9.2f}  {p['stop_reason']}")


def write_json(out, results, slopes, probe, device):
    payload = dict(device=device, results=results, slopes=slopes, probe=probe)
    p = out / "allops_results.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-probe", action="store_true",
                    help="also run the max-capacity probe (subprocess-isolated)")
    ap.add_argument("--ops", default="all",
                    help="comma list of ops or 'all'")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to cpu")
        device = "cpu"

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.mkdir(parents=True, exist_ok=True)

    op_names = list(OPS) if args.ops == "all" else args.ops.split(",")

    nthreads = torch.get_num_threads()
    print(f"device={device}  quick={args.quick}  max_probe={args.max_probe}")
    print(f"torch threads={nthreads}  psutil={_HAVE_PSUTIL}  "
          f"strumpack={is_strumpack_available()}")

    # --- size sweeps per op ---
    # poisson_2d(side): DOF=side^2, nnz~5*DOF.
    # cheap ops (spmv/norm/transpose/cc/matmat) push to ~1e6 DOF (side~1024).
    # direct (lu/det/det_backward) cap early due to fill-in.
    if args.quick:
        cheap = [32, 64, 128, 200]
        mid = [32, 64, 128]
        direct = [16, 32, 48, 64]
        eig = [32, 64, 96]
    else:
        cheap = [32, 64, 128, 256, 448, 640, 832, 1024]   # DOF 1e3 .. ~1.05e6
        mid = [32, 64, 128, 256, 448, 640]                 # CG / logdet / strumpack
        direct = [16, 32, 48, 64, 96, 128, 192, 256, 320]  # LU / det fill-in heavy
        eig = [32, 64, 96, 128, 192, 256, 350]

    sides_map = {
        "spmv": cheap, "norm": cheap, "transpose": cheap, "cc": cheap, "matmat": cheap,
        "solve_cg": mid, "logdet": mid, "solve_strumpack": mid,
        "solve_lu": direct, "det": direct, "det_backward": direct,
        "eigsh": eig,
    }
    sides_map = {op: sides_map[op] for op in op_names}

    t0 = time.perf_counter()
    results = run_sweep(op_names, sides_map, device)
    sweep_t = time.perf_counter() - t0
    print(f"\nSweep wall time: {sweep_t:.1f}s")

    slopes = {op: fit_slope([r["dof"] for r in rows], [r["time_s"] for r in rows])
              for op, rows in results.items() if len(rows) >= 2}

    probe = {}
    if args.max_probe:
        print("\n" + "=" * 60)
        print("MAX-CAPACITY PROBE (subprocess-isolated)")
        print("=" * 60)
        time_cap = 8.0 if args.quick else 30.0
        probe = max_probe(op_names, device, time_cap=time_cap)

    # --- plots ---
    print("\nGenerating plots ...", flush=True)
    paths = []
    paths += plot_time_per_op(out, results, slopes, device)
    paths.append(plot_combined_latency(out, results, device))
    paths.append(plot_combined_mem(out, results, device))
    paths.append(plot_combined_tput(out, results, device))
    if probe:
        cp = plot_capacity(out, probe, device)
        if cp:
            paths.append(cp)
    jpath = write_json(out, results, slopes, probe, device)

    # --- tables ---
    print_results(results, slopes, device)
    if probe:
        print_capacity(probe)

    print("\nGenerated files:")
    for p in paths:
        print(f"  {p}")
    print(f"  {jpath}")
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
