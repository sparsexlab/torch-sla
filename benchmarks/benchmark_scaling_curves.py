#!/usr/bin/env python
"""DOF-vs-throughput scaling curves + scaling analysis for torch-sla.

Sweeps problem size (degrees of freedom, DOF) over a range and times a set of
core algorithms, producing log-log time-vs-DOF curves, fitted empirical scaling
exponents, throughput curves, and an automatic "is the implementation OK?"
verdict per algorithm.

All problems come from ``torch_sla.datasets`` (no hand-built matrices). The
only composition done here is stacking several independent dataset problems
block-diagonally to obtain a matrix with a known non-trivial connected-component
structure -- that is composing datasets, not constructing a one-off matrix.

Algorithms timed
-----------------
1. connected_components  -- A.connected_components()  (vs scipy.csgraph ref)
2. spmv                  -- A @ x
3. cg                    -- spsolve(backend='pytorch', method='cg')  on poisson_2d (SPD)
4. lu                    -- spsolve(backend='scipy',   method='lu')  on poisson_2d
5. eigsh                 -- A.eigsh(k=6, which='SA')   on laplacian_2d

Run::

    ~/.venvs/torchsla/bin/python benchmarks/benchmark_scaling_curves.py
    ~/.venvs/torchsla/bin/python benchmarks/benchmark_scaling_curves.py --quick
"""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")  # silence pytorch sparse beta/invariant warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_sla.datasets as d  # noqa: E402
from torch_sla import SparseTensor, spsolve  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import scipy.sparse as sp  # noqa: E402
from scipy.sparse.csgraph import connected_components as scipy_cc  # noqa: E402


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def time_call(fn, *, reps: int, device: str, warmup: int = 1) -> float:
    """Median wall-clock seconds of ``fn`` over ``reps`` repeats, after warmup."""
    for _ in range(warmup):
        fn()
        _sync(device)
    samples = []
    for _ in range(reps):
        gc.collect()
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def fit_slope(dofs, times):
    """Least-squares slope of log(time) vs log(DOF). Returns (slope, intercept)."""
    x = np.log(np.asarray(dofs, dtype=float))
    y = np.log(np.asarray(times, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Problem builders (all via torch_sla.datasets)
# ---------------------------------------------------------------------------
def make_sparsetensor(problem, device: str) -> SparseTensor:
    val, row, col, shape = problem.coo()
    A = SparseTensor(val.to(device), row.to(device), col.to(device), shape)
    return A


def block_diag_poisson(block_side: int, k: int, device: str) -> SparseTensor:
    """Stack ``k`` independent poisson_2d(block_side) blocks block-diagonally.

    Produces a matrix with exactly ``k`` connected components (one per block),
    DOF = k * block_side**2. This is dataset composition, not a hand-built
    one-off: each block is a real ``datasets.poisson_2d`` problem, simply
    placed on disjoint index ranges.
    """
    base = d.poisson_2d(block_side)
    bval, brow, bcol, bshape = base.coo()
    n = bshape[0]
    vals, rows, cols = [], [], []
    for b in range(k):
        off = b * n
        vals.append(bval)
        rows.append(brow + off)
        cols.append(bcol + off)
    val = torch.cat(vals).to(device)
    row = torch.cat(rows).to(device)
    col = torch.cat(cols).to(device)
    N = k * n
    return SparseTensor(val, row, col, (N, N)), k


def to_scipy_csr(A: SparseTensor) -> sp.csr_matrix:
    val = A.values.detach().cpu().numpy()
    row = A.row_indices.detach().cpu().numpy()
    col = A.col_indices.detach().cpu().numpy()
    M, N = A.sparse_shape
    return sp.coo_matrix((val, (row, col)), shape=(M, N)).tocsr()


# ---------------------------------------------------------------------------
# Per-algorithm benchmarks
# ---------------------------------------------------------------------------
def bench_connected_components(sides, device, reps):
    """torch-sla connected_components vs scipy.csgraph, on:
      - single-component grids (poisson_2d, 1 component)
      - block-diagonal stacks (k>1 components)
    Returns dict with two sub-series (torch & scipy) for both structures.
    """
    rows_single, rows_multi = [], []
    for side in sides:
        # --- single component grid ---
        p = d.poisson_2d(side)
        A = make_sparsetensor(p, device)
        dof, nnz = A.sparse_shape[0], A.nnz
        t_ts = time_call(lambda: A.connected_components(), reps=reps, device=device)
        Acsr = to_scipy_csr(A)
        t_sp = time_call(lambda: scipy_cc(Acsr, directed=False), reps=reps, device="cpu")
        _, ncomp = A.connected_components()
        rows_single.append(dict(dof=dof, nnz=nnz, t_ts=t_ts, t_sp=t_sp, ncomp=ncomp))

        # --- multi-component block-diagonal stack (k blocks) ---
        # choose block side so each block is modest; many components.
        block_side = max(8, side // 3)
        k = max(4, side // 2)
        Am, kc = block_diag_poisson(block_side, k, device)
        dofm, nnzm = Am.sparse_shape[0], Am.nnz
        tm_ts = time_call(lambda: Am.connected_components(), reps=reps, device=device)
        Amcsr = to_scipy_csr(Am)
        tm_sp = time_call(lambda: scipy_cc(Amcsr, directed=False), reps=reps, device="cpu")
        _, ncm = Am.connected_components()
        assert ncm == kc, f"expected {kc} components, got {ncm}"
        rows_multi.append(dict(dof=dofm, nnz=nnzm, t_ts=tm_ts, t_sp=tm_sp, ncomp=ncm))
    return dict(single=rows_single, multi=rows_multi)


def bench_spmv(sides, device, reps):
    rows = []
    for side in sides:
        p = d.poisson_2d(side)
        A = make_sparsetensor(p, device)
        dof, nnz = A.sparse_shape[0], A.nnz
        x = torch.randn(dof, dtype=A.dtype, device=device)
        t = time_call(lambda: A @ x, reps=reps, device=device, warmup=2)
        rows.append(dict(dof=dof, nnz=nnz, time=t, tput=dof / t))
    return rows


def bench_cg(sides, device, reps):
    rows = []
    for side in sides:
        p = d.poisson_2d(side)
        val, row, col, shape = p.coo()
        val, row, col = val.to(device), row.to(device), col.to(device)
        b = p.rhs.to(device)
        A = SparseTensor(val, row, col, shape)
        dof, nnz = shape[0], A.nnz

        def run():
            return spsolve(val, row, col, shape, b,
                           backend="pytorch", method="cg", is_spd=True,
                           tol=1e-8, maxiter=20000)

        # warmup once also lets us measure residual / accuracy
        x = run()
        resid = float((A @ x - b).norm() / (b.norm() + 1e-30))
        t = time_call(run, reps=reps, device=device, warmup=0)
        rows.append(dict(dof=dof, nnz=nnz, time=t, tput=dof / t, resid=resid))
    return rows


def bench_lu(sides, device, reps):
    rows = []
    for side in sides:
        p = d.poisson_2d(side)
        val, row, col, shape = p.coo()
        b = p.rhs
        A = SparseTensor(val, row, col, shape)
        dof, nnz = shape[0], A.nnz

        def run():
            return spsolve(val, row, col, shape, b, backend="scipy", method="lu")

        x = run()
        resid = float((A @ x - b).norm() / (b.norm() + 1e-30))
        t = time_call(run, reps=reps, device="cpu", warmup=0)
        rows.append(dict(dof=dof, nnz=nnz, time=t, tput=dof / t, resid=resid))
    return rows


def bench_eigsh(sides, device, reps):
    rows = []
    for side in sides:
        p = d.laplacian_2d(side)
        A = make_sparsetensor(p, device)
        dof, nnz = A.sparse_shape[0], A.nnz

        def run():
            return A.eigsh(k=6, which="SA", return_eigenvectors=False)

        t = time_call(run, reps=reps, device=device, warmup=1)
        rows.append(dict(dof=dof, nnz=nnz, time=t, tput=dof / t))
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def loglog_plot(out, fname, title, series):
    """series: list of (label, dofs, times, slope) -> log-log time vs DOF."""
    plt.figure(figsize=(7, 5))
    for label, dofs, times, slope in series:
        lbl = f"{label} (slope={slope:.2f})" if slope is not None else label
        plt.loglog(dofs, times, "o-", label=lbl)
    # reference O(N) and O(N^2) guide lines anchored to first series
    if series:
        d0 = np.array(series[0][1], dtype=float)
        t0 = np.array(series[0][2], dtype=float)
        anchor = t0[0] / d0[0]
        plt.loglog(d0, anchor * d0, "k--", alpha=0.3, label="O(N) ref")
        plt.loglog(d0, anchor * d0[0] * (d0 / d0[0]) ** 2, "k:", alpha=0.3, label="O(N^2) ref")
    plt.xlabel("DOF (N)")
    plt.ylabel("time [s]")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = out / fname
    plt.savefig(path, dpi=110)
    plt.close()
    return path


def throughput_plot(out, fname, series):
    """series: list of (label, dofs, tputs)."""
    plt.figure(figsize=(7, 5))
    for label, dofs, tputs in series:
        plt.loglog(dofs, tputs, "o-", label=label)
    plt.xlabel("DOF (N)")
    plt.ylabel("throughput [DOF / s]")
    plt.title("Throughput vs DOF (all algorithms)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = out / fname
    plt.savefig(path, dpi=110)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_table(name, rows, cols):
    print(f"\n=== {name} ===")
    header = "  ".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                line.append(f"{v:>14.4g}")
            else:
                line.append(f"{v:>14}")
        print("  ".join(line))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--quick", action="store_true", help="fast smoke sweep")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu")
        device = "cpu"
    torch.set_num_threads(max(1, torch.get_num_threads()))

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out
    out.mkdir(parents=True, exist_ok=True)

    # ---- sweeps ----
    # poisson_2d(side) gives DOF = side**2, nnz ~ 5*DOF.
    # full sweep: side 32..448  => DOF ~1e3 .. ~2e5
    if args.quick:
        sides_main = [16, 24, 32, 48]          # DOF 256 .. 2304
        sides_eig = [16, 24, 32]
        sides_lu = [16, 24, 32, 48]
        reps = 2
    else:
        sides_main = [32, 48, 64, 96, 128, 192, 256, 350, 448]  # DOF ~1e3..2e5
        sides_eig = [32, 48, 64, 96, 128, 192, 256]             # eigsh is pricier
        sides_lu = [32, 48, 64, 96, 128, 192, 256, 350]         # LU fill-in heavy
        reps = 3

    print(f"device={device}  quick={args.quick}  reps={reps}  out={out}")
    print(f"torch threads = {torch.get_num_threads()}")

    results = {}
    t_start = time.perf_counter()

    # 1. connected components
    print("\n[1/5] connected_components (torch-sla vs scipy) ...")
    cc = bench_connected_components(sides_main, device, reps)
    results["cc"] = cc

    # 2. spmv
    print("[2/5] spmv (A @ x) ...")
    spmv = bench_spmv(sides_main, device, reps)
    results["spmv"] = spmv

    # 3. cg
    print("[3/5] CG solve (pytorch backend) ...")
    cg = bench_cg(sides_main, device, reps)
    results["cg"] = cg

    # 4. lu
    print("[4/5] direct LU solve (scipy backend) ...")
    lu = bench_lu(sides_lu, device, reps)
    results["lu"] = lu

    # 5. eigsh
    print("[5/5] eigsh (k=6, SA) on laplacian_2d ...")
    eig = bench_eigsh(sides_eig, device, reps)
    results["eigsh"] = eig

    elapsed = time.perf_counter() - t_start
    print(f"\nMeasurement wall time: {elapsed:.1f}s")

    # ---------------------------------------------------------------- fits
    slopes = {}

    # connected components fits (single + multi, both torch & scipy)
    cc_single, cc_multi = cc["single"], cc["multi"]
    s_ts_single, _ = fit_slope([r["dof"] for r in cc_single], [r["t_ts"] for r in cc_single])
    s_sp_single, _ = fit_slope([r["dof"] for r in cc_single], [r["t_sp"] for r in cc_single])
    s_ts_multi, _ = fit_slope([r["dof"] for r in cc_multi], [r["t_ts"] for r in cc_multi])
    s_sp_multi, _ = fit_slope([r["dof"] for r in cc_multi], [r["t_sp"] for r in cc_multi])
    slopes["cc_torch_single"] = s_ts_single
    slopes["cc_scipy_single"] = s_sp_single
    slopes["cc_torch_multi"] = s_ts_multi
    slopes["cc_scipy_multi"] = s_sp_multi

    def simple_slope(rows):
        return fit_slope([r["dof"] for r in rows], [r["time"] for r in rows])[0]

    slopes["spmv"] = simple_slope(spmv)
    slopes["cg"] = simple_slope(cg)
    slopes["lu"] = simple_slope(lu)
    slopes["eigsh"] = simple_slope(eig)

    # ---------------------------------------------------------------- plots
    paths = {}
    paths["cc"] = loglog_plot(
        out, "connected_components_scaling.png",
        "connected_components: time vs DOF (torch-sla vs scipy)",
        [
            ("torch-sla single-comp", [r["dof"] for r in cc_single], [r["t_ts"] for r in cc_single], s_ts_single),
            ("scipy single-comp", [r["dof"] for r in cc_single], [r["t_sp"] for r in cc_single], s_sp_single),
            ("torch-sla multi-comp", [r["dof"] for r in cc_multi], [r["t_ts"] for r in cc_multi], s_ts_multi),
            ("scipy multi-comp", [r["dof"] for r in cc_multi], [r["t_sp"] for r in cc_multi], s_sp_multi),
        ],
    )
    paths["spmv"] = loglog_plot(
        out, "spmv_scaling.png", "SpMV (A @ x): time vs DOF",
        [("spmv", [r["dof"] for r in spmv], [r["time"] for r in spmv], slopes["spmv"])],
    )
    paths["cg"] = loglog_plot(
        out, "cg_scaling.png", "CG solve: time vs DOF (poisson_2d, SPD)",
        [("cg", [r["dof"] for r in cg], [r["time"] for r in cg], slopes["cg"])],
    )
    paths["lu"] = loglog_plot(
        out, "lu_scaling.png", "Direct LU (scipy): time vs DOF (poisson_2d)",
        [("lu", [r["dof"] for r in lu], [r["time"] for r in lu], slopes["lu"])],
    )
    paths["eigsh"] = loglog_plot(
        out, "eigsh_scaling.png", "eigsh (k=6, SA): time vs DOF (laplacian_2d)",
        [("eigsh", [r["dof"] for r in eig], [r["time"] for r in eig], slopes["eigsh"])],
    )
    paths["throughput"] = throughput_plot(
        out, "throughput.png",
        [
            ("connected_components (multi)", [r["dof"] for r in cc_multi], [r["dof"] / r["t_ts"] for r in cc_multi]),
            ("spmv", [r["dof"] for r in spmv], [r["tput"] for r in spmv]),
            ("cg", [r["dof"] for r in cg], [r["tput"] for r in cg]),
            ("lu", [r["dof"] for r in lu], [r["tput"] for r in lu]),
            ("eigsh", [r["dof"] for r in eig], [r["tput"] for r in eig]),
        ],
    )

    # ---------------------------------------------------------------- tables
    print_table("connected_components -- single component (poisson_2d grid)",
                cc_single, ["dof", "nnz", "ncomp", "t_ts", "t_sp"])
    print(f"  fitted slope (torch-sla) = {s_ts_single:.3f}   (scipy) = {s_sp_single:.3f}")

    print_table("connected_components -- multi component (block-diagonal stack)",
                cc_multi, ["dof", "nnz", "ncomp", "t_ts", "t_sp"])
    print(f"  fitted slope (torch-sla) = {s_ts_multi:.3f}   (scipy) = {s_sp_multi:.3f}")

    print_table("spmv (A @ x)", spmv, ["dof", "nnz", "time", "tput"])
    print(f"  fitted slope = {slopes['spmv']:.3f}")

    print_table("CG solve", cg, ["dof", "nnz", "time", "tput", "resid"])
    print(f"  fitted slope = {slopes['cg']:.3f}")

    print_table("direct LU (scipy)", lu, ["dof", "nnz", "time", "tput", "resid"])
    print(f"  fitted slope = {slopes['lu']:.3f}")

    print_table("eigsh (k=6, SA)", eig, ["dof", "nnz", "time", "tput"])
    print(f"  fitted slope = {slopes['eigsh']:.3f}")

    # ---------------------------------------------------------------- analysis
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS / IMPLEMENTATION VERDICT")
    print("=" * 70)
    flags = []

    # connected_components
    print("\n[connected_components]")
    print(f"  torch-sla slope: single={s_ts_single:.2f}  multi={s_ts_multi:.2f}  "
          f"(expect ~1.0-1.3, linear in DOF/nnz)")
    print(f"  scipy slope:     single={s_sp_single:.2f}  multi={s_sp_multi:.2f}")
    # speed ratio at largest size
    r_big_s = cc_single[-1]
    r_big_m = cc_multi[-1]
    ratio_s = r_big_s["t_ts"] / r_big_s["t_sp"]
    ratio_m = r_big_m["t_ts"] / r_big_m["t_sp"]
    print(f"  torch/scipy time ratio @largest: single={ratio_s:.1f}x  multi={ratio_m:.1f}x "
          f"(>1 means torch slower)")
    if max(s_ts_single, s_ts_multi) > 1.4:
        flags.append(
            f"connected_components is SUPERLINEAR (slope up to "
            f"{max(s_ts_single, s_ts_multi):.2f}); label-propagation/pointer-jumping "
            f"loop in torch_sla/sparse_tensor/graph.py (_connected_components_labels "
            f"while-loop) likely runs too many rounds or does O(N) work per round.")
    if max(ratio_s, ratio_m) > 8.0:
        flags.append(
            f"connected_components is much slower than scipy.csgraph "
            f"(up to {max(ratio_s, ratio_m):.0f}x). The pure-torch Shiloach-Vishkin "
            f"loop in graph.py has high per-round overhead (each round: scatter_reduce "
            f"+ pointer-jump + torch.equal over all nodes); investigate round count "
            f"and whether torch.equal early-exit dominates.")
    if not flags:
        print("  OK: roughly linear and competitive with scipy.")

    # spmv
    print("\n[spmv]")
    print(f"  slope={slopes['spmv']:.2f} (expect ~1.0; nnz ~ 5*DOF so linear)")
    if slopes["spmv"] > 1.4:
        flags.append(f"spmv slope {slopes['spmv']:.2f} > 1.4: should be ~linear; "
                     f"check CSR build/caching in matmul path.")
    else:
        print("  OK: linear.")

    # cg
    print("\n[cg]")
    print(f"  slope={slopes['cg']:.2f}")
    print("  Poisson 2D: kappa ~ O(N), CG iters ~ O(sqrt(kappa)) ~ O(N^0.5),")
    print("  cost/iter ~ O(nnz) ~ O(N), so unpreconditioned CG time ~ O(N^1.5).")
    resid_max = max(r["resid"] for r in cg)
    print(f"  max relative residual across sizes = {resid_max:.2e}")
    if slopes["cg"] > 1.9:
        flags.append(f"CG slope {slopes['cg']:.2f} > 1.9: steeper than the O(N^1.5) "
                     f"expected for unpreconditioned CG on Poisson; check iteration cap / "
                     f"preconditioner effectiveness.")
    if resid_max > 1e-4:
        flags.append(f"CG max residual {resid_max:.1e} is large: not converging to "
                     f"requested tolerance within maxiter.")
    if slopes["cg"] <= 1.9 and resid_max <= 1e-4:
        print("  OK: consistent with ~O(N^1.5) unpreconditioned CG and converging.")

    # lu
    print("\n[lu]")
    print(f"  slope={slopes['lu']:.2f} (sparse LU on 2D grid: fill-in gives ~O(N^1.5),")
    print("  superlinear is EXPECTED; only flag if wildly off e.g. >2.2)")
    if slopes["lu"] > 2.2:
        flags.append(f"LU slope {slopes['lu']:.2f} > 2.2: worse than nested-dissection "
                     f"O(N^1.5) for 2D; check ordering / scipy splu config.")
    else:
        print("  OK: superlinear as expected for sparse direct factorization.")

    # eigsh
    print("\n[eigsh]")
    print(f"  slope={slopes['eigsh']:.2f} (iterative LOBPCG/Lanczos; per-iter ~O(nnz),")
    print("  iters grow slowly; expect mildly superlinear ~1.0-1.8)")
    if slopes["eigsh"] > 2.2:
        flags.append(f"eigsh slope {slopes['eigsh']:.2f} > 2.2: steeper than expected "
                     f"for an iterative eigensolver; check restart / convergence behavior.")
    else:
        print("  OK: within expected range for iterative eigensolver.")

    print("\n" + "-" * 70)
    if flags:
        print("FLAGGED POTENTIAL IMPLEMENTATION ISSUES:")
        for i, f in enumerate(flags, 1):
            print(f"  {i}. {f}")
    else:
        print("No implementation issues flagged. All slopes within expected ranges.")

    print("\nGenerated plots:")
    for k, v in paths.items():
        print(f"  {k:12s} -> {v}")

    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
