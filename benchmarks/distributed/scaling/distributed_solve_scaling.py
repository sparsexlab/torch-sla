#!/usr/bin/env python
"""Canonical distributed *linear-solve* scaling benchmark for torch-sla.

This is the single, reproducible benchmark for **distributed
linear-solve scaling**. It launches under ``torchrun`` (one rank per
process), builds a row-sharded :class:`~torch_sla.DSparseTensor` from a
reproducible Poisson problem, runs the unified distributed
:func:`~torch_sla.solve`, and records the metrics that matter for a
scaling study: wall-clock solve time, the relative residual
``||A x - b|| / ||b||`` (correctness gate), and parallel efficiency.

It is meant to be **handed off**: a newcomer should be able to run it on
a multi-GPU node (or multi-process CPU) without tribal knowledge. See
``docs/source/distributed_scaling.rst`` for launch commands, what the
metrics mean, and how to extend it.

Three experiment modes (pick with ``--mode``):

* ``weak``        fixed DOF *per rank*; total DOF grows with the world
                  size. Ideal solve time is **flat** as ranks grow.
* ``strong``      fixed *total* DOF; world size grows. Ideal solve time
                  *halves* each time you double the ranks.
* ``throughput``  DOF processed per second vs ranks (derived from the
                  same timed solve). Ideal curve grows linearly.

All three are measured from the same instrumented solve, so a single
launch records everything for the current world size; the plot/JSON
accumulate across world sizes (re-run with different ``--nproc_per_node``
and the results append into one file).

Backend is chosen automatically: ``nccl`` on CUDA, ``gloo`` on CPU. The
problem, RHS, and partition are seeded so two runs at the same world
size produce identical matrices.

Quick reference
---------------
    # 1 GPU / 1 process (smoke test, establishes the p=1 baseline)
    torchrun --standalone --nproc_per_node=1 \
        benchmarks/distributed/scaling/distributed_solve_scaling.py \
        --mode weak --dof-per-rank 40000

    # 4 GPUs, weak scaling (run after the p=1 baseline so efficiency
    # has something to divide by)
    torchrun --standalone --nproc_per_node=4 \
        benchmarks/distributed/scaling/distributed_solve_scaling.py \
        --mode weak --dof-per-rank 40000

    # render the accumulated curve once you have several world sizes
    python benchmarks/distributed/scaling/distributed_solve_scaling.py --plot-only

Outputs
-------
    benchmarks/results/distributed_solve_scaling.json     # all rows
    assets/benchmarks/distributed_solve_scaling.png       # scaling plot
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Repo root on sys.path so ``torch_sla`` imports when run via torchrun
# from the repo root or from this directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_JSON = _REPO_ROOT / "benchmarks" / "results" / "distributed_solve_scaling.json"
DEFAULT_PLOT = _REPO_ROOT / "assets" / "benchmarks" / "distributed_solve_scaling.png"

SEED = 1234

# Plot palette / style — matches the existing distributed benchmarks
# (benchmark_distributed_scaling.py, benchmark_distributed.py).
_MODE_COLOR = {"weak": "#2E86AB", "strong": "#E94F37", "throughput": "#3CB371"}


# --------------------------------------------------------------------------- #
# Problem construction
# --------------------------------------------------------------------------- #
def _grid_side(dof: int, dim: int) -> int:
    """Largest grid side ``m`` with ``m**dim <= dof`` (>=2)."""
    side = max(2, int(round(dof ** (1.0 / dim))))
    return side


def build_problem(dof_target: int, dim: int):
    """Build a reproducible Poisson problem with ~``dof_target`` DOF.

    Returns ``(SparseTensor A, b_global, side, actual_dof)``. The matrix
    and RHS are identical on every rank (each rank rebuilds the global
    problem, then ``DSparseTensor.partition`` shards it deterministically).
    """
    import torch_sla.datasets as datasets
    from torch_sla import SparseTensor

    torch.manual_seed(SEED)
    if dim == 2:
        side = _grid_side(dof_target, 2)
        prob = datasets.poisson_2d(side)
    elif dim == 3:
        side = _grid_side(dof_target, 3)
        prob = datasets.poisson_3d(side)
    else:
        raise ValueError(f"--dim must be 2 or 3, got {dim}")

    n = prob.shape[0]
    A = SparseTensor(prob.val.to(torch.float64), prob.row, prob.col, prob.shape)
    # RHS for a *scaling* study: a seeded RANDOM vector, NOT the dataset's
    # smooth manufactured ``f``. The manufactured RHS is dominated by a few
    # low-frequency modes (nearly an eigenvector of A), so CG converges in
    # ~O(1) iterations regardless of N -- the solve then finishes in ~1 ms
    # even at 40M DOF and measures fixed overhead, not the solver. A random
    # RHS has full spectral content, forcing CG to do its real ~O(sqrt(N))
    # work. Re-seeded here so every rank builds an identical b (each rank
    # constructs the global problem, then partitions/scatters it).
    torch.manual_seed(SEED + 1)
    b_global = torch.randn(n, dtype=torch.float64).contiguous()
    return A, b_global, side, n


def _grid_coords(side: int, dim: int) -> torch.Tensor:
    """Geometric coordinates of the Poisson grid nodes, for the
    coordinate (RCB) partitioner. Shape ``[side**dim, dim]``."""
    axes = [torch.arange(side, dtype=torch.float64) for _ in range(dim)]
    grids = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([g.flatten() for g in grids], dim=1).contiguous()


# Map the user-facing partitioner names to the library's method strings.
# "coordinate" -> recursive-coordinate-bisection (needs coords).
_PARTITIONER_ALIASES = {
    "simple": ("simple", False),
    "coordinate": ("rcb", True),
    "metis": ("metis", False),
}


# --------------------------------------------------------------------------- #
# One measured solve (runs inside every rank)
# --------------------------------------------------------------------------- #
def run_once(args, rank: int, world_size: int, device: str) -> dict:
    """Build, partition, solve, measure. Returns rank-0's metrics dict."""
    import torch.distributed as dist
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError:  # torch < 2.2
        from torch.distributed._tensor.device_mesh import init_device_mesh
    from torch_sla import DSparseTensor, solve

    # ---- pick the DOF target for this mode ------------------------------- #
    if args.mode == "strong":
        dof_target = args.total_dof
    else:  # weak / throughput: fixed DOF per rank
        dof_target = args.dof_per_rank * world_size

    A, b_global, side, n = build_problem(dof_target, args.dim)
    A = A.to(device)
    b_global = b_global.to(device)

    method_str, needs_coords = _PARTITIONER_ALIASES[args.partitioner]
    coords = None
    if needs_coords:
        coords = _grid_coords(side, args.dim).to(device)

    # init_device_mesh wants a device *type* ("cuda" / "cpu"), not an indexed
    # device ("cuda:0") -- newer torch rejects the latter. Each rank's GPU is
    # already pinned via torch.cuda.set_device(local_rank) above.
    mesh = init_device_mesh(torch.device(device).type, (world_size,))
    D = DSparseTensor.partition(A, mesh, partition_method=method_str, coords=coords)
    b_dt = D.scatter(b_global)

    owned = int(D.spec.placement.partition.owned_nodes.numel())

    solve_kw = dict(method=args.method, atol=args.atol, rtol=args.rtol,
                    maxiter=args.maxiter)

    # ---- warmup (also forces lazy CSR / buffer allocation) --------------- #
    warm_kw = dict(solve_kw, maxiter=min(args.warmup_iters, args.maxiter))
    for _ in range(args.warmup):
        try:
            _ = solve(D, b_dt, **warm_kw)
        except Exception:
            break
    dist.barrier()
    if device != "cpu":
        torch.cuda.synchronize()

    # ---- timed solve (best of --repeat) ---------------------------------- #
    times = []
    x_dt = None
    for _ in range(args.repeat):
        dist.barrier()
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_dt = solve(D, b_dt, **solve_kw)
        if device != "cpu":
            torch.cuda.synchronize()
        dist.barrier()
        times.append(time.perf_counter() - t0)
    solve_time = min(times)

    # ---- correctness gate: relative residual ||A x - b|| / ||b|| --------- #
    # Public ops only: ``D @ x_dt`` is the distributed matvec. We avoid
    # ``full_tensor()`` (its generic redistribute path assumes uniform
    # shard sizes and trips a gloo size-mismatch when owned-row counts
    # differ across ranks). Owned rows are disjoint, so the global 2-norm
    # is sqrt(sum_ranks ||local owned slice||^2) via a single all_reduce.
    r_dt = b_dt - D @ x_dt
    r_local = r_dt.to_local()
    b_local = b_dt.to_local()
    r_sq = (r_local.double() ** 2).sum()
    b_sq = (b_local.double() ** 2).sum()
    if dist.is_initialized() and world_size > 1:
        dist.all_reduce(r_sq, op=dist.ReduceOp.SUM)
        dist.all_reduce(b_sq, op=dist.ReduceOp.SUM)
    rel_res = float((r_sq.sqrt() / (b_sq.sqrt() + 1e-30)).item())

    # ---- per-GPU peak memory (max over ranks) ---------------------------- #
    if device != "cpu":
        local_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        t = torch.tensor([local_peak], device=device)
        gathered = [torch.zeros(1, device=device) for _ in range(world_size)]
        dist.all_gather(gathered, t)
        peak_mem_gb = max(float(g.item()) for g in gathered)
    else:
        peak_mem_gb = 0.0

    return {
        "mode": args.mode,
        "world_size": world_size,
        "dof_global": int(n),
        "dof_per_rank": int(round(n / world_size)),
        "owned_rows_rank0": owned,
        "global_nnz": int(D.global_nnz()),
        "side": int(side),
        "dim": args.dim,
        "method": args.method,
        "partitioner": args.partitioner,
        "atol": args.atol,
        "rtol": args.rtol,
        "maxiter": args.maxiter,
        # NOTE: the distributed Krylov shard solvers return only the
        # solution vector; iteration count is not exposed by the public
        # API, so we record it as null. The residual below is the
        # authoritative correctness signal.
        "iterations": None,
        "solve_time_s": solve_time,
        "rel_residual": rel_res,
        "peak_mem_gb": peak_mem_gb,
        "device": device,
        "backend": dist.get_backend(),
        "torch": torch.__version__,
        "timestamp": time.time(),
    }


# --------------------------------------------------------------------------- #
# Results persistence (one accumulating JSON across world sizes)
# --------------------------------------------------------------------------- #
def append_result(json_path: Path, row: dict) -> list:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if json_path.exists():
        try:
            rows = json.loads(json_path.read_text())
        except (ValueError, OSError):
            rows = []
    # Replace any prior row with the same (mode, world_size, dof_per_rank/total).
    key = (row["mode"], row["world_size"], row["dof_per_rank"], row["partitioner"],
           row["method"])
    rows = [r for r in rows
            if (r.get("mode"), r.get("world_size"), r.get("dof_per_rank"),
                r.get("partitioner"), r.get("method")) != key]
    rows.append(row)
    json_path.write_text(json.dumps(rows, indent=2))
    return rows


def _efficiency(rows_for_mode: list) -> list:
    """Annotate rows (one mode) with parallel efficiency vs the p=1 baseline.

    * weak:        efficiency = T(1) / T(p)             (ideal = 1, time flat)
    * strong:      efficiency = T(1) / (p * T(p))       (ideal = 1)
    * throughput:  efficiency = (thr(p)/p) / thr(1)     (ideal = 1)
    """
    rows = sorted(rows_for_mode, key=lambda r: r["world_size"])
    base = next((r for r in rows if r["world_size"] == 1), None)
    t1 = base["solve_time_s"] if base else None
    for r in rows:
        p = r["world_size"]
        t = r["solve_time_s"]
        r["throughput_dof_s"] = r["dof_global"] / t if t else None
        if t1 is None:
            r["efficiency"] = None
            r["speedup"] = None
            continue
        if r["mode"] == "weak":
            r["speedup"] = None
            r["efficiency"] = t1 / t
        elif r["mode"] == "strong":
            r["speedup"] = t1 / t
            r["efficiency"] = (t1 / t) / p
        else:  # throughput
            thr1 = base["dof_global"] / t1
            r["speedup"] = None
            r["efficiency"] = (r["throughput_dof_s"] / p) / thr1 if thr1 else None
    return rows


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot(rows: list, plot_path: Path, no_title: bool = False) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed -- skipping PNG. The JSON is "
              "saved; run `pip install matplotlib` then re-render offline with "
              "`--plot-only`.")
        return plot_path

    modes = [m for m in ("weak", "strong", "throughput")
             if any(r["mode"] == m for r in rows)]
    if not modes:
        print("[plot] no rows to plot.")
        return plot_path

    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5),
                             squeeze=False)
    axes = axes[0]

    for ax, mode in zip(axes, modes):
        mrows = _efficiency([r for r in rows if r["mode"] == mode])
        ranks = [r["world_size"] for r in mrows]
        color = _MODE_COLOR[mode]

        if mode == "weak":
            ys = [r["solve_time_s"] for r in mrows]
            ax.plot(ranks, ys, "o-", color=color, lw=2.2, ms=8,
                    markeredgecolor="white", markeredgewidth=1, label="measured")
            if ys:
                ax.axhline(ys[0], ls="--", color=color, alpha=0.4,
                           label="ideal (flat)")
            ax.set_ylabel("solve time (s)  [lower = better]", fontsize=11)
            if not no_title:
                ax.set_title("Weak scaling\n(fixed DOF/rank)", fontsize=12,
                             fontweight="bold")
        elif mode == "strong":
            sp = [r["speedup"] for r in mrows]
            ax.plot(ranks, ranks, "k--", lw=1.4, alpha=0.6, label="ideal linear")
            ax.plot(ranks, sp, "o-", color=color, lw=2.2, ms=8,
                    markeredgecolor="white", markeredgewidth=1, label="measured")
            ax.set_ylabel("speedup  T(1) / T(p)", fontsize=11)
            if not no_title:
                ax.set_title("Strong scaling\n(fixed total DOF)", fontsize=12,
                             fontweight="bold")
        else:  # throughput
            thr = [r["throughput_dof_s"] for r in mrows]
            ax.plot(ranks, thr, "o-", color=color, lw=2.2, ms=8,
                    markeredgecolor="white", markeredgewidth=1, label="measured")
            if thr:
                ax.plot(ranks, [thr[0] * p for p in ranks], "--", color=color,
                        alpha=0.4, label="ideal linear")
            ax.set_ylabel("throughput (DOF / s)", fontsize=11)
            if not no_title:
                ax.set_title("Throughput\n(DOF/s vs ranks)", fontsize=12,
                             fontweight="bold")

        ax.set_xlabel("# ranks (world size)", fontsize=11)
        if ranks:
            ax.set_xticks(sorted(set(ranks)))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    if not no_title:
        ex = rows[0]
        fig.suptitle(
            f"torch-sla distributed solve scaling "
            f"({ex['method']}, {ex['partitioner']} partition, "
            f"{ex['device']}/{ex['backend']})",
            fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96) if not no_title else None)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    # Transparent canvas so the PNG drops onto any slide background.
    fig.savefig(plot_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return plot_path


def print_table(rows: list) -> None:
    for mode in ("weak", "strong", "throughput"):
        mrows = _efficiency([r for r in rows if r["mode"] == mode])
        if not mrows:
            continue
        print(f"\n{'=' * 78}\n{mode.upper()} SCALING\n{'=' * 78}")
        print(f"{'ranks':>6} {'DOF(global)':>13} {'DOF/rank':>10} "
              f"{'time(s)':>10} {'rel.res':>10} {'thr(DOF/s)':>13} "
              f"{'efficiency':>11}")
        for r in mrows:
            eff = r.get("efficiency")
            thr = r.get("throughput_dof_s")
            print(f"{r['world_size']:>6} {r['dof_global']:>13,} "
                  f"{r['dof_per_rank']:>10,} {r['solve_time_s']:>10.4f} "
                  f"{r['rel_residual']:>10.2e} "
                  f"{(f'{thr:,.0f}' if thr else '-'):>13} "
                  f"{(f'{eff*100:.1f}%' if eff is not None else '-'):>11}")


# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["weak", "strong", "throughput"],
                    default="weak",
                    help="weak=fixed DOF/rank, strong=fixed total DOF, "
                         "throughput=DOF/s vs ranks")
    ap.add_argument("--dof-per-rank", type=int, default=40000,
                    help="weak/throughput: target DOF per rank")
    ap.add_argument("--total-dof", type=int, default=160000,
                    help="strong: fixed total DOF (split across ranks)")
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3],
                    help="Poisson dimension (2D 5-point / 3D 7-point)")
    ap.add_argument("--partitioner", choices=["simple", "coordinate", "metis"],
                    default="simple",
                    help="row partitioner; 'coordinate' uses grid coords (RCB)")
    ap.add_argument("--method", default="cg",
                    help="distributed Krylov method: cg / bicgstab / gmres / "
                         "minres")
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--warmup-iters", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=3,
                    help="timed solves; the minimum is recorded")
    ap.add_argument("--json", type=str, default=str(DEFAULT_JSON))
    ap.add_argument("--plot", type=str, default=str(DEFAULT_PLOT))
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render plot + table from the existing JSON")
    ap.add_argument("--no-title", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    json_path = Path(args.json)
    plot_path = Path(args.plot)

    # ---- offline plot/report (no torchrun needed) ------------------------ #
    if args.plot_only:
        if not json_path.exists():
            print(f"[plot-only] no results at {json_path}")
            return
        rows = json.loads(json_path.read_text())
        print_table(rows)
        out = plot(rows, plot_path, no_title=args.no_title)
        print(f"\n[plot-only] wrote {out}")
        return

    # ---- distributed run ------------------------------------------------- #
    import torch.distributed as dist
    if "RANK" not in os.environ:
        prog = "benchmarks/distributed/scaling/distributed_solve_scaling.py"
        print("This benchmark must be launched with torchrun, e.g.:\n"
              f"  torchrun --standalone --nproc_per_node=4 {prog} "
              f"--mode {args.mode}\n"
              "Use --plot-only (plain python) to render the accumulated curve.")
        sys.exit(1)

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if not use_cuda:
        # Avoid BLAS oversubscription when many ranks share one CPU.
        torch.set_num_threads(1)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        torch.cuda.reset_peak_memory_stats()
    else:
        device = "cpu"

    if rank == 0:
        print("=" * 78)
        print("torch-sla CANONICAL distributed solve scaling")
        print("=" * 78)
        print(f"  mode        : {args.mode}")
        print(f"  world size  : {world_size}")
        print(f"  device      : {device}  (backend={backend})")
        print(f"  method      : {args.method}   partitioner: {args.partitioner}")
        print(f"  dim         : {args.dim}D Poisson")
        if args.mode == "strong":
            print(f"  total DOF   : {args.total_dof:,}")
        else:
            print(f"  DOF/rank    : {args.dof_per_rank:,} "
                  f"(global ~{args.dof_per_rank * world_size:,})")
        print(f"  atol/rtol   : {args.atol} / {args.rtol}   maxiter={args.maxiter}")
        print("=" * 78, flush=True)

    try:
        row = run_once(args, rank, world_size, device)
    finally:
        dist.destroy_process_group()

    if rank != 0:
        return

    print(f"\nworld_size={row['world_size']}  DOF={row['dof_global']:,}  "
          f"time={row['solve_time_s']:.4f}s  "
          f"rel_res={row['rel_residual']:.2e}  "
          f"peak_mem={row['peak_mem_gb']:.3f}GB/gpu")
    if row["rel_residual"] > max(1e-4, 100 * row["rtol"]):
        print(f"  WARNING: relative residual {row['rel_residual']:.2e} is "
              f"high — solve may not have converged (raise --maxiter or "
              f"check the partitioner).")

    rows = append_result(json_path, row)
    print(f"\n[json] {json_path}  ({len(rows)} rows total)")
    print_table(rows)
    out = plot(rows, plot_path, no_title=args.no_title)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
