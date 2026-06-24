#!/usr/bin/env python
"""Distributed strong/weak scaling benchmark for torch-sla (CPU / gloo).

macor7 has a single iGPU, so this benchmark stresses the *distributed*
code paths via **multiprocess CPU** with the ``gloo`` backend, varying the
world size (number of ranks) over ``--ranks 1,2,4,8``. It spawns one
process group per rank-count via ``torch.multiprocessing`` (spawn) and a
``Queue`` to ship per-rank timings back to the parent.

It measures only the already-stable distributed ops:

* distributed **matvec**  -- ``D @ x``  (many reps)
* distributed **solve**   -- unified ``solve(D, b, method="cg")``
                             (backend=pytorch Krylov)
* distributed **eigsh**   -- ``D.eigsh(k, which="SA")``  (backend=lobpcg)

Two experiments:

1. **Strong scaling** -- a FIXED problem size (``poisson_2d(side)`` with
   ``side`` ~256 -> ~65k DOF), world size varied over ``--ranks``. We
   report ``speedup = T(1)/T(p)`` and ``efficiency = speedup / p``.

2. **Weak scaling** -- DOF grows ~proportionally with the rank count so
   that per-rank work stays fixed. Ideal time-per-op curve is flat.

NOTE on interpretation: gloo-on-CPU has no real network/GPU and pays
``all_reduce`` / halo-exchange overhead on every Krylov / LOBPCG
iteration with no compute to hide it behind. Strong-scaling speedup is
therefore expected to be limited and often *negative* (slowdown) once
communication dominates -- the *shape* of the curve and the point where
comm overhead takes over is the signal, and it is reported honestly.

Usage
-----
    PY=~/.venvs/torchsla/bin/python
    $PY benchmarks/benchmark_distributed_scaling.py \
        --ranks 1,2,4 --out benchmarks/results
    $PY benchmarks/benchmark_distributed_scaling.py --quick   # fast smoke

Outputs (to ``--out``):
    dist_strong_scaling.png   speedup vs #ranks (+ ideal-linear ref)
    dist_weak_scaling.png     time/op vs #ranks (ideal = flat)
    dist_throughput.png       DOF/s vs #ranks
    dist_scaling_results.json full numbers
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --------------------------------------------------------------------------- #
# Worker: runs inside each spawned rank.
# --------------------------------------------------------------------------- #
def _worker(rank: int, world_size: int, port: int,
            cfg: dict, out_queue: "mp.Queue") -> None:
    """Run the full op suite for one (world_size, problem) on one rank.

    Only rank 0's timing dict is shipped back, but every rank participates
    in the collectives. Each rank is pinned to a single BLAS thread so the
    multiprocess comparison is apples-to-apples (no oversubscription).
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    # Avoid BLAS oversubscription: p ranks * T threads must not exceed cores.
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch_sla import SparseTensor, DSparseTensor, solve
        import torch_sla.datasets as d

        side = cfg["side"]
        dtype = torch.float64

        # ---- build global problem (cheap, every rank builds identical) ---- #
        prob = d.poisson_2d(side)
        n = prob.shape[0]
        A = SparseTensor(prob.val.to(dtype), prob.row, prob.col, prob.shape)

        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A, mesh, partition_method="simple")

        b_global = torch.ones(n, dtype=dtype)
        b_dt = D.scatter(b_global)
        x_dt = b_dt.clone()  # a valid Shard(0) vector for matvec

        reps_mv = cfg["reps_matvec"]
        maxiter = cfg["maxiter"]
        k = cfg["eig_k"]
        eig_maxiter = cfg["eig_maxiter"]

        timings: dict = {}

        # ---- distributed matvec: D @ x (many reps) ------------------------ #
        # warmup
        for _ in range(cfg["warmup"]):
            _ = D @ x_dt
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(reps_mv):
            y = D @ x_dt
        dist.barrier()
        mv_total = time.perf_counter() - t0
        timings["matvec_total_s"] = mv_total
        timings["matvec_per_call_ms"] = mv_total / reps_mv * 1e3

        # ---- distributed solve: unified CG ------------------------------- #
        solve_kw = dict(method="cg", atol=1e-10, rtol=0.0, maxiter=maxiter)
        try:
            _ = solve(D, b_dt, **dict(solve_kw, maxiter=min(20, maxiter)))  # warmup
        except Exception:
            pass
        dist.barrier()
        t0 = time.perf_counter()
        try:
            xs = solve(D, b_dt, **solve_kw)
            dist.barrier()
            solve_s = time.perf_counter() - t0
            # residual via public ops
            r_dt = b_dt - D @ xs
            rel_res = float((r_dt.full_tensor().norm()
                             / (b_dt.full_tensor().norm() + 1e-30)).item())
            timings["solve_cg_s"] = solve_s
            timings["solve_cg_residual"] = rel_res
        except Exception as e:  # pragma: no cover - report, don't crash suite
            dist.barrier()
            timings["solve_cg_s"] = None
            timings["solve_cg_error"] = str(e)[:200]

        # ---- distributed eigsh: LOBPCG, smallest-algebraic --------------- #
        # eigsh returns (eigenvalues, eigenvectors); request vectors so the
        # LOBPCG core has a well-defined return path.
        try:
            _ = D.eigsh(k=k, which="SA", maxiter=min(20, eig_maxiter),
                        tol=1e-6, return_eigenvectors=True)  # warmup
        except Exception:
            pass
        dist.barrier()
        t0 = time.perf_counter()
        try:
            evals, _evecs = D.eigsh(k=k, which="SA", maxiter=eig_maxiter,
                                    tol=1e-7, return_eigenvectors=True)
            dist.barrier()
            eig_s = time.perf_counter() - t0
            timings["eigsh_s"] = eig_s
            ev = evals.tolist() if hasattr(evals, "tolist") else list(evals)
            timings["eigsh_smallest"] = float(min(ev)) if ev else None
        except Exception as e:  # pragma: no cover
            dist.barrier()
            timings["eigsh_s"] = None
            timings["eigsh_error"] = str(e)[:200]

        if rank == 0:
            timings["world_size"] = world_size
            timings["dof"] = int(n)
            timings["side"] = int(side)
            timings["nnz"] = int(A.values.numel())
            timings["reps_matvec"] = reps_mv
            timings["maxiter"] = maxiter
            timings["eig_k"] = k
            out_queue.put(timings)
    finally:
        dist.destroy_process_group()


def _run_point(world_size: int, cfg: dict, port: int) -> dict:
    """Spawn ``world_size`` ranks for one config, return rank-0 timings."""
    ctx = mp.get_context("spawn")
    q: "mp.Queue" = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(r, world_size, port, cfg, q))
             for r in range(world_size)]
    for p in procs:
        p.start()
    result = None
    try:
        result = q.get(timeout=cfg.get("timeout", 1200))
    except Exception:
        result = {"world_size": world_size, "error": "timeout/no-result"}
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
    # surface non-zero exits
    bad = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    if bad and "error" not in result:
        result["worker_exitcodes"] = bad
    return result


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
OP_LABELS = {
    "matvec_total_s": "matvec (D @ x)",
    "solve_cg_s": "solve cg (backend=pytorch)",
    "eigsh_s": "eigsh (backend=lobpcg)",
}
OP_COLORS = {
    "matvec_total_s": "#2E86AB",
    "solve_cg_s": "#E94F37",
    "eigsh_s": "#3CB371",
}


def _plot(results: dict, out_dir: Path, no_title: bool = False) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strong = sorted(results["strong"], key=lambda r: r["world_size"])
    weak = sorted(results["weak"], key=lambda r: r["world_size"])
    paths = []

    # ---- 1. Strong scaling: speedup vs ranks -------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    ranks = [r["world_size"] for r in strong]
    ax.plot(ranks, ranks, "k--", lw=1.5, label="ideal linear", alpha=0.6)
    for op, label in OP_LABELS.items():
        base = next((r for r in strong if r["world_size"] == 1), None)
        if base is None or base.get(op) is None:
            continue
        t1 = base[op]
        xs, ys = [], []
        for r in strong:
            if r.get(op) is None:
                continue
            xs.append(r["world_size"])
            ys.append(t1 / r[op])
        if xs:
            ax.plot(xs, ys, "o-", color=OP_COLORS[op], lw=2.2, ms=8,
                    label=label, markeredgecolor="white", markeredgewidth=1)
    ax.set_xlabel("# ranks (world size)", fontsize=12)
    ax.set_ylabel("speedup  T(1) / T(p)", fontsize=12)
    ax.set_xticks(ranks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    if not no_title and strong:
        dof = strong[0]["dof"]
        ax.set_title(f"Distributed strong scaling (CPU / gloo)\n"
                     f"fixed poisson_2d, DOF={dof:,}", fontsize=12,
                     fontweight="bold")
    fig.tight_layout()
    p = out_dir / "dist_strong_scaling.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # ---- 2. Weak scaling: time/op vs ranks (ideal flat) --------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    ranks_w = [r["world_size"] for r in weak]
    for op, label in OP_LABELS.items():
        xs, ys = [], []
        for r in weak:
            if r.get(op) is None:
                continue
            xs.append(r["world_size"])
            # per-call for matvec; total for solve/eigsh
            if op == "matvec_total_s":
                ys.append(r["matvec_per_call_ms"] / 1e3)
            else:
                ys.append(r[op])
        if xs:
            ax.plot(xs, ys, "o-", color=OP_COLORS[op], lw=2.2, ms=8,
                    label=label, markeredgecolor="white", markeredgewidth=1)
            # ideal-flat reference anchored at p=1
            base = next((r for r in weak if r["world_size"] == 1), None)
            if base is not None and base.get(op) is not None:
                y0 = (base["matvec_per_call_ms"] / 1e3
                      if op == "matvec_total_s" else base[op])
                ax.plot(xs, [y0] * len(xs), "--", color=OP_COLORS[op],
                        alpha=0.4, lw=1.2)
    ax.set_xlabel("# ranks (world size, DOF grows with p)", fontsize=12)
    ax.set_ylabel("time per op (s)  [matvec: per call]", fontsize=12)
    ax.set_xticks(ranks_w)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")
    if not no_title:
        ax.set_title("Distributed weak scaling (CPU / gloo)\n"
                     "per-rank DOF fixed; dashed = ideal (flat)",
                     fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "dist_weak_scaling.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # ---- 3. Throughput: DOF/s vs ranks (strong) ----------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    for op, label in (("solve_cg_s", "solve cg (backend=pytorch)"),
                      ("eigsh_s", "eigsh (backend=lobpcg)")):
        xs, ys = [], []
        for r in strong:
            if r.get(op) is None:
                continue
            xs.append(r["world_size"])
            ys.append(r["dof"] / r[op])
        if xs:
            ax.plot(xs, ys, "o-", color=OP_COLORS[op], lw=2.2, ms=8,
                    label=label, markeredgecolor="white", markeredgewidth=1)
    # matvec throughput = DOF * reps / total time
    xs, ys = [], []
    for r in strong:
        if r.get("matvec_total_s") is None:
            continue
        xs.append(r["world_size"])
        ys.append(r["dof"] * r["reps_matvec"] / r["matvec_total_s"])
    if xs:
        ax.plot(xs, ys, "o-", color=OP_COLORS["matvec_total_s"], lw=2.2, ms=8,
                label="matvec (D @ x)", markeredgecolor="white",
                markeredgewidth=1)
    ax.set_xlabel("# ranks (world size)", fontsize=12)
    ax.set_ylabel("throughput  (DOF processed / s)", fontsize=12)
    ax.set_xticks([r["world_size"] for r in strong])
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")
    if not no_title and strong:
        ax.set_title(f"Distributed throughput (CPU / gloo)\n"
                     f"fixed poisson_2d, DOF={strong[0]['dof']:,}",
                     fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = out_dir / "dist_throughput.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _print_tables(results: dict) -> None:
    strong = sorted(results["strong"], key=lambda r: r["world_size"])
    weak = sorted(results["weak"], key=lambda r: r["world_size"])

    def _speedup_eff(rows, op):
        base = next((r for r in rows if r["world_size"] == 1), None)
        t1 = base.get(op) if base else None
        out = []
        for r in rows:
            t = r.get(op)
            if t is None or t1 is None:
                out.append((r["world_size"], t, None, None))
            else:
                sp = t1 / t
                out.append((r["world_size"], t, sp, sp / r["world_size"]))
        return out

    print("\n" + "=" * 72)
    print("STRONG SCALING (fixed DOF=%s)" %
          (f"{strong[0]['dof']:,}" if strong else "?"))
    print("=" * 72)
    for op, label in OP_LABELS.items():
        print(f"\n  {label}:")
        print(f"    {'ranks':>6} {'time(s)':>12} {'speedup':>10} "
              f"{'efficiency':>12}")
        for ws, t, sp, eff in _speedup_eff(strong, op):
            ts = f"{t:.4f}" if t is not None else "FAIL"
            sps = f"{sp:.3f}" if sp is not None else "-"
            effs = f"{eff*100:.1f}%" if eff is not None else "-"
            print(f"    {ws:>6} {ts:>12} {sps:>10} {effs:>12}")

    print("\n" + "=" * 72)
    print("WEAK SCALING (per-rank DOF fixed; ideal = flat time)")
    print("=" * 72)
    print(f"\n  {'ranks':>6} {'DOF':>10} | "
          f"{'matvec/call(ms)':>16} {'solve cg(s)':>12} {'eigsh(s)':>10}")
    for r in weak:
        mv = r.get("matvec_per_call_ms")
        sc = r.get("solve_cg_s")
        eg = r.get("eigsh_s")
        print(f"  {r['world_size']:>6} {r.get('dof', 0):>10,} | "
              f"{(f'{mv:.3f}' if mv is not None else '-'):>16} "
              f"{(f'{sc:.4f}' if sc is not None else '-'):>12} "
              f"{(f'{eg:.4f}' if eg is not None else '-'):>10}")
    print()


def _markdown_summary(results: dict) -> str:
    strong = sorted(results["strong"], key=lambda r: r["world_size"])
    weak = sorted(results["weak"], key=lambda r: r["world_size"])
    meta = results["meta"]

    def _row_strong(op):
        base = next((r for r in strong if r["world_size"] == 1), None)
        t1 = base.get(op) if base else None
        cells = []
        for r in strong:
            t = r.get(op)
            if t is None or t1 is None:
                cells.append("FAIL")
            else:
                cells.append(f"{t:.3f}s ({t1/t:.2f}x)")
        return cells

    ranks = [r["world_size"] for r in strong]
    lines = []
    lines.append("### torch-sla distributed scaling (CPU / gloo)\n")
    lines.append(f"Backend `gloo`, multiprocess on CPU "
                 f"({meta['ncores']} cores, 1 BLAS thread/rank). "
                 f"Strong scaling fixes `poisson_2d` at "
                 f"DOF={strong[0]['dof']:,}; solve=`cg` (pytorch Krylov), "
                 f"eigsh=LOBPCG.\n")
    lines.append("**Strong scaling** -- speedup `T(1)/T(p)` in parens:\n")
    hdr = "| op | " + " | ".join(f"p={p}" for p in ranks) + " |"
    sep = "|----|" + "|".join(["----"] * len(ranks)) + "|"
    lines.append(hdr)
    lines.append(sep)
    for op, label in OP_LABELS.items():
        lines.append(f"| {label} | " + " | ".join(_row_strong(op)) + " |")
    lines.append("")
    lines.append("**Weak scaling** -- per-rank DOF fixed; ideal time is "
                 "flat:\n")
    lines.append("| ranks | DOF | matvec/call (ms) | solve cg (s) | "
                 "eigsh (s) |")
    lines.append("|----|----|----|----|----|")
    for r in weak:
        mv = r.get("matvec_per_call_ms")
        sc = r.get("solve_cg_s")
        eg = r.get("eigsh_s")
        lines.append(
            f"| {r['world_size']} | {r.get('dof', 0):,} | "
            f"{(f'{mv:.3f}' if mv is not None else '-')} | "
            f"{(f'{sc:.3f}' if sc is not None else '-')} | "
            f"{(f'{eg:.3f}' if eg is not None else '-')} |")
    lines.append("")
    lines.append("> On CPU/gloo with no real network or GPU, every Krylov "
                 "/ LOBPCG iteration pays an `all_reduce` + halo exchange "
                 "with no compute to hide it. Strong-scaling speedup is "
                 "therefore comm-bound and often sub-linear or negative; "
                 "the curves quantify where communication starts to "
                 "dominate. Matvec (one collective per call) degrades "
                 "least; iterative `solve`/`eigsh` (a collective every "
                 "iteration) degrade most.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ranks", type=str, default="1,2,4,8",
                    help="comma-separated world sizes, e.g. 1,2,4,8")
    ap.add_argument("--out", type=str, default="benchmarks/results")
    ap.add_argument("--side", type=int, default=256,
                    help="strong-scaling poisson_2d side (DOF=side^2)")
    ap.add_argument("--weak-base-side", type=int, default=160,
                    help="weak-scaling side at p=1 (DOF grows ~p)")
    ap.add_argument("--reps-matvec", type=int, default=200)
    ap.add_argument("--maxiter", type=int, default=500)
    ap.add_argument("--eig-k", type=int, default=4)
    ap.add_argument("--eig-maxiter", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--base-port", type=int, default=29760)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke: small DOF, few reps/iters")
    ap.add_argument("--only-plot", action="store_true",
                    help="replot from existing dist_scaling_results.json")
    ap.add_argument("--no-title", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dist_scaling_results.json"

    if args.only_plot:
        with open(json_path) as fh:
            results = json.load(fh)
        paths = _plot(results, out_dir, no_title=args.no_title)
        print("replotted:", *[str(p) for p in paths], sep="\n  ")
        return

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]

    if args.quick:
        side = 64
        weak_base = 48
        reps_mv = 50
        maxiter = 100
        eig_maxiter = 60
        warmup = 2
    else:
        side = args.side
        weak_base = args.weak_base_side
        reps_mv = args.reps_matvec
        maxiter = args.maxiter
        eig_maxiter = args.eig_maxiter
        warmup = args.warmup

    ncores = os.cpu_count() or 1
    print("=" * 72)
    print("torch-sla DISTRIBUTED scaling benchmark (CPU / gloo, multiprocess)")
    print("=" * 72)
    print(f"  ranks         : {ranks}")
    print(f"  cores         : {ncores} (1 BLAS thread/rank)")
    print(f"  strong side   : {side}  (DOF={side*side:,})")
    print(f"  weak base side: {weak_base}  (DOF grows ~p)")
    print(f"  matvec reps   : {reps_mv}   cg maxiter: {maxiter}   "
          f"eigsh maxiter: {eig_maxiter} (k={args.eig_k})")
    print("=" * 72, flush=True)

    common = dict(reps_matvec=reps_mv, maxiter=maxiter, eig_k=args.eig_k,
                  eig_maxiter=eig_maxiter, warmup=warmup, timeout=2400)

    results = {"meta": {"ncores": ncores, "backend": "gloo", "device": "cpu",
                        "ranks": ranks, "strong_side": side,
                        "weak_base_side": weak_base, "torch": torch.__version__},
               "strong": [], "weak": []}

    port = args.base_port
    # ---- strong scaling: fixed side -------------------------------------- #
    print("\n[strong scaling] fixed DOF=%d" % (side * side), flush=True)
    for ws in ranks:
        cfg = dict(common, side=side)
        t0 = time.perf_counter()
        r = _run_point(ws, cfg, port)
        port += 1
        results["strong"].append(r)
        wall = time.perf_counter() - t0
        mv = r.get("matvec_per_call_ms")
        sc = r.get("solve_cg_s")
        eg = r.get("eigsh_s")
        print(f"  p={ws:>2}  matvec={ (f'{mv:.3f}ms' if mv else 'FAIL') }  "
              f"cg={ (f'{sc:.3f}s' if sc else 'FAIL') }  "
              f"eigsh={ (f'{eg:.3f}s' if eg else 'FAIL') }  "
              f"[wall {wall:.1f}s]", flush=True)
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2)

    # ---- weak scaling: DOF ~ p ------------------------------------------- #
    # per-rank DOF fixed: total DOF(p) ~= weak_base^2 * p -> side(p) =
    # round(weak_base * sqrt(p)). poisson_2d side = that.
    print("\n[weak scaling] per-rank DOF fixed (~%d/rank)"
          % (weak_base * weak_base), flush=True)
    for ws in ranks:
        wside = int(round(weak_base * (ws ** 0.5)))
        cfg = dict(common, side=wside)
        t0 = time.perf_counter()
        r = _run_point(ws, cfg, port)
        port += 1
        results["weak"].append(r)
        wall = time.perf_counter() - t0
        mv = r.get("matvec_per_call_ms")
        sc = r.get("solve_cg_s")
        eg = r.get("eigsh_s")
        print(f"  p={ws:>2}  side={wside}  DOF={r.get('dof', 0):>8,}  "
              f"matvec={ (f'{mv:.3f}ms' if mv else 'FAIL') }  "
              f"cg={ (f'{sc:.3f}s' if sc else 'FAIL') }  "
              f"eigsh={ (f'{eg:.3f}s' if eg else 'FAIL') }  "
              f"[wall {wall:.1f}s]", flush=True)
        with open(json_path, "w") as fh:
            json.dump(results, fh, indent=2)

    # ---- plots + report -------------------------------------------------- #
    paths = _plot(results, out_dir, no_title=args.no_title)
    _print_tables(results)
    md = _markdown_summary(results)
    results["markdown_summary"] = md
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nPLOTS:")
    for p in paths:
        print("  ", p)
    print("  ", json_path)
    print("\n" + "-" * 72)
    print("MARKDOWN SUMMARY (ready to paste):")
    print("-" * 72)
    print(md)


if __name__ == "__main__":
    main()
