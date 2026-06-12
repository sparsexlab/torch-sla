#!/usr/bin/env python3
"""Multi-GPU distributed-CG scaling benchmark

Launch with torchrun, e.g.:
    torchrun --standalone --nproc_per_node=4 \
        benchmarks/benchmark_multi_gpu_scaling.py --sizes 10000 14000 17000

For each grid size n, builds a 2D Poisson 5-point stencil with
N = n^2 unknowns, partitions it across world_size processes via
``DSparseTensor.partition``, and runs distributed CG with NCCL.
Records wall time, peak memory per GPU, and final residual.

All non-rank-0 processes are silent. Rank 0 writes a JSON record per
(world_size, DOF) into the output directory.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from torch.distributed.device_mesh import init_device_mesh
except ImportError:
    from torch.distributed._tensor.device_mesh import init_device_mesh

from torch_sla import SparseTensor, DSparseTensor, solve, SolverConfig


def make_poisson_2d_global(n_grid, dtype):
    """Generate global 2D Poisson COO arrays on CPU."""
    N = n_grid * n_grid
    diag = 4.0 * np.ones(N)
    off = -np.ones(N - 1)
    off[np.arange(1, N) % n_grid == 0] = 0.0
    A = sp.diags(
        [diag, off, off, -np.ones(N - n_grid), -np.ones(N - n_grid)],
        [0, -1, 1, -n_grid, n_grid],
        format="coo",
    )
    val = torch.tensor(A.data, dtype=dtype)
    row = torch.tensor(A.row, dtype=torch.long)
    col = torch.tensor(A.col, dtype=torch.long)
    return val, row, col, (N, N)


def reset_mem(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", required=True,
                        help="Grid sides n (DOF = n^2)")
    parser.add_argument("--maxiter", type=int, default=1000,
                        help="CG iterations to run (paper uses 1000 with "
                             "Jacobi)")
    parser.add_argument("--rtol", type=float, default=1e-12,
                        help="Set very low and rely on maxiter to dominate")
    parser.add_argument("--preconditioner", type=str, default="polynomial",
                        help="distributed PCG preconditioner. "
                             "'polynomial' (Chebyshev deg 5) gives good "
                             "convergence in 1000 iters at 1M+ DOF; "
                             "'jacobi' is much weaker.")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--out", type=str,
                        default="results/benchmark_multi_gpu_scaling")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]

    if rank == 0:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        results_path = out_dir / "results.json"
        if results_path.exists():
            with open(results_path) as fh:
                rows = json.load(fh).get("rows", [])
        else:
            rows = []
        completed = {(r["world_size"], r["dof"]) for r in rows
                     if "error" not in r}
        gpu_name = torch.cuda.get_device_name(device)
        print(f"[setup] world_size={world_size}  GPU={gpu_name}  "
              f"sizes={args.sizes}  maxiter={args.maxiter}",
              flush=True)
    else:
        rows = None
        completed = None
        results_path = None
        out_dir = None
        gpu_name = ""

    for n_grid in args.sizes:
        N = n_grid * n_grid

        # Skip already-completed (rank-0-driven decision)
        if rank == 0:
            skip = (world_size, N) in completed
        else:
            skip = False
        skip_t = torch.tensor([1 if skip else 0], device=device)
        dist.broadcast(skip_t, src=0)
        if skip_t.item() == 1:
            if rank == 0:
                print(f"\n=== n={n_grid}  DOF={N:,}  ws={world_size}: "
                      f"already completed", flush=True)
            continue

        if rank == 0:
            print(f"\n=== n={n_grid}  DOF={N:,}  ws={world_size} ===",
                  flush=True)

        # Build global arrays on every rank (cheap; needed for partitioning)
        try:
            val_g, row_g, col_g, shape = make_poisson_2d_global(n_grid, dtype)
        except MemoryError:
            if rank == 0:
                rows.append({"world_size": world_size, "dof": N,
                             "n_grid": n_grid, "error": "host MemoryError"})
                with open(results_path, "w") as fh:
                    json.dump({"rows": rows}, fh, indent=2)
                print(f"  host MemoryError building COO arrays", flush=True)
            continue

        try:
            reset_mem(device)
            mesh = init_device_mesh("cuda", (world_size,))
            A_global = SparseTensor(val_g.to(device), row_g.to(device),
                                     col_g.to(device), shape)
            D = DSparseTensor.partition(A_global, mesh,
                                         partition_method="simple")
            b_global = torch.ones(shape[0], dtype=dtype, device=device)
            b_dt = D.scatter(b_global)

            scope = SolverConfig(method="cg", preconditioner=args.preconditioner,
                                  atol=0.0, rtol=args.rtol,
                                  maxiter=args.maxiter)

            # Warmup
            for _ in range(args.warmup):
                with scope:
                    _ = solve(D, b_dt)
            torch.cuda.synchronize(device)
            dist.barrier()
            torch.cuda.reset_peak_memory_stats(device)

            times = []
            x_dt = None
            for _ in range(args.num_runs):
                torch.cuda.synchronize(device)
                dist.barrier()
                t0 = time.perf_counter()
                with scope:
                    x_dt = solve(D, b_dt)
                torch.cuda.synchronize(device)
                dist.barrier()
                times.append((time.perf_counter() - t0) * 1000)

            # Per-GPU peak memory and aggregate
            peak_local = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            peak_t = torch.tensor([peak_local], device=device)
            peak_max_t = peak_t.clone()
            dist.all_reduce(peak_max_t, op=dist.ReduceOp.MAX)

            # Distributed residual ||b - A x|| / ||b||, via public ops.
            r_dt = b_dt - D @ x_dt
            r_full = r_dt.full_tensor()
            b_full = b_dt.full_tensor()
            residual = float(
                (r_full.norm() / (b_full.norm() + 1e-30)).item())

            if rank == 0:
                row = {
                    "world_size": world_size,
                    "n_grid": n_grid,
                    "dof": N,
                    "time_ms_mean": float(np.mean(times)),
                    "time_ms_min": float(np.min(times)),
                    "time_ms_max": float(np.max(times)),
                    "memory_max_mb_per_gpu": float(peak_max_t.item()),
                    "memory_local_mb": float(peak_local),
                    "residual": residual,
                    "maxiter": args.maxiter,
                    "preconditioner": args.preconditioner,
                    "gpu_name": gpu_name,
                }
                rows.append(row)
                with open(results_path, "w") as fh:
                    json.dump({"rows": rows}, fh, indent=2)
                print(f"  ws={world_size} DOF={N:,}  "
                      f"time={row['time_ms_mean']:.1f} ms  "
                      f"mem_max/GPU={row['memory_max_mb_per_gpu']:.0f} MB  "
                      f"res={residual:.2e}", flush=True)

            del D, A_global, b_global, b_dt, x_dt, r_dt, r_full, b_full
            del val_g, row_g, col_g
            reset_mem(device)
            dist.barrier()

        except torch.cuda.OutOfMemoryError as e:
            if rank == 0:
                rows.append({"world_size": world_size, "dof": N,
                             "n_grid": n_grid, "error": "CUDA OOM"})
                with open(results_path, "w") as fh:
                    json.dump({"rows": rows}, fh, indent=2)
                print(f"  CUDA OOM at DOF={N:,}", flush=True)
            reset_mem(device)
            dist.barrier()
            continue
        except RuntimeError as e:
            if rank == 0:
                rows.append({"world_size": world_size, "dof": N,
                             "n_grid": n_grid, "error": str(e)[:200]})
                with open(results_path, "w") as fh:
                    json.dump({"rows": rows}, fh, indent=2)
                print(f"  RuntimeError: {str(e)[:120]}", flush=True)
            reset_mem(device)
            dist.barrier()
            continue

    if rank == 0:
        print(f"\n[done] -> {results_path}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
