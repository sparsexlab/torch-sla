#!/usr/bin/env python3
"""Multi-GPU distributed-CG scaling benchmark

Launch with torchrun, e.g.:
    torchrun --standalone --nproc_per_node=4 \
        benchmarks/benchmark_multi_gpu_scaling.py --sizes 10000 14000 17000

For each grid size n, builds a 2D Poisson 5-point stencil with
N = n^2 unknowns, partitions it across world_size processes via
``DSparseMatrix.from_global``, and runs distributed CG with NCCL.
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
from torch_sla.distributed import DSparseMatrix


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
            dsparse = DSparseMatrix.from_global(
                val_g, row_g, col_g, shape,
                num_partitions=world_size, my_partition=rank,
                device=device,
            )
            b = torch.ones(dsparse.num_owned, dtype=dtype, device=device)

            # Warmup
            for _ in range(args.warmup):
                _ = dsparse.solve(b, atol=0.0, rtol=args.rtol,
                                  maxiter=args.maxiter,
                                  preconditioner=args.preconditioner)
            torch.cuda.synchronize(device)
            dist.barrier()
            torch.cuda.reset_peak_memory_stats(device)

            times = []
            x = None
            for _ in range(args.num_runs):
                torch.cuda.synchronize(device)
                dist.barrier()
                t0 = time.perf_counter()
                x = dsparse.solve(b, atol=0.0, rtol=args.rtol,
                                  maxiter=args.maxiter,
                                  preconditioner=args.preconditioner)
                torch.cuda.synchronize(device)
                dist.barrier()
                times.append((time.perf_counter() - t0) * 1000)

            # Per-GPU peak memory and aggregate
            peak_local = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            peak_t = torch.tensor([peak_local], device=device)
            peak_max_t = peak_t.clone()
            dist.all_reduce(peak_max_t, op=dist.ReduceOp.MAX)

            # Distributed residual: r_owned = b - (A x)_owned ; ||r||^2 reduced
            # matvec needs num_local input (owned + halo); pad and let
            # the halo exchange fill in neighbor values.
            x_local = torch.zeros(dsparse.num_local, dtype=dtype,
                                  device=device)
            x_local[:dsparse.num_owned] = x
            Ax = dsparse.matvec(x_local, exchange_halo=True)[:dsparse.num_owned]
            r = b - Ax
            local_rr = torch.dot(r, r).detach()
            local_bb = torch.dot(b, b).detach()
            global_rr = local_rr.clone()
            global_bb = local_bb.clone()
            dist.all_reduce(global_rr, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_bb, op=dist.ReduceOp.SUM)
            residual = float((global_rr / global_bb).sqrt().item())

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

            del dsparse, b, x, Ax, r, val_g, row_g, col_g
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
