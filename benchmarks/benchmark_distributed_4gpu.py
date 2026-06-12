#!/usr/bin/env python
"""
4-GPU Distributed Solver Performance Benchmark

Run with:
    torchrun --standalone --nproc_per_node=4 benchmark_distributed_4gpu.py

Compare preconditioner choices in true multi-GPU distributed setting:
1. Baseline:     plain CG (no preconditioner)
2. +Jacobi:      diagonal inverse on owned rows
3. +Block-Jacobi: dense LU on the owned-by-owned block per rank
4. +SSOR:        symmetric SOR sweep on the owned-by-owned block
"""

import argparse
import time
import os
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.distributed as dist

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BenchmarkResult:
    config_name: str
    n: int
    dof: int
    num_gpus: int
    time_ms: float
    memory_mb: float
    residual: float
    speedup: float = 1.0


def create_2d_poisson(n: int, dtype=torch.float64):
    """Create 2D Poisson matrix (5-point stencil)."""
    N = n * n
    row, col, val = [], [], []
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            val.append(4.0)
            row.append(idx)
            col.append(idx)
            if j > 0:
                val.append(-1.0)
                row.append(idx)
                col.append(idx - 1)
            if j < n - 1:
                val.append(-1.0)
                row.append(idx)
                col.append(idx + 1)
            if i > 0:
                val.append(-1.0)
                row.append(idx)
                col.append(idx - n)
            if i < n - 1:
                val.append(-1.0)
                row.append(idx)
                col.append(idx + n)
    
    return (
        torch.tensor(val, dtype=dtype),
        torch.tensor(row, dtype=torch.int64),
        torch.tensor(col, dtype=torch.int64),
        (N, N)
    )


def get_gpu_memory_mb(device) -> float:
    """Get peak GPU memory in MB."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0


def reset_memory(device):
    """Reset memory tracking."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def benchmark_solve(
    D,
    b_dt,
    preconditioner: str,
    rtol: float,
    maxiter: int,
    warmup: int,
    repeat: int,
    rank: int,
    world_size: int,
) -> tuple:
    """Benchmark a solve configuration."""
    from torch_sla import solve, SolverConfig

    device = b_dt.to_local().device

    scope = SolverConfig(method="cg",
                          preconditioner=(None if preconditioner == "none"
                                          else preconditioner),
                          rtol=rtol, atol=0.0, maxiter=maxiter)
    warmup_scope = SolverConfig(method="cg",
                                 preconditioner=(None if preconditioner == "none"
                                                 else preconditioner),
                                 rtol=rtol, atol=0.0,
                                 maxiter=min(50, maxiter))

    # Warmup
    for _ in range(warmup):
        with warmup_scope:
            _ = solve(D, b_dt)

    dist.barrier()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    reset_memory(device)

    # Benchmark
    times = []
    x_dt = None
    for _ in range(repeat):
        dist.barrier()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        with scope:
            x_dt = solve(D, b_dt)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        dist.barrier()

        times.append(time.perf_counter() - t0)

    memory_mb = get_gpu_memory_mb(device)

    # Residual via public ops -- ``D @ x_dt`` is the distributed matvec.
    r_dt = b_dt - D @ x_dt
    residual = float(
        (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item())

    return min(times) * 1000, memory_mb, residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[200, 500, 1000, 2000])
    parser.add_argument('--rtol', type=float, default=1e-6)
    parser.add_argument('--maxiter', type=int, default=2000)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--output', type=str, default='results/distributed_4gpu')
    args = parser.parse_args()
    
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set device
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError:
        from torch.distributed._tensor.device_mesh import init_device_mesh

    from torch_sla import SparseTensor, DSparseTensor

    if rank == 0:
        print("=" * 80)
        print(f"4-GPU Distributed Solver Benchmark (World Size: {world_size})")
        print("=" * 80)
        print(f"GPUs: {world_size} x {torch.cuda.get_device_name(0)}")
        print(f"Sizes: {args.sizes}")
        print(f"rtol: {args.rtol}, maxiter: {args.maxiter}")
        print()
    
    # Configurations: (name, preconditioner)
    configs = [
        ("Baseline",       "none"),
        ("+Jacobi",        "jacobi"),
        ("+Block-Jacobi",  "block_jacobi"),
        ("+SSOR",          "ssor"),
    ]
    mesh = init_device_mesh("cuda", (world_size,))
    
    all_results = []
    
    for n in args.sizes:
        N = n * n
        
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Grid {n}×{n} = {N:,} DOF, {world_size} GPUs ({N//world_size:,} DOF/GPU)")
            print(f"{'='*80}")
        
        # Build SparseTensor on every rank, then row-shard across the mesh.
        val, row, col, shape = create_2d_poisson(n)
        A_global = SparseTensor(val.to(device), row.to(device),
                                 col.to(device), shape)
        D = DSparseTensor.partition(A_global, mesh,
                                     partition_method="simple")

        b_global = torch.ones(shape[0], dtype=torch.float64, device=device)
        b_dt = D.scatter(b_global)

        if rank == 0:
            partition = D._spec.placement.partition
            owned = int(partition.owned_nodes.numel())
            halo  = int(partition.halo_nodes.numel())
            print(f"Rank 0: owned={owned}, halo={halo}, local_nnz={D.nnz}")
            print()
            print(f"{'Config':<20} {'Time (ms)':>12} {'Memory (MB)':>12} {'Residual':>12} {'Speedup':>10}")
            print("-" * 70)

        dist.barrier()

        baseline_time = None

        for name, precond in configs:
            time_ms, mem_mb, residual = benchmark_solve(
                D, b_dt, precond,
                args.rtol, args.maxiter, args.warmup, args.repeat,
                rank, world_size,
            )
            
            if baseline_time is None:
                baseline_time = time_ms
            
            speedup = baseline_time / time_ms if time_ms > 0 else 1.0
            
            if rank == 0:
                print(f"{name:<20} {time_ms:>12.2f} {mem_mb:>12.1f} {residual:>12.2e} {speedup:>10.2f}x")
            
            all_results.append(BenchmarkResult(
                config_name=name,
                n=n,
                dof=N,
                num_gpus=world_size,
                time_ms=time_ms,
                memory_mb=mem_mb,
                residual=residual,
                speedup=speedup
            ))
    
    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"Summary: Speedup vs Baseline ({world_size} GPUs)")
        print("=" * 80)
        
        by_config = defaultdict(list)
        for r in all_results:
            by_config[r.config_name].append(r)
        
        header = f"{'Config':<20}"
        for n in args.sizes:
            header += f" {n}×{n}".rjust(12)
        print(header)
        print("-" * (20 + 12 * len(args.sizes)))
        
        for name, _, _ in configs:
            line = f"{name:<20}"
            for r in by_config[name]:
                line += f" {r.speedup:>10.2f}x "
            print(line)
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        with open(f'{args.output}/results_{world_size}gpu.json', 'w') as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to {args.output}/results_{world_size}gpu.json")
        
        # Plot
        if HAS_MATPLOTLIB:
            plot_results(all_results, configs, args, world_size)
    
    dist.destroy_process_group()


def plot_results(results: List[BenchmarkResult], configs, args, world_size: int):
    """Generate plots."""
    sizes = args.sizes
    n_sizes = len(sizes)
    n_configs = len(configs)
    
    by_config = defaultdict(list)
    for r in results:
        by_config[r.config_name].append(r)
    
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#9b59b6']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Distributed Solver Performance ({world_size} GPUs)', fontsize=14, fontweight='bold')
    
    # Time
    ax = axes[0]
    x = np.arange(n_sizes)
    width = 0.18
    for i, (name, _, _) in enumerate(configs):
        times = [r.time_ms for r in by_config[name]]
        ax.bar(x + (i - n_configs/2 + 0.5) * width, times, width, 
               label=name, color=colors[i], edgecolor='white')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Solve Time')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Speedup
    ax = axes[1]
    markers = ['o', 's', '^', 'D']
    for i, (name, _, _) in enumerate(configs):
        speedups = [r.speedup for r in by_config[name]]
        ax.plot(range(n_sizes), speedups, marker=markers[i], linestyle='-',
                label=name, color=colors[i], linewidth=2, markersize=8)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Speedup vs Baseline')
    ax.set_title('Optimization Speedup')
    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Memory
    ax = axes[2]
    for i, (name, _, _) in enumerate(configs):
        mems = [r.memory_mb for r in by_config[name]]
        ax.bar(x + (i - n_configs/2 + 0.5) * width, mems, width,
               label=name, color=colors[i], edgecolor='white')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Memory/GPU (MB)')
    ax.set_title('Memory Usage')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{args.output}/perf_{world_size}gpu.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}/perf_{world_size}gpu.png")
    plt.close()


if __name__ == '__main__':
    main()

