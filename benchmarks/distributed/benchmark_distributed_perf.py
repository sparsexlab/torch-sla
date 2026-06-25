#!/usr/bin/env python
"""
Distributed Solver Performance Benchmark

Compare different optimization strategies with controlled variables:
- Fixed partition count (P=4 by default)
- Same problem size
- Same tolerance

Optimization strategies compared:
1. Baseline: No caching, no preconditioner
2. +CSR Cache: Cached CSR format for matvec
3. +Jacobi Precond: Diagonal preconditioner
4. +Comm Overlap: Communication-computation overlap

Measures:
- Solve time (ms)
- Peak memory (MB)
- Iterations to convergence
- Speedup vs baseline

Usage:
    python benchmark_distributed_perf.py
    python benchmark_distributed_perf.py --sizes 100 200 500 1000 --partitions 4
"""

import argparse
import time
import gc
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from collections import defaultdict

import torch

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
    """Result of a single benchmark run."""
    config_name: str
    n: int
    dof: int
    num_partitions: int
    time_ms: float
    memory_mb: float
    iterations: int
    residual: float
    speedup: float = 1.0


def create_2d_poisson(n: int, dtype=torch.float64):
    """Create 2D Poisson matrix with 5-point stencil."""
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
    """Get current GPU memory usage in MB."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return 0


def reset_memory(device):
    """Reset memory tracking."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


class SolveConfig:
    """Configuration for a solve run."""
    def __init__(self, name: str, use_csr_cache: bool = True, 
                 preconditioner: str = 'none', overlap: bool = False):
        self.name = name
        self.use_csr_cache = use_csr_cache
        self.preconditioner = preconditioner
        self.overlap = overlap


def benchmark_config(
    dsparse,
    b: torch.Tensor,
    config: SolveConfig,
    rtol: float = 1e-6,
    maxiter: int = 2000,
    warmup: int = 2,
    repeat: int = 3,
) -> Tuple[float, float, int, float]:
    """
    Benchmark a single configuration.
    
    Returns: (time_ms, memory_mb, iterations, residual)
    """
    device = dsparse.device
    
    # Clear caches if testing without CSR cache
    if not config.use_csr_cache:
        dsparse._invalidate_cache()
    
    # Warmup
    for _ in range(warmup):
        x = dsparse.solve(
            b,
            preconditioner=config.preconditioner,
            overlap=config.overlap,
            rtol=rtol,
            maxiter=min(50, maxiter),
            verbose=False
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    
    # Reset memory tracking
    reset_memory(device)
    
    # Benchmark runs
    times = []
    for _ in range(repeat):
        # Invalidate cache each run if testing without cache
        if not config.use_csr_cache:
            dsparse._csr_cache = None
        
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        
        t0 = time.perf_counter()
        x = dsparse.solve(
            b,
            preconditioner=config.preconditioner,
            overlap=config.overlap,
            rtol=rtol,
            maxiter=maxiter,
            verbose=False
        )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    
    # Get peak memory
    memory_mb = get_gpu_memory_mb(device)
    
    # Compute residual
    x_full = torch.zeros(dsparse.num_local, dtype=b.dtype, device=device)
    x_full[:dsparse.num_owned] = x
    dsparse.halo_exchange(x_full)
    Ax = dsparse.matvec(x_full, exchange_halo=False)
    
    b_full = torch.zeros(dsparse.num_local, dtype=b.dtype, device=device)
    b_full[:dsparse.num_owned] = b
    residual = (Ax[:dsparse.num_owned] - b[:dsparse.num_owned]).norm().item()
    
    # Estimate iterations (based on time ratio to baseline)
    iterations = maxiter  # Placeholder
    
    return min(times) * 1000, memory_mb, iterations, residual


def run_benchmark(args):
    """Run the benchmark."""
    from torch_sla import SparseTensor
    
    device = torch.device(args.device)
    dtype = torch.float64
    P = args.partitions  # Fixed partition count
    
    print("=" * 75)
    print(f"DSparseTensor Performance Benchmark (P={P} partitions)")
    print("=" * 75)
    print(f"Device: {device}")
    print(f"Problem sizes: {args.sizes}")
    print(f"Tolerance: {args.rtol}")
    print(f"Max iterations: {args.maxiter}")
    print()
    
    # Define configurations to compare
    configs = [
        SolveConfig("Baseline", use_csr_cache=False, preconditioner='none', overlap=False),
        SolveConfig("+CSR Cache", use_csr_cache=True, preconditioner='none', overlap=False),
        SolveConfig("+Jacobi", use_csr_cache=True, preconditioner='jacobi', overlap=False),
        SolveConfig("+Overlap", use_csr_cache=True, preconditioner='jacobi', overlap=True),
    ]
    
    all_results = []
    
    for n in args.sizes:
        N = n * n
        print(f"\n{'='*75}")
        print(f"Grid {n}×{n} = {N:,} DOF, P={P} partitions")
        print(f"{'='*75}")
        
        # Create matrix and partition
        val, row, col, shape = create_2d_poisson(n, dtype=dtype)
        A = SparseTensor(val, row, col, shape).to(device)
        D = A.partition(num_partitions=P)
        
        # Use partition 0 for testing
        dsparse = D._partitions[0]
        
        # RHS vector
        b = torch.ones(dsparse.num_owned, dtype=dtype, device=device)
        
        # Info
        print(f"Partition 0: owned={dsparse.num_owned}, halo={dsparse.num_halo}, "
              f"nnz={dsparse.nnz}, nnz/node={dsparse.nnz/dsparse.num_owned:.1f}")
        
        # Build decomposition for interior ratio
        dsparse._build_interior_boundary_decomposition()
        stats = dsparse._overlap_stats
        print(f"Interior ratio: {stats.get('interior_ratio', 0):.1%}")
        print()
        
        # Header
        print(f"{'Configuration':<20} {'Time (ms)':>12} {'Memory (MB)':>12} "
              f"{'Residual':>12} {'Speedup':>10}")
        print("-" * 70)
        
        baseline_time = None
        
        for config in configs:
            time_ms, mem_mb, iters, residual = benchmark_config(
                dsparse, b, config,
                rtol=args.rtol,
                maxiter=args.maxiter,
                warmup=args.warmup,
                repeat=args.repeat
            )
            
            if baseline_time is None:
                baseline_time = time_ms
            
            speedup = baseline_time / time_ms if time_ms > 0 else 1.0
            
            print(f"{config.name:<20} {time_ms:>12.2f} {mem_mb:>12.1f} "
                  f"{residual:>12.2e} {speedup:>10.2f}x")
            
            all_results.append(BenchmarkResult(
                config_name=config.name,
                n=n,
                dof=N,
                num_partitions=P,
                time_ms=time_ms,
                memory_mb=mem_mb,
                iterations=iters,
                residual=residual,
                speedup=speedup,
            ))
    
    # Summary table
    print("\n" + "=" * 75)
    print(f"Summary: Speedup vs Baseline (P={P})")
    print("=" * 75)
    
    by_config = defaultdict(list)
    for r in all_results:
        by_config[r.config_name].append(r)
    
    # Header
    header = f"{'Configuration':<20}"
    for n in args.sizes:
        header += f" {n}×{n}".rjust(12)
    print(header)
    print("-" * (20 + 12 * len(args.sizes)))
    
    for config in configs:
        line = f"{config.name:<20}"
        for r in by_config[config.name]:
            line += f" {r.speedup:>10.2f}x "
        print(line)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    results_file = f'{args.output}/perf_P{P}.json'
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Generate plots
    if HAS_MATPLOTLIB:
        plot_results(all_results, configs, args, P)
    
    return all_results


def plot_results(results: List[BenchmarkResult], configs: List[SolveConfig], args, P: int):
    """Generate performance comparison plots."""
    sizes = args.sizes
    n_sizes = len(sizes)
    n_configs = len(configs)
    
    # Group by config
    by_config = defaultdict(list)
    for r in results:
        by_config[r.config_name].append(r)
    
    # Colors
    colors = ['#95a5a6', '#3498db', '#2ecc71', '#9b59b6']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'DSparseTensor Performance (P={P} partitions)', fontsize=14, fontweight='bold')
    
    # Plot 1: Solve Time (bar chart)
    ax = axes[0]
    x = np.arange(n_sizes)
    width = 0.18
    
    for i, config in enumerate(configs):
        times = [r.time_ms for r in by_config[config.name]]
        offset = (i - n_configs/2 + 0.5) * width
        ax.bar(x + offset, times, width, label=config.name, color=colors[i], edgecolor='white')
    
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Solve Time')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8, loc='upper left')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Speedup
    ax = axes[1]
    markers = ['o', 's', '^', 'D']
    
    for i, config in enumerate(configs):
        speedups = [r.speedup for r in by_config[config.name]]
        ax.plot(range(n_sizes), speedups, marker=markers[i], linestyle='-',
                label=config.name, color=colors[i], linewidth=2, markersize=8)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Speedup vs Baseline')
    ax.set_title('Optimization Speedup')
    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    # Plot 3: Memory Usage
    ax = axes[2]
    for i, config in enumerate(configs):
        mems = [r.memory_mb for r in by_config[config.name]]
        offset = (i - n_configs/2 + 0.5) * width
        ax.bar(x + offset, mems, width, label=config.name, color=colors[i], edgecolor='white')
    
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('Peak Memory (MB)')
    ax.set_title('Memory Usage')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in sizes])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{args.output}/perf_P{P}.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}/perf_P{P}.png")
    plt.close()
    
    # Scaling plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    dofs = [n*n for n in sizes]
    
    for i, config in enumerate(configs):
        times = [r.time_ms for r in by_config[config.name]]
        ax.loglog(dofs, times, marker=markers[i], linestyle='-',
                  label=config.name, color=colors[i], linewidth=2, markersize=8)
    
    # O(n) reference
    if len(dofs) > 1:
        baseline_times = [r.time_ms for r in by_config['Baseline']]
        ref = [baseline_times[0] * (d / dofs[0]) for d in dofs]
        ax.loglog(dofs, ref, '--', color='gray', alpha=0.5, label='O(n)')
    
    ax.set_xlabel('DOF')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Scaling: Time vs Problem Size (P={P})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{args.output}/scaling_P{P}.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to {args.output}/scaling_P{P}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='DSparseTensor Performance Benchmark')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 200, 500, 1000],
                        help='Grid sizes to test (n for n×n grid)')
    parser.add_argument('--partitions', '-P', type=int, default=4,
                        help='Number of partitions (fixed for controlled comparison)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--rtol', type=float, default=1e-6,
                        help='Relative tolerance for convergence')
    parser.add_argument('--maxiter', type=int, default=2000,
                        help='Maximum CG iterations')
    parser.add_argument('--warmup', type=int, default=2,
                        help='Number of warmup runs')
    parser.add_argument('--repeat', type=int, default=3,
                        help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default='results/distributed_perf',
                        help='Output directory')
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()
