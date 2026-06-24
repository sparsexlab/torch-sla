#!/usr/bin/env python
"""
Benchmark for sparse matrix operations.

Tests:
1. Sparse-Vector multiplication (SpMV): A @ x
2. Sparse-Dense multiplication (SpMM): A @ B  
3. Sparse-Sparse multiplication (SpSpM): A @ A^T

Generates three separate plots for each operation type showing performance
across different matrix sizes.

Usage:
    python benchmark_mm.py              # Run full benchmark
    python benchmark_mm.py --only-plot  # Only regenerate plots from cached data
    python benchmark_mm.py --no-title   # Generate plots without titles
    python benchmark_mm.py --dtype float32  # Test with float32
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Output directories
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "benchmark_mm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    operation: str
    backend: str
    n: int
    nnz: int
    time_ms: float
    bandwidth_gbs: float
    gflops: float
    peak_memory_mb: float
    success: bool
    error_msg: Optional[str] = None


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "memory_gb": 0, "peak_bandwidth_gbs": 100}
    
    props = torch.cuda.get_device_properties(0)
    name = props.name
    memory_gb = props.total_memory / (1024**3)
    
    # Estimate peak bandwidth
    bandwidth_map = {
        'h100': 3350, 'h200': 4800, 'a100': 2039, 'v100': 900,
        '4090': 1008, '3090': 936, 'a6000': 768
    }
    peak_bw = 1000
    for key, bw in bandwidth_map.items():
        if key in name.lower():
            peak_bw = bw
            break
    
    return {
        "name": name,
        "memory_gb": memory_gb,
        "peak_bandwidth_gbs": peak_bw
    }


def reset_cuda():
    """Reset CUDA state."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def get_peak_memory_mb() -> float:
    """Get peak CUDA memory in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0


def create_poisson_2d(grid_n: int, device: str = 'cuda', dtype=torch.float64):
    """Create 2D Poisson matrix (5-point stencil)."""
    N = grid_n * grid_n
    idx = torch.arange(N, device=device)
    i, j = idx // grid_n, idx % grid_n
    
    entries = [
        (idx, idx, torch.full((N,), 4.0, dtype=dtype, device=device)),
        (idx[j > 0], idx[j > 0] - 1, torch.full(((j > 0).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[j < grid_n-1], idx[j < grid_n-1] + 1, torch.full(((j < grid_n-1).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[i > 0], idx[i > 0] - grid_n, torch.full(((i > 0).sum(),), -1.0, dtype=dtype, device=device)),
        (idx[i < grid_n-1], idx[i < grid_n-1] + grid_n, torch.full(((i < grid_n-1).sum(),), -1.0, dtype=dtype, device=device)),
    ]
    
    rows = torch.cat([e[0] for e in entries])
    cols = torch.cat([e[1] for e in entries])
    vals = torch.cat([e[2] for e in entries])
    
    # Create CSR matrix
    indices = torch.stack([rows, cols], dim=0)
    coo = torch.sparse_coo_tensor(indices, vals, (N, N), device=device, dtype=dtype)
    csr = coo.to_sparse_csr()
    
    return csr, len(vals)


def create_random_sparse(n: int, density: float = 0.01, device: str = 'cuda', 
                         dtype=torch.float64):
    """Create random sparse matrix with given density."""
    nnz = int(n * n * density)
    rows = torch.randint(0, n, (nnz,), device=device)
    cols = torch.randint(0, n, (nnz,), device=device)
    vals = torch.randn(nnz, dtype=dtype, device=device)
    
    indices = torch.stack([rows, cols], dim=0)
    coo = torch.sparse_coo_tensor(indices, vals, (n, n), device=device, dtype=dtype)
    csr = coo.coalesce().to_sparse_csr()
    
    actual_nnz = csr.values().shape[0]
    return csr, actual_nnz


# ============================================================================
# SpMV Benchmark
# ============================================================================

def bench_spmv(A_csr, x, n_iters: int = 100) -> BenchmarkResult:
    """Benchmark Sparse-Vector multiplication: y = A @ x"""
    n = A_csr.shape[0]
    nnz = A_csr.values().shape[0]
    dtype_size = 8 if A_csr.dtype == torch.float64 else 4
    
    try:
        # Warmup
        for _ in range(10):
            y = torch.mv(A_csr, x)
        torch.cuda.synchronize()
        
        reset_cuda()
        
        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            y = torch.mv(A_csr, x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        time_per_iter = elapsed / n_iters * 1000  # ms
        
        # Memory traffic: values + col_indices + row_ptr + x + y
        bytes_per_iter = (nnz * dtype_size + nnz * 4 + (n + 1) * 4 + 
                         n * dtype_size + n * dtype_size)
        bandwidth_gbs = bytes_per_iter * n_iters / elapsed / 1e9
        
        # FLOPs: 2 * nnz (multiply-add)
        flops = 2 * nnz
        gflops = flops * n_iters / elapsed / 1e9
        
        return BenchmarkResult(
            operation="spmv",
            backend="pytorch",
            n=n,
            nnz=nnz,
            time_ms=time_per_iter,
            bandwidth_gbs=bandwidth_gbs,
            gflops=gflops,
            peak_memory_mb=get_peak_memory_mb(),
            success=True
        )
    except Exception as e:
        return BenchmarkResult(
            operation="spmv", backend="pytorch", n=n, nnz=nnz,
            time_ms=-1, bandwidth_gbs=-1, gflops=-1, peak_memory_mb=-1,
            success=False, error_msg=str(e)[:80]
        )


# ============================================================================
# SpMM Benchmark (Sparse @ Dense)
# ============================================================================

def bench_spmm(A_csr, B_dense, n_iters: int = 100) -> BenchmarkResult:
    """Benchmark Sparse-Dense multiplication: C = A @ B"""
    n = A_csr.shape[0]
    k = B_dense.shape[1]
    nnz = A_csr.values().shape[0]
    dtype_size = 8 if A_csr.dtype == torch.float64 else 4
    
    try:
        # Warmup
        for _ in range(10):
            C = torch.mm(A_csr, B_dense)
        torch.cuda.synchronize()
        
        reset_cuda()
        
        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            C = torch.mm(A_csr, B_dense)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        time_per_iter = elapsed / n_iters * 1000  # ms
        
        # Memory traffic (approximate): A_values + A_indices + B + C
        bytes_per_iter = (nnz * dtype_size + nnz * 4 + (n + 1) * 4 +
                         n * k * dtype_size + n * k * dtype_size)
        bandwidth_gbs = bytes_per_iter * n_iters / elapsed / 1e9
        
        # FLOPs: 2 * nnz * k
        flops = 2 * nnz * k
        gflops = flops * n_iters / elapsed / 1e9
        
        return BenchmarkResult(
            operation=f"spmm_k{k}",
            backend="pytorch",
            n=n,
            nnz=nnz,
            time_ms=time_per_iter,
            bandwidth_gbs=bandwidth_gbs,
            gflops=gflops,
            peak_memory_mb=get_peak_memory_mb(),
            success=True
        )
    except Exception as e:
        return BenchmarkResult(
            operation=f"spmm_k{k}", backend="pytorch", n=n, nnz=nnz,
            time_ms=-1, bandwidth_gbs=-1, gflops=-1, peak_memory_mb=-1,
            success=False, error_msg=str(e)[:80]
        )


# ============================================================================
# SpSpM Benchmark (Sparse @ Sparse)
# ============================================================================

def bench_spspm(A_csr, B_csr, n_iters: int = 10) -> BenchmarkResult:
    """Benchmark Sparse-Sparse multiplication: C = A @ B"""
    n = A_csr.shape[0]
    nnz_A = A_csr.values().shape[0]
    nnz_B = B_csr.values().shape[0]
    dtype_size = 8 if A_csr.dtype == torch.float64 else 4
    
    try:
        # Convert to COO for sparse-sparse multiplication
        A_coo = A_csr.to_sparse_coo()
        B_coo = B_csr.to_sparse_coo()
        
        # Warmup
        for _ in range(3):
            C = torch.sparse.mm(A_coo, B_coo)
        torch.cuda.synchronize()
        
        reset_cuda()
        
        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            C = torch.sparse.mm(A_coo, B_coo)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        
        time_per_iter = elapsed / n_iters * 1000  # ms
        nnz_C = C._nnz()
        
        # Memory traffic (approximate)
        bytes_per_iter = ((nnz_A + nnz_B + nnz_C) * dtype_size + 
                         (nnz_A + nnz_B + nnz_C) * 2 * 4)
        bandwidth_gbs = bytes_per_iter * n_iters / elapsed / 1e9
        
        # FLOPs estimation (very approximate)
        flops = 2 * nnz_A * (nnz_B / n)  # Rough estimate
        gflops = flops * n_iters / elapsed / 1e9
        
        return BenchmarkResult(
            operation="spspm",
            backend="pytorch",
            n=n,
            nnz=nnz_A,
            time_ms=time_per_iter,
            bandwidth_gbs=bandwidth_gbs,
            gflops=gflops,
            peak_memory_mb=get_peak_memory_mb(),
            success=True
        )
    except Exception as e:
        return BenchmarkResult(
            operation="spspm", backend="pytorch", n=n, nnz=nnz_A,
            time_ms=-1, bandwidth_gbs=-1, gflops=-1, peak_memory_mb=-1,
            success=False, error_msg=str(e)[:80]
        )


# ============================================================================
# Plotting
# ============================================================================

def generate_plots(results: List[Dict], output_dir: Path, dtype_name: str,
                   no_title: bool = False):
    """Generate performance plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    # Group by operation
    ops = {}
    for r in results:
        op = r['operation']
        if op not in ops:
            ops[op] = {'n': [], 'time': [], 'bw': [], 'gflops': [], 'mem': []}
        if r['success']:
            ops[op]['n'].append(r['n'])
            ops[op]['time'].append(r['time_ms'])
            ops[op]['bw'].append(r['bandwidth_gbs'])
            ops[op]['gflops'].append(r['gflops'])
            ops[op]['mem'].append(r['peak_memory_mb'])
    
    colors = {
        'spmv': '#e74c3c',
        'spmm_k8': '#3498db',
        'spmm_k32': '#2980b9',
        'spmm_k128': '#1a5276',
        'spspm': '#27ae60',
    }
    
    markers = {'spmv': 'o', 'spmm_k8': 's', 'spmm_k32': 'D', 'spmm_k128': '^', 'spspm': 'v'}
    
    gpu_info = get_gpu_info()
    peak_bw = gpu_info['peak_bandwidth_gbs']
    
    # 1. Performance (time vs N)
    fig, ax = plt.subplots(figsize=(10, 6))
    for op, data in ops.items():
        if data['n']:
            color = colors.get(op, '#7f8c8d')
            marker = markers.get(op, 'o')
            ax.plot(data['n'], data['time'], marker=marker, label=op,
                   color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Matrix Size N', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    if not no_title:
        ax.set_title(f'Sparse Matrix Operations Performance ({dtype_name})', 
                    fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'performance_{dtype_name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'performance_{dtype_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    
    # 2. Bandwidth utilization
    fig, ax = plt.subplots(figsize=(10, 6))
    for op, data in ops.items():
        if data['n']:
            color = colors.get(op, '#7f8c8d')
            marker = markers.get(op, 'o')
            ax.plot(data['n'], data['bw'], marker=marker, label=op,
                   color=color, linewidth=2, markersize=6)
    ax.axhline(y=peak_bw, color='#2ecc71', linestyle='--', linewidth=2, 
               label=f'Peak ({peak_bw:.0f} GB/s)')
    ax.set_xlabel('Matrix Size N', fontsize=11)
    ax.set_ylabel('Memory Bandwidth (GB/s)', fontsize=11)
    if not no_title:
        ax.set_title(f'Bandwidth Utilization ({dtype_name})', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'bandwidth_{dtype_name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'bandwidth_{dtype_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    
    # 3. Memory usage
    fig, ax = plt.subplots(figsize=(10, 6))
    for op, data in ops.items():
        if data['n'] and any(m > 0 for m in data['mem']):
            valid = [(n, m) for n, m in zip(data['n'], data['mem']) if m > 0]
            if valid:
                ns, mems = zip(*valid)
                color = colors.get(op, '#7f8c8d')
                marker = markers.get(op, 'o')
                ax.plot(ns, mems, marker=marker, label=op,
                       color=color, linewidth=2, markersize=6)
    ax.set_xlabel('Matrix Size N', fontsize=11)
    ax.set_ylabel('Peak GPU Memory (MB)', fontsize=11)
    if not no_title:
        ax.set_title(f'Memory Usage ({dtype_name})', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'memory_{dtype_name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'memory_{dtype_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plots saved to: {output_dir}")


# ============================================================================
# Main
# ============================================================================

def run_benchmark(dtype, dtype_name: str) -> List[Dict]:
    """Run benchmark for a specific dtype."""
    print(f"\n{'='*80}")
    print(f"Running Sparse Matrix Operations Benchmark ({dtype_name})")
    print(f"{'='*80}")
    
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('name', 'N/A')}")
    print(f"Peak Bandwidth: {gpu_info.get('peak_bandwidth_gbs', 'N/A')} GB/s")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test sizes (grid_n for Poisson matrix -> N = grid_n^2)
    grid_sizes = [32, 50, 100, 158, 224, 316, 447, 632, 894, 1000, 1265, 1414, 2000]
    
    results = []
    
    # SpMV benchmark
    print("\n  SpMV (Sparse @ Vector):")
    for grid_n in grid_sizes:
        n = grid_n * grid_n
        A_csr, nnz = create_poisson_2d(grid_n, device, dtype)
        x = torch.randn(n, dtype=dtype, device=device)
        
        result = bench_spmv(A_csr, x)
        results.append(asdict(result))
        
        if result.success:
            print(f"    N={n:>10,}: {result.time_ms:.3f}ms, "
                  f"{result.bandwidth_gbs:.1f} GB/s, {result.gflops:.2f} GFLOPS")
        else:
            print(f"    N={n:>10,}: FAILED ({result.error_msg})")
        
        del A_csr, x
        reset_cuda()
    
    # SpMM benchmark (different k values)
    for k in [8, 32, 128]:
        print(f"\n  SpMM (Sparse @ Dense, k={k}):")
        for grid_n in grid_sizes:
            n = grid_n * grid_n
            A_csr, nnz = create_poisson_2d(grid_n, device, dtype)
            B = torch.randn(n, k, dtype=dtype, device=device)
            
            result = bench_spmm(A_csr, B)
            results.append(asdict(result))
            
            if result.success:
                print(f"    N={n:>10,}: {result.time_ms:.3f}ms, "
                      f"{result.bandwidth_gbs:.1f} GB/s, {result.gflops:.2f} GFLOPS")
            else:
                print(f"    N={n:>10,}: FAILED ({result.error_msg})")
            
            del A_csr, B
            reset_cuda()
    
    # SpSpM benchmark (smaller sizes due to memory)
    print("\n  SpSpM (Sparse @ Sparse):")
    for grid_n in grid_sizes[:8]:  # Limit size for SpSpM
        n = grid_n * grid_n
        A_csr, nnz = create_poisson_2d(grid_n, device, dtype)
        
        result = bench_spspm(A_csr, A_csr)
        results.append(asdict(result))
        
        if result.success:
            print(f"    N={n:>10,}: {result.time_ms:.3f}ms, "
                  f"{result.bandwidth_gbs:.1f} GB/s")
        else:
            print(f"    N={n:>10,}: FAILED ({result.error_msg})")
        
        del A_csr
        reset_cuda()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark sparse matrix operations')
    parser.add_argument('--only-plot', action='store_true',
                       help='Only regenerate plots from cached data')
    parser.add_argument('--no-title', action='store_true',
                       help='Generate plots without titles')
    parser.add_argument('--dtype', choices=['float64', 'float32', 'both'], default='both',
                       help='Data type to test (default: both)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("torch-sla Sparse Matrix Operations Benchmark")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    dtypes_to_test = []
    if args.dtype in ['float64', 'both']:
        dtypes_to_test.append((torch.float64, 'float64'))
    if args.dtype in ['float32', 'both']:
        dtypes_to_test.append((torch.float32, 'float32'))
    
    all_results = {}
    
    for dtype, dtype_name in dtypes_to_test:
        cache_file = OUTPUT_DIR / f'benchmark_{dtype_name}.json'
        
        if args.only_plot:
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    results = json.load(f)
                print(f"Loaded {len(results)} results from {cache_file}")
            else:
                print(f"Cache file not found: {cache_file}")
                continue
        else:
            results = run_benchmark(dtype, dtype_name)
            
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {cache_file}")
        
        all_results[dtype_name] = results
        generate_plots(results, OUTPUT_DIR, dtype_name, args.no_title)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for dtype_name, results in all_results.items():
        print(f"\n{dtype_name}:")
        
        # Group by operation
        ops = {}
        for r in results:
            if r['success']:
                op = r['operation']
                if op not in ops:
                    ops[op] = []
                ops[op].append(r)
        
        for op, op_results in ops.items():
            if op_results:
                best = max(op_results, key=lambda x: x['n'])
                print(f"  {op:15s}: max N={best['n']:>10,}, "
                      f"{best['time_ms']:.2f}ms, {best['bandwidth_gbs']:.1f} GB/s")
    
    print("\nDone!")


if __name__ == '__main__':
    main()


