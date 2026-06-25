#!/usr/bin/env python
"""
Benchmark for torch-sla sparse solvers.

Tests different methods (pcg+amg, lu, cholesky, etc.) and backends 
(cudss, pytorch, scipy) across various DOF sizes.

Generates three separate plots:
1. Performance (time vs DOF)
2. Memory usage (MB vs DOF)  
3. Accuracy (residual vs DOF)

Usage:
    python benchmark_solvers.py              # Run full benchmark
    python benchmark_solvers.py --only-plot  # Only regenerate plots from cached data
    python benchmark_solvers.py --no-title   # Generate plots without titles
    python benchmark_solvers.py --dtype float32  # Test with float32
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch_sla as sla

# Output directories
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "benchmark_solvers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timeout for each solve
TIMEOUT_SEC = 120


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Solver timed out")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    backend: str
    method: str
    preconditioner: str
    dof: int
    time_ms: float
    residual: float
    peak_memory_mb: float
    success: bool
    error_msg: Optional[str] = None


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"name": "CPU", "memory_gb": 0}
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "memory_gb": props.total_memory / (1024**3),
        "cuda_version": torch.version.cuda
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
    
    return vals, rows, cols, (N, N)


def sparse_matvec(val, row, col, shape, x):
    """Sparse matrix-vector product."""
    indices = torch.stack([row, col], dim=0)
    A = torch.sparse_coo_tensor(indices, val, shape, device=x.device, dtype=x.dtype)
    return torch.mv(A.to_sparse_csr(), x)


def compute_residual(val, row, col, shape, x, b) -> float:
    """Compute relative residual ||Ax - b|| / ||b||."""
    Ax = sparse_matvec(val, row, col, shape, x)
    return (torch.norm(Ax - b) / torch.norm(b)).item()


# ============================================================================
# Solver Configurations
# ============================================================================

def get_solver_configs(dtype) -> List[Dict[str, Any]]:
    """Get list of solver configurations to benchmark."""
    configs = []
    is_float64 = dtype == torch.float64
    tol = 1e-10 if is_float64 else 1e-6
    
    # SciPy backends (CPU)
    if sla.is_scipy_available():
        configs.extend([
            {
                "name": "scipy+superlu",
                "backend": "scipy",
                "method": "superlu",
                "device": "cpu",
                "kwargs": {"atol": tol},
                "max_dof": 4_000_000,
            },
        ])
    
    # CUDA backends
    if torch.cuda.is_available():
        # cuDSS (direct solver)
        if sla.is_cudss_available():
            configs.extend([
                {
                    "name": "cudss+cholesky",
                    "backend": "cudss",
                    "method": "cholesky",
                    "device": "cuda",
                    "kwargs": {},
                    "max_dof": 2_000_000,
                },
                {
                    "name": "cudss+lu",
                    "backend": "cudss",
                    "method": "lu",
                    "device": "cuda",
                    "kwargs": {},
                    "max_dof": 2_000_000,
                },
            ])
        
        # PyTorch iterative solvers
        configs.extend([
            {
                "name": "pytorch+cg+jacobi",
                "backend": "pytorch",
                "method": "cg",
                "device": "cuda",
                "kwargs": {"preconditioner": "jacobi", "atol": tol, "maxiter": 30000},
                "max_dof": 10_000_000,
            },
            {
                "name": "pytorch+cg+amg",
                "backend": "pytorch",
                "method": "cg",
                "device": "cuda",
                "kwargs": {"preconditioner": "amg", "atol": tol, "maxiter": 30000},
                "max_dof": 10_000_000,
            },
            {
                "name": "pytorch+cg+ic0",
                "backend": "pytorch",
                "method": "cg",
                "device": "cuda",
                "kwargs": {"preconditioner": "ic0", "atol": tol, "maxiter": 30000},
                "max_dof": 10_000_000,
            },
            {
                "name": "pytorch+bicgstab+jacobi",
                "backend": "pytorch",
                "method": "bicgstab",
                "device": "cuda",
                "kwargs": {"preconditioner": "jacobi", "atol": tol, "maxiter": 30000},
                    "max_dof": 10_000_000,
                },
            ])
    
    return configs


def bench_solver(config: Dict, val, row, col, shape, b) -> BenchmarkResult:
    """Run a single solver benchmark."""
    name = config["name"]
    backend = config["backend"]
    method = config["method"]
    device = config["device"]
    kwargs = config["kwargs"]
    dof = shape[0]
    
    try:
        reset_cuda()
        
        # Move data to correct device
        if device == "cpu":
            val_dev = val.cpu()
            row_dev = row.cpu()
            col_dev = col.cpu()
            b_dev = b.cpu()
        else:
            val_dev = val.cuda() if not val.is_cuda else val
            row_dev = row.cuda() if not row.is_cuda else row
            col_dev = col.cuda() if not col.is_cuda else col
            b_dev = b.cuda() if not b.is_cuda else b
        
        # Set timeout
        old_handler = None
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(TIMEOUT_SEC)
        except (AttributeError, ValueError):
            pass
        
        try:
            if device == "cuda":
                torch.cuda.synchronize()
            
            t0 = time.perf_counter()
            x = sla.spsolve(val_dev, row_dev, col_dev, shape, b_dev,
                           backend=backend, method=method, **kwargs)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - t0) * 1000
            peak_mem = get_peak_memory_mb() if device == "cuda" else 0
            
            # Cancel timeout
            try:
                signal.alarm(0)
            except (AttributeError, ValueError):
                pass
            
            # Compute residual
            residual = compute_residual(val_dev, row_dev, col_dev, shape, x, b_dev)
            
            return BenchmarkResult(
                backend=backend,
                method=method,
                preconditioner=kwargs.get("preconditioner", "none"),
                dof=dof,
                time_ms=elapsed,
                residual=residual,
                peak_memory_mb=peak_mem,
                success=True
            )
            
        except TimeoutError:
            return BenchmarkResult(
                backend=backend, method=method,
                preconditioner=kwargs.get("preconditioner", "none"),
                dof=dof, time_ms=-1, residual=-1, peak_memory_mb=-1,
                success=False, error_msg="timeout"
            )
        finally:
            try:
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)
            except (AttributeError, ValueError):
                pass
                
    except Exception as e:
        return BenchmarkResult(
            backend=backend, method=method,
            preconditioner=kwargs.get("preconditioner", "none"),
            dof=dof, time_ms=-1, residual=-1, peak_memory_mb=-1,
            success=False, error_msg=str(e)[:80]
        )


# ============================================================================
# Plotting
# ============================================================================

def generate_plots(results: List[Dict], output_dir: Path, dtype_name: str, 
                   no_title: bool = False):
    """Generate performance, memory, and accuracy plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("matplotlib not available, skipping plots")
        return
    
    # Group results by solver name
    solvers = {}
    for r in results:
        name = f"{r['backend']}+{r['method']}"
        if r.get('preconditioner') and r['preconditioner'] != 'none':
            name += f"+{r['preconditioner']}"
        if name not in solvers:
            solvers[name] = {'dofs': [], 'times': [], 'residuals': [], 'memories': []}
        if r['success']:
            solvers[name]['dofs'].append(r['dof'])
            solvers[name]['times'].append(r['time_ms'])
            solvers[name]['residuals'].append(r['residual'])
            solvers[name]['memories'].append(r['peak_memory_mb'])
    
    # Color scheme
    colors = {
        'scipy+superlu': '#2ecc71',
        'cudss+cholesky': '#e74c3c',
        'cudss+lu': '#c0392b',
        'pytorch+cg+jacobi': '#3498db',
        'pytorch+cg+amg': '#2980b9',
        'pytorch+cg+ic0': '#1abc9c',
        'pytorch+bicgstab+jacobi': '#9b59b6',
    }
    
    markers = {
        'scipy+superlu': 'o',
        'cudss+cholesky': 's',
        'cudss+lu': 'D',
        'pytorch+cg+jacobi': '^',
        'pytorch+cg+amg': 'v',
        'pytorch+cg+ic0': '<',
        'pytorch+bicgstab+jacobi': '>',
    }
    
    def plot_single(ax, data_key, ylabel, yscale='log', title=None):
        for name, data in solvers.items():
            if data['dofs'] and data[data_key]:
                valid = [(d, v) for d, v in zip(data['dofs'], data[data_key]) 
                        if v is not None and v > 0]
                if valid:
                    dofs, vals = zip(*valid)
                    color = colors.get(name, '#7f8c8d')
                    marker = markers.get(name, 'o')
                    linestyle = '--' if 'pytorch' in name else '-'
                    ax.plot(dofs, vals, marker=marker, label=name,
                           color=color, linestyle=linestyle, linewidth=2, markersize=6)
        
        ax.set_xlabel('Degrees of Freedom (DOF)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        if title and not no_title:
            ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
    
    # 1. Performance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_single(ax, 'times', 'Time (ms)', 'log', f'Solver Performance ({dtype_name})')
    plt.tight_layout()
    fig.savefig(output_dir / f'performance_{dtype_name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'performance_{dtype_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    
    # 2. Memory plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_single(ax, 'memories', 'Peak GPU Memory (MB)', 'log', f'Memory Usage ({dtype_name})')
    plt.tight_layout()
    fig.savefig(output_dir / f'memory_{dtype_name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'memory_{dtype_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    
    # 3. Accuracy plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_single(ax, 'residuals', 'Relative Residual ||Ax-b||/||b||', 'log', 
                f'Solver Accuracy ({dtype_name})')
    # Add machine epsilon line
    eps = 1e-14 if dtype_name == 'float64' else 1e-6
    ax.axhline(y=eps, color='gray', linestyle=':', alpha=0.7, label=f'~machine eps')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(output_dir / f'accuracy_{dtype_name}.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'accuracy_{dtype_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plots saved to: {output_dir}")


# ============================================================================
# Main
# ============================================================================

def run_benchmark(dtype, dtype_name: str) -> List[Dict]:
    """Run benchmark for a specific dtype."""
    print(f"\n{'='*80}")
    print(f"Running Solver Benchmark ({dtype_name})")
    print(f"{'='*80}")
    
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info.get('name', 'N/A')}")
    if 'memory_gb' in gpu_info:
        print(f"GPU Memory: {gpu_info['memory_gb']:.1f} GB")
    
    # Test sizes (grid_n x grid_n = DOF)
    grid_sizes = [32, 50, 100, 158, 224, 316, 447, 632, 894, 1000, 1265, 1414]
    
    configs = get_solver_configs(dtype)
    print(f"Testing {len(configs)} solver configurations")
    print(f"Grid sizes: {grid_sizes} (DOF: {[n*n for n in grid_sizes]})")
    
    results = []
    
    for config in configs:
        name = config["name"]
        max_dof = config.get("max_dof", 10_000_000)
        device = config["device"]
        
        print(f"\n  {name}:")
        skip_rest = False
        
        for grid_n in grid_sizes:
            dof = grid_n * grid_n
            
            if dof > max_dof or skip_rest:
                print(f"    DOF {dof:>10,}: skipped")
                continue
            
            # Create test problem
            val, row, col, shape = create_poisson_2d(grid_n, device, dtype)
            b = torch.randn(shape[0], dtype=dtype, device=device)
            
            result = bench_solver(config, val, row, col, shape, b)
            results.append(asdict(result))
            
            if result.success:
                time_str = f"{result.time_ms/1000:.2f}s" if result.time_ms >= 1000 else f"{result.time_ms:.0f}ms"
                mem_str = f"{result.peak_memory_mb:.0f}MB" if result.peak_memory_mb > 0 else "N/A"
                print(f"    DOF {dof:>10,}: {time_str:>8}  mem={mem_str:>8}  res={result.residual:.1e}")
                
                # Skip if too slow
                if result.time_ms > 60000:
                    skip_rest = True
            else:
                print(f"    DOF {dof:>10,}: FAILED ({result.error_msg})")
                if result.error_msg == "timeout":
                    skip_rest = True
            
            # Cleanup
            del val, row, col, b
            reset_cuda()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark sparse solvers')
    parser.add_argument('--only-plot', action='store_true', 
                       help='Only regenerate plots from cached data')
    parser.add_argument('--no-title', action='store_true',
                       help='Generate plots without titles')
    parser.add_argument('--dtype', choices=['float64', 'float32', 'both'], default='both',
                       help='Data type to test (default: both)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("torch-sla Solver Benchmark")
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
            # Load from cache
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    results = json.load(f)
                print(f"Loaded {len(results)} results from {cache_file}")
            else:
                print(f"Cache file not found: {cache_file}")
                continue
        else:
            # Run benchmark
            results = run_benchmark(dtype, dtype_name)
            
            # Save cache
            with open(cache_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {cache_file}")
        
        all_results[dtype_name] = results
        
        # Generate plots
        generate_plots(results, OUTPUT_DIR, dtype_name, args.no_title)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    for dtype_name, results in all_results.items():
        print(f"\n{dtype_name}:")
        
        # Find best time for largest DOF
        if results:
            max_dof = max(r['dof'] for r in results if r['success'])
            best = [r for r in results if r['dof'] == max_dof and r['success']]
            best.sort(key=lambda x: x['time_ms'])
            
            print(f"  Best results at DOF = {max_dof:,}:")
            for r in best[:5]:
                name = f"{r['backend']}+{r['method']}"
                if r.get('preconditioner') and r['preconditioner'] != 'none':
                    name += f"+{r['preconditioner']}"
                print(f"    {name:30s}: {r['time_ms']:8.1f}ms, res={r['residual']:.1e}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
