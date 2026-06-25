#!/usr/bin/env python
"""
SuiteSparse Matrix Collection Benchmark with torch-sla

This script demonstrates how to use torch-sla to solve linear systems
from the SuiteSparse Matrix Collection (formerly University of Florida
Sparse Matrix Collection).

Key features:
- Downloads matrices from SuiteSparse via ssgetpy or direct URL
- Solves Ax = b using torch-sla's SparseTensor
- Benchmarks different backends (scipy, pytorch, cudss)
- Compares iterative vs direct solvers

Usage:
    python suitesparse_benchmark.py [--matrix NAME] [--device cpu|cuda] [--backend auto|scipy|pytorch|cudss]
"""

import argparse
import time
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import urllib.request
import tarfile
import gzip
import shutil

import torch
import numpy as np

# Add parent directory to path for local torch_sla import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch_sla import SparseTensor
from torch_sla.io import load_mtx, load_mtx_info


# =============================================================================
# SuiteSparse Matrix Download Utilities
# =============================================================================

SUITESPARSE_URL = "https://suitesparse-collection-website.herokuapp.com"
MATRIX_MARKET_BASE = "https://sparse.tamu.edu/MM"

# Popular matrices for testing (various sizes and properties)
POPULAR_MATRICES = {
    # Small matrices for quick testing
    "bcsstk01": ("HB", "bcsstk01"),  # 48x48, stiffness matrix
    "bcsstk02": ("HB", "bcsstk02"),  # 66x66, stiffness matrix
    "nos1": ("HB", "nos1"),  # 237x237, symmetric
    "nos2": ("HB", "nos2"),  # 957x957, symmetric
    "nos3": ("HB", "nos3"),  # 960x960, symmetric
    "nos4": ("HB", "nos4"),  # 100x100, symmetric
    "nos5": ("HB", "nos5"),  # 468x468, symmetric
    "nos6": ("HB", "nos6"),  # 675x675, symmetric
    "nos7": ("HB", "nos7"),  # 729x729, symmetric
    
    # Medium matrices for benchmarking
    "1138_bus": ("HB", "1138_bus"),  # 1138x1138, power network
    "494_bus": ("HB", "494_bus"),  # 494x494, power network
    "662_bus": ("HB", "662_bus"),  # 662x662, power network
    "685_bus": ("HB", "685_bus"),  # 685x685, power network
    
    # Poisson/Laplacian matrices (SPD)
    "mesh1e1": ("Pothen", "mesh1e1"),  # 48x48
    "mesh1em1": ("Pothen", "mesh1em1"),  # 48x48
    "mesh1em6": ("Pothen", "mesh1em6"),  # 48x48
    "mesh2e1": ("Pothen", "mesh2e1"),  # 306x306
    "mesh2em5": ("Pothen", "mesh2em5"),  # 306x306
    "mesh3e1": ("Pothen", "mesh3e1"),  # 289x289
    "mesh3em5": ("Pothen", "mesh3em5"),  # 289x289
    
    # Larger SPD matrices
    "apache1": ("GHS_psdef", "apache1"),  # 80800x80800, FEM
    "apache2": ("GHS_psdef", "apache2"),  # 715176x715176, FEM
    "G3_circuit": ("AMD", "G3_circuit"),  # 1585478x1585478, circuit
    
    # Structural engineering
    "offshore": ("Um", "offshore"),  # 259789x259789
    "parabolic_fem": ("Wissgott", "parabolic_fem"),  # 525825x525825
    
    # CFD
    "atmosmodd": ("Bourchtein", "atmosmodd"),  # 1270432x1270432
    "atmosmodl": ("Bourchtein", "atmosmodl"),  # 1489752x1489752
}


def get_cache_dir() -> Path:
    """Get or create cache directory for downloaded matrices."""
    cache_dir = Path.home() / ".cache" / "suitesparse"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_matrix(name: str, group: Optional[str] = None) -> Path:
    """
    Download a matrix from SuiteSparse Matrix Collection.
    
    Parameters
    ----------
    name : str
        Matrix name (e.g., 'bcsstk01', 'apache1')
    group : str, optional
        Matrix group (e.g., 'HB', 'GHS_psdef'). If None, looks up in POPULAR_MATRICES.
    
    Returns
    -------
    Path
        Path to the downloaded .mtx file
    """
    cache_dir = get_cache_dir()
    
    # Look up group if not provided
    if group is None:
        if name in POPULAR_MATRICES:
            group, name = POPULAR_MATRICES[name]
        else:
            raise ValueError(
                f"Unknown matrix '{name}'. Either provide group or use one of: "
                f"{list(POPULAR_MATRICES.keys())[:10]}..."
            )
    
    # Check if already downloaded
    mtx_path = cache_dir / f"{name}.mtx"
    if mtx_path.exists():
        print(f"Using cached matrix: {mtx_path}")
        return mtx_path
    
    # Download from SuiteSparse
    url = f"{MATRIX_MARKET_BASE}/{group}/{name}.tar.gz"
    tar_path = cache_dir / f"{name}.tar.gz"
    
    print(f"Downloading {name} from {url}...")
    try:
        urllib.request.urlretrieve(url, tar_path)
    except Exception as e:
        raise RuntimeError(f"Failed to download matrix: {e}")
    
    # Extract
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(cache_dir)
    
    # Find .mtx file
    extracted_dir = cache_dir / name
    mtx_files = list(extracted_dir.glob("*.mtx"))
    if not mtx_files:
        raise RuntimeError(f"No .mtx file found in {extracted_dir}")
    
    # Move to cache root and cleanup
    src_mtx = mtx_files[0]
    shutil.move(str(src_mtx), str(mtx_path))
    shutil.rmtree(extracted_dir)
    tar_path.unlink()
    
    print(f"Matrix saved to: {mtx_path}")
    return mtx_path


def list_available_matrices() -> None:
    """Print available matrices with their properties."""
    print("\nAvailable matrices in POPULAR_MATRICES:")
    print("-" * 60)
    print(f"{'Name':<20} {'Group':<15} {'Description'}")
    print("-" * 60)
    for name, (group, _) in sorted(POPULAR_MATRICES.items()):
        print(f"{name:<20} {group:<15}")
    print("-" * 60)


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_solve(
    A: SparseTensor,
    b: torch.Tensor,
    backend: str = "auto",
    method: str = "auto",
    num_runs: int = 3,
    warmup: int = 1,
) -> Dict:
    """
    Benchmark solving Ax = b.
    
    Returns
    -------
    dict
        Results including time, error, memory usage
    """
    device = A.device
    n = A.shape[0]
    
    # Warmup
    for _ in range(warmup):
        try:
            x = A.solve(b, backend=backend, method=method)
            torch.cuda.synchronize() if device.type == "cuda" else None
        except Exception as e:
            return {"error": str(e), "time_ms": float("inf")}
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        x = A.solve(b, backend=backend, method=method)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    # Compute residual error
    residual = A @ x - b
    rel_error = torch.norm(residual) / torch.norm(b)
    
    # Memory usage (CUDA only)
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_mb = 0
    
    return {
        "backend": backend,
        "method": method,
        "time_ms": np.mean(times),
        "time_std_ms": np.std(times),
        "rel_error": rel_error.item(),
        "mem_mb": mem_mb,
        "n": n,
        "nnz": A.nnz,
    }


def run_benchmark(
    matrix_name: str,
    device: str = "cpu",
    backends: Optional[List[str]] = None,
    dtype: torch.dtype = torch.float64,
) -> List[Dict]:
    """
    Run comprehensive benchmark on a matrix.
    
    Parameters
    ----------
    matrix_name : str
        Matrix name or path to .mtx file
    device : str
        'cpu' or 'cuda'
    backends : list, optional
        List of backends to test. If None, tests all available.
    dtype : torch.dtype
        Data type for computation
    
    Returns
    -------
    list
        List of benchmark results
    """
    # Load matrix
    if os.path.exists(matrix_name):
        mtx_path = Path(matrix_name)
    else:
        mtx_path = download_matrix(matrix_name)
    
    # Get matrix info
    info = load_mtx_info(mtx_path)
    print(f"\nMatrix: {mtx_path.stem}")
    print(f"  Shape: {info['shape']}")
    print(f"  NNZ: {info['nnz']}")
    print(f"  Symmetry: {info['symmetry']}")
    print(f"  Field: {info['field']}")
    
    # Load matrix
    print(f"\nLoading matrix to {device}...")
    A = load_mtx(mtx_path, dtype=dtype, device=device)
    
    # Ensure matrix is square
    if A.shape[0] != A.shape[1]:
        print(f"Warning: Matrix is not square ({A.shape}), skipping...")
        return []
    
    # Create RHS vector (random or based on known solution)
    n = A.shape[0]
    x_true = torch.ones(n, dtype=dtype, device=device)
    b = A @ x_true
    
    print(f"\nProblem size: n={n:,}, nnz={A.nnz:,}")
    print(f"Density: {A.nnz / (n * n) * 100:.4f}%")
    
    # Determine backends to test
    if backends is None:
        if device == "cuda":
            backends = ["cudss", "pytorch"]
        else:
            backends = ["scipy", "pytorch"]
    
    # Backend-method combinations to test
    test_configs = []
    for backend in backends:
        if backend == "scipy":
            test_configs.extend([
                ("scipy", "superlu"),
            ])
        elif backend == "cudss":
            test_configs.extend([
                ("cudss", "cholesky"),
                ("cudss", "lu"),
            ])
        elif backend == "pytorch":
            test_configs.extend([
                ("pytorch", "cg"),
                ("pytorch", "bicgstab"),
            ])
        else:
            test_configs.append((backend, "auto"))
    
    # Run benchmarks
    results = []
    print("\nBenchmarking...")
    print("-" * 70)
    print(f"{'Backend':<15} {'Method':<12} {'Time (ms)':<12} {'Rel Error':<12} {'Mem (MB)':<10}")
    print("-" * 70)
    
    for backend, method in test_configs:
        try:
            result = benchmark_solve(A, b, backend=backend, method=method)
            result["matrix"] = mtx_path.stem
            results.append(result)
            
            if "error" in result:
                print(f"{backend:<15} {method:<12} {'FAILED':<12} {result['error']}")
            else:
                print(f"{result['backend']:<15} {result['method']:<12} "
                      f"{result['time_ms']:<12.2f} {result['rel_error']:<12.2e} "
                      f"{result['mem_mb']:<10.1f}")
        except Exception as e:
            print(f"{backend:<15} {method:<12} {'ERROR':<12} {str(e)[:30]}")
            results.append({
                "backend": backend,
                "method": method,
                "error": str(e),
                "matrix": mtx_path.stem,
            })
    
    print("-" * 70)
    
    return results


def benchmark_multiple_matrices(
    matrix_names: List[str],
    device: str = "cpu",
    output_file: Optional[str] = None,
) -> None:
    """Benchmark multiple matrices and save results."""
    import json
    
    all_results = []
    
    for name in matrix_names:
        print(f"\n{'='*70}")
        print(f"Benchmarking: {name}")
        print(f"{'='*70}")
        
        try:
            results = run_benchmark(name, device=device)
            all_results.extend(results)
        except Exception as e:
            print(f"Failed to benchmark {name}: {e}")
    
    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Group by matrix
    from collections import defaultdict
    by_matrix = defaultdict(list)
    for r in all_results:
        if "error" not in r:
            by_matrix[r["matrix"]].append(r)
    
    for matrix, results in sorted(by_matrix.items()):
        print(f"\n{matrix}:")
        best = min(results, key=lambda x: x["time_ms"])
        print(f"  Best: {best['backend']}+{best['method']} ({best['time_ms']:.2f} ms)")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark torch-sla with SuiteSparse Matrix Collection"
    )
    parser.add_argument(
        "--matrix", "-m",
        type=str,
        default=None,
        help="Matrix name or path to .mtx file"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default=None,
        help="Backend to use (auto, scipy, pytorch, cudss)"
    )
    parser.add_argument(
        "--list-matrices", "-l",
        action="store_true",
        help="List available matrices"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch benchmark on multiple matrices"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    
    args = parser.parse_args()
    
    if args.list_matrices:
        list_available_matrices()
        return
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    # Set up backends
    backends = None
    if args.backend:
        backends = [args.backend]
    
    if args.batch:
        # Run on multiple matrices (small to medium for quick testing)
        test_matrices = [
            "bcsstk01", "bcsstk02", "nos1", "nos4", "nos5",
            "494_bus", "662_bus", "685_bus",
        ]
        benchmark_multiple_matrices(
            test_matrices,
            device=args.device,
            output_file=args.output or "suitesparse_results.json",
        )
    elif args.matrix:
        # Run on single matrix
        run_benchmark(
            args.matrix,
            device=args.device,
            backends=backends,
        )
    else:
        # Demo with a small matrix
        print("No matrix specified. Running demo with 'bcsstk01'...")
        print("Use --list-matrices to see available matrices")
        print("Use --matrix NAME to benchmark a specific matrix")
        print()
        run_benchmark("bcsstk01", device=args.device, backends=backends)


if __name__ == "__main__":
    main()

