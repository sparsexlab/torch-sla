# Benchmark Results

This directory contains performance benchmarks for torch-sla.

## Layout

Benchmarks are organized by domain:

- `scaling/`     — op/device scaling curves (`benchmark_all_ops_scaling.py`, `benchmark_scaling_curves.py`, `benchmark_op_scaling.py`, `benchmark_single_gpu_scaling.py`, `benchmark_multi_gpu_scaling.py`, `plot_device_compare.py`)
- `distributed/` — multi-GPU / distributed (`benchmark_distributed.py`, `benchmark_distributed_4gpu.py`, `benchmark_distributed_perf.py`, `benchmark_distributed_scaling.py`)
- `solvers/`     — sparse solvers (`benchmark_solvers.py`, `benchmark_large_scale.py`, `benchmark_complex_solve.py`, `benchmark_suitesparse.py`)
- `ops/`         — individual ops (`benchmark_mm.py`, `benchmark_determinant.py`, `benchmark_adjoint_vs_naive.py`, `benchmark_inverse_coefficient.py`)
- `results/`     — shared generated output dir (gitignored)
- `dev/`         — experimental / in-progress benchmarks

## Running Benchmarks

```bash
# Run specific benchmark
python benchmarks/ops/benchmark_determinant.py

# Results are saved to
ls benchmarks/results/benchmark_determinant/
```

## Available Benchmarks

### 1. Determinant Computation

**Script**: `ops/benchmark_determinant.py`  
**Results**: `results/benchmark_determinant/`

#### Key Findings

| Matrix Size | CPU (Sparse) | CUDA (Dense) | CPU-for-CUDA | Gradient |
|-------------|--------------|--------------|--------------|----------|
| n=10        | 0.56 ms      | 0.49 ms      | 0.57 ms      | 4.04 ms  |
| n=100       | 0.57 ms      | 0.31 ms      | 0.60 ms      | 39.19 ms |
| n=500       | 0.97 ms      | 1.12 ms      | 0.95 ms      | 253 ms   |
| n=1000      | 1.36 ms      | 2.51 ms      | 1.33 ms      | 674 ms   |

**Recommendations**:
1. ✅ **Always use CPU** for sparse determinant computation
2. ❌ **Avoid CUDA** - requires dense conversion (3-180x slower)
3. ⚡ **Use `.cpu().det()`** even for CUDA tensors

```python
# ❌ Slow for sparse matrices
det = A_cuda.det()  # 2.5 ms

# ✅ Fast - use CPU even for CUDA tensors
det = A_cuda.cpu().det()  # 1.3 ms (1.9x faster!)
```

**Why is CUDA slow?**
- cuDSS doesn't expose sparse determinant computation
- Must convert to dense: O(n²) memory + O(n³) computation
- CPU sparse LU: O(nnz^1.5) ≈ O(n^1.5) for tridiagonal matrices

#### Visualization

![Determinant Benchmark](results/benchmark_determinant/benchmark_determinant.png)

### 2. Distributed Computing

**Script**: `distributed/benchmark_distributed.py`  
**Results**: `results/benchmark_distributed/`

See distributed benchmark results for scaling analysis.

## Adding New Benchmarks

1. Create `benchmark_<feature>.py` in the appropriate domain subdir (`scaling/`, `distributed/`, `solvers/`, `ops/`)
2. Results will be saved to `results/benchmark_<feature>/`
3. Follow the template in `CONTRIBUTING.md`
4. Update this README with your findings

## System Information

Benchmarks were run on:
- **GPU**: NVIDIA GPU with CUDA support
- **PyTorch**: Latest version with CUDA
- **Python**: 3.10+

Performance may vary depending on hardware.

