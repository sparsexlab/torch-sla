# Changelog

All notable changes to this project are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-06-17

First minor release after the v0.2.1 baseline. This is a **breaking
release** -- expect import-path and API changes if you were pinned to
0.2.x.

### Added

- **Unified `solve()` API** (`torch_sla.solve`). Single entry point with a
  `SolverConfig` dataclass; scoped defaults via context manager / decorator;
  `SolverConfig.for_spd_gpu()`, `for_matrix(A)`, and other presets.
- **AmgX backend** for GPU AMG + Krylov via the new
  [`torch-amgx`](https://github.com/sparsexlab/torch-amgx) bridge. Inner
  preconditioner kwarg is plumbed end-to-end.
- **PyAMG-hybrid backend** (CPU setup + `torch.sparse` V-cycle) -- runs
  cross-platform, GPU when available.
- **`torch.sparse.spsolve` wrapper backend** (experimental).
- **LRU solver cache** (`SOLVER_CACHE`) -- keys by sparsity layout, reuses
  PyAMG / AmgX hierarchies across solves with matching structure.
- **Distributed (`DSparseTensor`) DTensor-mirror surface**:
  `from_local` / `to_local` / `full_tensor` / `matmul` → `Shard(0)`; new
  placement vocabulary (`DSparseSpec`, `VertexShard`,
  `VertexShardReplicated`, `BatchShard`, `Replicated`, `SparseShard`).
- **Distributed Krylov + preconditioner shard-space stack**:
  block-Jacobi, AMG (PyAMG + torch-amgx) on shards, gather-owned-to-global,
  vectorised scatter.
- **Single-process load of a sharded archive**:
  ``load_dsparse(dir, target_world_size=1)`` (and
  ``DSparseTensor.load(..., target_world_size=1)``) stitches every
  shard back into one ``mesh=None`` trivial DSparseTensor, no
  process group required. Useful for offline inspection. True
  ``stored != target != 1`` repartition raises ``NotImplementedError``
  with a workaround hint (deferred to ``redistribute()`` in 0.4).
- **Differentiable Hutchinson logdet** (single + distributed); SVD adjoint
  via Townsend's formula.
- **CUDA backward for `det()`** via cuDSS chunked solve (drops the dense
  `O(n^2)` inverse).
- **`sampled_addmm` fast path** for `SparseSparseMatmul` backward (no
  `to_dense`).
- **LOBPCG rewrite** (#43, in response to @TrinitroCat's review in
  #32): proper 3-block `[X | R | P]` subspace, pre-allocated buffers
  (no `torch.cat` per iter), CGS2 reorthogonalisation in place of
  full QR. Shared `_lobpcg_core` between single-device and
  distributed `eigsh`. ~8.5× fewer matvecs on clustered spectra
  (see `examples/lobpcg_convergence_benchmark.py`). Convergence
  criterion and reorthogonalisation further refined in PR #45 —
  switched to LAPACK QR for the inner reorth (~2× faster than
  `torch.lobpcg` on CUDA at correct precision); MPS is now flagged
  not-recommended due to upstream gaps.
- **Complex dtype support** + Wirtinger adjoint; Hermitian / HPD matrix
  types auto-detected.
- **Benchmark API** + SuiteSparse / Synthetic PDE / DIMACS10 datasets.

### Changed

- **Package layout**: `torch_sla/sparse_tensor.py` (4.8k-line monolith)
  split into a `torch_sla.sparse_tensor` package (`core`, `autograd`,
  `linalg`, `convert`, `matmul`, `ops`, `reductions`, `structural`,
  `graph`, `list`, `viz`, `utils`). Public `from torch_sla import
  SparseTensor` is unchanged; deep imports moved.
- **`torch_sla.distributed.distributed.py`** split into `partition.py` +
  `distributed_solve.py`.
- **AmgX backend** switched from `pyamgx` (Python) to in-house
  `torch-amgx` (C++/pybind11) for first-class CUDA wheel support.
- **CG backward** check converges every iteration (no 0/0 NaN near early
  convergence).

### Removed (breaking)

- **`DSparseMatrix`** is gone -- use `DSparseTensor` everywhere. The two
  collapsed: `DSparseTensor` now is the only public distributed sparse
  type.
- Distributed I/O helpers: `load_sparse_as_partition`, `save_distributed`,
  `load_partition`, `load_distributed_as_sparse` removed in favour of the
  symmetric `save_dsparse` / `load_dsparse` + `save_sparse_sharded` /
  `load_sparse_shard` pair.

### Fixed

- Block-Jacobi distributed preconditioner: real per-rank LU instead of the
  earlier placeholder; smart fallback when no CUDA backend is available.
- `DSparseTensor.partition` broadcasts `partition_ids` from rank 0 so all
  ranks agree byte-for-byte (NCCL backend safe).
- Eigsh: regression where shrinking the internal block size below
  `min(2k, k+2)` made clustered-spectrum extremes converge to the wrong
  pair on small problems.
- CUDA `svd_lowrank` path: raise `NotImplementedError` instead of the
  silent scipy round-trip (defer Lanczos bidiagonalisation to a future PR).
- Many import / packaging fixes after the package split (`scipy_lu`,
  `LUFactorization`, `SparseTensorList`).

### Migration

```python
# Before (0.2.x)
from torch_sla.distributed import DSparseMatrix
A_d = DSparseMatrix(...)

# After (0.3.x)
from torch_sla.distributed import DSparseTensor
A_d = DSparseTensor.from_local(...)
```

```python
# Before
from torch_sla.io import load_partition, save_distributed

# After
from torch_sla.io import load_dsparse, save_dsparse
```

The old single-file `from torch_sla.sparse_tensor import X` style still
works -- the package `__init__` re-exports the public surface.

## [0.2.1] - 2026-05-20

Last 0.2.x release. See git history for details.
