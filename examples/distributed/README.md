# Distributed sparse tensor examples

Multi-process distributed sparse linear algebra via `DSparseTensor` —
row-sharded matvec, Krylov solves, LOBPCG eigsh, and persistence.

## Quick start

```bash
chmod +x launch.sh
./launch.sh all 4          # run every example with 4 procs

# or one at a time
torchrun --standalone --nproc_per_node=4 distributed_matvec.py
torchrun --standalone --nproc_per_node=4 distributed_solve.py
torchrun --standalone --nproc_per_node=4 distributed_eigsh.py
torchrun --standalone --nproc_per_node=4 distributed_persistence.py
```

The examples use `gloo` so they work on CPU-only machines. Swap to
`nccl` (`dist.init_process_group(backend="nccl")`) for multi-GPU.

## Examples

| File                          | Operation                       | Key API                                                  |
|-------------------------------|---------------------------------|----------------------------------------------------------|
| `distributed_matvec.py`       | `y = A @ x`                     | `DSparseTensor.partition` → `D.scatter` → `D @ x_dt`     |
| `distributed_solve.py`        | `A x = b` (CG + Jacobi)         | `solve(D, b_dt)` under `SolverConfig`                    |
| `distributed_eigsh.py`        | `A v = λ v` (LOBPCG)            | `D.eigsh(k, which="LM")`                                 |
| `distributed_persistence.py`  | save → load → re-matvec         | `D.save(dir)` / `DSparseTensor.load(dir, mesh)`          |

## API at a glance

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch_sla import DSparseTensor, SparseTensor, solve, SolverConfig

dist.init_process_group(backend="gloo")
rank, world = dist.get_rank(), dist.get_world_size()

# Build (or load) a global SparseTensor on every rank.
A = SparseTensor(values, row, col, shape=(N, N))

# Row-shard across the device mesh.
mesh = init_device_mesh("cpu", (world,))
D = DSparseTensor.partition(A, mesh, partition_method="simple")  # or hilbert/metis/rcb

# Distributed ops:
y_dt   = D @ D.scatter(x_global)            # matvec, returns DTensor[Shard(0)]
x_dt   = solve(D, D.scatter(b))             # unified Krylov dispatch
λ, V   = D.eigsh(k=5, which="LM")           # distributed LOBPCG

# Tensor-mirror props:
D.shape, D.ndim, D.dtype, D.device, D.is_square, D.is_cuda
D.nnz                # local
D.global_nnz()       # all-rank reduce

# Reductions / math:
D.sum(); D.mean(); D.max(); D.min(); D.norm("fro")
(D + 1.5) * 2.0 / D.norm("fro")             # element-wise, returns DSparseTensor

# Persistence:
D.save("path/")                              # per-rank shard + metadata.json
D2 = DSparseTensor.load("path/", mesh)
```

## Architecture

```
Global A (N×N) partitioned across P ranks:

Rank 0:  [ owned rows 0..k_0    +  halo cols ]
Rank 1:  [ owned rows k_0..k_1  +  halo cols ]
Rank 2:  [ owned rows k_1..k_2  +  halo cols ]
…

Each rank holds only its own COO chunk in local coords + a Partition
struct (owned_nodes / halo_nodes / send_indices / recv_indices /
local_to_global). Communication:

* Halo exchange    — point-to-point with neighbour ranks (NCCL/gloo P2P)
* Global reductions— all_reduce for dot products, residual checks, eigsh RR
```

## Partition methods

* `"simple"`   — contiguous slices of row indices. Fast, no quality guarantees.
* `"rcb"`      — Recursive Coordinate Bisection (needs `coords`).
* `"hilbert"`  — Hilbert space-filling curve (needs 2-D / 3-D `coords`).
* `"metis"`    — METIS graph partitioner via `pymetis` (falls back to `"simple"`).

## See also

* `tests/test_distributed_*_multiprocess.py` — production-style multiproc tests.
* `docs/source/architecture.rst`             — full DSparseTensor architecture.
* `benchmarks/benchmark_distributed.py`      — 2× A100 NCCL scaling numbers.
