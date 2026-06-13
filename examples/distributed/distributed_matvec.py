#!/usr/bin/env python
"""Distributed matrix-vector multiplication via ``DSparseTensor``.

Each rank holds a row-sharded slice of the same global matrix and runs
``D @ x`` with automatic NCCL/gloo halo exchange. Result is compared
against the single-process baseline so any drift fails the example.

Run::

    torchrun --standalone --nproc_per_node=4 distributed_matvec.py
"""

import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"{'=' * 60}\nDistributed Matvec: y = A @ x  (world={world_size})\n{'=' * 60}")

    from torch.distributed.device_mesh import init_device_mesh
    from torch_sla import DSparseTensor, SparseTensor

    # ── Build the global matrix (every rank holds the COO -- only the
    # partitioner output is rank-specific). For multi-TB matrices use
    # ``save_sparse_sharded`` / ``DSparseTensor.load`` instead.
    n = 200
    idx = torch.arange(n)
    val = torch.cat([
        torch.full((n,), 4.0, dtype=torch.float64),
        torch.full((n - 1,), -1.0, dtype=torch.float64),
        torch.full((n - 1,), -1.0, dtype=torch.float64),
    ])
    row = torch.cat([idx, idx[1:], idx[:-1]])
    col = torch.cat([idx, idx[:-1], idx[1:]])
    A = SparseTensor(val, row, col, shape=(n, n))

    # ── Partition into row shards across the device mesh. ──
    mesh = init_device_mesh("cpu", (world_size,))
    D = DSparseTensor.partition(A, mesh, partition_method="simple")

    # ── Scatter a global x to Shard(0) DTensor, run matvec, gather. ──
    torch.manual_seed(0)
    x_global = torch.randn(n, dtype=torch.float64)
    x_dt = D.scatter(x_global)
    y_dt = D @ x_dt
    y_global = y_dt.full_tensor()

    # ── Verify against single-process baseline. ──
    y_ref = A @ x_global
    err = (y_ref - y_global).abs().max().item()

    print(f"[rank {rank}] owned={D._spec.placement.partition.owned_nodes.numel()} "
          f"nnz={D.nnz} max|y_dist - y_ref|={err:.2e}")
    if rank == 0:
        assert err < 1e-12, f"distributed matvec diverged ({err:.2e})"
        print(f"\nDistributed matvec completed. global nnz={D.global_nnz()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
