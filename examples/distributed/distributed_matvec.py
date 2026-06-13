#!/usr/bin/env python
"""Distributed matrix-vector multiplication via ``DSparseTensor``.

Each rank holds a row-sharded slice of the same global matrix and runs
``D @ x`` with automatic NCCL/gloo halo exchange. Result is compared
against the single-process baseline so any drift fails the example.

Run::

    torchrun --standalone --nproc_per_node=4 distributed_matvec.py
"""

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

    # One-line SPD tridiagonal -- diag=4, off=-1.
    n = 200
    A = SparseTensor.tridiagonal(n, diag=4.0, off_diag=-1.0)

    # Row-shard across the device mesh.
    mesh = init_device_mesh("cpu", (world_size,))
    D = DSparseTensor.partition(A, mesh, partition_method="simple")

    # Scatter a global x to Shard(0) DTensor, run matvec, gather.
    torch.manual_seed(0)
    x_global = torch.randn(n, dtype=torch.float64)
    x_dt = D.scatter(x_global)
    y_dt = D @ x_dt
    y_global = y_dt.full_tensor()

    # Collective reductions must run on EVERY rank.
    global_nnz = D.global_nnz()
    err = (A @ x_global - y_global).abs().max().item()

    print(f"[rank {rank}] owned={D._spec.placement.partition.owned_nodes.numel()} "
          f"local_nnz={D.nnz} max|y_dist - y_ref|={err:.2e}")

    if rank == 0:
        assert err < 1e-12, f"distributed matvec diverged ({err:.2e})"
        print(f"\nDistributed matvec completed. global nnz={global_nnz}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
