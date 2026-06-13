#!/usr/bin/env python
"""Distributed ``DSparseTensor`` save / load round-trip.

Each rank writes its own shard to ``./distributed_save/`` and then
reads it back. The reloaded tensor reproduces the original matvec
bit-for-bit. Rank 0 also writes a ``metadata.json`` sidecar that single
inspector processes can read via ``torch_sla.load_metadata`` /
``torch_sla.load_sparse_shard``.

Run::

    torchrun --standalone --nproc_per_node=4 distributed_persistence.py
"""

import os
import shutil
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    from torch.distributed.device_mesh import init_device_mesh
    from torch_sla import DSparseTensor, SparseTensor, load_metadata

    n = 200
    A = SparseTensor.tridiagonal(n, diag=4.0, off_diag=-1.0)

    mesh = init_device_mesh("cpu", (world_size,))
    D = DSparseTensor.partition(A, mesh, partition_method="simple")

    out_dir = os.path.join(os.path.dirname(__file__), "distributed_save")
    if rank == 0:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    dist.barrier()

    # Each rank persists its own shard.
    D.save(out_dir)
    dist.barrier()

    # Each rank reads its own shard back.
    D2 = DSparseTensor.load(out_dir, mesh=mesh)

    # Verify zero matvec drift -- collective work; every rank runs this.
    torch.manual_seed(0)
    x_global = torch.randn(n, dtype=torch.float64)
    y_ref = (D @ D.scatter(x_global)).full_tensor()
    y_back = (D2 @ D2.scatter(x_global)).full_tensor()
    err = (y_ref - y_back).abs().max().item()

    print(f"[rank {rank}] round-trip err={err:.2e}")

    if rank == 0:
        meta = load_metadata(out_dir)
        print(f"{'=' * 60}\nDistributed persistence  (world={world_size})\n{'=' * 60}")
        print(f"metadata: shape={meta['shape']} dtype={meta['dtype']} "
              f"num_partitions={meta['num_partitions']}")
        print(f"out_dir : {out_dir}")
        print(f"round-trip max |y - y'| = {err:.2e}")
        assert err == 0.0, "persistence drift"
        print("\nLoad round-trip verified.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
