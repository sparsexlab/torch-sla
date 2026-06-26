#!/usr/bin/env python
"""Distributed ``DSparseTensor`` save / load round-trip.

Each rank writes its own shard to ``./distributed_save/`` and then
reads it back. The reloaded tensor reproduces the original matvec
bit-for-bit. Rank 0 also writes a ``metadata.json`` sidecar that single
inspector processes can read via ``torch_sla.load_metadata`` /
``torch_sla.load_sparse_shard``.

Device-aware: uses NCCL + CUDA (one GPU per ``LOCAL_RANK``) when a GPU
is visible, else falls back to gloo + CPU so it still runs on a laptop.

Run::

    # single node (one box, N procs)
    torchrun --standalone --nproc_per_node=4 distributed_persistence.py

    # multiple nodes (run on EVERY node; HEAD_NODE_IP reachable by all)
    torchrun --nnodes=2 --nproc_per_node=4 \
        --rdzv-id=sla --rdzv-backend=c10d \
        --rdzv-endpoint=HEAD_NODE_IP:29500 distributed_persistence.py
"""

import os
import shutil
import torch
import torch.distributed as dist


def main():
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        mesh_device = "cuda"
    else:
        device = torch.device("cpu")
        mesh_device = "cpu"

    from torch.distributed.device_mesh import init_device_mesh
    from torch_sla import DSparseTensor, SparseTensor, load_metadata

    n = 200
    A = SparseTensor.tridiagonal(n, diag=4.0, off_diag=-1.0).to(device)

    mesh = init_device_mesh(mesh_device, (world_size,))
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
    x_global = torch.randn(n, dtype=torch.float64, device=device)
    y_ref = (D @ D.scatter(x_global)).full_tensor()
    y_back = (D2 @ D2.scatter(x_global)).full_tensor()
    err = (y_ref - y_back).abs().max().item()

    print(f"[rank {rank}] round-trip err={err:.2e}")

    if rank == 0:
        meta = load_metadata(out_dir)
        print(f"{'=' * 60}\nDistributed persistence  "
              f"(world={world_size}, backend={backend}, device={mesh_device})"
              f"\n{'=' * 60}")
        print(f"metadata: shape={meta['shape']} dtype={meta['dtype']} "
              f"num_partitions={meta['num_partitions']}")
        print(f"out_dir : {out_dir}")
        print(f"round-trip max |y - y'| = {err:.2e}")
        assert err == 0.0, "persistence drift"
        print("\nLoad round-trip verified.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
