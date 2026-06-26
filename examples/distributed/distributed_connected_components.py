#!/usr/bin/env python
"""Distributed connected components via ``DSparseTensor.connected_components``.

Row-shards a block-diagonal adjacency (several disjoint path graphs) and
runs distributed label propagation with boundary-label halo exchange.
Each rank gets its owned slice of the contiguous component labelling; the
global component count and the induced node partition are cross-checked
against a single-process ``scipy.sparse.csgraph.connected_components`` on
the full graph, so any divergence fails the example.

Device-aware: uses NCCL + CUDA (one GPU per ``LOCAL_RANK``) when a GPU
is visible, else falls back to gloo + CPU so it still runs on a laptop.

Run::

    # single node (one box, N procs)
    torchrun --standalone --nproc_per_node=4 distributed_connected_components.py

    # multiple nodes (run on EVERY node; HEAD_NODE_IP reachable by all)
    torchrun --nnodes=2 --nproc_per_node=4 \
        --rdzv-id=sla --rdzv-backend=c10d \
        --rdzv-endpoint=HEAD_NODE_IP:29500 distributed_connected_components.py
"""

import os
import torch
import torch.distributed as dist


def build_block_path_graph(num_blocks: int, block_size: int, device):
    """Symmetric adjacency of ``num_blocks`` disjoint path graphs.

    Node ``b * block_size + i`` is connected to ``b * block_size + i+1``
    inside its block only. Each node also carries a self-loop so every
    row is present (self-loops are ignored by connectivity but keep the
    partition map well-defined). Returns a :class:`SparseTensor`.
    """
    from torch_sla import SparseTensor

    rows, cols = [], []
    n = num_blocks * block_size
    for b in range(num_blocks):
        base = b * block_size
        for i in range(block_size):
            node = base + i
            rows.append(node)            # self-loop (diagonal)
            cols.append(node)
            if i + 1 < block_size:
                nxt = base + i + 1
                rows.append(node)        # forward edge
                cols.append(nxt)
                rows.append(nxt)         # backward edge (undirected)
                cols.append(node)
    row = torch.tensor(rows, dtype=torch.int64, device=device)
    col = torch.tensor(cols, dtype=torch.int64, device=device)
    val = torch.ones(row.numel(), dtype=torch.float64, device=device)
    return SparseTensor(val, row, col, shape=(n, n))


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

    if rank == 0:
        print(f"{'=' * 60}\nDistributed connected_components  "
              f"(world={world_size}, backend={backend}, device={mesh_device})"
              f"\n{'=' * 60}")

    from torch.distributed.device_mesh import init_device_mesh
    from torch_sla import DSparseTensor
    from torch_sla.distributed import gather_owned_to_global

    # A known graph: NUM_BLOCKS disjoint paths -> NUM_BLOCKS components.
    num_blocks, block_size = 6, 20
    n = num_blocks * block_size
    A = build_block_path_graph(num_blocks, block_size, device)

    mesh = init_device_mesh(mesh_device, (world_size,))
    D = DSparseTensor.partition(A, mesh, partition_method="simple")

    # Distributed connected components -- collective; every rank runs it.
    labels_owned, num_components = D.connected_components()

    # Assemble the global label vector on every rank for the cross-check.
    owned = D._spec.placement.partition.owned_nodes.to(
        device=device, dtype=torch.int64)
    labels_global = gather_owned_to_global(
        owned, labels_owned.to(torch.float64), n).to(torch.int64).cpu()

    print(f"[rank {rank}] owned={owned.numel()} "
          f"local_labels(min..max)="
          f"{int(labels_owned.min())}..{int(labels_owned.max())} "
          f"num_components={num_components}")

    # Cross-check on rank 0 against scipy (the canonical reference). We
    # compare the *induced partition*, which is invariant to how each
    # method numbers its components.
    if rank == 0:
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components as scc

        A_cpu = A.to("cpu")
        A_sp = sp.coo_matrix(
            (A_cpu.values.numpy(),
             (A_cpu.row_indices.numpy(), A_cpu.col_indices.numpy())),
            shape=(n, n),
        ).tocsr()
        ref_n, ref_labels = scc(A_sp, directed=False)

        ours = labels_global.numpy()

        def canonical(lab):
            # Relabel by first-appearance order so two equivalent
            # partitions compare equal regardless of component numbering.
            remap, out = {}, np.empty_like(lab)
            for i, v in enumerate(lab):
                if v not in remap:
                    remap[v] = len(remap)
                out[i] = remap[v]
            return out

        same_partition = bool(
            np.array_equal(canonical(ours), canonical(ref_labels)))

        print(f"\n     distributed num_components = {num_components}")
        print(f"     scipy       num_components = {ref_n}  "
              f"(expected {num_blocks})")
        print(f"     induced partition matches scipy = {same_partition}")

        assert num_components == ref_n == num_blocks, (
            f"component count mismatch: dist={num_components} "
            f"scipy={ref_n} expected={num_blocks}")
        assert same_partition, "distributed labelling disagrees with scipy"
        print("\nDistributed connected_components verified.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
