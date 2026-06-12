#!/usr/bin/env python
"""DSparseTensor matvec backed by SparseTensor + spec.partition.

Verifies the new ``DSparseTensor.from_sparse_local`` constructor + the
``_matmul_row_shard_via_sparse_tensor`` code path produce numerically
identical results to the legacy DSparseMatrix-backed path. This is the
"DSparseMatrix dissolution" load-bearing step -- once both paths agree
on every test we can flip the default constructor + start removing
DSparseMatrix.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _matvec_worker(rank: int, world_size: int, port: int,
               out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        from torch_sla import DSparseTensor, SparseTensor
        from torch_sla.datasets import Synthetic

        bench = Synthetic["poisson_2d_16"]
        N = bench.shape[0]
        A_global = SparseTensor(bench.val, bench.row, bench.col,
                                 bench.shape)

        # Old path: partition_for_rank → DSparseMatrix → from_local.
        local_mat = A_global.partition_for_rank(
            rank, world_size, partition_method="simple")
        mesh = init_device_mesh("cpu", (world_size,))
        D_old = DSparseTensor.from_local(local_mat, mesh)
        partition = local_mat.partition

        # SparseTensor-backed path: extract_partition + from_sparse_local.
        local_st = A_global.extract_partition(partition)
        D_new = DSparseTensor.from_sparse_local(
            local_st, mesh, partition,
            global_shape=A_global.shape,
        )

        # Verify the instance uses the SparseTensor backing.
        assert D_new._local_tensor is not None
        assert D_new._local_matrix is None
        assert D_old._local_tensor is None
        assert D_old._local_matrix is not None

        # Shared input.
        torch.manual_seed(0)
        x_global = torch.randn(N, dtype=torch.float64)
        x_owned = x_global[partition.owned_nodes]
        x_dt = DTensor.from_local(x_owned, mesh, [Shard(0)])

        y_old_dt = D_old @ x_dt
        y_new_dt = D_new @ x_dt
        y_old = y_old_dt.to_local()
        y_new = y_new_dt.to_local()

        # Old vs new path agree to machine precision.
        diff = (y_old - y_new).abs().max().item()
        rel_to_old = float(diff / (y_old.norm().item() + 1e-30))

        # Both vs a global reference: convert global A to dense and
        # check the owned-row slice.
        A_dense = A_global.to_dense()
        y_global_ref = A_dense @ x_global
        y_ref_owned = y_global_ref[partition.owned_nodes]
        rel_new_vs_ref = float(
            (y_new - y_ref_owned).norm().item()
            / (y_ref_owned.norm().item() + 1e-30))

        out_queue.put({
            "rank": rank,
            "diff_old_vs_new": diff,
            "rel_old_vs_new": rel_to_old,
            "rel_new_vs_ref": rel_new_vs_ref,
            "num_owned": int(partition.owned_nodes.numel()),
            "num_local": int(partition.local_to_global.numel()),
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_sparse_tensor_backed_matvec_matches_dsparse_matrix():
    """SparseTensor-backed matvec must agree with the legacy DSparseMatrix path to
    machine precision *and* with the global reference up to numerical
    error."""
    world_size = 2
    port = 29601
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_matvec_worker,
                            args=(rank, world_size, port, out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=120) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, \
                f"rank {procs.index(p)} exited with {p.exitcode}"
        for r in results:
            assert r["rel_old_vs_new"] < 1e-12, \
                f"rank {r['rank']}: SparseTensor-backed vs " \
                f"DSparseMatrix mismatch rel-diff {r['rel_old_vs_new']:.2e}"
            assert r["rel_new_vs_ref"] < 1e-10, \
                f"rank {r['rank']}: SparseTensor-backed vs global ref " \
                f"mismatch rel-err {r['rel_new_vs_ref']:.2e}"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    test_sparse_tensor_backed_matvec_matches_dsparse_matrix()
    print("OK: SparseTensor-backed matvec matches DSparseMatrix path "
          "and global reference")
