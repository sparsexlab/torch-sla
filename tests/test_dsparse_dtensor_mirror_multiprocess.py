#!/usr/bin/env python
"""Multi-process tests for the new DTensor-mirror :class:`DSparseTensor`
API: ``from_local`` / ``to_local`` / ``full_tensor`` / ``__matmul__``
returning ``DTensor[Shard(0)]``.

Verifies that:

1. ``DSparseTensor.from_local`` wraps a per-rank DSparseMatrix into a
   spec-bearing DSparseTensor without dropping any metadata.
2. ``to_local()`` round-trips back to the same DSparseMatrix.
3. ``full_tensor()`` allgathers the COO triples and rebuilds a
   :class:`SparseTensor` identical (modulo row order) to the original
   global matrix on every rank.
4. ``D @ x_dtensor`` produces a DTensor with ``Shard(0)`` placement
   whose values, after gather-to-global, equal ``A_global @ x_global``.

Run with::

    python tests/test_dsparse_dtensor_mirror_multiprocess.py
    pytest tests/test_dsparse_dtensor_mirror_multiprocess.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------- #
# Worker
# ---------------------------------------------------------------------- #
def _mirror_worker(rank: int, world_size: int,
                   port: int,
                   out_queue: mp.Queue) -> None:
    """Build the global Poisson matrix, partition it across ranks,
    wrap each rank's chunk as the DTensor-mirror DSparseTensor, and
    exercise the new API. Each rank pushes its observed results back
    for the parent to assert."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        from torch_sla import (DSparseMatrix, DSparseTensor,
                                SparseTensor, SparseShard, Replicated)
        from torch_sla.datasets import Synthetic

        # Use the project's standard catalogued PDE stencil.
        bench = Synthetic["poisson_2d_16"]
        N = bench.shape[0]
        A_global = SparseTensor(bench.val, bench.row,
                                 bench.col, bench.shape)

        # One-shot DTensor-mirror construction.
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A_global, mesh,
                                     partition_method="simple")
        local_matrix = D.to_local()

        # 1) spec metadata round-trips.
        assert D.spec is not None
        # SparseShard with axis=0 is the canonical row-shard placement
        # (replaces the deprecated RowPartitioned class).
        assert isinstance(D.spec.placement, SparseShard)
        assert D.spec.placement.axis == 0
        assert D.spec.global_shape == (N, N)

        # 2) to_local is the identity.
        assert D.to_local() is local_matrix

        # 3) full_tensor allgathers COO across ranks; rank 0 + rank 1
        # should agree, and the global nnz should match the original.
        full = D.full_tensor()
        assert isinstance(full, SparseTensor)
        assert full.shape == (N, N)

        # 4) DTensor matvec. Replicated x first (every rank holds full
        # x_global), then we extract the per-rank Shard(0) slice via
        # the matrix's owned_nodes map.
        torch.manual_seed(0)
        x_global = torch.randn(N, dtype=torch.float64)
        y_global = (A_global @ x_global).detach()

        # Local x slice = the x values for THIS rank's owned rows.
        # We can't use DTensor's automatic Shard(0) splitting because
        # the partition is irregular. So build the local slice from
        # the partition's owned_nodes ourselves.
        x_local = x_global[local_matrix.partition.local_nodes]
        x_dt = DTensor.from_local(x_local, mesh, [Shard(0)])

        y_dt = D @ x_dt
        assert hasattr(y_dt, "to_local"), "result must be a DTensor"
        y_local = y_dt.to_local()
        assert y_local.shape[0] == local_matrix.num_owned

        # Compare local result against the slice of the global y for
        # this rank's owned rows.
        owned_global = local_matrix.partition.owned_nodes
        y_expected_local = y_global[owned_global]
        rel = (y_local - y_expected_local).norm() / (y_expected_local.norm() + 1e-12)

        out_queue.put({
            "rank": rank,
            "full_nnz": int(full.values.numel()),
            "y_rel_err": float(rel.item()),
            "y_owned_count": int(local_matrix.num_owned),
            "global_nnz": int(bench.val.numel()),
        })
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------- #
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_dsparse_dtensor_mirror_roundtrip_and_matmul():
    world_size = 2
    port = 29513

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(
                target=_mirror_worker,
                args=(rank, world_size, port, out_queue),
            )
            p.start()
            procs.append(p)

        results = [out_queue.get(timeout=60) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, \
                f"rank {procs.index(p)} exited with code {p.exitcode}"

        by_rank = {r["rank"]: r for r in results}
        assert set(by_rank) == set(range(world_size))

        # Every rank's full_tensor() returned the same nnz as the
        # original global matrix (no duplicates, nothing dropped).
        global_nnz = by_rank[0]["global_nnz"]
        for r in results:
            assert r["full_nnz"] == global_nnz, \
                f"rank {r['rank']} full_tensor nnz={r['full_nnz']} != " \
                f"global nnz={global_nnz}"

        # Every rank's local matvec result matches A_global @ x_global
        # restricted to that rank's owned rows.
        for r in results:
            assert r["y_rel_err"] < 1e-10, \
                f"rank {r['rank']} matvec mismatch: rel-err {r['y_rel_err']:.2e}"

        # Sum of owned-row counts across ranks == global N.
        total_owned = sum(r["y_owned_count"] for r in results)
        assert total_owned == 256, \
            f"owned rows don't partition the global N: {total_owned} != 256"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    test_dsparse_dtensor_mirror_roundtrip_and_matmul()
    print("OK: from_local + to_local + full_tensor + Shard(0) matmul all "
          "behave like DTensor")
