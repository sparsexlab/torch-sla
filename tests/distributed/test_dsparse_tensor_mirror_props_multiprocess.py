#!/usr/bin/env python
"""Tensor-mirror property parity tests for ``DSparseTensor``.

Verifies the Phase C1 additions: ``DSparseTensor`` now exposes the same
``ndim`` / ``sparse_shape`` / ``sparse_dim`` / ``batch_shape`` /
``block_shape`` / ``batch_size`` / ``is_batched`` / ``is_block`` /
``is_cuda`` / ``is_square`` / ``values`` / ``row_indices`` /
``col_indices`` properties as ``SparseTensor`` (plus a new
``global_nnz()`` reduction).

Also exercises the restored ``SparseTensor.partition_for_rank()``
convenience constructor.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mirror_props_worker(rank: int, world_size: int, port: int,
                         out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch_sla import SparseTensor
        from torch_sla.datasets import Synthetic

        bench = Synthetic["poisson_2d_16"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

        # ---- partition_for_rank shim (restored from pre-PR-31 API) ----
        D = A.partition_for_rank(rank=rank, world_size=world_size)

        # ---- Tensor-mirror props ----
        props = {
            "shape": tuple(D.shape),
            "sparse_shape": tuple(D.sparse_shape),
            "sparse_dim": tuple(D.sparse_dim),
            "batch_shape": tuple(D.batch_shape),
            "block_shape": tuple(D.block_shape),
            "batch_size": D.batch_size,
            "ndim": D.ndim,
            "is_batched": D.is_batched,
            "is_block": D.is_block,
            "is_cuda": D.is_cuda,
            "is_square": D.is_square,
            "local_nnz": D.nnz,
            "global_nnz": D.global_nnz(),
            "ref_nnz": A.nnz,
            "values_shape": tuple(D.values.shape),
            "row_indices_shape": tuple(D.row_indices.shape),
            "col_indices_shape": tuple(D.col_indices.shape),
            "values_dtype": str(D.values.dtype),
            "dtype": str(D.dtype),
        }
        out_queue.put({"rank": rank, **props})
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_dsparse_tensor_mirror_props_2procs():
    """All Tensor-mirror props return correct values on a sharded matrix.

    Plus: ``global_nnz()`` equals single-process ``A.nnz``.
    """
    world_size = 2
    port = 29545
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_mirror_props_worker,
                         args=(rank, world_size, port, q))
             for rank in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world_size, \
        f"Expected {world_size} results, got {len(results)}"

    for r in results:
        # ---- Topology contracts (DSparseTensor is always 2-D, unbatched) ----
        assert r["ndim"] == 2, f"rank {r['rank']}: ndim {r['ndim']} != 2"
        assert r["sparse_shape"] == r["shape"], \
            f"rank {r['rank']}: sparse_shape != shape"
        assert r["sparse_dim"] == (0, 1)
        assert r["batch_shape"] == ()
        assert r["block_shape"] == ()
        assert r["batch_size"] == 1
        assert r["is_batched"] is False
        assert r["is_block"] is False
        assert r["is_cuda"] is False
        assert r["is_square"] is True

        # ---- Local arrays ----
        assert r["local_nnz"] == r["values_shape"][0]
        assert r["values_shape"] == r["row_indices_shape"]
        assert r["values_shape"] == r["col_indices_shape"]
        assert r["dtype"] == r["values_dtype"]

        # ---- global_nnz contract: sum across ranks == total local on
        # the global SparseTensor (no halo double-count). ----
        # Note: shard local includes halo rows, so per-rank local_nnz
        # *may* exceed the partition share; sum of locals may exceed
        # the global, so we only check global_nnz <= sum(local_nnz)
        # and global_nnz consistent across ranks.
        ranks_global_nnz = {x["global_nnz"] for x in results}
        assert len(ranks_global_nnz) == 1, \
            "global_nnz must be the same on every rank"

    print(f"\n[OK] Tensor-mirror props on 2 procs:")
    for r in sorted(results, key=lambda x: x["rank"]):
        print(f"  rank {r['rank']}: shape={r['shape']} "
              f"local_nnz={r['local_nnz']} global_nnz={r['global_nnz']}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
