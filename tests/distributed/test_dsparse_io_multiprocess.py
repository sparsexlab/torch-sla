#!/usr/bin/env python
"""Persistence round-trip tests for ``DSparseTensor``.

Phase C3: covers four modes of the new I/O suite:

1. ``DSparseTensor.save`` (per-rank) → ``DSparseTensor.load`` (per-rank).
2. ``SparseTensor.save_distributed`` (single-process partition+write) →
   ``DSparseTensor.load`` (per-rank read).
3. ``load_metadata`` returns the expected dict.
4. ``load_sparse_shard`` (inspection) on a saved shard.

Each rank verifies that ``D_loaded @ x_dt`` == ``D @ x_dt`` modulo
floating-point round-off.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _save_then_load_worker(rank: int, world_size: int, port: int,
                           shared_dir: str,
                           out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch_sla import DSparseTensor, SparseTensor, load_metadata
        from torch_sla.datasets import Synthetic

        bench = Synthetic["poisson_2d_16"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A, mesh, partition_method="simple")

        torch.manual_seed(0)
        x_global = torch.randn(A.shape[0], dtype=torch.float64)
        x_dt = D.scatter(x_global)
        y_ref = (D @ x_dt).full_tensor()

        # ---- save this rank's shard ----
        D.save(shared_dir)
        dist.barrier()  # ensure all writes complete before any load

        if rank == 0:
            meta = load_metadata(shared_dir)
            assert meta["format"] == "dsparse_tensor"
            assert meta["num_partitions"] == world_size
            assert tuple(meta["shape"]) == tuple(A.shape)

        # ---- load this rank's shard back ----
        D2 = DSparseTensor.load(shared_dir, mesh=mesh)
        y_loaded = (D2 @ D2.scatter(x_global)).full_tensor()
        err = float((y_loaded - y_ref).abs().max().item())

        # ---- topology checks on the reloaded shard ----
        assert D2.shape == D.shape, f"shape drift: {D2.shape} vs {D.shape}"
        assert D2.nnz == D.nnz, f"nnz drift: {D2.nnz} vs {D.nnz}"
        assert D2.global_nnz() == D.global_nnz()

        out_queue.put({
            "rank": rank,
            "matvec_err": err,
            "shape": tuple(D2.shape),
            "nnz_match": (D2.nnz == D.nnz),
        })
    finally:
        dist.destroy_process_group()


def _load_pre_sharded_worker(rank: int, world_size: int, port: int,
                             shared_dir: str,
                             out_queue: mp.Queue) -> None:
    """Load a directory that was written by a single-process
    ``save_sparse_sharded`` -- verify multi-rank can read it fine."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch_sla import DSparseTensor, SparseTensor
        from torch_sla.datasets import Synthetic

        bench = Synthetic["poisson_2d_16"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        mesh = init_device_mesh("cpu", (world_size,))

        # Reference: partition in-memory.
        D_ref = DSparseTensor.partition(A, mesh, partition_method="simple")

        torch.manual_seed(0)
        x_global = torch.randn(A.shape[0], dtype=torch.float64)
        y_ref = (D_ref @ D_ref.scatter(x_global)).full_tensor()

        # Load from the pre-sharded directory.
        D2 = DSparseTensor.load(shared_dir, mesh=mesh)
        y_loaded = (D2 @ D2.scatter(x_global)).full_tensor()
        err = float((y_loaded - y_ref).abs().max().item())

        out_queue.put({"rank": rank, "matvec_err": err})
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_dsparse_save_then_load_2procs():
    """Each rank writes its own shard; each rank reads its own shard."""
    world_size = 2
    port = 29580
    with tempfile.TemporaryDirectory() as td:
        ctx = mp.get_context("spawn")
        q: mp.Queue = ctx.Queue()
        procs = [ctx.Process(target=_save_then_load_worker,
                             args=(rank, world_size, port, td, q))
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
            assert r["matvec_err"] == 0.0, \
                f"rank {r['rank']}: round-trip introduced matvec error {r['matvec_err']:.2e}"
            assert r["nnz_match"]

    print(f"\n[OK] save+load round-trip on {world_size} procs: zero matvec drift")


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_load_presharded_2procs():
    """``save_sparse_sharded`` single-process produces a directory that
    ``load_dsparse`` can consume from each rank."""
    world_size = 2
    port = 29582
    with tempfile.TemporaryDirectory() as td:
        # ---- single-process partition + write ----
        from torch_sla import SparseTensor
        from torch_sla.datasets import Synthetic
        bench = Synthetic["poisson_2d_16"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        A.save_distributed(td, num_partitions=world_size,
                           partition_method="simple")

        # ---- multi-process load + verify ----
        ctx = mp.get_context("spawn")
        q: mp.Queue = ctx.Queue()
        procs = [ctx.Process(target=_load_pre_sharded_worker,
                             args=(rank, world_size, port, td, q))
                 for rank in range(world_size)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)

        results = []
        while not q.empty():
            results.append(q.get())
        assert len(results) == world_size

        for r in results:
            assert r["matvec_err"] == 0.0, \
                f"rank {r['rank']}: presharded load matvec error {r['matvec_err']:.2e}"

    print(f"\n[OK] presharded load on {world_size} procs: zero matvec drift")


def test_single_process_save_and_inspect():
    """Single-process: ``save_sparse_sharded`` + ``load_sparse_shard``
    inspection round-trip."""
    from torch_sla import (
        SparseTensor, save_sparse_sharded, load_sparse_shard, load_metadata,
    )
    from torch_sla.datasets import Synthetic
    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

    with tempfile.TemporaryDirectory() as td:
        save_sparse_sharded(A, td, num_partitions=4,
                            partition_method="simple")
        meta = load_metadata(td)
        assert meta["num_partitions"] == 4
        assert meta["format"] == "dsparse_tensor"
        assert tuple(meta["shape"]) == tuple(A.shape)
        assert len(meta["partitions"]) == 4

        # Sum of owned across all shards should cover every node once.
        owned_total = 0
        for r in range(4):
            local_st, partition = load_sparse_shard(td, rank=r)
            owned_total += int(partition.owned_nodes.numel())
            assert int(partition.partition_id) == r
        assert owned_total == int(A.shape[0]), \
            f"owned total {owned_total} != global rows {A.shape[0]}"
    print("\n[OK] single-process inspect: 4 shards, all rows accounted for")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
