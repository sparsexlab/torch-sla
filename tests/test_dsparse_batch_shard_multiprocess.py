#!/usr/bin/env python
"""``BatchShard`` placement -- partition + matvec parity vs single proc.

Phase B1: batched ``SparseTensor`` is value-sharded along a batch axis;
indices replicated; matvec runs locally on each rank with zero
inter-rank communication. After ``full_tensor()`` allgather, the result
must equal the single-process batched matvec to machine precision.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_batched_A(B: int = 4, n: int = 12) -> "SparseTensor":
    from torch_sla import SparseTensor

    idx = torch.arange(n)
    row = torch.cat([idx, idx[1:], idx[:-1]])
    col = torch.cat([idx, idx[:-1], idx[1:]])
    val = torch.zeros(B, len(row), dtype=torch.float64)
    for b in range(B):
        val[b] = torch.cat([
            torch.full((n,), 4.0 + b, dtype=torch.float64),
            torch.full((n - 1,), -1.0, dtype=torch.float64),
            torch.full((n - 1,), -1.0, dtype=torch.float64),
        ])
    return SparseTensor(val, row, col, shape=(B, n, n))


def _batch_matvec_worker(rank: int, world: int, port: int, q: mp.Queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from torch_sla import DSparseTensor

        B, n = 4, 12
        A = _build_batched_A(B, n)
        D = DSparseTensor.partition_batch(A, mesh=None, axis=0)

        torch.manual_seed(0)
        x_full = torch.randn(B, n, dtype=torch.float64)
        y_local = D @ x_full              # zero-comm, per-batch slice

        # Allgather the per-rank batch slices to reproduce the global y.
        local_size = y_local.shape[0]
        sizes = [torch.zeros(1, dtype=torch.long) for _ in range(world)]
        dist.all_gather(sizes, torch.tensor([local_size]))
        sizes = [int(s.item()) for s in sizes]

        max_size = max(sizes)
        pad = torch.zeros(max_size - local_size, n, dtype=torch.float64)
        y_padded = torch.cat([y_local, pad], dim=0)
        all_y = [torch.zeros_like(y_padded) for _ in range(world)]
        dist.all_gather(all_y, y_padded)
        y_global = torch.cat(
            [g[:sz] for g, sz in zip(all_y, sizes)], dim=0,
        )

        y_ref = A @ x_full
        err = float((y_global - y_ref).abs().max().item())
        q.put({"rank": rank, "err": err,
               "local_batch": (D._spec.placement.start,
                               D._spec.placement.end)})
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(),
                    reason="torch.distributed not available")
def test_batch_shard_matvec_2procs():
    world_size = 2
    port = 29710
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_batch_matvec_worker,
                         args=(rank, world_size, port, q))
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
        assert r["err"] == 0.0, \
            f"rank {r['rank']}: batch-shard matvec diverged ({r['err']:.2e})"

    print(f"\n[OK] batch shard matvec, 2 procs, 4 batches:")
    for r in sorted(results, key=lambda x: x["rank"]):
        print(f"  rank {r['rank']}: owns batch {r['local_batch']}  err={r['err']}")


def test_batch_shard_single_process():
    """Single-process partition_batch + matvec parity (world=1)."""
    from torch_sla import DSparseTensor

    A = _build_batched_A(B=4, n=12)
    D = DSparseTensor.partition_batch(A, mesh=None, axis=0)
    assert D._spec.placement.start == 0
    assert D._spec.placement.end == 4

    torch.manual_seed(0)
    x = torch.randn(4, 12, dtype=torch.float64)
    assert torch.allclose(D @ x, A @ x, atol=0.0)


def test_batch_shard_rejects_unbatched():
    from torch_sla import SparseTensor, DSparseTensor

    A = SparseTensor.tridiagonal(8, 4.0, -1.0)
    with pytest.raises(ValueError, match="batched"):
        DSparseTensor.partition_batch(A, mesh=None, axis=0)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
