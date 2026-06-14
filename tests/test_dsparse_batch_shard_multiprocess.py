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


def _full_op_worker(rank: int, world: int, port: int, q: mp.Queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from torch_sla import SparseTensor, DSparseTensor

        B, n = 4, 12
        A = _build_batched_A(B, n)
        D = DSparseTensor.partition_batch(A, mesh=None, axis=0)

        # full_tensor allgather
        full = D.full_tensor()
        full_match = bool(torch.allclose(full.values, A.values))

        # Cross-rank reductions
        sum_d = float(D.sum().item())
        sum_ref = float(A.sum().item())

        mean_d = float(D.mean().item())
        mean_ref = float(A.mean().item())

        max_d = float(D.max().item())
        max_ref = float(A.max().item())

        norm_d = float(D.norm("fro").item())
        norm_ref = float((A.norm("fro") ** 2).sum().sqrt().item())

        # Per-batch eigsh on local slice
        evals_local, _ = D.eigsh(k=3, which="LM")
        evals_ref, _ = A.eigsh(k=3, which="LM")
        evals_local_first = evals_local[0].tolist()
        evals_ref_first = evals_ref[D._spec.placement.start].tolist()

        # solve_batch_shard parity vs single-process solve_batch
        torch.manual_seed(0)
        b = torch.randn(B, n, dtype=torch.float64)
        x_local = D.solve_batch_shard(b)
        # Reference: use SparseTensor.solve_batch on the full A
        from torch_sla import SparseTensor as _ST
        ref_template = _ST(A.values[0], A.row_indices, A.col_indices, (n, n))
        x_ref = ref_template.solve_batch(A.values, b)
        local_slice = x_ref.narrow(0, D._spec.placement.start,
                                    D._spec.placement.end - D._spec.placement.start)
        solve_err = float((x_local - local_slice).abs().max().item())

        q.put({
            "rank": rank,
            "full_match": full_match,
            "sum_diff": abs(sum_d - sum_ref),
            "mean_diff": abs(mean_d - mean_ref),
            "max_diff": abs(max_d - max_ref),
            "norm_diff": abs(norm_d - norm_ref),
            "evals_local_first": evals_local_first,
            "evals_ref_first": evals_ref_first,
            "solve_err": solve_err,
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(),
                    reason="torch.distributed not available")
def test_batch_shard_full_op_2procs():
    """End-to-end on 2 procs: full_tensor + reductions + eigsh + solve_batch."""
    world_size = 2
    port = 29712
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_full_op_worker,
                         args=(rank, world_size, port, q))
             for rank in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world_size

    for r in results:
        assert r["full_match"]
        assert r["sum_diff"] < 1e-9
        assert r["mean_diff"] < 1e-9
        assert r["max_diff"] < 1e-9
        assert r["norm_diff"] < 1e-9
        for got, ref in zip(r["evals_local_first"], r["evals_ref_first"]):
            assert abs(got - ref) < 1e-6
        assert r["solve_err"] < 1e-9, \
            f"rank {r['rank']}: solve mismatch {r['solve_err']:.2e}"

    print(f"\n[OK] batch shard end-to-end on 2 procs:")
    for r in sorted(results, key=lambda x: x["rank"]):
        print(f"  rank {r['rank']}: solve_err={r['solve_err']:.2e} "
              f"norm_diff={r['norm_diff']:.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
