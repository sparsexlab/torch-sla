#!/usr/bin/env python
"""Distributed Hutchinson log-det -- no full_tensor() gather in the
inner loop, only ``_shard_matvec`` + a single ``all_gather`` per probe.
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _logdet_worker(rank: int, world: int, port: int, q: mp.Queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    try:
        from torch_sla import SparseTensor, DetConfig
        from torch_sla.datasets import Synthetic

        bench = Synthetic["poisson_2d_16"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        D = A.partition_for_rank(rank=rank, world_size=world)

        with DetConfig(method="hutchinson", num_probes=40, lanczos_iter=40):
            ld = float(D.logdet())

        # Single-process reference
        with DetConfig(method="hutchinson", num_probes=40, lanczos_iter=40):
            ld_ref = float(A.logdet())
        q.put({"rank": rank, "ld": ld, "ld_ref": ld_ref})
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(),
                    reason="torch.distributed not available")
def test_dsparse_logdet_hutchinson_2procs():
    world_size = 2
    port = 29730
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_logdet_worker,
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

    # All ranks should converge to the same estimate (same seed).
    # Hutchinson noise: expect ~few % spread across probes; we use 40 so
    # rel-err vs the single-process estimate should be small.
    for r in results:
        rel = abs(r["ld"] - r["ld_ref"]) / max(1e-12, abs(r["ld_ref"]))
        assert rel < 0.10, (
            f"rank {r['rank']}: D.logdet={r['ld']:.4f} ref={r['ld_ref']:.4f} "
            f"rel={rel:.2%}"
        )

    print(f"\n[OK] distributed Hutchinson logdet on 2 procs:")
    for r in sorted(results, key=lambda x: x["rank"]):
        print(f"  rank {r['rank']}: D.logdet={r['ld']:.4f} ref={r['ld_ref']:.4f}")


def test_sparse_tensor_logdet_modes():
    """Smoke test the dispatcher modes on a single-process SPD matrix."""
    from torch_sla import SparseTensor, DetConfig

    A = SparseTensor.tridiagonal(64, 4.0, -1.0)
    ld_ref = math.log(abs(float(A.det())))

    with DetConfig(method="hutchinson", num_probes=80, lanczos_iter=40):
        ld_h = float(A.logdet())
    with DetConfig(method="lu"):
        ld_lu = float(A.logdet())

    assert abs(ld_lu - ld_ref) < 1e-9
    assert abs(ld_h - ld_ref) / abs(ld_ref) < 0.05


def test_det_block_per_block():
    """``A.det()`` on a block-sparse SparseTensor returns one scalar
    per stored block. Pattern unchanged, ``block_shape`` collapsed."""
    from torch_sla import SparseTensor

    nnz, K = 3, 3
    torch.manual_seed(0)
    block_vals = torch.randn(nnz, K, K, dtype=torch.float64) + 5 * torch.eye(K)
    row = torch.tensor([0, 0, 1])
    col = torch.tensor([0, 1, 1])
    A = SparseTensor(block_vals, row, col, shape=(2, 2, K, K), sparse_dim=(0, 1))
    assert A.block_shape == (K, K)

    D = A.det()
    assert D.shape == (2, 2)
    assert D.sparse_shape == (2, 2)
    assert D.block_shape == ()
    assert D.values.shape == (nnz,)

    ref = torch.stack([torch.linalg.det(b) for b in block_vals])
    assert torch.allclose(D.values, ref)


def test_det_block_batched():
    """Batched block-sparse: ``[B, M_blk, N_blk, K, K]`` -> det values
    have shape ``[B, nnz]``."""
    from torch_sla import SparseTensor

    nnz, K, B = 3, 3, 2
    torch.manual_seed(1)
    block_vals = torch.randn(B, nnz, K, K, dtype=torch.float64) + 5 * torch.eye(K)
    row = torch.tensor([0, 0, 1])
    col = torch.tensor([0, 1, 1])
    A = SparseTensor(block_vals, row, col,
                     shape=(B, 2, 2, K, K), sparse_dim=(1, 2))
    D = A.det()
    assert D.batch_shape == (B,)
    assert D.sparse_shape == (2, 2)
    assert D.values.shape == (B, nnz)
    for b in range(B):
        ref = torch.stack([torch.linalg.det(blk) for blk in block_vals[b]])
        assert torch.allclose(D.values[b], ref)


def test_det_components_disconnected():
    """Block-diagonal sparse matrix -> det = product of block dets."""
    from torch_sla import SparseTensor, DetConfig

    # Two disconnected 4x4 tridiagonal blocks.
    A1 = SparseTensor.tridiagonal(4, 4.0, -1.0)
    A2 = SparseTensor.tridiagonal(4, 5.0, -2.0)
    # Build the block-diagonal SparseTensor by hand.
    val = torch.cat([A1.values, A2.values])
    row = torch.cat([A1.row_indices, A2.row_indices + 4])
    col = torch.cat([A1.col_indices, A2.col_indices + 4])
    A = SparseTensor(val, row, col, shape=(8, 8))

    with DetConfig(method="components"):
        d = float(A.det())
    expected = float(A1.det()) * float(A2.det())
    assert abs(d - expected) / abs(expected) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
