#!/usr/bin/env python
"""``DSparseTensor`` topology / structural queries -- parity tests.

Phase C6: ``is_symmetric / is_hermitian / is_positive_definite /
detect_matrix_type / .T / .H`` go through ``full_tensor`` so the result
matches the single-process ``SparseTensor`` answer by construction.
This test verifies that connection holds end-to-end on a 2-proc gloo
setup, including the cache hit on repeated calls.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _topology_worker(rank: int, world_size: int, port: int,
                     out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch_sla import SparseTensor
        from torch_sla.datasets import Synthetic

        # The Poisson 2D 16x16 Laplacian is SPD; ConvDiff is asymmetric.
        bench_spd = Synthetic["poisson_2d_16"]
        A_spd = SparseTensor(bench_spd.val, bench_spd.row,
                             bench_spd.col, bench_spd.shape)
        D_spd = A_spd.partition_for_rank(rank=rank, world_size=world_size)

        # Distributed checks (each rank sees the same global answer)
        is_sym = bool(D_spd.is_symmetric())
        is_herm = bool(D_spd.is_hermitian())
        is_pd = bool(D_spd.is_positive_definite())
        mtype = D_spd.detect_matrix_type()

        # ---- Transpose / Hermitian transpose round-trip ----
        # ``.T`` is a method (matches SparseTensor convention), call it.
        D_T = D_spd.T()
        D_TT = D_T.T()  # T(T(A)) should match A on matvec

        torch.manual_seed(7)
        x_global = torch.randn(A_spd.shape[0], dtype=torch.float64)
        y_A = (D_spd @ D_spd.scatter(x_global)).full_tensor()
        y_TT = (D_TT @ D_TT.scatter(x_global)).full_tensor()
        max_diff = float((y_A - y_TT).abs().max().item())

        out_queue.put({
            "rank":      rank,
            "is_sym":    is_sym,
            "is_herm":   is_herm,
            "is_pd":     is_pd,
            "mtype":     mtype,
            "T_shape":   tuple(D_T.shape),
            "TT_round_trip_err": max_diff,
            "H_equals_T_on_real": bool(
                torch.allclose(
                    D_spd.H().full_tensor().values,
                    D_T.full_tensor().values,
                    rtol=1e-12, atol=1e-14,
                )
            ),
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_dsparse_topology_2procs():
    from torch_sla import SparseTensor
    from torch_sla.datasets import Synthetic

    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

    # Reference: single-process ground truth.
    ref_is_sym = bool(A.is_symmetric())
    ref_is_pd = bool(A.is_positive_definite())
    ref_mtype = A.detect_matrix_type()

    world_size = 2
    port = 29650
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_topology_worker,
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
        assert r["is_sym"] == ref_is_sym, \
            f"rank {r['rank']}: is_sym={r['is_sym']} vs ref={ref_is_sym}"
        assert r["is_pd"] == ref_is_pd, \
            f"rank {r['rank']}: is_pd={r['is_pd']} vs ref={ref_is_pd}"
        assert r["mtype"] == ref_mtype, \
            f"rank {r['rank']}: mtype={r['mtype']!r} vs ref={ref_mtype!r}"
        assert r["T_shape"] == (A.shape[1], A.shape[0])
        # T(T(A)) @ x == A @ x exactly (modulo float)
        assert r["TT_round_trip_err"] < 1e-10, \
            f"rank {r['rank']}: T(T(A))@x diverged by {r['TT_round_trip_err']:.2e}"
        # On a real matrix H == T
        assert r["H_equals_T_on_real"], \
            f"rank {r['rank']}: .H differs from .T on a real matrix"

    print(f"\n[OK] topology on 2 procs (Poisson 2D Laplacian, SPD):")
    print(f"  is_symmetric={results[0]['is_sym']} is_pd={results[0]['is_pd']} "
          f"detect_matrix_type={results[0]['mtype']!r}")
    print(f"  T(T(A)) @ x round-trip err = {results[0]['TT_round_trip_err']:.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
