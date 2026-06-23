#!/usr/bin/env python
"""``DSparseTensor`` reductions parity tests vs single-process.

Phase C4: ``sum / mean / prod / max / min / norm`` reduce cross-rank
via ``all_reduce``. Verified by comparing the distributed result to
``A.sum() / A.norm() / ...`` on the same global ``SparseTensor`` in
single process.
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


def _reduction_worker(rank: int, world_size: int, port: int,
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
        D = A.partition_for_rank(rank=rank, world_size=world_size)

        # Distributed reductions (each rank gets the global answer).
        d_sum = float(D.sum().item())
        d_mean = float(D.mean().item())
        d_prod = float(D.prod().item())
        d_max = float(D.max().item())
        d_min = float(D.min().item())
        d_fro = float(D.norm("fro").item())
        d_l1 = float(D.norm(1).item())
        d_linf = float(D.norm(float("inf")).item())

        # Vector reductions: axis=0 (col sums), axis=1 (row sums)
        col_sums = D.sum(axis=0)
        row_sums = D.sum(axis=1)
        col_means = D.mean(axis=0)
        row_means = D.mean(axis=1)

        out_queue.put({
            "rank": rank,
            "sum": d_sum,
            "mean": d_mean,
            "prod": d_prod,
            "max": d_max,
            "min": d_min,
            "fro": d_fro,
            "l1": d_l1,
            "linf": d_linf,
            "col_sums_first10": col_sums[:10].tolist(),
            "row_sums_first10": row_sums[:10].tolist(),
            "col_means_first10": col_means[:10].tolist(),
            "row_means_first10": row_means[:10].tolist(),
            "col_sums_shape": tuple(col_sums.shape),
            "row_sums_shape": tuple(row_sums.shape),
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_dsparse_reductions_2procs():
    """Cross-rank reductions match single-process ground truth."""
    from torch_sla import SparseTensor
    from torch_sla.datasets import Synthetic

    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

    # Reference: single-process reductions on the global SparseTensor.
    ref = {
        "sum":  float(A.sum().item()),
        "mean": float(A.mean().item()),
        "prod": float(A.prod().item()),
        "max":  float(A.max().item()),
        "min":  float(A.min().item()),
        "fro":  float(A.norm("fro").item()),
    }
    # SparseTensor.sum(axis=...) returns dense vectors.
    ref_col_sums = A.sum(axis=0)
    ref_row_sums = A.sum(axis=1)

    world_size = 2
    port = 29610
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_reduction_worker,
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
        for key in ["sum", "max", "min", "fro"]:
            assert math.isclose(r[key], ref[key], rel_tol=1e-12), \
                f"rank {r['rank']}: {key} {r[key]} vs ref {ref[key]}"
        # prod / mean depend on global_nnz semantics matching
        # SparseTensor's (mean = sum / nnz). Verify by reconstructing.
        assert math.isclose(r["mean"], ref["mean"], rel_tol=1e-10), \
            f"rank {r['rank']}: mean {r['mean']} vs ref {ref['mean']}"
        assert math.isclose(r["prod"], ref["prod"], rel_tol=1e-10) or \
               (math.isnan(r["prod"]) and math.isnan(ref["prod"])) or \
               (r["prod"] == 0.0 and ref["prod"] == 0.0), \
            f"rank {r['rank']}: prod {r['prod']} vs ref {ref['prod']}"
        # vector reductions
        assert tuple(r["col_sums_shape"]) == (A.shape[1],)
        assert tuple(r["row_sums_shape"]) == (A.shape[0],)
        d_col = torch.tensor(r["col_sums_first10"], dtype=torch.float64)
        d_row = torch.tensor(r["row_sums_first10"], dtype=torch.float64)
        assert torch.allclose(d_col, ref_col_sums[:10], rtol=1e-12, atol=1e-14), \
            f"rank {r['rank']}: col sums diverge"
        assert torch.allclose(d_row, ref_row_sums[:10], rtol=1e-12, atol=1e-14), \
            f"rank {r['rank']}: row sums diverge"

        # ||·||_1 = max column abs-sum on the GLOBAL matrix. Reference:
        # build dense |A|, sum cols, take max.
        from torch import sparse
        ref_l1 = float(A.values.new_zeros(A.shape[1]).scatter_add_(
            0, A.col_indices, A.values.abs()).max().item())
        ref_linf = float(A.values.new_zeros(A.shape[0]).scatter_add_(
            0, A.row_indices, A.values.abs()).max().item())
        assert math.isclose(r["l1"], ref_l1, rel_tol=1e-12), \
            f"rank {r['rank']}: ||A||_1 {r['l1']} vs ref {ref_l1}"
        assert math.isclose(r["linf"], ref_linf, rel_tol=1e-12), \
            f"rank {r['rank']}: ||A||_inf {r['linf']} vs ref {ref_linf}"

    print(f"\n[OK] reductions on 2 procs vs single-process reference:")
    r0 = results[0]
    print(f"  sum={r0['sum']:.6e} mean={r0['mean']:.6e} max={r0['max']:.6e} "
          f"min={r0['min']:.6e}")
    print(f"  fro={r0['fro']:.6e} L1={r0['l1']:.6e} Linf={r0['linf']:.6e}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
