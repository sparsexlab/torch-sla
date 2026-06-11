#!/usr/bin/env python
"""Multi-process test for ``DSparseTensor.solve_distributed_shard``.

Verifies that distributed CG running entirely in ``Shard(0)`` space --
no global vectors materialised on any rank, ``dist.all_reduce`` for
inner products, ``halo_exchange`` per matvec -- converges to the same
solution as a single-process SciPy CG solve on the global matrix.

Run with::

    python tests/test_distributed_cg_shard_multiprocess.py
    pytest tests/test_distributed_cg_shard_multiprocess.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _shard_cg_worker(rank: int, world_size: int,
                     port: int,
                     out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        from torch_sla import (DSparseTensor, SparseTensor, RowPartitioned)
        from torch_sla.datasets import Synthetic

        # Standard catalogued 2D Poisson stencil -- SPD, well-conditioned.
        bench = Synthetic["poisson_2d_16"]
        N = bench.shape[0]
        A_global = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

        local_matrix = A_global.partition_for_rank(
            rank, world_size, partition_method="simple")
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.from_local(
            local_matrix, mesh, placement=RowPartitioned())

        # Manufacture a known global RHS so every rank can derive its
        # owned slice without prior global gather.
        torch.manual_seed(0)
        b_global = torch.randn(N, dtype=torch.float64)
        b_owned = b_global[local_matrix.partition.owned_nodes]
        b_dt = DTensor.from_local(b_owned, mesh, [Shard(0)])

        # Shard-space CG.
        x_dt = D.solve_distributed_shard(
            b_dt, atol=1e-12, rtol=1e-10, maxiter=2000)
        x_owned = x_dt.to_local()

        # Reference: scipy CG on the global matrix.
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        A_sp = sp.coo_matrix(
            (bench.val.numpy(), (bench.row.numpy(), bench.col.numpy())),
            shape=bench.shape).tocsr()
        x_ref, _ = spla.cg(A_sp, b_global.numpy(),
                            rtol=1e-12, atol=1e-12, maxiter=2000)

        owned_np = local_matrix.partition.owned_nodes.cpu().numpy()
        x_ref_owned = torch.from_numpy(x_ref[owned_np])
        rel = (x_owned - x_ref_owned).norm() / (x_ref_owned.norm() + 1e-12)

        out_queue.put({
            "rank": rank,
            "x_owned_norm": float(x_owned.norm().item()),
            "rel_err_vs_scipy": float(rel.item()),
            "x_owned_size": int(x_owned.numel()),
            "num_owned": int(local_matrix.num_owned),
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_shard_space_cg_matches_scipy_cg():
    world_size = 2
    port = 29514

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_shard_cg_worker,
                            args=(rank, world_size, port, out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=120) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, \
                f"rank {procs.index(p)} exited with code {p.exitcode}"

        by_rank = {r["rank"]: r for r in results}
        assert set(by_rank) == set(range(world_size))

        # x slice on each rank has size == num_owned (Shard(0) shape).
        for r in results:
            assert r["x_owned_size"] == r["num_owned"], \
                f"rank {r['rank']}: x_owned size {r['x_owned_size']} != " \
                f"num_owned {r['num_owned']}"

        # The owned slice of the distributed CG solution matches the
        # SciPy single-process CG solution to ~1e-6 relative error.
        for r in results:
            assert r["rel_err_vs_scipy"] < 1e-6, \
                f"rank {r['rank']}: distributed CG vs SciPy CG mismatch " \
                f"rel-err {r['rel_err_vs_scipy']:.2e}"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    test_shard_space_cg_matches_scipy_cg()
    print("OK: shard-space CG matches scipy single-process CG on every rank")
