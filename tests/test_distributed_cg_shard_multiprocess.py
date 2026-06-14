#!/usr/bin/env python
"""End-to-end distributed CG via the unified ``solve`` API.

The worker body reads like example user code: build the matrix, build
the distributed tensor, build the RHS via :meth:`DSparseTensor.scatter`,
hand both to :func:`solve`, and verify with public ops only. No
``_shard_matvec``, no raw ``dist.all_reduce``, no partition-internal
indexing -- if a user has to write that, the API failed.

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

        from torch_sla import DSparseTensor, SparseTensor, solve, SolverConfig
        from torch_sla.datasets import Synthetic

        # ---- user-side setup ---------------------------------------- #
        bench = Synthetic["poisson_2d_16"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A, mesh, partition_method="simple")

        torch.manual_seed(0)
        b_global = torch.randn(A.shape[0], dtype=torch.float64)
        b_dt = D.scatter(b_global)

        # ---- distributed solve -------------------------------------- #
        with SolverConfig(method="cg", atol=1e-12, rtol=1e-10,
                          maxiter=2000):
            x_dt = solve(D, b_dt)

        # ---- verification using public ops only --------------------- #
        # Distributed residual: ``||b - A x|| / ||b||``. ``D @ x_dt`` is
        # the public distributed matvec; the subtraction stays in
        # Shard(0) space; ``.full_tensor()`` allgathers the result so we
        # can take a global norm without touching ``dist`` directly.
        r_dt = b_dt - D @ x_dt
        rel_residual = float(
            (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item())

        # Sanity overlay: a SciPy CG single-process run on the same
        # system should land near the same solution.
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        A_sp = sp.coo_matrix(
            (bench.val.numpy(), (bench.row.numpy(), bench.col.numpy())),
            shape=bench.shape).tocsr()
        x_ref_np, _ = spla.cg(A_sp, b_global.numpy(),
                               rtol=1e-12, atol=1e-12, maxiter=2000)
        x_full = x_dt.full_tensor()
        x_ref = torch.from_numpy(x_ref_np).to(x_full.dtype)
        rel_to_scipy = float(
            ((x_full - x_ref).norm() / (x_ref.norm() + 1e-12)).item())

        out_queue.put({
            "rank": rank,
            "rel_residual": rel_residual,
            "rel_to_scipy": rel_to_scipy,
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

        for r in results:
            assert r["rel_residual"] < 1e-8, \
                f"rank {r['rank']}: ||b - A x|| / ||b|| = " \
                f"{r['rel_residual']:.2e}"
            assert r["rel_to_scipy"] < 1e-6, \
                f"rank {r['rank']}: distributed CG vs SciPy CG mismatch " \
                f"rel-err {r['rel_to_scipy']:.2e}"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    test_shard_space_cg_matches_scipy_cg()
    print("OK: shard-space CG matches scipy single-process CG on every rank")
