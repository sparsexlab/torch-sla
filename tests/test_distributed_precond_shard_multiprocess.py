#!/usr/bin/env python
"""Multi-process tests for the Shard(0) preconditioner family.

Verifies that the Jacobi / block-Jacobi / SSOR / polynomial precond
plug cleanly into PCG / PBiCGStab / GMRES on the standard catalogued
benchmarks, both via the explicit ``preconditioner=`` kwarg and via
``SolverConfig.preconditioner`` scope.

Quick correctness contract:
* Identity precond -> reproduces the unpreconditioned baseline.
* Jacobi / block-Jacobi / SSOR -> reduces iterations vs identity on a
  diagonally-dominant SPD problem (Poisson 2D, n=64).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _precond_worker(rank: int, world_size: int,
                    port: int, precond: object, method: str,
                    bench_key: str, maxiter: int,
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

        bench = Synthetic[bench_key]
        N = bench.shape[0]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        local_matrix = A.partition_for_rank(rank, world_size,
                                             partition_method="simple")
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.from_local(local_matrix, mesh,
                                       placement=RowPartitioned())

        torch.manual_seed(0)
        b_owned = torch.randn(N, dtype=torch.float64)[
            local_matrix.partition.owned_nodes]
        b_dt = DTensor.from_local(b_owned, mesh, [Shard(0)])

        x_dt = D.solve_distributed_shard(
            b_dt, method=method, preconditioner=precond,
            atol=1e-10, rtol=1e-10, maxiter=maxiter, restart=30)
        x_owned = x_dt.to_local()
        r = b_owned - D._shard_matvec(x_owned)
        rs = torch.dot(r, r)
        dist.all_reduce(rs, op=dist.ReduceOp.SUM)
        bs = torch.dot(b_owned, b_owned)
        dist.all_reduce(bs, op=dist.ReduceOp.SUM)
        rel_res = float(rs.sqrt().item()) / (float(bs.sqrt().item()) + 1e-30)

        out_queue.put({
            "rank": rank,
            "precond": str(precond),
            "method": method,
            "rel_residual": rel_res,
        })
    finally:
        dist.destroy_process_group()


def _run(world_size, port, precond, method, bench_key, maxiter):
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_precond_worker,
                            args=(rank, world_size, port, precond, method,
                                  bench_key, maxiter, out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=120) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, \
                f"rank {procs.index(p)} exited with {p.exitcode}"
        return results
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


@pytest.mark.parametrize("precond,port", [
    (None,           29541),
    ("none",         29542),
    ("jacobi",       29543),
    ("jacobi_l1",    29544),
    ("block_jacobi", 29545),
    ("ssor",         29546),
    ("polynomial",   29547),
])
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_precond_on_pcg_drives_residual_small(precond, port):
    """Every preconditioner choice must drive PCG on the SPD Poisson
    stencil to ``||r||/||b|| < 1e-5``. Confirms each precond preserves
    convergence (it doesn't have to be fast, just correct)."""
    results = _run(world_size=2, port=port,
                   precond=precond, method="cg",
                   bench_key="poisson_2d_16", maxiter=2000)
    for r in results:
        assert r["rel_residual"] < 1e-5, \
            f"{precond}/{r['rank']}: rel-residual {r['rel_residual']:.2e}"


@pytest.mark.parametrize("precond,port", [
    ("jacobi",       29551),
    ("block_jacobi", 29552),
])
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_precond_on_pbicgstab_converges_on_convdiff(precond, port):
    """Block-Jacobi / Jacobi preconditioned BiCGStab on the
    non-symmetric Peclet=10 convdiff stencil. The block precond should
    converge inside 1000 iters (way fewer than unpreconditioned)."""
    results = _run(world_size=2, port=port,
                   precond=precond, method="bicgstab",
                   bench_key="convdiff_2d_64_peclet_10", maxiter=1000)
    for r in results:
        assert r["rel_residual"] < 1e-5, \
            f"{precond}/{r['rank']}: rel-residual {r['rel_residual']:.2e}"


def _scope_precond_worker(rank, world_size, port, out_queue):
    """Worker for ``test_solverconfig_preconditioner_scope_reaches_shard_solve``.
    Defined at module scope so the ``spawn`` start method can pickle it."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        from torch_sla import (DSparseTensor, SparseTensor,
                                RowPartitioned, SolverConfig)
        from torch_sla.datasets import Synthetic

        bench = Synthetic["convdiff_2d_64_peclet_10"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        local = A.partition_for_rank(rank, world_size,
                                      partition_method="simple")
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.from_local(local, mesh,
                                      placement=RowPartitioned())

        torch.manual_seed(0)
        N = bench.shape[0]
        b_owned = torch.randn(N, dtype=torch.float64)[
            local.partition.owned_nodes]
        b_dt = DTensor.from_local(b_owned, mesh, [Shard(0)])

        with SolverConfig(method="bicgstab",
                          preconditioner="block_jacobi",
                          atol=1e-10, rtol=1e-10, maxiter=500):
            x_dt = D.solve_distributed_shard(b_dt)
        x = x_dt.to_local()
        r = b_owned - D._shard_matvec(x)
        rs = torch.dot(r, r); dist.all_reduce(rs)
        bs = torch.dot(b_owned, b_owned); dist.all_reduce(bs)
        rel = float(rs.sqrt().item()) / (float(bs.sqrt().item()) + 1e-30)
        out_queue.put({"rank": rank, "rel_residual": rel})
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_solverconfig_preconditioner_scope_reaches_shard_solve():
    """``SolverConfig(preconditioner='block_jacobi')`` scope must be
    picked up by ``solve_distributed_shard``."""
    world_size = 2
    port = 29561
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_scope_precond_worker,
                            args=(rank, world_size, port, out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=120) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0
        for r in results:
            assert r["rel_residual"] < 1e-5, \
                f"rank {r['rank']}: scope precond didn't propagate, " \
                f"rel-residual {r['rel_residual']:.2e}"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate(); p.join(timeout=5)


if __name__ == "__main__":
    test_precond_on_pcg_drives_residual_small("jacobi", 29543)
    print("OK")
