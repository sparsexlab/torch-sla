#!/usr/bin/env python
"""End-to-end preconditioner family via the unified ``solve`` API.

Worker bodies read like example user code: ``A`` →
``DSparseTensor.partition`` → ``D.scatter(b_global)`` → ``solve(D, b)``
under a ``SolverConfig(preconditioner=...)`` scope → residual via
``D @ x_dt``. No ``_shard_matvec`` / no raw ``dist.all_reduce`` /
no partition-internal indexing.

Quick correctness contract:

* Every preconditioner (None / 'none' / 'jacobi' / 'jacobi_l1' /
  'block_jacobi' / 'ssor' / 'polynomial') must keep PCG on the
  SPD Poisson stencil converging.
* Block-Jacobi / Jacobi must make BiCGStab converge on the
  non-symmetric convdiff stencil.
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

        from torch_sla import (DSparseTensor, SparseTensor, solve,
                                 SolverConfig)
        from torch_sla.datasets import Synthetic

        bench = Synthetic[bench_key]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A, mesh, partition_method="simple")

        torch.manual_seed(0)
        b_global = torch.randn(A.shape[0], dtype=torch.float64)
        b_dt = D.scatter(b_global)

        with SolverConfig(method=method, preconditioner=precond,
                          atol=1e-10, rtol=1e-10,
                          maxiter=maxiter, restart=30):
            x_dt = solve(D, b_dt)

        r_dt = b_dt - D @ x_dt
        rel_res = float(
            (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item())

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
    Defined at module scope so ``spawn`` can pickle it."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from torch_sla import (DSparseTensor, SparseTensor,
                                solve, SolverConfig)
        from torch_sla.datasets import Synthetic

        bench = Synthetic["convdiff_2d_64_peclet_10"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A, mesh, partition_method="simple")

        torch.manual_seed(0)
        b_global = torch.randn(A.shape[0], dtype=torch.float64)
        b_dt = D.scatter(b_global)

        with SolverConfig(method="bicgstab",
                          preconditioner="block_jacobi",
                          atol=1e-10, rtol=1e-10, maxiter=500):
            x_dt = solve(D, b_dt)

        r_dt = b_dt - D @ x_dt
        rel = float(
            (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item())
        out_queue.put({"rank": rank, "rel_residual": rel})
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_solverconfig_preconditioner_scope_reaches_shard_solve():
    """``SolverConfig(preconditioner='block_jacobi')`` scope must be
    picked up by the unified ``solve(D, b)`` entry."""
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
