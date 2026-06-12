#!/usr/bin/env python
"""End-to-end Krylov methods via the unified ``solve`` API.

Worker bodies read like example user code: ``A`` →
``DSparseTensor.partition`` → ``D.scatter(b_global)`` → ``solve(D, b)``
→ residual via ``D @ x_dt`` (all public ops, no ``_shard_matvec`` /
no raw ``dist.all_reduce`` for inner products).

Covered methods:

* BiCGStab + GMRES + FGMRES on the non-symmetric
  ``Synthetic["convdiff_2d_64_peclet_10"]`` benchmark -- exercises
  the indefinite path.
* MINRES on the SPD ``Synthetic["poisson_2d_16"]`` benchmark.

Run with::

    python tests/test_distributed_krylov_shard_multiprocess.py
    pytest tests/test_distributed_krylov_shard_multiprocess.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _krylov_worker(rank: int, world_size: int,
                   port: int, bench_key: str, method: str,
                   atol: float, rtol: float, maxiter: int,
                   restart: int,
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

        # ---- user-side setup ---------------------------------------- #
        bench = Synthetic[bench_key]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A, mesh, partition_method="simple")

        torch.manual_seed(0)
        b_global = torch.randn(A.shape[0], dtype=torch.float64)
        b_dt = D.scatter(b_global)

        # ---- distributed solve via the unified API ------------------ #
        # ``restart`` only matters for gmres / fgmres but the kwarg
        # is harmless elsewhere.
        with SolverConfig(method=method,
                          atol=atol, rtol=rtol,
                          maxiter=maxiter, restart=restart):
            x_dt = solve(D, b_dt)

        # ---- residual via public ops only --------------------------- #
        r_dt = b_dt - D @ x_dt
        rel_residual = float(
            (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item())

        # ---- sanity overlay vs SciPy single-process reference ------- #
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        A_sp = sp.coo_matrix(
            (bench.val.numpy(), (bench.row.numpy(), bench.col.numpy())),
            shape=bench.shape).tocsr()
        if method == "minres":
            scipy_kwargs = dict(rtol=1e-12, maxiter=maxiter)
        else:
            scipy_kwargs = dict(rtol=1e-12, atol=1e-12, maxiter=maxiter)
        scipy_fn = {
            "bicgstab": spla.bicgstab,
            "gmres":    lambda *a, **kw: spla.gmres(*a, restart=restart, **kw),
            "fgmres":   lambda *a, **kw: spla.gmres(*a, restart=restart, **kw),
            "minres":   spla.minres,
        }[method]
        x_ref_np, _ = scipy_fn(A_sp, b_global.numpy(), **scipy_kwargs)

        x_full = x_dt.full_tensor()
        x_ref = torch.from_numpy(x_ref_np).to(x_full.dtype)
        rel_to_scipy = float(
            ((x_full - x_ref).norm() / (x_ref.norm() + 1e-12)).item())

        out_queue.put({
            "rank": rank,
            "method": method,
            "rel_residual": rel_residual,
            "rel_to_scipy": rel_to_scipy,
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method,bench_key,port", [
    ("bicgstab", "convdiff_2d_64_peclet_10", 29521),
    ("gmres",    "convdiff_2d_64_peclet_10", 29522),
    ("fgmres",   "convdiff_2d_64_peclet_10", 29523),
    ("minres",   "poisson_2d_16",            29524),
])
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_shard_krylov_matches_scipy(method, bench_key, port):
    """Each Krylov method, run in Shard(0) space across 2 ranks, must
    drive ``||b - A x|| / ||b|| < 1e-5`` and produce a global solution
    that matches scipy's single-process result to rel-err 1e-2.
    """
    world_size = 2
    maxiter = 4000 if bench_key.startswith("convdiff") else 2000

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_krylov_worker,
                            args=(rank, world_size, port, bench_key, method,
                                  1e-10, 1e-10, maxiter, 50, out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=300) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=300)
            assert p.exitcode == 0, \
                f"rank {procs.index(p)}/{method} exited with {p.exitcode}"

        by_rank = {r["rank"]: r for r in results}
        assert set(by_rank) == set(range(world_size))

        for r in results:
            assert r["rel_residual"] < 1e-5, \
                f"{method}/rank {r['rank']}: rel-residual " \
                f"{r['rel_residual']:.2e}"
            assert r["rel_to_scipy"] < 1e-2, \
                f"{method}/rank {r['rank']}: distributed vs scipy " \
                f"rel-err {r['rel_to_scipy']:.2e}"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


# ---------------------------------------------------------------------- #
# SolverConfig scope integration
# ---------------------------------------------------------------------- #
def _solverconfig_scope_worker(rank: int, world_size: int,
                                port: int,
                                out_queue: mp.Queue) -> None:
    """Verify that the unified ``solve(D, b)`` entry point honours the
    active :class:`SolverConfig` scope. The non-symmetric convdiff
    benchmark is deliberate -- the hard-coded default ``method="cg"``
    does not converge here, so a passing residual proves the scope was
    actually consulted."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from torch_sla import (DSparseTensor, SparseTensor, solve,
                                 SolverConfig)
        from torch_sla.datasets import Synthetic

        bench = Synthetic["convdiff_2d_64_peclet_10"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A, mesh, partition_method="simple")

        torch.manual_seed(0)
        b_global = torch.randn(A.shape[0], dtype=torch.float64)
        b_dt = D.scatter(b_global)

        # Under SolverConfig(method="bicgstab"), the unified entry
        # point must pick bicgstab.
        with SolverConfig(method="bicgstab", atol=1e-10, rtol=1e-10,
                          maxiter=4000):
            x_dt = solve(D, b_dt)

        r_dt = b_dt - D @ x_dt
        rel_res = float(
            (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item())

        # Explicit kwarg should still override the surrounding scope.
        with SolverConfig(method="bicgstab", maxiter=4000):
            try:
                x_cg = solve(D, b_dt, method="cg", maxiter=50,
                              atol=1e-99, rtol=0)
                kwarg_override_ran = (x_cg is not None)
            except Exception:
                kwarg_override_ran = False

        out_queue.put({
            "rank": rank,
            "rel_residual_under_scope": rel_res,
            "kwarg_override_ran": kwarg_override_ran,
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_solverconfig_scope_propagates_to_shard_solve():
    """``SolverConfig(method=..., atol=..., ...)`` scope MUST be picked
    up by the unified ``solve(D, b)`` entry point -- otherwise the
    convdiff matrix silently runs through CG and produces garbage."""
    world_size = 2
    port = 29531
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_solverconfig_scope_worker,
                            args=(rank, world_size, port, out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=120) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, \
                f"rank {procs.index(p)} exited with {p.exitcode}"

        for r in results:
            assert r["rel_residual_under_scope"] < 1e-5, \
                f"rank {r['rank']}: scope=bicgstab but residual=" \
                f"{r['rel_residual_under_scope']:.2e}; scope likely ignored."
            assert r["kwarg_override_ran"], \
                f"rank {r['rank']}: explicit kwarg override didn't run"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="bicgstab")
    parser.add_argument("--bench",  default="convdiff_2d_64_peclet_10")
    parser.add_argument("--port",   type=int, default=29521)
    args = parser.parse_args()
    test_shard_krylov_matches_scipy(args.method, args.bench, args.port)
    print(f"OK: shard-{args.method} matches scipy on {args.bench}")
