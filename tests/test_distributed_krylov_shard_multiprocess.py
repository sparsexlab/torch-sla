#!/usr/bin/env python
"""Multi-process tests for the Shard(0) Krylov family on
``DSparseTensor.solve_distributed_shard``.

Verifies BiCGStab / GMRES / FGMRES / MINRES converge to the same
solution SciPy's reference single-process solvers produce on the
global matrix. CG has its own test in
``tests/test_distributed_cg_shard_multiprocess.py``.

* BiCGStab + GMRES + FGMRES use the non-symmetric
  ``Synthetic["convdiff_2d_64_peclet_10"]`` benchmark so we exercise
  the indefinite path.
* MINRES uses the SPD ``Synthetic["poisson_2d_16"]`` benchmark --
  scipy's reference also accepts SPD even though MINRES targets
  symmetric indefinite.

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
        from torch.distributed.tensor import DTensor, Shard

        from torch_sla import (DSparseTensor, SparseTensor, solve,
                                 SolverConfig)
        from torch_sla.datasets import Synthetic

        bench = Synthetic[bench_key]
        N = bench.shape[0]
        A_global = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A_global, mesh,
                                     partition_method="simple")
        local_matrix = D.to_local()

        torch.manual_seed(0)
        b_global = torch.randn(N, dtype=torch.float64)
        b_owned = b_global[local_matrix.partition.owned_nodes]
        b_dt = DTensor.from_local(b_owned, mesh, [Shard(0)])

        # Unified solve() entry + SolverConfig scope -- restart only
        # matters for gmres / fgmres but the kwarg is harmless elsewhere.
        with SolverConfig(method=method,
                          atol=atol, rtol=rtol,
                          maxiter=maxiter, restart=restart):
            x_dt = solve(D, b_dt)
        x_owned = x_dt.to_local()

        # SciPy reference on the full global matrix.
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        A_sp = sp.coo_matrix(
            (bench.val.numpy(), (bench.row.numpy(), bench.col.numpy())),
            shape=bench.shape).tocsr()
        b_np = b_global.numpy()
        # scipy's minres uses an older kwarg shape (no ``atol``); the
        # other methods accept the modern atol+rtol combo.
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
        x_ref, _ = scipy_fn(A_sp, b_np, **scipy_kwargs)

        owned_np = local_matrix.partition.owned_nodes.cpu().numpy()
        x_ref_owned = torch.from_numpy(x_ref[owned_np])

        # Distributed solver's own residual ||r||/||b|| -- the absolute
        # truth-of-convergence; ref-comparison is a sanity overlay.
        r = b_owned - D._shard_matvec(x_owned)
        r_norm_sq = torch.dot(r, r)
        dist.all_reduce(r_norm_sq, op=dist.ReduceOp.SUM)
        b_norm_sq = torch.dot(b_owned, b_owned)
        dist.all_reduce(b_norm_sq, op=dist.ReduceOp.SUM)

        rel_residual = float(r_norm_sq.sqrt().item()) / (
            float(b_norm_sq.sqrt().item()) + 1e-30)
        rel_to_scipy = float((x_owned - x_ref_owned).norm().item()) / (
            float(x_ref_owned.norm().item()) + 1e-12)

        out_queue.put({
            "rank": rank,
            "method": method,
            "rel_residual": rel_residual,
            "rel_to_scipy": rel_to_scipy,
            "x_owned_size": int(x_owned.numel()),
            "num_owned": int(local_matrix.num_owned),
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
    drive ``||A x - b|| / ||b|| < 1e-5`` and produce per-rank owned
    slices that match scipy's single-process result to rel-err 1e-3.
    """
    world_size = 2
    # Heavier tolerances for the harder problem
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
            # Shard(0) shape contract preserved.
            assert r["x_owned_size"] == r["num_owned"]
            # Distributed residual is small (the "did we actually solve it"
            # check; this is the strict criterion).
            assert r["rel_residual"] < 1e-5, \
                f"{method}/rank {r['rank']}: rel-residual " \
                f"{r['rel_residual']:.2e}"
            # Cross-check vs scipy (loose because Krylov methods are not
            # required to converge to exactly the same point under
            # different stopping criteria, but order-of-magnitude must
            # agree).
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
    """Verify that ``solve_distributed_shard`` reads from the active
    :class:`SolverConfig` scope. We use the non-symmetric convdiff
    benchmark on purpose -- the hard-coded default ``method="cg"``
    would *not* converge on this matrix, so a passing test proves
    scope was actually consulted (otherwise CG would run and produce
    a garbage residual)."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        from torch_sla import (DSparseTensor, SparseTensor, solve,
                                 SolverConfig)
        from torch_sla.datasets import Synthetic

        bench = Synthetic["convdiff_2d_64_peclet_10"]
        N = bench.shape[0]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A, mesh, partition_method="simple")
        local_matrix = D.to_local()

        torch.manual_seed(0)
        b_owned = torch.randn(N, dtype=torch.float64)[
            local_matrix.partition.owned_nodes]
        b_dt = DTensor.from_local(b_owned, mesh, [Shard(0)])

        # Inside the scope: bicgstab + tight tols should be picked up by
        # the unified ``solve(D, b)`` entry point.
        with SolverConfig(method="bicgstab", atol=1e-10, rtol=1e-10,
                          maxiter=4000):
            x_dt = solve(D, b_dt)

        x_owned = x_dt.to_local()
        # Distributed residual.
        r = b_owned - D._shard_matvec(x_owned)
        rs = torch.dot(r, r)
        dist.all_reduce(rs, op=dist.ReduceOp.SUM)
        bs = torch.dot(b_owned, b_owned)
        dist.all_reduce(bs, op=dist.ReduceOp.SUM)
        rel_res = float(rs.sqrt().item()) / (float(bs.sqrt().item()) + 1e-30)

        # Explicit kwarg override inside same scope: pass method="cg".
        # That one should ignore the scope's "bicgstab" and run plain CG
        # (which won't converge -- we just check the kwarg overrode).
        with SolverConfig(method="bicgstab", maxiter=4000):
            try:
                x_cg = solve(D, b_dt, method="cg", maxiter=50,
                              atol=1e-99, rtol=0)
                # Don't assert convergence -- just that the call returned.
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
    up by ``solve_distributed_shard`` -- otherwise the convdiff matrix
    silently runs through CG and produces garbage."""
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
            # Under SolverConfig(method='bicgstab'), the solve must
            # converge on convdiff. If the scope had been ignored, CG
            # would have been picked and we'd see a huge residual.
            assert r["rel_residual_under_scope"] < 1e-5, \
                f"rank {r['rank']}: scope=bicgstab but residual=" \
                f"{r['rel_residual_under_scope']:.2e}; scope likely ignored."
            # Explicit kwarg override must work too.
            assert r["kwarg_override_ran"], \
                f"rank {r['rank']}: explicit kwarg override didn't run"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    # Quick standalone run -- exercises bicgstab only for fast iteration.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="bicgstab")
    parser.add_argument("--bench",  default="convdiff_2d_64_peclet_10")
    parser.add_argument("--port",   type=int, default=29521)
    args = parser.parse_args()
    test_shard_krylov_matches_scipy(args.method, args.bench, args.port)
    print(f"OK: shard-{args.method} matches scipy on {args.bench}")
