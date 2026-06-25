#!/usr/bin/env python
"""Sugar-method <-> ``*_shard`` equivalence on ``DSparseTensor``.

The clean delegating methods added to :class:`DSparseTensor`
(``solve`` / ``nonlinear_solve`` / ``connected_components`` / ``lsqr`` /
``lsmr``) are thin wrappers that must return *exactly* what the
underlying ``*_shard`` implementation / free function returns. This test
runs both sides in the same distributed process and asserts bit-for-bit
(``torch.equal``) agreement, so any future logic drift in the wrappers is
caught immediately.

Covered:

* ``D.solve(b)``                == ``D.solve_distributed_shard(b)``
* ``D.lsqr(b)``                 == ``lsqr_shard(D, b, ...)``
* ``D.lsmr(b)``                 == ``lsmr_shard(D, b, ...)``
* ``D.nonlinear_solve(...)``    == ``D.nonlinear_solve_distributed_shard(...)``
* ``D.connected_components()``  == ``connected_components_shard(D)``

Verified rank-count invariant (world 2 and 4) on simple + rcb partitions.

Run with::

    pytest tests/distributed/test_dsparse_sugar_methods_multiprocess.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _coords_2d(m: int) -> torch.Tensor:
    ii, jj = torch.meshgrid(torch.arange(m), torch.arange(m), indexing="ij")
    return torch.stack([ii.flatten(), jj.flatten()], dim=1).to(torch.float64)


def _sugar_worker(rank, world_size, port, part, m, out_queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from torch_sla import DSparseTensor, SparseTensor, SolverConfig
        from torch_sla.distributed.solve import lsqr_shard, lsmr_shard
        from torch_sla.distributed.graph import connected_components_shard
        import torch_sla.datasets as d

        # SPD problem for solve(); non-symmetric isn't needed here since
        # we only compare wrapper vs delegate (identical operator).
        prob = d.poisson_2d(m)
        A = SparseTensor(prob.val, prob.row, prob.col, prob.shape)
        n = A.shape[0]
        mesh = init_device_mesh("cpu", (world_size,))
        if part == "rcb":
            D = DSparseTensor.partition(
                A, mesh, partition_method="rcb", coords=_coords_2d(m))
        else:
            D = DSparseTensor.partition(A, mesh, partition_method=part)

        torch.manual_seed(0)
        b_global = torch.randn(n, dtype=torch.float64)
        b_dt = D.scatter(b_global)
        b_owned = b_dt.to_local().contiguous()

        # ---- solve: sugar vs delegate (same SolverConfig scope) -------- #
        with SolverConfig(method="cg", atol=1e-12, rtol=1e-12, maxiter=2000):
            x_sugar = D.solve(b_dt).to_local()
            x_deleg = D.solve_distributed_shard(b_dt).to_local()
        same_solve = bool(torch.equal(x_sugar, x_deleg))

        # ---- lsqr / lsmr: sugar vs free function ----------------------- #
        ls_kw = dict(atol=1e-10, btol=1e-10, maxiter=2000)
        xq_sugar = D.lsqr(b_owned, **ls_kw)
        xq_deleg = lsqr_shard(D, b_owned, **ls_kw)
        same_lsqr = bool(torch.equal(xq_sugar, xq_deleg))

        xm_sugar = D.lsmr(b_owned, **ls_kw)
        xm_deleg = lsmr_shard(D, b_owned, **ls_kw)
        same_lsmr = bool(torch.equal(xm_sugar, xm_deleg))

        # ---- connected_components: sugar vs free function -------------- #
        lab_sugar, nc_sugar = D.connected_components()
        lab_deleg, nc_deleg = connected_components_shard(D)
        same_cc = (bool(torch.equal(lab_sugar, lab_deleg))
                   and nc_sugar == nc_deleg)

        # ---- nonlinear_solve: sugar vs delegate ------------------------ #
        lam = 0.5

        def residual(u_owned, Dop):
            return Dop._shard_matvec(u_owned) - lam * torch.exp(u_owned)

        def jac_diag(u_owned, Dop):
            return -lam * torch.exp(u_owned)

        u0 = D.scatter(torch.zeros(n, dtype=torch.float64))
        u_sugar = D.nonlinear_solve(
            residual, u0, jac_diag_fn=jac_diag, max_iter=30).to_local()
        u0b = D.scatter(torch.zeros(n, dtype=torch.float64))
        u_deleg = D.nonlinear_solve_distributed_shard(
            residual, u0b, jac_diag_fn=jac_diag, max_iter=30).to_local()
        same_nl = bool(torch.equal(u_sugar, u_deleg))

        out_queue.put({
            "rank": rank, "part": part,
            "same_solve": same_solve, "same_lsqr": same_lsqr,
            "same_lsmr": same_lsmr, "same_cc": same_cc,
            "same_nl": same_nl,
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("part,world_size,port", [
    ("simple", 2, 29631),
    ("simple", 4, 29632),
    ("rcb",    4, 29633),
])
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_sugar_methods_match_shard_delegates(part, world_size, port):
    """Each DSparseTensor sugar method returns bit-for-bit what its
    ``*_shard`` delegate returns, across world 2 & 4 and a non-monotone
    rcb partition."""
    m = 12
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_sugar_worker,
                            args=(rank, world_size, port, part, m, out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=300) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=300)
            assert p.exitcode == 0, f"rank exited with {p.exitcode}"

        for r in results:
            tag = f"{part}/rank {r['rank']}"
            assert r["same_solve"], f"{tag}: solve != solve_distributed_shard"
            assert r["same_lsqr"], f"{tag}: lsqr != lsqr_shard"
            assert r["same_lsmr"], f"{tag}: lsmr != lsmr_shard"
            assert r["same_cc"], f"{tag}: connected_components mismatch"
            assert r["same_nl"], \
                f"{tag}: nonlinear_solve != nonlinear_solve_distributed_shard"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    test_sugar_methods_match_shard_delegates("rcb", 4, 29633)
    print("OK")
