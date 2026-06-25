#!/usr/bin/env python
"""Distributed nonlinear solve (Newton + IFT adjoint) on ``DSparseTensor``.

The 1-D Bratu problem ``F(u) = A u - lam exp(u) = 0`` with
``A = (1/h^2) tridiag(-1, 2, -1)`` is the canonical semilinear elliptic
test. Each Newton step solves the Jacobian system
``J du = -F``, ``J = A - lam diag(exp(u))``, via distributed GMRES
against the operator ``J v = A v - lam exp(u) * v`` (diagonal shift).
The reverse-halo transpose matvec gives ``Jᵀ`` for the IFT adjoint.

Asserts:

* the distributed solution matches the closed-form analytical Bratu
  solution to O(h^2) discretization error;
* it matches the single-process ``SparseTensor.nonlinear_solve`` to
  ~machine precision;
* the distributed IFT adjoint ``λ`` (``Jᵀ λ = dL/du``) matches a dense
  reference solve of the same transposed Jacobian system.

Verified rank-count invariant (world 2 and 4) and on a non-monotone
``rcb`` partition.

Run with::

    pytest tests/distributed/test_dsparse_nonlinear_bratu_multiprocess.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _bratu_worker(rank, world_size, port, part, n, lam, out_queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from torch_sla import DSparseTensor, SparseTensor
        import torch_sla.datasets as d
        from torch_sla.distributed.collectives import gather_owned_to_global

        prob = d.bratu_1d(n, lam=lam)
        A = SparseTensor(prob.val, prob.row, prob.col, prob.shape)
        mesh = init_device_mesh("cpu", (world_size,))
        if part == "rcb":
            # 2-D folded coords (node i at (i mod 20, i div 20)) so the
            # recursive-coordinate-bisection cuts interleave node ids
            # across ranks -> a genuinely NON-MONOTONE owned set (the
            # owned ids are not a contiguous arange block in rank order).
            # A 1-D arange coord would just reproduce the contiguous
            # ``simple`` partition and never exercise the
            # gather_owned_to_global reconstruction path.
            idx = torch.arange(n)
            coords = torch.stack(
                [(idx % 20).to(torch.float64),
                 (idx // 20).to(torch.float64)], dim=1)
            D = DSparseTensor.partition(
                A, mesh, partition_method="rcb", coords=coords)
        else:
            D = DSparseTensor.partition(A, mesh, partition_method=part)

        def residual(u_owned, Dop):
            return Dop._shard_matvec(u_owned) - lam * torch.exp(u_owned)

        def jac_diag(u_owned, Dop):
            # J = A - lam diag(exp(u))  =>  diagonal shift d = -lam exp(u)
            return -lam * torch.exp(u_owned)

        partition = D._spec.placement.partition
        owned = partition.owned_nodes.to(torch.int64)

        # ---- Newton forward solve + IFT adjoint in one call ------------ #
        u0 = D.scatter(torch.zeros(n, dtype=torch.float64))
        # adjoint RHS dL/du: an arbitrary owned-slice vector for L = sum(u)
        dLdu_global = torch.ones(n, dtype=torch.float64)
        dLdu_dt = D.scatter(dLdu_global)
        u_dt, lam_dt = D.nonlinear_solve_distributed_shard(
            residual, u0, jac_diag_fn=jac_diag, max_iter=50,
            adjoint_dLdu=dLdu_dt)

        u_g = gather_owned_to_global(
            owned, u_dt.to_local().contiguous(), n)
        lam_g = gather_owned_to_global(
            owned, lam_dt.to_local().contiguous(), n)

        err_exact = (u_g - prob.exact).abs().max().item()

        # ---- single-process reference (forward) ------------------------ #
        def _resid(u, Asp, lm):
            return Asp @ u - lm * torch.exp(u)
        u_sp = A.nonlinear_solve(
            _resid, torch.zeros(n, dtype=torch.float64),
            torch.tensor(lam), linear_method="lu")
        rel_sp = ((u_g - u_sp).norm() / (u_sp.norm() + 1e-30)).item()

        # ---- dense reference for the IFT adjoint Jᵀ λ = dL/du ---------- #
        import scipy.sparse as sp
        A_sp = sp.coo_matrix(
            (prob.val.numpy(), (prob.row.numpy(), prob.col.numpy())),
            shape=prob.shape).tocsr()
        import numpy as np
        Jdense = A_sp.toarray() - lam * np.diag(np.exp(u_sp.numpy()))
        lam_ref = np.linalg.solve(Jdense.T, dLdu_global.numpy())
        lam_ref = torch.from_numpy(lam_ref)
        rel_adj = ((lam_g - lam_ref).norm() / (lam_ref.norm() + 1e-30)).item()

        # ``owned_nodes`` is built via nonzero() so it is always sorted
        # ASCENDING within a rank; "non-monotone" here means the owned
        # set is not a single contiguous arange block (so rank-order
        # concatenation would permute the global vector and only
        # gather_owned_to_global reconstructs it correctly).
        nonmono = (
            not bool(torch.equal(
                owned,
                torch.arange(int(owned.min()), int(owned.max()) + 1)))
            if owned.numel() > 1 else False
        )

        out_queue.put({
            "rank": rank, "part": part,
            "err_exact": err_exact, "rel_singleproc": rel_sp,
            "rel_adjoint": rel_adj, "nonmono": nonmono,
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("part,world_size,port", [
    ("simple", 2, 29621),
    ("simple", 4, 29622),
    ("rcb",    4, 29623),
])
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_shard_bratu_newton_and_adjoint(part, world_size, port):
    """Distributed Bratu Newton solve matches the analytical solution
    (O(h^2)) and the single-process solver (~machine precision); the
    distributed IFT adjoint matches a dense transposed-Jacobian solve.
    Rank-count invariant + non-monotone rcb partition."""
    n, lam = 200, 1.0
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_bratu_worker,
                            args=(rank, world_size, port, part, n, lam,
                                  out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=300) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=300)
            assert p.exitcode == 0, f"rank exited with {p.exitcode}"

        for r in results:
            assert r["err_exact"] < 1e-3, \
                f"{part}/rank {r['rank']}: vs analytical {r['err_exact']:.2e}"
            assert r["rel_singleproc"] < 1e-8, \
                f"{part}/rank {r['rank']}: vs single-proc {r['rel_singleproc']:.2e}"
            assert r["rel_adjoint"] < 1e-7, \
                f"{part}/rank {r['rank']}: IFT adjoint {r['rel_adjoint']:.2e}"
        if part == "rcb":
            assert any(r["nonmono"] for r in results), \
                "rcb partition is rank-monotone; non-monotone path not exercised"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    test_shard_bratu_newton_and_adjoint("rcb", 4, 29623)
    print("OK")
