#!/usr/bin/env python
"""Distributed least-squares Krylov (LSQR / LSMR) on ``DSparseTensor``.

LSQR and LSMR are the Golub-Kahan bidiagonalisation least-squares
solvers. They need both ``A @ x`` and ``Aᵀ @ y`` -- the latter via the
new transpose-shard matvec (``D._shard_rmatvec`` / reverse halo
exchange). Here we solve the square but **non-symmetric**
``advection_diffusion_2d`` problem (which exercises the ``Aᵀ`` path,
since for a non-symmetric ``A`` LSQR ≠ CG) and assert that the
distributed solution matches scipy's single-process ``lsqr`` / ``lsmr``
to solver tolerance.

Verified rank-count invariant (world 2 and 4) and on a non-monotone
``rcb`` partition (owned node ids out of rank order -> exercises the
``owned_nodes`` gather in ``full_tensor`` / ``scatter``).

Run with::

    pytest tests/distributed/test_dsparse_lsqr_lsmr_multiprocess.py -v
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


def _lsqr_worker(rank, world_size, port, method, part, m, out_queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from torch_sla import DSparseTensor, SparseTensor, solve, SolverConfig
        import torch_sla.datasets as d

        prob = d.advection_diffusion_2d(m)
        A = SparseTensor(prob.val, prob.row, prob.col, prob.shape)
        mesh = init_device_mesh("cpu", (world_size,))
        if part == "rcb":
            D = DSparseTensor.partition(
                A, mesh, partition_method="rcb", coords=_coords_2d(m))
        else:
            D = DSparseTensor.partition(A, mesh, partition_method=part)

        torch.manual_seed(0)
        b_global = torch.randn(A.shape[0], dtype=torch.float64)
        b_dt = D.scatter(b_global)

        with SolverConfig(method=method, atol=1e-12, rtol=1e-12, maxiter=2000):
            x_dt = solve(D, b_dt)

        r_dt = b_dt - D @ x_dt
        rel_res = float(
            (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item())

        # scipy single-process reference
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        A_sp = sp.coo_matrix(
            (prob.val.numpy(), (prob.row.numpy(), prob.col.numpy())),
            shape=prob.shape).tocsr()
        if method == "lsqr":
            x_ref = spla.lsqr(A_sp, b_global.numpy(),
                              atol=1e-12, btol=1e-12, iter_lim=2000)[0]
        else:
            x_ref = spla.lsmr(A_sp, b_global.numpy(),
                              atol=1e-12, btol=1e-12, maxiter=2000)[0]
        x_ref = torch.from_numpy(x_ref)
        x_full = x_dt.full_tensor()
        rel_to_scipy = float(
            ((x_full - x_ref).norm() / (x_ref.norm() + 1e-12)).item())

        # Confirm the rcb partition genuinely exercises the owned<->global
        # mapping, i.e. that ``full_tensor`` cannot just rank-order
        # concatenate the owned slices. The relevant non-monotonicity is
        # *global* (across ranks), not within a single rank's owned_nodes:
        # ``rcb`` sorts ids within each rank, so the old within-rank check
        # ``owned[1:] < owned[:-1]`` was always False and never fired.
        #
        # Each rank advertises the [min, max] span of its owned ids; in
        # rank order these spans must be strictly increasing & disjoint for
        # the layout to be rank-monotone (rank-order concat == sorted). If
        # any later rank's span overlaps/precedes an earlier rank's, the
        # rank-order concatenation is a non-identity permutation of the
        # global vector -- exactly what owned_nodes / gather_owned_to_global
        # must undo. Detect that collectively.
        owned = D._spec.placement.partition.owned_nodes
        span = torch.tensor(
            [int(owned.min().item()), int(owned.max().item())]
            if owned.numel() > 0 else [-1, -1],
            dtype=torch.int64,
        )
        spans = [torch.zeros(2, dtype=torch.int64) for _ in range(world_size)]
        dist.all_gather(spans, span)
        mins = [int(s[0].item()) for s in spans]
        maxs = [int(s[1].item()) for s in spans]
        # rank-monotone iff spans are non-overlapping and ordered by rank.
        nonmono = any(maxs[r] > mins[r + 1] for r in range(world_size - 1))

        out_queue.put({
            "rank": rank, "method": method, "part": part,
            "rel_residual": rel_res, "rel_to_scipy": rel_to_scipy,
            "nonmono": nonmono,
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method,part,world_size,port", [
    ("lsqr", "simple", 2, 29611),
    ("lsqr", "simple", 4, 29612),
    ("lsqr", "rcb",    4, 29613),
    ("lsmr", "simple", 2, 29614),
    ("lsmr", "simple", 4, 29615),
    ("lsmr", "rcb",    4, 29616),
])
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_shard_lsqr_lsmr_matches_scipy(method, part, world_size, port):
    """Distributed LSQR/LSMR on the non-symmetric advection-diffusion
    problem must match scipy single-process to solver tolerance, for
    world 2 & 4 and for a non-monotone rcb partition."""
    m = 16
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_lsqr_worker,
                            args=(rank, world_size, port, method, part, m,
                                  out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=300) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=300)
            assert p.exitcode == 0, \
                f"rank exited with {p.exitcode}"

        for r in results:
            assert r["rel_residual"] < 1e-6, \
                f"{method}/{part}/rank {r['rank']}: rel-residual " \
                f"{r['rel_residual']:.2e}"
            assert r["rel_to_scipy"] < 1e-6, \
                f"{method}/{part}/rank {r['rank']}: vs scipy " \
                f"{r['rel_to_scipy']:.2e}"
        if part == "rcb":
            assert any(r["nonmono"] for r in results), \
                "rcb partition is rank-monotone; non-monotone path not exercised"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    test_shard_lsqr_lsmr_matches_scipy("lsqr", "rcb", 4, 29613)
    print("OK")
