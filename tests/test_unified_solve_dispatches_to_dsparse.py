#!/usr/bin/env python
"""Verify the unified :func:`torch_sla.solve` recognises
:class:`DSparseTensor` inputs and routes to ``solve_distributed_shard``.

Before this PR the only path to the Shard(0) distributed solver was to
call ``D.solve_distributed_shard(b, ...)`` directly. After it::

    x = solve(D, b)                          # auto-routes to shard solve
    x = solve(D, b, method="bicgstab")       # kwarg overrides
    with SolverConfig.spd().gpu():           # scope flows in
        x = solve(D, b)
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _unified_worker(rank: int, world_size: int,
                    port: int, kind: str,
                    out_queue: mp.Queue) -> None:
    """``kind`` selects which dispatch path to exercise:

    * ``"explicit"``  -- ``solve(D, b, method="bicgstab", preconditioner="block_jacobi")``
    * ``"scope"``     -- explicit kwargs absent, settings come from
                         ``SolverConfig`` scope
    * ``"default"``   -- no kwargs and no scope; ``solve`` falls back to
                         its hard-coded defaults inside the shard solver
                         (cg + identity precond on an SPD matrix)
    """
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

        # SPD Poisson works with every path (cg/bicgstab/gmres);
        # convdiff would also work but BiCGStab is fine on Poisson too.
        bench = Synthetic["poisson_2d_16"]
        N = bench.shape[0]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.partition(A, mesh, partition_method="simple")
        local = D.to_local()

        torch.manual_seed(0)
        b_owned = torch.randn(N, dtype=torch.float64)[
            local.partition.owned_nodes]
        b_dt = DTensor.from_local(b_owned, mesh, [Shard(0)])

        if kind == "explicit":
            x_dt = solve(D, b_dt,
                         method="bicgstab",
                         preconditioner="block_jacobi",
                         atol=1e-10, rtol=1e-10, maxiter=2000)
        elif kind == "scope":
            with SolverConfig(method="bicgstab",
                              preconditioner="block_jacobi",
                              atol=1e-10, rtol=1e-10, maxiter=2000):
                x_dt = solve(D, b_dt)
        elif kind == "default":
            # No kwargs, no scope -- shard solver picks cg / identity.
            x_dt = solve(D, b_dt, atol=1e-10, rtol=1e-10, maxiter=2000)
        else:
            raise ValueError(kind)

        # Result must be a DTensor (so the rest of the FSDP / TP
        # ecosystem can compose with it).
        assert hasattr(x_dt, "to_local"), \
            f"{kind}: solve(D, b) didn't return a DTensor"
        x_owned = x_dt.to_local()
        assert x_owned.shape[0] == local.num_owned, \
            f"{kind}: result size {x_owned.shape[0]} != num_owned {local.num_owned}"

        # Residual check.
        r = b_owned - D._shard_matvec(x_owned)
        rs = torch.dot(r, r); dist.all_reduce(rs)
        bs = torch.dot(b_owned, b_owned); dist.all_reduce(bs)
        rel_res = float(rs.sqrt().item()) / (float(bs.sqrt().item()) + 1e-30)

        out_queue.put({"rank": rank, "kind": kind,
                        "rel_residual": rel_res})
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind,port", [
    ("explicit", 29571),
    ("scope",    29572),
    ("default",  29573),
])
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_unified_solve_routes_to_dsparse_shard(kind, port):
    """Each dispatch path -- explicit kwargs, ``SolverConfig`` scope,
    and hard-coded defaults -- must drive the residual below 1e-5 on
    the 256-DOF Poisson stencil. Proves ``solve(D, b)`` actually
    reached ``solve_distributed_shard`` (a bypass would either error or
    diverge)."""
    world_size = 2
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_unified_worker,
                            args=(rank, world_size, port, kind, out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=120) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=120)
            assert p.exitcode == 0, \
                f"rank {procs.index(p)} ({kind}) exited with {p.exitcode}"
        for r in results:
            assert r["rel_residual"] < 1e-5, \
                f"{kind}/{r['rank']}: rel-residual {r['rel_residual']:.2e}"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    for k, port in [("explicit", 29571), ("scope", 29572), ("default", 29573)]:
        test_unified_solve_routes_to_dsparse_shard(k, port)
        print(f"OK: solve(D, b) routes correctly under {k!r} dispatch path")
