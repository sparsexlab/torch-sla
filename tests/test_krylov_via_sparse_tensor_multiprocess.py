#!/usr/bin/env python
"""Krylov methods on the SparseTensor-backed DSparseTensor path.

Same end-to-end style as the other distributed Krylov tests: build
``A``, build ``D`` (this time through the SparseTensor /
``DSparseTensor.from_sparse_local`` route rather than
``.partition``), build the RHS via ``D.scatter``, hand both to
``solve``, verify with ``D @ x_dt``.

If this test passes, the unified solver stack works just as well on a
``DSparseTensor`` that was constructed through the SparseTensor
backing -- proving the Krylov methods no longer depend on the legacy
``DSparseMatrix`` backing.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _krylov_worker(rank: int, world_size: int, port: int,
                   method: str, precond, out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from torch_sla import DSparseTensor, SparseTensor, solve, SolverConfig
        from torch_sla.datasets import Synthetic

        # ---- user-side setup: SparseTensor-only path ---------------- #
        bench = Synthetic["poisson_2d_16"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

        # Build the local SparseTensor + partition; wrap as DSparseTensor.
        local_meta = A.partition_for_rank(
            rank, world_size, partition_method="simple")
        partition = local_meta.partition
        local_st = A.extract_partition(partition)

        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.from_sparse_local(
            local_st, mesh, partition,
            global_shape=A.shape,
        )
        # Sanity: we really took the SparseTensor path, not the legacy
        # DSparseMatrix backing.
        assert D._local_tensor is not None
        assert D._local_matrix is None

        torch.manual_seed(0)
        b_global = torch.randn(A.shape[0], dtype=torch.float64)
        b_dt = D.scatter(b_global)

        with SolverConfig(method=method, preconditioner=precond,
                          atol=1e-10, rtol=1e-10, maxiter=2000):
            x_dt = solve(D, b_dt)

        r_dt = b_dt - D @ x_dt
        rel = float(
            (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item())
        out_queue.put({"rank": rank, "rel_residual": rel})
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("method,precond,port", [
    ("cg",       None,           29611),
    ("cg",       "block_jacobi", 29612),
    ("bicgstab", "jacobi",       29613),
    ("gmres",    None,           29614),
    ("minres",   None,           29615),
])
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_krylov_on_sparse_tensor_backed_dsparse(method, precond, port):
    """Each Krylov method, run on a DSparseTensor built via the
    SparseTensor path, must drive ``||r||/||b|| < 1e-5`` on the
    SPD Poisson stencil. Proves the unified solver no longer depends
    on DSparseMatrix."""
    world_size = 2
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_krylov_worker,
                            args=(rank, world_size, port, method, precond,
                                  out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=180) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=180)
            assert p.exitcode == 0, \
                f"{method}/{precond} rank {procs.index(p)} exited with " \
                f"{p.exitcode}"
        for r in results:
            assert r["rel_residual"] < 1e-5, \
                f"{method}/{precond}/{r['rank']}: rel-residual " \
                f"{r['rel_residual']:.2e}"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
