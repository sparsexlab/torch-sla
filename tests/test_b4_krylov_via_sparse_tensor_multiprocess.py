#!/usr/bin/env python
"""B4: Krylov methods on a B3-backed DSparseTensor.

Verifies that ``solve_distributed_shard`` (and every Krylov method it
dispatches to) runs end-to-end on a SparseTensor-backed DSparseTensor,
with the same correctness as the legacy DSparseMatrix-backed path.

If this test passes, ``_distributed_*_shard`` no longer needs
``DSparseMatrix`` -- they go through ``_shard_matvec`` and
``_make_preconditioner``, both of which now read from either backing.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _b4_worker(rank: int, world_size: int, port: int,
               method: str, precond, out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Shard

        from torch_sla import DSparseTensor, SparseTensor
        from torch_sla.datasets import Synthetic

        bench = Synthetic["poisson_2d_16"]
        N = bench.shape[0]
        A_global = SparseTensor(bench.val, bench.row, bench.col,
                                 bench.shape)

        # Build partition + local SparseTensor → B3 DSparseTensor.
        local_mat = A_global.partition_for_rank(
            rank, world_size, partition_method="simple")
        partition = local_mat.partition
        local_st = A_global.extract_partition(partition)
        mesh = init_device_mesh("cpu", (world_size,))
        D = DSparseTensor.from_sparse_local(
            local_st, mesh, partition,
            global_shape=A_global.shape,
        )
        assert D._local_tensor is not None
        assert D._local_matrix is None

        torch.manual_seed(0)
        b_owned = torch.randn(N, dtype=torch.float64)[partition.owned_nodes]
        b_dt = DTensor.from_local(b_owned, mesh, [Shard(0)])

        x_dt = D.solve_distributed_shard(
            b_dt, method=method, preconditioner=precond,
            atol=1e-10, rtol=1e-10, maxiter=2000)
        x = x_dt.to_local()

        r = b_owned - D._shard_matvec(x)
        rs = torch.dot(r, r); dist.all_reduce(rs)
        bs = torch.dot(b_owned, b_owned); dist.all_reduce(bs)
        rel = float(rs.sqrt().item()) / (float(bs.sqrt().item()) + 1e-30)
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
def test_b4_krylov_on_b3_backed_dsparse(method, precond, port):
    """Each Krylov method, run on a DSparseTensor built via the B3
    SparseTensor path, must drive ``||r||/||b|| < 1e-5`` on the
    SPD Poisson stencil. Proves the Krylov stack no longer depends
    on DSparseMatrix."""
    world_size = 2
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_b4_worker,
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
