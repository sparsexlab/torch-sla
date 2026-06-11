#!/usr/bin/env python
"""Multi-process test for ``DSparseMatrix.gather_global``.

The bug we're guarding against: the distributed gather path allocated a
zero-filled ``x_global`` on rank 0, gathered the values into a temporary
list, and then returned ``x_global`` *without* placing the gathered
values into it -- making every gather on rank 0 a no-op that silently
returned a zero vector. The single-process branch (``world_size==1``)
worked, so the bug only surfaced once a job ran on multiple ranks.

This test spawns 2 ranks via ``torch.multiprocessing.spawn`` (gloo
backend so it works without CUDA) and verifies that rank 0 actually
receives the original global vector after gather. Run with::

    pytest tests/test_gather_global_multiprocess.py -v
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# Make sure the in-tree package is importable when this file is
# invoked through pytest (the multiprocessing.spawn'd children inherit
# the current Python path, so this also covers the subprocess case).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------- #
# Worker that exercises gather_global and reports back to the parent
# ---------------------------------------------------------------------- #
def _gather_global_worker(rank: int, world_size: int,
                          out_queue: mp.Queue) -> None:
    """Per-rank worker. Each rank owns a contiguous block of the 2-D
    Poisson stencil; every rank calls ``gather_global`` (symmetric
    Allgather semantics) and pushes its result back to the parent.
    The parent asserts both ranks observed the same global vector."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch_sla import SparseTensor
        from torch_sla.datasets import Synthetic

        # Use the project's standard Synthetic benchmark catalogue
        # (poisson_2d_16 = a 256-DOF 2D Poisson stencil, the smallest
        # catalogued one). No network IO, but it's the same primitive
        # the rest of the test suite uses, so we stay consistent with
        # ``benchmark_small_real`` instead of rolling a one-off stencil.
        bench = Synthetic["poisson_2d_16"]
        A_global = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        n = bench.shape[0]

        # Each rank materialises its own subdomain via simple striping.
        D = A_global.partition_for_rank(rank, world_size,
                                        partition_method="simple")

        # Set each owned entry to its own global index value so the
        # gathered vector is trivially predictable: x_global[i] == i.
        owned = D.partition.owned_nodes
        x_local = torch.zeros(D.num_local, dtype=torch.float64)
        x_local[:D.num_owned] = owned.to(torch.float64)

        x_global = D.gather_global(x_local)
        out_queue.put((rank, x_global.cpu()))
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------- #
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_gather_global_returns_filled_vector_on_every_rank():
    """After the fix, EVERY rank must receive a vector whose entries
    equal their global indices (0, 1, 2, ..., n-1) -- not the all-zero
    rank-0-only result the previous code returned."""
    world_size = 2

    # ``mp.get_context("spawn")`` is the only context that works
    # reliably across pytest invocations + CUDA tensors.
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(
                target=_gather_global_worker,
                args=(rank, world_size, out_queue),
            )
            p.start()
            procs.append(p)

        results = [out_queue.get(timeout=60) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, \
                f"rank {procs.index(p)} exited with code {p.exitcode}"

        # Every rank should observe an identical full global vector
        # whose entries equal their global indices.
        per_rank = {rank: gathered for rank, gathered in results}
        assert set(per_rank) == set(range(world_size)), \
            f"missing ranks in result set: {set(per_rank)}"
        expected = torch.arange(per_rank[0].numel(), dtype=torch.float64)
        for rank, gathered in per_rank.items():
            torch.testing.assert_close(gathered, expected,
                                       msg=f"rank {rank} got the wrong vector")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    # Allow ``python tests/test_gather_global_multiprocess.py`` for quick
    # iteration outside pytest.
    test_gather_global_returns_filled_vector_on_every_rank()
    print("OK: every rank received the filled global vector after gather")
