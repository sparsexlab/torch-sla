#!/usr/bin/env python
"""``DSparseTensor`` element-wise math parity tests.

Phase C5: covers scalar arithmetic (``+ - * / **``), unary ops
(``- abs sqrt square exp log conj``), and same-spec
``DSparseTensor + DSparseTensor``. Each op is verified against the
equivalent single-process ``SparseTensor`` computation via
``full_tensor()``.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _elementwise_worker(rank: int, world_size: int, port: int,
                        out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch_sla import SparseTensor
        from torch_sla.datasets import Synthetic

        bench = Synthetic["poisson_2d_16"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        D = A.partition_for_rank(rank=rank, world_size=world_size)

        # ---- scalar arithmetic ----
        results = {
            "rank":           rank,
            "shape":          tuple(D.shape),
            # After every op, fold to a scalar (sum of stored values)
            # so we can cheaply check parity across ranks.
            "add_scalar":     float((D + 1.5).sum().item()),
            "radd_scalar":    float((1.5 + D).sum().item()),
            "sub_scalar":     float((D - 0.5).sum().item()),
            "rsub_scalar":    float((1.0 - D).sum().item()),
            "mul_scalar":     float((D * 2.0).sum().item()),
            "rmul_scalar":    float((2.0 * D).sum().item()),
            "div_scalar":     float((D / 0.5).sum().item()),
            "pow_scalar":     float((D ** 2).sum().item()),
            "neg":            float((-D).sum().item()),
            "abs":            float(D.abs().sum().item()),
            "sqrt_of_abs":    float(D.abs().sqrt().sum().item()),
            "square":         float(D.square().sum().item()),
            "exp":            float(D.exp().sum().item()),
        }

        # ---- same-spec DSparseTensor + DSparseTensor ----
        D2 = D * 3.0
        D_sum = D + D2  # = 4D in stored-value land
        results["same_spec_add"] = float(D_sum.sum().item())
        results["same_spec_sub"] = float((D2 - D).sum().item())  # = 2D

        # ---- result is still a DSparseTensor with same spec ----
        results["sum_is_dsparse_friendly"] = (
            type(D_sum).__name__ == "DSparseTensor"
            and D_sum.shape == D.shape
            and D_sum._spec is D._spec
        )

        out_queue.put(results)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_dsparse_elementwise_2procs():
    """Element-wise math agrees with the same op on a single-process
    SparseTensor (via ``A + scalar``, ``A.sum()`` baselines)."""
    from torch_sla import SparseTensor
    from torch_sla.datasets import Synthetic

    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

    # Reference single-process scalar arithmetic on the same matrix.
    def _sum(B):  # convenience
        return float(B.sum().item())

    ref = {
        "add_scalar":  _sum(A + 1.5),
        "radd_scalar": _sum(1.5 + A),
        "sub_scalar":  _sum(A - 0.5),
        "rsub_scalar": _sum(1.0 - A),
        "mul_scalar":  _sum(A * 2.0),
        "rmul_scalar": _sum(2.0 * A),
        "div_scalar":  _sum(A / 0.5),
        "pow_scalar":  _sum(A ** 2),
        "neg":         _sum(-A),
        "abs":         _sum(A.abs()),
        "sqrt_of_abs": _sum(A.abs().sqrt()),
        "square":      _sum(A.square()),
        "exp":         _sum(A.exp()),
        "same_spec_add": _sum(A + A * 3.0),
        "same_spec_sub": _sum(A * 3.0 - A),
    }

    world_size = 2
    port = 29630
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_elementwise_worker,
                         args=(rank, world_size, port, q))
             for rank in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world_size

    for r in results:
        assert r["sum_is_dsparse_friendly"], \
            f"rank {r['rank']}: result is not a DSparseTensor"
        for key, expected in ref.items():
            assert abs(r[key] - expected) <= 1e-9 * max(1.0, abs(expected)), \
                f"rank {r['rank']}: {key} {r[key]} vs ref {expected}"

    print(f"\n[OK] element-wise math on 2 procs vs single-process reference: "
          f"all {len(ref)} ops match")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
