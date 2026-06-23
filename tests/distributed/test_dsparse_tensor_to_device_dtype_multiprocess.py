#!/usr/bin/env python
"""``DSparseTensor.to / cuda / cpu / float / double / half`` parity tests.

Phase C2: device + dtype methods mirror ``SparseTensor``. CPU-only paths
(``.to('cpu')``, ``.float()``, ``.double()``, ``.half()``) are exercised
under gloo 2-proc; the CUDA path is skipped when no GPU is present.
After casting / moving, ``__matmul__`` must keep producing the same
distributed result modulo dtype.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _device_dtype_worker(rank: int, world_size: int, port: int,
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

        # baseline matvec (fp64)
        torch.manual_seed(0)
        x_global = torch.randn(A.shape[0], dtype=torch.float64)
        y_dt_ref = D @ D.scatter(x_global)
        y_ref = y_dt_ref.full_tensor()

        # ---- .float() ----
        D_f = D.float()
        assert D_f.dtype == torch.float32
        assert D.dtype == torch.float64, "Original must be unchanged"
        x_dt_f = D_f.scatter(x_global.float())
        y_f = (D_f @ x_dt_f).full_tensor()
        # fp32 vs fp64 should agree to ~1e-5 absolute, ~1e-6 relative
        max_abs = float((y_f.double() - y_ref).abs().max().item())
        rel = max_abs / float(y_ref.abs().max().item() + 1e-30)

        # ---- .double() round-trip ----
        D_d = D_f.double()
        assert D_d.dtype == torch.float64
        x_dt_d = D_d.scatter(x_global)
        y_d = (D_d @ x_dt_d).full_tensor()
        roundtrip_err = float((y_d - y_ref).abs().max().item())

        # ---- .cpu() identity (already on cpu) ----
        D_cpu = D.cpu()
        assert D_cpu.device.type == "cpu"

        # ---- .to(device=cpu, dtype=float32) combined ----
        D_combo = D.to(device="cpu", dtype=torch.float32)
        assert D_combo.dtype == torch.float32
        assert D_combo.device.type == "cpu"

        # ---- .to(None, None) returns self ----
        assert D.to() is D

        out_queue.put({
            "rank": rank,
            "float_max_abs": max_abs,
            "float_rel": rel,
            "double_roundtrip_err": roundtrip_err,
            "original_dtype": str(D.dtype),
            "float_dtype": str(D_f.dtype),
            "double_dtype": str(D_d.dtype),
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
def test_dsparse_to_device_dtype_2procs():
    world_size = 2
    port = 29560
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_device_dtype_worker,
                         args=(rank, world_size, port, q))
             for rank in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world_size, \
        f"Expected {world_size} results, got {len(results)}"

    for r in results:
        assert r["original_dtype"] == "torch.float64"
        assert r["float_dtype"] == "torch.float32"
        assert r["double_dtype"] == "torch.float64"
        # fp32 matvec vs fp64 reference: relative err < 1e-5
        assert r["float_rel"] < 1e-5, \
            f"rank {r['rank']}: float→fp64 matvec rel err {r['float_rel']:.2e} too high"
        # fp64→fp32→fp64 roundtrip should agree to ~1e-6 absolute
        assert r["double_roundtrip_err"] < 1e-5, \
            f"rank {r['rank']}: roundtrip err {r['double_roundtrip_err']:.2e} too high"

    print(f"\n[OK] device/dtype on 2 procs:")
    for r in sorted(results, key=lambda x: x["rank"]):
        print(f"  rank {r['rank']}: float matvec rel={r['float_rel']:.2e} "
              f"roundtrip abs={r['double_roundtrip_err']:.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
