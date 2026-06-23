#!/usr/bin/env python
"""End-to-end check for :func:`gather_owned_to_global`.

Each rank holds a non-overlapping slice of a global index range and a
value tensor; the helper must round-trip them into a length-N global
vector identical to a single-process reference.

Also covers an uneven-partition case (rank 0 owns more than rank 1)
since :func:`all_gather_into_tensor` is sensitive to non-uniform sizes
-- the helper pads internally to dodge that.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(rank: int, world_size: int, port: int,
             scheme: str, out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch_sla.distributed import gather_owned_to_global

        N_global = 17  # prime, makes the uneven split unmistakable
        full = torch.arange(N_global, dtype=torch.float64) * 0.5 + 1.0

        if scheme == "even":
            # Round-robin: rank r owns indices {r, r+world, r+2*world, ...}.
            owned = torch.arange(rank, N_global, world_size, dtype=torch.long)
        else:  # "uneven" -- rank 0 takes the first chunk, rank 1 the rest.
            cut = 11 if rank == 0 else N_global
            lo = 0 if rank == 0 else 11
            owned = torch.arange(lo, cut, dtype=torch.long)

        val_owned = full[owned].clone()
        recovered = gather_owned_to_global(owned, val_owned, N_global)

        out_queue.put((rank, "ok",
                        torch.allclose(recovered, full),
                        float((recovered - full).abs().max().item())))
    except Exception as e:  # noqa: BLE001
        import traceback
        out_queue.put((rank, "err", str(e), traceback.format_exc()))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scheme", ["even", "uneven"])
def test_gather_owned_to_global_2procs(scheme):
    world = 2
    port = 29512 if scheme == "even" else 29513
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                          args=(r, world, port, scheme, q))
             for r in range(world)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world, f"missing ranks: got {results}"
    for r in results:
        assert r[1] == "ok", f"rank {r[0]} crashed: {r[2:]}"
        assert r[2], f"rank {r[0]} mismatch, max diff {r[3]}"


if __name__ == "__main__":
    test_gather_owned_to_global_2procs("even")
    test_gather_owned_to_global_2procs("uneven")
    print("OK")
