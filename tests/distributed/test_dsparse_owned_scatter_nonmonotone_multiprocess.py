#!/usr/bin/env python
"""Regression: owned-aware Shard(0) round-trip under NON-MONOTONE partitions.

A plain ``DTensor[Shard(0)]`` reconstructs the global vector in
``full_tensor()`` by concatenating each rank's local slice in rank
order, IGNORING ``Partition.owned_nodes``. For geometric / graph
partitions (``rcb`` / ``hilbert`` / real ``metis``) the owned node-ids
are NOT globally sorted in rank order, so that concatenation returns a
silently *permuted* vector.

This test partitions a ``laplacian_2d`` problem with ``rcb`` (a
genuinely non-monotone partition) under ``world=4`` and asserts:

  (a) ``D.scatter(x).full_tensor()`` round-trips ``x`` to ~0 error,
      and the same holds for the distributed matvec result
      ``(D @ D.scatter(x)).full_tensor()`` vs the dense reference;
  (b) distributed ``eigsh(which="SA")`` recovers the analytical
      Laplacian spectrum -- in particular POSITIVE eigenvalues for the
      SPD Laplacian (the buggy path returned NEGATIVE eigenvalues).

Before the fix this test FAILS (permuted round-trip / negative
eigenvalues); after the fix it PASSES.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _coords_2d(m: int) -> torch.Tensor:
    ii, jj = torch.meshgrid(torch.arange(m), torch.arange(m), indexing="ij")
    return torch.stack([ii.flatten(), jj.flatten()], dim=1).to(torch.float64)


def _worker(rank: int, world_size: int, port: int, m: int, k: int,
            out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    res = {"rank": rank}
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch_sla import SparseTensor
        from torch_sla.distributed import DSparseTensor
        import torch_sla.datasets as d
        import scipy.sparse as sp

        torch.manual_seed(0)
        mesh = init_device_mesh("cpu", (world_size,))
        coords = _coords_2d(m)

        # --- SPD Laplacian (matvec + round-trip + eigsh) ---
        lap = d.laplacian_2d(m)
        lv, lr, lc, lshape = lap.coo()
        N = lshape[0]
        AL = SparseTensor(lv, lr, lc, lshape)
        DL = DSparseTensor.partition(
            AL, mesh, partition_method="rcb", coords=coords)

        # Confirm this partition is genuinely NON-MONOTONE for this rank,
        # i.e. owned_nodes are not a sorted contiguous block in rank
        # order -- otherwise the test would pass by coincidence and not
        # exercise the bug.
        owned = DL._spec.placement.partition.owned_nodes
        res["owned_is_sorted_contig"] = bool(
            torch.equal(owned, torch.arange(int(owned.min()),
                                            int(owned.max()) + 1))
        ) if owned.numel() > 0 else True

        AL_sp = sp.coo_matrix(
            (lv.numpy(), (lr.numpy(), lc.numpy())), shape=lshape).tocsr()

        # (a) scatter round-trip: x -> owned slice -> global vector == x
        x_global = torch.randn(N, dtype=torch.float64)
        x_round = DL.scatter(x_global).full_tensor()
        res["roundtrip_err"] = float((x_round - x_global).norm().item())

        # (a') distributed matvec reconstructs the dense reference
        y_ref = torch.from_numpy(AL_sp @ x_global.numpy())
        y_full = (DL @ DL.scatter(x_global)).full_tensor()
        res["matvec_err"] = float((y_full - y_ref).norm().item())

        # (b) distributed eigsh smallest eigenvalues vs analytic spectrum
        evals, _ = DL.eigsh(k=k, which="SA", maxiter=500, tol=1e-9)
        evals = torch.sort(evals.real).values
        ana = torch.sort(d.laplacian_2d_eigenvalues(m)).values[:k]
        res["eigsh_err"] = float((evals - ana).abs().max().item())
        res["eigvals"] = evals.tolist()
        res["min_eigval"] = float(evals.min().item())
    except Exception:
        import traceback
        res["exception"] = traceback.format_exc()
    finally:
        out_queue.put(res)
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(),
                    reason="torch.distributed not available")
def test_owned_scatter_roundtrip_and_eigsh_rcb_world4():
    world_size = 4
    port = 29688
    m = 16
    k = 4

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_worker,
                         args=(rank, world_size, port, m, k, q))
             for rank in range(world_size)]
    for p in procs:
        p.start()
    results = [q.get(timeout=240) for _ in range(world_size)]
    for p in procs:
        p.join(timeout=240)

    exc = [r for r in results if "exception" in r]
    assert not exc, exc[0]["exception"]
    assert len(results) == world_size

    # The rcb partition must be non-monotone on at least one rank,
    # otherwise the regression wouldn't actually exercise the bug.
    assert not all(r["owned_is_sorted_contig"] for r in results), \
        "rcb partition is rank-monotone here; test does not exercise the bug"

    for r in results:
        # (a) scatter round-trip is exact
        assert r["roundtrip_err"] < 1e-12, \
            f"rank {r['rank']}: scatter round-trip err {r['roundtrip_err']:.3e}"
        # (a') matvec matches dense reference
        assert r["matvec_err"] < 1e-9, \
            f"rank {r['rank']}: matvec err {r['matvec_err']:.3e}"
        # (b) eigsh matches analytic spectrum and is POSITIVE (SPD)
        assert r["eigsh_err"] < 1e-6, \
            f"rank {r['rank']}: eigsh err {r['eigsh_err']:.3e}"
        assert r["min_eigval"] > 0.0, \
            f"rank {r['rank']}: non-positive eigenvalue {r['min_eigval']:.6f}"

    print("\n[OK] rcb world=4 owned-scatter round-trip + eigsh:")
    print(f"  roundtrip_err max = {max(r['roundtrip_err'] for r in results):.2e}")
    print(f"  matvec_err    max = {max(r['matvec_err'] for r in results):.2e}")
    print(f"  eigsh_err     max = {max(r['eigsh_err'] for r in results):.2e}")
    print(f"  eigvals (rank0)   = {[f'{v:.6f}' for v in results[0]['eigvals']]}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
