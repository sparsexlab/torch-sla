#!/usr/bin/env python
"""Distributed connected components on ``DSparseTensor``.

Label-propagation (Shiloach-Vishkin style) with boundary-label halo
exchange: every node starts with its own global id, the per-edge
minimum label is propagated over the local subdomain, halo labels are
refreshed from their owners each sweep, and a global ``all_reduce``
label-sum detects convergence. Components spanning several shards
agree on a single global-minimum root label.

The test graph is a **multi-component path forest** whose components
are interleaved across the node ordering (node ``i`` -> component
``i % ncomp``), so a contiguous ``simple`` partition splits every
component across all ranks -- the cross-rank propagation is genuinely
exercised (verified: every component spans every partition at
world=4). Component labels are compared to ``scipy.sparse.csgraph``
as a partition (grouping-invariant to label permutation).

Verified rank-count invariant (world 2 and 4) and a non-monotone
``rcb`` partition.

Run with::

    pytest tests/distributed/test_dsparse_connected_components_multiprocess.py -v
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _build_multicomponent(N: int, ncomp: int):
    """Interleaved multi-component path forest as a COO triple.

    Node ``i`` belongs to component ``i % ncomp``; nodes of a component
    are chained into an undirected path. Self loops make every node a
    row so the matrix is well formed."""
    comp = torch.arange(N) % ncomp
    rows, cols = [], []
    for c in range(ncomp):
        nodes = (comp == c).nonzero().squeeze(1).tolist()
        for a, b in zip(nodes[:-1], nodes[1:]):
            rows += [a, b]
            cols += [b, a]
    for i in range(N):
        rows.append(i)
        cols.append(i)
    row = torch.tensor(rows, dtype=torch.int64)
    col = torch.tensor(cols, dtype=torch.int64)
    val = torch.ones(row.numel(), dtype=torch.float64)
    return val, row, col, (N, N), comp


def _canon(labels: torch.Tensor) -> torch.Tensor:
    """Canonicalise a labelling to first-occurrence order so two
    labellings of the same partition compare equal regardless of the
    actual label values."""
    seen = {}
    out = torch.empty_like(labels)
    for i, v in enumerate(labels.tolist()):
        if v not in seen:
            seen[v] = len(seen)
        out[i] = seen[v]
    return out


def _cc_worker(rank, world_size, port, part, N, ncomp, out_queue):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from torch_sla import DSparseTensor, SparseTensor
        from torch_sla.distributed.collectives import gather_owned_to_global

        val, row, col, shape, comp = _build_multicomponent(N, ncomp)
        A = SparseTensor(val, row, col, shape)
        mesh = init_device_mesh("cpu", (world_size,))
        if part == "rcb":
            # 2-D folded coords (node i at (i mod 12, i div 12)) so the
            # recursive-coordinate-bisection cuts interleave node ids
            # across ranks -> a genuinely NON-MONOTONE owned set. A coord
            # whose first axis is arange(N) would bisect into contiguous
            # blocks and reproduce the ``simple`` partition, never
            # exercising the boundary-label halo exchange between
            # non-contiguous owned slices.
            idx = torch.arange(N)
            coords = torch.stack(
                [(idx % 12).to(torch.float64),
                 (idx // 12).to(torch.float64)], dim=1)
            D = DSparseTensor.partition(
                A, mesh, partition_method="rcb", coords=coords)
        else:
            D = DSparseTensor.partition(A, mesh, partition_method=part)

        labels_owned, ncc = D.connected_components_distributed_shard()
        partition = D._spec.placement.partition
        owned = partition.owned_nodes.to(torch.int64)
        labels_g = gather_owned_to_global(
            owned, labels_owned.to(torch.float64), N).to(torch.int64)

        import scipy.sparse as sp
        from scipy.sparse.csgraph import connected_components as scc
        A_sp = sp.coo_matrix(
            (val.numpy(), (row.numpy(), col.numpy())), shape=shape).tocsr()
        n_ref, lab_ref = scc(A_sp, directed=False)
        lab_ref = torch.from_numpy(lab_ref)
        match = bool(torch.equal(_canon(labels_g), _canon(lab_ref))) and ncc == n_ref

        # ``owned_nodes`` is always sorted ASCENDING within a rank
        # (built via nonzero()); "non-monotone" means the owned set is
        # not a single contiguous arange block, i.e. rank-order
        # concatenation would permute the global vector.
        nonmono = (
            not bool(torch.equal(
                owned,
                torch.arange(int(owned.min()), int(owned.max()) + 1)))
            if owned.numel() > 1 else False
        )

        out_queue.put({"rank": rank, "ncc": ncc, "n_ref": int(n_ref),
                       "match": match, "nonmono": nonmono})
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("part,world_size,port", [
    ("simple", 2, 29631),
    ("simple", 4, 29632),
    ("rcb",    4, 29633),
])
@pytest.mark.skipif(
    not hasattr(dist, "is_available") or not dist.is_available(),
    reason="torch.distributed not available",
)
def test_shard_connected_components_matches_scipy(part, world_size, port):
    """Distributed connected components must reproduce scipy.csgraph's
    component partition on a multi-component graph whose components are
    split across every rank. Rank-count invariant + non-monotone rcb."""
    N, ncomp = 60, 5
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    procs = []
    try:
        for rank in range(world_size):
            p = ctx.Process(target=_cc_worker,
                            args=(rank, world_size, port, part, N, ncomp,
                                  out_queue))
            p.start()
            procs.append(p)
        results = [out_queue.get(timeout=180) for _ in range(world_size)]
        for p in procs:
            p.join(timeout=180)
            assert p.exitcode == 0, f"rank exited with {p.exitcode}"

        for r in results:
            assert r["ncc"] == r["n_ref"] == ncomp, \
                f"{part}/rank {r['rank']}: ncc={r['ncc']} ref={r['n_ref']}"
            assert r["match"], \
                f"{part}/rank {r['rank']}: labelling differs from scipy"
        if part == "rcb":
            assert any(r["nonmono"] for r in results), \
                "rcb partition is rank-monotone; non-monotone path not exercised"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)


if __name__ == "__main__":
    test_shard_connected_components_matches_scipy("rcb", 4, 29633)
    print("OK")
