"""Distributed graph algorithms for
:class:`~torch_sla.distributed.DSparseTensor`.

Currently: distributed connected components via label propagation with
boundary-label halo exchange.

The single-process algorithm (``torch_sla.sparse_tensor.graph``) is a
Shiloach-Vishkin label-propagation: every node starts with its own
global id as label and the minimum label is propagated across every
edge until the labelling stops changing. The distributed version keeps
that idea but runs it on each rank's local subdomain (owned rows +
halo columns) and exchanges the boundary labels across ranks each
sweep, so a component spanning several shards eventually agrees on a
single global minimum label.

Correctness relies on the same structural-symmetry assumption used by
the transpose matvec (an edge ``i->j`` implies ``j->i`` in the
sparsity pattern); ``connected_components`` treats the matrix as an
undirected graph, so this always holds after symmetrising the local
adjacency.
"""
from __future__ import annotations

from typing import Tuple

import torch

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

from .matvec import halo_exchange


def connected_components_shard(D) -> Tuple[torch.Tensor, int]:
    """Distributed connected components.

    Returns ``(labels_owned, num_components)`` where ``labels_owned`` is
    this rank's owned-slice of the contiguous component labelling
    (``0..num_components-1``, with component ids assigned by global
    minimum node id so the result is identical to the single-process /
    scipy labelling) and ``num_components`` is the global component
    count (identical on every rank).

    Algorithm (label propagation with boundary halo exchange):

    1. ``lab[k] = global_id(local_node k)`` for owned + halo slots.
    2. Repeat until the global label-sum stabilises:
       a. propagate the per-edge minimum over the local (symmetrised)
          adjacency: ``lab[i] = min(lab[i], lab[j])`` for every local
          edge ``(i, j)``;
       b. halo-exchange ``lab`` so each rank's halo slots carry the
          owner's current label;
       c. ``all_reduce(SUM)`` the local label-sum -> global convergence
          test.
    3. Globally renumber the surviving root labels to ``0..C-1``.
    """
    partition = D._spec.placement.partition
    if partition is None:
        raise RuntimeError("connected_components requires a partition")

    device = D._local_tensor.values.device
    num_owned = int(partition.owned_nodes.numel())
    num_local = int(partition.local_to_global.numel())
    l2g = partition.local_to_global.to(device=device, dtype=torch.int64)

    # Build the local symmetric edge list in LOCAL coords. The local
    # tensor's rows are owned (0..num_owned), cols span owned+halo. To
    # treat the graph as undirected we add both (r, c) and (c, r); the
    # halo->owned direction is what carries a neighbour's label into our
    # owned nodes after the halo exchange refreshes halo labels.
    st = D._local_tensor
    r = st.row_indices.to(torch.int64)
    c = st.col_indices.to(torch.int64)
    mask = r != c                       # drop self loops
    r, c = r[mask], c[mask]
    src = torch.cat([r, c])
    dst = torch.cat([c, r])

    # Labels indexed by LOCAL node id; initial label = global id.
    lab = l2g.clone()

    def _propagate(labels: torch.Tensor) -> torch.Tensor:
        out = labels.clone()
        if src.numel() > 0:
            out.scatter_reduce_(0, dst, labels[src], reduce="amin",
                                include_self=True)
        return out

    prev_sum = None
    max_sweeps = 10 * num_local + 50
    for _ in range(max_sweeps):
        lab = _propagate(lab)
        # Refresh halo labels from their owners (forward halo exchange).
        # halo_exchange operates on float buffers internally via
        # index_select/index_copy_; labels are integer so we run it on
        # the int64 tensor directly (index ops are dtype-agnostic).
        halo_exchange(D, lab, partition)

        local_sum = lab[:num_owned].sum().to(torch.float64)
        red = local_sum.clone()
        if _DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(red, op=dist.ReduceOp.SUM)
        cur = float(red.item())
        if prev_sum is not None and cur == prev_sum:
            break
        prev_sum = cur

    # ``lab[:num_owned]`` now holds each owned node's root global id.
    owned_roots = lab[:num_owned].clone()

    # Global contiguous renumbering: gather the set of distinct root ids
    # across all ranks, sort, map root -> dense index. Done identically
    # on every rank so labels agree globally.
    if _DIST_AVAILABLE and dist.is_initialized():
        from .collectives import gather_owned_to_global
        owned = partition.owned_nodes.to(device=device, dtype=torch.int64)
        N = int(D.shape[0])
        # Assemble the global root vector on every rank.
        roots_global = gather_owned_to_global(
            owned, owned_roots.to(torch.float64), N).to(torch.int64)
        unique = torch.unique(roots_global)            # sorted ascending
        # Map each owned root to its dense id.
        remap = torch.searchsorted(unique, owned_roots)
        num_components = int(unique.numel())
        return remap.to(torch.long), num_components

    # Single-process fallback.
    unique, inverse = torch.unique(owned_roots, return_inverse=True)
    return inverse.to(torch.long), int(unique.numel())
