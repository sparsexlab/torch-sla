"""Distributed collective utilities for owned-slice <-> global-vector
transforms.

Every DSparseTensor row-shard holds an irregular subset of global node
ids in ``Partition.owned_nodes``; assembling per-rank computed values
into a length-``N_global`` vector across the world is a routine
operation (residual checks, allgather-for-comparison, dense matvec
emulation, recovery in PDE-condensed test harnesses).

The naive implementation pads to ``N_global``, runs three
``dist.all_gather`` calls, then loops in Python to scatter each rank's
slice back -- which is what the codebase used to do (closure inside
:meth:`DSparseTensor.det`). This module replaces that loop with a
single :func:`torch.Tensor.index_put_` over the concatenated buffers
returned by ``all_gather_into_tensor``.
"""
from __future__ import annotations

import torch

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


def gather_owned_to_global(
    owned: torch.Tensor,
    val: torch.Tensor,
    N_global: int,
) -> torch.Tensor:
    """Gather per-rank ``(owned_index, value)`` pairs into a global vector.

    Each rank holds:

    * ``owned`` -- ``int64`` tensor of global node ids it owns
      (typically ``Partition.owned_nodes``)
    * ``val`` -- the corresponding values, shape ``[owned.numel()]``

    Returns a length-``N_global`` tensor materialised on **every rank**,
    with the gathered values scattered into their global positions
    (and zeros at any unfilled slot).

    The implementation pads to ``max(owned.numel())`` across ranks (so
    the gathers see a uniform shape), stacks every rank's pad into one
    ``(world, max_n)`` tensor, flattens it to a single buffer, masks
    out the padding entries with a single vectorised comparison, then
    does ONE ``index_put_`` -- no Python loop over ranks.

    Uses the list-based :func:`torch.distributed.all_gather` rather
    than ``all_gather_into_tensor`` because torch 2.1's Gloo backend
    raises ``no support for _allgather_base in Gloo process group``.
    NCCL supports both; the list API works on every backend.

    Falls back to a plain ``out[owned] = val`` when called outside a
    process group (e.g. unit tests on a single CPU).
    """
    device = val.device
    dtype = val.dtype

    if not (_DIST_AVAILABLE and dist.is_initialized()):
        out = torch.zeros(N_global, dtype=dtype, device=device)
        out[owned] = val
        return out

    world = dist.get_world_size()

    # Gather sizes first so we know how much padding to drop later.
    size_local = torch.tensor([owned.numel()], dtype=torch.long, device=device)
    all_sizes_list = [torch.zeros_like(size_local) for _ in range(world)]
    dist.all_gather(all_sizes_list, size_local)
    all_sizes = torch.cat(all_sizes_list)  # (world,)

    max_n = int(all_sizes.max().item())
    if max_n == 0:
        return torch.zeros(N_global, dtype=dtype, device=device)

    idx_pad = torch.zeros(max_n, dtype=torch.long, device=device)
    val_pad = torch.zeros(max_n, dtype=dtype, device=device)
    idx_pad[: owned.numel()] = owned.to(torch.long)
    val_pad[: val.numel()] = val

    all_idx_list = [torch.zeros_like(idx_pad) for _ in range(world)]
    all_val_list = [torch.zeros_like(val_pad) for _ in range(world)]
    dist.all_gather(all_idx_list, idx_pad)
    dist.all_gather(all_val_list, val_pad)
    # Stack -> (world, max_n), flatten to a contiguous (world*max_n,)
    # buffer suitable for the masked index_put_ below.
    big_idx = torch.stack(all_idx_list, dim=0).reshape(-1)
    big_val = torch.stack(all_val_list, dim=0).reshape(-1)

    # Vectorised valid-position mask: position p on rank r is real iff
    # p < all_sizes[r]. Lay out as (world*max_n,) and compare elementwise.
    positions = torch.arange(max_n, device=device).repeat(world)
    rank_of_pos = torch.arange(world, device=device).repeat_interleave(max_n)
    valid = positions < all_sizes[rank_of_pos]

    out = torch.zeros(N_global, dtype=dtype, device=device)
    out.index_put_((big_idx[valid],), big_val[valid])
    return out
