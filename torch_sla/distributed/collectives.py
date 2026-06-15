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

    The implementation pads to ``max(owned.numel())`` across ranks
    (so ``all_gather_into_tensor`` sees a uniform shape), concatenates
    every rank's pad into one contiguous buffer, masks out the padding
    entries with a single vectorised comparison, then does ONE
    ``index_put_`` -- no Python loop over ranks.

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

    # all_gather_into_tensor needs contiguous output sized world * input.
    # Gather sizes first so we know how much padding to drop later.
    size_local = torch.tensor([owned.numel()], dtype=torch.long, device=device)
    all_sizes = torch.zeros(world, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(all_sizes, size_local)

    max_n = int(all_sizes.max().item())
    if max_n == 0:
        return torch.zeros(N_global, dtype=dtype, device=device)

    idx_pad = torch.zeros(max_n, dtype=torch.long, device=device)
    val_pad = torch.zeros(max_n, dtype=dtype, device=device)
    idx_pad[: owned.numel()] = owned.to(torch.long)
    val_pad[: val.numel()] = val

    big_idx = torch.empty(world * max_n, dtype=torch.long, device=device)
    big_val = torch.empty(world * max_n, dtype=dtype, device=device)
    dist.all_gather_into_tensor(big_idx, idx_pad)
    dist.all_gather_into_tensor(big_val, val_pad)

    # Vectorised valid-position mask: position p on rank r is real iff
    # p < all_sizes[r]. Lay out as (world*max_n,) and compare elementwise.
    positions = torch.arange(max_n, device=device).repeat(world)
    rank_of_pos = torch.arange(world, device=device).repeat_interleave(max_n)
    valid = positions < all_sizes[rank_of_pos]

    out = torch.zeros(N_global, dtype=dtype, device=device)
    out.index_put_((big_idx[valid],), big_val[valid])
    return out
