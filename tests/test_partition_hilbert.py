"""Unit tests for the Hilbert space-filling curve partitioner.

No multi-process / no torch.distributed required -- the partitioner is
a pure function from ``(coords, num_parts)`` to ``partition_ids``."""
from __future__ import annotations

import pytest
import torch

from torch_sla.distributed import (
    partition_coordinates,
    _hilbert_sort_indices,
    _hilbert_index_from_axes,
)


# ---------------------------------------------------------------------- #
# 2-D regular grid -- the canonical PDE/mesh case
# ---------------------------------------------------------------------- #
def _grid_coords(n: int, d: int = 2) -> torch.Tensor:
    if d == 2:
        ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        return torch.stack([xs.flatten(), ys.flatten()], dim=1).float()
    if d == 3:
        zs, ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n),
                                     torch.arange(n), indexing="ij")
        return torch.stack([xs.flatten(), ys.flatten(),
                             zs.flatten()], dim=1).float()
    raise ValueError(d)


def test_hilbert_partition_2d_grid_equal_sized():
    """Hilbert partitioner on an n×n grid into k parts (n²/k divides) must
    produce equal-sized partitions."""
    n = 8
    k = 4
    coords = _grid_coords(n)
    parts = partition_coordinates(coords, num_parts=k, method="hilbert")
    sizes = [int((parts == p).sum().item()) for p in range(k)]
    assert sizes == [n * n // k] * k, sizes


def test_hilbert_partition_preserves_locality_better_than_random():
    """Sanity: a 2-D Hilbert partition of a grid should give each
    partition a *small* bounding box compared to a random partition.
    Quantifies that the curve actually respects geometry."""
    n = 16
    k = 4
    coords = _grid_coords(n)

    parts_hilbert = partition_coordinates(coords, num_parts=k, method="hilbert")
    # Random partition as baseline
    torch.manual_seed(0)
    parts_random = torch.randint(0, k, (n * n,))

    def avg_bbox_area(parts):
        total = 0.0
        for p in range(k):
            mask = parts == p
            if mask.sum() == 0:
                continue
            sub = coords[mask]
            box = (sub.max(0).values - sub.min(0).values).prod().item()
            total += box
        return total / k

    h_area = avg_bbox_area(parts_hilbert)
    r_area = avg_bbox_area(parts_random)
    # Random partitions span the full grid; Hilbert chunks span ~1/k of
    # the grid linear dimensions in each axis, so bbox-area is way smaller.
    assert h_area < r_area / 2, \
        f"Hilbert bbox area {h_area:.1f} not significantly smaller than " \
        f"random {r_area:.1f}"


def test_hilbert_partition_3d_grid_equal_sized():
    n = 4
    k = 2
    coords = _grid_coords(n, d=3)
    parts = partition_coordinates(coords, num_parts=k, method="hilbert")
    sizes = [int((parts == p).sum().item()) for p in range(k)]
    assert sizes == [n * n * n // k] * k, sizes


def test_hilbert_sort_indices_returns_permutation():
    n = 16
    coords = _grid_coords(n)
    idx = _hilbert_sort_indices(coords)
    # Every grid point appears exactly once
    assert sorted(idx.tolist()) == list(range(n * n))


def test_hilbert_index_distinct_on_unique_axes():
    """Distinct quantised coordinates produce distinct Hilbert indices."""
    order = 4
    seen = set()
    for x in range(1 << order):
        for y in range(1 << order):
            h = _hilbert_index_from_axes([x, y], order)
            assert h not in seen, (x, y, h)
            seen.add(h)
    assert len(seen) == (1 << (2 * order))


# ---------------------------------------------------------------------- #
# Errors
# ---------------------------------------------------------------------- #
def test_hilbert_rejects_unsupported_dim():
    bad = torch.randn(10, 5)  # 5-D coords
    with pytest.raises(ValueError, match="2-D / 3-D"):
        _hilbert_sort_indices(bad)


def test_hilbert_rejects_non_2d_input():
    bad = torch.randn(10)
    with pytest.raises(ValueError, match="coords must be 2-D"):
        _hilbert_sort_indices(bad)
