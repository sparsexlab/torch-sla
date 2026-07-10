"""Regression tests for sparse-axis reductions.

Covers two bugs:
  * ``sum``/``mean`` with ``keepdim=True`` over a sparse axis dropped the
    reduced axis instead of keeping it as size 1.
  * ``mean`` over a single sparse axis divided every slice by the *total*
    nnz instead of that slice's own nnz count.
"""
from __future__ import annotations

import torch

from torch_sla import SparseTensor


def _mat():
    #  [[1, 2, 0],
    #   [0, 3, 4],
    #   [5, 0, 6]]
    return SparseTensor.from_dense(
        torch.tensor([[1.0, 2, 0], [0, 3, 4], [5, 0, 6]])
    )


def test_sum_keepdim_shapes():
    A = _mat()
    assert tuple(A.sum(axis=1, keepdim=True).shape) == (3, 1)
    assert tuple(A.sum(axis=0, keepdim=True).shape) == (1, 3)
    assert tuple(A.sum(axis=(0, 1), keepdim=True).shape) == (1, 1)
    # keepdim=False unchanged
    assert tuple(A.sum(axis=1).shape) == (3,)


def test_sum_keepdim_values_match_flat():
    A = _mat()
    flat = A.sum(axis=1)
    kept = A.sum(axis=1, keepdim=True)
    assert torch.allclose(kept.squeeze(-1), flat)
    assert torch.allclose(flat, torch.tensor([3.0, 7.0, 11.0]))


def test_mean_over_sparse_axis_uses_per_slice_nnz():
    A = _mat()
    # per-row mean of stored values: [ (1+2)/2, (3+4)/2, (5+6)/2 ]
    assert torch.allclose(A.mean(axis=1), torch.tensor([1.5, 3.5, 5.5]))
    # per-column mean of stored values: [ (1+5)/2, (2+3)/2, (4+6)/2 ]
    assert torch.allclose(A.mean(axis=0), torch.tensor([3.0, 2.5, 5.0]))


def test_mean_over_both_sparse_axes_is_mean_of_nnz():
    A = _mat()
    # 6 stored values summing to 21 -> 3.5
    assert torch.isclose(A.mean(axis=(0, 1)), torch.tensor(3.5))
    assert torch.isclose(A.mean(), torch.tensor(3.5))


def test_mean_keepdim_shape():
    A = _mat()
    assert tuple(A.mean(axis=1, keepdim=True).shape) == (3, 1)


def test_mean_empty_slice_is_zero():
    # Row 0 has no stored entries -> its mean is 0 (not NaN).
    A = SparseTensor.from_dense(
        torch.tensor([[0.0, 0, 0], [0, 3, 4], [5, 0, 6]])
    )
    out = A.mean(axis=1)
    assert torch.isfinite(out).all()
    assert torch.isclose(out[0], torch.tensor(0.0))
