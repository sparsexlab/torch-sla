#!/usr/bin/env python
"""Factory classmethods on ``SparseTensor`` -- ``eye / diag / tridiagonal``.

Verifies shape / nnz / symmetry / matvec equivalence vs the explicit
COO construction the helpers replace.
"""
from __future__ import annotations

import os
import sys

import torch
import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_eye_basic():
    from torch_sla import SparseTensor

    I = SparseTensor.eye(8)
    assert I.shape == (8, 8)
    assert I.nnz == 8
    assert I.dtype == torch.float64
    x = torch.randn(8, dtype=torch.float64)
    assert torch.equal(I @ x, x)


def test_eye_dtype_device():
    from torch_sla import SparseTensor

    I = SparseTensor.eye(4, dtype=torch.float32)
    assert I.dtype == torch.float32


def test_diag_basic():
    from torch_sla import SparseTensor

    vals = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float64)
    D = SparseTensor.diag(vals)
    assert D.shape == (4, 4)
    assert D.nnz == 4
    x = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.equal(D @ x, vals)


def test_diag_rejects_2d():
    from torch_sla import SparseTensor

    with pytest.raises(ValueError, match="1-D tensor"):
        SparseTensor.diag(torch.eye(3))


def test_tridiagonal_replicates_manual_pattern():
    """The very pattern from the distributed examples should reduce to
    one line with identical numerics."""
    from torch_sla import SparseTensor

    n = 200
    A_new = SparseTensor.tridiagonal(n, diag=4.0, off_diag=-1.0)

    # The 7-line manual version.
    idx = torch.arange(n)
    val = torch.cat([
        torch.full((n,), 4.0, dtype=torch.float64),
        torch.full((n - 1,), -1.0, dtype=torch.float64),
        torch.full((n - 1,), -1.0, dtype=torch.float64),
    ])
    row = torch.cat([idx, idx[1:], idx[:-1]])
    col = torch.cat([idx, idx[:-1], idx[1:]])
    A_old = SparseTensor(val, row, col, shape=(n, n))

    x = torch.randn(n, dtype=torch.float64)
    assert torch.equal(A_new @ x, A_old @ x)
    assert A_new.shape == A_old.shape
    assert A_new.nnz == A_old.nnz
    assert bool(A_new.is_symmetric().item())


def test_tridiagonal_tensor_inputs():
    from torch_sla import SparseTensor

    n = 6
    diag = torch.arange(1.0, n + 1, dtype=torch.float64)
    off = -torch.arange(1.0, n, dtype=torch.float64)
    A = SparseTensor.tridiagonal(n, diag=diag, off_diag=off)
    assert A.shape == (n, n)
    assert A.nnz == n + 2 * (n - 1)

    # Build the dense reference and compare matvec.
    dense = torch.diag(diag) + torch.diag(off, 1) + torch.diag(off, -1)
    x = torch.randn(n, dtype=torch.float64)
    assert torch.allclose(A @ x, dense @ x)


def test_tridiagonal_rejects_wrong_shape():
    from torch_sla import SparseTensor

    with pytest.raises(ValueError, match=r"diag tensor must have shape"):
        SparseTensor.tridiagonal(5, diag=torch.zeros(4), off_diag=-1.0)
    with pytest.raises(ValueError, match=r"off_diag tensor must have shape"):
        SparseTensor.tridiagonal(5, diag=2.0,
                                  off_diag=torch.zeros(10))


def test_tridiagonal_dtype_propagates():
    from torch_sla import SparseTensor

    A = SparseTensor.tridiagonal(10, diag=4.0, off_diag=-1.0,
                                  dtype=torch.float32)
    assert A.dtype == torch.float32


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
