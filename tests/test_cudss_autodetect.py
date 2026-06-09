"""Tests for the cuDSS auto matrix-type detection.

Drives the public :meth:`SparseTensor.detect_matrix_type` method (the
API a user calls indirectly via ``A.solve(b, backend='cudss',
matrix_type='auto')``) across all three catalogued benchmark sources:

* ``SuiteSparse`` — real-world FE / CFD / structural matrices
* ``DIMACS10`` — graph Laplacians (irregular / scale-free sparsity)
* ``Synthetic`` — programmatic PDE stencils

The detector itself runs on pure PyTorch tensors and does not need
CUDA / nvmath-python; the integration with the actual cuDSS solve is
covered separately by a CUDA-gated test.
"""
from __future__ import annotations

import pytest
import torch

from torch_sla import SparseTensor


# =====================================================================
# Each catalogued benchmark, exercised through the public API
# =====================================================================

def test_detect_on_suitesparse(benchmark_suitesparse):
    """``SparseTensor.detect_matrix_type()`` matches the SuiteSparse
    catalogue's expected ``detected_kind`` (which may be more
    conservative than the true ``math_kind`` because Gershgorin is
    a sufficient but not necessary positive-definite test)."""
    b = benchmark_suitesparse
    A = SparseTensor(b.val, b.row, b.col, b.shape)
    got = A.detect_matrix_type()
    assert got == b.detected_kind, (
        f"{b.name}: expected {b.detected_kind!r}, got {got!r}"
    )


def test_detect_on_dimacs(benchmark_dimacs):
    """Every catalogued DIMACS10 graph Laplacian (``L + eps*I``) detects
    as ``spd`` -- it is strictly diagonally dominant by construction."""
    b = benchmark_dimacs
    A = SparseTensor(b.val, b.row, b.col, b.shape)
    got = A.detect_matrix_type()
    assert got == b.detected_kind == "spd", (
        f"{b.name}: expected spd, got {got!r}"
    )


def test_detect_on_synthetic(benchmark_synthetic):
    """Every catalogued Synthetic stencil matches its declared
    ``detected_kind``."""
    b = benchmark_synthetic
    A = SparseTensor(b.val, b.row, b.col, b.shape)
    got = A.detect_matrix_type()
    assert got == b.detected_kind, (
        f"{b.name}: expected {b.detected_kind!r}, got {got!r}"
    )


# =====================================================================
# Catalogue-level sanity
# =====================================================================

def test_catalogues_cover_every_mathematical_kind():
    """Across all three sources the catalogue exercises every distinct
    mathematical kind the detector might encounter
    (general, symmetric, spd, hpd)."""
    from torch_sla.datasets import SuiteSparse, DIMACS10, Synthetic
    kinds: set[str] = set()
    for registry in (SuiteSparse, DIMACS10, Synthetic):
        for key in registry:
            # Synthetic builds locally; SuiteSparse / DIMACS10 may
            # need network. We only read math_kind which is in the
            # catalogue tuple, so we can read it after lazy instantiation
            # but it's safer to read directly from the catalogue.
            pass
    # Read math_kinds from catalogue tuples without triggering downloads.
    sk = {v[2] for v in SuiteSparse.catalog().values()}
    # DIMACS10 entries are post-processed to L+eps*I so all are SPD.
    dk = {"spd"}
    sy = {v[2] for v in Synthetic.catalog().values()}
    kinds = sk | dk | sy
    assert {"spd", "hpd", "symmetric", "general"}.issubset(kinds), (
        f"missing kinds; collected: {kinds}"
    )


# =====================================================================
# Synthetic edge cases the catalogue doesn't cover
# =====================================================================

def _from_dense(A: torch.Tensor) -> SparseTensor:
    n = A.shape[0]
    mask = A != 0
    idx = mask.nonzero(as_tuple=False)
    row = idx[:, 0].contiguous()
    col = idx[:, 1].contiguous()
    val = A[row, col]
    return SparseTensor(val, row, col, (n, n))


def test_detect_empty_matrix_is_general():
    """An all-zero matrix has no diagonal -- Gershgorin says nothing,
    detector falls back to ``general``."""
    A = SparseTensor(
        torch.zeros(0, dtype=torch.float64),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        (3, 3),
    )
    assert A.detect_matrix_type() == "general"


def test_detect_strict_diag_dom_hpd():
    """Strictly diagonally dominant Hermitian -> ``hpd``."""
    A = _from_dense(torch.tensor([
        [10.+0j,   1.+1j, 0.+0j],
        [1.-1j,  12.+0j,  2.+0j],
        [0.+0j,   2.+0j, 11.+0j],
    ], dtype=torch.complex128))
    assert A.detect_matrix_type() == "hpd"


def test_detect_real_symmetric_indefinite():
    """Symmetric with a negative diagonal entry -> conservative
    ``symmetric`` (Gershgorin sees diag <= 0, can't claim SPD)."""
    A = _from_dense(torch.tensor([
        [-1., 2., 0.],
        [ 2., 3., 1.],
        [ 0., 1., 4.],
    ], dtype=torch.float64))
    assert A.detect_matrix_type() == "symmetric"


def test_detect_real_nonsymmetric_general():
    A = _from_dense(torch.tensor([
        [3., 1., 0.],
        [0., 2., 5.],
        [1., 0., 4.],
    ], dtype=torch.float64))
    assert A.detect_matrix_type() == "general"


def test_detect_complex_symmetric_not_hermitian():
    """``A = A^T`` but NOT ``A = A^H`` (diagonal not real) -> ``symmetric``."""
    A = _from_dense(torch.tensor([
        [1.+1j, 2.+3j, 0.],
        [2.+3j, 4.+1j, 5.],
        [0.,    5.,    6.],
    ], dtype=torch.complex128))
    assert A.detect_matrix_type() == "symmetric"


# =====================================================================
# Guard rails on the public method
# =====================================================================

def test_detect_rejects_batched():
    """Batched / block-sparse tensors raise -- matrix-type is only
    defined for a single 2-D matrix."""
    val = torch.ones(2, 3, dtype=torch.float64)
    row = torch.tensor([0, 1, 2])
    col = torch.tensor([0, 1, 2])
    A = SparseTensor(val, row, col, (2, 3, 3))
    with pytest.raises(ValueError, match="non-batched"):
        A.detect_matrix_type()
