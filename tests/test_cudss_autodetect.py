"""Tests for the cuDSS auto matrix-type detection, exercised through
:meth:`SparseTensor.detect_matrix_type` (the public API a user actually
calls when they do ``A.solve(b, backend='cudss', matrix_type='auto')``).

CUDA / nvmath-python are *not* required: the detection logic is pure
PyTorch and runs on CPU. The cuDSS forward solve itself is covered
separately by an integration test gated on ``torch.cuda.is_available()``.

The detection is exercised against real SuiteSparse Matrix Collection
matrices via the ``benchmark`` fixture (every catalogued kind, see
``torch_sla.datasets.SuiteSparse``) plus a small number of synthetic
edge cases that pin down behaviour on cases the catalogue doesn't cover
(empty matrix, real symmetric indefinite, etc.).
"""
from __future__ import annotations

import pytest
import torch

from torch_sla import SparseTensor


# =====================================================================
# Real SuiteSparse coverage -- the actual user-facing path
# =====================================================================

def test_detect_matches_catalogue(benchmark):
    """``SparseTensor.detect_matrix_type()`` returns the expected label
    on every catalogued SuiteSparse matrix.

    Uses ``benchmark.detected_kind`` (what the heuristic *is expected
    to* return, may be more conservative than the true mathematical
    kind because Gershgorin is sufficient but not necessary)."""
    A = SparseTensor(benchmark.val, benchmark.row, benchmark.col, benchmark.shape)
    assert A.detect_matrix_type() == benchmark.detected_kind, (
        f"{benchmark.name}: expected {benchmark.detected_kind!r}, "
        f"got {A.detect_matrix_type()!r}"
    )


def test_catalogue_covers_every_kind():
    """The catalogue together exercises every distinct mathematical
    kind the detector might encounter (general, symmetric, spd,
    hermitian, hpd)."""
    from torch_sla.datasets import SuiteSparse
    math_kinds = {SuiteSparse[k].math_kind for k in SuiteSparse}
    assert math_kinds == {"spd", "hpd", "symmetric", "general"}, (
        f"missing kinds: {math_kinds}"
    )


# =====================================================================
# Synthetic edge cases the catalogue doesn't cover
# =====================================================================
# These pin down detector behaviour on inputs that don't appear in the
# SuiteSparse catalogue: empty matrices, indefinite symmetric matrices,
# strict-diag-dominant cases (which the catalogue lacks for SPD/HPD).


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


def test_detect_strict_diag_dom_spd():
    """Strictly diagonally dominant real symmetric -> ``spd``."""
    A = _from_dense(torch.tensor([
        [10., 1., 0.],
        [1., 12., 2.],
        [0., 2., 11.],
    ], dtype=torch.float64))
    assert A.detect_matrix_type() == "spd"


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
    """Complex symmetric (A = A^T) but NOT Hermitian (diagonal not real)
    -> ``symmetric``."""
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
    # Build a tiny batched SparseTensor: 2 batches of a 3x3.
    val = torch.ones(2, 3, dtype=torch.float64)
    row = torch.tensor([0, 1, 2])
    col = torch.tensor([0, 1, 2])
    A = SparseTensor(val, row, col, (2, 3, 3))
    with pytest.raises(ValueError, match="non-batched"):
        A.detect_matrix_type()
