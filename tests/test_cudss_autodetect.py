"""Tests for the auto-detection logic in the cuDSS (nvmath-python) backend.

These tests do not require CUDA or nvmath-python -- they exercise only the
pure-torch matrix-type detection that runs *before* the cuDSS call. The
cuDSS forward solve itself requires a working CUDA + nvmath install and is
covered separately by an integration test gated on torch.cuda.is_available().
"""
import pytest
import torch

from torch_sla.backends.nvmath_backend import (
    detect_matrix_type,
    _check_symmetry,
    _check_positive_definite_gershgorin,
)


# ---------------- helpers ---------------- #
def dense_to_coo(A: torch.Tensor):
    """Convert a dense matrix to COO (val, row, col), keeping non-zeros only."""
    n = A.shape[0]
    mask = A != 0
    idx = mask.nonzero(as_tuple=False)
    row = idx[:, 0].contiguous()
    col = idx[:, 1].contiguous()
    val = A[row, col]
    return val, row, col, (n, n)


# ---------------- real matrices ---------------- #
def test_detect_real_general():
    A = torch.tensor([[3., 1., 0.],
                      [0., 2., 5.],
                      [1., 0., 4.]], dtype=torch.float64)
    assert detect_matrix_type(*dense_to_coo(A)) == "general"


def test_detect_real_symmetric_indefinite():
    # symmetric but not PD (negative on diagonal)
    A = torch.tensor([[-1., 2., 0.],
                      [ 2., 3., 1.],
                      [ 0., 1., 4.]], dtype=torch.float64)
    assert detect_matrix_type(*dense_to_coo(A)) == "symmetric"


def test_detect_real_spd():
    # strictly diagonally dominant, positive diagonal → Gershgorin PD
    A = torch.tensor([[10., 1., 0.],
                      [ 1., 10., 1.],
                      [ 0., 1., 10.]], dtype=torch.float64)
    assert detect_matrix_type(*dense_to_coo(A)) == "spd"


# ---------------- complex matrices ---------------- #
def test_detect_complex_general():
    A = torch.tensor([[3+0j, 1+1j],
                      [2+1j, 4+0j]], dtype=torch.complex128)
    assert detect_matrix_type(*dense_to_coo(A)) == "general"


def test_detect_complex_symmetric():
    # A = A^T (no conjugation) — complex diagonal allowed
    A = torch.tensor([[3+1j, 1-1j],
                      [1-1j, 4+2j]], dtype=torch.complex128)
    assert detect_matrix_type(*dense_to_coo(A)) == "symmetric"


def test_detect_complex_hermitian_indefinite():
    # A = A^H — diagonal must be real
    A = torch.tensor([[-2+0j, 1-1j],
                      [ 1+1j,  3+0j]], dtype=torch.complex128)
    assert detect_matrix_type(*dense_to_coo(A)) == "hermitian"


def test_detect_complex_hpd():
    # Hermitian + strictly diagonally dominant ⇒ HPD via Gershgorin
    A = torch.tensor([[10+0j, 1-1j, 0+0j],
                      [ 1+1j, 10+0j, 0-2j],
                      [ 0+0j, 0+2j, 10+0j]], dtype=torch.complex128)
    assert detect_matrix_type(*dense_to_coo(A)) == "hpd"


# ---------------- primitive checks ---------------- #
def test_check_symmetry_real():
    A = torch.tensor([[1., 2.],
                      [2., 3.]], dtype=torch.float64)
    val, row, col, _ = dense_to_coo(A)
    assert _check_symmetry(val, row, col, 2, conjugate=False) is True
    assert _check_symmetry(val, row, col, 2, conjugate=True) is True  # real → same


def test_check_symmetry_complex_distinguishes_T_vs_H():
    # Complex symmetric (A = A^T but NOT A^H)
    A = torch.tensor([[1+1j, 2+0j],
                      [2+0j, 3+1j]], dtype=torch.complex128)
    val, row, col, _ = dense_to_coo(A)
    assert _check_symmetry(val, row, col, 2, conjugate=False) is True
    assert _check_symmetry(val, row, col, 2, conjugate=True)  is False


def test_check_pd_gershgorin_diagonally_dominant():
    A = torch.tensor([[5., 1., 0.],
                      [1., 5., 0.],
                      [0., 0., 5.]], dtype=torch.float64)
    val, row, col, _ = dense_to_coo(A)
    assert _check_positive_definite_gershgorin(val, row, col, 3) is True


def test_check_pd_gershgorin_negative_diagonal():
    A = torch.tensor([[-1., 0.],
                      [ 0., -1.]], dtype=torch.float64)
    val, row, col, _ = dense_to_coo(A)
    assert _check_positive_definite_gershgorin(val, row, col, 2) is False
