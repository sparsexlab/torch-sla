"""
Complex sparse linear solve tests.

Forward solve correctness, conjugate transpose, and Wirtinger adjoint
gradient are all exercised against real SuiteSparse Matrix Collection
benchmarks rather than hand-rolled synthetics. The catalogue covers
every mathematically distinct case the complex adjoint must handle:

  Bai/mhd1280b   Hermitian SPD       (A = A^H, real diagonal)
  Bai/qc324      Complex symmetric   (A = A^T, complex diagonal)
  Bai/mhd1280a   General complex     (no symmetry; pairs with mhd1280b)
  HB/young1c     General complex     (acoustic, David Young)

``autograd.gradcheck`` numerical FD is O(nnz) full solves per check, so
it runs on the smallest catalogued matrix (qc324, n=324) with
``fast_mode=True`` (one random Jacobian-vector direction). That's the
gold-standard validation for the adjoint formula

    grad_b      = A^{-H} @ grad_u
    grad_val[k] = -grad_b[row[k]] * conj(u[col[k]])

(see ROADMAP.md item 1(b)).
"""
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from torch_sla import SparseTensor


# =====================================================================
# Forward solve vs scipy reference
# =====================================================================

def test_complex_forward_solve_matches_scipy(suitesparse_complex):
    """forward solve agrees with scipy.sparse.linalg.spsolve to machine
    precision, on every catalogued complex matrix."""
    fx = suitesparse_complex
    n = fx["shape"][0]
    A_sp = sp.coo_matrix((fx["val"].numpy(),
                          (fx["row"].numpy(), fx["col"].numpy())),
                         shape=fx["shape"]).tocsr()
    torch.manual_seed(0)
    b = torch.randn(n, dtype=torch.complex128)
    x_ref = torch.from_numpy(np.asarray(spla.spsolve(A_sp, b.numpy())))

    A = SparseTensor(fx["val"], fx["row"], fx["col"], fx["shape"])
    x = A.solve(b)
    err = (x - x_ref).abs().max().item()
    assert err < 1e-9, f"{fx['key']}: forward err = {err}"


# =====================================================================
# Wirtinger adjoint via autograd.gradcheck (gold-standard FD validation)
# =====================================================================

def test_complex_adjoint_gradcheck(suitesparse_complex_small):
    """``autograd.gradcheck`` with ``fast_mode=True`` (random Jacobian-
    vector probe -- O(1) full solves rather than O(nnz)) verifies the
    complex Wirtinger adjoint against numerical FD on a real SuiteSparse
    matrix (qc324, n=324, complex symmetric, quantum chemistry)."""
    val, row, col, shape = suitesparse_complex_small
    val = val.contiguous().requires_grad_(True)
    torch.manual_seed(7)
    b = torch.randn(shape[0], dtype=torch.complex128)

    def fn(v):
        return SparseTensor(v, row, col, shape).solve(b)

    assert torch.autograd.gradcheck(
        fn, (val,),
        eps=1e-6, atol=1e-4, rtol=1e-3,
        check_grad_dtypes=True,
        fast_mode=True,
    ), "complex adjoint gradcheck failed on qc324"


# =====================================================================
# .H / .conj on real SuiteSparse matrices
# =====================================================================

def test_H_is_conjugate_transpose_on_real_matrix(suitesparse_complex):
    """``A.H()`` reproduces ``conj(A.T)`` on each catalogued complex matrix."""
    fx = suitesparse_complex
    A = SparseTensor(fx["val"], fx["row"], fx["col"], fx["shape"])

    # Reference: dense conj-transpose
    A_dense = torch.zeros(*fx["shape"], dtype=torch.complex128).index_put_(
        (fx["row"], fx["col"]), fx["val"], accumulate=True
    )
    AH = A.H().to_dense()
    assert torch.allclose(AH, A_dense.conj().T, atol=1e-10), (
        f"{fx['key']}: .H differs from conj(A^T) by "
        f"{(AH - A_dense.conj().T).abs().max().item()}"
    )


def test_conj_is_elementwise_on_real_matrix(suitesparse_complex):
    """``A.conj()`` element-wise conjugates and preserves sparsity."""
    fx = suitesparse_complex
    A = SparseTensor(fx["val"], fx["row"], fx["col"], fx["shape"])

    A_dense = torch.zeros(*fx["shape"], dtype=torch.complex128).index_put_(
        (fx["row"], fx["col"]), fx["val"], accumulate=True
    )
    Ac = A.conj().to_dense()
    assert torch.allclose(Ac, A_dense.conj(), atol=1e-10)


# =====================================================================
# Property sanity on the SuiteSparse benchmarks
# =====================================================================

def test_hermitian_benchmark_has_real_diagonal():
    """Mathematical sanity on Bai/mhd1280b: Hermitian implies real diagonal."""
    from tests._suitesparse import load_matrix
    val, row, col, _shape = load_matrix("complex_hpd", dtype=np.complex128)
    diag_mask = row == col
    diag_vals = val[diag_mask]
    assert diag_vals.imag.abs().max().item() < 1e-12, (
        "mhd1280b diagonal has non-trivial imaginary part -- contradicts Hermitian"
    )


def test_complex_symmetric_benchmark_may_have_complex_diagonal():
    """Math check on Bai/qc324: complex symmetric (A=A^T, not A^H) is
    allowed to have a complex diagonal (and qc324 actually does)."""
    from tests._suitesparse import load_matrix
    val, row, col, _shape = load_matrix("complex_sym", dtype=np.complex128)
    diag_mask = row == col
    diag_vals = val[diag_mask]
    assert diag_vals.imag.abs().max().item() > 1e-3, (
        "qc324 diagonal looks real -- expected complex entries"
    )


# =====================================================================
# Backwards compat: real-tensor .H == .T
# =====================================================================

def test_H_equals_T_on_real_tensor():
    """For real-valued matrices ``.H()`` must equal ``.T()`` -- ``.conj()``
    is a no-op on real tensors, so the conjugate-transpose collapses to
    the plain transpose. Defends against silently breaking real users."""
    rows = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.long)
    cols = torch.tensor([0, 1, 2, 3, 1, 2, 3], dtype=torch.long)
    vals = torch.tensor([2., 3., 4., 5., -1., -1., -1.], dtype=torch.float64)
    A = SparseTensor(vals, rows, cols, (4, 4))
    assert torch.allclose(A.H().to_dense(), A.T().to_dense())
