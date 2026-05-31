"""
Complex sparse linear solve tests.

Covers the three matrix-type cases that have *distinct* mathematical
behaviour for complex inputs:

  - Hermitian SPD               (A = A^H,  real diagonal,    cuDSS LDL^H path)
  - Complex symmetric           (A = A^T,  complex diagonal, cuDSS LDL^T path)
  - General non-Hermitian       (no symmetry,                LU path)

The scipy backend already supports all three via SuperLU's native complex LU,
so these tests validate (a) forward correctness vs ``torch.linalg.solve`` and
(b) gradient correctness via ``torch.autograd.gradcheck`` (the gold standard
that bypasses any hand-written reference). The complex adjoint formula is

    grad_b      = A^{-H} @ grad_u
    grad_val[k] = -grad_b[row[k]] * conj(u[col[k]])

(see ROADMAP.md item 1(b)).  For real tensors ``.conj()`` is a no-op so the
same code path stays correct -- regression-covered indirectly by
``test_sparse_tensor.py`` / ``test_spsolve.py``.
"""
import pytest
import torch
import numpy as np
from torch_sla import SparseTensor


def _make_solver(row, col, shape, b):
    """Return a closure (val) -> x that lets gradcheck differentiate w.r.t. val."""
    def fn(val):
        return SparseTensor(val, row, col, shape).solve(b)
    return fn


def _hermitian_spd(n, seed=0, dtype=torch.complex128):
    """Build a small Hermitian positive-definite complex sparse matrix."""
    torch.manual_seed(seed)
    # tridiagonal with strictly dominant real diagonal
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(complex(2.0 + n, 0.0))
        if i + 1 < n:
            v = complex(torch.randn(1).item(), torch.randn(1).item())
            rows.append(i);   cols.append(i+1); vals.append(v)
            rows.append(i+1); cols.append(i);   vals.append(v.conjugate())  # Hermitian
    return (torch.tensor(vals, dtype=dtype),
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long))


def _complex_symmetric(n, seed=1, dtype=torch.complex128):
    """Build a small complex-symmetric (A = A^T) matrix with complex diagonal."""
    torch.manual_seed(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        # complex diagonal (allowed for complex symmetric, forbidden for Hermitian)
        rows.append(i); cols.append(i); vals.append(complex(3.0 + n, 0.5))
        if i + 1 < n:
            v = complex(torch.randn(1).item(), torch.randn(1).item())
            rows.append(i);   cols.append(i+1); vals.append(v)
            rows.append(i+1); cols.append(i);   vals.append(v)  # symmetric, no conj
    return (torch.tensor(vals, dtype=dtype),
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long))


def _nonhermitian(n, seed=2, dtype=torch.complex128):
    """Build a small diagonally-dominant non-Hermitian complex matrix."""
    torch.manual_seed(seed)
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(complex(5.0, 1.0))
        if i + 1 < n:
            rows.append(i);   cols.append(i+1); vals.append(complex(1.0, 2.0))
            rows.append(i+1); cols.append(i);   vals.append(complex(-1.0, 0.5))
    return (torch.tensor(vals, dtype=dtype),
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long))


@pytest.mark.parametrize("builder,name", [
    (_hermitian_spd,     "hermitian_spd"),
    (_complex_symmetric, "complex_symmetric"),
    (_nonhermitian,      "nonhermitian"),
])
def test_complex_solve_forward(builder, name):
    """Forward solve matches a dense torch.linalg.solve reference."""
    n = 6
    val, row, col = builder(n)
    A = SparseTensor(val, row, col, (n, n))
    torch.manual_seed(42)
    b = torch.randn(n, dtype=torch.complex128)

    # dense reference
    A_dense = torch.zeros(n, n, dtype=torch.complex128).index_put_((row, col), val, accumulate=True)
    x_ref = torch.linalg.solve(A_dense, b)

    x = A.solve(b)
    assert torch.allclose(x, x_ref, atol=1e-10), f"{name}: forward err = {(x-x_ref).abs().max()}"


@pytest.mark.parametrize("builder,name", [
    (_hermitian_spd,     "hermitian_spd"),
    (_complex_symmetric, "complex_symmetric"),
    (_nonhermitian,      "nonhermitian"),
])
def test_complex_solve_gradcheck(builder, name):
    """Wirtinger gradient verified by torch.autograd.gradcheck (numerical FD)."""
    n = 5  # small so FD is fast (one FD pass per nnz)
    val, row, col = builder(n)
    val = val.requires_grad_(True)
    torch.manual_seed(7)
    b = torch.randn(n, dtype=torch.complex128)
    fn = _make_solver(row, col, (n, n), b)
    # autograd.gradcheck supports complex inputs out of the box
    assert torch.autograd.gradcheck(fn, (val,), eps=1e-6, atol=1e-4, rtol=1e-3,
                                    check_grad_dtypes=True), f"{name}: gradcheck failed"


def test_H_is_conjugate_transpose():
    """.H() returns A^H = conj(A^T)."""
    n = 4
    val, row, col = _nonhermitian(n)
    A = SparseTensor(val, row, col, (n, n))
    A_dense = torch.zeros(n, n, dtype=torch.complex128).index_put_((row, col), val, accumulate=True)
    assert torch.allclose(A.H().to_dense(), A_dense.conj().T)


def test_conj_is_elementwise():
    """.conj() conjugates entries, keeps sparsity pattern."""
    n = 4
    val, row, col = _nonhermitian(n)
    A = SparseTensor(val, row, col, (n, n))
    A_dense = torch.zeros(n, n, dtype=torch.complex128).index_put_((row, col), val, accumulate=True)
    assert torch.allclose(A.conj().to_dense(), A_dense.conj())


def test_H_equals_T_on_real():
    """For a real-valued matrix .H() must equal .T() (no-op conjugation)."""
    n = 4
    rows = torch.tensor([0,1,2,3,0,1,2], dtype=torch.long)
    cols = torch.tensor([0,1,2,3,1,2,3], dtype=torch.long)
    vals = torch.tensor([2.,3.,4.,5.,-1.,-1.,-1.], dtype=torch.float64)
    A = SparseTensor(vals, rows, cols, (n, n))
    assert torch.allclose(A.H().to_dense(), A.T().to_dense())


def test_hermitian_has_real_diagonal():
    """Math check: Hermitian implies real diagonal (sanity, not library test)."""
    n = 5
    val, row, col = _hermitian_spd(n)
    diag_mask = row == col
    diag_vals = val[diag_mask]
    assert torch.allclose(diag_vals.imag, torch.zeros_like(diag_vals.imag))


def test_complex_symmetric_can_have_complex_diagonal():
    """Math check: complex symmetric (A=A^T, not A^H) permits complex diagonal."""
    n = 5
    val, row, col = _complex_symmetric(n)
    diag_mask = row == col
    diag_vals = val[diag_mask]
    # Our builder uses 0.5j on the diagonal — confirm it's non-zero
    assert diag_vals.imag.abs().max() > 0.1
