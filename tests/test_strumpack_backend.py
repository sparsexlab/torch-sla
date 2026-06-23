"""Tests for the STRUMPACK direct-solver backend (``backend="strumpack"``).

Two layers:

1. **Adapter / autograd logic** -- runs everywhere. We inject a SciPy-SuperLU
   stand-in for torch-strumpack's ``factor`` / ``solve`` / ``solve_transpose``
   primitives so the torch-sla integration (COO->CSR, dispatch, the adjoint in
   ``SparseLinearSolveStrumpack``) is exercised *without* a compiled STRUMPACK
   extension. This verifies our code, not STRUMPACK's numerics.

2. **Real STRUMPACK smoke test** -- skipped unless a torch-strumpack wheel with
   the compiled extension is installed (the cpu / cuda / rocm path). On a GPU
   box this is the portable (incl. AMD ROCm) direct-solver check.
"""
from __future__ import annotations

import pytest
import torch

from torch_sla import SparseTensor, solve
from torch_sla.backends import is_strumpack_available


def _spd(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    return M @ M.t() + n * torch.eye(n, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# 1. Adapter + autograd, against a SciPy stand-in (no real STRUMPACK needed)
# --------------------------------------------------------------------------- #
@pytest.fixture
def strumpack_stand_in(monkeypatch):
    sp = pytest.importorskip("scipy.sparse")
    spla = pytest.importorskip("scipy.sparse.linalg")

    class _Fac:
        def __init__(self, A_csc):
            self.lu = spla.splu(A_csc)

    def _np(t):
        t = t.detach().cpu()
        return t.to(torch.complex128).numpy() if torch.is_complex(t) \
            else t.to(torch.float64).numpy()

    def factor(crow, col, values, n):
        A = sp.csr_matrix(
            (_np(values), col.detach().cpu().numpy(), crow.detach().cpu().numpy()),
            shape=(n, n),
        )
        return _Fac(A.tocsc())

    def solve(fac, b):
        x = fac.lu.solve(_np(b))
        return torch.from_numpy(x).to(device=b.device, dtype=b.dtype)

    def solve_transpose(fac, b):
        x = fac.lu.solve(_np(b), trans="T")
        return torch.from_numpy(x).to(device=b.device, dtype=b.dtype)

    import torch_sla.backends.strumpack_backend as spb
    import torch_sla.linear_solve as ls
    monkeypatch.setattr(spb, "factor", factor)
    monkeypatch.setattr(spb, "solve", solve)
    monkeypatch.setattr(spb, "solve_transpose", solve_transpose)
    monkeypatch.setattr(ls, "is_strumpack_available", lambda: True)


def test_strumpack_forward(strumpack_stand_in):
    n = 8
    A = _spd(n)
    b = torch.randn(n, dtype=torch.float64)
    x = solve(SparseTensor.from_dense(A), b, backend="strumpack")
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-9)


def test_strumpack_multi_rhs(strumpack_stand_in):
    n, k = 7, 4
    A = _spd(n, seed=3)
    B = torch.randn(n, k, dtype=torch.float64)
    X = solve(SparseTensor.from_dense(A), B, backend="strumpack")
    assert torch.allclose(X, torch.linalg.solve(A, B), atol=1e-9)


@pytest.mark.parametrize("k", [None, 3])
def test_strumpack_gradcheck(strumpack_stand_in, k):
    n = 6
    A = _spd(n, seed=1)
    idx = A.nonzero(as_tuple=False).t()
    row, col = idx[0], idx[1]
    val = A[row, col].clone().requires_grad_(True)
    shape = (n, n)
    b_shape = (n,) if k is None else (n, k)
    b = torch.randn(*b_shape, dtype=torch.float64, requires_grad=True)

    def f(v, b_):
        return solve(SparseTensor(v, row, col, shape), b_, backend="strumpack")

    assert torch.autograd.gradcheck(f, (val, b), atol=1e-5, rtol=1e-3)


def _hpd(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.complex128)
    return M @ M.conj().t() + n * torch.eye(n, dtype=torch.complex128)


def test_strumpack_complex_forward(strumpack_stand_in):
    n = 8
    A = _hpd(n)
    b = torch.randn(n, dtype=torch.complex128)
    x = solve(SparseTensor.from_dense(A), b, backend="strumpack")
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-9)


def test_strumpack_complex_gradcheck(strumpack_stand_in):
    """Complex adjoint (A^H + conj) must match dense autograd."""
    n = 5
    A = _hpd(n, seed=1)
    idx = A.nonzero(as_tuple=False).t()
    row, col = idx[0], idx[1]
    val = A[row, col].clone().requires_grad_(True)
    b = torch.randn(n, dtype=torch.complex128, requires_grad=True)

    def f(v, b_):
        return solve(SparseTensor(v, row, col, (n, n)), b_, backend="strumpack")

    assert torch.autograd.gradcheck(f, (val, b), atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------- #
# 2. Real STRUMPACK (only when the compiled extension is installed)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not is_strumpack_available(),
                    reason="torch-strumpack (compiled extension) not installed")
def test_strumpack_real_extension():
    n = 12
    A = _spd(n, seed=5)
    b = torch.randn(n, dtype=torch.float64)
    x = solve(SparseTensor.from_dense(A), b, backend="strumpack")
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-8)
