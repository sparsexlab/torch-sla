"""LSQR / LSMR least-squares solvers (pytorch backend, device-agnostic).

Pure-torch ports of scipy.sparse.linalg.lsqr / lsmr -> run on CPU/CUDA/ROCm,
replacing the cupy backend's GPU least-squares. Verified against scipy on
square (consistent), overdetermined (rectangular), and damped systems; the
square case is checked differentiable via gradcheck through the solve() API.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from torch_sla import SparseTensor, solve
from torch_sla.backends.pytorch_backend import lsqr_solve, lsmr_solve

torch.manual_seed(0)
_FN = {"lsqr": (lsqr_solve, spla.lsqr), "lsmr": (lsmr_solve, spla.lsmr)}


def _coo(A):
    A = A.tocoo()
    return (torch.tensor(A.data, dtype=torch.float64),
            torch.tensor(A.row, dtype=torch.long),
            torch.tensor(A.col, dtype=torch.long), A.shape)


def _rel(x, xr):
    return float(np.linalg.norm(x.detach().cpu().numpy() - xr) / (np.linalg.norm(xr) + 1e-30))


@pytest.mark.parametrize("method", ["lsqr", "lsmr"])
def test_square_spd_vs_scipy(method):
    fn, sc = _FN[method]
    n = 40
    M = np.random.RandomState(0).randn(n, n)
    A = sp.csr_matrix(M @ M.T + n * np.eye(n))
    b = torch.randn(n, dtype=torch.float64)
    x, _, _ = fn(*_coo(A), b, atol=1e-10, btol=1e-10, maxiter=2000)
    xr = sc(A, b.numpy(), atol=1e-10, btol=1e-10)[0]
    assert _rel(x, xr) < 1e-7


@pytest.mark.parametrize("method", ["lsqr", "lsmr"])
def test_overdetermined_vs_scipy(method):
    fn, sc = _FN[method]
    m, n = 60, 30
    A = sp.random(m, n, density=0.3, random_state=1, format="csr") + sp.eye(m, n) * 2
    b = torch.randn(m, dtype=torch.float64)
    x, _, _ = fn(*_coo(A), b, atol=1e-10, btol=1e-10, maxiter=4000)
    xr = sc(A, b.numpy(), atol=1e-10, btol=1e-10)[0]
    assert _rel(x, xr) < 1e-6


@pytest.mark.parametrize("method", ["lsqr", "lsmr"])
def test_damped_vs_scipy(method):
    fn, sc = _FN[method]
    m, n = 50, 25
    A = sp.random(m, n, density=0.3, random_state=2, format="csr") + sp.eye(m, n) * 2
    b = torch.randn(m, dtype=torch.float64)
    x, _, _ = fn(*_coo(A), b, damp=0.5, atol=1e-10, btol=1e-10, maxiter=4000)
    xr = sc(A, b.numpy(), damp=0.5, atol=1e-10, btol=1e-10)[0]
    assert _rel(x, xr) < 1e-6


@pytest.mark.parametrize("method", ["lsqr", "lsmr"])
def test_solve_api_square(method):
    fn = _FN[method][0]
    n = 20
    M = np.random.RandomState(3).randn(n, n)
    A = sp.csr_matrix(M @ M.T + n * np.eye(n))
    b = torch.randn(n, dtype=torch.float64)
    x = solve(SparseTensor(*_coo(A)), b, backend="pytorch", method=method)
    assert _rel(x, spla.spsolve(A.tocsc(), b.numpy())) < 1e-6


@pytest.mark.parametrize("method", ["lsqr", "lsmr"])
def test_gradcheck_square(method):
    """Gradient through solve() on a square system."""
    n = 6
    M = torch.randn(n, n, dtype=torch.float64)
    A0 = M @ M.t() + n * torch.eye(n, dtype=torch.float64)
    idx = A0.nonzero(as_tuple=False).t()
    r, c = idx[0], idx[1]
    val = A0[r, c].clone().requires_grad_(True)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)

    def f(v, b_):
        return solve(SparseTensor(v, r, c, (n, n)), b_, backend="pytorch", method=method)

    assert torch.autograd.gradcheck(f, (val, b), atol=1e-4, rtol=1e-3)
