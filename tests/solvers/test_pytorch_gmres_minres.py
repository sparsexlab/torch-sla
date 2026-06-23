"""Numerical correctness tests for the new PyTorch-native GMRES / MINRES.

These fill the gap that previously forced the ``cupy`` backend: GPU/CPU
``gmres`` (general, non-symmetric) and ``minres`` (symmetric/Hermitian,
possibly indefinite). Everything here runs on CPU; CUDA is exercised
opportunistically when available.

Run with::

    pytest tests/test_pytorch_gmres_minres.py -v

Each solver is checked against a dense reference (``torch.linalg.solve``)
to a tight tolerance, in float64, for real and complex systems, single
and multi-RHS. The autograd adjoint is checked via ``gradcheck`` on the
public ``solve`` path.
"""
from __future__ import annotations

import pytest
import torch

from torch_sla import SparseTensor, solve
from torch_sla.backends.pytorch_backend import pgmres_solve, minres_solve


torch.manual_seed(0)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# --------------------------------------------------------------------------- #
# Matrix builders (dense -> COO triple)
# --------------------------------------------------------------------------- #
def _coo(dense):
    idx = dense.nonzero(as_tuple=False).t().contiguous()
    row, col = idx[0], idx[1]
    val = dense[row, col]
    return val, row, col, dense.shape


def _spd(n, dtype, device):
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = A @ A.conj().t() + n * torch.eye(n, dtype=dtype, device=device)
    return A


def _hermitian_indefinite(n, dtype, device):
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = A + A.conj().t()            # Hermitian, generally indefinite
    return A


def _general(n, dtype, device):
    A = torch.randn(n, n, dtype=dtype, device=device)
    A = A + n * torch.eye(n, dtype=dtype, device=device)   # well-conditioned, non-symmetric
    return A


# --------------------------------------------------------------------------- #
# GMRES -- general / non-symmetric
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_gmres_general(device, dtype):
    n = 40
    A = _general(n, dtype, device)
    b = torch.randn(n, dtype=dtype, device=device)
    val, row, col, shape = _coo(A)

    x, iters, res = pgmres_solve(val, row, col, shape, b,
                                 atol=1e-12, rtol=1e-12, maxiter=500,
                                 preconditioner="jacobi", restart=30)
    x_ref = torch.linalg.solve(A, b)
    assert torch.allclose(x, x_ref, atol=1e-7, rtol=1e-7), \
        f"GMRES wrong (iters={iters}, res={res:.2e})"


@pytest.mark.parametrize("device", DEVICES)
def test_gmres_multi_rhs(device):
    n, k = 30, 4
    A = _general(n, torch.float64, device)
    B = torch.randn(n, k, dtype=torch.float64, device=device)
    At = SparseTensor.from_dense(A)
    X = solve(At, B, backend="pytorch", method="gmres", preconditioner="jacobi")
    assert torch.allclose(X, torch.linalg.solve(A, B), atol=1e-6, rtol=1e-6)


# --------------------------------------------------------------------------- #
# MINRES -- symmetric / Hermitian, including indefinite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_minres_spd(device, dtype):
    n = 40
    A = _spd(n, dtype, device)
    b = torch.randn(n, dtype=dtype, device=device)
    val, row, col, shape = _coo(A)
    x, iters, res = minres_solve(val, row, col, shape, b,
                                 atol=1e-12, rtol=1e-12, maxiter=500,
                                 preconditioner="jacobi")
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-7, rtol=1e-7), \
        f"MINRES(SPD) wrong (iters={iters}, res={res:.2e})"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_minres_indefinite(device, dtype):
    """The case BiCGStab/CG can't reliably handle: Hermitian indefinite.

    Uses an unpreconditioned run (M=I is SPD) so the SPD-preconditioner
    requirement holds even though A itself is indefinite.
    """
    n = 40
    A = _hermitian_indefinite(n, dtype, device)
    b = torch.randn(n, dtype=dtype, device=device)
    val, row, col, shape = _coo(A)
    x, iters, res = minres_solve(val, row, col, shape, b,
                                 atol=1e-12, rtol=1e-12, maxiter=1000,
                                 preconditioner="none")
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-6, rtol=1e-6), \
        f"MINRES(indefinite) wrong (iters={iters}, res={res:.2e})"


# --------------------------------------------------------------------------- #
# Autograd adjoint via the public solve() path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["gmres", "minres"])
def test_gradient_gradcheck(method):
    n = 12
    base = (_general if method == "gmres" else _spd)(n, torch.float64, "cpu")
    A = base.clone()
    val, row, col, shape = _coo(A)
    val = val.clone().requires_grad_(True)
    b = torch.randn(n, dtype=torch.float64, requires_grad=True)

    def f(v, b_):
        At = SparseTensor(v, row, col, shape)
        return solve(At, b_, backend="pytorch", method=method,
                     preconditioner="jacobi")

    assert torch.autograd.gradcheck(f, (val, b), atol=1e-4, rtol=1e-3, eps=1e-6)
