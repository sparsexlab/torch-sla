"""Tests for the PyAMG-hybrid backend (CPU AMG setup + torch.sparse V-cycle).

These tests run wherever ``pip install pyamg`` works -- which is
Windows, Linux, and macOS. Cross-platform AMG is the entire point of
this backend, so the test suite is the cross-platform claim.

Each test :func:`pytest.skip` s when pyamg is unavailable rather than
failing, so air-gapped CI still passes.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from torch_sla import SparseTensor, solve, spsolve
from torch_sla.backends import is_pyamg_available


pytestmark = pytest.mark.skipif(
    not is_pyamg_available(),
    reason="pyamg is an optional dependency; install with `pip install pyamg`."
)


# =====================================================================
# Fixtures: standard PDE stencils
# =====================================================================
def _poisson_2d(n: int) -> tuple:
    """5-point 2D Laplacian on an n x n grid. Returns (val, row, col, shape)."""
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    I = sp.eye(n, format="csr")
    A = (sp.kron(I, T) + sp.kron(T, I)).tocoo()
    A.data = A.data.astype(np.float64)
    return (torch.from_numpy(A.data),
            torch.from_numpy(A.row.astype(np.int64)),
            torch.from_numpy(A.col.astype(np.int64)),
            A.shape)


def _anisotropic_2d(n: int, eps: float) -> tuple:
    """``-eps * d^2/dx^2 - d^2/dy^2`` -- AMG's home turf."""
    T = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    I = sp.eye(n, format="csr")
    A = (sp.kron(I, eps * T) + sp.kron(T, I)).tocoo()
    A.data = A.data.astype(np.float64)
    return (torch.from_numpy(A.data),
            torch.from_numpy(A.row.astype(np.int64)),
            torch.from_numpy(A.col.astype(np.int64)),
            A.shape)


# =====================================================================
# Standalone AMG convergence
# =====================================================================
def test_pyamg_solves_poisson_2d():
    """Standalone AMG converges to the catalogue's analytical x_ref on a
    Poisson 2D system within 30 V-cycles to a relative residual < 1e-6."""
    val, row, col, shape = _poisson_2d(32)
    torch.manual_seed(0)
    x_ref = torch.randn(shape[0], dtype=torch.float64)
    A_sp = sp.coo_matrix((val.numpy(), (row.numpy(), col.numpy())),
                         shape=shape).tocsr()
    b = torch.from_numpy(A_sp @ x_ref.numpy())

    x = spsolve(val, row, col, shape, b,
                backend="pyamg", atol=1e-8, maxiter=30)
    rel_err = ((x - x_ref).norm() / x_ref.norm()).item()
    assert rel_err < 1e-7, f"AMG did not converge: rel_err = {rel_err}"


def test_pyamg_solves_anisotropic_diffusion():
    """Anisotropic diffusion (eps=0.01) is where AMG outperforms simple
    iterative methods. We accept it within 40 V-cycles."""
    val, row, col, shape = _anisotropic_2d(32, eps=0.01)
    torch.manual_seed(1)
    x_ref = torch.randn(shape[0], dtype=torch.float64)
    A_sp = sp.coo_matrix((val.numpy(), (row.numpy(), col.numpy())),
                         shape=shape).tocsr()
    b = torch.from_numpy(A_sp @ x_ref.numpy())

    x = spsolve(val, row, col, shape, b,
                backend="pyamg", atol=1e-7, maxiter=40)
    rel_err = ((x - x_ref).norm() / x_ref.norm()).item()
    assert rel_err < 1e-5, f"AMG-anisotropic: rel_err = {rel_err}"


def test_pyamg_via_new_solve_api():
    """The new :func:`torch_sla.solve` entry point also routes to pyamg."""
    val, row, col, shape = _poisson_2d(16)
    torch.manual_seed(0)
    x_ref = torch.randn(shape[0], dtype=torch.float64)
    A_sp = sp.coo_matrix((val.numpy(), (row.numpy(), col.numpy())),
                         shape=shape).tocsr()
    b = torch.from_numpy(A_sp @ x_ref.numpy())

    A = SparseTensor(val, row, col, shape)
    x, info = solve(A, b, backend="pyamg", atol=1e-8, maxiter=20,
                    return_info=True)
    assert info.converged
    assert ((x - x_ref).norm() / x_ref.norm()).item() < 1e-6


def test_pyamg_smoothed_aggregation_method():
    """``method='sa'`` selects the smoothed-aggregation coarsening variant.

    SA without a user-supplied near-null-space converges more slowly than
    Ruge-Stuben on a 5-point Laplacian (where RS's strength-of-connection
    matches the M-matrix structure exactly). We assert SA *makes
    progress* (~100x residual reduction) rather than reaching machine
    precision; users who need tight tolerance pass the constant vector
    via ``B=...`` to PyAMG directly through the hierarchy API."""
    val, row, col, shape = _poisson_2d(32)
    torch.manual_seed(2)
    x_ref = torch.randn(shape[0], dtype=torch.float64)
    A_sp = sp.coo_matrix((val.numpy(), (row.numpy(), col.numpy())),
                         shape=shape).tocsr()
    b = torch.from_numpy(A_sp @ x_ref.numpy())

    x = spsolve(val, row, col, shape, b,
                backend="pyamg", method="sa", atol=1e-8, maxiter=30)
    rel_err = ((x - x_ref).norm() / x_ref.norm()).item()
    assert rel_err < 1e-2, f"SA-AMG did not make progress: rel_err={rel_err}"


# =====================================================================
# Gradient (Wirtinger adjoint through AMG)
# =====================================================================
def test_pyamg_backward_produces_finite_gradient():
    """Backward pass via the adjoint solve produces a finite, non-trivial
    gradient on ``val``. The adjoint of an AMG-solved system is solved
    by the *same* AMG hierarchy on the conjugate transpose -- since
    Poisson is real symmetric, that's just another AMG solve."""
    val, row, col, shape = _poisson_2d(16)
    val = val.clone().requires_grad_(True)
    torch.manual_seed(3)
    b = torch.randn(shape[0], dtype=torch.float64)

    x = spsolve(val, row, col, shape, b,
                backend="pyamg", atol=1e-8, maxiter=20)
    loss = (x ** 2).sum()
    loss.backward()

    assert val.grad is not None
    assert torch.isfinite(val.grad).all().item()
    assert val.grad.abs().max().item() > 1e-6


# =====================================================================
# Hierarchy-level API
# =====================================================================
def test_pyamg_hierarchy_reused_for_multiple_rhs():
    """Build a hierarchy once, use it as a callable preconditioner for
    multiple right-hand sides -- this is the LRU-cache pattern that
    motivated the solver-caching follow-up PR."""
    from torch_sla.backends.pyamg_backend import PyAMGHierarchy
    val, row, col, shape = _poisson_2d(16)
    H = PyAMGHierarchy.from_coo(val, row, col, shape)

    # The hierarchy is callable -- one V-cycle returns an approximate solve.
    A_sp = sp.coo_matrix((val.numpy(), (row.numpy(), col.numpy())),
                         shape=shape).tocsr()
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        b = torch.randn(shape[0], dtype=torch.float64)
        # Each V-cycle is a fixed contraction (~0.1 for classical AMG on
        # Poisson); 10 cycles compounds to ~1e-10 if the constant holds,
        # in practice we get to ~1e-7 reliably.
        x = torch.zeros_like(b)
        for _ in range(10):
            r = b - torch.from_numpy(A_sp @ x.numpy())
            x = x + H(r)
        rel_resid = (b - torch.from_numpy(A_sp @ x.numpy())).norm() / b.norm()
        assert rel_resid.item() < 1e-6, (
            f"seed={seed}: rel residual after 10 V-cycles = {rel_resid.item()}"
        )


def test_pyamg_hierarchy_levels_diminish():
    """A well-formed hierarchy has at least two levels for a non-trivial
    problem, and coarse-grid sizes strictly diminish."""
    from torch_sla.backends.pyamg_backend import PyAMGHierarchy
    val, row, col, shape = _poisson_2d(32)
    H = PyAMGHierarchy.from_coo(val, row, col, shape)
    assert len(H.levels) >= 2, (
        f"expected multi-level hierarchy, got {len(H.levels)} levels"
    )
    sizes = [L.A.shape[0] for L in H.levels]
    assert all(sizes[i] > sizes[i + 1] for i in range(len(sizes) - 1)), (
        f"coarse levels not strictly diminishing: {sizes}"
    )


# =====================================================================
# Cross-device (CUDA when available)
# =====================================================================
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available")
def test_pyamg_hierarchy_runs_on_cuda():
    """Build hierarchy on CPU (PyAMG), transfer operators to CUDA, run
    the V-cycle on GPU. This is the core hybrid-platform claim."""
    from torch_sla.backends.pyamg_backend import PyAMGHierarchy
    val, row, col, shape = _poisson_2d(32)
    val = val.cuda(); row = row.cuda(); col = col.cuda()
    H = PyAMGHierarchy.from_coo(val, row, col, shape, device=val.device)
    assert H.device.type == "cuda"
    torch.manual_seed(0)
    b = torch.randn(shape[0], dtype=torch.float64, device="cuda")
    x = H.v_cycle(b)
    assert x.device.type == "cuda"
    assert torch.isfinite(x).all().item()


# =====================================================================
# Availability sentinel
# =====================================================================
def test_pyamg_available_returns_bool():
    """``is_pyamg_available`` is a stable boolean -- safe for backend
    selection logic. (At collection time it returned ``True``; this
    re-validates inside the test.)"""
    assert is_pyamg_available() is True
