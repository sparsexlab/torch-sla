"""Smoke tests for the torch-amgx-backed amgx backend.

Both the torch-amgx package and a CUDA device are gated -- on machines
that lack either, all tests in this module are skipped cleanly. macOS
will skip everything.
"""
from __future__ import annotations

import pytest
import torch

torch_amgx = pytest.importorskip("torch_amgx",
                                  reason="torch-amgx not installed")
pytestmark = pytest.mark.skipif(
    not torch_amgx.is_available(),
    reason="torch_amgx reports no CUDA / native lib unavailable",
)


def _poisson_1d_coo(n: int, device, dtype=torch.float64):
    """Build the standard 1-D Poisson stencil A x = b, x_exact = 1."""
    diag_idx = torch.arange(n, device=device)
    rows = [diag_idx, diag_idx[:-1], diag_idx[1:]]
    cols = [diag_idx, diag_idx[1:], diag_idx[:-1]]
    vals = [torch.full((n,), 2.0, dtype=dtype, device=device),
            torch.full((n - 1,), -1.0, dtype=dtype, device=device),
            torch.full((n - 1,), -1.0, dtype=dtype, device=device)]
    row = torch.cat(rows)
    col = torch.cat(cols)
    val = torch.cat(vals)
    x_exact = torch.ones(n, dtype=dtype, device=device)
    # b = A @ x_exact built from row/col/val without materializing A
    b = torch.zeros(n, dtype=dtype, device=device).index_add_(0, row, val * x_exact[col])
    return val, row, col, (n, n), b, x_exact


def test_is_amgx_available_truthy_on_cuda():
    from torch_sla.backends import is_amgx_available
    assert is_amgx_available() is True


def test_amgx_solve_recovers_poisson():
    from torch_sla.backends.amgx_backend import amgx_solve
    device = torch.device("cuda")
    val, row, col, shape, b, x_exact = _poisson_1d_coo(256, device)
    x = amgx_solve(val, row, col, shape, b,
                   tol=1e-10, maxiter=200, method="pbicgstab")
    err = (x - x_exact).norm() / x_exact.norm()
    assert err.item() < 1e-6, f"||x - x*|| / ||x*|| = {err.item():.2e}"


def test_amgx_solve_through_solve_api():
    """Hit the public torch_sla.solve() router with backend='amgx'."""
    from torch_sla import solve
    device = torch.device("cuda")
    val, row, col, shape, b, x_exact = _poisson_1d_coo(128, device)
    x = solve(val, row, col, shape, b,
              backend="amgx", tol=1e-10, maxiter=200, method="pbicgstab")
    err = (x - x_exact).norm() / x_exact.norm()
    assert err.item() < 1e-6, f"||x - x*|| / ||x*|| = {err.item():.2e}"


def test_amgx_solver_cache_reuses_setup():
    """Two solves on the same matrix should hit the cache, not rebuild
    the AmgX hierarchy."""
    from torch_sla.backends.amgx_backend import (
        amgx_preconditioner, AmgXSolver,
    )
    device = torch.device("cuda")
    val, row, col, shape, b, _ = _poisson_1d_coo(128, device)

    s1 = amgx_preconditioner(val, row, col, shape,
                             method="amg", tol=1e-3, maxiter=1)
    s2 = amgx_preconditioner(val, row, col, shape,
                             method="amg", tol=1e-3, maxiter=1)
    assert isinstance(s1, AmgXSolver)
    assert s1 is s2, "solver_cache did not hand back the same instance"
