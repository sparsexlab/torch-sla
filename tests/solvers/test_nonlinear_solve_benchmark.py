"""Public-benchmark validation of the nonlinear solve: the **1D Bratu problem**.

Bratu (Bratu-Gelfand / solid-fuel ignition) is the canonical nonlinear elliptic
benchmark -- e.g. PETSc SNES example ex5. In 1D it has a **closed-form analytical
solution**, so we can check the solver against a known answer, not just a
self-consistent residual.

    -u''(x) = lambda * exp(u(x)),   u(0) = u(1) = 0

Exact solution (lower branch, for lambda < lambda_c ~= 3.5138):
    u(x) = -2 * ln[ cosh( (c/2)(x - 1/2) ) / cosh(c/4) ],
    where c solves   c = sqrt(2*lambda) * cosh(c/4).
(One verifies u'' + lambda e^u = 0 reduces exactly to that relation.)

Standard 3-point finite differences give the discrete residual
    F(u) = A u - lambda exp(u),   A = (1/h^2) tridiag(-1, 2, -1),  h = 1/(n+1).

We check torch-sla's nonlinear_solve against (a) the analytical solution and
(b) scipy.optimize.root on the identical discrete system.
"""
import math

import numpy as np
import pytest
import torch

from torch_sla.sparse_tensor import SparseTensor

torch.set_default_dtype(torch.float64)


def _bratu_c(lmbda):
    """Lower-branch root of c = sqrt(2 lambda) cosh(c/4) by fixed-point."""
    k = math.sqrt(2.0 * lmbda)
    c = 0.1
    for _ in range(200):
        c = k * math.cosh(c / 4.0)
    return c


def _bratu_exact(x, lmbda):
    c = _bratu_c(lmbda)
    return -2.0 * np.log(np.cosh((c / 2.0) * (x - 0.5)) / math.cosh(c / 4.0))


def _laplacian_1d(n, h):
    """(1/h^2) tridiag(-1, 2, -1) as a torch-sla SparseTensor, interior nodes."""
    rows, cols, vals = [], [], []
    inv = 1.0 / (h * h)
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(2.0 * inv)
        if i + 1 < n:
            rows.append(i); cols.append(i + 1); vals.append(-inv)
            rows.append(i + 1); cols.append(i); vals.append(-inv)
    return SparseTensor(torch.tensor(vals), torch.tensor(rows), torch.tensor(cols), (n, n))


def _resid(u, A, lmbda):
    return A @ u - lmbda * torch.exp(u)


def test_bratu_1d_matches_analytical():
    """Discrete Bratu solve converges to the exact analytical solution
    (within O(h^2) discretization error, which shrinks as n grows)."""
    lmbda = 1.0
    errs = {}
    for n in (50, 200):
        h = 1.0 / (n + 1)
        x = torch.linspace(h, 1 - h, n, dtype=torch.float64)
        A = _laplacian_1d(n, h)
        u = A.nonlinear_solve(_resid, torch.zeros(n), torch.tensor(lmbda),
                              linear_method="lu")
        u_exact = torch.from_numpy(_bratu_exact(x.numpy(), lmbda))
        errs[n] = (u - u_exact).abs().max().item()
    # absolute accuracy at the fine grid, and O(h^2) convergence (4x n -> ~16x smaller)
    assert errs[200] < 1e-3, f"max error at n=200 too large: {errs[200]:.2e}"
    assert errs[50] / errs[200] > 8.0, f"not ~2nd-order: {errs[50]:.2e} -> {errs[200]:.2e}"


def test_bratu_1d_matches_scipy_root():
    """Same discrete nonlinear system solved by scipy.optimize.root (trusted
    public reference) -- must agree to ~machine precision."""
    root = pytest.importorskip("scipy.optimize").root
    import scipy.sparse as sp
    lmbda = 1.5
    n = 100
    h = 1.0 / (n + 1)
    A = _laplacian_1d(n, h)
    u = A.nonlinear_solve(_resid, torch.zeros(n), torch.tensor(lmbda),
                          linear_method="lu")

    inv = 1.0 / (h * h)
    A_np = sp.diags([-inv, 2 * inv, -inv], [-1, 0, 1], shape=(n, n)).tocsr()

    def F(u_np):
        return A_np @ u_np - lmbda * np.exp(u_np)

    sol = root(F, np.zeros(n), method="hybr", tol=1e-12)
    assert sol.success, f"scipy root failed: {sol.message}"
    rel = np.linalg.norm(u.numpy() - sol.x) / (np.linalg.norm(sol.x) + 1e-30)
    assert rel < 1e-7, f"torch-sla vs scipy.optimize.root rel diff = {rel:.2e}"
