"""Tests for the differentiable Newton nonlinear solve with
implicit-function-theorem gradients (``SparseTensor.nonlinear_solve``).

Covers:
  * forward correctness (residual ~ 1e-8 or better) for symmetric and
    non-symmetric Jacobians,
  * implicit-diff gradients w.r.t. parameters AND A's values, checked
    against finite differences and ``torch.autograd.gradcheck``,
  * the user-supplied Jacobian path,
  * O(1) graph: backward works without retaining the Newton iterations.
"""
import math

import pytest
import torch

from torch_sla.sparse_tensor import SparseTensor

torch.set_default_dtype(torch.float64)


def _tridiag(n, diag=2.0, off=-1.0):
    """Symmetric tridiagonal COO (SPD-ish) for a 1D Laplacian-like operator."""
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(diag)
        if i + 1 < n:
            rows.append(i); cols.append(i + 1); vals.append(off)
            rows.append(i + 1); cols.append(i); vals.append(off)
    return (torch.tensor(rows), torch.tensor(cols), torch.tensor(vals))


def _upper_bidiag(n, diag=3.0, up=1.5):
    """Non-symmetric (upper bidiagonal) COO -> non-symmetric Jacobian."""
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(diag)
        if i + 1 < n:
            rows.append(i); cols.append(i + 1); vals.append(up)
    return (torch.tensor(rows), torch.tensor(cols), torch.tensor(vals))


# --------------------------------------------------------------------------- #
# Forward residual                                                            #
# --------------------------------------------------------------------------- #

def test_forward_residual_bratu_symmetric():
    """Bratu-type A u + lam exp(u) - b = 0 (symmetric A)."""
    n = 10
    row, col, val = _tridiag(n)
    A = SparseTensor(val.clone(), row, col, (n, n))
    lam = torch.tensor(0.4)
    b = torch.linspace(0.1, 1.0, n)

    def resid(u, A, lam, b):
        return A @ u + lam * torch.exp(u) - b

    u = A.nonlinear_solve(resid, torch.zeros(n), lam, b, linear_method="lu")
    r = resid(u, A, lam, b)
    assert torch.linalg.vector_norm(r).item() < 1e-8


def test_forward_residual_nonsymmetric_jacobian():
    """Cubic nonlinearity with a non-symmetric A -> non-symmetric Jacobian.

    This is the case where CG-based Jacobian-free Newton diverges; a direct
    solve on the true Jacobian must converge.
    """
    n = 8
    row, col, val = _upper_bidiag(n)
    A = SparseTensor(val.clone(), row, col, (n, n))
    b = torch.ones(n)

    def resid(u, A, b):
        return A @ u + 0.1 * u ** 3 - b

    u = A.nonlinear_solve(resid, torch.zeros(n), b, linear_method="lu")
    r = resid(u, A, b)
    assert torch.linalg.vector_norm(r).item() < 1e-8


def test_forward_matches_reference_scalar():
    """Scalar-per-node: diagonal A so each node solves a u + u^3 = b
    independently; compare to a direct 1D Newton reference."""
    n = 6
    rows = torch.arange(n)
    a_vals = torch.full((n,), 2.0)
    A = SparseTensor(a_vals.clone(), rows, rows.clone(), (n, n))
    b = torch.linspace(1.0, 3.0, n)

    def resid(u, A, b):
        return A @ u + u ** 3 - b

    u = A.nonlinear_solve(resid, torch.zeros(n), b, linear_method="lu")

    # Reference: solve 2 x + x^3 = b_i per node.
    ref = torch.zeros(n)
    for i in range(n):
        x = 0.0
        for _ in range(100):
            f = 2 * x + x ** 3 - b[i].item()
            df = 2 + 3 * x ** 2
            x -= f / df
        ref[i] = x
    assert torch.allclose(u, ref, atol=1e-9)


# --------------------------------------------------------------------------- #
# Gradients                                                                   #
# --------------------------------------------------------------------------- #

def test_grad_vs_finite_difference():
    n = 5
    row, col, val0 = _tridiag(n, diag=3.0, off=-0.7)
    b0 = torch.linspace(0.5, 1.5, n)

    def solve_loss(val, b):
        A = SparseTensor(val, row, col, (n, n))

        def resid(u, A, b):
            return A @ u + 0.2 * torch.tanh(u) - b

        u = A.nonlinear_solve(resid, torch.zeros(n), b, linear_method="lu")
        return (u * torch.arange(1.0, n + 1)).sum()

    val = val0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    L = solve_loss(val, b)
    L.backward()
    ga, gb = val.grad.clone(), b.grad.clone()

    eps = 1e-6
    gfd_v = torch.zeros_like(val0)
    for k in range(val0.numel()):
        p = val0.clone(); p[k] += eps
        m = val0.clone(); m[k] -= eps
        gfd_v[k] = (solve_loss(p, b0).item() - solve_loss(m, b0).item()) / (2 * eps)
    gfd_b = torch.zeros_like(b0)
    for k in range(b0.numel()):
        p = b0.clone(); p[k] += eps
        m = b0.clone(); m[k] -= eps
        gfd_b[k] = (solve_loss(val0, p).item() - solve_loss(val0, m).item()) / (2 * eps)

    assert (ga - gfd_v).abs().max().item() < 1e-6
    assert (gb - gfd_b).abs().max().item() < 1e-6


def test_gradcheck():
    n = 4
    row, col, val0 = _tridiag(n, diag=3.0, off=-0.6)
    b0 = torch.linspace(0.5, 1.2, n)

    def f(val, b):
        A = SparseTensor(val, row, col, (n, n))

        def resid(u, A, b):
            return A @ u + 0.3 * torch.tanh(u) - b

        u = A.nonlinear_solve(resid, torch.zeros(n), b, linear_method="lu")
        return (u * torch.linspace(1.0, 2.0, n)).sum()

    inputs = (val0.clone().requires_grad_(True), b0.clone().requires_grad_(True))
    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_grad_only_param_requires_grad():
    """Only b requires grad; A's values must get no gradient and not error."""
    n = 5
    row, col, val0 = _tridiag(n, diag=3.0, off=-0.7)
    A = SparseTensor(val0.clone(), row, col, (n, n))  # no requires_grad
    b = torch.linspace(0.5, 1.5, n).requires_grad_(True)

    def resid(u, A, b):
        return A @ u + 0.2 * torch.tanh(u) - b

    u = A.nonlinear_solve(resid, torch.zeros(n), b, linear_method="lu")
    u.sum().backward()
    assert b.grad is not None
    assert b.grad.abs().sum().item() > 0


# --------------------------------------------------------------------------- #
# User-supplied Jacobian path                                                 #
# --------------------------------------------------------------------------- #

def test_explicit_jacobian_matches_autograd_jacobian():
    n = 6
    row, col, val0 = _tridiag(n, diag=3.0, off=-0.7)
    b0 = torch.linspace(0.5, 1.5, n)

    def resid(u, A, b):
        return A @ u + 0.5 * u ** 3 - b

    def jac(u, A, b):
        # J = A + diag(1.5 u^2).  Return COO for A's pattern plus the diagonal
        # contribution merged onto the diagonal entries.
        dval = A.values.clone()
        r, c = A.row_indices, A.col_indices
        diag_mask = r == c
        dval = dval.clone()
        dval[diag_mask] = dval[diag_mask] + 1.5 * u[r[diag_mask]] ** 2
        return dval, r, c, (n, n)

    A1 = SparseTensor(val0.clone().requires_grad_(True), row, col, (n, n))
    b1 = b0.clone().requires_grad_(True)
    u1 = A1.nonlinear_solve(resid, torch.zeros(n), b1, jac_fn=jac, linear_method="lu")
    (u1.sum()).backward()

    A2 = SparseTensor(val0.clone().requires_grad_(True), row, col, (n, n))
    b2 = b0.clone().requires_grad_(True)
    u2 = A2.nonlinear_solve(resid, torch.zeros(n), b2, linear_method="lu")
    (u2.sum()).backward()

    assert torch.allclose(u1, u2, atol=1e-9)
    assert torch.allclose(A1.values.grad, A2.values.grad, atol=1e-7)
    assert torch.allclose(b1.grad, b2.grad, atol=1e-7)


# --------------------------------------------------------------------------- #
# Implicit-diff property: graph stays O(1)                                    #
# --------------------------------------------------------------------------- #

def test_grad_independent_of_iteration_count():
    """Same problem, different max_iter -> identical converged solution AND
    identical gradients (gradient comes from IFT, not the Newton path)."""
    n = 5
    row, col, val0 = _tridiag(n, diag=3.0, off=-0.7)
    b0 = torch.linspace(0.5, 1.5, n)

    def run(max_iter):
        A = SparseTensor(val0.clone().requires_grad_(True), row, col, (n, n))
        b = b0.clone().requires_grad_(True)

        def resid(u, A, b):
            return A @ u + 0.2 * torch.tanh(u) - b

        u = A.nonlinear_solve(resid, torch.zeros(n), b, max_iter=max_iter,
                              linear_method="lu")
        u.sum().backward()
        return u.detach(), A.values.grad.clone(), b.grad.clone()

    u_a, ga_a, gb_a = run(50)
    u_b, ga_b, gb_b = run(8)
    assert torch.allclose(u_a, u_b, atol=1e-10)
    assert torch.allclose(ga_a, ga_b, atol=1e-9)
    assert torch.allclose(gb_a, gb_b, atol=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
