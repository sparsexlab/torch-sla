"""Tests for the bound-constrained nonlinear solve ``vi_solve``.

Covers:
  * KKT conditions and agreement with SciPy's L-BFGS-B on a bounded convex QP,
  * the 1D obstacle problem against its analytical contact set and solution,
  * fixed components (lower == upper), a nonlinear residual, the
    Jacobian-free path (GMRES and CG), the projected line search,
  * adjoint gradients w.r.t. parameters and both bounds (gradcheck),
  * Jacobian formats, a user linear solver, the info dict, input checks, CUDA.
"""
import math

import numpy as np
import pytest
import torch

from torch_sla import SparseTensor, vi_solve

torch.set_default_dtype(torch.float64)


def _tridiag(n, diag=2.0, off=-1.0, scale=1.0):
    i = torch.arange(n)
    row = torch.cat([i, i[1:], i[:-1]])
    col = torch.cat([i, i[:-1], i[1:]])
    val = torch.cat([torch.full((n,), diag), torch.full((2 * n - 2,), off)]) * scale
    return SparseTensor(val, row, col, (n, n))


def _kkt_violation(u, F, lower, upper, tol=1e-9):
    """Largest violation of the bounds and complementarity conditions."""
    at_lower = u <= lower + tol
    at_upper = u >= upper - tol
    inside = ~at_lower & ~at_upper
    worst = torch.zeros(())
    worst = torch.maximum(worst, (lower - u).clamp(min=0).max())
    worst = torch.maximum(worst, (u - upper).clamp(min=0).max())
    if inside.any():
        worst = torch.maximum(worst, F[inside].abs().max())
    if at_lower.any():
        worst = torch.maximum(worst, (-F[at_lower]).clamp(min=0).max())
    if at_upper.any():
        worst = torch.maximum(worst, F[at_upper].clamp(min=0).max())
    return float(worst)


def _bounded_qp(n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = _tridiag(n, diag=2.5)
    f = torch.randn(n, generator=g)
    return A, f, torch.full((n,), -0.3), torch.full((n,), 0.4)


# --------------------------------------------------------------------------- #
# Forward                                                                     #
# --------------------------------------------------------------------------- #

def test_bounded_qp_kkt_and_scipy_reference():
    from scipy.optimize import minimize

    A, f, lo, hi = _bounded_qp()
    u, info = vi_solve(lambda u, f: A @ u - f, torch.zeros_like(f), f, lower=lo, upper=hi,
                       jacobian_fn=lambda u, f: A, return_info=True)
    F = A @ u - f
    assert info["converged"]
    assert _kkt_violation(u, F, lo, hi) < 1e-10
    assert 0 < info["active"] < f.numel()  # both active and free components

    Ad = A.to_dense().numpy()
    fd = f.numpy()
    ref = minimize(lambda x: 0.5 * x @ Ad @ x - fd @ x, np.zeros_like(fd), jac=lambda x: Ad @ x - fd,
                   method="L-BFGS-B", bounds=[(-0.3, 0.4)] * fd.size,
                   options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10000})
    assert np.abs(u.numpy() - ref.x).max() < 1e-6


def test_obstacle_problem_matches_analytical_solution():
    """-u'' = -10 on (0, 1), u(0) = u(1) = 0, u >= -0.2: contact on [0.2, 0.8],
    u = 5x^2 - 2x on [0, 0.2] (and mirrored)."""
    n = 79
    h = 1.0 / (n + 1)
    x = torch.arange(1, n + 1) * h
    A = _tridiag(n, scale=1.0 / h**2)
    f = torch.full((n,), -10.0)
    u, info = vi_solve(lambda u, f: A @ u - f, torch.zeros(n), f, lower=-0.2,
                       jacobian_fn=lambda u, f: A, return_info=True)
    assert info["converged"]
    contact = x[u <= -0.2 + 1e-12]
    assert abs(float(contact.min()) - 0.2) < 2 * h and abs(float(contact.max()) - 0.8) < 2 * h
    s = torch.minimum(x, 1 - x)
    exact = torch.where(s < 0.2, 5 * s**2 - 2 * s, torch.full_like(s, -0.2))
    assert float((u - exact).abs().max()) < 2e-3
    # from u0 = 0 the first projected step overshoots the contact set to about
    # [0.05, 0.95]; each iteration then releases about one node per side
    assert info["iterations"] <= 0.2 * n

    # started from the solution, it is accepted at once
    u2, info2 = vi_solve(lambda u, f: A @ u - f, u, f, lower=-0.2,
                         jacobian_fn=lambda u, f: A, return_info=True)
    assert info2["iterations"] == 0 and torch.equal(u2, u)


def test_fixed_components():
    A, f, lo, hi = _bounded_qp()
    n = f.numel()
    fixed = torch.zeros(n, dtype=torch.bool)
    fixed[[0, 7, n - 1]] = True
    value = torch.tensor(0.05)
    lo = torch.where(fixed, value, lo)
    hi = torch.where(fixed, value, hi)
    u = vi_solve(lambda u, f: A @ u - f, torch.zeros(n), f, lower=lo, upper=hi,
                 jacobian_fn=lambda u, f: A)
    assert torch.equal(u[fixed], torch.full((3,), 0.05))
    F = A @ u - f
    free = ~fixed
    assert _kkt_violation(u[free], F[free], lo[free], hi[free]) < 1e-10


def _cubic(n=30):
    A = _tridiag(n)
    f = torch.linspace(-2.0, 3.0, n)

    def residual(u, f):
        return A @ u + u**3 - f

    def jacobian(u, f):
        return (A.to_dense() + torch.diag(3 * u**2)).to_sparse_coo()

    return A, f, residual, jacobian


def test_nonlinear_residual_with_bounds():
    _, f, residual, jacobian = _cubic()
    u, info = vi_solve(residual, torch.zeros_like(f), f, lower=-0.5, upper=0.8,
                       jacobian_fn=jacobian, return_info=True)
    assert info["converged"]
    assert _kkt_violation(u, residual(u, f), torch.full_like(f, -0.5), torch.full_like(f, 0.8)) < 1e-9


@pytest.mark.parametrize("method", ["gmres", "cg"])
def test_jacobian_free_matches_explicit(method):
    _, f, residual, jacobian = _cubic()
    explicit = vi_solve(residual, torch.zeros_like(f), f, lower=-0.5, upper=0.8, jacobian_fn=jacobian)
    free = vi_solve(residual, torch.zeros_like(f), f, lower=-0.5, upper=0.8, linear_method=method)
    assert torch.allclose(free, explicit, atol=1e-9)


def test_line_search_rescues_overshooting_newton():
    """F(u) = atan(u) - 0.5: full Newton steps from u0 = 8 overshoot; the
    projected backtracking converges to tan(0.5)."""
    n = 5

    def residual(u):
        return torch.atan(u) - 0.5

    def jacobian(u):
        i = torch.arange(n)
        return (1.0 / (1.0 + u**2), i, i, (n, n))

    with pytest.warns(UserWarning, match="did not converge"):
        u_plain, info_plain = vi_solve(residual, torch.full((n,), 8.0), lower=-50.0, upper=50.0,
                                       jacobian_fn=jacobian, max_iter=30, return_info=True)
    assert not info_plain["converged"]
    u, info = vi_solve(residual, torch.full((n,), 8.0), lower=-50.0, upper=50.0, jacobian_fn=jacobian,
                       line_search=True, return_info=True)
    assert info["converged"]
    assert torch.allclose(u, torch.full((n,), math.tan(0.5)), atol=1e-10)


# --------------------------------------------------------------------------- #
# Gradients                                                                   #
# --------------------------------------------------------------------------- #

def _strictly_complementary():
    """Small QP whose solution has components on the lower bound, on the upper
    bound and inside, each with a margin (so gradcheck's perturbations keep the
    active set)."""
    n = 6
    A = _tridiag(n, diag=3.0)
    f = torch.tensor([-4.0, 0.2, 0.1, -0.15, 3.0, 0.05], requires_grad=True)
    lo = torch.full((n,), -0.5, requires_grad=True)
    hi = torch.full((n,), 0.6, requires_grad=True)
    return A, f, lo, hi


@pytest.mark.parametrize("explicit", [True, False])
def test_gradcheck_params_and_bounds(explicit):
    A, f, lo, hi = _strictly_complementary()
    jac = (lambda u, f: A) if explicit else None

    def solve(f, lo, hi):
        return vi_solve(lambda u, f: A @ u - f, torch.zeros(6), f, lower=lo, upper=hi,
                        jacobian_fn=jac, linear_method="auto" if explicit else "cg")

    u = solve(f.detach(), lo.detach(), hi.detach())
    F = A @ u - f.detach()
    assert (u <= lo.detach() + 1e-12).any() and (u >= hi.detach() - 1e-12).any()
    assert ((u > lo.detach() + 1e-3) & (u < hi.detach() - 1e-3)).any()
    assert F[(u <= lo.detach() + 1e-12) | (u >= hi.detach() - 1e-12)].abs().min() > 1e-3
    assert torch.autograd.gradcheck(solve, (f, lo, hi), eps=1e-6, atol=1e-7)


def test_grad_through_fixed_components_goes_to_lower():
    """Dirichlet-style values built once and used for both bounds get the
    gradient once."""
    A, f, _, _ = _strictly_complementary()
    f = f.detach()
    g = torch.tensor(0.2, requires_grad=True)
    mask = torch.tensor([True, False, False, False, False, True])

    def solve(g):
        lo = torch.where(mask, g, torch.tensor(-math.inf))
        hi = torch.where(mask, g, torch.tensor(math.inf))
        return vi_solve(lambda u, f: A @ u - f, torch.zeros(6), f, lower=lo, upper=hi,
                        jacobian_fn=lambda u, f: A)

    assert torch.autograd.gradcheck(solve, (g,), eps=1e-6, atol=1e-7)


# --------------------------------------------------------------------------- #
# Interfaces                                                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fmt", ["sparse_tensor", "tuple", "torch_coo"])
def test_jacobian_formats(fmt):
    A, f, lo, hi = _bounded_qp(n=12, seed=1)

    def jac(u, f):
        if fmt == "sparse_tensor":
            return A
        if fmt == "tuple":
            return A.values, A.row_indices, A.col_indices, (12, 12)
        return A.to_dense().to_sparse_coo()

    ref = vi_solve(lambda u, f: A @ u - f, torch.zeros(12), f, lower=lo, upper=hi,
                   jacobian_fn=lambda u, f: A)
    u = vi_solve(lambda u, f: A @ u - f, torch.zeros(12), f, lower=lo, upper=hi, jacobian_fn=jac)
    assert torch.allclose(u, ref, atol=1e-12)


def test_callable_linear_solver():
    A, f, lo, hi = _bounded_qp(n=20, seed=2)
    calls = []

    def dense_solver(val, row, col, shape, b):
        calls.append(val.numel())
        M = torch.sparse_coo_tensor(torch.stack([row, col]), val, shape).to_dense()  # sums duplicates
        return torch.linalg.solve(M, b)

    ref = vi_solve(lambda u, f: A @ u - f, torch.zeros(20), f, lower=lo, upper=hi,
                   jacobian_fn=lambda u, f: A)
    u = vi_solve(lambda u, f: A @ u - f, torch.zeros(20), f, lower=lo, upper=hi,
                 jacobian_fn=lambda u, f: A, linear_solver=dense_solver)
    assert calls and len(set(calls)) == 1  # one sparsity pattern for every iteration
    assert torch.allclose(u, ref, atol=1e-12)


def test_info_and_unbounded_newton():
    A, f, _, _ = _bounded_qp(n=10, seed=3)
    u, info = vi_solve(lambda u, f: A @ u - f, torch.zeros(10), f, jacobian_fn=lambda u, f: A,
                       return_info=True)
    assert torch.allclose(A @ u, f, atol=1e-10)
    assert info["converged"] and info["iterations"] == 1 and info["active"] == 0
    assert set(info) == {"converged", "iterations", "residual", "function_evaluations", "active", "history"}


def test_input_checks():
    A, f, _, _ = _bounded_qp(n=4)
    with pytest.raises(ValueError, match="lower > upper"):
        vi_solve(lambda u, f: A @ u - f, torch.zeros(4), f, lower=1.0, upper=0.0, jacobian_fn=lambda u, f: A)
    with pytest.raises(ValueError, match="1-D"):
        vi_solve(lambda u, f: u, torch.zeros(2, 2), f)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("backend", ["pytorch", "scipy"])
def test_obstacle_problem_cuda(backend):
    n = 199
    h = 1.0 / (n + 1)
    A = _tridiag(n, scale=1.0 / h**2).cuda()
    f = torch.full((n,), -10.0, device="cuda")
    kwargs = {"linear_method": "cg", "linear_kwargs": {"atol": 1e-12}} if backend == "pytorch" else {}
    u = vi_solve(lambda u, f: A @ u - f, torch.zeros(n, device="cuda"), f, lower=-0.2,
                 jacobian_fn=lambda u, f: A, linear_solver=backend, **kwargs)
    u_cpu = vi_solve(lambda u, f: A.cpu() @ u - f, torch.zeros(n), f.cpu(), lower=-0.2,
                     jacobian_fn=lambda u, f: A.cpu())
    assert u.is_cuda
    assert torch.allclose(u.cpu(), u_cpu, atol=1e-8)
