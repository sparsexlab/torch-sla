"""Tests for the ILU(0) and additive-Schwarz preconditioners.

A preconditioner earns its keep by reducing iteration counts. These tests
assert that ``'ilu'`` and ``'asm'`` make CG / GMRES converge in FEWER
iterations than plain ``'jacobi'`` on ill-conditioned problems, while keeping
the solution accurate. Real and complex paths are both exercised.
"""
import numpy as np
import scipy.sparse as sp
import torch
import pytest

from torch_sla.backends.pytorch_backend import (
    CachedSparseMatrix,
    get_preconditioner,
    pcg_solve_fused,
    pgmres_solve,
)


def _anisotropic_laplacian_2d(nx, ny, eps=0.01):
    """Anisotropic 2D Laplacian: eps*d2/dx2 + d2/dy2 (SPD, ill-conditioned)."""
    Ix = sp.identity(nx)
    Iy = sp.identity(ny)
    Tx = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx))
    Ty = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(ny, ny))
    A = eps * sp.kron(Iy, Tx) + sp.kron(Ty, Ix)
    return A.tocoo()


def _coo_to_torch(A, dtype=torch.float64):
    A = A.tocoo()
    row = torch.tensor(A.row, dtype=torch.long)
    col = torch.tensor(A.col, dtype=torch.long)
    val = torch.tensor(A.data, dtype=dtype)
    return val, row, col, (A.shape[0], A.shape[1])


def test_ilu_apply_is_accurate_inverse_pattern():
    """M^{-1} should approximate A^{-1}: M^{-1}(A x) ~ x reasonably."""
    A = _anisotropic_laplacian_2d(12, 12, eps=0.05).tocsr()
    n = A.shape[0]
    val, row, col, shape = _coo_to_torch(A)
    Acm = CachedSparseMatrix(val, row, col, shape)
    M = get_preconditioner(Acm, 'ilu')

    rng = np.random.default_rng(0)
    r = torch.tensor(rng.standard_normal(n), dtype=torch.float64)
    z = M(r)
    # residual of the preconditioned system should be much smaller than r itself
    Az = torch.tensor(A @ z.cpu().numpy(), dtype=torch.float64)
    assert torch.norm(Az - r) < 0.5 * torch.norm(r)


@pytest.mark.parametrize("precond", ["ilu", "asm"])
def test_cg_fewer_iters_than_jacobi(precond):
    A = _anisotropic_laplacian_2d(24, 24, eps=0.01).tocsr()
    n = A.shape[0]
    rng = np.random.default_rng(1)
    b_np = rng.standard_normal(n)
    val, row, col, shape = _coo_to_torch(A)
    b = torch.tensor(b_np, dtype=torch.float64)
    x_dense = np.linalg.solve(A.toarray(), b_np)

    res_j = pcg_solve_fused(val, row, col, shape, b, rtol=1e-8,
                            maxiter=5000, preconditioner='jacobi')
    res_p = pcg_solve_fused(val, row, col, shape, b, rtol=1e-8,
                            maxiter=5000, preconditioner=precond)

    err_p = np.linalg.norm(res_p.x.cpu().numpy() - x_dense) / np.linalg.norm(x_dense)
    print(f"CG {precond}: jacobi={res_j.num_iters} {precond}={res_p.num_iters} "
          f"rel_err={err_p:.2e}")
    assert err_p < 1e-5
    assert res_p.num_iters < res_j.num_iters, (
        f"{precond} ({res_p.num_iters}) not fewer than jacobi ({res_j.num_iters})")


@pytest.mark.parametrize("precond", ["ilu", "asm"])
def test_gmres_fewer_iters_than_jacobi(precond):
    A = _anisotropic_laplacian_2d(20, 20, eps=0.01).tocsr()
    n = A.shape[0]
    rng = np.random.default_rng(2)
    b_np = rng.standard_normal(n)
    val, row, col, shape = _coo_to_torch(A)
    b = torch.tensor(b_np, dtype=torch.float64)
    x_dense = np.linalg.solve(A.toarray(), b_np)

    x_j, it_j, _ = pgmres_solve(val, row, col, shape, b, rtol=1e-8,
                                maxiter=5000, preconditioner='jacobi', restart=50)
    x_p, it_p, _ = pgmres_solve(val, row, col, shape, b, rtol=1e-8,
                                maxiter=5000, preconditioner=precond, restart=50)

    err_p = np.linalg.norm(x_p.cpu().numpy() - x_dense) / np.linalg.norm(x_dense)
    print(f"GMRES {precond}: jacobi={it_j} {precond}={it_p} rel_err={err_p:.2e}")
    assert err_p < 1e-5
    assert it_p < it_j, f"{precond} ({it_p}) not fewer than jacobi ({it_j})"


@pytest.mark.parametrize("precond", ["ilu", "asm"])
def test_complex_safe(precond):
    """Preconditioners must run on complex-Hermitian systems and solve them."""
    n_side = 8
    A = _anisotropic_laplacian_2d(n_side, n_side, eps=0.1).tocsr().astype(np.complex128)
    n = A.shape[0]
    val, row, col, shape = _coo_to_torch(A, dtype=torch.complex128)
    Acm = CachedSparseMatrix(val, row, col, shape)
    M = get_preconditioner(Acm, precond)

    rng = np.random.default_rng(3)
    r = torch.tensor(rng.standard_normal(n) + 1j * rng.standard_normal(n),
                     dtype=torch.complex128)
    z = M(r)
    assert z.dtype == torch.complex128
    assert torch.isfinite(z.real).all() and torch.isfinite(z.imag).all()

    # CG on this SPD (real-structure) complex system should converge
    b = torch.tensor(rng.standard_normal(n), dtype=torch.complex128)
    res = pcg_solve_fused(val, row, col, shape, b, rtol=1e-8,
                          maxiter=5000, preconditioner=precond)
    x_dense = np.linalg.solve(A.toarray(), b.cpu().numpy())
    err = np.linalg.norm(res.x.cpu().numpy() - x_dense) / np.linalg.norm(x_dense)
    print(f"complex {precond}: iters={res.num_iters} rel_err={err:.2e}")
    assert err < 1e-5
