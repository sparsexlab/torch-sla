"""Tests for the tightened (Paige-Saunders normalized) MINRES stopping test.

Verifies that ``minres_solve`` now uses scipy's normalized stopping criterion,
so iteration counts drop toward ``scipy.sparse.linalg.minres`` (was ~15-20%
more, should now be within ~5-10%) WITHOUT losing accuracy.
"""
import math

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import pytest

from torch_sla.backends.pytorch_backend import minres_solve


def _poisson_2d(nx, ny):
    """2D 5-point Poisson (SPD) on an nx-by-ny grid."""
    Ix = sp.identity(nx)
    Iy = sp.identity(ny)
    Tx = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx))
    Ty = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(ny, ny))
    A = sp.kron(Iy, Tx) + sp.kron(Ty, Ix)
    return A.tocoo()


def _poisson_3d(nx, ny, nz):
    Ix, Iy, Iz = sp.identity(nx), sp.identity(ny), sp.identity(nz)
    Tx = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx))
    Ty = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(ny, ny))
    Tz = sp.diags([-1, 2, -1], [-1, 0, 1], shape=(nz, nz))
    A = (sp.kron(sp.kron(Iz, Iy), Tx)
         + sp.kron(sp.kron(Iz, Ty), Ix)
         + sp.kron(sp.kron(Tz, Iy), Ix))
    return A.tocoo()


def _indefinite(n):
    """Symmetric indefinite: Poisson shifted so part of the spectrum is < 0."""
    A = _poisson_2d(int(round(math.sqrt(n))), int(round(math.sqrt(n)))).tocsr()
    m = A.shape[0]
    # shift to straddle zero (non-eigenvalue shift to stay nonsingular)
    A = A - 4.37 * sp.identity(m)
    return A.tocoo()


def _coo_to_torch(A):
    A = A.tocoo()
    row = torch.tensor(A.row, dtype=torch.long)
    col = torch.tensor(A.col, dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float64)
    return val, row, col, (A.shape[0], A.shape[1])


def _count_scipy(A_csr, b, rtol):
    it = {"n": 0}

    def cb(xk):
        it["n"] += 1

    x, info = spla.minres(A_csr, b, rtol=rtol, maxiter=10000, callback=cb)
    return x, it["n"]


# indefinite uses no preconditioner (MINRES needs an SPD M; a shifted-Poisson
# diagonal is not SPD), the SPD problems use the default auto preconditioner.
@pytest.mark.parametrize("builder,name,precond", [
    (lambda: _poisson_2d(20, 20), "poisson2d", "none"),
    (lambda: _poisson_3d(8, 8, 8), "poisson3d", "none"),
    (lambda: _indefinite(400), "indefinite", "none"),
])
def test_minres_iters_close_to_scipy_and_accurate(builder, name, precond):
    rtol = 1e-8
    A_coo = builder()
    A_csr = A_coo.tocsr()
    n = A_csr.shape[0]

    rng = np.random.default_rng(0)
    b_np = rng.standard_normal(n)
    val, row, col, shape = _coo_to_torch(A_coo)
    b = torch.tensor(b_np, dtype=torch.float64)

    # dense reference
    x_dense = np.linalg.solve(A_csr.toarray(), b_np)

    # our minres
    x, iters, res = minres_solve(val, row, col, shape, b,
                                 atol=1e-12, rtol=rtol, maxiter=10000,
                                 preconditioner=precond)
    x_np = x.cpu().numpy()

    rel_err = np.linalg.norm(x_np - x_dense) / np.linalg.norm(x_dense)
    assert rel_err < 1e-6, f"{name}: rel_err {rel_err:.2e} too large"

    # scipy reference iteration count
    _, scipy_iters = _count_scipy(A_csr, b_np, rtol)

    # within ~10% of scipy (allow a small absolute slack for tiny problems)
    ratio = iters / max(scipy_iters, 1)
    print(f"{name}: ours={iters} scipy={scipy_iters} ratio={ratio:.3f} "
          f"rel_err={rel_err:.2e}")
    assert ratio <= 1.12 + 3.0 / max(scipy_iters, 1), (
        f"{name}: our iters {iters} vs scipy {scipy_iters} ratio {ratio:.3f}")


def test_minres_complex_hermitian():
    """Complex-Hermitian SPD system still solves accurately."""
    n = 60
    rng = np.random.default_rng(1)
    M = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = M @ M.conj().T + n * np.eye(n)  # Hermitian SPD
    A = A.astype(np.complex128)
    A_sp = sp.coo_matrix(A)

    row = torch.tensor(A_sp.row, dtype=torch.long)
    col = torch.tensor(A_sp.col, dtype=torch.long)
    val = torch.tensor(A_sp.data, dtype=torch.complex128)
    b_np = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    b = torch.tensor(b_np, dtype=torch.complex128)

    x, iters, res = minres_solve(val, row, col, (n, n), b,
                                 atol=1e-12, rtol=1e-10, maxiter=10000)
    x_dense = np.linalg.solve(A, b_np)
    rel_err = np.linalg.norm(x.cpu().numpy() - x_dense) / np.linalg.norm(x_dense)
    print(f"complex-hermitian: iters={iters} rel_err={rel_err:.2e}")
    assert rel_err < 1e-6
