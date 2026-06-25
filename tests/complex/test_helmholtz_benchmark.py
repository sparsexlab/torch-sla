"""Public-benchmark validation of the **complex** solve: a 1-D **Helmholtz /
impedance** problem — the canonical complex-symmetric sparse system (time-harmonic
wave / PML), which is the motivating use case for torch-sla's complex support.

Discrete operator (complex-symmetric, A = Aᵀ ≠ Aᴴ):

    A = (1/h²) tridiag(-1, 2, -1)  -  k² I  -  i·α I

(the -k²I is the Helmholtz shift, the -iαI a lumped impedance/absorption term).
We manufacture a known complex solution u*, set b = A u*, solve, and check we
recover u* — against the exact vector and against ``scipy.sparse.linalg.spsolve``.
"""
import math

import numpy as np
import pytest
import torch

from torch_sla import spsolve
from torch_sla.datasets import helmholtz_1d

torch.set_default_dtype(torch.float64)


def _helmholtz_1d(n, k=8.0, alpha=2.0):
    """Complex-symmetric 1-D Helmholtz operator (from torch_sla.datasets).
    Returns (row, col, val, (n, n)) as torch complex128."""
    p = helmholtz_1d(n, k=k, alpha=alpha)
    val, row, col, shape = p.coo()
    return row, col, val, shape


def test_helmholtz_1d_manufactured():
    """Recover a manufactured complex solution u* from b = A u* (complex direct)."""
    n = 300
    row, col, val, shape = _helmholtz_1d(n)
    g = torch.Generator().manual_seed(0)
    u_star = (torch.randn(n, generator=g, dtype=torch.float64)
              + 1j * torch.randn(n, generator=g, dtype=torch.float64))
    # b = A u*  (dense reference for the RHS only)
    A = torch.zeros(n, n, dtype=torch.complex128).index_put((row, col), val, accumulate=True)
    b = A @ u_star

    u = spsolve(val, row, col, shape, b, backend="scipy", method="lu")
    rel = (u - u_star).abs().max() / u_star.abs().max()
    assert rel < 1e-9, f"complex Helmholtz recover rel err = {rel:.2e}"


def test_helmholtz_1d_vs_scipy():
    """Same complex-symmetric system vs scipy.sparse.linalg.spsolve (reference)."""
    spla = pytest.importorskip("scipy.sparse.linalg")
    import scipy.sparse as sp
    n = 200
    row, col, val, shape = _helmholtz_1d(n)
    g = torch.Generator().manual_seed(1)
    b = (torch.randn(n, generator=g, dtype=torch.float64)
         + 1j * torch.randn(n, generator=g, dtype=torch.float64))

    u = spsolve(val, row, col, shape, b, backend="scipy", method="lu")

    A_sp = sp.coo_matrix((val.numpy(), (row.numpy(), col.numpy())), shape=(n, n)).tocsr()
    u_ref = spla.spsolve(A_sp, b.numpy())
    rel = np.linalg.norm(u.numpy() - u_ref) / np.linalg.norm(u_ref)
    assert rel < 1e-10, f"complex solve vs scipy rel diff = {rel:.2e}"
