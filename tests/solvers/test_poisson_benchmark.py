"""Public-benchmark validation of the linear solve: the **2-D Poisson equation**
with a manufactured solution — the standard PDE-solver verification problem.

    -Δu = f  on (0,1)²,   u = 0 on ∂Ω,
    exact:  u(x,y) = sin(πx) sin(πy),   so  f = 2π² sin(πx) sin(πy).

Discretised with the 5-point stencil A = (1/h²)·penta(4, -1), h = 1/(m+1). We
check torch-sla against (a) the exact solution — second-order (O(h²)) convergence
— and (b) ``scipy.sparse.linalg.spsolve`` on the identical discrete system.
"""
import math

import numpy as np
import pytest
import torch

from torch_sla import spsolve
from torch_sla.datasets import poisson_2d

torch.set_default_dtype(torch.float64)


def _poisson_2d(m):
    """5-point Poisson on an m×m interior grid (from torch_sla.datasets).
    Returns (row, col, val, (n,n)), rhs f, exact u — all as torch float64."""
    p = poisson_2d(m)
    val, row, col, shape = p.coo()
    return row, col, val, shape, p.rhs, p.exact


def test_poisson_2d_manufactured_convergence():
    """Direct solve converges to the exact solution at second order."""
    errs = {}
    for m in (15, 31, 63):
        row, col, val, shape, f, u_exact = _poisson_2d(m)
        u = spsolve(val, row, col, shape, f, backend="scipy", method="lu")
        errs[m] = (u - u_exact).abs().max().item()
    assert errs[63] < 1e-3, f"max error at m=63 too large: {errs[63]:.2e}"
    # h halves between 15->31 and 31->63 -> error should drop ~4x each refinement
    assert errs[15] / errs[31] > 3.3, f"not ~2nd order: {errs[15]:.2e} -> {errs[31]:.2e}"
    assert errs[31] / errs[63] > 3.3, f"not ~2nd order: {errs[31]:.2e} -> {errs[63]:.2e}"


@pytest.mark.parametrize("method", ["lu", "cg"])
def test_poisson_2d_vs_scipy(method):
    """torch-sla (scipy LU and pytorch CG) vs scipy.sparse.linalg.spsolve on the
    identical discrete system — agreement to ~machine / solver tolerance."""
    spla = pytest.importorskip("scipy.sparse.linalg")
    import scipy.sparse as sp
    m = 40
    row, col, val, shape, f, _ = _poisson_2d(m)
    n = shape[0]

    if method == "lu":
        u = spsolve(val, row, col, shape, f, backend="scipy", method="lu")
    else:
        u = spsolve(val, row, col, shape, f, backend="pytorch", method="cg",
                    atol=1e-12, maxiter=20000)

    A_sp = sp.coo_matrix((val.numpy(), (row.numpy(), col.numpy())), shape=(n, n)).tocsr()
    u_ref = spla.spsolve(A_sp, f.numpy())

    rel = np.linalg.norm(u.numpy() - u_ref) / np.linalg.norm(u_ref)
    tol = 1e-5 if method == "cg" else 1e-10
    assert rel < tol, f"{method}: torch-sla vs scipy rel diff = {rel:.2e}"
