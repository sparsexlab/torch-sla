"""Public-benchmark validation of ``eigsh``: the **discrete Laplacian spectrum**.

The 1-D Dirichlet Laplacian ``tridiag(-1, 2, -1)`` (n interior nodes) is the
canonical sparse-eigensolver benchmark — its eigenvalues are known in **closed
form** (no discretization error):

    λ_k = 2 - 2 cos(k π / (n+1)) = 4 sin²(k π / (2(n+1))),   k = 1 … n

and the 2-D Laplacian on a grid is the Kronecker sum, so its eigenvalues are all
sums λ_i + λ_j of 1-D ones. We check ``SparseTensor.eigsh`` against (a) these
analytical eigenvalues and (b) ``scipy.sparse.linalg.eigsh`` (trusted public
reference) on the identical matrix.
"""
import math

import numpy as np
import pytest
import torch

from torch_sla.sparse_tensor import SparseTensor
from torch_sla.datasets import (
    laplacian_1d,
    laplacian_2d,
    laplacian_1d_eigenvalues,
)

torch.set_default_dtype(torch.float64)


def test_eigsh_1d_laplacian_analytical():
    """Smallest k eigenvalues of the 1-D Laplacian match the closed form."""
    n, k = 200, 6
    prob = laplacian_1d(n)
    val, row, col, shape = prob.coo()
    A = SparseTensor(val, row, col, shape)

    evals, _ = A.eigsh(k=k, which="SA")
    got = torch.sort(evals.real).values

    exact = torch.sort(laplacian_1d_eigenvalues(n)).values[:k]

    rel = (got - exact).abs() / exact
    assert rel.max() < 1e-5, f"eigsh vs analytical rel err = {rel.max():.2e}\n{got}\n{exact}"


def test_eigsh_1d_laplacian_vs_scipy():
    """Same eigenproblem solved by scipy.sparse.linalg.eigsh (public reference)."""
    sla = pytest.importorskip("scipy.sparse.linalg")
    import scipy.sparse as sp
    n, k = 150, 5
    prob = laplacian_1d(n)
    val, row, col, shape = prob.coo()
    A = SparseTensor(val, row, col, shape)
    evals, _ = A.eigsh(k=k, which="SA")
    got = np.sort(evals.real.numpy())

    A_sp = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n)).tocsr()
    ref = np.sort(sla.eigsh(A_sp, k=k, which="SA", return_eigenvectors=False))

    rel = np.abs(got - ref) / ref
    assert rel.max() < 1e-6, f"eigsh vs scipy rel err = {rel.max():.2e}"


def test_eigsh_2d_laplacian_analytical():
    """Smallest eigenvalues of the 2-D (m×m grid) Laplacian match λ_i+λ_j."""
    m, k = 20, 4
    # 5-point Laplacian on an m×m grid, Dirichlet (graph form: diag 4, -1 nbrs)
    prob = laplacian_2d(m)
    val, row, col, shape = prob.coo()
    A = SparseTensor(val, row, col, shape)
    evals, _ = A.eigsh(k=k, which="SA")
    got = torch.sort(evals.real).values

    # 1-D spectrum, then all pairwise sums; take the smallest k
    from torch_sla.datasets import laplacian_2d_eigenvalues
    exact = torch.sort(laplacian_2d_eigenvalues(m)).values[:k]

    rel = (got - exact).abs() / exact
    assert rel.max() < 1e-5, f"2D eigsh vs analytical rel err = {rel.max():.2e}\n{got}\n{exact}"
