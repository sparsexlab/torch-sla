"""Complex sparse linear solve tests.

Forward solve correctness, conjugate transpose, and Wirtinger adjoint
gradient are exercised entirely through the public benchmark API in
``torch_sla.datasets`` (:class:`Benchmark` and the
:data:`SuiteSparse` / :data:`Synthetic` catalogues). The benchmark
catalogue covers every distinct case the complex adjoint must handle:

  Bai/mhd1280b           Hermitian SPD     (A = A^H, real diagonal)
  Bai/qc324              Complex symmetric (A = A^T, complex diagonal)
  Bai/mhd1280a           General complex   (no symmetry; pairs with mhd1280b)
  HB/young1c             General complex   (acoustic, David Young)
  Synthetic helmholtz_2d Complex symmetric (parametric Helmholtz)

``autograd.gradcheck`` numerical FD is O(nnz) full solves per check, so
it runs on the smallest catalogued complex matrix (qc324, n=324) with
``fast_mode=True`` (one random Jacobian-vector direction). That is the
gold-standard validation for the adjoint formula

    grad_b      = A^{-H} @ grad_u
    grad_val[k] = -grad_b[row[k]] * conj(u[col[k]])

(see ROADMAP.md item 1(b)).
"""
import pytest
import torch

from torch_sla import SparseTensor
from torch_sla.datasets import SuiteSparse


# Catalogued matrices whose condition number is so large that a random
# ``b = A @ x_ref`` round-trip cannot recover ``x_ref`` to working
# precision under any backend. Their purpose in the catalogue is as
# *structural* tests (cuDSS LU path, complex general matvec), not
# solve-accuracy benchmarks; they are skipped from solve-tolerance tests
# and exercised by ``.H`` / ``.conj`` / structural tests instead.
_PATHOLOGICAL_FOR_SOLVE = {
    "Bai/mhd1280a",   # cond ~ 1e14, the A matrix in the MHD Ax = lambda Bx pair
}


def _solver(val, row, col, shape, rhs):
    """Reference solver: torch-sla's :meth:`SparseTensor.solve` -- the
    callable plugs into :meth:`Benchmark.evaluate`."""
    return SparseTensor(val, row, col, shape).solve(rhs)


# =====================================================================
# Forward solve round-trip via Benchmark.evaluate
# =====================================================================

def test_complex_forward_solve_roundtrip(benchmark_complex):
    """:meth:`Benchmark.evaluate` round-trips every catalogued complex
    benchmark: each case stores ``b = A @ x_ref``, so a correct solver
    must recover ``x_ref`` to working precision.

    This is the canonical use of the public ``Benchmark`` API: no
    hand-rolled reference, no ``scipy.spsolve`` shadow -- just feed
    :meth:`SparseTensor.solve` to the benchmark and read the error.
    """
    if benchmark_complex.name in _PATHOLOGICAL_FOR_SOLVE:
        pytest.skip(
            f"{benchmark_complex.name}: pathologically ill-conditioned; "
            f"round-trip via random x_ref cannot recover x to working "
            f"precision under any backend (catalogued for structural "
            f"coverage, not solve accuracy)."
        )
    errs = benchmark_complex.evaluate(_solver, metric="rel_l2")
    assert max(errs) < 1e-8, f"{benchmark_complex.name}: errs = {errs}"


# =====================================================================
# Wirtinger adjoint via autograd.gradcheck
# =====================================================================

def test_complex_adjoint_gradcheck(benchmark_small_complex):
    """``autograd.gradcheck`` with ``fast_mode=True`` (random Jacobian-
    vector probe -- O(1) full solves rather than O(nnz)) verifies the
    complex Wirtinger adjoint against numerical FD on a real SuiteSparse
    matrix (qc324, n=324)."""
    b = benchmark_small_complex
    val = b.val.clone().contiguous().requires_grad_(True)
    rhs = b[0]["b"]  # reuse a stored reference RHS instead of inventing one

    def fn(v):
        return SparseTensor(v, b.row, b.col, b.shape).solve(rhs)

    assert torch.autograd.gradcheck(
        fn, (val,),
        eps=1e-6, atol=1e-4, rtol=1e-3,
        check_grad_dtypes=True,
        fast_mode=True,
    ), f"complex adjoint gradcheck failed on {b.name}"


# =====================================================================
# .H / .conj across the complex catalogue
# =====================================================================

def test_H_is_conjugate_transpose(benchmark_complex):
    """``A.H()`` reproduces ``conj(A.T)`` on each catalogued complex matrix."""
    b = benchmark_complex
    A = SparseTensor(b.val, b.row, b.col, b.shape)

    A_dense = torch.zeros(*b.shape, dtype=torch.complex128).index_put_(
        (b.row, b.col), b.val.to(torch.complex128), accumulate=True
    )
    AH = A.H().to_dense()
    assert torch.allclose(AH, A_dense.conj().T, atol=1e-10), (
        f"{b.name}: .H differs from conj(A^T) by "
        f"{(AH - A_dense.conj().T).abs().max().item()}"
    )


def test_conj_is_elementwise(benchmark_complex):
    """``A.conj()`` element-wise conjugates and preserves sparsity."""
    b = benchmark_complex
    A = SparseTensor(b.val, b.row, b.col, b.shape)

    A_dense = torch.zeros(*b.shape, dtype=torch.complex128).index_put_(
        (b.row, b.col), b.val.to(torch.complex128), accumulate=True
    )
    Ac = A.conj().to_dense()
    assert torch.allclose(Ac, A_dense.conj(), atol=1e-10)


# =====================================================================
# Property sanity on the SuiteSparse complex catalogue entries
# =====================================================================

def test_hermitian_benchmark_has_real_diagonal():
    """Sanity on Bai/mhd1280b: Hermitian implies real diagonal."""
    b = SuiteSparse["complex_hpd"]
    diag_mask = b.row == b.col
    assert b.val[diag_mask].imag.abs().max().item() < 1e-12, (
        f"{b.name}: diagonal has non-trivial imaginary part -- contradicts Hermitian"
    )


def test_complex_symmetric_benchmark_may_have_complex_diagonal():
    """Sanity on Bai/qc324: complex symmetric (A=A^T, not A^H) is
    allowed to have a complex diagonal, and qc324 actually does."""
    b = SuiteSparse["complex_sym"]
    diag_mask = b.row == b.col
    assert b.val[diag_mask].imag.abs().max().item() > 1e-3, (
        f"{b.name}: diagonal looks real -- expected complex entries"
    )


# =====================================================================
# Backwards compat: real-tensor .H == .T
# =====================================================================

def test_H_equals_T_on_real_tensor():
    """For real-valued matrices ``.H()`` must equal ``.T()`` -- ``.conj()``
    is a no-op on real tensors, so the conjugate-transpose collapses to
    the plain transpose. Defends against silently breaking real users."""
    rows = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.long)
    cols = torch.tensor([0, 1, 2, 3, 1, 2, 3], dtype=torch.long)
    vals = torch.tensor([2., 3., 4., 5., -1., -1., -1.], dtype=torch.float64)
    A = SparseTensor(vals, rows, cols, (4, 4))
    assert torch.allclose(A.H().to_dense(), A.T().to_dense())
