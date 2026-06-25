#!/usr/bin/env python
"""Op ``setup(A, dof, device) -> callable()`` builders for the per-op benchmarks.

The headline ops that already existed -- spmv, matmat, solve_cg, solve_lu,
solve_strumpack, solve_cudss, solve_pyamg, det, det_backward, logdet, eigsh,
norm, transpose, connected_components -- have working setup/verify functions in
the original monolith ``benchmark_all_ops_scaling.py``. We import and REUSE them
here rather than copy-pasting, so the two stay in lock-step.

The four ops that were MISSING -- nonlinear_solve, svd, condition_number,
solve_batch -- get new setup functions written below, following the same
pattern and the real signatures in ``torch_sla/sparse_tensor/`` and
``torch_sla/distributed/core.py``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from torch_sla import SparseTensor  # noqa: F401  (re-exported for symmetry)

# --- load the monolith by path (benchmarks/ is not a package) --------------
_MONOLITH = Path(__file__).resolve().parents[1] / "benchmark_all_ops_scaling.py"
_spec = importlib.util.spec_from_file_location("_allops_monolith", _MONOLITH)
_M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_M)

# Existing setups, reused verbatim.
setup_spmv = _M._setup_spmv
setup_matmat = _M._setup_matmat
setup_solve_cg = _M._setup_solve_cg
setup_solve_lu = _M._setup_solve_lu
setup_solve_strumpack = _M._setup_solve_strumpack
setup_solve_cudss = _M._setup_solve_cudss
setup_solve_pyamg = _M._setup_solve_pyamg
setup_det = _M._setup_det
setup_det_backward = _M._setup_det_backward
setup_logdet = _M._setup_logdet
setup_eigsh = _M._setup_eigsh
setup_norm = _M._setup_norm
setup_transpose = _M._setup_transpose
setup_connected_components = _M._setup_cc

# Existing verify helpers, reused.
verify_solve = _M._verify_solve
verify_eigsh = _M._verify_eigsh

# Backend / availability predicates, reused.
is_strumpack_available = _M.is_strumpack_available
is_cudss_available = _M.is_cudss_available
is_pyamg_available = _M.is_pyamg_available


# ---------------------------------------------------------------------------
# NEW setups for the four previously-missing ops.
# ---------------------------------------------------------------------------
def setup_nonlinear_solve(A, dof, device):
    """Newton solve of F(u, A, f) = A @ u + u**3 - f = 0.

    Signature: ``A.nonlinear_solve(residual_fn, u0, *params, method=..., ...)``
    with ``residual_fn(u, A, *params)`` (A is passed in automatically). A is the
    SPD poisson_2d Laplacian, so the Jacobian dF/du = A + diag(3 u**2) stays SPD
    and CG converges. linear_solver='pytorch'/'cg' keeps it device-portable.
    """
    f = torch.ones(dof, dtype=A.dtype, device=device)
    u0 = torch.zeros(dof, dtype=A.dtype, device=device)

    def residual(u, Amat, rhs):
        return (Amat @ u) + u ** 3 - rhs

    def run():
        return A.nonlinear_solve(
            residual, u0, f, method="newton",
            tol=1e-8, max_iter=50, line_search=True,
            linear_solver="pytorch", linear_method="cg",
        )
    return run


def _nonlinear_residual_norm(A, dof, device):
    """Verify: ||F(u*, A, f)|| / ||f|| for the converged nonlinear solution."""
    f = torch.ones(dof, dtype=A.dtype, device=device)
    u0 = torch.zeros(dof, dtype=A.dtype, device=device)

    def residual(u, Amat, rhs):
        return (Amat @ u) + u ** 3 - rhs

    u = A.nonlinear_solve(
        residual, u0, f, method="newton", tol=1e-8, max_iter=50,
        line_search=True, linear_solver="pytorch", linear_method="cg",
    )
    r = (A @ u + u ** 3 - f).norm() / f.norm()
    return float(r.detach().cpu())


def setup_svd(A, dof, device):
    """Truncated SVD: ``A.svd(k=6)`` -> (U, S, Vt).

    CPU only (SciPy svds); CUDA path raises NotImplementedError, so this op's
    availability is gated to device == 'cpu'.
    """
    return lambda: A.svd(k=6)


def _svd_check(A, dof, device):
    """Verify: relative reconstruction residual ||A v_0 - s_0 u_0|| / s_0 for
    the leading singular triple (must be small)."""
    U, S, Vt = A.svd(k=6)
    v0 = Vt[0]
    s0 = S[0]
    u0 = U[:, 0]
    resid = (A @ v0 - s0 * u0).norm() / (s0 + 1e-30)
    return float(resid.detach().cpu())


def setup_condition_number(A, dof, device):
    """Spectral condition number estimate: ``A.condition_number(ord=2)``.

    Uses A.svd internally on CPU -> CPU only (gated like svd).
    """
    return lambda: A.condition_number(ord=2)


def setup_solve_batch(A, dof, device, batch=4):
    """Batched linear solve: stack `batch` copies into a [batch, M, N]
    SparseTensor (shared row/col, per-batch values) and solve [batch, M] RHS.

    A batched SparseTensor takes values of shape [batch, nnz] with shared
    row/col and shape (batch, M, N); ``A_batch.solve(b)`` loops the solve over
    batch elements (see SparseTensor.solve). The poisson_2d Laplacian is SPD so
    the auto backend picks CG.
    """
    M, N = A.sparse_shape
    vals = A.values.unsqueeze(0).repeat(batch, 1).contiguous()
    # perturb each batch element a touch so they are not byte-identical
    scales = (1.0 + 0.01 * torch.arange(batch, dtype=A.dtype, device=device))
    vals = vals * scales.unsqueeze(1)
    A_batch = SparseTensor(vals, A.row_indices, A.col_indices, (batch, M, N))
    b = torch.ones(batch, M, dtype=A.dtype, device=device)
    return lambda: A_batch.solve(b, backend="pytorch", method="cg",
                                 tol=1e-8, maxiter=20000)


def _solve_batch_check(A, dof, device, batch=4):
    """Verify: max relative residual over the batch."""
    M, N = A.sparse_shape
    vals = A.values.unsqueeze(0).repeat(batch, 1).contiguous()
    scales = (1.0 + 0.01 * torch.arange(batch, dtype=A.dtype, device=device))
    vals = vals * scales.unsqueeze(1)
    A_batch = SparseTensor(vals, A.row_indices, A.col_indices, (batch, M, N))
    b = torch.ones(batch, M, dtype=A.dtype, device=device)
    x = A_batch.solve(b, backend="pytorch", method="cg", tol=1e-8, maxiter=20000)
    worst = 0.0
    for i in range(batch):
        Ai = SparseTensor(vals[i], A.row_indices, A.col_indices, (M, N))
        r = (Ai @ x[i] - b[i]).norm() / b[i].norm()
        worst = max(worst, float(r.detach().cpu()))
    return worst
