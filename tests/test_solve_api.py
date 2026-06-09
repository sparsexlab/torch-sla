"""Tests for the new top-level :func:`torch_sla.solve` API.

Covers:

* the five accepted matrix-input formats (SparseTensor, tuple,
  scipy.sparse, dense tensor, matrix-free callable)
* the ``return_info=True`` ``(x, SolveInfo)`` return
* ``PreconditionerConfig`` interop with string / dataclass forms
* parity with the legacy ``torch_sla.spsolve``
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from torch_sla import (
    PreconditionerConfig,
    SolveInfo,
    SparseTensor,
    solve,
    spsolve,
)


# =====================================================================
# Builds a small SPD diagonal system used across the suite
# =====================================================================
@pytest.fixture
def tiny_spd():
    n = 5
    diag = torch.tensor([4., 5., 6., 7., 8.], dtype=torch.float64)
    val = diag.clone()
    row = torch.arange(n, dtype=torch.long)
    col = torch.arange(n, dtype=torch.long)
    b = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
    x_ref = b / diag
    return dict(n=n, diag=diag, val=val, row=row, col=col, b=b, x_ref=x_ref)


# =====================================================================
# Input formats
# =====================================================================
def test_solve_accepts_dense_tensor(tiny_spd):
    A = torch.diag(tiny_spd["diag"])
    x = solve(A, tiny_spd["b"])
    assert torch.allclose(x, tiny_spd["x_ref"], atol=1e-10)


def test_solve_accepts_sparse_tensor(tiny_spd):
    A = SparseTensor(tiny_spd["val"], tiny_spd["row"], tiny_spd["col"],
                     (tiny_spd["n"], tiny_spd["n"]))
    x = solve(A, tiny_spd["b"])
    assert torch.allclose(x, tiny_spd["x_ref"], atol=1e-10)


def test_solve_accepts_4_tuple(tiny_spd):
    x = solve(
        (tiny_spd["val"], tiny_spd["row"], tiny_spd["col"],
         (tiny_spd["n"], tiny_spd["n"])),
        tiny_spd["b"],
    )
    assert torch.allclose(x, tiny_spd["x_ref"], atol=1e-10)


def test_solve_accepts_scipy_sparse(tiny_spd):
    A_sp = sp.diags(tiny_spd["diag"].numpy()).tocoo()
    x = solve(A_sp, tiny_spd["b"])
    assert torch.allclose(x, tiny_spd["x_ref"], atol=1e-10)


def test_solve_accepts_matrix_free_callable(tiny_spd):
    diag = tiny_spd["diag"]
    matvec = lambda x: diag * x  # noqa: E731 -- compact for the test
    x = solve(matvec, tiny_spd["b"], shape=(tiny_spd["n"], tiny_spd["n"]))
    assert torch.allclose(x, tiny_spd["x_ref"], atol=1e-10)


def test_solve_callable_without_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        solve(lambda x: x, torch.ones(3))


def test_solve_rejects_unknown_type():
    with pytest.raises(TypeError, match="Unsupported"):
        solve("not-a-matrix", torch.ones(3))


# =====================================================================
# (x, info) return
# =====================================================================
def test_return_info_yields_solveinfo(tiny_spd):
    A = SparseTensor(tiny_spd["val"], tiny_spd["row"], tiny_spd["col"],
                     (tiny_spd["n"], tiny_spd["n"]))
    out = solve(A, tiny_spd["b"], return_info=True)
    assert isinstance(out, tuple) and len(out) == 2
    x, info = out
    assert torch.allclose(x, tiny_spd["x_ref"], atol=1e-10)
    assert isinstance(info, SolveInfo)
    assert info.converged
    assert info.residual < 1e-9


def test_return_info_default_off(tiny_spd):
    A = SparseTensor(tiny_spd["val"], tiny_spd["row"], tiny_spd["col"],
                     (tiny_spd["n"], tiny_spd["n"]))
    out = solve(A, tiny_spd["b"])  # default
    assert isinstance(out, torch.Tensor)


# =====================================================================
# PreconditionerConfig
# =====================================================================
def test_preconditioner_string_shortcut(tiny_spd):
    A = SparseTensor(tiny_spd["val"], tiny_spd["row"], tiny_spd["col"],
                     (tiny_spd["n"], tiny_spd["n"]))
    x = solve(A, tiny_spd["b"], method="cg", preconditioner="jacobi")
    assert torch.allclose(x, tiny_spd["x_ref"], atol=1e-9)


def test_preconditioner_dataclass(tiny_spd):
    A = SparseTensor(tiny_spd["val"], tiny_spd["row"], tiny_spd["col"],
                     (tiny_spd["n"], tiny_spd["n"]))
    cfg = PreconditionerConfig(kind="jacobi")
    x = solve(A, tiny_spd["b"], method="cg", preconditioner=cfg)
    assert torch.allclose(x, tiny_spd["x_ref"], atol=1e-9)


def test_preconditioner_none(tiny_spd):
    """``preconditioner=None`` should be honoured and routed as ``'none'``."""
    A = SparseTensor(tiny_spd["val"], tiny_spd["row"], tiny_spd["col"],
                     (tiny_spd["n"], tiny_spd["n"]))
    x = solve(A, tiny_spd["b"], method="cg", preconditioner=None)
    assert torch.allclose(x, tiny_spd["x_ref"], atol=1e-9)


def test_preconditioner_wrong_type_raises():
    A = SparseTensor(torch.ones(3), torch.arange(3), torch.arange(3), (3, 3))
    with pytest.raises(TypeError, match="preconditioner"):
        solve(A, torch.ones(3), preconditioner=42)  # type: ignore[arg-type]


def test_preconditioner_config_defaults():
    cfg = PreconditionerConfig()
    assert cfg.kind == "jacobi"
    assert cfg.omega == 1.0
    assert cfg.amg_strength == 0.25


def test_preconditioner_config_is_frozen():
    """The dataclass is frozen so callers can stash it in a dict / use it
    as a cache key without worrying about silent mutation."""
    cfg = PreconditionerConfig(kind="ssor", omega=1.4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.omega = 1.0  # type: ignore[misc]


# =====================================================================
# Parity with the legacy spsolve
# =====================================================================
def test_solve_matches_legacy_spsolve(tiny_spd):
    val, row, col, b = (tiny_spd["val"], tiny_spd["row"],
                        tiny_spd["col"], tiny_spd["b"])
    x_legacy = spsolve(val, row, col, (tiny_spd["n"], tiny_spd["n"]), b)
    x_new = solve((val, row, col, (tiny_spd["n"], tiny_spd["n"])), b)
    assert torch.allclose(x_legacy, x_new, atol=1e-12)


# =====================================================================
# Larger SuiteSparse smoke -- exercises one real benchmark via solve()
# =====================================================================
def test_solve_on_suitesparse_real_spd(benchmark_small_real):
    """``solve`` works on a real SuiteSparse / synthetic benchmark via the
    new API; round-trips A @ x_ref = b back to x_ref."""
    b = benchmark_small_real
    rhs = b[0]["b"]
    x_ref = b[0]["x"]
    A = SparseTensor(b.val, b.row, b.col, b.shape)
    x = solve(A, rhs)
    rel_err = (x - x_ref).norm() / x_ref.norm()
    assert rel_err.item() < 1e-9
