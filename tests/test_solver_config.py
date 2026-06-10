"""Tests for :class:`SolverConfig` -- scoped defaults for ``solve()``.

Three forms exercised:

* **Context manager** -- ``with SolverConfig(...): solve(...)`` picks
  up the scope's defaults.
* **Decorator** -- ``@SolverConfig(...) def f(): solve(...)`` attaches
  defaults to a function.
* **Nested scopes** -- inner non-None fields override outer; explicit
  kwargs on the ``solve`` call always beat both.

All matrices come from ``torch_sla.datasets`` -- no hand-rolled
stencils.
"""
from __future__ import annotations

import threading

import pytest
import torch

from torch_sla import (
    PreconditionerConfig,
    SolverConfig,
    SparseTensor,
    solve,
)
from torch_sla.solve import _STACK, _active_defaults
from torch_sla.datasets import Synthetic


@pytest.fixture(autouse=True)
def _reset_stack():
    """Make every test start with an empty defaults stack."""
    _STACK.stack.clear()
    yield
    _STACK.stack.clear()


# =====================================================================
# Pure stack mechanics (no solve involved)
# =====================================================================
def test_inactive_defaults_dict_is_empty():
    assert _active_defaults() == {}


def test_single_scope_populates_defaults():
    with SolverConfig(backend="pytorch", method="cg"):
        d = _active_defaults()
    assert d == {"backend": "pytorch", "method": "cg"}
    assert _active_defaults() == {}, "exit must pop the layer"


def test_nested_scopes_merge_inner_wins():
    with SolverConfig(backend="pytorch", atol=1e-8):
        with SolverConfig(atol=1e-12, method="cg"):
            d = _active_defaults()
            assert d["backend"] == "pytorch"   # from outer
            assert d["method"]  == "cg"        # from inner
            assert d["atol"]    == 1e-12       # inner wins


def test_none_fields_are_inactive():
    """Leaving a field at its default (None) means \"don't touch\" --
    the merged dict shouldn't carry it."""
    with SolverConfig(method="cg"):
        d = _active_defaults()
    assert "method" in d
    assert "atol" not in d, "unset fields should not appear in defaults"


def test_solverconfig_is_frozen():
    """Frozen dataclass -- can't tweak after construction. That's the
    contract that lets us hash it / reuse it across scopes."""
    cfg = SolverConfig(method="cg")
    with pytest.raises((AttributeError,)):
        cfg.method = "bicgstab"  # type: ignore[misc]


# =====================================================================
# End-to-end: defaults reach ``solve``
# =====================================================================
def test_scoped_backend_changes_what_solve_dispatches_to():
    """In a SolverConfig(backend='pytorch') scope, solve uses pytorch
    even though we never passed backend explicitly. Demonstrated by
    asking for the dispatch info."""
    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
    rhs = bench[0]["b"]

    with SolverConfig(backend="pytorch", method="cg", atol=1e-10, maxiter=500):
        x, info = solve(A, rhs, return_info=True)
    assert info.backend == "pytorch"
    assert info.method == "cg"


def test_explicit_kwarg_beats_scope():
    """Explicit kwargs at the call site override the scope."""
    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
    rhs = bench[0]["b"]

    with SolverConfig(backend="pytorch", method="cg"):
        x, info = solve(A, rhs, backend="scipy", method="lu",
                        return_info=True)
    assert info.backend == "scipy"
    assert info.method == "lu"


def test_decorator_form_attaches_scope_to_function():
    """``@SolverConfig(...)`` applied to a function makes every solve
    inside that function inherit the scope."""
    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
    rhs = bench[0]["b"]

    @SolverConfig(backend="pytorch", method="cg", atol=1e-10, maxiter=500)
    def my_pipeline():
        return solve(A, rhs, return_info=True)

    x, info = my_pipeline()
    assert info.backend == "pytorch"
    assert info.method == "cg"


def test_decorator_does_not_leak_after_return():
    """After the decorated function returns, the scope is gone --
    subsequent solves see empty defaults."""
    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
    rhs = bench[0]["b"]

    @SolverConfig(backend="pytorch", method="cg")
    def scoped():
        return solve(A, rhs, return_info=True)

    scoped()
    assert _active_defaults() == {}


def test_nested_scopes_inner_wins_in_solve():
    """Two nested scopes; inner's tolerance wins."""
    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
    rhs = bench[0]["b"]

    # outer atol=1e-2 (would stop early); inner atol=1e-10 should
    # demonstrably push the solve much tighter.
    with SolverConfig(backend="pytorch", method="cg", atol=1e-2):
        with SolverConfig(atol=1e-10):
            x, info = solve(A, rhs, maxiter=2000, return_info=True)
    assert info.residual < 1e-7, (
        f"inner atol=1e-10 should beat outer atol=1e-2; got resid={info.residual}"
    )


def test_preconditioner_config_inside_solver_config():
    """A ``PreconditionerConfig`` instance can be the value of
    ``SolverConfig.preconditioner`` -- the dataclass nests cleanly."""
    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
    rhs = bench[0]["b"]

    with SolverConfig(
        backend="pytorch",
        method="cg",
        preconditioner=PreconditionerConfig(kind="jacobi"),
        atol=1e-10, maxiter=500,
    ):
        x, info = solve(A, rhs, return_info=True)
    assert info.converged
    assert ((x - bench[0]["x"]).norm() / bench[0]["x"].norm()).item() < 1e-7


def test_preconditioner_none_is_a_valid_choice():
    """``preconditioner=None`` is a legitimate user choice meaning 'no
    preconditioning'. The dataclass uses an internal sentinel so we
    can distinguish 'unset' from 'explicitly None'."""
    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
    rhs = bench[0]["b"]

    with SolverConfig(backend="pytorch", method="cg",
                      preconditioner=None,
                      atol=1e-10, maxiter=2000):
        x, info = solve(A, rhs, return_info=True)
    assert info.backend == "pytorch"
    assert info.method == "cg"


# =====================================================================
# Thread isolation
# =====================================================================
def test_stack_is_per_thread():
    """A scope opened on one thread should not be visible to another."""
    barrier = threading.Barrier(2)
    seen_in_worker = {}

    def worker():
        # Wait for the main thread to open its scope.
        barrier.wait()
        seen_in_worker["defaults"] = _active_defaults()
        barrier.wait()

    t = threading.Thread(target=worker)
    t.start()
    with SolverConfig(backend="pytorch", method="cg"):
        barrier.wait()                  # let the worker peek
        barrier.wait()                  # let the worker finish
    t.join()

    assert seen_in_worker["defaults"] == {}, (
        "the worker thread should see no defaults"
    )


# =====================================================================
# Regression: solve() without any scope behaves like before
# =====================================================================
def test_solve_without_scope_is_unchanged():
    """No scope active = solve() runs with the same hard-coded defaults
    as before this PR."""
    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
    rhs = bench[0]["b"]
    x = solve(A, rhs)
    assert ((x - bench[0]["x"]).norm() / bench[0]["x"].norm()).item() < 1e-12
