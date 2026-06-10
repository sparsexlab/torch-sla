"""Tests for the LRU solver cache.

The cache is backend-agnostic and currently consumed by the PyAMG
hybrid backend (PR #14 / #15). These tests cover both the cache class
itself in isolation *and* its integration through ``solve()``.
"""
from __future__ import annotations

import pytest
import torch

from torch_sla import (
    SOLVER_CACHE,
    SolverCache,
    SparseTensor,
    SparsityKey,
    solve,
)
from torch_sla.solver_cache import make_key
from torch_sla.backends import is_pyamg_available


# =====================================================================
# Pure-cache behaviour (no backends involved)
# =====================================================================
def test_cache_hit_returns_same_object():
    cache = SolverCache(max_size=4)
    sentinel = object()
    built = []

    def build():
        built.append(1)
        return sentinel

    a = cache.get_or_build("k", build)
    b = cache.get_or_build("k", build)
    assert a is sentinel and b is sentinel
    assert len(built) == 1, "build should run once across two hits"
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1


def test_cache_lru_eviction_evicts_oldest():
    cache = SolverCache(max_size=2)
    cache.get_or_build("a", lambda: "A")
    cache.get_or_build("b", lambda: "B")
    cache.get_or_build("c", lambda: "C")    # 'a' should be evicted
    assert "a" not in cache
    assert "b" in cache and "c" in cache


def test_cache_touch_moves_to_front():
    cache = SolverCache(max_size=2)
    cache.get_or_build("a", lambda: "A")
    cache.get_or_build("b", lambda: "B")
    # Touch 'a' so it becomes most-recently-used.
    cache.get_or_build("a", lambda: "REBUILT")
    cache.get_or_build("c", lambda: "C")    # 'b' should now be evicted, not 'a'
    assert "a" in cache and "c" in cache
    assert "b" not in cache


def test_cache_clear_drops_entries_keeps_counters():
    cache = SolverCache(max_size=4)
    cache.get_or_build("a", lambda: "A")
    cache.get_or_build("a", lambda: "A")    # hit
    cache.clear()
    s = cache.stats()
    assert s["size"] == 0
    assert s["hits"] == 1 and s["misses"] == 1, (
        "clear() preserves cumulative hit/miss counters"
    )


def test_cache_set_max_size_evicts_overflow():
    cache = SolverCache(max_size=4)
    for k in "abcd":
        cache.get_or_build(k, lambda v=k: v)
    cache.set_max_size(2)
    assert len(cache) == 2, "shrinking max_size should evict LRU entries"


def test_cache_max_size_zero_raises():
    with pytest.raises(ValueError, match="max_size"):
        SolverCache(max_size=0)


def test_cache_repr_includes_stats():
    cache = SolverCache(max_size=4)
    cache.get_or_build("a", lambda: "A")
    s = repr(cache)
    assert "hits=0" in s and "misses=1" in s and "size=1" in s


# =====================================================================
# SparsityKey identity
# =====================================================================
def test_same_matrix_makes_equal_keys():
    n = 16
    val = torch.tensor([2.0, 3.0, 4.0])
    row = torch.tensor([0, 1, 2])
    col = torch.tensor([0, 1, 2])
    k1 = make_key(val, row, col, (n, n))
    k2 = make_key(val, row, col, (n, n))
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_perturbed_values_make_unequal_keys():
    n = 16
    row = torch.tensor([0, 1, 2])
    col = torch.tensor([0, 1, 2])
    k1 = make_key(torch.tensor([2.0, 3.0, 4.0]), row, col, (n, n))
    k2 = make_key(torch.tensor([2.0, 3.0, 5.0]), row, col, (n, n))
    assert k1 != k2, "different values should produce different keys"


def test_different_shapes_make_unequal_keys():
    val = torch.tensor([1.0])
    row = torch.tensor([0])
    col = torch.tensor([0])
    k1 = make_key(val, row, col, (3, 3))
    k2 = make_key(val, row, col, (4, 4))
    assert k1 != k2


# =====================================================================
# End-to-end: PyAMG hierarchy reuse via solve()
# =====================================================================
pyamg_only = pytest.mark.skipif(
    not is_pyamg_available(),
    reason="pyamg is an optional dependency; install with `pip install pyamg`."
)


@pyamg_only
def test_solve_pyamg_reuses_hierarchy_across_rhs(benchmark_small_real):
    """Solving the same A with three different RHS only builds the
    hierarchy once."""
    SOLVER_CACHE.clear()
    before = SOLVER_CACHE.stats()
    bench = benchmark_small_real
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

    for case in bench:
        solve(A, case["b"], backend="pyamg", atol=1e-8, maxiter=30)

    after = SOLVER_CACHE.stats()
    misses = after["misses"] - before["misses"]
    hits = after["hits"] - before["hits"]
    assert misses == 1, f"expected 1 build, got {misses}"
    assert hits == len(bench) - 1, (
        f"expected {len(bench) - 1} hits, got {hits}"
    )


@pyamg_only
def test_solve_pyamg_cache_false_forces_rebuild(benchmark_small_real):
    """Passing ``cache=False`` through the explicit pyamg_solve entry
    point skips both the lookup and the insert."""
    from torch_sla.backends.pyamg_backend import pyamg_solve
    SOLVER_CACHE.clear()
    before = SOLVER_CACHE.stats()
    bench = benchmark_small_real
    for case in bench:
        pyamg_solve(bench.val, bench.row, bench.col, bench.shape,
                    case["b"], cache=False, tol=1e-8, maxiter=30)
    after = SOLVER_CACHE.stats()
    assert after == before, (
        f"cache=False should not touch SOLVER_CACHE; before={before}, after={after}"
    )


@pyamg_only
def test_solve_pyamg_different_matrix_misses(benchmark_small_real):
    """Two structurally different matrices both miss on first solve."""
    from torch_sla.datasets import Synthetic
    SOLVER_CACHE.clear()
    a = benchmark_small_real
    Aa = SparseTensor(a.val, a.row, a.col, a.shape)
    b = Synthetic["poisson_2d_64"]
    Ab = SparseTensor(b.val, b.row, b.col, b.shape)

    solve(Aa, a[0]["b"], backend="pyamg", maxiter=30)
    solve(Ab, b[0]["b"], backend="pyamg", maxiter=30)

    stats = SOLVER_CACHE.stats()
    assert stats["misses"] >= 2
    assert stats["size"] >= 2


@pyamg_only
def test_pyamg_preconditioner_factory_uses_cache(benchmark_small_real):
    """The standalone ``pyamg_preconditioner`` factory returns the
    cached hierarchy when called twice on the same matrix."""
    from torch_sla.backends.pyamg_backend import pyamg_preconditioner
    SOLVER_CACHE.clear()
    b = benchmark_small_real
    H1 = pyamg_preconditioner(b.val, b.row, b.col, b.shape)
    H2 = pyamg_preconditioner(b.val, b.row, b.col, b.shape)
    assert H1 is H2, (
        "second factory call should return the cached hierarchy"
    )
