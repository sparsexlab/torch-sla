"""Tests for the LRU solver cache.

Two layers:

* The :class:`SolverCache` class itself is exercised with synthetic
  build-thunks and arbitrary keys; no matrices involved.
* Backend integration is exercised through the public
  :func:`torch_sla.solve` API on real catalogued benchmarks from
  ``torch_sla.datasets`` (no hand-rolled poisson stencils). The cache
  is *transparent*: there is no ``cache=`` kwarg to flip, repeated
  calls to ``solve`` just reuse setup state on a sparsity-key hit.
  These tests reach into :data:`SOLVER_CACHE.stats()` to assert the
  reuse, but user code never has to.
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
from torch_sla.datasets import Synthetic


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
    cache.get_or_build("a", lambda: "REBUILT")  # touch -> 'a' becomes MRU
    cache.get_or_build("c", lambda: "C")        # 'b' should now be evicted
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
# SparsityKey identity, driven by the catalogue
# =====================================================================
def test_same_matrix_makes_equal_keys():
    """Two ``make_key`` calls on the same catalogued matrix must agree
    bit-for-bit -- otherwise the cache fingerprint is unstable."""
    b = Synthetic["poisson_2d_16"]
    k1 = make_key(b.val, b.row, b.col, b.shape)
    k2 = make_key(b.val, b.row, b.col, b.shape)
    assert k1 == k2
    assert hash(k1) == hash(k2)


def test_perturbed_values_make_unequal_keys():
    """Mutating one entry of ``val`` must shift the key -- guards
    against same-pattern-different-values cache collisions."""
    b = Synthetic["poisson_2d_16"]
    val2 = b.val.clone()
    val2[0] += 1.0
    k1 = make_key(b.val, b.row, b.col, b.shape)
    k2 = make_key(val2, b.row, b.col, b.shape)
    assert k1 != k2


def test_different_catalogue_matrices_make_unequal_keys():
    """Distinct catalogued matrices produce distinct keys."""
    a = Synthetic["poisson_2d_16"]
    b = Synthetic["poisson_2d_64"]
    ka = make_key(a.val, a.row, a.col, a.shape)
    kb = make_key(b.val, b.row, b.col, b.shape)
    assert ka != kb


# =====================================================================
# End-to-end: cache works transparently through the public solve() API
# =====================================================================
pyamg_only = pytest.mark.skipif(
    not is_pyamg_available(),
    reason="pyamg is an optional dependency; install with `pip install pyamg`."
)


@pyamg_only
def test_solve_reuses_hierarchy_across_rhs_on_same_matrix(benchmark_small_real):
    """Solving the same A with several different right-hand sides
    only builds the AMG hierarchy once -- the user notices the speedup
    without touching any ``cache=`` flag, since there isn't one."""
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
def test_solve_misses_on_distinct_catalogue_matrices(benchmark_small_real):
    """Two structurally different catalogued matrices both miss on
    their first solve and the cache holds both entries afterwards."""
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
def test_cache_clear_forces_rebuild_on_next_solve(benchmark_small_real):
    """``SOLVER_CACHE.clear()`` is the escape hatch when users do need
    to force a rebuild (testing, benchmarking, deliberate invalidation).
    Demonstrated to exist; doesn't appear in user APIs."""
    bench = benchmark_small_real
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

    SOLVER_CACHE.clear()
    baseline = SOLVER_CACHE.stats()["misses"]

    solve(A, bench[0]["b"], backend="pyamg", maxiter=30)
    assert SOLVER_CACHE.stats()["misses"] - baseline == 1

    SOLVER_CACHE.clear()
    solve(A, bench[0]["b"], backend="pyamg", maxiter=30)
    assert SOLVER_CACHE.stats()["misses"] - baseline == 2, (
        "after clear the second solve must miss again"
    )


@pyamg_only
def test_pyamg_preconditioner_factory_reuses_cache(benchmark_small_real):
    """The standalone ``pyamg_preconditioner`` factory returns the
    cached hierarchy when called twice on the same matrix -- the
    factory and the solver share one cache."""
    from torch_sla.backends.pyamg_backend import pyamg_preconditioner
    SOLVER_CACHE.clear()
    b = benchmark_small_real
    H1 = pyamg_preconditioner(b.val, b.row, b.col, b.shape)
    H2 = pyamg_preconditioner(b.val, b.row, b.col, b.shape)
    assert H1 is H2, (
        "second factory call should return the cached hierarchy"
    )
