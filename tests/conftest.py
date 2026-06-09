"""Shared pytest fixtures backed by ``torch_sla.datasets``.

Fixtures are deliberately split by source / kind so individual tests can
opt into the appropriate tier instead of every test paying for every
matrix:

==============================  ==================================  =======
Fixture                         Yields                              Network
==============================  ==================================  =======
``benchmark_suitesparse``       every SuiteSparse benchmark         yes
``benchmark_dimacs``            every DIMACS10 graph Laplacian      yes
``benchmark_synthetic``         every Synthetic PDE stencil         no
``benchmark_real``              every real-dtype benchmark          partial
``benchmark_complex``           every complex-dtype benchmark       partial
``benchmark_small_real``        a single tiny synthetic (n^2=256)   no
==============================  ==================================  =======

When a matrix cannot be downloaded (no internet, mirror down, ...) the
test is skipped via :func:`pytest.skip` rather than failing -- so the
library is air-gapped-CI friendly.

Tests that genuinely want exhaustive coverage can iterate
``torch_sla.datasets.iter_benchmarks()`` directly.
"""
from __future__ import annotations

import pytest

from torch_sla.benchmark import Benchmark
from torch_sla.datasets import (
    Benchmarks,
    DatasetUnavailable,
    Synthetic,
)


def _safe_get(source: str, key: str) -> Benchmark:
    try:
        return Benchmarks[source][key]
    except DatasetUnavailable as e:
        pytest.skip(f"{source}:{key}: {e}")


# ---------------------------------------------------------------------- #
# By source
# ---------------------------------------------------------------------- #
@pytest.fixture(params=list(Benchmarks["suitesparse"].keys()))
def benchmark_suitesparse(request) -> Benchmark:
    """Each catalogued SuiteSparse matrix, one per test invocation.

    Needs network on first call; subsequent runs hit the cache.
    """
    return _safe_get("suitesparse", request.param)


@pytest.fixture(params=list(Benchmarks["dimacs10"].keys()))
def benchmark_dimacs(request) -> Benchmark:
    """Each catalogued DIMACS10 graph Laplacian.

    Downloaded from the SuiteSparse mirror (group ``DIMACS10``) and
    converted to ``L + eps*I`` so the solve target is SPD.
    """
    return _safe_get("dimacs10", request.param)


@pytest.fixture(params=list(Benchmarks["synthetic"].keys()))
def benchmark_synthetic(request) -> Benchmark:
    """Each Synthetic PDE stencil. No network required."""
    return _safe_get("synthetic", request.param)


# ---------------------------------------------------------------------- #
# By dtype (mix all sources)
# ---------------------------------------------------------------------- #
# Hardcoded carve-out at the catalogue-key level so we don't have to
# instantiate every benchmark (which would trigger network downloads
# during pytest collection) just to read its dtype. ``"complex"`` in the
# SuiteSparse key and ``"helmholtz"`` in the Synthetic key are the only
# complex-dtype entries; everything else is real.
def _is_complex_key(source: str, key: str) -> bool:
    if source == "suitesparse":
        return "complex" in key
    if source == "synthetic":
        return "helmholtz" in key
    return False  # DIMACS10 entries are real Laplacians


_ALL_KEYS = [
    (src, key) for src, coll in Benchmarks.items() for key in coll
]
_REAL_KEYS = [(s, k) for s, k in _ALL_KEYS if not _is_complex_key(s, k)]
_COMPLEX_KEYS = [(s, k) for s, k in _ALL_KEYS if _is_complex_key(s, k)]


@pytest.fixture(params=_REAL_KEYS, ids=[f"{s}:{k}" for s, k in _REAL_KEYS])
def benchmark_real(request) -> Benchmark:
    """Every real-dtype benchmark across all three sources."""
    source, key = request.param
    return _safe_get(source, key)


@pytest.fixture(params=_COMPLEX_KEYS, ids=[f"{s}:{k}" for s, k in _COMPLEX_KEYS])
def benchmark_complex(request) -> Benchmark:
    """Every complex-dtype benchmark across all three sources."""
    source, key = request.param
    return _safe_get(source, key)


# ---------------------------------------------------------------------- #
# Fast smoke
# ---------------------------------------------------------------------- #
@pytest.fixture
def benchmark_small_real() -> Benchmark:
    """A single 16x16 = 256-DOF synthetic Poisson 2D -- for very fast
    correctness smoke checks (gradcheck, residual sweep, ...)."""
    return Synthetic["poisson_2d_16"]


@pytest.fixture
def benchmark_small_complex() -> Benchmark:
    """The smallest catalogued complex matrix (Bai/qc324, n=324) -- for
    complex-dtype fast tests like ``autograd.gradcheck`` whose runtime is
    ``O(nnz)`` per finite-difference probe."""
    return _safe_get("suitesparse", "complex_sym")
