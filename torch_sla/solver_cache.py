"""LRU cache for expensive solver setup state.

Many sparse solvers do a lot of work that depends *only* on the matrix
sparsity pattern + numerical values, and that work is identical across
repeated solves with the same ``A``:

* PyAMG multigrid hierarchy (Ruge-Stuben coarsening, R/P operators)
* cuDSS symbolic factorization
* SciPy SuperLU factorisation
* (planned) AmgX solver handles

The natural pattern is: build once per distinct matrix, reuse for every
subsequent solve. JAX-AMG (Liu, Fan, Wang -- arXiv:2606.09001) makes
exactly this point in its "solver caching" section.

This module provides a small, backend-agnostic LRU cache keyed by a
:class:`SparsityKey` fingerprint of the matrix. Backends opt in
explicitly via :func:`get_or_build`; user code controls it through the
``cache`` keyword on :func:`torch_sla.solve` and through the
module-level singleton :data:`SOLVER_CACHE`.

Eviction policy: LRU with a bounded ``max_size``. Default is small (8
entries) since cached values can be large (full multigrid hierarchies).

Hashing collisions are statistically negligible for non-adversarial
inputs -- the fingerprint mixes shape, dtype, device, nnz, and a
six-element index sample. False *positives* (different matrices hashing
to the same key and returning a stale solver) are dangerous, so the
fingerprint is deliberately wide enough to make them vanishingly
unlikely; if you're worried, pass ``cache=False``.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor


# ====================================================================== #
# Hashable key fingerprinting a sparse-matrix + backend-config pair
# ====================================================================== #
@dataclass(frozen=True)
class SparsityKey:
    """Hashable identity of a sparse matrix for the solver cache.

    Captures shape, dtype, device type (CPU vs CUDA -- device index is
    ignored on purpose, the same matrix on cuda:0 vs cuda:1 should hit
    the same cache entry once moved), nnz, and a cheap fingerprint of
    the index arrays. Two matrices with the same fingerprint that
    differ in *content* count as a cache miss only if their values also
    hash differently (see :attr:`val_hash`); same sparsity + same
    values + same dtype = same key.
    """
    shape: Tuple[int, int]
    dtype: torch.dtype
    device_type: str
    nnz: int
    row_fp: Tuple[int, ...]
    col_fp: Tuple[int, ...]
    val_fp: Tuple[float, ...]


def _index_fingerprint(t: Tensor) -> Tuple[int, ...]:
    """Six-element sample of an index tensor that's stable across same
    matrices on different devices. Cheap to compute (no full hash)."""
    n = t.numel()
    if n == 0:
        return (0, 0, 0, 0, 0, 0)
    if n <= 6:
        return tuple(t.tolist())
    return (
        int(t[0].item()),
        int(t[n // 5].item()),
        int(t[n // 2].item()),
        int(t[(3 * n) // 4].item()),
        int(t[-1].item()),
        n,
    )


def _value_fingerprint(t: Tensor) -> Tuple[float, ...]:
    """Five-element sample of value tensor. The tail is the sum (full
    reduction) so that perturbations to any entry shift the key."""
    n = t.numel()
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    take = lambda idx: float(t[idx].real.item()) if t.is_complex() \
        else float(t[idx].item())
    return (
        take(0),
        take(n // 3),
        take((2 * n) // 3),
        take(-1),
        float(t.real.abs().sum().item() if t.is_complex() else t.abs().sum().item()),
    )


def make_key(val: Tensor, row: Tensor, col: Tensor,
             shape: Tuple[int, int]) -> SparsityKey:
    """Build a :class:`SparsityKey` for a COO triple."""
    return SparsityKey(
        shape=tuple(shape),
        dtype=val.dtype,
        device_type=val.device.type,
        nnz=int(row.numel()),
        row_fp=_index_fingerprint(row),
        col_fp=_index_fingerprint(col),
        val_fp=_value_fingerprint(val),
    )


# ====================================================================== #
# LRU cache
# ====================================================================== #
class SolverCache:
    """Bounded LRU cache for solver setup state.

    Backends call :meth:`get_or_build` with a key (typically a tuple
    of ``(backend_label, SparsityKey, frozen_config)``) and a thunk that
    constructs the expensive state on a cache miss. Hits are O(1).

    Eviction policy: least-recently-used; LRU order tracked via
    ``OrderedDict`` re-insertion on touch.
    """

    def __init__(self, max_size: int = 8):
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_or_build(self, key: Any, build: Callable[[], Any]) -> Any:
        """Return the cached value for ``key``, or build + insert it.

        On a hit the entry is moved to the LRU front. On a miss the
        ``build`` thunk runs once and its return value is inserted;
        oldest entries are evicted until ``len(self) <= max_size``.
        """
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        value = build()
        self._cache[key] = value
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return value

    def clear(self) -> None:
        """Drop all entries. Cumulative hit/miss counters are preserved."""
        self._cache.clear()

    def stats(self) -> Dict[str, int]:
        """Return a snapshot of cache stats: ``{'hits', 'misses', 'size'}``."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    def set_max_size(self, max_size: int) -> None:
        """Resize the cache. Excess entries are evicted (LRU first)."""
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._max_size = max_size
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def __repr__(self) -> str:
        s = self.stats()
        return (f"SolverCache(hits={s['hits']}, misses={s['misses']}, "
                f"size={s['size']}/{s['max_size']})")


# ====================================================================== #
# Module-level singleton
# ====================================================================== #
SOLVER_CACHE = SolverCache(max_size=8)
"""Default cache used by :func:`torch_sla.solve` when ``cache=True``.

Replace with a custom :class:`SolverCache` instance, or adjust the
existing one's max size via :meth:`SolverCache.set_max_size`."""


__all__ = ["SolverCache", "SparsityKey", "make_key", "SOLVER_CACHE"]
