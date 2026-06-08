"""Curated sparse-matrix benchmarks exposed as :class:`Benchmark` mappings.

Three benchmark families are provided:

* :data:`SuiteSparse` — curated subset of the SuiteSparse Matrix Collection
  (Tim Davis et al.), real-world FE / CFD / structural matrices.
* :data:`DIMACS10` — curated graph Laplacians from the 10th DIMACS
  Implementation Challenge (graph partitioning / clustering), downloaded
  through the SuiteSparse mirror. Adjacency matrices are converted to
  regularised Laplacians ``L + eps*I`` so they are solvable.
* :data:`Synthetic` — programmatic PDE stencil generators (2D/3D Poisson,
  anisotropic, convection-diffusion, Helmholtz). No download required.

Each entry is a :class:`torch_sla.benchmark.Benchmark` instance with
three random ``(x_ref, b)`` reference cases.

The download cache directory is controlled by the ``TORCH_SLA_DATASET``
environment variable, defaulting to ``~/.cache/torch_sla/datasets``::

    export TORCH_SLA_DATASET=/path/to/large/disk
"""

from __future__ import annotations

import io as _io
import os
import tarfile
import urllib.request
from collections.abc import Mapping
from typing import Any, Callable, Dict, Iterator, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from scipy.io import mmread

from .benchmark import Benchmark


# ---------------------------------------------------------------------- #
# Cache directory
# ---------------------------------------------------------------------- #
def cache_dir() -> str:
    """Return the dataset cache directory, honouring ``TORCH_SLA_DATASET``.

    Default: ``~/.cache/torch_sla/datasets``. Created on first call.
    """
    raw = os.environ.get("TORCH_SLA_DATASET",
                         os.path.join("~", ".cache", "torch_sla", "datasets"))
    path = os.path.expanduser(raw)
    os.makedirs(path, exist_ok=True)
    return path


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset cannot be fetched (no internet, mirror down, ...).

    Library code raises this rather than calling ``pytest.skip``; tests
    catch it and skip individually.
    """


# ====================================================================== #
# SuiteSparse-mirror download helper (shared by SuiteSparse + DIMACS10)
# ====================================================================== #
_MIRROR = ("https://suitesparse-collection-website.herokuapp.com/MM/"
           "{group}/{name}.tar.gz")


def _ensure_downloaded(group: str, name: str) -> str:
    """Return ``<cache>/<name>.mtx``, downloading from the SuiteSparse mirror
    if not yet cached. Raises :class:`DatasetUnavailable` on failure."""
    out = os.path.join(cache_dir(), f"{name}.mtx")
    if os.path.exists(out):
        return out
    url = _MIRROR.format(group=group, name=name)
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
    except Exception as exc:
        raise DatasetUnavailable(
            f"failed to download {group}/{name} from {url}: {exc}"
        ) from exc
    try:
        with tarfile.open(fileobj=_io.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                if m.name.endswith(f"{name}.mtx"):
                    with open(out, "wb") as f:
                        f.write(tf.extractfile(m).read())
                    return out
    except Exception as exc:
        raise DatasetUnavailable(
            f"failed to extract {name}.mtx from tarball: {exc}"
        ) from exc
    raise DatasetUnavailable(
        f"tarball for {group}/{name} did not contain {name}.mtx"
    )


def _load_as_triple(path: str, *, dtype) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, Tuple[int, int]]:
    A = mmread(path).tocoo()
    if dtype is not None:
        A = A.astype(dtype)
    val = torch.from_numpy(A.data)
    row = torch.from_numpy(A.row.astype(np.int64))
    col = torch.from_numpy(A.col.astype(np.int64))
    return val, row, col, (A.shape[0], A.shape[1])


def _scipy_to_triple(A: sp.spmatrix) -> Tuple[torch.Tensor, torch.Tensor,
                                              torch.Tensor, Tuple[int, int]]:
    A = A.tocoo()
    val = torch.from_numpy(np.ascontiguousarray(A.data))
    row = torch.from_numpy(A.row.astype(np.int64))
    col = torch.from_numpy(A.col.astype(np.int64))
    return val, row, col, (A.shape[0], A.shape[1])


# ====================================================================== #
# SuiteSparse Matrix Collection (curated)
# ====================================================================== #
# Each entry: (group, name, mathematical_kind, detected_kind, notes).
#
# ``mathematical_kind`` is what the matrix really is.  ``detected_kind``
# is what the Gershgorin-based heuristic in
# :func:`torch_sla.backends.nvmath_backend.detect_matrix_type` is
# expected to return.  Gershgorin is a *sufficient* PD test, so SPD/HPD
# matrices that are not strictly diagonally dominant report as plain
# symmetric / hermitian -- that is correct safe behaviour.
_SUITESPARSE_CATALOG: Dict[str, Tuple[str, str, str, str, str]] = {
    "real_spd":            ("HB",  "bcsstk16",  "spd",       "symmetric",
                            "Harwell-Boeing structural stiffness; SPD but not diag dominant"),
    "complex_hpd":         ("Bai", "mhd1280b",  "hpd",       "hermitian",
                            "MHD Alfven spectra, HPD but not strictly diag dominant"),
    "complex_sym":         ("Bai", "qc324",     "symmetric", "symmetric",
                            "Quantum chemistry complex symmetric (small smoke)"),
    "complex_general_mhd": ("Bai", "mhd1280a",  "general",   "general",
                            "MHD system A matrix (NOT symmetric); pairs with mhd1280b"),
    "complex_general":     ("HB",  "young1c",   "general",   "general",
                            "Acoustic non-symmetric complex (David Young)"),
}


class _SuiteSparse(Mapping):
    """Lazy ``Mapping[str, Benchmark]`` over a curated SuiteSparse subset.

    Each value is a :class:`Benchmark` with three random ``(x_ref, b)``
    reference cases. Matrices are downloaded from the heroku mirror on
    first access; subsequent access hits the cache (see :func:`cache_dir`).

    >>> from torch_sla.datasets import SuiteSparse
    >>> bench = SuiteSparse["complex_hpd"]      # downloads on first call
    >>> bench.math_kind, bench.detected_kind
    ('hpd', 'hermitian')
    """

    def __init__(self):
        self._loaded: dict = {}

    def __getitem__(self, key: str) -> Benchmark:
        if key in self._loaded:
            return self._loaded[key]
        if key not in _SUITESPARSE_CATALOG:
            raise KeyError(
                f"Unknown SuiteSparse key {key!r}; available: "
                f"{sorted(_SUITESPARSE_CATALOG)}"
            )
        group, name, math_kind, det_kind, _notes = _SUITESPARSE_CATALOG[key]
        dt = np.complex128 if "complex" in key else np.float64
        path = _ensure_downloaded(group, name)
        val, row, col, shape = _load_as_triple(path, dtype=dt)
        bench = Benchmark(
            name=f"{group}/{name}",
            val=val, row=row, col=col, shape=shape,
            math_kind=math_kind, detected_kind=det_kind,
        )
        self._loaded[key] = bench
        return bench

    def __iter__(self) -> Iterator[str]:
        return iter(_SUITESPARSE_CATALOG)

    def __len__(self) -> int:
        return len(_SUITESPARSE_CATALOG)

    def notes(self, key: str) -> str:
        return _SUITESPARSE_CATALOG[key][4]

    def catalog(self) -> dict:
        return dict(_SUITESPARSE_CATALOG)


SuiteSparse = _SuiteSparse()


# ====================================================================== #
# DIMACS10 graph Laplacians (downloaded through SuiteSparse mirror)
# ====================================================================== #
# DIMACS10 stores adjacency matrices of undirected graphs. We convert to
# the regularised Laplacian ``L = D - A + eps*I`` so the matrix is SPD
# (strictly diag dominant for any eps > 0) and solvable -- the original
# adjacency is rank-deficient.
_DIMACS10_CATALOG: Dict[str, Tuple[str, float, str]] = {
    # key                 SuiteSparse-name     eps   notes
    "delaunay_small":    ("delaunay_n10",      1.0,
                          "1024-vertex planar Delaunay mesh; regular ~6-degree"),
    "delaunay_medium":   ("delaunay_n12",      1.0,
                          "4096-vertex planar Delaunay mesh; regular ~6-degree"),
    "scale_free":        ("preferentialAttachment", 1.0,
                          "100K-vertex Barabasi-Albert preferential-attachment graph; "
                          "power-law degree distribution"),
    "small_world":       ("smallworld",        1.0,
                          "100K-vertex Watts-Strogatz small-world graph; "
                          "high clustering + short paths"),
}


def _adjacency_to_regularised_laplacian(
    A: sp.spmatrix, eps: float
) -> sp.spmatrix:
    """Convert an adjacency matrix to ``L + eps*I`` where ``L = D - A``.

    Forces symmetric pattern (``A := (A + A^T)/2``) in case the source
    has minor asymmetries from the .mtx round-trip. Returns CSR.
    """
    A = A.tocsr()
    # Symmetrise: many DIMACS10 files store only one triangle.
    A = (A + A.T).tocsr() * 0.5
    # Zero the diagonal (in case it leaked in).
    A.setdiag(0)
    A.eliminate_zeros()
    n = A.shape[0]
    degree = np.asarray(A.sum(axis=1)).ravel()
    D = sp.diags(degree, 0, format="csr")
    L = D - A
    return L + eps * sp.eye(n, format="csr")


class _DIMACS10(Mapping):
    """Lazy ``Mapping[str, Benchmark]`` of DIMACS10 graph-Laplacian benchmarks.

    The DIMACS10 Implementation Challenge focuses on graph partitioning;
    its matrices are sparse adjacency matrices of undirected real-world
    networks (planar meshes, social, web, road, ...). We download the
    adjacency through the SuiteSparse mirror (where DIMACS10 lives under
    group ``DIMACS10``), then convert to the regularised Laplacian
    ``L + eps*I`` to obtain an SPD operator that exercises the
    irregular / scale-free sparsity patterns FE matrices don't have.

    >>> from torch_sla.datasets import DIMACS10
    >>> bench = DIMACS10["delaunay_small"]
    >>> bench.shape
    (1024, 1024)
    """

    def __init__(self):
        self._loaded: dict = {}

    def __getitem__(self, key: str) -> Benchmark:
        if key in self._loaded:
            return self._loaded[key]
        if key not in _DIMACS10_CATALOG:
            raise KeyError(
                f"Unknown DIMACS10 key {key!r}; available: "
                f"{sorted(_DIMACS10_CATALOG)}"
            )
        name, eps, _notes = _DIMACS10_CATALOG[key]
        path = _ensure_downloaded("DIMACS10", name)
        A_adj = mmread(path)
        L_reg = _adjacency_to_regularised_laplacian(A_adj, eps=eps).astype(np.float64)
        val, row, col, shape = _scipy_to_triple(L_reg)
        bench = Benchmark(
            name=f"DIMACS10/{name}+{eps}*I",
            val=val, row=row, col=col, shape=shape,
            math_kind="spd",
            # L+eps*I is strictly diag dominant by construction, so Gershgorin
            # correctly identifies it as SPD.
            detected_kind="spd",
        )
        self._loaded[key] = bench
        return bench

    def __iter__(self) -> Iterator[str]:
        return iter(_DIMACS10_CATALOG)

    def __len__(self) -> int:
        return len(_DIMACS10_CATALOG)

    def notes(self, key: str) -> str:
        return _DIMACS10_CATALOG[key][2]

    def catalog(self) -> dict:
        return dict(_DIMACS10_CATALOG)


DIMACS10 = _DIMACS10()


# ====================================================================== #
# Synthetic PDE stencils (programmatic, no download)
# ====================================================================== #
def _laplacian_1d(n: int) -> sp.csr_matrix:
    """1D Laplacian ``-d^2/dx^2`` on ``n`` interior points, Dirichlet BC."""
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    return sp.diags([off, main, off], [-1, 0, 1], format="csr")


def _backward_diff_1d(n: int) -> sp.csr_matrix:
    """First-order backward difference ``(u_i - u_{i-1})``."""
    return sp.diags([-np.ones(n - 1), np.ones(n)], [-1, 0], format="csr")


def poisson_2d(n: int = 64) -> Tuple[sp.csr_matrix, str]:
    """5-point Laplacian on an ``n x n`` grid with Dirichlet BC; SPD."""
    T = _laplacian_1d(n)
    I = sp.eye(n, format="csr")
    A = sp.kron(I, T) + sp.kron(T, I)
    return A.tocsr(), f"poisson_2d(n={n})"


def poisson_3d(n: int = 16) -> Tuple[sp.csr_matrix, str]:
    """7-point Laplacian on an ``n x n x n`` grid; SPD."""
    T = _laplacian_1d(n)
    I = sp.eye(n, format="csr")
    A = (sp.kron(sp.kron(I, I), T)
         + sp.kron(sp.kron(I, T), I)
         + sp.kron(sp.kron(T, I), I))
    return A.tocsr(), f"poisson_3d(n={n})"


def anisotropic_2d(n: int = 64, eps: float = 0.01) -> Tuple[sp.csr_matrix, str]:
    """Anisotropic diffusion ``-eps * d^2/dx^2 - d^2/dy^2``; SPD but
    ill-conditioned for ``eps << 1`` (cond ~ 1/eps)."""
    T = _laplacian_1d(n)
    I = sp.eye(n, format="csr")
    A = sp.kron(I, eps * T) + sp.kron(T, I)
    return A.tocsr(), f"anisotropic_2d(n={n}, eps={eps})"


def convdiff_2d(n: int = 64, peclet: float = 10.0) -> Tuple[sp.csr_matrix, str]:
    """Convection-diffusion ``-Laplace u + Pe * (d/dx + d/dy) u`` with
    upwind first-order convection; real **non-symmetric**."""
    T = _laplacian_1d(n)
    D = _backward_diff_1d(n)
    I = sp.eye(n, format="csr")
    diff = sp.kron(I, T) + sp.kron(T, I)
    conv = peclet * (sp.kron(I, D) + sp.kron(D, I))
    return (diff + conv).tocsr(), f"convdiff_2d(n={n}, peclet={peclet})"


def helmholtz_2d(n: int = 64, k: float = 5.0, sigma: float = 0.5
                 ) -> Tuple[sp.csr_matrix, str]:
    """Helmholtz with first-order absorption: ``-Laplace u - k^2 u + i*sigma*u``.

    The matrix is **complex symmetric** (``A = A^T`` but not Hermitian
    because the diagonal carries an imaginary ``+i*sigma`` term)."""
    T = _laplacian_1d(n)
    I = sp.eye(n, format="csr")
    L = sp.kron(I, T) + sp.kron(T, I)
    N = n * n
    A = L.astype(np.complex128) + (-k * k + 1j * sigma) * sp.eye(N, format="csr",
                                                                 dtype=np.complex128)
    return A.tocsr(), f"helmholtz_2d(n={n}, k={k}, sigma={sigma})"


# Each entry: (builder_callable, kwargs, math_kind, detected_kind, notes).
_SYNTHETIC_CATALOG: Dict[str, Tuple[Callable, Dict[str, Any], str, str, str]] = {
    "poisson_2d_16": (
        poisson_2d, {"n": 16}, "spd", "symmetric",
        "5-point 2D Laplacian, 256 unknowns; tiny smoke-test size"),
    "poisson_2d_64": (
        poisson_2d, {"n": 64}, "spd", "symmetric",
        "5-point 2D Laplacian, n^2 unknowns; classic SPD test problem"),
    "poisson_3d_16": (
        poisson_3d, {"n": 16}, "spd", "symmetric",
        "7-point 3D Laplacian, n^3 unknowns"),
    "anisotropic_2d_64_eps_001": (
        anisotropic_2d, {"n": 64, "eps": 0.01}, "spd", "symmetric",
        "Anisotropic diffusion; eps=0.01 gives cond ~ 100"),
    "convdiff_2d_64_peclet_10": (
        convdiff_2d, {"n": 64, "peclet": 10.0}, "general", "general",
        "Convection-diffusion with upwind; real non-symmetric"),
    "helmholtz_2d_64_k_5": (
        helmholtz_2d, {"n": 64, "k": 5.0, "sigma": 0.5}, "symmetric", "symmetric",
        "Helmholtz with absorption; complex symmetric (A = A^T, not Hermitian)"),
}


class _Synthetic(Mapping):
    """Lazy ``Mapping[str, Benchmark]`` of programmatic PDE stencils.

    No network required -- the matrices are built on demand via
    ``scipy.sparse`` Kronecker products. Useful for parameter sweeps
    (size, condition number, Peclet, wavenumber) that real-world
    catalogues cannot offer.

    >>> from torch_sla.datasets import Synthetic
    >>> bench = Synthetic["poisson_2d_64"]
    >>> bench.shape
    (4096, 4096)
    """

    def __init__(self):
        self._loaded: dict = {}

    def __getitem__(self, key: str) -> Benchmark:
        if key in self._loaded:
            return self._loaded[key]
        if key not in _SYNTHETIC_CATALOG:
            raise KeyError(
                f"Unknown Synthetic key {key!r}; available: "
                f"{sorted(_SYNTHETIC_CATALOG)}"
            )
        builder, kwargs, math_kind, det_kind, _notes = _SYNTHETIC_CATALOG[key]
        A_sp, label = builder(**kwargs)
        val, row, col, shape = _scipy_to_triple(A_sp)
        bench = Benchmark(
            name=label,
            val=val, row=row, col=col, shape=shape,
            math_kind=math_kind, detected_kind=det_kind,
        )
        self._loaded[key] = bench
        return bench

    def __iter__(self) -> Iterator[str]:
        return iter(_SYNTHETIC_CATALOG)

    def __len__(self) -> int:
        return len(_SYNTHETIC_CATALOG)

    def notes(self, key: str) -> str:
        return _SYNTHETIC_CATALOG[key][4]

    def catalog(self) -> dict:
        return dict(_SYNTHETIC_CATALOG)


Synthetic = _Synthetic()


# ====================================================================== #
# Convenience: every benchmark in one place
# ====================================================================== #
def all_benchmarks() -> Dict[str, Benchmark]:
    """Return a flat ``{qualified_name: Benchmark}`` over every catalogue
    (SuiteSparse, DIMACS10, Synthetic). Network failures degrade
    gracefully -- the offending entry is omitted.

    Keys are prefixed by source: ``"suitesparse:real_spd"``,
    ``"dimacs10:delaunay_small"``, ``"synthetic:poisson_2d_64"``.
    """
    out: Dict[str, Benchmark] = {}
    for prefix, registry in (("suitesparse", SuiteSparse),
                             ("dimacs10", DIMACS10),
                             ("synthetic", Synthetic)):
        for key in registry:
            try:
                out[f"{prefix}:{key}"] = registry[key]
            except DatasetUnavailable:
                continue
    return out


__all__ = [
    "SuiteSparse", "DIMACS10", "Synthetic",
    "DatasetUnavailable", "cache_dir", "all_benchmarks",
    "poisson_2d", "poisson_3d", "anisotropic_2d",
    "convdiff_2d", "helmholtz_2d",
]
