"""Curated sparse-matrix datasets exposed as :class:`Benchmark` mappings.

Currently provides :class:`SuiteSparse`, a lazy ``Mapping[str, Benchmark]``
that downloads matrices from the SuiteSparse Matrix Collection on first
access and caches them locally.

Cache directory is controlled by the ``TORCH_SLA_DATASET`` environment
variable (default ``~/.cache/torch_sla/datasets``). Override at any time:

    export TORCH_SLA_DATASET=/path/to/large/disk

The default catalogue covers every matrix kind the cuDSS auto-detect
must handle (general / symmetric / spd / hermitian / hpd).
"""

from __future__ import annotations

import io as _io
import os
import tarfile
import urllib.request
from collections.abc import Mapping
from typing import Iterator, Optional, Tuple

import numpy as np
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
    """Raised when a dataset cannot be downloaded (no internet, etc.).

    Library code raises this rather than calling ``pytest.skip`` so the
    benchmark API does not pull in pytest as a runtime dependency. Test
    code may catch this and skip individually.
    """


# ---------------------------------------------------------------------- #
# SuiteSparse catalogue
# ---------------------------------------------------------------------- #
# Each entry: (group, name, mathematical_kind, detected_kind, notes).
#
# ``mathematical_kind`` is what the matrix really is.  ``detected_kind``
# is what the Gershgorin-based heuristic in
# :func:`torch_sla.backends.nvmath_backend.detect_matrix_type` is
# expected to return.  Gershgorin is a *sufficient* PD test, so SPD/HPD
# matrices that are not strictly diagonally dominant report as plain
# symmetric / hermitian -- that is correct safe behaviour (cuDSS would
# fail at factorisation if we declared SPD wrongly).
_CATALOG = {
    # --- Real -------------------------------------------------------------- #
    "real_spd":            ("HB",  "bcsstk16",  "spd",       "symmetric",
                            "Harwell-Boeing structural stiffness; SPD but not diag dominant"),
    # --- Complex HPD ------------------------------------------------------- #
    "complex_hpd":         ("Bai", "mhd1280b",  "hpd",       "hermitian",
                            "MHD Alfven spectra, HPD but not strictly diag dominant"),
    # --- Complex symmetric (A = A^T, complex diagonal allowed) ------------ #
    "complex_sym":         ("Bai", "qc324",     "symmetric", "symmetric",
                            "Quantum chemistry complex symmetric (small smoke)"),
    # --- General complex (no symmetry: A != A^T and A != A^H) ------------- #
    "complex_general_mhd": ("Bai", "mhd1280a",  "general",   "general",
                            "MHD system A matrix (NOT symmetric); pairs with mhd1280b"),
    "complex_general":     ("HB",  "young1c",   "general",   "general",
                            "Acoustic non-symmetric complex (David Young)"),
}


_HEROKU = ("https://suitesparse-collection-website.herokuapp.com/MM/"
           "{group}/{name}.tar.gz")


def _ensure_downloaded(group: str, name: str) -> str:
    """Return path to ``<name>.mtx`` in cache, downloading + extracting if
    missing. Raises :class:`DatasetUnavailable` on network failure."""
    out = os.path.join(cache_dir(), f"{name}.mtx")
    if os.path.exists(out):
        return out
    url = _HEROKU.format(group=group, name=name)
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
    except Exception as exc:  # network / firewall / heroku down
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


# ---------------------------------------------------------------------- #
# SuiteSparse Mapping
# ---------------------------------------------------------------------- #
class _SuiteSparse(Mapping):
    """Lazy ``Mapping[str, Benchmark]`` over the curated catalogue.

    >>> from torch_sla.datasets import SuiteSparse
    >>> bench = SuiteSparse["complex_hpd"]       # downloads on first call
    >>> len(bench)                               # number of (x, b) cases
    3
    >>> bench[0]["b"].shape
    torch.Size([1280])

    The cache directory is taken from the ``TORCH_SLA_DATASET``
    environment variable (default ``~/.cache/torch_sla/datasets``).
    """

    def __init__(self):
        self._loaded: dict = {}

    # Mapping interface ------------------------------------------------- #
    def __getitem__(self, key: str) -> Benchmark:
        if key in self._loaded:
            return self._loaded[key]
        if key not in _CATALOG:
            raise KeyError(
                f"Unknown SuiteSparse key {key!r}; available: {sorted(_CATALOG)}"
            )
        group, name, math_kind, det_kind, _notes = _CATALOG[key]
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
        return iter(_CATALOG)

    def __len__(self) -> int:
        return len(_CATALOG)

    # Convenience ------------------------------------------------------- #
    def notes(self, key: str) -> str:
        return _CATALOG[key][4]

    def catalog(self) -> dict:
        """Return the catalogue dict (read-only view of the static metadata)."""
        return dict(_CATALOG)


SuiteSparse = _SuiteSparse()
"""Singleton instance. Treat as ``Mapping[str, Benchmark]``."""

__all__ = ["SuiteSparse", "DatasetUnavailable", "cache_dir"]
