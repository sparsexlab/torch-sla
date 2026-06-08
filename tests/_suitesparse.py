"""Shared SuiteSparse Matrix Collection fixtures.

Downloads and caches a curated set of small-to-medium open-source sparse
test matrices, used uniformly across the complex / cudss test suites.

Cache location: ``data/suitesparse_test/<name>.mtx`` relative to the
working directory; matrices are downloaded once on first access.

If a matrix cannot be downloaded (no internet, GFW, heroku down, ...) the
fixture calls ``pytest.skip``, so tests gracefully degrade rather than fail
on environments without internet.
"""
import io as pyio
import os
import tarfile
import urllib.request
from typing import Tuple

import numpy as np
import pytest
import torch
from scipy.io import mmread

CACHE_DIR = "data/suitesparse_test"

# Curated catalogue. Each entry: group, name, expected matrix-type label, notes.
# Picked for COVERAGE of every matrix class while staying small enough for
# the test suite (largest is mhd1280b at 1280x22778, runs in <1s).
CATALOG = {
    # Each entry: (group, name, mathematical_kind, detected_kind, notes).
    #
    # ``mathematical_kind`` is what the matrix really is.
    # ``detected_kind`` is what ``detect_matrix_type`` actually returns. It
    # can be more conservative than the math truth: Gershgorin only proves a
    # *sufficient* condition for positive definiteness, so SPD/HPD matrices
    # that aren't strictly diagonally dominant report as plain
    # symmetric/hermitian. That's correct behaviour -- a wrongly declared
    # SPD/HPD would fail at cuDSS factorisation; under-declaring is safe.
    #
    # --- Real ---------------------------------------------------------------
    "real_spd":        ("HB",       "bcsstk16",      "spd",       "symmetric",
                        "Harwell-Boeing structural stiffness; SPD but not diag dominant"),
    # --- Complex HPD (Hermitian + PD) --------------------------------------
    "complex_hpd":     ("Bai",      "mhd1280b",      "hpd",       "hermitian",
                        "MHD Alfven spectra, HPD but not strictly diag dominant"),
    # --- Complex symmetric (A = A^T, complex diagonal allowed) ------------
    "complex_sym":     ("Bai",      "qc324",         "symmetric", "symmetric",
                        "Quantum chemistry complex symmetric (small smoke)"),
    # --- General complex (no symmetry: A != A^T and A != A^H) -------------
    "complex_general_mhd": ("Bai",  "mhd1280a",      "general",   "general",
                        "MHD system A matrix (NOT symmetric); pairs with mhd1280b"),
    "complex_general":     ("HB",   "young1c",       "general",   "general",
                        "Acoustic non-symmetric complex (David Young)"),
}


def _download(group: str, name: str) -> str | None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.mtx")
    if os.path.exists(path):
        return path
    url = f"https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
        with tarfile.open(fileobj=pyio.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                if m.name.endswith(f"{name}.mtx"):
                    with open(path, "wb") as f:
                        f.write(tf.extractfile(m).read())
                    return path
    except Exception:
        return None
    return None


def load_matrix(key: str, *, dtype=None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Load a catalogued SuiteSparse matrix as (val, row, col, (n, n)).

    Skips the calling test (pytest.skip) if the matrix can't be downloaded
    (no internet, blocked, ...).  Use the keys defined in ``CATALOG``.
    """
    if key not in CATALOG:
        raise KeyError(f"Unknown SuiteSparse fixture key {key!r}; "
                       f"options: {sorted(CATALOG)}")
    group, name, _math_kind, _detected_kind, _notes = CATALOG[key]
    path = _download(group, name)
    if path is None:
        pytest.skip(f"SuiteSparse matrix {group}/{name} not available "
                    f"(no internet?); skipping {key}")
    A = mmread(path).tocsr()
    if dtype is not None:
        A = A.astype(dtype)
    n = A.shape[0]
    coo = A.tocoo()
    val = torch.from_numpy(coo.data)
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    return val, row, col, (n, n)


def mathematical_kind(key: str) -> str:
    """The matrix's true mathematical kind (spd / hpd / hermitian / symmetric / general)."""
    return CATALOG[key][2]


def detected_kind(key: str) -> str:
    """The kind ``detect_matrix_type`` is expected to return (may be more
    conservative than ``mathematical_kind`` because Gershgorin is sufficient
    but not necessary for positive definiteness)."""
    return CATALOG[key][3]


# Pytest fixtures (parametrise over the catalogue for sweep-style tests) ----

@pytest.fixture(params=list(CATALOG.keys()))
def suitesparse_any(request):
    """Iterates over every catalogued matrix.

    Yields a dict with keys: ``key``, ``math_kind``, ``detected_kind``,
    ``val``, ``row``, ``col``, ``shape``.
    """
    key = request.param
    dt = np.complex128 if "complex" in key else np.float64
    val, row, col, shape = load_matrix(key, dtype=dt)
    return {
        "key": key,
        "math_kind": mathematical_kind(key),
        "detected_kind": detected_kind(key),
        "val": val, "row": row, "col": col, "shape": shape,
    }


@pytest.fixture(params=[k for k in CATALOG if "complex" in k])
def suitesparse_complex(request):
    """Complex-only catalogue iteration. Same shape as ``suitesparse_any``."""
    key = request.param
    val, row, col, shape = load_matrix(key, dtype=np.complex128)
    return {
        "key": key,
        "math_kind": mathematical_kind(key),
        "detected_kind": detected_kind(key),
        "val": val, "row": row, "col": col, "shape": shape,
    }


@pytest.fixture
def suitesparse_complex_small():
    """A single small complex matrix (qc324, n=324) — good for fast tests
    like gradcheck. Returns (val, row, col, shape)."""
    return load_matrix("complex_sym", dtype=np.complex128)
