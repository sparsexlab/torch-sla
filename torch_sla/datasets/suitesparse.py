"""Public-dataset loader for the SuiteSparse Matrix Collection.

Downloads + caches ``.mtx`` matrices from https://sparse.tamu.edu/MM (via
``ssgetpy`` when available, else direct URL), parses the Matrix Market header
for symmetry / field, and returns a :class:`SparseProblem`.

Cache lives under ``~/.cache/suitesparse`` (matching
``benchmarks/benchmark_suitesparse.py``).
"""

from __future__ import annotations

import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .problems import SparseProblem

__all__ = [
    "load_suitesparse",
    "is_suitesparse_available",
    "RECOMMENDED",
    "download_suitesparse",
]

MATRIX_MARKET_BASE = "https://sparse.tamu.edu/MM"


class SuiteSparseUnavailable(RuntimeError):
    """Raised when a matrix cannot be obtained (offline + not cached)."""


# Curated small/medium matrices spanning the property spectrum.  Names + groups
# are accurate to sparse.tamu.edu.
RECOMMENDED: Dict[str, Dict] = {
    # ---- SPD ----
    "bcsstk16": {"group": "HB", "spd": True, "note": "structural stiffness, 4884, SPD"},
    "nos3": {"group": "HB", "spd": True, "note": "biharmonic FE, 960, SPD"},
    "494_bus": {"group": "HB", "spd": True, "note": "power network, 494, SPD"},
    "mesh2e1": {"group": "Pothen", "spd": True, "note": "structural mesh, 306, SPD"},
    # ---- symmetric indefinite ----
    "bcsstk14": {"group": "HB", "symmetric": True, "note": "roof structure, 1806, symmetric"},
    "1138_bus": {"group": "HB", "symmetric": True, "note": "power network, 1138, symmetric"},
    # ---- nonsymmetric ----
    "cage5": {"group": "vanHeukelum", "symmetric": False, "note": "DNA electrophoresis, 37, nonsymmetric"},
    "cage9": {"group": "vanHeukelum", "symmetric": False, "note": "DNA electrophoresis, 3534, nonsymmetric"},
    "sherman1": {"group": "HB", "symmetric": False, "note": "oil reservoir, 1000, nonsymmetric"},
    # ---- complex ----
    "qc324": {"group": "Bai", "complex": True, "note": "H2+ in magnetic field, 324, complex symmetric"},
    "mhd1280b": {"group": "Bai", "complex": True, "note": "Alfven spectra MHD, 1280, complex"},
    # ---- large -> skip by default ----
    "G3_circuit": {"group": "AMD", "spd": True, "large": True,
                   "note": "circuit sim, 1.5M, SPD -- large, skip"},
}


def cache_dir() -> Path:
    """Get (creating if needed) the SuiteSparse download cache directory."""
    d = Path.home() / ".cache" / "suitesparse"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _resolve_group(name: str, group: Optional[str]) -> Optional[str]:
    if group is not None:
        return group
    if name in RECOMMENDED:
        return RECOMMENDED[name].get("group")
    return None


def _download_via_ssgetpy(name: str, group: Optional[str]) -> Optional[Path]:
    """Try ssgetpy; return the cached .mtx path or None if unavailable."""
    try:
        import ssgetpy
    except Exception:
        return None
    try:
        kwargs = {"name": name}
        if group:
            kwargs["group"] = group
        results = ssgetpy.search(**kwargs, limit=50)
        match = None
        for m in results:
            if m.name == name and (group is None or m.group == group):
                match = m
                break
        if match is None and len(results) == 1:
            match = results[0]
        if match is None:
            return None
        # fetch returns (mm_path, rb_path) style depending on version
        paths = match.download(format="MM", destpath=str(cache_dir()), extract=True)
        # locate the .mtx
        mtx = _find_mtx(cache_dir(), name)
        return mtx
    except Exception:
        return None


def _find_mtx(root: Path, name: str) -> Optional[Path]:
    direct = root / f"{name}.mtx"
    if direct.exists():
        return direct
    # ssgetpy extracts into <root>/<name>/<name>.mtx
    candidates = list(root.glob(f"**/{name}.mtx"))
    if candidates:
        return candidates[0]
    candidates = list(root.glob(f"{name}/*.mtx"))
    if candidates:
        return candidates[0]
    return None


def _download_via_url(name: str, group: str) -> Path:
    """Direct download from sparse.tamu.edu MM mirror; returns .mtx path."""
    d = cache_dir()
    mtx_path = d / f"{name}.mtx"
    if mtx_path.exists():
        return mtx_path
    url = f"{MATRIX_MARKET_BASE}/{group}/{name}.tar.gz"
    tar_path = d / f"{name}.tar.gz"
    try:
        urllib.request.urlretrieve(url, tar_path)
    except (urllib.error.URLError, OSError) as e:
        raise SuiteSparseUnavailable(
            f"Failed to download {group}/{name} from {url}: {e}"
        ) from e
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(d)
    extracted = d / name
    mtx_files = list(extracted.glob("*.mtx")) if extracted.exists() else []
    if not mtx_files:
        mtx_files = list(d.glob(f"{name}/*.mtx"))
    if not mtx_files:
        raise SuiteSparseUnavailable(f"No .mtx found after extracting {name}")
    shutil.move(str(mtx_files[0]), str(mtx_path))
    if extracted.exists():
        shutil.rmtree(extracted, ignore_errors=True)
    tar_path.unlink(missing_ok=True)
    return mtx_path


def download_suitesparse(name: str, group: Optional[str] = None) -> Path:
    """Download + cache the named matrix, returning the local ``.mtx`` path.

    Tries ssgetpy first, then a direct URL download.  Raises
    :class:`SuiteSparseUnavailable` (a clear error, not a crash) when offline
    and not cached.
    """
    group = _resolve_group(name, group)

    cached = _find_mtx(cache_dir(), name)
    if cached is not None:
        return cached

    # ssgetpy path (handles group lookup itself)
    mtx = _download_via_ssgetpy(name, group)
    if mtx is not None:
        return mtx

    if group is None:
        raise SuiteSparseUnavailable(
            f"Matrix {name!r} not cached and group unknown; pass group=... "
            f"or add it to RECOMMENDED (ssgetpy unavailable or offline)."
        )
    return _download_via_url(name, group)


def _parse_mm_header(path: Path) -> Tuple[str, str]:
    """Return ``(field, symmetry)`` from the Matrix Market banner."""
    with open(path, "r") as f:
        header = f.readline().strip()
    parts = header.split()
    field = parts[3].lower() if len(parts) > 3 else "real"
    symmetry = parts[4].lower() if len(parts) > 4 else "general"
    return field, symmetry


def load_suitesparse(name: str, group: Optional[str] = None) -> SparseProblem:
    """Load a SuiteSparse matrix as a :class:`SparseProblem`.

    Downloads + caches the ``.mtx``, parses the MM header for symmetry /
    complexity, and returns a COO :class:`SparseProblem`.  Raises
    :class:`SuiteSparseUnavailable` when offline and not cached.
    """
    from ..io import load_mtx

    mtx_path = download_suitesparse(name, group)
    field, symmetry = _parse_mm_header(mtx_path)
    is_complex = field in ("complex",)
    dtype = torch.complex128 if is_complex else torch.float64

    A = load_mtx(mtx_path, dtype=dtype, device="cpu")
    # torch-sla SparseTensor exposes COO via tensor attributes.
    val = A.values
    row = A.row_indices
    col = A.col_indices
    shape = tuple(A.shape)

    symmetric = symmetry in ("symmetric", "hermitian", "skew-symmetric")
    properties = {
        "complex": bool(is_complex),
        "symmetric": bool(symmetric),
        "spd": bool(RECOMMENDED.get(name, {}).get("spd", False)),
        "mm_symmetry": symmetry,
        "mm_field": field,
    }
    meta = {
        "dof": shape[0],
        "source": "suitesparse",
        "name": name,
        "group": _resolve_group(name, group),
        "path": str(mtx_path),
    }
    return SparseProblem(
        name=f"suitesparse:{name}",
        val=val.to(dtype), row=row.long(), col=col.long(), shape=shape,
        properties=properties, meta=meta,
    )


def is_suitesparse_available(name: Optional[str] = None,
                             group: Optional[str] = None) -> bool:
    """Return ``True`` if SuiteSparse matrices can be obtained.

    If ``name`` is given, checks that specific matrix is cached or downloadable;
    otherwise checks for ssgetpy availability or network reachability of the
    MM mirror.
    """
    if name is not None:
        if _find_mtx(cache_dir(), name) is not None:
            return True
    # cached anything?
    if name is None and any(cache_dir().glob("*.mtx")):
        return True
    # ssgetpy present?
    try:
        import ssgetpy  # noqa: F401
        has_ssgetpy = True
    except Exception:
        has_ssgetpy = False
    # network reachable?
    try:
        urllib.request.urlopen(MATRIX_MARKET_BASE, timeout=5)
        network = True
    except Exception:
        network = False
    return bool(has_ssgetpy or network)
