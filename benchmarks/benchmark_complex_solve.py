"""
Benchmark complex sparse linear solve on SuiteSparse Matrix Collection
matrices.  Validates that the complex forward solve and the complex adjoint
(Wirtinger gradient) work correctly across three mathematically distinct
classes of complex matrices, and measures per-matrix wall time.

Open-source test matrices (Bai / HB groups, freely downloadable):

  Bai/mhd1280b   Hermitian SPD       (A = A^H, real diagonal)        MHD plasma
  Bai/mhd1280a   Complex symmetric   (A = A^T, complex diagonal)     MHD plasma
  Bai/qc324      Complex symmetric                                   quantum chemistry
  HB/young1c     General complex     (no symmetry)                   acoustics

The script downloads each matrix on first run (~50 MB total) and caches under
``data/suitesparse_complex/``.  Falls back gracefully if any matrix can't be
fetched (e.g. CI without internet).

Results are saved to: results/benchmark_complex_solve/
"""
import os
import io as pyio
import json
import time
import tarfile
import urllib.request
import warnings

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy.io import mmread

from torch_sla import SparseTensor

warnings.filterwarnings("ignore")

CACHE_DIR = "data/suitesparse_complex"
RESULT_DIR = "results/benchmark_complex_solve"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MATRICES = [
    # (group, name, "kind", "application")
    ("Bai", "mhd1280b", "Hermitian SPD",     "MHD plasma (Alfven spectra)"),
    ("Bai", "mhd1280a", "Complex symmetric", "MHD plasma (A matrix)"),
    ("Bai", "qc324",    "Complex symmetric", "Quantum chemistry"),
    ("HB",  "young1c",  "General complex",   "Acoustic (Young)"),
]


def download_matrix(group: str, name: str) -> str | None:
    """Download <group>/<name>.tar.gz from SuiteSparse, extract the .mtx file,
    and return the local path. Returns None on failure (e.g. no internet)."""
    path = os.path.join(CACHE_DIR, f"{name}.mtx")
    if os.path.exists(path):
        return path
    url = (
        f"https://suitesparse-collection-website.herokuapp.com/MM/{group}/{name}.tar.gz"
    )
    print(f"  downloading {url}", flush=True)
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
        with tarfile.open(fileobj=pyio.BytesIO(data), mode="r:gz") as tf:
            for m in tf.getmembers():
                if m.name.endswith(f"{name}.mtx"):
                    with open(path, "wb") as f:
                        f.write(tf.extractfile(m).read())
                    return path
    except Exception as e:  # network down, GFW, etc.
        print(f"  download failed: {e}", flush=True)
        return None
    return None


def benchmark_matrix(group: str, name: str, kind: str, app: str) -> dict | None:
    path = download_matrix(group, name)
    if path is None:
        print(f"  [skip] {name}: not available locally and could not download")
        return None

    A_sp = mmread(path).tocsr().astype(np.complex128)
    n = A_sp.shape[0]
    nnz = A_sp.nnz

    # Property checks (math sanity)
    diag_imag_max = float(np.abs(A_sp.diagonal().imag).max())
    is_hermitian = bool(np.allclose((A_sp - A_sp.conj().T).data, 0))
    is_complex_sym = bool(np.allclose((A_sp - A_sp.T).data, 0))

    # Build SparseTensor
    coo = A_sp.tocoo()
    val = torch.from_numpy(coo.data)
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    A = SparseTensor(val, row, col, (n, n))

    torch.manual_seed(0)
    b = torch.randn(n, dtype=torch.complex128)
    x_ref = torch.from_numpy(np.asarray(spla.spsolve(A_sp, b.numpy())))

    # --- forward solve ---
    for _ in range(2):
        _ = A.solve(b)  # warmup (cache scipy factorisation if any)
    t0 = time.perf_counter()
    x = A.solve(b)
    t_fwd = time.perf_counter() - t0
    fwd_err = (x - x_ref).abs().max().item()

    # --- backward (Wirtinger adjoint via scipy backend) ---
    v = val.clone().requires_grad_(True)
    A_g = SparseTensor(v, row, col, (n, n))
    x_g = A_g.solve(b)
    loss = (x_g.conj() * x_g).real.sum()  # ||x||^2
    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0
    bwd_finite = bool(torch.isfinite(v.grad.abs()).all().item())

    result = {
        "matrix":  f"{group}/{name}",
        "kind":    kind,
        "app":     app,
        "n":       n,
        "nnz":     nnz,
        "diag_imag_max":      diag_imag_max,
        "is_hermitian":       is_hermitian,
        "is_complex_sym":     is_complex_sym,
        "fwd_ms":             t_fwd * 1e3,
        "fwd_err_vs_scipy":   fwd_err,
        "bwd_ms":             t_bwd * 1e3,
        "bwd_finite":         bwd_finite,
    }
    print(
        f"  {name:10s} | n={n:>5d} nnz={nnz:>8d} | {kind:18s} "
        f"| fwd {t_fwd*1e3:7.2f}ms err {fwd_err:.0e} "
        f"| bwd {t_bwd*1e3:7.2f}ms finite={bwd_finite}",
        flush=True,
    )
    return result


def gradcheck_validation() -> bool:
    """Gold-standard numerical-FD check of the Wirtinger adjoint, on a small
    matrix (gradcheck on a 1k+ matrix would be O(nnz) full solves)."""
    n = 5
    rows = torch.tensor([0, 1, 2, 3, 4, 0, 1, 1, 2, 2, 3, 3, 4], dtype=torch.long)
    cols = torch.tensor([0, 1, 2, 3, 4, 1, 0, 2, 1, 3, 2, 4, 3], dtype=torch.long)
    vals_list = [
        complex(10, 0), complex(11, 0), complex(12, 0), complex(13, 0), complex(14, 0),
        complex(1, -1), complex(1,  1),
        complex(2, -2), complex(2,  2),
        complex(3, -3), complex(3,  3),
        complex(1, -1), complex(1,  1),
    ]  # Hermitian SPD tridiagonal-ish
    val = torch.tensor(vals_list, dtype=torch.complex128, requires_grad=True)
    torch.manual_seed(7)
    b = torch.randn(n, dtype=torch.complex128)

    def fn(v):
        return SparseTensor(v, rows, cols, (n, n)).solve(b)

    return torch.autograd.gradcheck(fn, (val,), eps=1e-6, atol=1e-4, rtol=1e-3,
                                    check_grad_dtypes=True)


def plot(results):
    if not results:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = [r["matrix"].split("/")[-1] for r in results]
    fwd   = [r["fwd_ms"] for r in results]
    bwd   = [r["bwd_ms"] for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.20, fwd, 0.40, label="forward solve")
    ax.bar(x + 0.20, bwd, 0.40, label="backward (adjoint)")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("wall time (ms)"); ax.set_yscale("log")
    ax.set_title("Complex sparse solve on SuiteSparse matrices "
                 "(scipy backend, complex128)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "timings.png")
    plt.savefig(out, dpi=150)
    print(f"\n  plot saved -> {out}")


def main():
    print("=== Benchmark: complex sparse solve on SuiteSparse matrices ===\n")
    print(f"  cache:  {CACHE_DIR}")
    print(f"  output: {RESULT_DIR}\n")

    results = []
    for spec in MATRICES:
        r = benchmark_matrix(*spec)
        if r is not None:
            results.append(r)

    with open(os.path.join(RESULT_DIR, "timings.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Wirtinger gradient validation (autograd.gradcheck) ===")
    try:
        ok = gradcheck_validation()
        print(f"  gradcheck: {'PASS' if ok else 'FAIL'}")
    except Exception as e:
        print(f"  gradcheck FAIL: {type(e).__name__}: {e}")

    plot(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
