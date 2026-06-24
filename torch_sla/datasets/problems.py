"""Analytical / PDE benchmark problems exposed as :class:`SparseProblem`.

Each generator returns a :class:`SparseProblem` carrying the COO triple, the
shape, optional right-hand side / exact solution, and ``properties`` / ``meta``
dictionaries.  The matrices are built with the *exact* formulas used by the
correctness tests so that benchmarks and tests can pull a single source of
truth from here instead of building stencils inline.

All matrices use ``torch.float64`` (or ``torch.complex128`` for the complex
problems).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "SparseProblem",
    "laplacian_1d",
    "laplacian_2d",
    "laplacian_3d",
    "poisson_2d",
    "poisson_3d",
    "bratu_1d",
    "helmholtz_1d",
    "anisotropic_diffusion_2d",
    "advection_diffusion_2d",
    "laplacian_1d_eigenvalues",
    "laplacian_2d_eigenvalues",
    "PROBLEMS",
    "list_problems",
    "get",
    "default_sizes",
]


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #
@dataclass
class SparseProblem:
    """A single sparse benchmark / test problem in COO form.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. ``"poisson_2d(m=31)"``).
    val, row, col : Tensor
        COO values (float64 / complex128), row indices (long), col indices
        (long).
    shape : tuple
        ``(n, n)``.
    rhs : Tensor or None
        A right-hand side ``b`` if the problem defines one.
    exact : Tensor or None
        Exact / manufactured solution ``u*`` if known.
    properties : dict
        Boolean-ish flags such as ``{'spd': True, 'symmetric': True,
        'complex': False}``.
    meta : dict
        Free-form metadata, e.g. ``{'dof': n, 'source': 'pde',
        'pde': 'poisson2d'}``.
    """

    name: str
    val: Tensor
    row: Tensor
    col: Tensor
    shape: tuple
    rhs: Optional[Tensor] = None
    exact: Optional[Tensor] = None
    properties: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def coo(self):
        """Return ``(val, row, col, shape)``."""
        return self.val, self.row, self.col, self.shape

    def nnz(self):
        """Number of stored entries."""
        return self.val.numel()

    def to_dense(self) -> Tensor:
        """Materialise a dense matrix (mostly for the RHS / reference work)."""
        n, m = self.shape
        A = torch.zeros(n, m, dtype=self.val.dtype)
        A.index_put_((self.row, self.col), self.val, accumulate=True)
        return A

    def matvec(self, x: Tensor) -> Tensor:
        """Compute ``A @ x`` from the COO triple without densifying."""
        n, _ = self.shape
        out = torch.zeros(n, dtype=torch.result_type(self.val, x))
        out.index_add_(0, self.row, self.val * x[self.col])
        return out


# --------------------------------------------------------------------------- #
# Helpers for building stencils
# --------------------------------------------------------------------------- #
def _coo(rows, cols, vals, dtype=torch.float64):
    return (
        torch.tensor(vals, dtype=dtype),
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(cols, dtype=torch.long),
    )


def _tridiag_1d(n: int, scale: float = 1.0):
    """``scale * tridiag(-1, 2, -1)`` (n interior nodes)."""
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(2.0 * scale)
        if i + 1 < n:
            rows.append(i); cols.append(i + 1); vals.append(-1.0 * scale)
            rows.append(i + 1); cols.append(i); vals.append(-1.0 * scale)
    return rows, cols, vals


# --------------------------------------------------------------------------- #
# Analytical-spectrum helpers
# --------------------------------------------------------------------------- #
def laplacian_1d_eigenvalues(n: int) -> Tensor:
    """Closed-form eigenvalues of ``tridiag(-1, 2, -1)`` (n interior nodes)::

        lambda_k = 2 - 2 cos(k pi / (n+1)),   k = 1 .. n
    """
    ks = torch.arange(1, n + 1, dtype=torch.float64)
    return 2 - 2 * torch.cos(ks * math.pi / (n + 1))


def laplacian_2d_eigenvalues(m: int) -> Tensor:
    """All ``m*m`` eigenvalues of the 5-point ``m x m`` graph Laplacian.

    The 2-D spectrum is the set of pairwise sums of the 1-D spectrum.
    """
    lam1d = laplacian_1d_eigenvalues(m)
    return (lam1d[:, None] + lam1d[None, :]).flatten()


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #
def laplacian_1d(n: int = 200) -> SparseProblem:
    """1-D Dirichlet Laplacian ``tridiag(-1, 2, -1)`` with ``n`` interior nodes.

    SPD + symmetric.  Eigenvalues are ``2 - 2 cos(k pi / (n+1))``.
    """
    rows, cols, vals = _tridiag_1d(n, scale=1.0)
    val, row, col = _coo(rows, cols, vals)
    return SparseProblem(
        name=f"laplacian_1d(n={n})",
        val=val, row=row, col=col, shape=(n, n),
        properties={"spd": True, "symmetric": True, "complex": False},
        meta={
            "dof": n, "source": "pde", "pde": "laplacian1d",
            "eigenvalues": "2-2cos(k*pi/(n+1))", "n": n,
        },
    )


def laplacian_2d(m: int = 30) -> SparseProblem:
    """5-point graph Laplacian on an ``m x m`` grid (diag 4, off -1).

    SPD + symmetric.  2-D spectrum = pairwise sums of the 1-D spectrum.
    """
    n = m * m
    idx = lambda i, j: i * m + j
    rows, cols, vals = [], [], []
    for i in range(m):
        for j in range(m):
            p = idx(i, j)
            rows.append(p); cols.append(p); vals.append(4.0)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    rows.append(p); cols.append(idx(ii, jj)); vals.append(-1.0)
    val, row, col = _coo(rows, cols, vals)
    return SparseProblem(
        name=f"laplacian_2d(m={m})",
        val=val, row=row, col=col, shape=(n, n),
        properties={"spd": True, "symmetric": True, "complex": False},
        meta={"dof": n, "source": "pde", "pde": "laplacian2d", "m": m},
    )


def laplacian_3d(m: int = 10) -> SparseProblem:
    """7-point graph Laplacian on an ``m x m x m`` grid (diag 6, off -1).

    SPD + symmetric.
    """
    n = m * m * m
    idx = lambda i, j, k: (i * m + j) * m + k
    rows, cols, vals = [], [], []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                p = idx(i, j, k)
                rows.append(p); cols.append(p); vals.append(6.0)
                for di, dj, dk in (
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1),
                ):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < m and 0 <= jj < m and 0 <= kk < m:
                        rows.append(p); cols.append(idx(ii, jj, kk)); vals.append(-1.0)
    val, row, col = _coo(rows, cols, vals)
    return SparseProblem(
        name=f"laplacian_3d(m={m})",
        val=val, row=row, col=col, shape=(n, n),
        properties={"spd": True, "symmetric": True, "complex": False},
        meta={"dof": n, "source": "pde", "pde": "laplacian3d", "m": m},
    )


def poisson_2d(m: int = 31) -> SparseProblem:
    """2-D Poisson with manufactured solution.

    ``A = (1/h^2) * 5-point(4, -1)``, ``h = 1/(m+1)``, interior grid
    ``x_i = i*h``.  Exact ``u = sin(pi x) sin(pi y)`` and
    ``f = 2 pi^2 sin(pi x) sin(pi y)``.  SPD.
    """
    n = m * m
    h = 1.0 / (m + 1)
    inv = 1.0 / (h * h)
    idx = lambda i, j: i * m + j
    rows, cols, vals = [], [], []
    for i in range(m):
        for j in range(m):
            p = idx(i, j)
            rows.append(p); cols.append(p); vals.append(4.0 * inv)
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    rows.append(p); cols.append(idx(ii, jj)); vals.append(-inv)
    val, row, col = _coo(rows, cols, vals)
    xs = torch.arange(1, m + 1, dtype=torch.float64) * h
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    u_exact = (torch.sin(math.pi * X) * torch.sin(math.pi * Y)).flatten()
    f = (2 * math.pi ** 2 * torch.sin(math.pi * X) * torch.sin(math.pi * Y)).flatten()
    return SparseProblem(
        name=f"poisson_2d(m={m})",
        val=val, row=row, col=col, shape=(n, n),
        rhs=f, exact=u_exact,
        properties={"spd": True, "symmetric": True, "complex": False},
        meta={"dof": n, "source": "pde", "pde": "poisson2d", "m": m, "h": h,
              "manufactured": "sin(pi x) sin(pi y)"},
    )


def poisson_3d(m: int = 10) -> SparseProblem:
    """3-D Poisson with manufactured solution.

    ``A = (1/h^2) * 7-point(6, -1)``, ``h = 1/(m+1)``.  Exact
    ``u = sin(pi x) sin(pi y) sin(pi z)`` and ``f = 3 pi^2 u``.  SPD.
    """
    n = m * m * m
    h = 1.0 / (m + 1)
    inv = 1.0 / (h * h)
    idx = lambda i, j, k: (i * m + j) * m + k
    rows, cols, vals = [], [], []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                p = idx(i, j, k)
                rows.append(p); cols.append(p); vals.append(6.0 * inv)
                for di, dj, dk in (
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1),
                ):
                    ii, jj, kk = i + di, j + dj, k + dk
                    if 0 <= ii < m and 0 <= jj < m and 0 <= kk < m:
                        rows.append(p); cols.append(idx(ii, jj, kk)); vals.append(-inv)
    val, row, col = _coo(rows, cols, vals)
    xs = torch.arange(1, m + 1, dtype=torch.float64) * h
    X, Y, Z = torch.meshgrid(xs, xs, xs, indexing="ij")
    u_exact = (torch.sin(math.pi * X) * torch.sin(math.pi * Y)
               * torch.sin(math.pi * Z)).flatten()
    f = (3 * math.pi ** 2 * torch.sin(math.pi * X) * torch.sin(math.pi * Y)
         * torch.sin(math.pi * Z)).flatten()
    return SparseProblem(
        name=f"poisson_3d(m={m})",
        val=val, row=row, col=col, shape=(n, n),
        rhs=f, exact=u_exact,
        properties={"spd": True, "symmetric": True, "complex": False},
        meta={"dof": n, "source": "pde", "pde": "poisson3d", "m": m, "h": h,
              "manufactured": "sin(pi x) sin(pi y) sin(pi z)"},
    )


def _bratu_c(lam: float) -> float:
    """Lower-branch root of ``c = sqrt(2 lam) cosh(c/4)`` by fixed-point."""
    k = math.sqrt(2.0 * lam)
    c = 0.1
    for _ in range(200):
        c = k * math.cosh(c / 4.0)
    return c


def bratu_1d(n: int = 100, lam: float = 1.0) -> SparseProblem:
    """1-D Bratu (solid-fuel ignition) benchmark.

    Stores the *linear* operator ``A = (1/h^2) tridiag(-1, 2, -1)`` of the
    Bratu residual ``F(u) = A u - lam exp(u)``, ``h = 1/(n+1)``.  ``.exact``
    holds the closed-form analytical Bratu solution::

        u(x_i) = -2 ln( cosh((c/2)(x_i - 1/2)) / cosh(c/4) )

    where ``c`` solves ``c = sqrt(2 lam) cosh(c/4)``.  ``rhs`` is ``None``
    (the system is nonlinear).
    """
    h = 1.0 / (n + 1)
    inv = 1.0 / (h * h)
    rows, cols, vals = _tridiag_1d(n, scale=inv)
    val, row, col = _coo(rows, cols, vals)
    x = torch.linspace(h, 1 - h, n, dtype=torch.float64)
    c = _bratu_c(lam)
    u_exact = -2.0 * torch.log(
        torch.cosh((c / 2.0) * (x - 0.5)) / math.cosh(c / 4.0)
    )
    return SparseProblem(
        name=f"bratu_1d(n={n}, lam={lam})",
        val=val, row=row, col=col, shape=(n, n),
        rhs=None, exact=u_exact,
        properties={"spd": True, "symmetric": True, "complex": False},
        meta={"dof": n, "source": "pde", "pde": "bratu1d", "n": n, "h": h,
              "lam": lam, "nonlinear": True, "residual": "A@u - lam*exp(u)",
              "x": x},
    )


def helmholtz_1d(n: int = 300, k: float = 8.0, alpha: float = 2.0) -> SparseProblem:
    """1-D Helmholtz / impedance problem (complex-symmetric).

    ``A = (1/h^2) tridiag(-1, 2, -1) - k^2 I - 1j alpha I``, ``h = 1/(n+1)``.
    ``A = A^T != A^H`` (complex-symmetric, not Hermitian).  ``rhs`` is None.
    """
    h = 1.0 / (n + 1)
    inv = 1.0 / (h * h)
    shift = -(k * k) - 1j * alpha
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i); cols.append(i); vals.append(2.0 * inv + shift)
        if i + 1 < n:
            rows.append(i); cols.append(i + 1); vals.append(-inv + 0j)
            rows.append(i + 1); cols.append(i); vals.append(-inv + 0j)
    val, row, col = _coo(rows, cols, vals, dtype=torch.complex128)
    return SparseProblem(
        name=f"helmholtz_1d(n={n}, k={k}, alpha={alpha})",
        val=val, row=row, col=col, shape=(n, n),
        rhs=None, exact=None,
        properties={"complex": True, "symmetric": True, "spd": False},
        meta={"dof": n, "source": "pde", "pde": "helmholtz1d", "n": n, "h": h,
              "k": k, "alpha": alpha},
    )


def anisotropic_diffusion_2d(m: int = 30, eps: float = 1e-2) -> SparseProblem:
    """2-D anisotropic diffusion ``-eps u_xx - u_yy`` (5-point) on ``m x m``.

    SPD but ill-conditioned for small ``eps`` -- a harder SPD test problem.
    ``h = 1/(m+1)``.
    """
    n = m * m
    h = 1.0 / (m + 1)
    inv = 1.0 / (h * h)
    cx = eps * inv   # weight for x-direction (u_xx) couplings
    cy = 1.0 * inv   # weight for y-direction (u_yy) couplings
    idx = lambda i, j: i * m + j
    rows, cols, vals = [], [], []
    for i in range(m):
        for j in range(m):
            p = idx(i, j)
            rows.append(p); cols.append(p); vals.append(2.0 * (cx + cy))
            # x-neighbours (vary i)
            for di in (1, -1):
                ii = i + di
                if 0 <= ii < m:
                    rows.append(p); cols.append(idx(ii, j)); vals.append(-cx)
            # y-neighbours (vary j)
            for dj in (1, -1):
                jj = j + dj
                if 0 <= jj < m:
                    rows.append(p); cols.append(idx(i, jj)); vals.append(-cy)
    val, row, col = _coo(rows, cols, vals)
    return SparseProblem(
        name=f"anisotropic_diffusion_2d(m={m}, eps={eps})",
        val=val, row=row, col=col, shape=(n, n),
        properties={"spd": True, "symmetric": True, "complex": False},
        meta={"dof": n, "source": "pde", "pde": "anisotropic2d", "m": m,
              "h": h, "eps": eps},
    )


def advection_diffusion_2d(m: int = 30, peclet: float = 20.0) -> SparseProblem:
    """2-D convection-diffusion (upwind convection) -- NON-symmetric.

    ``-Delta u + Pe (u_x + u_y) = f`` discretised with the 5-point Laplacian
    plus first-order upwind differencing of the convective term, ``h =
    1/(m+1)``.  Upwinding makes ``A`` non-symmetric -- a good non-symmetric
    test problem.
    """
    n = m * m
    h = 1.0 / (m + 1)
    inv = 1.0 / (h * h)
    # convective coefficient (positive velocity -> upwind takes the "left" node)
    b = peclet / h
    idx = lambda i, j: i * m + j
    rows, cols, vals = [], [], []
    for i in range(m):
        for j in range(m):
            p = idx(i, j)
            diag = 4.0 * inv + 2.0 * b
            # diffusion neighbours
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    rows.append(p); cols.append(idx(ii, jj)); vals.append(-inv)
            # upwind convection: velocity (+1,+1) -> backward difference,
            # couples to (i-1, j) and (i, j-1) with -b
            for di, dj in ((-1, 0), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < m and 0 <= jj < m:
                    rows.append(p); cols.append(idx(ii, jj)); vals.append(-b)
            rows.append(p); cols.append(p); vals.append(diag)
    val, row, col = _coo(rows, cols, vals)
    return SparseProblem(
        name=f"advection_diffusion_2d(m={m}, peclet={peclet})",
        val=val, row=row, col=col, shape=(n, n),
        properties={"spd": False, "symmetric": False, "complex": False},
        meta={"dof": n, "source": "pde", "pde": "advdiff2d", "m": m, "h": h,
              "peclet": peclet},
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
PROBLEMS: Dict[str, Callable[..., SparseProblem]] = {
    "laplacian_1d": laplacian_1d,
    "laplacian_2d": laplacian_2d,
    "laplacian_3d": laplacian_3d,
    "poisson_2d": poisson_2d,
    "poisson_3d": poisson_3d,
    "bratu_1d": bratu_1d,
    "helmholtz_1d": helmholtz_1d,
    "anisotropic_diffusion_2d": anisotropic_diffusion_2d,
    "advection_diffusion_2d": advection_diffusion_2d,
}


def list_problems() -> List[str]:
    """Names of all registered analytical / PDE problems."""
    return list(PROBLEMS.keys())


def get(name: str, **kwargs) -> SparseProblem:
    """Construct a registered problem by name, forwarding ``**kwargs``."""
    if name not in PROBLEMS:
        raise KeyError(
            f"Unknown problem {name!r}. Available: {list_problems()}"
        )
    return PROBLEMS[name](**kwargs)


# DOF sweep (small -> large) per problem, suitable for benchmark scaling.
_DEFAULT_SIZES: Dict[str, List[dict]] = {
    "laplacian_1d": [{"n": n} for n in (100, 500, 2000, 10000)],
    "laplacian_2d": [{"m": m} for m in (16, 32, 64, 128)],
    "laplacian_3d": [{"m": m} for m in (8, 16, 24, 32)],
    "poisson_2d": [{"m": m} for m in (15, 31, 63, 127)],
    "poisson_3d": [{"m": m} for m in (8, 16, 24, 32)],
    "bratu_1d": [{"n": n} for n in (50, 200, 800, 3200)],
    "helmholtz_1d": [{"n": n} for n in (100, 400, 1600, 6400)],
    "anisotropic_diffusion_2d": [{"m": m} for m in (16, 32, 64, 128)],
    "advection_diffusion_2d": [{"m": m} for m in (16, 32, 64, 128)],
}


def default_sizes(name: str) -> List[dict]:
    """Return a reasonable DOF sweep (list of kwarg dicts) for ``name``.

    Each element can be splatted into :func:`get`, e.g.::

        for kw in default_sizes("poisson_2d"):
            p = get("poisson_2d", **kw)
    """
    if name not in _DEFAULT_SIZES:
        raise KeyError(
            f"Unknown problem {name!r}. Available: {list_problems()}"
        )
    return [dict(kw) for kw in _DEFAULT_SIZES[name]]
