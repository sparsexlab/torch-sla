"""Analytical / PDE benchmark problems exposed as :class:`SparseProblem`.

Each generator returns a :class:`SparseProblem` carrying the COO triple, the
shape, optional right-hand side / exact solution, and ``properties`` / ``meta``
dictionaries.  The matrices are built with the *exact* formulas used by the
correctness tests so that benchmarks and tests can pull a single source of
truth from here instead of building stencils inline.

The structured-grid stencils are assembled with a single **vectorised** builder
(:func:`_grid_coo`) -- meshgrid coordinates + boolean in-bounds masks, no
per-node Python loops -- so even large grids build instantly.

All matrices use ``torch.float64`` (or ``torch.complex128`` for the complex
problems).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence

import torch
from torch import Tensor

from .sparse_problem import SparseProblem

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
# Vectorised structured-grid stencil builder
# --------------------------------------------------------------------------- #
def _grid_coo(dims: Sequence[int], diag, edges, dtype=torch.float64):
    """Assemble a structured-grid finite-difference stencil as COO, vectorised.

    Parameters
    ----------
    dims : sequence of int
        Per-axis grid sizes; nodes are flattened **row-major** (last axis
        fastest), i.e. ``p = ((i0 * d1 + i1) * d2 + i2) ...`` -- matching
        ``idx(i, j) = i * m + j`` / ``idx(i, j, k) = (i*m + j)*m + k``.
    diag : number
        Value placed on every diagonal entry.
    edges : iterable of ``(axis, step, weight)``
        Each coupling adds an off-diagonal entry from every node to its
        neighbour ``step`` cells along ``axis`` (``step`` is +1/-1), with value
        ``weight``, but only where the neighbour stays in-bounds. Multiple edges
        to the same neighbour stay as separate entries (they sum on coalesce),
        matching the inline stencils.
    dtype : torch.dtype
        ``float64`` (default) or ``complex128``.

    Returns
    -------
    (val, row, col) : Tensors  (``col``/``row`` are long)
    """
    dims = tuple(int(d) for d in dims)
    ndim = len(dims)
    n = 1
    for d in dims:
        n *= d
    # row-major strides (last axis fastest)
    strides = [1] * ndim
    for a in range(ndim - 2, -1, -1):
        strides[a] = strides[a + 1] * dims[a + 1]

    # per-node coordinate along each axis (each length n)
    grids = torch.meshgrid(*[torch.arange(d) for d in dims], indexing="ij")
    coords = [g.reshape(-1) for g in grids]
    p = torch.arange(n)

    rows = [p]
    cols = [p]
    vals = [torch.full((n,), diag, dtype=dtype)]
    for axis, step, weight in edges:
        nb = coords[axis] + step
        mask = (nb >= 0) & (nb < dims[axis])
        src = p[mask]
        dst = src + step * strides[axis]
        rows.append(src)
        cols.append(dst)
        vals.append(torch.full((src.numel(),), weight, dtype=dtype))

    return (torch.cat(vals),
            torch.cat(rows).to(torch.long),
            torch.cat(cols).to(torch.long))


def _axis_edges(ndim: int, weight) -> List[tuple]:
    """``(axis, +-1, weight)`` couplings along every axis (isotropic stencil)."""
    return [(a, s, weight) for a in range(ndim) for s in (1, -1)]


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
    val, row, col = _grid_coo((n,), 2.0, _axis_edges(1, -1.0))
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
    val, row, col = _grid_coo((m, m), 4.0, _axis_edges(2, -1.0))
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
    val, row, col = _grid_coo((m, m, m), 6.0, _axis_edges(3, -1.0))
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
    val, row, col = _grid_coo((m, m), 4.0 * inv, _axis_edges(2, -inv))
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
    val, row, col = _grid_coo((m, m, m), 6.0 * inv, _axis_edges(3, -inv))
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
    val, row, col = _grid_coo((n,), 2.0 * inv, _axis_edges(1, -inv))
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
    diag = 2.0 * inv - (k * k) - 1j * alpha
    val, row, col = _grid_coo((n,), diag, _axis_edges(1, -inv + 0j),
                              dtype=torch.complex128)
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
    cx = eps * inv   # x-direction (u_xx) coupling weight (axis 0)
    cy = 1.0 * inv   # y-direction (u_yy) coupling weight (axis 1)
    edges = [(0, 1, -cx), (0, -1, -cx), (1, 1, -cy), (1, -1, -cy)]
    val, row, col = _grid_coo((m, m), 2.0 * (cx + cy), edges)
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
    b = peclet / h   # convective coefficient
    # diffusion (4-pt, weight -inv) + upwind convection toward (i-1,j),(i,j-1)
    # (velocity (+1,+1) -> backward difference) with weight -b. Same-neighbour
    # diffusion + convection entries are kept separate (they sum on coalesce).
    edges = _axis_edges(2, -inv) + [(0, -1, -b), (1, -1, -b)]
    val, row, col = _grid_coo((m, m), 4.0 * inv + 2.0 * b, edges)
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
