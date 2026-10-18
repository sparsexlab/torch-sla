"""
Bound-constrained nonlinear solve (variational inequality) with adjoint gradients.

For a residual ``F(u, θ)`` and bounds ``lower <= u <= upper`` this solves the
box-constrained variational inequality (mixed complementarity problem)

    find u with lower <= u <= upper such that, for every component i,
        F_i(u, θ) >= 0   where u_i = lower_i,
        F_i(u, θ) <= 0   where u_i = upper_i,
        F_i(u, θ)  = 0   where lower_i < u_i < upper_i.

When ``F`` is the gradient of an energy ``E`` these are the first-order (KKT)
conditions of minimizing ``E`` subject to the bounds: obstacle and contact
problems, damage and phase-field fracture with irreversibility, bounded
optimization. Components with ``lower_i == upper_i`` are fixed at that value,
which is how Dirichlet conditions are imposed.

Usage
-----
    from torch_sla import vi_solve

    # obstacle problem: minimize 1/2 u^T A u - f^T u  subject to  u >= psi
    u = vi_solve(lambda u, f: A @ u - f, torch.zeros(n), f,
                 lower=psi, jacobian_fn=lambda u, f: A)
    u.sum().backward()      # f.grad and, through the active set, psi.grad
"""

import math
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.autograd import Function

from .linear_solve import spsolve

__all__ = ["vi_solve", "VISolveAdjoint"]


# ============================================================================
# Helpers
# ============================================================================

def _full_bound(u: Tensor, bound, fill: float) -> Tensor:
    """Bound as a tensor shaped like ``u`` (None -> ±inf, scalars broadcast)."""
    if bound is None:
        return torch.full_like(u, fill)
    if not isinstance(bound, Tensor):
        return torch.full_like(u, float(bound))
    return bound.to(dtype=u.dtype, device=u.device).expand_as(u)


def _project(u: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    return torch.minimum(torch.maximum(u, lower), upper)


def _inactive(u: Tensor, F: Tensor, lower: Tensor, upper: Tensor,
              fixed: Tensor, zero_tol: float) -> Tensor:
    """Complement of the active set, as PETSc's ``SNESVIGetActiveSetIS``:
    a component is active when it sits on a bound and ``F`` pushes it out,
    ``u_i <= l_i + tol and F_i > 0`` or ``u_i >= r_i - tol and F_i < 0``.
    Fixed components (``l_i == r_i``) are always active."""
    off_lower = (u > lower + zero_tol) | (F <= 0)
    off_upper = (u < upper - zero_tol) | (F >= 0)
    return off_lower & off_upper & ~fixed


def _as_coo(J, n: int) -> Tuple[Tensor, Tensor, Tensor]:
    """``(val, row, col)`` of a Jacobian given as a SparseTensor (or subclass),
    a ``(val, row, col, shape)`` tuple, or a torch sparse COO tensor."""
    if isinstance(J, Tensor):  # before the duck-typed check: torch tensors have row_indices() too
        if J.layout != torch.sparse_coo:
            raise TypeError(f"vi_solve: a torch.Tensor Jacobian must be sparse COO, got {J.layout}")
        J = J.coalesce()
        idx = J.indices()
        return J.values(), idx[0], idx[1]
    if hasattr(J, "row_indices") and hasattr(J, "col_indices"):
        if getattr(J, "is_batched", False):
            raise ValueError("vi_solve: the Jacobian must not be batched")
        return J.values, J.row_indices, J.col_indices
    if isinstance(J, (tuple, list)) and len(J) == 4:
        val, row, col, shape = J
        if tuple(shape) != (n, n):
            raise ValueError(f"vi_solve: Jacobian shape {tuple(shape)} != ({n}, {n})")
        return val, row, col
    raise TypeError(
        "vi_solve: jacobian_fn must return a SparseTensor, a (val, row, col, shape) "
        f"tuple or a torch sparse COO tensor, got {type(J).__name__}")


def _reduced_coo(val: Tensor, row: Tensor, col: Tensor, keep: Tensor, n: int):
    """The Jacobian restricted to the inactive set, padded with the identity on
    the active and fixed components: ``[J_II 0; 0 I]``.

    The rows and columns of active components are zeroed and one diagonal
    entry per component is appended (value 1 if active, 0 otherwise), so the
    sparsity pattern does not depend on the active set and a direct solver can
    reuse its symbolic analysis. Duplicate entries are summed by the solvers.
    """
    v = torch.where(keep[row] & keep[col], val, torch.zeros_like(val))
    d = torch.arange(n, device=val.device)
    return (torch.cat([v, (~keep).to(val.dtype)]),
            torch.cat([row, d]), torch.cat([col, d]))


def _linear_solve(cfg: Dict[str, Any], val: Tensor, row: Tensor, col: Tensor,
                  n: int, b: Tensor) -> Tensor:
    solver = cfg["linear_solver"]
    if callable(solver):
        return solver(val, row, col, (n, n), b)
    return spsolve(val, row, col, (n, n), b, backend=solver, method=cfg["linear_method"],
                   **cfg["linear_kwargs"])


def _cg(matvec: Callable, b: Tensor, rtol: float, maxiter: int) -> Tensor:
    """Conjugate gradients for a symmetric positive definite operator."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = torch.dot(r, r)
    stop = rtol**2 * rs
    for _ in range(maxiter):
        if rs <= stop:
            break
        Ap = matvec(p)
        alpha = rs / torch.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def _gmres(matvec: Callable, b: Tensor, rtol: float, maxiter: int, restart: int = 50) -> Tensor:
    """Restarted GMRES (modified Gram-Schmidt Arnoldi, Givens rotations)."""
    x = torch.zeros_like(b)
    bnorm = float(torch.linalg.vector_norm(b))
    done = 0
    while done < maxiter:
        r = b - matvec(x) if done else b.clone()
        beta = float(torch.linalg.vector_norm(r))
        if beta <= rtol * bnorm:
            break
        m = min(restart, maxiter - done)
        V = [r / beta]
        H = torch.zeros(m + 1, m, dtype=torch.float64)
        g = torch.zeros(m + 1, dtype=torch.float64)
        g[0] = beta
        cs, sn = [], []
        k = m
        for j in range(m):
            w = matvec(V[j])
            done += 1
            for i in range(j + 1):
                H[i, j] = float(torch.dot(w, V[i]))
                w = w - H[i, j].item() * V[i]
            h_next = float(torch.linalg.vector_norm(w))
            for i in range(j):  # previous rotations
                h0, h1 = H[i, j].item(), H[i + 1, j].item()
                H[i, j], H[i + 1, j] = cs[i] * h0 + sn[i] * h1, -sn[i] * h0 + cs[i] * h1
            denom = math.hypot(H[j, j].item(), h_next)
            if denom == 0.0:  # breakdown: the Krylov space stopped growing
                k = j
                break
            cs.append(H[j, j].item() / denom)
            sn.append(h_next / denom)
            H[j, j] = denom
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j] * g[j]
            if h_next > 0:
                V.append(w / h_next)
            if abs(g[j + 1].item()) <= rtol * bnorm or h_next == 0:
                k = j + 1
                break
        y = torch.linalg.solve_triangular(H[:k, :k], g[:k, None], upper=True)[:, 0]
        for i in range(k):
            x = x + y[i].item() * V[i]
    return x


def _krylov_solve(cfg: Dict[str, Any], matvec: Callable, n: int, b: Tensor) -> Tensor:
    """Matrix-free solve of the reduced system, relative to ``||b||``."""
    method = cfg["linear_method"]
    if method in ("auto", "gmres"):
        return _gmres(matvec, b, cfg["krylov_rtol"], cfg["krylov_maxiter"])
    if method == "cg":
        return _cg(matvec, b, cfg["krylov_rtol"], cfg["krylov_maxiter"])
    raise ValueError(f"vi_solve: matrix-free solves support linear_method 'gmres' or 'cg', got {method!r}")


def _jvp(residual_fn: Callable, u: Tensor, params: Tuple, v: Tensor) -> Tensor:
    """J(u) v with a forward-mode-equivalent double-backward product."""
    with torch.enable_grad():
        return torch.autograd.functional.jvp(lambda x: residual_fn(x, *params), u.detach(), v)[1]


def _vjp(residual_fn: Callable, u: Tensor, params: Tuple, v: Tensor) -> Tensor:
    """J(u)^T v."""
    with torch.enable_grad():
        x = u.detach().requires_grad_(True)
        F = residual_fn(x, *params)
        return torch.autograd.grad(F, x, grad_outputs=v)[0]


# ============================================================================
# Reduced-space active-set Newton
# ============================================================================

def _active_set_newton(u: Tensor, lower: Tensor, upper: Tensor, params: Tuple,
                       cfg: Dict[str, Any]):
    residual_fn, jacobian_fn = cfg["residual_fn"], cfg["jacobian_fn"]
    zero_tol, n = cfg["zero_tol"], u.numel()
    fixed = lower == upper
    nfev = 0

    def evaluate(x):
        nonlocal nfev
        nfev += 1
        F = residual_fn(x, *params)
        keep = _inactive(x, F, lower, upper, fixed, zero_tol)
        return F, keep, float(torch.linalg.vector_norm(torch.where(keep, F, torch.zeros_like(F))))

    u = _project(u, lower, upper)
    F, keep, fnorm = evaluate(u)
    fnorm0 = fnorm
    history = [fnorm]

    def monitor(it):
        if cfg["verbose"]:
            print(f"  VI iter {it}: ||F_inactive|| = {fnorm:.3e}, active = {int((~keep).sum())}")

    monitor(0)
    converged, it = fnorm <= cfg["atol"], 0
    while not converged and it < cfg["max_iter"]:
        it += 1
        rhs = torch.where(keep, -F, torch.zeros_like(F))
        if jacobian_fn is not None:
            val, row, col = _as_coo(jacobian_fn(u, *params), n)
            d = _linear_solve(cfg, *_reduced_coo(val, row, col, keep, n), n, rhs)
        else:
            def matvec(v, keep=keep, u=u):
                Jv = _jvp(residual_fn, u, params, torch.where(keep, v, torch.zeros_like(v)))
                return torch.where(keep, Jv, v)
            d = _krylov_solve(cfg, matvec, n, rhs)
        d = torch.where(keep, d, torch.zeros_like(d))
        if cfg["line_search"]:
            # projected backtracking on the inactive-set residual norm
            lam = 1.0
            for _ in range(cfg["line_search_max_iter"]):
                u_t = _project(u + lam * d, lower, upper)
                F_t, keep_t, fnorm_t = evaluate(u_t)
                if fnorm_t <= (1.0 - 1e-4 * lam) * fnorm:
                    break
                lam *= 0.5
            u, F, keep, fnorm = u_t, F_t, keep_t, fnorm_t
        else:
            u = _project(u + d, lower, upper)
            F, keep, fnorm = evaluate(u)
        history.append(fnorm)
        monitor(it)
        if not math.isfinite(fnorm):
            raise RuntimeError(f"vi_solve: residual is not finite at iteration {it}")
        converged = fnorm <= cfg["atol"] or fnorm <= cfg["tol"] * fnorm0
    if not converged:
        warnings.warn(f"vi_solve did not converge in {cfg['max_iter']} iterations, "
                      f"||F_inactive|| = {fnorm:.2e}")
    info = {"converged": converged, "iterations": it, "residual": fnorm,
            "function_evaluations": nfev, "active": int((~keep).sum()), "history": history}
    return u, F, keep, info


# ============================================================================
# Autograd Function
# ============================================================================

class VISolveAdjoint(Function):
    """Active-set Newton forward, implicit-function adjoint backward.

    At the solution the inactive components satisfy ``F_I(u, θ) = 0`` and the
    active ones sit on their bounds. With the active set held fixed (strict
    complementarity), ``∂L/∂θ = -λᵀ ∂F/∂θ`` with ``J_IIᵀ λ_I = ∂L/∂u_I``,
    ``λ_A = 0``, and an active bound receives ``∂L/∂u_A - (Jᵀλ)_A``.
    """

    @staticmethod
    def forward(ctx, u0: Tensor, lower: Tensor, upper: Tensor, cfg: Dict[str, Any], *params):
        params_detached = tuple(p.detach() if isinstance(p, Tensor) else p for p in params)
        lower, upper = lower.detach(), upper.detach()
        u, F, keep, info = _active_set_newton(u0.detach().clone(), lower, upper, params_detached, cfg)
        cfg["info"].update(info)

        fixed = lower == upper
        ctx.at_lower = ~keep & ~fixed & (F > 0)
        ctx.at_upper = ~keep & ~fixed & (F < 0)
        ctx.fixed = fixed
        ctx.keep = keep
        ctx.cfg = cfg
        ctx.param_is_tensor = [isinstance(p, Tensor) for p in params]
        ctx.save_for_backward(u, *[p for p in params if isinstance(p, Tensor)])
        ctx.params_other = [None if isinstance(p, Tensor) else p for p in params]
        return u

    @staticmethod
    def backward(ctx, grad_u: Tensor):
        u, *tensors = ctx.saved_tensors
        it = iter(tensors)
        params = [next(it) if is_t else other for is_t, other in zip(ctx.param_is_tensor, ctx.params_other)]
        cfg, keep, n = ctx.cfg, ctx.keep, u.numel()
        residual_fn, jacobian_fn = cfg["residual_fn"], cfg["jacobian_fn"]
        need_lower, need_upper = ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        need_params = [is_t and ctx.needs_input_grad[4 + i] for i, is_t in enumerate(ctx.param_is_tensor)]
        params_detached = tuple(p.detach() if isinstance(p, Tensor) else p for p in params)

        # adjoint: [J_II 0; 0 I]^T lambda = [g_I; 0]
        rhs = torch.where(keep, grad_u, torch.zeros_like(grad_u))
        J = None
        if jacobian_fn is not None:
            with torch.no_grad():
                val, row, col = _as_coo(jacobian_fn(u, *params_detached), n)
            rv, rr, rc = _reduced_coo(val.detach(), row, col, keep, n)
            lam = _linear_solve(cfg, rv, rc, rr, n, rhs)  # transpose: swap row and col
            J = (val.detach(), row, col)
        else:
            def matvec_t(v):
                JTv = _vjp(residual_fn, u, params_detached, torch.where(keep, v, torch.zeros_like(v)))
                return torch.where(keep, JTv, v)
            lam = _krylov_solve(cfg, matvec_t, n, rhs)
        lam = torch.where(keep, lam, torch.zeros_like(lam))

        grads_params = [None] * len(params)
        if any(need_params):
            with torch.enable_grad():
                var = [p.detach().requires_grad_(True) if need else
                       (p.detach() if isinstance(p, Tensor) else p)
                       for p, need in zip(params, need_params)]
                F = residual_fn(u.detach(), *var)
                wrt = [v for v, need in zip(var, need_params) if need]
                g = torch.autograd.grad(F, wrt, grad_outputs=lam, allow_unused=True)
            g = iter(g)
            for i, need in enumerate(need_params):
                if need:
                    gi = next(g)
                    grads_params[i] = None if gi is None else -gi

        grad_lower = grad_upper = None
        if need_lower or need_upper:
            if J is not None:
                val, row, col = J
                JTlam = torch.zeros_like(lam).index_add_(0, col, val * lam[row])
            else:
                JTlam = _vjp(residual_fn, u, params_detached, lam)
            g_bound = grad_u - JTlam
            zero = torch.zeros_like(g_bound)
            if need_lower:
                # fixed components (lower == upper) report to `lower` only
                grad_lower = torch.where(ctx.at_lower | ctx.fixed, g_bound, zero)
            if need_upper:
                grad_upper = torch.where(ctx.at_upper, g_bound, zero)
        return (None, grad_lower, grad_upper, None, *grads_params)


# ============================================================================
# High-level API
# ============================================================================

def vi_solve(
    residual_fn: Callable,
    u0: Tensor,
    *params,
    lower: Union[Tensor, float, None] = None,
    upper: Union[Tensor, float, None] = None,
    jacobian_fn: Optional[Callable] = None,
    tol: float = 1e-10,
    atol: float = 1e-10,
    max_iter: int = 50,
    line_search: bool = False,
    line_search_max_iter: int = 20,
    zero_tol: float = 1e-8,
    linear_solver: Union[str, Callable] = "auto",
    linear_method: str = "auto",
    linear_kwargs: Optional[Dict[str, Any]] = None,
    krylov_rtol: float = 1e-10,
    krylov_maxiter: int = 10000,
    verbose: bool = False,
    return_info: bool = False,
):
    r"""
    Solve the bound-constrained nonlinear problem (variational inequality)

    .. math::

        \text{find } u,\ l \le u \le r:\quad
        F_i(u,\theta) \ge 0 \text{ if } u_i = l_i,\quad
        F_i(u,\theta) \le 0 \text{ if } u_i = r_i,\quad
        F_i(u,\theta) = 0 \text{ otherwise},

    with adjoint gradients. For :math:`F = \nabla E` these are the KKT
    conditions of :math:`\min E(u)` subject to :math:`l \le u \le r`
    (obstacle and contact problems, damage and phase-field models with
    irreversibility, bound-constrained optimization). Components with
    :math:`l_i = r_i` are fixed at that value, e.g. Dirichlet conditions.

    Algorithm
    ---------
    **Reduced-space active-set Newton**, the method of PETSc's
    ``SNESVINEWTONRSLS``. Each iteration takes the active set
    :math:`\mathcal A = \{u_i \le l_i + \epsilon, F_i > 0\} \cup
    \{u_i \ge r_i - \epsilon, F_i < 0\} \cup \{l_i = r_i\}`, solves the Newton
    system on the inactive components :math:`\mathcal I` only, and projects:

    .. code-block:: text

        u = clip(u0, l, r)
        repeat until ||F_I(u)|| <= atol or <= tol * ||F_I(u0)||:
            I  = inactive set of (u, F(u))
            d  = -[J_II 0; 0 I]^{-1} [F_I; 0]    # active components do not move
            u  = clip(u + lambda * d, l, r)       # lambda = 1, or backtracking

    The active rows and columns of :math:`J` are replaced by the identity
    rather than removed, so the matrix keeps one sparsity pattern for the
    whole solve and direct solvers can reuse their symbolic analysis. For a
    convex quadratic energy the iteration is a primal-dual active-set method
    and stops once the active set is right. Each iteration moves the boundary
    of the active set locally (by about one layer of components where it is
    still wrong), so the iteration count depends on the initial guess: in an
    incremental (load-stepping) setting, start from the previous solution.

    Complexity
    ----------
    One sparse linear solve of size :math:`n` and one residual evaluation per
    iteration (plus the backtracking evaluations with ``line_search``); the
    iteration count is usually small and largely independent of :math:`n`.

    Backward
    --------
    **Implicit differentiation** at the solution with the active set held
    fixed: one adjoint solve :math:`J_{\mathcal{II}}^{T}\lambda_{\mathcal I} =
    \partial L/\partial u_{\mathcal I}` (:math:`\lambda_{\mathcal A} = 0`),
    then :math:`\partial L/\partial\theta = -\lambda^{T}\,\partial F/\partial\theta`
    for the parameters and :math:`\partial L/\partial u_{\mathcal A} -
    (J^{T}\lambda)_{\mathcal A}` for the bounds an active component sits on
    (a fixed component reports to ``lower``). The derivative assumes strict
    complementarity; it is not defined where a component is on a bound with
    :math:`F_i = 0`.

    Parameters
    ----------
    residual_fn : Callable
        ``F(u, *params) -> Tensor`` shaped like ``u`` (1-D).
    u0 : torch.Tensor
        Initial guess, projected onto the bounds.
    *params : torch.Tensor or object
        Passed to ``residual_fn`` and ``jacobian_fn``; tensors with
        ``requires_grad=True`` receive gradients.
    lower, upper : torch.Tensor, float or None
        Bounds, broadcast to ``u``. ``None`` means unbounded. Tensors that
        require grad receive gradients through the active set.
    jacobian_fn : Callable, optional
        ``J(u, *params)`` returning :math:`\partial F/\partial u` as a
        :class:`~torch_sla.SparseTensor`, a ``(val, row, col, shape)`` tuple
        or a torch sparse COO tensor. If ``None``, Jacobian-vector products
        from autograd are used with a Krylov solver (``linear_method``,
        default GMRES).
    tol, atol : float
        Relative and absolute tolerance on the norm of ``F`` over the
        inactive set.
    max_iter : int
        Maximum Newton iterations (a warning is issued if reached).
    line_search : bool
        Projected backtracking on the inactive residual norm. Default
        ``False``: full projected Newton steps, which suit (piecewise)
        quadratic energies.
    line_search_max_iter : int
        Maximum step halvings per iteration.
    zero_tol : float
        Distance to a bound below which a component counts as on the bound.
    linear_solver : str or Callable
        Backend for :func:`~torch_sla.spsolve` (``"auto"``, ``"scipy"``,
        ``"cudss"``, ``"pytorch"``, ...) or a callable
        ``solver(val, row, col, shape, b) -> x``; the COO input may contain
        duplicate entries, which are to be summed.
    linear_method : str
        Method for :func:`~torch_sla.spsolve` or, without ``jacobian_fn``,
        the Krylov method (``"gmres"``, ``"cg"``, ``"bicgstab"``, ...).
    linear_kwargs : dict, optional
        Extra keyword arguments for :func:`~torch_sla.spsolve`, e.g.
        ``{"matrix_type": "spd"}``.
    krylov_rtol, krylov_maxiter : float, int
        Tolerance and iteration cap of the matrix-free Krylov solves.
    verbose : bool
        Print the residual norm and the size of the active set per iteration.
    return_info : bool
        Also return a dict with ``converged``, ``iterations``, ``residual``,
        ``function_evaluations``, ``active`` (size of the final active set)
        and ``history`` (residual norms).

    Returns
    -------
    torch.Tensor or (torch.Tensor, dict)
        The solution ``u``, and the info dict if ``return_info``.

    Examples
    --------
    >>> # obstacle problem: -u'' = f with u >= psi, u(0) = u(1) = 0
    >>> import torch
    >>> from torch_sla import SparseTensor, vi_solve
    >>> n = 99; h = 1.0 / (n + 1)
    >>> i = torch.arange(n)
    >>> row = torch.cat([i, i[1:], i[:-1]]); col = torch.cat([i, i[:-1], i[1:]])
    >>> val = torch.cat([torch.full((n,), 2.0), torch.full((2 * n - 2,), -1.0)]) / h**2
    >>> A = SparseTensor(val.double(), row, col, (n, n))
    >>> f = torch.full((n,), -10.0, dtype=torch.float64, requires_grad=True)
    >>> psi = -0.2 + 0.0 * f.detach()
    >>> u = vi_solve(lambda u, f: A @ u - f, torch.zeros(n, dtype=torch.float64), f,
    ...              lower=psi, jacobian_fn=lambda u, f: A)
    >>> u.sum().backward()   # f.grad is zero where u rests on the obstacle
    """
    if u0.dim() != 1:
        raise ValueError(f"vi_solve: u0 must be 1-D, got shape {tuple(u0.shape)}")
    lower_t = _full_bound(u0, lower, -math.inf)
    upper_t = _full_bound(u0, upper, math.inf)
    if bool((lower_t > upper_t).any()):
        raise ValueError("vi_solve: lower > upper for some components")
    cfg = {
        "residual_fn": residual_fn,
        "jacobian_fn": jacobian_fn,
        "tol": tol,
        "atol": atol,
        "max_iter": max_iter,
        "line_search": line_search,
        "line_search_max_iter": line_search_max_iter,
        "zero_tol": zero_tol,
        "linear_solver": linear_solver,
        "linear_method": linear_method,
        "linear_kwargs": dict(linear_kwargs or {}),
        "krylov_rtol": krylov_rtol,
        "krylov_maxiter": krylov_maxiter,
        "verbose": verbose,
        "info": {},
    }
    u = VISolveAdjoint.apply(u0, lower_t, upper_t, cfg, *params)
    return (u, cfg["info"]) if return_info else u
