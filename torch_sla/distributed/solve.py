"""Shard(0)-space Krylov methods and preconditioners for
:class:`~torch_sla.distributed.DSparseTensor`.

Every routine here runs entirely in the row-sharded space: vectors
stay distributed across the device mesh, inner products go through a
single ``dist.all_reduce(SUM)`` per use, and matvecs route through
``D._shard_matvec`` (halo exchange + local SpMV). No rank ever
materialises the global solution.

These functions are deliberately written as free functions rather
than methods on :class:`DSparseTensor` -- keeps the class focused on
data layout and lets the Krylov implementations evolve independently.
:meth:`DSparseTensor.solve_distributed_shard` is the public dispatcher.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import torch

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


# ====================================================================== #
# Preconditioner factory                                                 #
# ====================================================================== #
def make_preconditioner(D, kind: Any, *, omega: float = 1.0,
                         degree: int = 2) -> Callable:
    """Build a ``apply(r_owned) -> z_owned`` Shard(0) preconditioner.

    Accepts:

    * ``None`` / ``"none"`` -- identity (no precond)
    * ``"jacobi"`` / ``"jacobi_l1"`` -- diagonal inverse on owned
      rows (the l1 variant scales by row-sum-of-abs instead of
      diag(A); more robust for indefinite spectra, Anzt et al. 2015)
    * ``"block_jacobi"`` -- one-time dense LU on the owned-by-owned
      block; ``apply`` does two triangular solves
    * ``"ssor"`` -- forward + backward symmetric SOR sweep on the
      owned-by-owned block, ``omega``-damped
    * ``"amg"`` / ``"pyamg"`` -- PyAMG V-cycle per call, hierarchy built
      from the owned-rows × owned-cols block (block-Jacobi AMG).
      Cached on the DSparseTensor instance via
      ``D._amg_hierarchy_cache``; call :func:`invalidate_precond_cache`
      after the matrix values change. **Hierarchy build runs on CPU**
      (PyAMG limitation); for production-scale CUDA workloads prefer
      ``"amgx"`` below.
    * ``"amgx"`` / ``"torch_amgx"`` -- AmgX V-cycle, hierarchy build +
      apply both on GPU via the ``torch-amgx`` package. Same
      block-Jacobi structure as PyAMG but stays on CUDA the entire
      time. Cached on ``D._amgx_solver_cache``; same invalidation
      semantics. Requires Linux/Windows CUDA + ``pip install torch-amgx``.
    * ``"polynomial"`` / ``"neumann"`` -- Neumann series
      M⁻¹ ≈ τ⁻¹ Σ_{k=0..degree-1} (I - A/τ)^k. Uses
      ``D._shard_matvec`` so halo exchange happens inside the
      preconditioner -- costly but fully distributed.
    * A callable ``f(r_owned) -> z_owned`` is returned unchanged.
    """
    if callable(kind):
        return kind
    if kind is None or (isinstance(kind, str) and kind.lower() == "none"):
        return lambda r: r

    partition = D._spec.placement.partition
    no = int(partition.owned_nodes.numel())
    device = D._local_tensor.values.device

    def _owned_block() -> torch.Tensor:
        """Materialise the owned-by-owned dense block on first use.

        Filter the local SparseTensor's COO entries down to (row < no
        AND col < no) and scatter into a ``(no, no)`` dense tensor.
        **O(no²) memory** -- only call for preconditioners that genuinely
        need the full block (block-Jacobi LU, SSOR sweep). Jacobi /
        jacobi_l1 / polynomial all bypass this and stay O(nnz).
        """
        if getattr(D, "_owned_block_cache", None) is not None:
            return D._owned_block_cache
        st = D._local_tensor
        rows = st.row_indices
        cols = st.col_indices
        mask = (rows < no) & (cols < no)
        block = torch.zeros((no, no), dtype=st.values.dtype, device=device)
        block.index_put_((rows[mask], cols[mask]), st.values[mask],
                          accumulate=True)
        D._owned_block_cache = block
        return block

    kind_l = str(kind).lower()
    if kind_l in ("jacobi", "jacobi_l1"):
        # Diagonal / row-l1 preconditioners only need O(no) memory.
        # Materialising the full ``(no, no)`` block (a la _owned_block)
        # explodes at scale -- a 1M-owned partition needs 8 TB of dense.
        st = D._local_tensor
        rows = st.row_indices
        cols = st.col_indices
        vals = st.values
        if kind_l == "jacobi":
            # Sum diagonal entries (handles COO duplicates).
            mask = (rows == cols) & (rows < no)
            diag = torch.zeros(no, dtype=vals.dtype, device=device)
            diag.scatter_add_(0, rows[mask], vals[mask])
            scale = diag.clamp_min(1e-30)
        else:  # jacobi_l1: sum |A_ij| over each owned row.
            mask = rows < no
            row_l1 = torch.zeros(no, dtype=vals.dtype, device=device)
            row_l1.scatter_add_(0, rows[mask], vals[mask].abs())
            scale = row_l1.clamp_min(1e-30)
        inv_scale = 1.0 / scale
        return lambda r: inv_scale * r

    if kind_l == "block_jacobi":
        block = _owned_block()
        LU, pivots = torch.linalg.lu_factor(block)

        def apply_block_jacobi(r):
            return torch.linalg.lu_solve(
                LU, pivots, r.unsqueeze(-1)).squeeze(-1)

        return apply_block_jacobi

    if kind_l == "ssor":
        block = _owned_block()
        # block = L + D + U; ω-SSOR is (D/ω + L) (D/ω)⁻¹ (D/ω + U) z = r.
        D_diag = torch.diag(block.diagonal().clamp_min(1e-30))
        L = torch.tril(block, diagonal=-1)
        U = torch.triu(block, diagonal=1)
        D_over_om = D_diag / omega
        M1 = D_over_om + L
        M2 = D_over_om + U

        def apply_ssor(r):
            y = torch.linalg.solve_triangular(
                M1, r.unsqueeze(-1), upper=False).squeeze(-1)
            w = D_over_om @ y
            return torch.linalg.solve_triangular(
                M2, w.unsqueeze(-1), upper=True).squeeze(-1)

        return apply_ssor

    if kind_l in ("amgx", "torch_amgx"):
        # CUDA-native AMG preconditioner via torch-amgx. AmgX runs the
        # full hierarchy build + V-cycle entirely on the GPU -- no PyAMG
        # CPU roundtrip. Cached on D._amgx_solver_cache.
        cached = getattr(D, "_amgx_solver_cache", None)
        if cached is not None:
            return cached

        try:
            from torch_amgx import Config as _AmgxConfig, Solver as _AmgxSolver
        except ImportError as e:
            raise RuntimeError(
                f"AmgX preconditioner needs torch-amgx: {e}. "
                "Install via: pip install torch-amgx (Linux/Windows CUDA only)."
            )

        st = D._local_tensor
        rows = st.row_indices
        cols = st.col_indices
        vals = st.values
        mask = (rows < no) & (cols < no)
        # AmgX requires CSR with int32 indices on CUDA. Build the owned
        # sub-block as a torch.sparse_coo, coalesce, convert to CSR.
        owned_indices = torch.stack(
            [rows[mask].to(torch.int64), cols[mask].to(torch.int64)])
        sparse_coo = torch.sparse_coo_tensor(
            owned_indices, vals[mask], (no, no)).coalesce()
        sparse_csr = sparse_coo.to_sparse_csr()
        crow = sparse_csr.crow_indices().to(torch.int32)
        ccol = sparse_csr.col_indices().to(torch.int32)
        cval = sparse_csr.values()

        # max_iters=1, tolerance=0 forces exactly one V-cycle per solve()
        # call regardless of residual (we are a preconditioner, the outer
        # CG decides convergence).
        config = _AmgxConfig(
            method="amg", maxiter=1, tol=0.0,
            presweeps=1, postsweeps=1,
        )
        solver = _AmgxSolver(config, device=device)
        solver.setup_csr(crow, ccol, cval, no)

        def apply_amgx(r):
            return solver.solve(r.contiguous())

        # Hold a reference so the AmgX C handles stay alive for the
        # lifetime of D, and the closure for the user.
        D._amgx_solver_cache = apply_amgx
        D._amgx_solver_handle = solver        # keep handle alive
        return apply_amgx

    if kind_l in ("amg", "pyamg"):
        # PyAMG block-Jacobi preconditioner: each rank builds an AMG
        # hierarchy on its owned-rows × owned-cols block, applies one
        # V-cycle per call. Cached on the DSparseTensor instance so
        # repeated solves on the same matrix (time-stepping, inverse
        # design, multi-RHS) pay setup once.
        cached = getattr(D, "_amg_hierarchy_cache", None)
        if cached is not None:
            return cached

        try:
            from ..backends.pyamg_backend import PyAMGHierarchy
        except ImportError as e:
            raise RuntimeError(
                f"AMG preconditioner needs PyAMG: {e}. "
                "Install with: pip install pyamg"
            )

        import scipy.sparse as sp
        st = D._local_tensor
        rows = st.row_indices
        cols = st.col_indices
        vals = st.values
        # Filter to owned-rows × owned-cols block (sparse, no dense alloc).
        mask = (rows < no) & (cols < no)
        r_owned = rows[mask].cpu().numpy()
        c_owned = cols[mask].cpu().numpy()
        v_owned = vals[mask].cpu().numpy()
        A_owned_csr = sp.coo_matrix(
            (v_owned, (r_owned, c_owned)), shape=(no, no)).tocsr()

        hierarchy = PyAMGHierarchy.from_scipy_csr(
            A_owned_csr,
            method="ruge_stuben",
            device=device, dtype=st.values.dtype,
            max_levels=10, max_coarse=128,
            num_pre_smooth=1, num_post_smooth=1,
        )
        # Cache the callable on D so subsequent solves reuse it.
        D._amg_hierarchy_cache = hierarchy
        return hierarchy

    if kind_l in ("polynomial", "neumann"):
        # τ ≈ ||A||_∞ on owned rows; take max across ranks for safety.
        block = _owned_block()
        tau_local = block.abs().sum(dim=1).max()
        if _DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(tau_local, op=dist.ReduceOp.MAX)
        tau = float(tau_local.item()) if float(tau_local.item()) > 0 else 1.0
        d = int(degree)

        def apply_neumann(r):
            # Horner form of τ⁻¹ Σ_{k=0..d-1} (I - A/τ)^k r.
            z = r / tau
            for _ in range(d - 1):
                Az = D._shard_matvec(z)
                z = (r + tau * z - Az) / tau
            return z

        return apply_neumann

    raise ValueError(
        f"Unknown preconditioner {kind!r}; expected one of None / "
        "'none' / 'jacobi' / 'jacobi_l1' / 'block_jacobi' / "
        "'ssor' / 'polynomial' / 'neumann' / 'amg' / 'amgx', "
        "or a callable."
    )


def invalidate_precond_cache(D) -> None:
    """Drop cached preconditioner factors. Call after the matrix's
    local values change (the cache is keyed implicitly on the
    DSparseTensor instance, not on a hash of the values)."""
    D._owned_block_cache = None
    D._amg_hierarchy_cache = None
    D._amgx_solver_cache = None
    D._amgx_solver_handle = None


# ====================================================================== #
# Conjugate Gradient (Saad §9.2)                                         #
# ====================================================================== #
def cg_shard(D, b_owned: torch.Tensor, *, M_apply: Callable,
              atol: float, rtol: float, maxiter: int,
              verbose: bool) -> torch.Tensor:
    """Preconditioned CG in Shard(0) space.

    Saad §9.2 PCG: ``rho_k = <r_k, z_k>`` with ``z_k = M⁻¹ r_k``;
    ``p_k = z_k + beta_{k-1} p_{k-1}``. Identity ``M_apply`` recovers
    plain CG.
    """
    no = D._num_owned()
    if b_owned.shape[0] != no:
        raise ValueError(
            f"b_owned size {b_owned.shape[0]} != num_owned {no}; "
            "Shard(0) CG requires b to be the local owned slice.")

    # Allocate the working set once.
    x = torch.zeros_like(b_owned)
    Ax0 = D._shard_matvec(x)
    r = b_owned - Ax0
    z = M_apply(r)
    p = z.clone()

    # Fuse the initial <r, z> + ||b|| reduces into ONE all_reduce
    # (NCCL latency is dominated by per-call overhead, not payload).
    init2 = torch.stack([torch.dot(r, z), torch.dot(b_owned, b_owned)])
    if _DIST_AVAILABLE and dist.is_initialized():
        dist.all_reduce(init2, op=dist.ReduceOp.SUM)
    rs_old = init2[0].clone()
    b_norm = float(init2[1].sqrt().item())
    tol = max(atol, rtol * b_norm)

    # Hot-loop optimisations:
    # * 2 fused all_reduce per iter instead of 3 (combined <r,r> + <r,z>).
    # * No CPU sync inside the hot path -- convergence is checked every
    #   ``check_every`` iters. CG is monotone in ||r||, so worst-case we
    #   overshoot by <check_every iterations.
    # * In-place axpy via ``addcmul`` / ``mul_().add_()`` -- zero fresh
    #   allocations in the loop body.
    check_every = 10
    r_norm_sq = None

    for k in range(maxiter):
        Ap = D._shard_matvec(p)
        pAp = D._shard_dot(p, Ap)             # all_reduce #1 / iter
        alpha = rs_old / pAp
        torch.addcmul(x, alpha, p, value=1.0, out=x)
        torch.addcmul(r, alpha, Ap, value=-1.0, out=r)

        z = M_apply(r)
        # Fused: <r, r> AND <r, z> in one 2-element all_reduce.
        loc2 = torch.stack([torch.dot(r, r), torch.dot(r, z)])
        if _DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(loc2, op=dist.ReduceOp.SUM)  # all_reduce #2 / iter
        r_norm_sq = loc2[0]
        rs_new = loc2[1]

        # Convergence check + verbose log: only every ``check_every`` iter.
        if k % check_every == 0:
            r_norm = float(r_norm_sq.sqrt().item())
            if verbose and (k % 100 == 0 or k < 5):
                print(f"[shard-PCG] iter {k}: ||r||={r_norm:.3e}")
            if r_norm < tol:
                if verbose:
                    print(f"[shard-PCG] converged at iter {k}, "
                          f"||r||={r_norm:.3e}")
                break

        beta = rs_new / rs_old
        p.mul_(beta).add_(z)                  # p = z + beta * p   in-place
        rs_old = rs_new

    return x


# ====================================================================== #
# BiCGStab (Saad §7.4.2)                                                 #
# ====================================================================== #
def bicgstab_shard(D, b_owned: torch.Tensor, *, M_apply: Callable,
                    atol: float, rtol: float, maxiter: int,
                    verbose: bool) -> torch.Tensor:
    """Preconditioned BiCGStab (Saad §9.3): apply ``M⁻¹`` to the
    search directions ``p`` and ``s`` before the matvec. Identity
    ``M_apply`` recovers plain BiCGStab.

    Optimisations vs textbook PBiCGStab:
    * Fuse ``<t,t>`` and ``<t,s>`` into one 2-element all_reduce
      (5 -> 4 reduces per iter).
    * Convergence check + breakdown check only every ``check_every``
      iters, so CPU doesn't sync with the GPU launch pipeline on
      every iteration.
    """
    no = D._num_owned()
    if b_owned.shape[0] != no:
        raise ValueError(
            f"b_owned size {b_owned.shape[0]} != num_owned {no}")

    x = torch.zeros_like(b_owned)
    r = b_owned - D._shard_matvec(x)
    r_hat = r.clone()
    rho = alpha = omega = torch.tensor(1.0, dtype=b_owned.dtype,
                                        device=b_owned.device)
    p = torch.zeros_like(b_owned)
    v = torch.zeros_like(b_owned)

    b_norm = float(D._shard_norm(b_owned).item())
    tol = max(atol, rtol * b_norm)

    check_every = 10
    last_r_norm = float("inf")

    for k in range(maxiter):
        rho_new = D._shard_dot(r_hat, r)               # all_reduce #1
        if k == 0:
            p = r.clone()
        else:
            beta = (rho_new / rho) * (alpha / omega)
            p = r + beta * (p - omega * v)
        p_hat = M_apply(p)
        v = D._shard_matvec(p_hat)
        denom = D._shard_dot(r_hat, v)                 # all_reduce #2
        alpha = rho_new / denom                        # NaN if breakdown
        s = r - alpha * v
        s_norm_sq = D._shard_dot(s, s)                 # all_reduce #3

        if k % check_every == 0:
            s_norm = float(s_norm_sq.sqrt().item())
            if s_norm < tol:
                x = x + alpha * p_hat
                if verbose:
                    print(f"[shard-PBiCGStab] half-iter {k}: "
                          f"||s||={s_norm:.3e}")
                return x

        s_hat = M_apply(s)
        t = D._shard_matvec(s_hat)
        # Fuse <t,t> + <t,s> into a single 2-element all_reduce.
        loc2 = torch.stack([torch.dot(t, t), torch.dot(t, s)])
        if _DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(loc2, op=dist.ReduceOp.SUM)  # all_reduce #4
        omega = loc2[1] / loc2[0]                      # NaN if breakdown
        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t
        r_norm_sq = D._shard_dot(r, r)                 # all_reduce #5

        if k % check_every == 0:
            r_norm = float(r_norm_sq.sqrt().item())
            last_r_norm = r_norm
            if verbose and (k % 100 == 0 or k < 5):
                print(f"[shard-PBiCGStab] iter {k}: ||r||={r_norm:.3e}")
            if r_norm < tol or math.isnan(r_norm):
                if verbose:
                    print(f"[shard-PBiCGStab] stop at iter {k}, "
                          f"||r||={r_norm:.3e}")
                break

        rho = rho_new

    return x


# ====================================================================== #
# Restarted GMRES(m) (Saad §6.5.1) -- optionally flexible (FGMRES)       #
# ====================================================================== #
def gmres_shard(D, b_owned: torch.Tensor, *, M_apply: Callable,
                 atol: float, rtol: float, maxiter: int, restart: int,
                 flexible: bool = False,
                 verbose: bool = False) -> torch.Tensor:
    no = D._num_owned()
    if b_owned.shape[0] != no:
        raise ValueError(
            f"b_owned size {b_owned.shape[0]} != num_owned {no}")

    dtype = b_owned.dtype
    device = b_owned.device
    m = restart

    x = torch.zeros_like(b_owned)
    b_norm = float(D._shard_norm(b_owned).item())
    if b_norm == 0.0:
        return x
    tol = max(atol, rtol * b_norm)

    total_iters = 0
    for cycle in range((maxiter + m - 1) // m):
        r = b_owned - D._shard_matvec(x)
        beta = float(D._shard_norm(r).item())
        if beta < tol:
            if verbose:
                print(f"[shard-{'F' if flexible else ''}GMRES] "
                      f"converged cycle {cycle}, ||r||={beta:.3e}")
            return x

        V = torch.zeros((m + 1, no), dtype=dtype, device=device)
        V[0] = r / beta
        # Right-preconditioned GMRES (Saad §9.3.2): apply A M⁻¹ at every
        # Arnoldi step. For FGMRES the preconditioner may vary per step,
        # so we remember Z[:, j] = M_j⁻¹ V[j] for the final update.
        # Plain GMRES could reuse V (since M is fixed), but storing Z
        # lets both branches share one code path.
        Z = torch.zeros((m, no), dtype=dtype, device=device)
        H = torch.zeros((m + 1, m), dtype=dtype, device=device)
        g = torch.zeros(m + 1, dtype=dtype, device=device)
        g[0] = beta
        cs = torch.zeros(m, dtype=dtype, device=device)
        sn = torch.zeros(m, dtype=dtype, device=device)

        j_max = 0
        for j in range(m):
            Z[j] = M_apply(V[j])
            w = D._shard_matvec(Z[j])
            # Modified Gram-Schmidt (Saad's stable variant for GMRES).
            for i in range(j + 1):
                H[i, j] = D._shard_dot(V[i], w)
                w = w - H[i, j] * V[i]
            H[j + 1, j] = D._shard_norm(w)
            if float(H[j + 1, j].abs().item()) > 1e-30:
                V[j + 1] = w / H[j + 1, j]
            else:
                V[j + 1] = w  # lucky breakdown -- next iter will catch

            # Apply previous Givens rotations to column j of H.
            # ``.clone()`` is mandatory: H[i, j] returns a 0-d view;
            # writing to H[i, j] in the first assignment would otherwise
            # leak into the second through the h_ij alias.
            for i in range(j):
                h_ij  = H[i, j].clone()
                h_ipj = H[i + 1, j].clone()
                H[i,     j] =  cs[i] * h_ij + sn[i] * h_ipj
                H[i + 1, j] = -sn[i] * h_ij + cs[i] * h_ipj
            # New Givens rotation eliminating H[j+1, j].
            denom = (H[j, j] * H[j, j] + H[j + 1, j] * H[j + 1, j]).sqrt()
            if float(denom.abs().item()) < 1e-30:
                cs[j] = torch.tensor(1.0, dtype=dtype, device=device)
                sn[j] = torch.tensor(0.0, dtype=dtype, device=device)
            else:
                cs[j] = H[j, j] / denom
                sn[j] = H[j + 1, j] / denom
            H[j, j]     = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = torch.tensor(0.0, dtype=dtype, device=device)
            # ``.clone()`` is essential here too: g[j] is a 0-d view;
            # the next two writes would otherwise see the partially
            # updated value through the alias.
            g_j = g[j].clone()
            g[j]     =  cs[j] * g_j
            g[j + 1] = -sn[j] * g_j

            total_iters += 1
            j_max = j + 1
            if float(g[j + 1].abs().item()) < tol:
                break
            if total_iters >= maxiter:
                break

        # Back-solve H[:j_max, :j_max] y = g[:j_max].
        y = torch.zeros(j_max, dtype=dtype, device=device)
        for i in range(j_max - 1, -1, -1):
            s = g[i].clone()
            for k in range(i + 1, j_max):
                s = s - H[i, k] * y[k]
            y[i] = s / H[i, i]
        # Right-preconditioned update: x += Z y. The two cases coincide
        # when M is fixed per-call; FGMRES is identical here because
        # Z[j] = M_j⁻¹ V[j] is stored above.
        for i in range(j_max):
            x = x + y[i] * Z[i]

        if verbose:
            r_norm = float(D._shard_norm(
                b_owned - D._shard_matvec(x)).item())
            print(f"[shard-{'F' if flexible else ''}GMRES] "
                  f"cycle {cycle}: ||r||={r_norm:.3e}")
        if total_iters >= maxiter:
            break

    return x


# ====================================================================== #
# MINRES (Paige & Saunders 1975) -- symmetric indefinite                 #
# ====================================================================== #
def minres_shard(D, b_owned: torch.Tensor, *, M_apply: Callable,
                  atol: float, rtol: float, maxiter: int,
                  verbose: bool) -> torch.Tensor:
    """Distributed MINRES in Shard(0) space (Paige & Saunders 1975).

    Faithful port of SciPy's ``scipy.sparse.linalg.minres`` Lanczos
    + 2-step Givens recurrence -- variable names match SciPy's source
    so the math is reviewable against it. Every inner product runs
    through ``D._shard_dot`` (local dot + all_reduce SUM); matvecs
    route through ``D._shard_matvec`` (halo + local SpMV). Identity
    ``M_apply`` recovers plain MINRES.
    """
    no = D._num_owned()
    if b_owned.shape[0] != no:
        raise ValueError(
            f"b_owned size {b_owned.shape[0]} != num_owned {no}")

    eps_floor = 1e-30
    x = torch.zeros_like(b_owned)

    # Initial residual r = b - A x = b (since x = 0).
    r1 = b_owned.clone()
    y = M_apply(r1)
    beta1_sq = float(D._shard_dot(r1, y).item())
    if beta1_sq <= 0.0:
        return x
    beta1 = beta1_sq ** 0.5
    tol = max(atol, rtol * beta1)

    # Lanczos / Givens scalars (track previous two).
    oldb = 0.0
    beta = beta1
    dbar = 0.0
    epsln = 0.0
    phibar = beta1
    cs = -1.0
    sn = 0.0
    w  = torch.zeros_like(b_owned)
    w2 = torch.zeros_like(b_owned)
    r2 = r1.clone()

    for k in range(maxiter):
        s_scale = 1.0 / max(beta, eps_floor)
        v = s_scale * y
        y = D._shard_matvec(v)
        if k > 0:
            y = y - (beta / oldb) * r1

        alfa = float(D._shard_dot(v, y).item())
        y = y - (alfa / beta) * r2
        r1 = r2
        r2 = y
        y = M_apply(r2)
        oldb = beta
        beta_sq = float(D._shard_dot(r2, y).item())
        beta = (beta_sq if beta_sq > 0.0 else 0.0) ** 0.5

        # Apply previous rotation Q_{k-1}.
        oldeps = epsln
        delta  = cs * dbar + sn * alfa
        gbar   = sn * dbar - cs * alfa
        epsln  = sn * beta
        dbar   = -cs * beta

        # New rotation Q_k.
        gamma = max((gbar * gbar + beta * beta) ** 0.5, eps_floor)
        cs    = gbar / gamma
        sn    = beta / gamma
        phi   = cs * phibar
        phibar = sn * phibar

        # Update solution and search direction.
        denom_inv = 1.0 / gamma
        w1 = w2
        w2 = w
        w  = (v - oldeps * w1 - delta * w2) * denom_inv
        x  = x + phi * w

        rnorm = abs(phibar)
        if verbose and (k % 100 == 0 or k < 5):
            print(f"[shard-MINRES] iter {k}: ||r||~={rnorm:.3e}")
        if rnorm < tol:
            if verbose:
                print(f"[shard-MINRES] converged at iter {k}, "
                      f"||r||~={rnorm:.3e}")
            break

    return x
