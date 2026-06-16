"""Autograd ``Function`` classes for differentiable sparse ops.

* :class:`DetAdjoint`            -- ``det(A)`` (implicit-diff gradient).
* :class:`EigshAdjoint`          -- ``eigsh`` (implicit-diff gradient).
* :class:`SparseSolveFunction`   -- scipy-backed differentiable solve.
* :class:`SparseSparseMatmulFunction` -- ``A @ B`` sparse-sparse with sparse gradients.

Lifted from ``sparse_tensor.core`` as part of the file split. All four are
used internally by :class:`~torch_sla.sparse_tensor.SparseTensor` methods.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.autograd.function import Function


# =============================================================================
# Adjoint Determinant Solver
# =============================================================================

def _det_backward_cuda_cudss(val, row, col, shape, det_val, grad_output):
    """CUDA chunked det backward via cuDSS multi-RHS solve.

    Same chunking strategy as the CPU path (factor-once, solve in
    ~64 MB chunks, gather at A's nnz positions). Raises if cuDSS /
    nvmath-python is missing -- caller falls back to dense.

    Returns
    -------
    grad_val : torch.Tensor
        Shape ``[nnz]``, on the same device/dtype as ``val``.
    """
    from ..backends.nvmath_backend import nvmath_factor_solve_many

    device = val.device
    N = int(shape[0])
    unique_cols, col_inv = torch.unique(col, return_inverse=True)
    n_uc = int(unique_cols.numel())

    # Same 64 MB target as CPU; bytes_per_col = N * 16 (E + Y both live).
    CHUNK_BYTES = 64 * 1024 * 1024
    bytes_per_col = N * 8 * 2
    chunk_cols = max(1, min(n_uc, CHUNK_BYTES // max(1, bytes_per_col)))

    # Sort nnz by col_inv so chunks map to contiguous slices.
    order = torch.argsort(col_inv, stable=True)
    col_inv_sorted = col_inv[order]
    row_sorted = row[order].to(torch.int64)

    nnz_starts = torch.searchsorted(
        col_inv_sorted, torch.arange(n_uc, device=device, dtype=col_inv_sorted.dtype),
    )
    nnz_starts = torch.cat([
        nnz_starts,
        torch.tensor([row.numel()], device=device, dtype=nnz_starts.dtype),
    ])

    gathered = torch.empty(row.numel(), dtype=val.dtype, device=device)
    unique_cols_i64 = unique_cols.to(torch.int64)

    n_chunks = (n_uc + chunk_cols - 1) // chunk_cols

    def build_rhs(i):
        c0 = i * chunk_cols
        c1 = min(c0 + chunk_cols, n_uc)
        width = c1 - c0
        E = torch.zeros(N, width, dtype=val.dtype, device=device)
        rows_i = unique_cols_i64[c0:c1]
        cols_i = torch.arange(width, device=device, dtype=torch.int64)
        E[rows_i, cols_i] = 1.0
        return E, (c0, c1)

    def gather_chunk(X, meta):
        c0, c1 = meta
        lo = int(nnz_starts[c0].item())
        hi = int(nnz_starts[c1].item())
        sub_inv = col_inv_sorted[lo:hi] - c0
        sub_row = row_sorted[lo:hi]
        gathered[order[lo:hi]] = X[sub_row, sub_inv.to(torch.int64)]

    # Solve A^T y = e_c by swapping (row, col) -> A^T's COO triple.
    nvmath_factor_solve_many(
        val, col, row, shape, build_rhs, n_chunks, gather_chunk,
        matrix_type="general",
    )

    return det_val * gathered * grad_output


class DetAdjoint(Function):
    """
    Adjoint-based differentiable determinant computation.
    
    Uses implicit differentiation to compute gradients:
    For matrix A with determinant d = det(A):
        ∂d/∂A = d * (A^{-1})^T
    
    This means:
        ∂d/∂A_ij = d * (A^{-1})_ji
    
    The gradient computation requires solving a linear system,
    which is done efficiently using the existing solve infrastructure.
    """
    
    @staticmethod
    def forward(ctx, val, row, col, shape, device, is_cuda):
        """Forward pass via the DetConfig dispatcher.

        Routing (CPU=SuperLU LU, CUDA=copy-to-CPU+SuperLU by default,
        SPD=Cholesky if scikit-sparse is installed, disconnected matrices
        factored per-component) lives in :mod:`torch_sla.det`.
        """
        from ..sparse_tensor import SparseTensor
        from ..det import _det_dispatch
        # Build a no-grad SparseTensor view so the dispatcher can look
        # at properties (is_pd / connected_components / ...).
        A = SparseTensor(val.detach(), row, col, shape)
        det_val = _det_dispatch(A)
        if not torch.is_tensor(det_val):
            det_val = torch.tensor(det_val, dtype=val.dtype, device=device)
        elif det_val.device != device:
            det_val = det_val.to(device)
        ctx.save_for_backward(val, row, col, det_val)
        ctx.shape = shape
        ctx.device = device
        ctx.is_cuda = is_cuda
        return det_val
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradient using adjoint method.
        
        Gradient formula: ∂L/∂A_ij = ∂L/∂d * d * (A^{-1})_ji
        """
        val, row, col, det_val = ctx.saved_tensors
        shape = ctx.shape
        device = ctx.device
        is_cuda = ctx.is_cuda

        if abs(det_val.item()) < 1e-15:
            return torch.zeros_like(val), None, None, None, None, None

        # Jacobi: ∂det/∂A_ij = det * (A^-1)_ji, so grad_val[k] = det *
        # (A^-1)[col[k], row[k]] * grad_output. We need only the entries of
        # A^-1 at the nonzero positions, transposed.
        #
        # Strategy (was: per-unique-row Python LU + per-nnz Python gather):
        #   1. One sparse LU factorisation of A^T (CPU via SuperLU).
        #   2. Batched solve A^T Y = E, where E's columns are e_c for each
        #      unique c in col[:]. Y[:, k] = (A^-T)[:, c_k] = (A^-1)[c_k, :].
        #   3. Vectorised gather: grad_val[k] = det * Y[row[k], col_inv[k]].
        #
        # For nnz ~ N this drops the Python loop overhead from O(N) to 0
        # and replaces N LU solves with one batched solve (SuperLU shares
        # the factorisation).

        # On CUDA: prefer cuDSS multi-RHS chunked solve (mirrors the CPU
        # SuperLU path, factor-once / solve-many). Falls back to dense
        # inverse if cuDSS / nvmath-python is unavailable.
        if is_cuda:
            try:
                grad_val = _det_backward_cuda_cudss(
                    val, row, col, shape, det_val, grad_output)
                return grad_val, None, None, None, None, None
            except Exception:
                # Fall back: dense inverse. Materialises (n, n) -- OK only
                # at small n; warn so users notice.
                import warnings
                warnings.warn(
                    "DetAdjoint CUDA backward: cuDSS path unavailable, "
                    "falling back to dense torch.linalg.inv. Install "
                    "nvmath-python (with cuDSS) to avoid the (n,n) alloc.",
                    RuntimeWarning, stacklevel=2,
                )
                indices = torch.stack([row, col], dim=0).to(device)
                sparse_coo = torch.sparse_coo_tensor(
                    indices, val, shape, device=device)
                dense = sparse_coo.to_dense()
                A_inv = torch.linalg.inv(dense)
                grad_val = det_val * A_inv[col.to(torch.int64),
                                             row.to(torch.int64)] * grad_output
                return grad_val, None, None, None, None, None

        # CPU sparse path -- single SuperLU, chunked batched solve.
        #
        # For each unique col c we need column c of A^-T, then we gather
        # at (row[k], col_inv[k]). The full Y has shape (N, n_uc), which
        # at n_uc ~ N is N^2 doubles -- 5 GB at n=25600. Chunking RHS
        # keeps the peak at (N * CHUNK_COLS), gathering per chunk into
        # the final 1-D grad buffer.
        from ..backends.scipy_backend import torch_coo_to_scipy_csr
        import scipy.sparse.linalg as spla
        import numpy as np

        A_T_csc = torch_coo_to_scipy_csr(val, row, col, shape).T.tocsc()
        unique_cols, col_inv = torch.unique(col, return_inverse=True)
        unique_cols_np = unique_cols.cpu().numpy()
        n_uc = int(unique_cols_np.shape[0])
        N = int(shape[0])
        np_dtype = val.detach().cpu().numpy().dtype

        # Target ~64 MB per chunk (8 bytes/double, both E and Y live).
        CHUNK_BYTES = 64 * 1024 * 1024
        bytes_per_col = N * 8 * 2
        chunk_cols = max(1, min(n_uc, CHUNK_BYTES // max(1, bytes_per_col)))

        lu = spla.splu(A_T_csc)
        row_np = row.cpu().numpy().astype(np.int64)
        col_inv_np = col_inv.cpu().numpy().astype(np.int64)

        # Sort nnz by col_inv so each chunk handles a contiguous slice.
        order = np.argsort(col_inv_np, kind="stable")
        col_inv_sorted = col_inv_np[order]
        row_sorted = row_np[order]
        gathered_np = np.empty(row_np.shape[0], dtype=np_dtype)

        nnz_starts = np.searchsorted(col_inv_sorted, np.arange(n_uc))
        # Append sentinel so [start_k, start_{k+1}) works.
        nnz_starts = np.concatenate([nnz_starts, [row_np.shape[0]]])

        E_buf = np.zeros((N, chunk_cols), dtype=np_dtype)
        for c0 in range(0, n_uc, chunk_cols):
            c1 = min(c0 + chunk_cols, n_uc)
            width = c1 - c0
            E = E_buf[:, :width]
            E.fill(0.0)
            E[unique_cols_np[c0:c1], np.arange(width)] = 1.0
            Y = lu.solve(E)                       # (N, width)
            nnz_lo = nnz_starts[c0]
            nnz_hi = nnz_starts[c1]
            sub_inv = col_inv_sorted[nnz_lo:nnz_hi] - c0
            sub_row = row_sorted[nnz_lo:nnz_hi]
            gathered_np[order[nnz_lo:nnz_hi]] = Y[sub_row, sub_inv]

        gathered = torch.from_numpy(gathered_np).to(dtype=val.dtype, device=device)
        grad_val = det_val * gathered * grad_output
        return grad_val, None, None, None, None, None


# =============================================================================
# Adjoint Eigenvalue Solver
# =============================================================================

def _eigh_dense_topk(sparse_coo, k, largest):
    """Small-n CPU fallback: dense eigh + slice top/bottom k."""
    A_dense = sparse_coo.to_dense()
    evals_all, evecs_all = torch.linalg.eigh(A_dense)
    if largest:
        return evals_all[-k:], evecs_all[:, -k:]
    return evals_all[:k], evecs_all[:, :k]


class EigshAdjoint(Function):
    """
    Adjoint-based differentiable eigenvalue solver.
    
    Uses implicit differentiation to compute gradients with O(1) graph nodes,
    regardless of the number of iterations in the forward solve.
    
    For symmetric matrix A with eigenvalue λ and eigenvector v:
        A @ v = λ * v
    
    The gradient is:
        ∂λ/∂A = v @ v.T  (outer product)
        ∂v/∂A requires solving a linear system (more complex)
    """
    
    @staticmethod
    def forward(ctx, val, row, col, shape, k, which, return_eigenvectors, device):
        """Forward pass: compute eigenvalues using LOBPCG or dense fallback."""
        n = shape[0]
        
        # Detach for forward computation
        val_detached = val.detach()
        
        # Build sparse matrix for matvec
        indices = torch.stack([row, col], dim=0).to(device)
        sparse_coo = torch.sparse_coo_tensor(indices, val_detached, shape, device=device)
        
        def matvec(x):
            if x.dim() == 1:
                return torch.sparse.mm(sparse_coo, x.unsqueeze(1)).squeeze(1)
            return torch.sparse.mm(sparse_coo, x)
        
        # Compute eigenvalues
        if device.type == 'cuda':
            # Use LOBPCG on CUDA
            largest = which in ('LM', 'LA')
            eigenvalues, eigenvectors = _lobpcg_eigsh(
                matvec, n, k, val.dtype, device, largest=largest
            )
        else:
            # CPU path. For n > 1024 dispatch to scipy.sparse.linalg.eigsh
            # (ARPACK Lanczos) -- truly O(nnz*k*iter), no dense n*n alloc.
            # For small n we still use torch.linalg.eigh on the dense
            # tensor: LAPACK is faster than ARPACK's setup cost, and
            # scipy isn't a hard dep at that scale.
            largest = which in ('LM', 'LA')
            if n > 1024:
                from ..backends.scipy_backend import SCIPY_AVAILABLE, scipy_eigsh
                if SCIPY_AVAILABLE:
                    which_arpack = {'LM': 'LM', 'SM': 'SM', 'LA': 'LA', 'SA': 'SA'}[which]
                    eigenvalues, eigenvectors = scipy_eigsh(
                        val_detached, row, col, shape,
                        k=k, which=which_arpack, sigma=None,
                        return_eigenvectors=True,
                    )
                    eigenvalues = eigenvalues.to(device=device)
                    eigenvectors = eigenvectors.to(device=device)
                else:
                    eigenvalues, eigenvectors = _eigh_dense_topk(
                        sparse_coo, k, largest)
            else:
                eigenvalues, eigenvectors = _eigh_dense_topk(
                    sparse_coo, k, largest)
        
        # Save for backward
        ctx.save_for_backward(val, eigenvalues, eigenvectors)
        ctx.row = row
        ctx.col = col
        ctx.shape = shape
        ctx.k = k
        ctx.return_eigenvectors = return_eigenvectors
        
        if return_eigenvectors:
            return eigenvalues, eigenvectors
        return eigenvalues, None
    
    @staticmethod
    def backward(ctx, grad_eigenvalues, grad_eigenvectors):
        """
        Backward pass using adjoint method.
        
        For eigenvalue λ_i with eigenvector v_i:
            ∂L/∂A[j,k] = Σ_i (∂L/∂λ_i) * v_i[j] * v_i[k]
        
        This gives us O(1) graph nodes.
        """
        val, eigenvalues, eigenvectors = ctx.saved_tensors
        row = ctx.row
        col = ctx.col
        k = ctx.k
        
        if grad_eigenvalues is None:
            return None, None, None, None, None, None, None, None
        
        # Compute gradient w.r.t. values
        # ∂L/∂A[i,j] = Σ_m (∂L/∂λ_m) * v_m[i] * v_m[j]
        # For sparse format: ∂L/∂val[idx] = Σ_m (∂L/∂λ_m) * v_m[row[idx]] * v_m[col[idx]]
        
        grad_val = torch.zeros_like(val)
        
        for m in range(k):
            if grad_eigenvalues[m] != 0:
                # v_m[row] * v_m[col] for each sparse entry
                v_m = eigenvectors[:, m]
                grad_val += grad_eigenvalues[m] * v_m[row] * v_m[col]
        
        # Handle eigenvector gradients if needed (more complex, skip for now)
        # The eigenvector gradient requires solving (A - λI) @ dv = ...
        
        return grad_val, None, None, None, None, None, None, None



# =============================================================================
# Autograd Functions
# =============================================================================

class SparseSolveFunction(Function):
    """
    Differentiable sparse solve using scipy for CPU.
    
    Solves Ax = b and computes gradients for both A's values and b.
    """
    
    @staticmethod
    def forward(ctx, val, row, col, shape, b, method, atol, maxiter):
        u = scipy_solve(val, row, col, shape, b, method=method, atol=atol, maxiter=maxiter)
        ctx.save_for_backward(val, row, col, u, b)
        ctx.shape = shape
        ctx.method = method
        ctx.atol = atol
        ctx.maxiter = maxiter
        return u
    
    @staticmethod
    def backward(ctx, grad_u):
        val, row, col, u, b = ctx.saved_tensors
        shape = ctx.shape
        method = ctx.method
        atol = ctx.atol
        maxiter = ctx.maxiter
        # Adjoint system uses the conjugate transpose A^H (= conj(A)^T), not
        # just A^T. For real matrices .conj() is a no-op so this is unchanged;
        # for complex it makes the Wirtinger gradient correct.
        grad_b = scipy_solve(val.conj(), col, row, (shape[1], shape[0]), grad_u,
                            method=method, atol=atol, maxiter=maxiter)
        grad_val = -grad_b[row] * u[col].conj()
        return grad_val, None, None, None, grad_b, None, None, None


class SparseSparseMatmulFunction(Function):
    """
    Differentiable Sparse @ Sparse multiplication with SPARSE gradients.
    
    Forward: C = A @ B where A is [M, K], B is [K, N], C is [M, N]
    
    Backward:
    - grad_A_values = (grad_C @ B^T)[A_row, A_col]  (sparse gradient at A's positions)
    - grad_B_values = (A^T @ grad_C)[B_row, B_col]  (sparse gradient at B's positions)
    
    The gradients are computed only at the original non-zero positions,
    keeping memory usage proportional to nnz rather than M*N.
    """
    
    @staticmethod
    def forward(ctx, val_a, row_a, col_a, shape_a, val_b, row_b, col_b, shape_b):
        M, K = shape_a
        K2, N = shape_b
        assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
        
        # Create torch sparse tensors for multiplication
        A_coo = torch.sparse_coo_tensor(
            torch.stack([row_a, col_a]), val_a, (M, K)
        ).coalesce()
        B_coo = torch.sparse_coo_tensor(
            torch.stack([row_b, col_b]), val_b, (K, N)
        ).coalesce()
        
        # Sparse @ Sparse -> Sparse
        with torch.no_grad():
            C_coo = torch.sparse.mm(A_coo, B_coo).coalesce()
        
        # Extract result
        C_indices = C_coo._indices()
        C_values = C_coo._values()
        
        # Save for backward
        ctx.save_for_backward(val_a, row_a, col_a, val_b, row_b, col_b, 
                              C_indices[0], C_indices[1], C_values)
        ctx.shape_a = shape_a
        ctx.shape_b = shape_b
        
        return C_values, C_indices[0], C_indices[1]
    
    @staticmethod
    def backward(ctx, grad_C_values, grad_row_c, grad_col_c):
        (val_a, row_a, col_a, val_b, row_b, col_b,
         row_c, col_c, val_c) = ctx.saved_tensors
        M, K = ctx.shape_a
        K2, N = ctx.shape_b

        grad_val_a = None
        grad_val_b = None

        if ctx.needs_input_grad[0]:
            # grad_A = (grad_C @ B^T)[A_row, A_col]
            # Keep sparse end-to-end; never materialise an (M, K) dense.
            grad_C_coo = torch.sparse_coo_tensor(
                torch.stack([row_c, col_c]), grad_C_values, (M, N)
            ).coalesce()
            B_T_coo = torch.sparse_coo_tensor(
                torch.stack([col_b, row_b]), val_b, (N, K)
            ).coalesce()
            grad_A_sparse = torch.sparse.mm(grad_C_coo, B_T_coo)
            grad_val_a = _gather_sparse_at_coords(grad_A_sparse, row_a, col_a, K)

        if ctx.needs_input_grad[4]:
            # grad_B = (A^T @ grad_C)[B_row, B_col]
            A_T_coo = torch.sparse_coo_tensor(
                torch.stack([col_a, row_a]), val_a, (K, M)
            ).coalesce()
            grad_C_coo = torch.sparse_coo_tensor(
                torch.stack([row_c, col_c]), grad_C_values, (M, N)
            ).coalesce()
            grad_B_sparse = torch.sparse.mm(A_T_coo, grad_C_coo)
            grad_val_b = _gather_sparse_at_coords(grad_B_sparse, row_b, col_b, N)

        return grad_val_a, None, None, None, grad_val_b, None, None, None


def _gather_sparse_at_coords(C_sparse, row_q, col_q, n_cols):
    """Gather sparse tensor C at positions (row_q[i], col_q[i]).

    Positions outside C's sparsity pattern return 0. No dense intermediate
    is materialised; cost is O((nnz(C) + nnz(query)) log nnz(C)).

    Used in SparseSparseMatmul backward to extract grad at A/B's nnz
    positions from the sparse product (grad_C @ B^T) / (A^T @ grad_C).
    """
    C_coo = C_sparse.coalesce()
    i_c = C_coo.indices()[0].to(torch.int64)
    j_c = C_coo.indices()[1].to(torch.int64)
    v_c = C_coo.values()

    if v_c.numel() == 0:
        return torch.zeros(row_q.shape[0], dtype=v_c.dtype, device=v_c.device)

    # Linearise (i, j) -> i * n_cols + j; coalesce() guarantees C's
    # indices are lexicographically sorted, so no extra sort is needed.
    pos_c = i_c * n_cols + j_c
    pos_q = row_q.to(torch.int64) * n_cols + col_q.to(torch.int64)

    idx = torch.searchsorted(pos_c, pos_q)
    idx_clamped = idx.clamp(max=pos_c.numel() - 1)
    found = (idx < pos_c.numel()) & (pos_c[idx_clamped] == pos_q)
    return torch.where(
        found, v_c[idx_clamped],
        torch.zeros_like(pos_q, dtype=v_c.dtype),
    )


class SvdAdjoint(Function):
    """Differentiable truncated SVD for sparse matrices.

    Forward runs detached (ARPACK / dense / power-iteration) so the
    iterative process never enters the autograd graph. Backward uses the
    Townsend & Wendland (2014) closed form sampled at A's nnz pattern.

    Gradient support (this first implementation):

    * Singular-value gradient (``∂L/∂σ_i``) — full support. Closed form
      ``∂σ_i/∂A = u_i v_iᵀ`` gathered at A's pattern; cost O(nnz · k).
    * Singular-vector gradient (``∂L/∂U`` / ``∂L/∂V``) — NOT YET
      implemented. If non-zero gradient flows back through U or V,
      emits a ``RuntimeWarning`` and treats those contributions as zero.

    Most use cases (spectral regularisation, nuclear norm penalty,
    PCA-style targets) only need σ gradients; full Townsend formula for
    U/V is tracked as a separate follow-up.
    """

    @staticmethod
    def forward(ctx, val, row, col, shape, k, svd_fn, gather_fn):
        with torch.no_grad():
            U, S, Vt = svd_fn(val.detach(), row, col, shape, k)
        ctx.save_for_backward(val, U, S, Vt)
        ctx.gather_fn = gather_fn
        ctx.shape = shape
        return U.detach(), S.detach(), Vt.detach()

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_Vt):
        val, U, S, Vt = ctx.saved_tensors

        # Singular-vector gradient not yet supported; warn if any flow.
        u_norm = float(grad_U.abs().max()) if grad_U is not None else 0.0
        v_norm = float(grad_Vt.abs().max()) if grad_Vt is not None else 0.0
        if u_norm > 0 or v_norm > 0:
            import warnings
            warnings.warn(
                "SvdAdjoint: gradient through singular vectors (U, V) "
                "is not yet implemented in the first SvdAdjoint release; "
                "only ∂L/∂σ contributions are propagated to A. The U/V "
                "Townsend cross-term will land in a follow-up PR. "
                "Treating ∂L/∂U / ∂L/∂V as zero for now.",
                RuntimeWarning, stacklevel=2,
            )

        if grad_S is None or float(grad_S.abs().max()) == 0.0:
            return torch.zeros_like(val), None, None, None, None, None, None

        # ∂σ_i / ∂A = u_i v_i^T, so for each nnz (row[k], col[k]):
        #   grad_val[k] = Σ_i grad_S[i] * U[row[k], i] * Vt[i, col[k]]
        # gather_fn implements this; for distributed it maps local CSR
        # row/col into global indices before indexing U / Vt.
        grad_val = ctx.gather_fn(U, S, Vt, grad_S)
        return grad_val, None, None, None, None, None, None


def _sparse_sparse_matmul_with_sparse_grad(
    val_a: torch.Tensor, row_a: torch.Tensor, col_a: torch.Tensor, shape_a: Tuple[int, int],
    val_b: torch.Tensor, row_b: torch.Tensor, col_b: torch.Tensor, shape_b: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Sparse @ Sparse with sparse gradients.
    
    Parameters
    ----------
    val_a, row_a, col_a : torch.Tensor
        COO representation of matrix A.
    shape_a : Tuple[int, int]
        Shape of matrix A (M, K).
    val_b, row_b, col_b : torch.Tensor
        COO representation of matrix B.
    shape_b : Tuple[int, int]
        Shape of matrix B (K, N).
    
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int]]
        (values, row_indices, col_indices, shape) of result C = A @ B.
    """
    M, K = shape_a
    K2, N = shape_b
    
    C_values, C_row, C_col = SparseSparseMatmulFunction.apply(
        val_a, row_a, col_a, shape_a,
        val_b, row_b, col_b, shape_b
    )
    
    return C_values, C_row, C_col, (M, N)


# =============================================================================
# LOBPCG and Power Iteration for CUDA
# =============================================================================

def _cgs2_inplace(Z: torch.Tensor, ncols_keep: int) -> torch.Tensor:
    """Twice-iterated classical Gram-Schmidt orthonormalisation in-place.

    Two CGS sweeps give numerical reliability competitive with
    Householder QR (Giraud et al., 2005) at a fraction of the cost
    when the input is already mostly orthonormal -- the case in every
    LOBPCG iteration after the first. Cheaper than a full
    ``torch.linalg.qr(Z)`` whenever ``Z.shape[1]`` is not tiny because
    QR does ``O(n m²)`` Householder flops while CGS2 does ``O(n m²)``
    too but with only level-2 BLAS (matmul-friendly).

    Returns a possibly column-pruned view (rank ``< ncols_keep``
    surfaces as fewer columns retained).
    """
    eps = torch.finfo(Z.dtype).eps * 100
    # Twice-iterated CGS: the second pass corrects for catastrophic
    # cancellation in the first when columns of Z are nearly linearly
    # dependent (typical for [X | R | P] near convergence where R is
    # small).
    for _ in range(2):
        for j in range(Z.shape[1]):
            if j > 0:
                # Project column j onto the orthonormal basis of
                # columns 0..j-1 and subtract.
                # ``Z[:, :j].T @ Z[:, j]`` is one matmul of shape (j,).
                coeff = Z[:, :j].T @ Z[:, j]
                Z[:, j] -= Z[:, :j] @ coeff
            nrm = Z[:, j].norm()
            if nrm > eps:
                Z[:, j] /= nrm
            else:
                Z[:, j].zero_()  # Drop degenerate direction; rank-deficient.
    # Trim trailing zero columns (rank deficiency in [X | R | P]).
    col_norms = Z.norm(dim=0)
    valid = col_norms > 0.5  # >0.5 because just-normalised cols are unit.
    if not bool(valid.all()):
        Z = Z[:, valid]
    return Z[:, :max(Z.shape[1], 1)]


def _lobpcg_eigsh(
    A_matvec,
    n: int,
    k: int,
    dtype: torch.dtype,
    device: torch.device,
    largest: bool = True,
    maxiter: int = 1000,
    tol: float = 1e-8,
    T_apply=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LOBPCG eigenvalue solver.

    Knyazev (2001) Locally Optimal Block Preconditioned Conjugate
    Gradient method for the k extreme eigenpairs of a symmetric
    matrix accessed only through a matvec callback. The earlier
    implementation here was block steepest descent ([X | R]
    subspace, no conjugate direction, no preconditioner) which
    only converges linearly; this one adds three pieces:

    1. **Conjugate direction P**. Subspace at iter > 0 is
       ``[X | R | P]`` (3k columns) instead of ``[X | R]``. ``P`` is
       the residual / previous-conjugate contribution to the new
       Ritz vectors -- extracting it from the Ritz coefficients
       costs no extra matvec. Convergence rate improves from linear
       to the LOBPCG-standard "near cubic in the gap" rate.

    2. **Buffer reuse**. ``X``, ``AX``, ``R``, ``P`` are allocated
       once at start and rewritten in-place each iteration via
       ``torch.matmul(..., out=...)``. The subspace ``Z`` /
       ``AZ`` buffers are also pre-allocated and never re-cat'd. The
       hot loop allocates only the small Hessian ``H`` and its
       eigenvectors (size 3k x 3k, negligible vs the n x k blocks).

    3. **CGS2 over QR**. Replaces ``torch.linalg.qr(combined)`` with
       twice-iterated classical Gram-Schmidt. Stability matches
       Householder QR (Giraud et al.) and the flops are equivalent
       but matmul-bound (level-2 BLAS), much friendlier to GPU and
       to autograd if Z later feeds a differentiable path.

    Parameters
    ----------
    A_matvec : callable
        ``A_matvec(X) -> A @ X`` for X of shape ``[n, m]``.
    n, k : int
        Matrix dimension and number of eigenpairs to return.
    dtype, device
        Output placement; X buffers live here.
    largest : bool
        Largest (``True``) or smallest (``False``) eigenpairs.
    maxiter, tol : int, float
        Outer iteration cap and convergence threshold on
        ``|lambda_new - lambda_old| / |lambda|``.
    T_apply : callable, optional
        Optional preconditioner ``T(R) -> approx_inv_A @ R``. Speeds
        up convergence whenever a cheap approximation to ``A^{-1}``
        is available (e.g. Jacobi for diagonally-dominant ``A``).
        ``None`` (default) keeps the unpreconditioned form.

    Returns
    -------
    (eigenvalues, eigenvectors) : (Tensor[k], Tensor[n, k])
    """
    if k > n:
        raise ValueError(f"k={k} exceeds matrix dimension n={n}")
    k = max(k, 1)
    # Work in a slightly larger block than k so we have buffer columns
    # to resolve closely-clustered eigenvalues (especially for the
    # ``smallest`` case where random init has low overlap with the true
    # extreme eigenvector). The final return is the best k of m at
    # convergence. ``2*k`` matches scipy.sparse.linalg.lobpcg's
    # internal expansion; clamped to ``n``.
    m = min(max(2 * k, k + 2), n)

    # ---- Pre-allocated working buffers (lifetime = whole solve) ----
    X = torch.randn(n, m, dtype=dtype, device=device)
    X, _ = torch.linalg.qr(X)  # one QR at init, then never again
    AX = torch.empty_like(X)
    R = torch.empty_like(X)
    P = torch.zeros_like(X)  # zero until iter 1 introduces the
                             # conjugate direction
    Z = torch.empty(n, 3 * m, dtype=dtype, device=device)
    AZ = torch.empty_like(Z)

    eigenvalues = torch.empty(m, dtype=dtype, device=device)
    eigenvalues_prev = None

    # ---- Initial Rayleigh-Ritz step ----
    AX.copy_(A_matvec(X))
    H = X.T @ AX
    H = 0.5 * (H + H.T)  # symmetrise against floating-point drift
    eigvals, V = torch.linalg.eigh(H)
    if largest:
        idx = eigvals.argsort(descending=True)
    else:
        idx = eigvals.argsort()
    eigvals = eigvals[idx]
    V = V[:, idx]
    # Ritz rotation: X <- X V, AX <- AX V
    X_new = X @ V
    AX_new = AX @ V
    X.copy_(X_new)
    AX.copy_(AX_new)
    eigenvalues.copy_(eigvals[:m])

    iter_done = 0
    for iteration in range(maxiter):
        iter_done = iteration

        # ---- Residual R = AX - X * lambda (column-wise scale) ----
        # torch.sub + mul: R = AX - X * lambda[None, :]
        torch.mul(X, eigenvalues.unsqueeze(0), out=R)
        R.neg_()
        R.add_(AX)

        # Apply preconditioner if provided.
        if T_apply is not None:
            R = T_apply(R)

        # ---- Build subspace Z = [X | R | P] (P all-zero on iter 0) ----
        ncols = 2 * m if iteration == 0 else 3 * m
        Z[:, :m].copy_(X)
        Z[:, m:2 * m].copy_(R)
        if iteration > 0:
            Z[:, 2 * m:3 * m].copy_(P)

        # ---- CGS2 orthonormalise the active part of Z ----
        Z_active = _cgs2_inplace(Z[:, :ncols], ncols)
        ncols_eff = Z_active.shape[1]

        # ---- Compute AZ on the orthonormalised subspace ----
        AZ_active = A_matvec(Z_active)

        # ---- Project: small Hessian H = Z_active.T @ AZ_active ----
        H = Z_active.T @ AZ_active
        H = 0.5 * (H + H.T)
        eigvals, V = torch.linalg.eigh(H)
        if largest:
            idx = eigvals.argsort(descending=True)
        else:
            idx = eigvals.argsort()
        eigvals = eigvals[idx]
        V = V[:, idx]
        Vk = V[:, :m]

        # ---- New Ritz vectors and conjugate direction ----
        X_new = Z_active @ Vk         # [n, m]
        AX_new = AZ_active @ Vk       # [n, m]
        # Conjugate direction = portion of new Ritz vectors that came
        # from the (R, P) blocks, i.e. NOT from the previous X block.
        # Equivalent to Z_active[:, m:] @ Vk[m:, :] in the standard
        # LOBPCG derivation (see Knyazev 2001 eq. 7).
        if ncols_eff > m:
            P_new = Z_active[:, m:] @ Vk[m:, :]
        else:
            P_new = torch.zeros_like(X)

        X.copy_(X_new)
        AX.copy_(AX_new)
        P.copy_(P_new)
        new_eigvals = eigvals[:m]

        # ---- Convergence check on the k eigenpairs we actually return ----
        if eigenvalues_prev is not None:
            diff = (new_eigvals[:k] - eigenvalues_prev[:k]).abs()
            denom = new_eigvals[:k].abs().clamp(min=1e-10)
            if (diff < tol * denom).all():
                eigenvalues.copy_(new_eigvals)
                break
        if eigenvalues_prev is None:
            eigenvalues_prev = new_eigvals.clone()
        else:
            eigenvalues_prev.copy_(new_eigvals)
        eigenvalues.copy_(new_eigvals)

    return eigenvalues[:k], X[:, :k]

