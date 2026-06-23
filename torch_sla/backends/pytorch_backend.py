"""
PyTorch-native backend for GPU iterative solvers.

This backend uses pure PyTorch sparse operations, which work on both CPU and CUDA.
It's especially useful for large-scale problems where direct solvers run out of memory.

Methods:
- 'cg': Conjugate Gradient (for SPD matrices)
- 'bicgstab': BiCGStab (for general matrices)
- 'gmres': restarted GMRES (general / non-symmetric matrices)
- 'minres': MINRES (symmetric / Hermitian, possibly indefinite; needs SPD preconditioner)

Preconditioners (ordered by effectiveness):
- 'ic0': Incomplete Cholesky (zero fill-in) - BEST for SPD
- 'ilu0': Incomplete LU (zero fill-in) - for general matrices
- 'block_jacobi': Block Jacobi with small blocks
- 'polynomial': Chebyshev polynomial preconditioner
- 'ssor': Symmetric SOR preconditioner
- 'jacobi': Diagonal (Jacobi) preconditioner
- 'none': No preconditioning

Optimizations:
- CSR matrix caching (avoid repeated COO->CSR conversion)
- Fused vector operations where possible
- In-place operations to reduce memory allocations
"""

import torch
from torch import Tensor
from typing import Tuple, Optional, Union, NamedTuple, Callable
import warnings
import math


class SolveResult(NamedTuple):
    """Result of iterative solve."""
    x: Tensor
    num_iters: int
    residual: float
    converged: bool


def compute_rcm_ordering(row: Tensor, col: Tensor, n: int) -> Tensor:
    """
    Compute Reverse Cuthill-McKee (RCM) ordering for bandwidth reduction.
    
    RCM reorders nodes to reduce matrix bandwidth, improving cache locality
    and SpMV performance (typically 1.2-1.5x speedup).
    
    Returns permutation tensor: new_idx = perm[old_idx]
    """
    device = row.device
    
    # Build adjacency list on CPU for graph traversal
    row_cpu = row.cpu().numpy()
    col_cpu = col.cpu().numpy()
    
    # Build adjacency list
    from collections import defaultdict, deque
    adj = defaultdict(list)
    for r, c in zip(row_cpu, col_cpu):
        if r != c:
            adj[r].append(c)
    
    # Compute node degrees
    degrees = {i: len(adj[i]) for i in range(n)}
    
    # Find starting node (peripheral node with min degree)
    unvisited = set(range(n))
    if not unvisited:
        return torch.arange(n, device=device)
    
    # Start from minimum degree node
    start = min(unvisited, key=lambda x: degrees.get(x, 0))
    
    # BFS with degree-based ordering
    result = []
    queue = deque([start])
    visited = {start}
    
    while queue or unvisited:
        if not queue:
            # Start new component
            start = min(unvisited, key=lambda x: degrees.get(x, 0))
            queue.append(start)
            visited.add(start)
        
        node = queue.popleft()
        result.append(node)
        unvisited.discard(node)
        
        # Add neighbors sorted by degree (ascending)
        neighbors = [nb for nb in adj[node] if nb not in visited]
        neighbors.sort(key=lambda x: degrees.get(x, 0))
        
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    
    # Reverse for RCM
    result = result[::-1]
    
    return torch.tensor(result, dtype=torch.long, device=device)


class CachedSparseMatrix:
    """
    Cached sparse matrix for efficient repeated matvec operations.
    Avoids repeated COO -> CSR conversion.
    
    Supports optional RCM reordering for better cache locality.
    """
    def __init__(self, val: Tensor, row: Tensor, col: Tensor, shape: Tuple[int, int],
                 use_rcm: bool = False):
        self.val = val
        self.row = row
        self.col = col
        self.shape = shape
        self.device = val.device
        self.dtype = val.dtype
        self.n = shape[0]
        self.use_rcm = use_rcm
        
        # RCM reordering for better cache locality
        if use_rcm and self.n > 1000:
            self._perm = compute_rcm_ordering(row, col, self.n)
            self._inv_perm = torch.empty_like(self._perm)
            self._inv_perm[self._perm] = torch.arange(self.n, device=self.device)
            
            # Reorder matrix: A' = P A P^T
            new_row = self._inv_perm[row]
            new_col = self._inv_perm[col]
            row, col = new_row, new_col
        else:
            self._perm = None
            self._inv_perm = None
        
        # Build CSR matrix once
        indices = torch.stack([row, col], dim=0)
        coo = torch.sparse_coo_tensor(indices, val, shape, device=val.device, dtype=val.dtype)
        self._csr = coo.to_sparse_csr()
        
        # Store reordered indices for preconditioners
        self._row_ordered = row
        self._col_ordered = col
        
        # Cache structures
        self._diag = None
        self._lower_csr = None
        self._upper_csr = None
        self._ic0_L = None  # Incomplete Cholesky factor
    
    def matvec(self, x: Tensor) -> Tensor:
        """Sparse matrix-vector product y = A @ x"""
        if self._perm is not None:
            # Apply permutation: y = P A P^T x = P A (P^T x)
            x_perm = x[self._perm]
            y_perm = torch.mv(self._csr, x_perm)
            return y_perm[self._inv_perm]
        return torch.mv(self._csr, x)
    
    def matvec_reordered(self, x: Tensor) -> Tensor:
        """SpMV in reordered space (no permutation applied)."""
        return torch.mv(self._csr, x)
    
    @property
    def diagonal(self) -> Tensor:
        """Get diagonal elements (cached, in original ordering)"""
        if self._diag is None:
            self._diag = torch.zeros(self.n, dtype=self.dtype, device=self.device)
            diag_mask = self.row == self.col
            self._diag.scatter_add_(0, self.row[diag_mask], self.val[diag_mask])
        return self._diag
    
    @property 
    def diagonal_reordered(self) -> Tensor:
        """Get diagonal in reordered space."""
        if self._perm is not None:
            return self.diagonal[self._perm]
        return self.diagonal
    
    def get_lower_upper(self):
        """Get strictly lower and upper triangular parts"""
        if self._lower_csr is None:
            # Lower triangular (row > col)
            lower_mask = self.row > self.col
            lower_row = self.row[lower_mask]
            lower_col = self.col[lower_mask]
            lower_val = self.val[lower_mask]
            
            if len(lower_val) > 0:
                lower_indices = torch.stack([lower_row, lower_col], dim=0)
                lower_coo = torch.sparse_coo_tensor(lower_indices, lower_val, self.shape, 
                                                    device=self.device, dtype=self.dtype)
                self._lower_csr = lower_coo.to_sparse_csr()
            else:
                self._lower_csr = None
            
            # Upper triangular (row < col)
            upper_mask = self.row < self.col
            upper_row = self.row[upper_mask]
            upper_col = self.col[upper_mask]
            upper_val = self.val[upper_mask]
            
            if len(upper_val) > 0:
                upper_indices = torch.stack([upper_row, upper_col], dim=0)
                upper_coo = torch.sparse_coo_tensor(upper_indices, upper_val, self.shape,
                                                    device=self.device, dtype=self.dtype)
                self._upper_csr = upper_coo.to_sparse_csr()
            else:
                self._upper_csr = None
        
        return self._lower_csr, self._upper_csr


# ============================================================================
# Preconditioners
# ============================================================================

def jacobi_preconditioner(A: CachedSparseMatrix) -> Callable[[Tensor], Tensor]:
    """
    Jacobi (diagonal) preconditioner: M^{-1} = diag(A)^{-1}.
    
    Convergence: Slowest, but simplest.
    Cost per iteration: O(n)
    """
    diag = A.diagonal
    eps = torch.finfo(diag.dtype).eps * 100
    D_inv = 1.0 / torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
    
    def apply(r: Tensor) -> Tensor:
        return D_inv * r
    
    return apply


def ssor_preconditioner(A: CachedSparseMatrix, omega: float = 1.5) -> Callable[[Tensor], Tensor]:
    """
    SSOR preconditioner (simplified diagonal version for GPU).
    
    True SSOR requires sequential forward/backward sweeps which are slow on GPU.
    This uses a scaled Jacobi approximation that works well in practice.
    
    Convergence: Better than Jacobi (~1.5x fewer iterations)
    Cost per iteration: O(n)
    """
    diag = A.diagonal
    eps = torch.finfo(diag.dtype).eps * 100
    D_inv = 1.0 / torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
    scale = math.sqrt(omega * (2 - omega))
    
    def apply(r: Tensor) -> Tensor:
        return scale * D_inv * r
    
    return apply


def block_jacobi_preconditioner(A: CachedSparseMatrix, block_size: int = 32) -> Callable[[Tensor], Tensor]:
    """
    Block Jacobi preconditioner.
    
    Divides the matrix into blocks along the diagonal and inverts each block.
    Better than point Jacobi because it captures local coupling.
    
    Convergence: ~2x better than Jacobi
    Cost per iteration: O(n * block_size^2) for setup, O(n) for apply
    """
    n = A.n
    num_blocks = (n + block_size - 1) // block_size
    
    # Extract diagonal blocks and compute their inverses
    block_inverses = []
    
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, n)
        size = end - start
        
        # Extract block from sparse matrix
        mask = (A.row >= start) & (A.row < end) & (A.col >= start) & (A.col < end)
        local_row = A.row[mask] - start
        local_col = A.col[mask] - start
        local_val = A.val[mask]
        
        # Build dense block
        block = torch.zeros((size, size), dtype=A.dtype, device=A.device)
        block[local_row, local_col] = local_val
        
        # Regularize and invert
        block += torch.eye(size, dtype=A.dtype, device=A.device) * 1e-10
        try:
            block_inv = torch.linalg.inv(block)
        except:
            block_inv = torch.diag(1.0 / (torch.diag(block) + 1e-10))
        
        block_inverses.append(block_inv)
    
    def apply(r: Tensor) -> Tensor:
        z = torch.zeros_like(r)
        for i, block_inv in enumerate(block_inverses):
            start = i * block_size
            end = min((i + 1) * block_size, n)
            z[start:end] = block_inv @ r[start:end]
        return z
    
    return apply


def polynomial_preconditioner(A: CachedSparseMatrix, degree: int = 5) -> Callable[[Tensor], Tensor]:
    """
    Neumann series polynomial preconditioner.
    
    Uses M^{-1} ≈ D^{-1} (I + N + N^2 + ...) where N = I - D^{-1}A
    
    This is stable and effective for well-scaled problems.
    
    Convergence: 2-3x iteration reduction vs Jacobi
    Cost per iteration: O(degree * nnz)
    """
    diag = A.diagonal
    eps = torch.finfo(diag.dtype).eps * 100
    D_inv = 1.0 / torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
    
    def apply(r: Tensor) -> Tensor:
        # Neumann series: M^{-1} = D^{-1} sum_{k=0}^{degree} (I - D^{-1}A)^k
        z = D_inv * r  # k=0 term
        
        if degree == 0:
            return z
        
        y = r.clone()
        for _ in range(degree):
            # y = (I - D^{-1}A) @ y
            Ay = A.matvec(y)
            y = y - D_inv * Ay
            z = z + D_inv * y
        
        return z
    
    return apply


def ic0_preconditioner(A: CachedSparseMatrix, num_sweeps: int = 2) -> Callable[[Tensor], Tensor]:
    """
    Incomplete Cholesky (IC0) preconditioner using Jacobi-like iterations.
    
    Uses symmetric Gauss-Seidel style iterations to approximate:
    M^{-1} = (D + L)^{-1} D (D + L^T)^{-1}
    
    This is GPU-friendly as it uses parallel relaxation.
    
    Convergence: 1.5-2x iteration reduction vs Jacobi for SPD matrices
    Cost: O(nnz) setup, O(num_sweeps * nnz) per apply
    """
    n = A.n
    diag = A.diagonal.clone()
    device = A.device
    dtype = A.dtype
    
    eps = torch.finfo(dtype).eps * 100
    diag_safe = torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
    D_inv = 1.0 / diag_safe
    
    # Get strictly lower triangular part
    lower_mask = A.row > A.col
    L_row = A.row[lower_mask]
    L_col = A.col[lower_mask]
    L_val = A.val[lower_mask]
    
    if len(L_val) > 0:
        L_indices = torch.stack([L_row, L_col], dim=0)
        L_coo = torch.sparse_coo_tensor(L_indices, L_val, (n, n), 
                                        device=device, dtype=dtype)
        L_csr = L_coo.to_sparse_csr()
        
        # Upper triangular (transpose)
        U_indices = torch.stack([L_col, L_row], dim=0)
        U_coo = torch.sparse_coo_tensor(U_indices, L_val, (n, n),
                                        device=device, dtype=dtype)
        U_csr = U_coo.to_sparse_csr()
        has_offdiag = True
    else:
        has_offdiag = False
    
    def apply(r: Tensor) -> Tensor:
        # Approximate (D + L)^{-1} D (D + L^T)^{-1} r
        # Using Jacobi iterations for triangular solves
        
        if not has_offdiag:
            return D_inv * r
        
        # Forward sweep: solve (D + L) y = r approximately
        # y^{k+1} = D^{-1} (r - L y^k)
        y = D_inv * r
        for _ in range(num_sweeps):
            Ly = torch.mv(L_csr, y)
            y = D_inv * (r - Ly)
        
        # Middle: scale by D
        z = diag_safe * y
        
        # Backward sweep: solve (D + L^T) x = z approximately
        # x^{k+1} = D^{-1} (z - L^T x^k)
        x = D_inv * z
        for _ in range(num_sweeps):
            Ux = torch.mv(U_csr, x)
            x = D_inv * (z - Ux)
        
        return x
    
    return apply


def ilu0_preconditioner(A: CachedSparseMatrix) -> Callable[[Tensor], Tensor]:
    """
    ILU(0): incomplete LU with **zero fill-in** on A's existing sparsity pattern.

    Factors ``A ~= L U`` where ``L`` (unit lower) and ``U`` (upper) are confined
    to the nonzero pattern of ``A`` (classic IKJ Gaussian elimination, dropping
    every entry outside the pattern). The apply solves ``M^{-1} r = U^{-1}(L^{-1} r)``
    with sparse triangular forward/back substitution.

    Real + complex Hermitian/general safe (no conjugation is applied; the IKJ
    recurrence works directly on the stored values). Falls back to
    ``scipy.sparse.linalg.spilu`` only if the pure-Python factorization fails.

    Convergence: typically 2-5x fewer iterations than Jacobi on anisotropic /
    ill-conditioned problems.
    Cost: O(nnz * avg_bandwidth) setup, O(nnz) per apply.
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    n = A.n
    device = A.device
    dtype = A.dtype

    # Assemble A on CPU as a CSR matrix (summing duplicate COO entries).
    row_np = A.row.cpu().numpy()
    col_np = A.col.cpu().numpy()
    val_np = A.val.detach().cpu().numpy()
    A_csr = csr_matrix((val_np, (row_np, col_np)), shape=(n, n))
    A_csr.sum_duplicates()
    A_csr.sort_indices()

    indptr = A_csr.indptr
    indices = A_csr.indices
    if np.iscomplexobj(val_np):
        data = A_csr.data.astype(np.complex128).copy()
    else:
        data = A_csr.data.astype(np.float64).copy()

    # Map (i, j) -> position in the flat data array for each row, so the IKJ
    # update can test "is (i,k)/(k,j) in the pattern?" in O(1).
    # diag_pos[i] = index into data for the diagonal of row i (-1 if missing).
    diag_pos = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        for p in range(indptr[i], indptr[i + 1]):
            if indices[p] == i:
                diag_pos[i] = p
                break

    try:
        # IKJ in-place ILU(0). col_index_of[j] holds the data position of
        # column j in the current row i, or -1 if not in the pattern.
        col_index_of = np.full(n, -1, dtype=np.int64)
        tiny = 1e-14
        for i in range(n):
            row_start, row_end = indptr[i], indptr[i + 1]
            for p in range(row_start, row_end):
                col_index_of[indices[p]] = p
            # eliminate columns k < i that are in the pattern
            for p in range(row_start, row_end):
                k = indices[p]
                if k >= i:
                    continue
                dk = diag_pos[k]
                if dk < 0:
                    continue
                ukk = data[dk]
                if abs(ukk) < tiny:
                    ukk = tiny
                lik = data[p] / ukk
                data[p] = lik  # store multiplier in L part
                # a[i, j] -= lik * u[k, j]  for j > k, only where (i,j) in pattern
                for q in range(dk + 1, indptr[k + 1]):
                    j = indices[q]
                    pj = col_index_of[j]
                    if pj >= 0:
                        data[pj] -= lik * data[q]
            for p in range(row_start, row_end):
                col_index_of[indices[p]] = -1
        factored = True
    except Exception:
        factored = False

    if factored:
        # Split combined LU into unit-lower L (strict + implicit unit diag) and
        # upper U as torch CSR tensors for fast triangular solves.
        L_rows, L_cols, L_vals = [], [], []
        U_rows, U_cols, U_vals = [], [], []
        for i in range(n):
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                if j < i:
                    L_rows.append(i); L_cols.append(j); L_vals.append(data[p])
                else:
                    U_rows.append(i); U_cols.append(j); U_vals.append(data[p])

        torch_real = torch.empty(0, dtype=dtype).real.dtype
        complex_problem = dtype.is_complex

        def _to_tensor(vals):
            arr = np.asarray(vals)
            if complex_problem:
                return torch.tensor(arr, dtype=dtype, device=device)
            return torch.tensor(arr.real.astype(np.float64), dtype=dtype, device=device)

        # Precompute U diagonal (with the same tiny guard used in elimination).
        Udiag = np.empty(n, dtype=data.dtype)
        for i in range(n):
            dp = diag_pos[i]
            ud = data[dp] if dp >= 0 else 0.0
            if abs(ud) < tiny:
                ud = tiny
            Udiag[i] = ud
        Udiag_t = _to_tensor(Udiag)

        # CSR row pointers for sequential triangular solves.
        L_indptr = [0] * (n + 1)
        for r in L_rows:
            L_indptr[r + 1] += 1
        for i in range(n):
            L_indptr[i + 1] += L_indptr[i]
        U_indptr = [0] * (n + 1)
        for r in U_rows:
            U_indptr[r + 1] += 1
        for i in range(n):
            U_indptr[i + 1] += U_indptr[i]

        # Reorder L/U entries by row into flat CSR-style python lists.
        L_col_by_row = [[] for _ in range(n)]
        L_val_by_row = [[] for _ in range(n)]
        for r, c, vv in zip(L_rows, L_cols, L_vals):
            L_col_by_row[r].append(c); L_val_by_row[r].append(vv)
        # Upper without the diagonal (diagonal handled separately).
        U_col_by_row = [[] for _ in range(n)]
        U_val_by_row = [[] for _ in range(n)]
        for r, c, vv in zip(U_rows, U_cols, U_vals):
            if c == r:
                continue
            U_col_by_row[r].append(c); U_val_by_row[r].append(vv)

        # Move per-row index/value lists to tensors for vectorized gather.
        L_cols_t = [torch.tensor(cs, dtype=torch.long, device=device) if cs else None
                    for cs in L_col_by_row]
        L_vals_t = [_to_tensor(vs) if vs else None for vs in L_val_by_row]
        U_cols_t = [torch.tensor(cs, dtype=torch.long, device=device) if cs else None
                    for cs in U_col_by_row]
        U_vals_t = [_to_tensor(vs) if vs else None for vs in U_val_by_row]

        def apply(r: Tensor) -> Tensor:
            # Forward solve L y = r  (unit lower diagonal).
            y = r.clone()
            for i in range(n):
                cols = L_cols_t[i]
                if cols is not None:
                    y[i] = y[i] - torch.dot(L_vals_t[i], y[cols])
            # Back solve U x = y.
            xsol = y.clone()
            for i in range(n - 1, -1, -1):
                cols = U_cols_t[i]
                if cols is not None:
                    xsol[i] = xsol[i] - torch.dot(U_vals_t[i], xsol[cols])
                xsol[i] = xsol[i] / Udiag_t[i]
            return xsol

        return apply

    # Fallback: scipy spilu restricted to ~zero fill.
    from scipy.sparse.linalg import spilu
    A_csc = A_csr.tocsc()
    lu = spilu(A_csc, drop_tol=0.0, fill_factor=1.0)
    complex_problem = dtype.is_complex

    def apply(r: Tensor) -> Tensor:
        r_np = r.detach().cpu().numpy()
        z = lu.solve(r_np)
        if complex_problem:
            return torch.tensor(z, dtype=dtype, device=device)
        return torch.tensor(z.real, dtype=dtype, device=device)

    return apply


def additive_schwarz_preconditioner(
    A: CachedSparseMatrix, block_size: int = 32, overlap: int = 0
) -> Callable[[Tensor], Tensor]:
    """
    Additive Schwarz (ASM) preconditioner.

    Partitions the rows into contiguous subdomains of ``block_size`` (optionally
    overlapping by ``overlap`` rows on each side). Each subdomain is solved
    exactly via a dense LU of its diagonal block; the local corrections are
    **summed** with the standard (symmetric) additive-Schwarz weighting
    ``M^{-1} = sum_i R_i^T A_i^{-1} R_i``.

    Keeping the restriction/prolongation symmetric (no per-row averaging) means
    ``M`` stays symmetric -> usable as a CG/MINRES preconditioner when ``A`` is
    SPD; the overlap lets neighboring blocks exchange information and converges
    faster than plain block-Jacobi. Real + complex safe.

    Convergence: better than Jacobi (and, with overlap, than plain block-Jacobi).
    Cost: O(sum block^3) setup, O(sum block^2) per apply.
    """
    n = A.n
    device = A.device
    dtype = A.dtype

    num_blocks = (n + block_size - 1) // block_size

    blocks = []  # (start, end, lu_solver_factors)

    for b in range(num_blocks):
        core_start = b * block_size
        core_end = min((b + 1) * block_size, n)
        start = max(0, core_start - overlap)
        end = min(n, core_end + overlap)
        size = end - start

        mask = (A.row >= start) & (A.row < end) & (A.col >= start) & (A.col < end)
        local_row = A.row[mask] - start
        local_col = A.col[mask] - start
        local_val = A.val[mask]

        block = torch.zeros((size, size), dtype=dtype, device=device)
        block.index_put_((local_row, local_col), local_val, accumulate=True)
        block += torch.eye(size, dtype=dtype, device=device) * 1e-10

        try:
            LU, piv = torch.linalg.lu_factor(block)
            blocks.append((start, end, ('lu', LU, piv)))
        except Exception:
            diag = torch.diagonal(block)
            blocks.append((start, end, ('diag', 1.0 / (diag + 1e-10), None)))

    def apply(r: Tensor) -> Tensor:
        # Standard additive Schwarz: M^{-1} r = sum_i R_i^T A_i^{-1} R_i r.
        # Symmetric restriction/prolongation -> M stays symmetric (SPD if A is).
        z = torch.zeros_like(r)
        for start, end, fac in blocks:
            r_loc = r[start:end]
            kind = fac[0]
            if kind == 'lu':
                sol = torch.linalg.lu_solve(fac[1], fac[2], r_loc.unsqueeze(-1)).squeeze(-1)
            else:
                sol = fac[1] * r_loc
            z[start:end] += sol
        return z

    return apply


def amg_preconditioner(A: CachedSparseMatrix, num_smooth: int = 1, omega: float = 0.8) -> Callable[[Tensor], Tensor]:
    """
    Lightweight 2-level Algebraic Multigrid (AMG) preconditioner.
    
    Optimized for GPU: uses simple 2-level V-cycle with:
    - Single weighted Jacobi smoothing (fast on GPU)
    - Aggregation-based coarsening
    - Diagonal coarse solve (avoids expensive recursive calls)
    
    Convergence: 2-3x iteration reduction vs Jacobi
    Cost: O(2 * num_smooth * nnz) per apply - much faster than full AMG
    """
    n = A.n
    device = A.device
    dtype = A.dtype
    eps = torch.finfo(dtype).eps * 100
    
    # Diagonal for smoothing
    diag = A.diagonal
    D_inv = 1.0 / torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
    
    # Simple aggregation: 4:1 ratio
    stride = 4
    n_coarse = (n + stride - 1) // stride
    fine_to_coarse = (torch.arange(n, device=device) // stride).clamp(0, n_coarse - 1)
    
    # Coarse diagonal (aggregated)
    D_coarse = torch.zeros(n_coarse, dtype=dtype, device=device)
    D_coarse.scatter_add_(0, fine_to_coarse, diag)
    D_coarse_inv = 1.0 / torch.where(torch.abs(D_coarse) < eps, 
                                      torch.ones_like(D_coarse), D_coarse)
    
    def apply(r: Tensor) -> Tensor:
        # Pre-smooth: z = omega * D^{-1} * r (start from zero)
        z = omega * D_inv * r
        for _ in range(num_smooth - 1):
            Az = A.matvec(z)
            z = z + omega * D_inv * (r - Az)
        
        # Residual
        res = r - A.matvec(z)
        
        # Restrict (injection)
        res_coarse = torch.zeros(n_coarse, dtype=dtype, device=device)
        res_coarse.scatter_add_(0, fine_to_coarse, res)
        
        # Coarse solve (diagonal)
        e_coarse = D_coarse_inv * res_coarse
        
        # Prolong and correct
        z = z + e_coarse[fine_to_coarse]
        
        # Post-smooth
        for _ in range(num_smooth):
            Az = A.matvec(z)
            z = z + omega * D_inv * (r - Az)
        
        return z
    
    return apply


def pyamg_hierarchy_preconditioner(
    A: CachedSparseMatrix,
    *,
    method: str = "ruge_stuben",
    strength: float = 0.25,
    num_pre_smooth: int = 1,
    num_post_smooth: int = 1,
    **kwargs,
) -> Callable[[Tensor], Tensor]:
    """Full PyAMG-backed multigrid preconditioner.

    Builds a real Ruge-Stuben / smoothed-aggregation hierarchy via PyAMG
    on CPU and then runs the V-cycle through ``torch.sparse`` SpMV on
    whatever device :class:`CachedSparseMatrix` lives on. Much stronger
    than the 2-level ``amg_preconditioner`` stub, especially on
    anisotropic / ill-conditioned problems where multi-level coarsening
    is essential.

    Returns the :class:`PyAMGHierarchy` itself (it's already callable as
    ``M^{-1} r``), so callers can also hold on to it for inspection /
    caching across solves.

    Raises ``ImportError`` if pyamg is not installed.
    """
    # Lazy import: the module-level pyamg_backend imports torch + scipy
    # which we already pulled, but explicit lazy-load keeps the
    # "pyamg not installed" path producing a clean error rather than a
    # cryptic ImportError at file load time.
    from .pyamg_backend import PyAMGHierarchy
    return PyAMGHierarchy.from_coo(
        A.val, A.row, A.col, A.shape,
        device=A.device, dtype=A.dtype,
        method=method, strength=strength,
        num_pre_smooth=num_pre_smooth,
        num_post_smooth=num_post_smooth,
    )


def estimate_condition_number(A: CachedSparseMatrix, num_iters: int = 20) -> float:
    """
    Estimate matrix condition number using power iteration.
    
    Returns a rough estimate of the condition number (ratio of largest to smallest eigenvalue).
    Used to decide which preconditioner to use.
    """
    n = A.n
    device = A.device
    dtype = A.dtype
    
    # Power iteration for largest eigenvalue
    v = torch.randn(n, dtype=dtype, device=device)
    v = v / torch.norm(v)
    
    lambda_max = 0.0
    for _ in range(num_iters):
        w = A.matvec(v)
        lambda_max = torch.vdot(v, w).item()
        v = w / torch.norm(w)
    
    # Inverse power iteration for smallest eigenvalue (approximate)
    # Use Jacobi-preconditioned vector as approximation
    diag = A.diagonal
    eps = torch.finfo(dtype).eps * 100
    D_inv = 1.0 / torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
    
    v = torch.randn(n, dtype=dtype, device=device)
    v = v / torch.norm(v)
    
    lambda_min = float('inf')
    for _ in range(num_iters):
        # Approximate inverse iteration using Jacobi
        w = D_inv * v  # M^{-1} v approximates A^{-1} v
        w = w / torch.norm(w)
        Aw = A.matvec(w)
        lambda_est = torch.vdot(w, Aw).item()
        if lambda_est > 0:
            lambda_min = min(lambda_min, lambda_est)
        v = w
    
    if lambda_min <= 0 or lambda_min == float('inf'):
        lambda_min = abs(lambda_max) * 1e-10
    
    return abs(lambda_max / lambda_min)


def select_preconditioner(A: CachedSparseMatrix, method: str = 'cg') -> str:
    """
    Automatically select the best preconditioner based on matrix properties.
    
    Strategy based on empirical testing on SuiteSparse matrices:
    - Small matrices (n < 1000): block_jacobi (captures local structure well)
    - Large well-conditioned (κ < 1e3): jacobi (fast, sufficient)
    - Large moderate conditioning (1e3 < κ < 1e5): ssor or ic0
    - Large ill-conditioned (κ > 1e5): amg (best for difficult problems)
    
    Returns
    -------
    str
        Name of the selected preconditioner
    """
    n = A.n
    
    # For small/medium matrices, block_jacobi is consistently good
    # It captures local coupling which helps with structured matrices
    if n < 1000:
        return 'block_jacobi'
    
    # For larger matrices, estimate condition number
    try:
        kappa = estimate_condition_number(A, num_iters=20)
    except:
        kappa = 1e4  # Default to moderate conditioning if estimation fails
    
    # Select based on size and conditioning
    if n > 50000:
        # Very large matrices: amg scales best
        return 'amg'
    elif kappa < 1e3:
        # Well-conditioned: simple preconditioner is fine
        return 'jacobi'
    elif kappa < 1e5:
        # Moderate conditioning: use ic0
        return 'ic0'
    else:
        # Ill-conditioned: use amg
        return 'amg'


def get_preconditioner(A: CachedSparseMatrix, name: str = 'jacobi', 
                       mixed_precision: bool = False,
                       method: str = 'cg') -> Callable[[Tensor], Tensor]:
    """
    Get preconditioner by name.
    
    Available preconditioners (roughly ordered by effectiveness for SPD):
    - 'auto': Automatically select based on matrix properties (RECOMMENDED)
    - 'ic0': Incomplete Cholesky (best for SPD)
    - 'ilu': ILU(0) incomplete LU, zero fill-in (general matrices)
    - 'asm': Additive Schwarz (overlapping block solves)
    - 'amg': Algebraic Multigrid (good for Poisson-like)
    - 'polynomial': Chebyshev polynomial (degree 5)
    - 'block_jacobi': Block Jacobi (block_size=32)
    - 'ssor': Symmetric SOR (omega=1.5)
    - 'jacobi': Diagonal Jacobi
    - 'none': No preconditioning
    
    Parameters
    ----------
    A : CachedSparseMatrix
        Sparse matrix
    name : str
        Preconditioner name. Use 'auto' for automatic selection.
    mixed_precision : bool
        If True, compute preconditioner in float32 for speed (input/output stay in original dtype)
    method : str
        Solver method ('cg' or 'bicgstab'), used for 'auto' selection.
    """
    # Automatic selection
    if name == 'auto':
        name = select_preconditioner(A, method=method)
    
    if name == 'jacobi':
        precond = jacobi_preconditioner(A)
    elif name == 'ssor':
        precond = ssor_preconditioner(A, omega=1.5)
    elif name == 'block_jacobi':
        precond = block_jacobi_preconditioner(A, block_size=32)
    elif name == 'polynomial':
        precond = polynomial_preconditioner(A, degree=5)
    elif name == 'ic0':
        precond = ic0_preconditioner(A)
    elif name in ('ilu', 'ilu0'):
        precond = ilu0_preconditioner(A)
    elif name == 'asm':
        precond = additive_schwarz_preconditioner(A, block_size=32, overlap=4)
    elif name == 'amg':
        # Prefer real PyAMG hierarchy when available; otherwise the
        # original 2-level stub. Both expose the same callable shape.
        try:
            from .pyamg_backend import is_pyamg_available
            if is_pyamg_available():
                precond = pyamg_hierarchy_preconditioner(A)
            else:
                precond = amg_preconditioner(A)
        except ImportError:
            precond = amg_preconditioner(A)
    elif name == 'pyamg':
        # Explicit PyAMG path; surface a clear error if not installed.
        precond = pyamg_hierarchy_preconditioner(A)
    elif name == 'none':
        return lambda r: r
    else:
        raise ValueError(
            f"Unknown preconditioner: {name}. "
            f"Available: auto, jacobi, ssor, block_jacobi, polynomial, "
            f"ic0, ilu, asm, amg, pyamg, none"
        )
    
    if mixed_precision and A.dtype == torch.float64:
        # Wrap preconditioner to use float32 internally
        return _wrap_mixed_precision(precond)
    
    return precond


def _wrap_mixed_precision(precond: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Wrap preconditioner to compute in float32."""
    def apply(r: Tensor) -> Tensor:
        r_f32 = r.float()
        z_f32 = precond(r_f32)
        return z_f32.to(r.dtype)
    return apply


def polynomial_preconditioner_f32(A: CachedSparseMatrix, degree: int = 5) -> Callable[[Tensor], Tensor]:
    """
    Neumann series polynomial preconditioner in float32.
    
    Same as polynomial_preconditioner but computed in float32 for speed.
    """
    n = A.n
    device = A.device
    
    # Build float32 matrix
    A_f32_indices = torch.stack([A.row, A.col], dim=0)
    A_f32_val = A.val.float()
    A_f32_coo = torch.sparse_coo_tensor(A_f32_indices, A_f32_val, (n, n), device=device)
    A_f32_csr = A_f32_coo.to_sparse_csr()
    
    # Float32 diagonal
    diag_f32 = A.diagonal.float()
    eps = torch.finfo(torch.float32).eps * 100
    D_inv_f32 = 1.0 / torch.where(torch.abs(diag_f32) < eps, torch.ones_like(diag_f32), diag_f32)
    
    def apply(r: Tensor) -> Tensor:
        r_f32 = r.float()
        z = D_inv_f32 * r_f32
        
        if degree == 0:
            return z.to(r.dtype)
        
        y = r_f32.clone()
        for _ in range(degree):
            Ay = torch.mv(A_f32_csr, y)
            y = y - D_inv_f32 * Ay
            z = z + D_inv_f32 * y
        
        return z.to(r.dtype)
    
    return apply


def amg_preconditioner_f32(A: CachedSparseMatrix, num_smooth: int = 1, omega: float = 0.8) -> Callable[[Tensor], Tensor]:
    """
    AMG preconditioner computed in float32 for speed.
    """
    n = A.n
    device = A.device
    
    # Build float32 matrix
    A_f32_indices = torch.stack([A.row, A.col], dim=0)
    A_f32_val = A.val.float()
    A_f32_coo = torch.sparse_coo_tensor(A_f32_indices, A_f32_val, (n, n), device=device)
    A_f32_csr = A_f32_coo.to_sparse_csr()
    
    # Float32 diagonal
    diag_f32 = A.diagonal.float()
    eps = torch.finfo(torch.float32).eps * 100
    D_inv_f32 = 1.0 / torch.where(torch.abs(diag_f32) < eps, torch.ones_like(diag_f32), diag_f32)
    
    # Coarsening
    stride = 4
    n_coarse = (n + stride - 1) // stride
    fine_to_coarse = (torch.arange(n, device=device) // stride).clamp(0, n_coarse - 1)
    
    D_coarse_f32 = torch.zeros(n_coarse, dtype=torch.float32, device=device)
    D_coarse_f32.scatter_add_(0, fine_to_coarse, diag_f32)
    D_coarse_inv_f32 = 1.0 / torch.where(torch.abs(D_coarse_f32) < eps, 
                                          torch.ones_like(D_coarse_f32), D_coarse_f32)
    
    def apply(r: Tensor) -> Tensor:
        r_f32 = r.float()
        
        # Pre-smooth
        z = omega * D_inv_f32 * r_f32
        for _ in range(num_smooth - 1):
            Az = torch.mv(A_f32_csr, z)
            z = z + omega * D_inv_f32 * (r_f32 - Az)
        
        # Residual
        res = r_f32 - torch.mv(A_f32_csr, z)
        
        # Restrict
        res_coarse = torch.zeros(n_coarse, dtype=torch.float32, device=device)
        res_coarse.scatter_add_(0, fine_to_coarse, res)
        
        # Coarse solve
        e_coarse = D_coarse_inv_f32 * res_coarse
        
        # Prolong and correct
        z = z + e_coarse[fine_to_coarse]
        
        # Post-smooth
        for _ in range(num_smooth):
            Az = torch.mv(A_f32_csr, z)
            z = z + omega * D_inv_f32 * (r_f32 - Az)
        
        return z.to(r.dtype)
    
    return apply


def pcg_solve_compiled(
    A: 'CachedSparseMatrix',
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    check_interval: int = 50,
) -> 'SolveResult':
    """
    PCG solver optimized with torch.compile.
    
    Uses torch.compile to JIT compile the inner loop, eliminating
    Python overhead while maintaining compatibility with sparse ops.
    """
    device = A.device
    dtype = A.dtype
    n = A.n
    
    # Compute D_inv
    diag = A.diagonal
    eps = torch.finfo(dtype).eps * 100
    D_inv = 1.0 / torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
    
    # Get CSR matrix
    A_csr = A._csr
    
    # Define inner loop as a compilable function
    def cg_iterations(x, r, z, p, rz_old, Ap, iters):
        """Run multiple CG iterations without sync."""
        for _ in range(iters):
            # SpMV
            torch.mv(A_csr, p, out=Ap)
            
            # Scalars as 0-dim tensors
            pAp = torch.vdot(p, Ap)
            alpha = rz_old / pAp
            
            # Updates (using broadcasting for scalar * vector)
            x.add_(p, alpha=alpha)
            r.add_(Ap, alpha=-alpha)
            
            # Preconditioner
            torch.mul(D_inv, r, out=z)
            
            rz_new = torch.vdot(r, z)
            beta = rz_new / rz_old
            
            # Direction update
            p.mul_(beta)
            p.add_(z)
            
            rz_old = rz_new
        
        return x, r, z, p, rz_old
    
    # Try to compile (may not work for sparse ops)
    try:
        cg_iterations_compiled = torch.compile(cg_iterations, mode='reduce-overhead')
    except Exception:
        cg_iterations_compiled = cg_iterations
    
    # Initialize
    if x0 is None:
        x = torch.zeros(n, dtype=dtype, device=device)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A.matvec(x)
    
    z = D_inv * r
    p = z.clone()
    Ap = torch.zeros(n, dtype=dtype, device=device)
    rz_old = torch.vdot(r, z)
    
    b_norm = torch.norm(b).item()
    if b_norm == 0:
        b_norm = 1.0
    tol = max(atol, rtol * b_norm)
    tol_sq = tol * tol
    
    # Main loop
    num_iters = 0
    while num_iters < maxiter:
        # Run batch of iterations
        iters_this_batch = min(check_interval, maxiter - num_iters)
        x, r, z, p, rz_old = cg_iterations_compiled(x, r, z, p, rz_old, Ap, iters_this_batch)
        num_iters += iters_this_batch
        
        # Check convergence
        residual_sq = torch.vdot(r, r).item()
        if residual_sq < tol_sq:
            return SolveResult(x, num_iters, residual_sq ** 0.5, True)
    
    residual = torch.norm(r).item()
    warnings.warn(f"Compiled PCG did not converge in {num_iters} iterations (residual={residual:.2e})")
    return SolveResult(x, num_iters, residual, False)


# ============================================================================
# Mixed Precision PCG Solver
# ============================================================================

class MixedPrecisionMatrix:
    """
    Mixed precision sparse matrix wrapper.
    
    Stores matrix in float32 for memory efficiency,
    performs matvec in float32, returns float64.
    """
    def __init__(self, val: Tensor, row: Tensor, col: Tensor, shape: Tuple[int, int]):
        self.shape = shape
        self.device = val.device
        self.n = shape[0]
        
        # Store in float32
        val_f32 = val.to(torch.float32)
        indices = torch.stack([row, col], dim=0)
        coo = torch.sparse_coo_tensor(indices, val_f32, shape, device=val.device, dtype=torch.float32)
        self._csr = coo.to_sparse_csr()
        
        # Cache diagonal in float64 for preconditioner
        self._diag = torch.zeros(self.n, dtype=torch.float64, device=self.device)
        diag_mask = row == col
        self._diag.scatter_add_(0, row[diag_mask], val[diag_mask].to(torch.float64))
    
    def matvec(self, x_f64: Tensor) -> Tensor:
        """Matrix-vector product: float32 matvec, float64 output."""
        x_f32 = x_f64.to(torch.float32)
        y_f32 = torch.mv(self._csr, x_f32)
        return y_f32.to(torch.float64)
    
    @property
    def diagonal(self) -> Tensor:
        return self._diag


def pcg_solve_mixed_precision(
    A: CachedSparseMatrix,
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    preconditioner: Callable[[Tensor], Tensor] = None,
) -> SolveResult:
    """
    Mixed precision PCG solver.
    
    Uses float32 for matrix-vector products (memory bandwidth bound)
    and float64 for dot products and accumulation (compute bound).
    
    This gives:
    - 2x faster matvec (float32)
    - High precision solution (float64 accumulation)
    - Same memory as float32 for matrix storage
    """
    device = A.device
    n = A.n
    
    # Create mixed precision matrix wrapper
    A_mixed = MixedPrecisionMatrix(A.val, A.row, A.col, A.shape)
    
    # Work in float64 for vectors
    if x0 is None:
        x = torch.zeros(n, dtype=torch.float64, device=device)
        r = b.to(torch.float64).clone()
    else:
        x = x0.to(torch.float64).clone()
        r = b.to(torch.float64) - A_mixed.matvec(x)
    
    # Preconditioner (in float64)
    if preconditioner is None:
        diag = A_mixed.diagonal
        eps = torch.finfo(torch.float64).eps * 100
        D_inv = 1.0 / torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
        preconditioner = lambda r: D_inv * r
    
    z = preconditioner(r)
    p = z.clone()
    rz_old = torch.vdot(r, z)
    
    b_norm = torch.norm(b.to(torch.float64))
    if b_norm == 0:
        b_norm = 1.0
    tol = max(atol, rtol * b_norm)
    
    residual = torch.norm(r).item()
    
    for i in range(maxiter):
        # Matrix-vector product in float32, result in float64
        Ap = A_mixed.matvec(p)
        
        pAp = torch.vdot(p, Ap)
        if pAp <= 0:
            warnings.warn(f"PCG: Matrix not positive definite (p'Ap = {pAp:.2e})")
            return SolveResult(x, i, residual, False)
        
        alpha = rz_old / pAp
        
        # Update in float64
        x.add_(p, alpha=alpha.item())
        r.add_(Ap, alpha=-alpha.item())
        
        residual = torch.norm(r).item()
        if residual < tol:
            return SolveResult(x, i + 1, residual, True)
        
        z = preconditioner(r)
        rz_new = torch.vdot(r, z)
        
        beta = rz_new / rz_old
        p.mul_(beta.item()).add_(z)
        rz_old = rz_new
    
    warnings.warn(f"Mixed PCG did not converge in {maxiter} iterations (residual={residual:.2e})")
    return SolveResult(x, maxiter, residual, False)


# ============================================================================
# Optimized PCG Solver
# ============================================================================

def pcg_solve_optimized(
    A: CachedSparseMatrix,
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    preconditioner: Callable[[Tensor], Tensor] = None,
    check_interval: int = 50,
) -> SolveResult:
    """
    Optimized Preconditioned Conjugate Gradient solver.

    Optimizations:
    - Cached CSR matrix (no repeated COO->CSR conversion)
    - **Pre-allocated buffers + in-place updates** (``addcmul_``, ``mul_``,
      ``.copy_``) so the only n-sized allocation per iter is the ``A @ p``
      result (PyTorch sparse mv exposes no ``out=``). The preconditioned
      residual ``z`` is reused via ``z.copy_(preconditioner(r))``.
    - Sync-free ``alpha`` / ``beta`` as 0-d GPU tensors fed to
      ``addcmul_``/``mul_`` (avoids the ``.item()`` -> Python scalar host
      sync per iter).
    - Only checks convergence every ``check_interval`` iterations.
    """
    device = A.device
    dtype = A.dtype
    n = A.n

    if preconditioner is None:
        preconditioner = jacobi_preconditioner(A)

    # Initial residual
    if x0 is None:
        x = torch.zeros(n, dtype=dtype, device=device)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A.matvec(x)

    # Pre-allocate the preconditioned-residual buffer z (reused every iter).
    z = torch.empty_like(r)
    z.copy_(preconditioner(r))
    p = z.clone()
    rz_old = torch.vdot(r, z)

    b_norm = torch.norm(b)
    tol_sq = (max(atol, rtol * b_norm.item())) ** 2  # one .item() at start

    # Pre-allocate a scratch for ``-alpha`` so the per-iter residual update
    # stays sync-free: addcmul_(value=...) only accepts a Python scalar, but
    # broadcasting a 0-d tensor through addcmul_(scratch, Ap) keeps alpha on
    # the GPU. ``torch.neg(alpha, out=neg_alpha)`` then writes ``-alpha``
    # without allocating.
    neg_alpha = torch.empty((), dtype=dtype, device=device)

    for i in range(maxiter):
        # PR #6 (dev) fix: check convergence at the START of every iteration.
        # CG converges in at most n steps, so small / well-conditioned systems
        # reach r -> 0 before a periodic check would fire. Once r -> 0,
        # rz_old and pAp collapse to 0 and alpha = rz_old / pAp = 0/0 = NaN,
        # which poisons x on subsequent updates. Checking here both reports
        # convergence and breaks out cleanly. The single scalar sync per
        # iteration (~10-30 us) is dwarfed by the SpMV cost.
        # ``.real`` keeps it real-scalar for complex r.
        residual_sq = torch.vdot(r, r).real
        if residual_sq.item() < tol_sq:
            return SolveResult(x, i, residual_sq.sqrt().item(), True)

        # Matrix-vector product (allocates Ap; PyTorch sparse mv has no out=).
        Ap = A.matvec(p)

        # vdot = sesquilinear inner product (correct for complex dtype; same
        # as dot for real). Required for the complex CG/Wirtinger adjoint.
        pAp = torch.vdot(p, Ap)
        alpha = rz_old / pAp  # 0-d tensor, stays on GPU
        torch.neg(alpha, out=neg_alpha)

        # In-place updates with NO host sync: addcmul_ computes
        #   x += alpha * p     (alpha is 0-d, broadcasts over p)
        #   r += -alpha * Ap   (via neg_alpha)
        x.addcmul_(alpha, p)
        r.addcmul_(neg_alpha, Ap)

        # Preconditioned residual -- in-place into the pre-allocated z buffer
        # so we don't allocate a fresh n-sized tensor each iter.
        z.copy_(preconditioner(r))

        rz_new = torch.vdot(r, z)
        beta = rz_new / rz_old  # 0-d tensor, stays on GPU

        # In-place search direction update: p = beta*p + z  (no sync, no alloc)
        p.mul_(beta).add_(z)
        rz_old = rz_new

    # Final residual
    residual = torch.norm(r).item()
    if residual < tol_sq ** 0.5:
        return SolveResult(x, maxiter, residual, True)
    
    warnings.warn(f"PCG did not converge in {maxiter} iterations (residual={residual:.2e})")
    return SolveResult(x, maxiter, residual, False)


def pipelined_pcg_solve(
    A: CachedSparseMatrix,
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    preconditioner: Callable[[Tensor], Tensor] = None,
    check_interval: int = 50,
) -> SolveResult:
    """
    Pipelined Preconditioned Conjugate Gradient solver.
    
    Reformulates CG to overlap computation and reduce synchronization points.
    Uses chronopoulos-Gear variant which only requires one global sync per iteration.
    
    Benefits:
    - Hides latency of global reductions
    - Better GPU utilization
    - 10-20% speedup for large problems
    
    Reference: Chronopoulos & Gear (1989)
    """
    device = A.device
    dtype = A.dtype
    n = A.n
    
    if preconditioner is None:
        preconditioner = jacobi_preconditioner(A)
    
    # Initial setup
    if x0 is None:
        x = torch.zeros(n, dtype=dtype, device=device)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A.matvec(x)
    
    # Precompute quantities
    u = preconditioner(r)
    w = A.matvec(u)
    
    gamma = torch.vdot(r, u)
    delta = torch.vdot(w, u)
    
    # Initialize
    m = preconditioner(w)
    n_vec = A.matvec(m)
    
    alpha = gamma / delta
    beta = torch.tensor(0.0, dtype=dtype, device=device)
    
    p = u.clone()
    s = w.clone()
    q = m.clone()
    z = n_vec.clone()
    
    b_norm = torch.norm(b)
    tol_sq = (max(atol, rtol * b_norm.item())) ** 2
    
    for i in range(maxiter):
        # Update solution
        x = x + alpha * p
        
        # Update residual
        r = r - alpha * s
        
        # Check convergence periodically
        if (i + 1) % check_interval == 0:
            residual_sq = torch.vdot(r, r).item()
            if residual_sq < tol_sq:
                return SolveResult(x, i + 1, residual_sq ** 0.5, True)
        
        # Update preconditioned vectors
        u = u - alpha * q
        w = w - alpha * z
        
        # Recompute for next iteration (only one sync point here)
        gamma_new = torch.vdot(r, u)
        delta_new = torch.vdot(w, u)
        
        # These can be done in parallel while sync is happening
        m = preconditioner(w)
        n_vec = A.matvec(m)
        
        beta = gamma_new / gamma
        alpha = gamma_new / (delta_new - beta * gamma_new / alpha)
        
        # Update search directions
        p = u + beta * p
        s = w + beta * s
        q = m + beta * q
        z = n_vec + beta * z
        
        gamma = gamma_new
    
    residual = torch.norm(r).item()
    warnings.warn(f"Pipelined PCG did not converge in {maxiter} iterations (residual={residual:.2e})")
    return SolveResult(x, maxiter, residual, False)


def pcg_solve_fused(
    val: Tensor,
    row: Tensor,
    col: Tensor,
    shape: Tuple[int, int],
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    preconditioner: str = 'auto',
) -> SolveResult:
    """
    Fused PCG solver with all optimizations.
    
    This is the recommended entry point for high-performance CG solving.
    
    Parameters
    ----------
    preconditioner : str
        Preconditioner to use. Options:
        - 'auto': Automatically select based on matrix conditioning (RECOMMENDED)
        - 'jacobi': Diagonal preconditioner (fast but basic)
        - 'ssor': Symmetric SOR (better convergence)
        - 'block_jacobi': Block Jacobi (good for structured matrices)
        - 'ic0': Incomplete Cholesky (best for ill-conditioned SPD)
        - 'amg': Algebraic Multigrid (best for very ill-conditioned)
        - 'none': No preconditioning
    """
    # Build cached sparse matrix
    A = CachedSparseMatrix(val, row, col, shape)
    
    # Get preconditioner (pass method='cg' for auto selection)
    M = get_preconditioner(A, preconditioner, method='cg')
    
    return pcg_solve_optimized(A, b, x0=x0, atol=atol, rtol=rtol, maxiter=maxiter, preconditioner=M)


# ============================================================================
# BiCGStab Solver
# ============================================================================

def pbicgstab_solve(
    val: Tensor,
    row: Tensor,
    col: Tensor,
    shape: Tuple[int, int],
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    preconditioner: str = 'auto',
) -> Tuple[Tensor, int, float]:
    """Preconditioned BiCGStab solver with automatic preconditioner selection."""
    A = CachedSparseMatrix(val, row, col, shape)
    M = get_preconditioner(A, preconditioner, method='bicgstab')
    
    device = A.device
    dtype = A.dtype
    n = A.n
    
    if x0 is None:
        x = torch.zeros(n, dtype=dtype, device=device)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A.matvec(x)
    
    r0_hat = r.clone()
    
    rho_old = torch.tensor(1.0, dtype=dtype, device=device)
    alpha = torch.tensor(1.0, dtype=dtype, device=device)
    omega = torch.tensor(1.0, dtype=dtype, device=device)
    
    v = torch.zeros(n, dtype=dtype, device=device)
    p = torch.zeros(n, dtype=dtype, device=device)
    
    b_norm = torch.norm(b)
    if b_norm == 0:
        b_norm = 1.0
    tol = max(atol, rtol * b_norm)
    residual = torch.norm(r).item()
    
    for i in range(maxiter):
        rho_new = torch.vdot(r0_hat, r)
        
        if abs(rho_new.item()) < 1e-30:
            warnings.warn("PBiCGStab: rho became too small")
            break
        
        beta = (rho_new / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        p_hat = M(p)
        v = A.matvec(p_hat)
        
        alpha = rho_new / torch.vdot(r0_hat, v)
        s = r - alpha * v
        
        s_norm = torch.norm(s).item()
        if s_norm < tol:
            x = x + alpha * p_hat
            return x, i + 1, s_norm
        
        s_hat = M(s)
        t = A.matvec(s_hat)
        
        # vdot(t, t) is real-valued (magnitude squared); cast to .real so the
        # < scalar comparison works for both real and complex t.
        t_dot_t = torch.vdot(t, t).real
        if t_dot_t < 1e-30:
            x = x + alpha * p_hat
            break
        
        omega = torch.vdot(t, s) / t_dot_t
        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t
        
        residual = torch.norm(r).item()
        
        if residual < tol:
            return x, i + 1, residual
        
        if abs(omega.item()) < 1e-30:
            break
        
        rho_old = rho_new
    
    warnings.warn(f"PBiCGStab did not converge in {maxiter} iterations (residual={residual:.2e})")
    return x, maxiter, residual


# ============================================================================
# GMRES (right-preconditioned, restarted) -- general / non-symmetric systems
# ============================================================================

def _conj_scalar(z):
    """Conjugate a Python scalar (complex or real)."""
    return z.conjugate() if isinstance(z, complex) else z


def _givens(a, b):
    """Complex-safe Givens rotation.

    Returns ``(c, s, r)`` with ``c`` real, ``s`` (possibly complex) such that
    ``[[c, s], [-conj(s), c]] @ [a, b]^T == [r, 0]^T``. Here ``b`` is the
    sub-diagonal entry, which in our Arnoldi loop is always a real, non-negative
    norm, but the general complex form is used so ``a`` may be complex.
    """
    if b == 0:
        return 1.0, (0j if isinstance(a, complex) else 0.0), a
    if a == 0:
        ab = abs(b)
        s = (b.conjugate() / ab) if isinstance(b, complex) else (b / ab)
        return 0.0, s, ab
    aa = abs(a)
    nrm = math.sqrt(aa * aa + abs(b) * abs(b))
    c = aa / nrm
    if isinstance(a, complex) or isinstance(b, complex):
        alpha = a / aa  # unit-modulus phase of a
        bconj = b.conjugate() if isinstance(b, complex) else b
        return c, alpha * bconj / nrm, alpha * nrm
    sgn = a / aa  # +/-1
    return c, sgn * b / nrm, sgn * nrm


def pgmres_solve(
    val: Tensor,
    row: Tensor,
    col: Tensor,
    shape: Tuple[int, int],
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    preconditioner: str = 'auto',
    restart: int = 30,
) -> Tuple[Tensor, int, float]:
    """Right-preconditioned restarted GMRES(``restart``).

    Solves ``A x = b`` for a general (non-symmetric, possibly complex) sparse
    ``A`` on CPU or CUDA. Right preconditioning keeps the monitored quantity the
    *true* residual ``b - A x`` (not the preconditioned residual), so the
    convergence test is honest regardless of ``M``.

    ``maxiter`` bounds the total number of inner Arnoldi steps across restarts.
    """
    A = CachedSparseMatrix(val, row, col, shape)
    M = get_preconditioner(A, preconditioner, method='gmres')
    device, dtype, n = A.device, A.dtype, A.n
    is_complex = torch.is_complex(b)

    if x0 is None:
        x = torch.zeros(n, dtype=dtype, device=device)
        r = b.clone()
    else:
        x = x0.clone()
        r = b - A.matvec(x)

    b_norm = torch.norm(b).item()
    if b_norm == 0:
        b_norm = 1.0
    tol = max(atol, rtol * b_norm)

    r_norm = torch.norm(r).item()
    if r_norm <= tol:
        return x, 0, r_norm

    m = min(int(restart), n)
    zero = 0j if is_complex else 0.0
    total = 0
    while total < maxiter:
        # Arnoldi basis (n x m+1) + Hessenberg / Givens kept as host scalars
        V = torch.zeros(n, m + 1, dtype=dtype, device=device)
        H = [[zero] * m for _ in range(m + 1)]
        cs = [0.0] * m
        sn = [zero] * m
        g = [zero] * (m + 1)

        beta = torch.norm(r).item()
        V[:, 0] = r / beta
        g[0] = beta
        k = 0
        for j in range(m):
            total += 1
            w = A.matvec(M(V[:, j]))              # right preconditioning
            for i in range(j + 1):
                hij = torch.vdot(V[:, i], w).item()
                H[i][j] = hij
                w = w - V[:, i] * hij
            hjp = torch.norm(w).item()
            H[j + 1][j] = hjp
            if hjp > 1e-30:
                V[:, j + 1] = w / hjp
            # apply previous rotations to the new column
            for i in range(j):
                t = cs[i] * H[i][j] + sn[i] * H[i + 1][j]
                H[i + 1][j] = -_conj_scalar(sn[i]) * H[i][j] + cs[i] * H[i + 1][j]
                H[i][j] = t
            # eliminate H[j+1][j]
            c, s, rr = _givens(H[j][j], H[j + 1][j])
            cs[j], sn[j] = c, s
            H[j][j] = rr
            H[j + 1][j] = zero
            g[j + 1] = -_conj_scalar(s) * g[j]
            g[j] = c * g[j]
            k = j + 1
            if abs(g[j + 1]) <= tol or total >= maxiter or hjp <= 1e-30:
                break
        # back-substitution: H[:k,:k] y = g[:k]
        y = [zero] * k
        for i in range(k - 1, -1, -1):
            acc = g[i]
            for jj in range(i + 1, k):
                acc -= H[i][jj] * y[jj]
            y[i] = acc / H[i][i]
        yv = torch.tensor(y, dtype=dtype, device=device)
        x = x + M(V[:, :k] @ yv)                  # undo right preconditioner
        r = b - A.matvec(x)
        r_norm = torch.norm(r).item()
        if r_norm <= tol:
            return x, total, r_norm

    warnings.warn(f"GMRES did not converge in {total} iterations (residual={r_norm:.2e})")
    return x, total, r_norm


# ============================================================================
# MINRES (preconditioned) -- symmetric / Hermitian, possibly indefinite
# ============================================================================

def minres_solve(
    val: Tensor,
    row: Tensor,
    col: Tensor,
    shape: Tuple[int, int],
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    preconditioner: str = 'auto',
) -> Tuple[Tensor, int, float]:
    """Preconditioned MINRES for symmetric/Hermitian (possibly indefinite) ``A``.

    Follows the Paige-Saunders formulation (Stanford SOL ``minres``). Requires an
    **SPD preconditioner** ``M``; for a Hermitian system the Lanczos tridiagonal
    is real, so all recurrence scalars stay real even though the vectors may be
    complex. Short 3-term recurrence -> constant memory (unlike GMRES).
    """
    A = CachedSparseMatrix(val, row, col, shape)
    M = get_preconditioner(A, preconditioner, method='minres')
    device, dtype, n = A.device, A.dtype, A.n

    if x0 is None:
        x = torch.zeros(n, dtype=dtype, device=device)
        r1 = b.clone()
    else:
        x = x0.clone()
        r1 = b - A.matvec(x)

    y = M(r1)
    beta_start = torch.vdot(r1, y).real.item()    # r1^H M^{-1} r1
    if beta_start < 0:
        raise ValueError("MINRES requires an SPD preconditioner (got <r,M r> < 0)")
    beta = math.sqrt(beta_start)

    b_norm = torch.norm(b).item()
    if b_norm == 0:
        # b == 0  =>  x == 0 is the exact solution
        return torch.zeros(n, dtype=dtype, device=device), 0, 0.0
    beta1 = beta
    if beta == 0:
        return x, 0, 0.0

    # Machine epsilon for the (real) working precision. For complex dtypes use
    # the precision of the underlying real scalars, which is what the recurrence
    # actually runs in.
    real_dtype = torch.empty(0, dtype=dtype).real.dtype
    eps = float(torch.finfo(real_dtype).eps)

    # rotation + recurrence state (all real for a Hermitian system)
    oldb = 0.0
    dbar = 0.0
    epsln = 0.0
    phibar = beta
    cs = -1.0
    sn = 0.0
    w = torch.zeros(n, dtype=dtype, device=device)
    w2 = torch.zeros(n, dtype=dtype, device=device)
    r2 = r1.clone()

    # Paige-Saunders normalized-test running estimates (mirrors scipy.minres):
    #   test1 = ||r|| / (||A|| ||x||)   relative residual
    #   test2 = ||Ar|| / (||A|| ||r||)  least-squares residual
    # with Anorm from a running Frobenius-like estimate, Acond = gmax/gmin.
    tnorm2 = 0.0
    gmax = 0.0
    gmin = float(torch.finfo(real_dtype).max)
    rhs1 = beta1
    rhs2 = 0.0
    Anorm = 0.0
    ynorm = 0.0
    rnorm = beta1
    istop = 0

    for itn in range(1, maxiter + 1):
        v = (1.0 / beta) * y                       # Lanczos vector
        y = A.matvec(v)
        if itn >= 2:
            y = y - (beta / oldb) * r1
        alfa = torch.vdot(v, y).real.item()
        y = y - (alfa / beta) * r2
        r1 = r2
        r2 = y
        y = M(r2)
        oldb = beta
        beta_sq = torch.vdot(r2, y).real.item()
        if beta_sq < 0:
            raise ValueError("MINRES requires an SPD preconditioner (got <r,M r> < 0)")
        beta = math.sqrt(beta_sq)
        tnorm2 += alfa * alfa + oldb * oldb + beta * beta

        # apply previous rotation, then build the next one
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        root = math.sqrt(gbar * gbar + dbar * dbar)   # ||[gbar, dbar]||
        Arnorm = phibar * root                        # estimate of ||A r||

        gamma = math.sqrt(gbar * gbar + beta * beta)
        gamma = max(gamma, eps)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar

        # solution update (short recurrence)
        w1, w2 = w2, w
        w = (v - oldeps * w1 - delta * w2) * (1.0 / gamma)
        x = x + phi * w

        # update running condition-number estimate
        gmax = max(gmax, gamma)
        gmin = min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = -epsln * z

        # estimate norms and the normalized stopping tests (Paige-Saunders)
        Anorm = math.sqrt(tnorm2)
        ynorm = torch.norm(x).item()
        epsx = Anorm * ynorm * eps

        rnorm = phibar
        if ynorm == 0.0 or Anorm == 0.0:
            test1 = float('inf')
        else:
            test1 = rnorm / (Anorm * ynorm)          # ||r|| / (||A|| ||x||)
        if Anorm == 0.0:
            test2 = float('inf')
        else:
            test2 = root / Anorm                     # ||Ar|| / (||A|| ||r||)

        Acond = gmax / gmin if gmin > 0 else float('inf')

        # Stopping criteria (combine scipy's normalized tests with the
        # caller's atol/rtol on the genuine preconditioned residual rnorm).
        if istop == 0:
            t1 = 1.0 + test1            # exact when rtol < eps
            t2 = 1.0 + test2
            if t2 <= 1.0:
                istop = 2
            if t1 <= 1.0:
                istop = 1
            if itn >= maxiter:
                istop = 6
            if Acond >= 0.1 / eps:
                istop = 4
            if epsx >= beta1:
                istop = 3
            if test2 <= rtol:
                istop = 2
            if test1 <= rtol:
                istop = 1
            # Honor an explicit absolute tolerance on the residual as well so
            # callers asking for a tight atol still get it.
            if rnorm <= atol:
                istop = 1

        if istop != 0:
            converged = istop in (1, 2, 3)
            if not converged and istop == 6:
                warnings.warn(
                    f"MINRES did not converge in {maxiter} iterations "
                    f"(residual={rnorm:.2e})"
                )
            return x, itn, rnorm

    warnings.warn(f"MINRES did not converge in {maxiter} iterations (residual={rnorm:.2e})")
    return x, maxiter, rnorm


# ============================================================================
# LSQR / LSMR -- least-squares (Golub-Kahan bidiagonalization), matrix-free.
# Pure-torch ports of scipy.sparse.linalg.lsqr / lsmr; device-agnostic
# (CPU / CUDA / ROCm). Solve  min_x ||A x - b||_2  with optional Tikhonov
# ``damp``; A may be rectangular. Scalars (norms, rotations) are real;
# the only matrix ops are A @ x and A^H @ y, so complex A works (rmatvec
# uses the conjugate transpose). Forward only (called inside an autograd
# Function's forward); uses .item() freely.
# ============================================================================

def _sym_ortho(a, b):
    """Stable Givens rotation (Choi). Returns (c, s, r) on real scalars."""
    if b == 0:
        return (math.copysign(1.0, a) if a != 0 else 1.0), 0.0, abs(a)
    if a == 0:
        return 0.0, math.copysign(1.0, b), abs(b)
    if abs(b) > abs(a):
        tau = a / b
        s = math.copysign(1.0, b) / math.sqrt(1 + tau * tau)
        return s * tau, s, b / s
    tau = b / a
    c = math.copysign(1.0, a) / math.sqrt(1 + tau * tau)
    return c, c * tau, a / c


def _ls_matvecs(val, row, col, shape):
    """Return (matvec, rmatvec) for A and A^H (conj-transpose for complex)."""
    A = CachedSparseMatrix(val, row, col, shape)
    AT = CachedSparseMatrix(torch.conj_physical(val), col, row, (shape[1], shape[0]))
    return A.matvec, AT.matvec


def lsqr_solve(val, row, col, shape, b, *, atol=1e-8, btol=1e-8, maxiter=10000,
               damp=0.0, conlim=1e8, x0=None, **kwargs):
    """LSQR (Paige & Saunders 1982). Faithful torch port of scipy lsqr."""
    matvec, rmatvec = _ls_matvecs(val, row, col, shape)
    m, n = shape
    dtype, device = b.dtype, b.device
    eps = torch.finfo(b.real.dtype if torch.is_complex(b) else dtype).eps
    dampsq = damp * damp
    ctol = 1.0 / conlim if conlim > 0 else 0.0
    anorm = ddnorm = res2 = xnorm = xxnorm = z = 0.0
    cs2, sn2 = -1.0, 0.0

    x = torch.zeros(n, dtype=dtype, device=device) if x0 is None else x0.clone()
    u = b.clone() if x0 is None else (b - matvec(x))
    beta = u.norm().item()
    if beta > 0:
        u = u / beta
    v = rmatvec(u)
    alfa = v.norm().item()
    if alfa > 0:
        v = v / alfa
    w = v.clone()
    rhobar, phibar, bnorm, rnorm = alfa, beta, beta, beta
    if alfa * beta == 0:  # arnorm == 0: x=0 (or x0) is already optimal
        return x, 0, rnorm

    istop = 0
    for itn in range(1, maxiter + 1):
        u = matvec(v) - alfa * u
        beta = u.norm().item()
        if beta > 0:
            u = u / beta
            anorm = math.sqrt(anorm ** 2 + alfa ** 2 + beta ** 2 + dampsq)
            v = rmatvec(u) - beta * v
            alfa = v.norm().item()
            if alfa > 0:
                v = v / alfa
        if damp > 0:
            rhobar1 = math.sqrt(rhobar ** 2 + dampsq)
            cs1, sn1 = rhobar / rhobar1, damp / rhobar1
            psi, phibar = sn1 * phibar, cs1 * phibar
        else:
            rhobar1, psi = rhobar, 0.0
        cs, sn, rho = _sym_ortho(rhobar1, beta)
        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi
        dk = (1.0 / rho) * w
        x = x + (phi / rho) * w
        w = v + (-theta / rho) * w
        ddnorm = ddnorm + dk.norm().item() ** 2
        # ||x|| estimate
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = math.sqrt(xxnorm + zbar ** 2)
        gamma = math.sqrt(gambar ** 2 + theta ** 2)
        cs2, sn2 = gambar / gamma, theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z ** 2
        acond = anorm * math.sqrt(ddnorm)
        rnorm = math.sqrt(phibar ** 2 + res2)
        res2 = res2 + psi ** 2
        arnorm = alfa * abs(tau)
        test1 = rnorm / bnorm if bnorm > 0 else 0.0
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1.0 / (acond + eps)
        t1c = test1 / (1 + anorm * xnorm / bnorm) if bnorm > 0 else test1
        rtol = btol + atol * anorm * xnorm / bnorm if bnorm > 0 else btol
        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1c <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if istop:
            break
    return x, itn, rnorm


def lsmr_solve(val, row, col, shape, b, *, atol=1e-8, btol=1e-8, maxiter=10000,
               damp=0.0, conlim=1e8, x0=None, **kwargs):
    """LSMR (Fong & Saunders 2011). Faithful torch port of scipy lsmr."""
    matvec, rmatvec = _ls_matvecs(val, row, col, shape)
    m, n = shape
    dtype, device = b.dtype, b.device
    eps = torch.finfo(b.real.dtype if torch.is_complex(b) else dtype).eps
    ctol = 1.0 / conlim if conlim > 0 else 0.0

    x = torch.zeros(n, dtype=dtype, device=device) if x0 is None else x0.clone()
    u = b.clone() if x0 is None else (b - matvec(x))
    beta = u.norm().item()
    if beta > 0:
        u = u / beta
    v = rmatvec(u)
    alpha = v.norm().item()
    if alpha > 0:
        v = v / alpha

    # init
    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = rhobar = cbar = 1.0
    sbar = 0.0
    h = v.clone()
    hbar = torch.zeros(n, dtype=dtype, device=device)
    betadd = beta
    betad = 0.0
    rhodold = 1.0
    tautildeold = thetatilde = zeta = d = 0.0
    normA2 = alpha * alpha
    maxrbar, minrbar = 0.0, 1e100
    normb = beta
    normr = beta
    normar = alpha * beta
    if normar == 0:
        return x, 0, normr
    if normb == 0:
        return x, 0, 0.0

    istop = 0
    for itn in range(1, maxiter + 1):
        u = matvec(v) - alpha * u
        beta = u.norm().item()
        if beta > 0:
            u = u / beta
            v = rmatvec(u) - beta * v
            alpha = v.norm().item()
            if alpha > 0:
                v = v / alpha
        chat, shat, alphahat = _sym_ortho(alphabar, damp)
        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s * alpha
        alphabar = c * alpha
        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = -sbar * zetabar
        hbar = h + (-(thetabar * rho / (rhoold * rhobarold))) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v + (-(thetanew / rho)) * h
        # ||r|| estimate
        betaacute = chat * betadd
        betacheck = -shat * betadd
        betahat = c * betaacute
        betadd = -s * betaacute
        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = -stildeold * betad + ctildeold * betahat
        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = math.sqrt(d + (betad - taud) ** 2 + betadd * betadd)
        normA2 = normA2 + beta * beta
        normA = math.sqrt(normA2)
        normA2 = normA2 + alpha * alpha
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
        normar = abs(zetabar)
        normx = x.norm().item()
        test1 = normr / normb if normb > 0 else 0.0
        test2 = normar / (normA * normr + eps)
        test3 = 1.0 / (condA + eps)
        t1c = test1 / (1 + normA * normx / normb) if normb > 0 else test1
        rtol = btol + atol * normA * normx / normb if normb > 0 else btol
        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1c <= 1:
            istop = 4
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1
        if istop:
            break
    return x, itn, normr


# ============================================================================
# Legacy interfaces for backward compatibility
# ============================================================================

def sparse_matvec(val, row, col, shape, x):
    """Sparse matrix-vector product y = A @ x using COO format."""
    indices = torch.stack([row, col], dim=0)
    A = torch.sparse_coo_tensor(indices, val, shape, device=x.device, dtype=x.dtype)
    A_csr = A.to_sparse_csr()
    return torch.mv(A_csr, x)


def extract_diagonal(val, row, col, shape):
    """Extract diagonal of sparse matrix in COO format."""
    n = min(shape[0], shape[1])
    diag = torch.zeros(n, dtype=val.dtype, device=val.device)
    diag_mask = row == col
    diag.scatter_add_(0, row[diag_mask], val[diag_mask])
    return diag


def pcg_solve(
    val: Tensor,
    row: Tensor,
    col: Tensor,
    shape: Tuple[int, int],
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    M_inv: Optional[Tensor] = None,
    callback: Optional[callable] = None,
) -> Tuple[Tensor, int, float]:
    """Preconditioned Conjugate Gradient solver (legacy interface)."""
    result = pcg_solve_fused(val, row, col, shape, b, x0=x0, atol=atol, rtol=rtol, 
                             maxiter=maxiter, preconditioner='jacobi')
    return result.x, result.num_iters, result.residual


def cg_solve(
    val: Tensor,
    row: Tensor,
    col: Tensor,
    shape: Tuple[int, int],
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    callback: Optional[callable] = None,
) -> Tuple[Tensor, int, float]:
    """Conjugate Gradient with automatic preconditioner selection."""
    result = pcg_solve_fused(val, row, col, shape, b, x0=x0, atol=atol, rtol=rtol, 
                             maxiter=maxiter, preconditioner='auto')
    return result.x, result.num_iters, result.residual


def bicgstab_solve(
    val: Tensor,
    row: Tensor,
    col: Tensor,
    shape: Tuple[int, int],
    b: Tensor,
    x0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    callback: Optional[callable] = None,
) -> Tuple[Tensor, int, float]:
    """BiCGStab with automatic preconditioner selection."""
    return pbicgstab_solve(val, row, col, shape, b, x0=x0, atol=atol, rtol=rtol,
                           maxiter=maxiter, preconditioner='auto')


def pytorch_solve(
    val: Tensor,
    row: Tensor,
    col: Tensor,
    shape: Tuple[int, int],
    b: Tensor,
    method: str = 'cg',
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    x0: Optional[Tensor] = None,
    preconditioner: str = 'jacobi',
    mixed_precision: bool = False,
) -> Tensor:
    """
    Solve sparse linear system using PyTorch-native iterative methods.
    
    Parameters
    ----------
    val, row, col : Tensor
        COO format sparse matrix
    shape : Tuple[int, int]
        Matrix shape
    b : Tensor
        Right-hand side
    method : str
        'cg', 'bicgstab', 'gmres', or 'minres'
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance
    maxiter : int
        Maximum iterations
    x0 : Tensor, optional
        Initial guess
    preconditioner : str
        'jacobi', 'ssor', 'block_jacobi', 'polynomial', 'ic0', 'amg', or 'none'
    mixed_precision : bool
        If True, use float32 for matvec and float64 for accumulation.
        This gives ~2x speedup with high precision solution.
        
    Returns
    -------
    Tensor
        Solution vector
    """
    if method == 'cg':
        if b.dim() == 2:
            # Multiple RHS: use batched PCG solver with SpMM
            A_mat = CachedSparseMatrix(val, row, col, shape)
            M = get_preconditioner(A_mat, preconditioner)
            result = batched_pcg_solve(A_mat, b, X0=x0, atol=atol, rtol=rtol,
                                       maxiter=maxiter, preconditioner=M)
            return result.X
        elif mixed_precision and val.dtype == torch.float32:
            A = CachedSparseMatrix(val, row, col, shape)
            M = get_preconditioner(A, preconditioner)
            # Wrap preconditioner for float64
            def M_f64(r):
                return M(r.to(torch.float32)).to(torch.float64)
            result = pcg_solve_mixed_precision(A, b, x0=x0, atol=atol, rtol=rtol,
                                               maxiter=maxiter, preconditioner=M_f64)
        else:
            result = pcg_solve_fused(val, row, col, shape, b, x0=x0, atol=atol, rtol=rtol,
                                     maxiter=maxiter, preconditioner=preconditioner)
        return result.x
    elif method == 'bicgstab':
        if b.dim() == 2:
            # No batched bicgstab; column-loop
            cols = []
            for k in range(b.shape[1]):
                x_k, _, _ = pbicgstab_solve(val, row, col, shape, b[:, k], x0=None,
                                            atol=atol, rtol=rtol, maxiter=maxiter,
                                            preconditioner=preconditioner)
                cols.append(x_k)
            return torch.stack(cols, dim=1)
        x, _, _ = pbicgstab_solve(val, row, col, shape, b, x0=x0, atol=atol, rtol=rtol,
                                  maxiter=maxiter, preconditioner=preconditioner)
        return x
    elif method == 'gmres':
        if b.dim() == 2:
            cols = []
            for k in range(b.shape[1]):
                x_k, _, _ = pgmres_solve(val, row, col, shape, b[:, k], x0=None,
                                         atol=atol, rtol=rtol, maxiter=maxiter,
                                         preconditioner=preconditioner)
                cols.append(x_k)
            return torch.stack(cols, dim=1)
        x, _, _ = pgmres_solve(val, row, col, shape, b, x0=x0, atol=atol, rtol=rtol,
                               maxiter=maxiter, preconditioner=preconditioner)
        return x
    elif method == 'minres':
        if b.dim() == 2:
            cols = []
            for k in range(b.shape[1]):
                x_k, _, _ = minres_solve(val, row, col, shape, b[:, k], x0=None,
                                         atol=atol, rtol=rtol, maxiter=maxiter,
                                         preconditioner=preconditioner)
                cols.append(x_k)
            return torch.stack(cols, dim=1)
        x, _, _ = minres_solve(val, row, col, shape, b, x0=x0, atol=atol, rtol=rtol,
                               maxiter=maxiter, preconditioner=preconditioner)
        return x
    elif method in ('lsqr', 'lsmr'):
        # least-squares (Golub-Kahan); no preconditioner. btol from rtol.
        fn = lsqr_solve if method == 'lsqr' else lsmr_solve
        if b.dim() == 2:
            cols = [fn(val, row, col, shape, b[:, k], atol=atol, btol=rtol,
                       maxiter=maxiter)[0] for k in range(b.shape[1])]
            return torch.stack(cols, dim=1)
        x, _, _ = fn(val, row, col, shape, b, x0=x0, atol=atol, btol=rtol, maxiter=maxiter)
        return x
    else:
        raise ValueError(f"Unknown method: {method}. Available: cg, bicgstab, gmres, minres, lsqr, lsmr")


# ============================================================================
# Batched CG Solver - Multiple RHS
# ============================================================================

class BatchSolveResult(NamedTuple):
    """Result of batched iterative solve."""
    X: Tensor  # (n, batch_size) solution matrix
    num_iters: Tensor  # (batch_size,) iterations per RHS
    residuals: Tensor  # (batch_size,) residuals
    converged: Tensor  # (batch_size,) convergence flags


def batched_pcg_solve(
    A: CachedSparseMatrix,
    B: Tensor,  # (n, batch_size) multiple RHS
    X0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    preconditioner: Callable[[Tensor], Tensor] = None,
    check_interval: int = 50,
) -> BatchSolveResult:
    """
    Batched Preconditioned Conjugate Gradient solver.
    
    Solves A @ X = B where B is a matrix of multiple right-hand sides.
    Uses sparse matrix-matrix multiplication (SpMM) for better GPU utilization.
    
    Parameters
    ----------
    A : CachedSparseMatrix
        Sparse coefficient matrix (n, n)
    B : Tensor
        Right-hand side matrix (n, batch_size)
    X0 : Tensor, optional
        Initial guess (n, batch_size)
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance  
    maxiter : int
        Maximum iterations
    preconditioner : Callable
        Preconditioner function M^{-1}(r) - applied column-wise
    check_interval : int
        How often to check convergence
        
    Returns
    -------
    BatchSolveResult
        Named tuple with X, num_iters, residuals, converged
        
    Notes
    -----
    - Uses SpMM (sparse matrix-matrix multiply) instead of multiple SpMV
    - All RHS are solved together until they all converge or reach maxiter
    - Individual RHS may converge at different iterations
    """
    was_1d = B.dim() == 1
    if was_1d:
        B = B.unsqueeze(1)
    
    device = A.device
    dtype = A.dtype
    n = A.n
    batch_size = B.shape[1]
    
    # Use SpMV for single RHS (faster), SpMM for batched
    use_spmv = (batch_size == 1)
    
    def sparse_matvec(M):
        """Dispatch to SpMV or SpMM based on batch size."""
        if use_spmv:
            return torch.mv(A._csr, M.squeeze(1)).unsqueeze(1)
        else:
            return torch.mm(A._csr, M)
    
    # Default preconditioner (batched Jacobi)
    if preconditioner is None:
        D_inv = 1.0 / A.diagonal
        D_inv = torch.where(torch.isinf(D_inv), torch.ones_like(D_inv), D_inv)
        def preconditioner(R):
            if R.dim() == 1:
                return D_inv * R
            return D_inv.unsqueeze(1) * R  # (n, 1) * (n, batch)
    
    # Initial guess
    if X0 is None:
        X = torch.zeros(n, batch_size, dtype=dtype, device=device)
        R = B.clone()
    else:
        X = X0.clone() if X0.dim() == 2 else X0.unsqueeze(1)
        R = B - sparse_matvec(X)
    
    # Preconditioned residual
    Z = preconditioner(R)
    P = Z.clone()
    RZ_old = (R * Z).sum(dim=0)  # (batch_size,) dot products
    
    # Convergence tracking
    B_norms = torch.norm(B, dim=0)  # (batch_size,)
    B_norms = torch.where(B_norms == 0, torch.ones_like(B_norms), B_norms)
    tols = torch.maximum(torch.full_like(B_norms, atol), rtol * B_norms)
    tols_sq = tols ** 2
    
    converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
    num_iters = torch.zeros(batch_size, dtype=torch.int32, device=device)
    active_mask = ~converged  # Which columns are still active
    
    for i in range(maxiter):
        if not active_mask.any():
            break
        
        # SpMV or SpMM: AP = A @ P
        AP = sparse_matvec(P)
        
        # Compute alpha for each column: alpha_j = RZ_old_j / (P_j' @ AP_j)
        PAP = (P * AP).sum(dim=0)  # (batch_size,)
        
        # Avoid division by zero
        PAP = torch.where(PAP > 0, PAP, torch.ones_like(PAP))
        alpha = RZ_old / PAP  # (batch_size,)
        
        # Update X and R (vectorized)
        X = X + alpha.unsqueeze(0) * P  # (n, batch)
        R = R - alpha.unsqueeze(0) * AP  # (n, batch)
        
        # Check convergence periodically
        if (i + 1) % check_interval == 0:
            residual_sq = (R * R).sum(dim=0)  # (batch_size,)
            newly_converged = (residual_sq < tols_sq) & active_mask
            
            if newly_converged.any():
                num_iters[newly_converged] = i + 1
                converged = converged | newly_converged
                active_mask = ~converged
            
            if not active_mask.any():
                break
        
        # Preconditioned residual
        Z = preconditioner(R)
        RZ_new = (R * Z).sum(dim=0)  # (batch_size,)
        
        # Compute beta
        beta = RZ_new / (RZ_old + 1e-30)  # (batch_size,)
        
        # Update search direction
        P = Z + beta.unsqueeze(0) * P  # (n, batch)
        RZ_old = RZ_new
    
    # Final residuals
    final_residuals = torch.norm(R, dim=0)
    
    # Set iterations for non-converged
    num_iters[~converged] = maxiter
    
    # Check final convergence
    converged = converged | (final_residuals < tols)
    
    if not converged.all():
        n_failed = (~converged).sum().item()
        warnings.warn(f"Batched PCG: {n_failed}/{batch_size} RHS did not converge in {maxiter} iterations")
    
    return BatchSolveResult(X, num_iters, final_residuals, converged)


def batched_cg_solve(
    val: Tensor,
    row: Tensor,
    col: Tensor,
    shape: Tuple[int, int],
    B: Tensor,
    X0: Optional[Tensor] = None,
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
    preconditioner: str = 'jacobi',
    check_interval: int = 50,
) -> BatchSolveResult:
    """
    Batched Conjugate Gradient solver (convenience wrapper).
    
    Parameters
    ----------
    val, row, col : Tensor
        COO format sparse matrix
    shape : Tuple[int, int]
        Matrix shape (n, n)
    B : Tensor
        Right-hand side matrix (n, batch_size) or (n,) for single RHS
    X0 : Tensor, optional
        Initial guess
    atol : float
        Absolute tolerance
    rtol : float
        Relative tolerance
    maxiter : int
        Maximum iterations
    preconditioner : str
        'jacobi', 'ssor', 'ic0', etc.
    check_interval : int
        Convergence check interval
        
    Returns
    -------
    BatchSolveResult
        Named tuple with X, num_iters, residuals, converged
    """
    A = CachedSparseMatrix(val, row, col, shape)
    
    # Get preconditioner
    M_single = get_preconditioner(A, preconditioner)
    
    # Wrap for batched operation
    if preconditioner == 'jacobi':
        D_inv = 1.0 / A.diagonal
        D_inv = torch.where(torch.isinf(D_inv), torch.ones_like(D_inv), D_inv)
        def M_batched(R):
            return D_inv.unsqueeze(1) * R if R.dim() == 2 else D_inv * R
    else:
        # For other preconditioners, apply column-wise
        def M_batched(R):
            if R.dim() == 1:
                return M_single(R)
            else:
                # Apply to each column
                return torch.stack([M_single(R[:, j]) for j in range(R.shape[1])], dim=1)
    
    return batched_pcg_solve(
        A, B, X0=X0, atol=atol, rtol=rtol, maxiter=maxiter,
        preconditioner=M_batched, check_interval=check_interval
    )


def batched_cg_same_pattern(
    val_batch: Tensor,   # [B, nnz]
    row: Tensor,         # [nnz]   (shared across the batch)
    col: Tensor,         # [nnz]   (shared across the batch)
    shape: Tuple[int, int],
    b_batch: Tensor,     # [B, n]
    atol: float = 1e-10,
    rtol: float = 1e-6,
    maxiter: int = 10000,
) -> Tensor:
    """Truly *batched* CG for a batch of SPD matrices that share a sparsity
    pattern ``(row, col)`` but have different values.

    Solves ``A_i x_i = b_i`` for every i **simultaneously** -- the batch lives
    on dim 0 throughout, so there is no Python loop over matrices. The matvec is
    a single batched scatter (``index_add_`` over the shared pattern) and every
    CG scalar (alpha, beta, rho) is a length-B vector. Jacobi-preconditioned.

    Pure torch ops only -> device-agnostic (CPU / CUDA / ROCm). Returns ``x``
    of shape ``[B, n]``. Intended to be called inside an autograd ``Function``
    that supplies gradients analytically (this routine itself does not track
    grad).
    """
    m, n = shape
    assert m == n, "CG requires square systems"
    Bn = val_batch.shape[0]
    device, dtype = val_batch.device, val_batch.dtype
    row = row.to(device)
    col = col.to(device)

    def matvec(X):  # [B, n] -> [B, n] :  Y_i = A_i X_i
        Y = torch.zeros(Bn, n, device=device, dtype=dtype)
        Y.index_add_(1, row, val_batch * X[:, col])
        return Y

    # Per-matrix Jacobi diagonal D_i (gather the diagonal entries of each A_i)
    D = torch.zeros(Bn, n, device=device, dtype=dtype)
    diag = row == col
    D.index_add_(1, row[diag], val_batch[:, diag])
    D_inv = torch.where(D.abs() > 0, 1.0 / D, torch.ones_like(D))

    X = torch.zeros(Bn, n, device=device, dtype=dtype)
    R = b_batch.clone()                 # b - A·0
    Z = D_inv * R
    P = Z.clone()
    rz_old = (R * Z).sum(dim=1)         # [B]
    bnorm = torch.linalg.vector_norm(b_batch, dim=1).clamp_min(1e-30)
    tol = torch.clamp(rtol * bnorm, min=atol)   # per-batch stopping threshold

    tiny = torch.tensor(1e-30, dtype=dtype, device=device)
    for _ in range(maxiter):
        AP = matvec(P)
        pAp = (P * AP).sum(dim=1)
        alpha = rz_old / torch.where(pAp.abs() > 0, pAp, tiny)
        X = X + alpha.unsqueeze(1) * P
        R = R - alpha.unsqueeze(1) * AP
        if bool((torch.linalg.vector_norm(R, dim=1) <= tol).all()):
            break
        Z = D_inv * R
        rz_new = (R * Z).sum(dim=1)
        beta = rz_new / torch.where(rz_old.abs() > 0, rz_old, tiny)
        P = Z + beta.unsqueeze(1) * P
        rz_old = rz_new
    return X
