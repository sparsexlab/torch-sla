"""Linear-algebra ops for SparseTensor.

Solve / Krylov dispatchers + eigenvalues / SVD / det / LU.
"""
from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union, Literal
import torch

from .core import SparseTensor  # noqa: E402
from .autograd import DetAdjoint, EigshAdjoint, SparseSolveFunction
from ..backends import is_scipy_available
from ..backends.scipy_backend import scipy_svds
from .utils import _power_iteration_svd


def solve(
    self,
    b: torch.Tensor,
    backend: BackendType = "auto",
    method: MethodType = "auto",
    atol: float = 1e-10,
    maxiter: int = 10000,
    tol: float = 1e-12,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Solve the sparse linear system Ax = b.
    
    Automatically handles batched tensors: if A is [...batch, M, N] and
    b is [...batch, M], returns x with shape [...batch, N].
    
    Parameters
    ----------
    b : torch.Tensor
        Right-hand side vector(s). Shape:
        - Non-batched: [M] or [M, K] for multiple RHS
        - Batched: [...batch, M] or [...batch, M, K]
    backend : {"auto", "scipy", "eigen", "cupy", "cudss"}, optional
        Solver backend. Default: "auto" (selects based on device).
        - "scipy": Uses SciPy's sparse solvers (CPU only)
        - "eigen": Uses Eigen C++ library (CPU only)
        - "cupy": Uses CuPy's sparse solvers (CUDA only)
        - "cudss": Uses NVIDIA cuDSS (CUDA only)
    method : str, optional
        Solver method. Default: "auto" (selects based on matrix properties).
        - Direct methods: "lu", "umfpack", "cholesky", "ldlt"
        - Iterative methods: "cg", "bicgstab", "gmres", "minres"
    atol : float, optional
        Absolute tolerance for iterative solvers. Default: 1e-10.
    maxiter : int, optional
        Maximum iterations for iterative solvers. Default: 10000.
    tol : float, optional
        Relative tolerance for direct solvers. Default: 1e-12.
    verbose : bool, optional
        If True, print a one-line summary of the auto-selected backend,
        method, and detected matrix properties (symmetric, SPD).
        Default: False.

    Returns
    -------
    torch.Tensor
        Solution x with same batch shape as b.
    
    Raises
    ------
    ValueError
        If matrix is not square.
    NotImplementedError
        If block sparse tensors are used (not yet supported).
    
    Examples
    --------
    >>> # Simple solve
    >>> A = SparseTensor(val, row, col, (3, 3))
    >>> b = torch.randn(3)
    >>> x = A.solve(b)
    
    >>> # Batched solve
    >>> A_batch = SparseTensor(val_batch, row, col, (4, 3, 3))
    >>> b_batch = torch.randn(4, 3)
    >>> x_batch = A_batch.solve(b_batch)
    
    >>> # Specify backend
    >>> x = A.solve(b, backend='scipy', method='cg')
    """
    if not self.is_square:
        raise ValueError("Matrix must be square for solve()")

    if self.is_block:
        raise NotImplementedError("solve() not yet supported for block sparse tensors")

    # Get matrix properties
    is_sym = self.is_symmetric().all().item() if self.is_batched else self.is_symmetric().item()
    is_pd = self.is_positive_definite().all().item() if self.is_batched else self.is_positive_definite().item()
    is_spd = is_sym and is_pd

    from ..linear_solve import spsolve

    M, N = self.sparse_shape

    if self.is_batched:
        batch_shape = self.batch_shape
        vals_flat = self.values.reshape(-1, self.nnz)
        b_flat = b.reshape(-1, M)

        results = []
        for i in range(vals_flat.size(0)):
            # Only print on the first batch element to avoid spam
            x = spsolve(
                vals_flat[i], self.row_indices, self.col_indices,
                (M, N), b_flat[i],
                backend=backend, method=method,
                atol=atol, maxiter=maxiter, tol=tol,
                is_symmetric=is_sym, is_spd=is_spd,
                verbose=verbose and i == 0,
            )
            results.append(x)

        return torch.stack(results).reshape(*batch_shape, N)
    else:
        return spsolve(
            self.values, self.row_indices, self.col_indices,
            (M, N), b,
            backend=backend, method=method,
            atol=atol, maxiter=maxiter, tol=tol,
            is_symmetric=is_sym, is_spd=is_spd,
            verbose=verbose,
        )

def solve_batch(
    self,
    values: torch.Tensor,
    b: torch.Tensor,
    backend: BackendType = "auto",
    method: MethodType = "auto",
    atol: float = 1e-10,
    maxiter: int = 10000,
    tol: float = 1e-12
) -> torch.Tensor:
    """
    Solve with different values but same sparsity structure.
    
    This is efficient when you have the same structure but different values
    (e.g., time-stepping, optimization, parameter sweeps).
    
    Parameters
    ----------
    values : torch.Tensor
        Matrix values. Shape [...batch, nnz] where ... are batch dimensions.
        All matrices share the same row_indices and col_indices.
    b : torch.Tensor
        Right-hand side. Shape [...batch, M].
    backend : {"auto", "scipy", "eigen", "cupy", "cudss"}, optional
        Solver backend. See solve() for details. Default: "auto".
    method : str, optional
        Solver method. See solve() for details. Default: "auto".
    atol : float, optional
        Absolute tolerance for iterative solvers. Default: 1e-10.
    maxiter : int, optional
        Maximum iterations for iterative solvers. Default: 10000.
    tol : float, optional
        Relative tolerance. Default: 1e-12.
    
    Returns
    -------
    torch.Tensor
        Solution x with shape [...batch, N].
    
    Examples
    --------
    >>> # Template matrix
    >>> A = SparseTensor(val, row, col, (10, 10))
    
    >>> # Batch of different values
    >>> val_batch = torch.stack([val * (1 + 0.1*i) for i in range(4)])  # [4, nnz]
    >>> b_batch = torch.randn(4, 10)
    
    >>> # Solve all at once
    >>> x_batch = A.solve_batch(val_batch, b_batch)  # [4, 10]
    """
    from ..linear_solve import spsolve
    
    M, N = self.sparse_shape
    
    # Check properties using first batch element
    temp = SparseTensor(values[0] if values.dim() > 1 else values, 
                       self.row_indices, self.col_indices, (M, N))
    is_sym = temp.is_symmetric().item()
    is_pd = temp.is_positive_definite().item()
    is_spd = is_sym and is_pd
    
    if values.dim() > 1:
        batch_shape = values.shape[:-1]
        vals_flat = values.reshape(-1, self.nnz)
        b_flat = b.reshape(-1, M)
        
        results = []
        for i in range(vals_flat.size(0)):
            x = spsolve(
                vals_flat[i], self.row_indices, self.col_indices, (M, N), b_flat[i],
                backend=backend, method=method,
                atol=atol, maxiter=maxiter, tol=tol,
                is_symmetric=is_sym, is_spd=is_spd
            )
            results.append(x)
        
        return torch.stack(results).reshape(*batch_shape, N)
    else:
        return spsolve(
            values, self.row_indices, self.col_indices, (M, N), b,
            backend=backend, method=method,
            atol=atol, maxiter=maxiter, tol=tol,
            is_symmetric=is_sym, is_spd=is_spd
        )

def nonlinear_solve(
    self,
    residual_fn,
    u0: torch.Tensor,
    *params,
    method: Literal['newton', 'picard', 'anderson'] = 'newton',
    tol: float = 1e-6,
    atol: float = 1e-10,
    max_iter: int = 50,
    line_search: bool = True,
    verbose: bool = False,
    linear_solver: BackendType = 'pytorch',
    linear_method: MethodType = 'cg',
) -> torch.Tensor:
    """
    Solve nonlinear equation F(u, A, θ) = 0 with adjoint-based gradients.
    
    The SparseTensor A is automatically passed as the first parameter to
    the residual function, enabling gradients to flow through A's values.
    
    Parameters
    ----------
    residual_fn : Callable
        Function F(u, A, *params) -> residual tensor.
        - u: Current solution estimate
        - A: This SparseTensor (passed automatically)
        - *params: Additional parameters with requires_grad=True
    u0 : torch.Tensor
        Initial guess for solution.
    *params : torch.Tensor
        Additional parameters (e.g., boundary conditions, coefficients).
        Tensors with requires_grad=True will receive gradients.
    method : {'newton', 'picard', 'anderson'}, optional
        Nonlinear solver method:
        - 'newton': Newton-Raphson with line search (default, fast)
        - 'picard': Fixed-point iteration (simple, slow)
        - 'anderson': Anderson acceleration (memory efficient)
    tol : float, optional
        Relative convergence tolerance. Default: 1e-6.
    atol : float, optional
        Absolute convergence tolerance. Default: 1e-10.
    max_iter : int, optional
        Maximum nonlinear iterations. Default: 50.
    line_search : bool, optional
        Use Armijo line search for Newton. Default: True.
    verbose : bool, optional
        Print convergence information. Default: False.
    linear_solver : str, optional
        Backend for linear solves. Default: 'pytorch'.
    linear_method : str, optional
        Method for linear solves. Default: 'cg'.
    
    Returns
    -------
    torch.Tensor
        Solution u* satisfying F(u*, A, θ) ≈ 0.
    
    Examples
    --------
    >>> # Nonlinear PDE: A @ u + u² = f
    >>> def residual(u, A, f):
    ...     return A @ u + u**2 - f
    ...
    >>> A = SparseTensor(val, row, col, (n, n))
    >>> f = torch.randn(n, requires_grad=True)
    >>> u0 = torch.zeros(n)
    >>> 
    >>> u = A.nonlinear_solve(residual, u0, f, method='newton')
    >>> 
    >>> # Gradients flow via adjoint method
    >>> loss = u.sum()
    >>> loss.backward()
    >>> print(f.grad)  # ∂u/∂f
    >>> print(A.values.grad)  # ∂u/∂A (if A.values.requires_grad)
    
    >>> # Nonlinear elasticity: K(u) @ u = F
    >>> def residual_elasticity(u, K, F, material):
    ...     # K depends on displacement through material nonlinearity
    ...     return K @ u - F + material * u**3
    ...
    >>> u = K.nonlinear_solve(residual_elasticity, u0, F, material)
    """
    from ..nonlinear_solve import nonlinear_solve as _nonlinear_solve
    
    # Wrap residual_fn to pass SparseTensor as matvec
    M, N = self.sparse_shape
    
    def wrapped_residual(u, *all_params):
        # First param is the values tensor, rest are user params
        # Reconstruct sparse matvec capability
        return residual_fn(u, self, *all_params)
    
    # Include self.values in params if it requires grad
    all_params = params
    
    return _nonlinear_solve(
        wrapped_residual, u0, *all_params,
        method=method, tol=tol, atol=atol, max_iter=max_iter,
        line_search=line_search, verbose=verbose,
        linear_solver=linear_solver, linear_method=linear_method,
        )

def eigs(
    self,
    k: int = 6,
    which: str = "LM",
    sigma: Optional[float] = None,
    return_eigenvectors: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute k eigenvalues and eigenvectors.
    
    For batched tensors, computes for each batch element.
    For CUDA tensors, uses LOBPCG algorithm.
    
    Parameters
    ----------
    k : int, optional
        Number of eigenvalues to compute. Default: 6.
    which : {"LM", "SM", "LR", "SR", "LA", "SA"}, optional
        Which eigenvalues to find:
        - "LM": Largest magnitude (default)
        - "SM": Smallest magnitude
        - "LR"/"SR": Largest/smallest real part
        - "LA"/"SA": Largest/smallest algebraic (for symmetric)
    sigma : float, optional
        Find eigenvalues near sigma (shift-invert mode).
    return_eigenvectors : bool, optional
        Whether to return eigenvectors. Default: True.
        
    Returns
    -------
    eigenvalues : torch.Tensor
        Shape [k] for non-batched, [*batch_shape, k] for batched.
    eigenvectors : torch.Tensor or None
        Shape [M, k] for non-batched, [*batch_shape, M, k] for batched.
        None if return_eigenvectors is False.
    
    Notes
    -----
    **Gradient Support:**
    
    - Both CPU and CUDA: Fully differentiable via adjoint method
    - Uses O(1) graph nodes regardless of iteration count
    - For symmetric matrices, prefer eigsh() for efficiency
    
    **Warning**: For non-symmetric matrices with complex eigenvalues,
    gradient computation is only supported for the real part.
    
    Examples
    --------
    >>> A = SparseTensor(val.requires_grad_(True), row, col, (n, n))
    >>> eigenvalues, eigenvectors = A.eigs(k=3)
    >>> loss = eigenvalues.real.sum()  # For complex eigenvalues
    >>> loss.backward()
    """
    M, N = self.sparse_shape
    
    if self.is_batched:
        batch_shape = self.batch_shape
        eigenvalues_list = []
        eigenvectors_list = []
        
        for idx in self._batch_indices():
            A_single = SparseTensor(
                self.values[idx], self.row_indices, self.col_indices, (M, N)
            )
            evals, evecs = A_single.eigs(k, which, sigma, return_eigenvectors)
            eigenvalues_list.append(evals)
            if return_eigenvectors:
                eigenvectors_list.append(evecs)
        
        eigenvalues = torch.stack(eigenvalues_list).reshape(*batch_shape, k)
        if return_eigenvectors:
            eigenvectors = torch.stack(eigenvectors_list).reshape(*batch_shape, M, k)
            return eigenvalues, eigenvectors
        return eigenvalues, None
    
    # For symmetric matrices or when using LA/SA, use eigsh (more efficient)
    if which in ("LA", "SA") or self.is_symmetric().item():
        return self.eigsh(k=k, which=which, sigma=sigma, return_eigenvectors=return_eigenvectors)
    
    # Use adjoint-based eigs for differentiability on all devices
    return EigshAdjoint.apply(
        self.values, self.row_indices, self.col_indices, (M, N),
        k, which, return_eigenvectors, self.device
    )

def eigsh(
    self,
    k: int = 6,
    which: str = "LM",
    sigma: Optional[float] = None,
    return_eigenvectors: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute k eigenvalues for symmetric matrices.
    
    More efficient than eigs() for symmetric matrices.
    
    Parameters
    ----------
    k : int, optional
        Number of eigenvalues to compute. Default: 6.
    which : {"LM", "SM", "LA", "SA"}, optional
        Which eigenvalues to find:
        - "LM": Largest magnitude (default)
        - "SM": Smallest magnitude
        - "LA"/"SA": Largest/smallest algebraic
    sigma : float, optional
        Find eigenvalues near sigma.
    return_eigenvectors : bool, optional
        Whether to return eigenvectors. Default: True.
        
    Returns
    -------
    eigenvalues : torch.Tensor
        Shape [k] for non-batched, [*batch_shape, k] for batched.
    eigenvectors : torch.Tensor or None
        Shape [M, k] for non-batched, [*batch_shape, M, k] for batched.
    
    Notes
    -----
    **Gradient Support:**
    
    - Both CPU and CUDA: Fully differentiable via adjoint method
    - Uses O(1) graph nodes regardless of iteration count
    - Gradient computed as: ∂L/∂A = Σ_i (∂L/∂λ_i) * v_i @ v_i.T
    
    Examples
    --------
    >>> A = SparseTensor(val.requires_grad_(True), row, col, (n, n))
    >>> eigenvalues, eigenvectors = A.eigsh(k=3)
    >>> loss = eigenvalues.sum()
    >>> loss.backward()  # Computes ∂loss/∂val
    """
    M, N = self.sparse_shape
    
    if self.is_batched:
        batch_shape = self.batch_shape
        eigenvalues_list = []
        eigenvectors_list = []
        
        for idx in self._batch_indices():
            A_single = SparseTensor(
                self.values[idx], self.row_indices, self.col_indices, (M, N)
            )
            evals, evecs = A_single.eigsh(k, which, sigma, return_eigenvectors)
            eigenvalues_list.append(evals)
            if return_eigenvectors:
                eigenvectors_list.append(evecs)
        
        eigenvalues = torch.stack(eigenvalues_list).reshape(*batch_shape, k)
        if return_eigenvectors:
            eigenvectors = torch.stack(eigenvectors_list).reshape(*batch_shape, M, k)
            return eigenvalues, eigenvectors
        return eigenvalues, None
    
    # Use adjoint-based eigsh for differentiability on all devices
    return EigshAdjoint.apply(
        self.values, self.row_indices, self.col_indices, (M, N),
        k, which, return_eigenvectors, self.device
    )

def svd(self, k: int = 6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute truncated SVD.
    
    Parameters
    ----------
    k : int, optional
        Number of singular values to compute. Default: 6.
        
    Returns
    -------
    U : torch.Tensor
        Left singular vectors. Shape [M, k] or [*batch_shape, M, k].
    S : torch.Tensor
        Singular values. Shape [k] or [*batch_shape, k].
    Vt : torch.Tensor
        Right singular vectors. Shape [k, N] or [*batch_shape, k, N].
    
    Notes
    -----
    **Gradient Support:**
    
    - CUDA: Fully differentiable (uses power iteration with PyTorch operations)
    - CPU: NOT differentiable (uses SciPy which breaks gradient chain)
    
    For differentiable SVD on CPU, use `A.to_dense()` and `torch.linalg.svd()`.
    """
    M, N = self.sparse_shape

    if self.is_batched:
        batch_shape = self.batch_shape
        U_list, S_list, Vt_list = [], [], []
        
        for idx in self._batch_indices():
            A_single = SparseTensor(
                self.values[idx], self.row_indices, self.col_indices, (M, N)
            )
            U, S, Vt = A_single.svd(k)
            U_list.append(U)
            S_list.append(S)
            Vt_list.append(Vt)
        
        U = torch.stack(U_list).reshape(*batch_shape, M, k)
        S = torch.stack(S_list).reshape(*batch_shape, k)
        Vt = torch.stack(Vt_list).reshape(*batch_shape, k, N)
        return U, S, Vt
    
    from .autograd import SvdAdjoint

    # Build the device-appropriate detached forward + sparse-pattern
    # gather, then run through SvdAdjoint so the result is a
    # differentiable tuple (single Function node, regardless of how
    # many iterations the inner solver took).
    if self.is_cuda:
        def _svd_forward(val_det, row, col, shape, kk):
            from .core import SparseTensor as _ST
            A = _ST(val_det, row, col, shape)
            matvec = lambda x: A._spmv_coo(x)
            matvec_T = lambda x: A.T()._spmv_coo(x)
            return _power_iteration_svd(
                matvec, matvec_T, shape[0], shape[1], kk,
                val_det.dtype, val_det.device,
            )
    elif is_scipy_available():
        def _svd_forward(val_det, row, col, shape, kk):
            U, S, Vt = scipy_svds(val_det, row, col, shape, k=kk)
            return U.to(val_det.device), S.to(val_det.device), Vt.to(val_det.device)
    else:
        raise RuntimeError("SciPy is required for SVD on CPU")

    row_i64 = self.row_indices.to(torch.int64)
    col_i64 = self.col_indices.to(torch.int64)

    def _gather_sv_grad(U, S, Vt, grad_S):
        # grad_val[k] = Σ_i grad_S[i] * U[row[k], i] * Vt[i, col[k]]
        U_at_row = U[row_i64]              # (nnz, k)
        Vt_at_col = Vt.t()[col_i64]        # (nnz, k)  -- Vt^T is V
        return (U_at_row * Vt_at_col * grad_S.unsqueeze(0)).sum(dim=1)

    U, S, Vt = SvdAdjoint.apply(
        self.values, self.row_indices, self.col_indices,
        (M, N), k, _svd_forward, _gather_sv_grad,
    )
    return U, S, Vt

def condition_number(self, ord: int = 2) -> torch.Tensor:
    """
    Estimate condition number.
    
    Parameters
    ----------
    ord : int, optional
        Norm order for condition number. Default: 2 (spectral).
        
    Returns
    -------
    torch.Tensor
        Condition number. Shape [] or [*batch_shape].
    """
    M, N = self.sparse_shape
    
    if self.is_batched:
        batch_shape = self.batch_shape
        cond_list = []
        
        for idx in self._batch_indices():
            A_single = SparseTensor(
                self.values[idx], self.row_indices, self.col_indices, (M, N)
            )
            cond_list.append(A_single.condition_number(ord))
        
        return torch.stack(cond_list).reshape(*batch_shape)
    
    if ord == 2:
        k = min(6, min(M, N) - 2)
        if k < 2:
            A_dense = self.to_dense()
            S = torch.linalg.svdvals(A_dense)
            return S.max() / S.min()
        _, S, _ = self.svd(k=k)
        return S.max() / S.min()
    
    norm_A = self.norm(ord=ord)
    e = torch.randn(M, dtype=self.dtype, device=self.device)
    e = e / e.norm()
    x = self.solve(e)
    return norm_A * x.norm() / e.norm()

def det(self) -> torch.Tensor:
    """
    Compute determinant of the sparse matrix with gradient support.
    
    Uses LU decomposition (CPU) or dense conversion (CUDA) to compute 
    the determinant efficiently. Supports automatic differentiation via
    the adjoint method.
    
    Returns
    -------
    torch.Tensor
        Determinant value. Shape [] for single matrix or [*batch_shape] for batched.
        
    Raises
    ------
    ValueError
        If matrix is not square
        
    Notes
    -----
    - Only square matrices have determinants
    - For large matrices, determinant values can overflow/underflow
    - Consider using log-determinant for numerical stability in such cases
    - Supports both CPU (via SciPy) and CUDA (via torch.linalg.det)
    - For batched tensors, computes determinant independently for each batch
    - Fully differentiable: gradients computed via adjoint method
    - Gradient formula: ∂det(A)/∂A = det(A) * (A^{-1})^T
    
    Performance Warning
    -------------------
    **CUDA performance is significantly slower than CPU for sparse matrices!**
    
    - CPU: Uses sparse LU decomposition (O(nnz^1.5)), ~0.3-0.8ms for n=10-1000
    - CUDA: Converts to dense (O(n²) memory + O(n³) compute), ~0.2-2.5ms
    
    The CUDA version requires converting the sparse matrix to dense format
    because cuDSS doesn't expose determinant computation for sparse
    matrices. This makes it inefficient for large sparse matrices.
    
    **Recommendation**: For sparse matrices, use `.cpu().det().cuda()` instead:
    
    >>> # Slow: CUDA with dense conversion
    >>> det_slow = A_cuda.det()  # ~2.5ms for n=1000
    >>> 
    >>> # Fast: CPU with sparse LU
    >>> det_fast = A_cuda.cpu().det()  # ~0.8ms for n=1000
    >>> det_fast = det_fast.cuda()  # Move result back if needed
    
    Examples
    --------
    >>> # Simple 2x2 matrix
    >>> val = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    >>> row = torch.tensor([0, 0, 1, 1])
    >>> col = torch.tensor([0, 1, 0, 1])
    >>> A = SparseTensor(val, row, col, (2, 2))
    >>> det = A.det()
    >>> print(det)  # Should be -2.0
    >>> det.backward()
    >>> print(val.grad)  # Gradient w.r.t. matrix values
    >>>
    >>> # CUDA support
    >>> A_cuda = A.cuda()
    >>> det_cuda = A_cuda.det()
    >>>
    >>> # Batched matrices
    >>> val_batch = val.unsqueeze(0).expand(3, -1).clone()
    >>> A_batch = SparseTensor(val_batch, row, col, (3, 2, 2))
    >>> det_batch = A_batch.det()
    >>> print(det_batch.shape)  # torch.Size([3])
    """
    M, N = self.sparse_shape

    if M != N:
        raise ValueError(f"Matrix must be square for determinant, got shape ({M}, {N})")

    # Block-dim det: per-block scalar det of each K x K dense block.
    # Sparse pattern stays the same; block_shape disappears; values
    # collapse from ``[..., nnz, K, K]`` to ``[..., nnz]``. The full
    # matrix's "sparse det" is not computed here -- this is the
    # block-local det that's useful for FEM Jacobian determinants and
    # other block-structured factors.
    if self.is_block:
        if self.block_shape[-2] != self.block_shape[-1]:
            raise ValueError(
                f"block-dim det requires square block, got {self.block_shape}")
        new_values = torch.linalg.det(self.values)  # broadcasts leading dims
        new_shape = (*self.batch_shape, *self.sparse_shape)
        return SparseTensor(
            new_values, self.row_indices, self.col_indices, new_shape,
        )

    if self.is_batched:
        batch_shape = self.batch_shape
        det_list = []

        for idx in self._batch_indices():
            A_single = SparseTensor(
                self.values[idx], self.row_indices, self.col_indices, (M, N)
            )
            det_list.append(A_single.det())

        return torch.stack(det_list).reshape(*batch_shape)

    # Sparse-dim det -- adjoint path; forward routes through DetConfig
    # dispatcher (see torch_sla/det.py).
    return DetAdjoint.apply(
        self.values,
        self.row_indices,
        self.col_indices,
        (M, N),
        self.device,
        self.is_cuda
    )


def logdet(self, **kwargs) -> torch.Tensor:
    """Log-determinant of this matrix. See :mod:`torch_sla.det`.

    For large SPD matrices, the default ``method='auto'`` selects the
    Hutchinson stochastic estimator (pure matvec, distributed-friendly).
    Smaller matrices use Cholesky / LU.
    """
    from ..det import logdet as _logdet
    return _logdet(self, **kwargs)

def lu(self) -> "LUFactorization":
    """
    Compute LU decomposition for repeated solves.
    
    Returns
    -------
    LUFactorization
        Factorization object with solve() method.
    
    Examples
    --------
    >>> A = SparseTensor(val, row, col, (10, 10))
    >>> lu = A.lu()
    >>> x1 = lu.solve(b1)
    >>> x2 = lu.solve(b2)  # Reuses factorization
    """
    if self.is_batched:
        raise NotImplementedError("lu() not supported for batched tensors")
    
    if self.is_cuda:
        raise NotImplementedError("LU decomposition on CUDA not yet supported")
    
    if not is_scipy_available():
        raise RuntimeError("SciPy is required for LU decomposition")
    
    M, N = self.sparse_shape
    lu = scipy_lu(self.values, self.row_indices, self.col_indices, (M, N))
    return LUFactorization(lu, (M, N), self.dtype, self.device)

