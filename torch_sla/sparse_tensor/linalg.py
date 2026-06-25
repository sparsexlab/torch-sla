"""Linear-algebra ops for SparseTensor.

Solve / Krylov dispatchers + eigenvalues / SVD / det / LU.
"""
from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union, Literal
import torch

from .core import SparseTensor, LUFactorization  # noqa: E402
from .autograd import DetAdjoint, EigshAdjoint, SparseSolveFunction
from ..backends import is_scipy_available
from ..backends.scipy_backend import scipy_svds, scipy_lu
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
    r"""
    Solve the sparse linear system :math:`Ax = b`.

    .. math::

        A x = b \quad\Longrightarrow\quad x = A^{-1} b

    Automatically handles batched tensors: if A is [...batch, M, N] and
    b is [...batch, M], returns x with shape [...batch, N].

    Algorithm
    ---------
    Dispatches (via :func:`torch_sla.linear_solve.spsolve`) to either a
    **direct** factorization (sparse LU / Cholesky / LDLᵀ) or a **Krylov**
    iterative method (CG for SPD, MINRES for symmetric-indefinite,
    BiCGStab / GMRES for general). ``method="auto"`` inspects symmetry and
    positive-definiteness to pick one. Krylov CG sketch:

    .. code-block:: text

        r = b - A @ x;  p = r;  rs = <r, r>
        repeat until ||r|| <= atol:
            Ap   = A @ p                 # one sparse matvec / iteration
            alpha = rs / <p, Ap>
            x    += alpha * p
            r    -= alpha * Ap
            rs_new = <r, r>
            p    = r + (rs_new / rs) * p
            rs   = rs_new

    Complexity
    ----------
    Krylov: time :math:`O(m \cdot nnz)` for ``m`` iterations, space
    :math:`O(n + nnz)`. Direct factorization: time between
    :math:`O(n^{1.5})` (2-D) and :math:`O(n^{2})` (3-D), space
    :math:`O(n\log n)` to :math:`O(n^{4/3})` for the fill-in factors.

    Backward
    --------
    Differentiable via the **adjoint method**: the backward pass solves
    :math:`A^{H}\lambda = \partial L/\partial x` (reusing the same
    factorization / Krylov operator) and accumulates
    :math:`\partial L/\partial A = -\lambda\, x^{H}`. This adds only
    :math:`O(1)` autograd graph nodes regardless of the iteration count
    ``m`` (one extra solve, not ``m`` recorded matvecs).

    Parameters
    ----------
    b : torch.Tensor
        Right-hand side vector(s). Shape:
        - Non-batched: [M] or [M, K] for multiple RHS
        - Batched: [...batch, M] or [...batch, M, K]
    backend : {"auto", "scipy", "pytorch", "cudss"}, optional
        Solver backend. Default: "auto" (selects based on device).
        - "scipy": Uses SciPy's sparse solvers (CPU only)
        - "pytorch": PyTorch-native iterative solvers (CPU & CUDA)
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
    >>> import torch
    >>> from torch_sla import SparseTensor
    >>> # 3x3 SPD matrix  A = diag(2,3,4)  (stored as 3 diagonal entries)
    >>> val = torch.tensor([2.0, 3.0, 4.0])
    >>> row = torch.tensor([0, 1, 2])
    >>> col = torch.tensor([0, 1, 2])
    >>> A = SparseTensor(val, row, col, (3, 3))
    >>> b = torch.tensor([2.0, 3.0, 4.0])
    >>> A.solve(b)
    tensor([1., 1., 1.])

    >>> # Batched solve: A is [4, 3, 3], b is [4, 3] -> x is [4, 3]
    >>> A_batch = SparseTensor(val.expand(4, 3).clone(), row, col, (4, 3, 3))
    >>> x_batch = A_batch.solve(b.expand(4, 3).clone())
    >>> x_batch.shape
    torch.Size([4, 3])

    >>> # Specify backend / method explicitly
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
    r"""
    Solve with different values but same sparsity structure.

    .. math::

        A_i x_i = b_i,\quad i = 1\ldots B
        \qquad\text{with } \mathrm{pattern}(A_i)\equiv(\text{row},\text{col})

    This is efficient when you have the same structure but different values
    (e.g., time-stepping, optimization, parameter sweeps): the symbolic
    factorization / sparsity analysis is shared across the batch.

    Algorithm
    ---------
    Detect symmetry/SPD once from the first batch element, then loop over
    the ``B`` value-vectors, calling :func:`spsolve` per system (each reuses
    the same ``row``/``col`` indices):

    .. code-block:: text

        props = analyze(values[0], row, col)      # symmetry / SPD, once
        for i in 1..B:
            x[i] = spsolve(values[i], row, col, b[i], props)

    Complexity
    ----------
    ``B`` independent solves: time :math:`O(B \cdot m \cdot nnz)` (Krylov)
    or :math:`O(B \cdot n^{1.5..2})` (direct); space :math:`O(n + nnz)` per
    system. Backward mirrors :meth:`solve` (one adjoint solve per system,
    :math:`O(1)` extra graph nodes each).

    Parameters
    ----------
    values : torch.Tensor
        Matrix values. Shape [...batch, nnz] where ... are batch dimensions.
        All matrices share the same row_indices and col_indices.
    b : torch.Tensor
        Right-hand side. Shape [...batch, M].
    backend : {"auto", "scipy", "pytorch", "cudss"}, optional
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
    jac_fn=None,
    method: Literal['newton'] = 'newton',
    tol: float = 1e-8,
    atol: float = 1e-12,
    max_iter: int = 50,
    line_search: bool = True,
    verbose: bool = False,
    linear_solver: BackendType = 'auto',
    linear_method: MethodType = 'auto',
) -> torch.Tensor:
    r"""
    Solve nonlinear equation :math:`F(u, A, \theta) = 0` with adjoint gradients.

    .. math::

        F(u^\*, A, \theta) = 0,\qquad
        u_{n+1} = u_n - J(u_n)^{-1} F(u_n),\quad
        J = \frac{\partial F}{\partial u}

    The SparseTensor A is automatically passed as the first parameter to
    the residual function, enabling gradients to flow through A's values.

    Algorithm
    ---------
    **Newton-Raphson**: at each step assemble the sparse Jacobian
    :math:`J = \partial F/\partial u` (explicit ``jac_fn`` or
    ``torch.autograd.functional.jacobian``) and solve the linear update via
    :func:`spsolve`; an optional Armijo line search damps the step.

    .. code-block:: text

        u = u0
        for it in 1..max_iter:
            F = residual(u, A, *params)
            if ||F|| <= atol or ||F|| <= tol*||F0||: break
            J  = jacobian(F, u)            # sparse dF/du
            du = spsolve(J, -F)            # Newton direction
            a  = armijo_line_search(u, du) # if line_search
            u  = u + a * du

    Complexity
    ----------
    Time :math:`O(m \cdot \text{solve})` for ``m`` Newton steps, where each
    ``solve`` is one sparse linear solve (Krylov :math:`O(m'\,nnz)` or
    direct); space :math:`O(\text{solve})`, i.e. that of a single linear
    solve plus the Jacobian.

    Backward
    --------
    Gradients use the **implicit-function theorem** (not unrolled Newton):
    backward solves one adjoint system :math:`J^{H}\lambda = \partial L/\partial u`
    at the converged :math:`u^\*` and back-propagates
    :math:`\partial L/\partial(\cdot) = -\lambda^{H}\,\partial F/\partial(\cdot)`.
    This is :math:`O(1)` extra graph nodes regardless of ``m``.

    Parameters
    ----------
    residual_fn : Callable
        Function ``F(u, A, *params)`` -> residual tensor.
        - u: Current solution estimate
        - A: This SparseTensor (passed automatically)
        - ``*params``: Additional parameters with requires_grad=True
    u0 : torch.Tensor
        Initial guess for solution.
    *params : torch.Tensor
        Additional parameters (e.g., boundary conditions, coefficients).
        Tensors with requires_grad=True will receive gradients.
    jac_fn : Callable, optional
        Optional explicit Jacobian ``J(u, A, *params) -> (val, row, col, shape)``
        returning the sparse dF/du in COO form. If ``None`` (default) the
        Jacobian is obtained via ``torch.autograd.functional.jacobian``.
    method : {'newton'}, optional
        Nonlinear solver method. Only Newton-Raphson (with implicit-diff
        gradients) is supported; each step solves the sparse Jacobian
        system via ``spsolve``.
    tol : float, optional
        Relative convergence tolerance on ``||F||``. Default: 1e-8.
    atol : float, optional
        Absolute convergence tolerance on ``||F||``. Default: 1e-12.
    max_iter : int, optional
        Maximum nonlinear iterations. Default: 50.
    line_search : bool, optional
        Use Armijo line search for Newton. Default: True.
    verbose : bool, optional
        Print convergence information. Default: False.
    linear_solver : str, optional
        Backend for the linear (Jacobian) solves. Default: 'auto'.
    linear_method : str, optional
        Method for the linear (Jacobian) solves. Default: 'auto'. Note: the
        Jacobian of a general nonlinear residual is NOT symmetric, so a
        direct method (e.g. 'lu') is recommended over 'cg'.
    
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
    if method != 'newton':
        raise ValueError(
            f"nonlinear_solve only supports method='newton', got {method!r}")
    if self.is_batched:
        raise NotImplementedError(
            "nonlinear_solve not supported for batched SparseTensors")

    from .autograd import NonlinearSolveFunction

    M, N = self.sparse_shape
    row, col = self.row_indices, self.col_indices

    # Residual seen by the autograd Function takes A's *values* as an
    # explicit tensor argument so gradients can flow back to A. We rebuild
    # the SparseTensor (sharing row/col/shape) inside the closure so the
    # user still writes ``residual_fn(u, A, *params)``.
    def residual_with_val(u, val, *user_params):
        A = SparseTensor(val, row, col, (M, N))
        return residual_fn(u, A, *user_params)

    jac_with_val = None
    if jac_fn is not None:
        def jac_with_val(u, val, *user_params):
            A = SparseTensor(val.detach(), row, col, (M, N))
            return jac_fn(u, A, *user_params)

    return NonlinearSolveFunction.apply(
        u0, len(params), residual_with_val, jac_with_val,
        row, col, (M, N),
        tol, atol, max_iter, line_search, verbose,
        linear_solver, linear_method,
        self.values, *params,
    )

def eigs(
    self,
    k: int = 6,
    which: str = "LM",
    sigma: Optional[float] = None,
    return_eigenvectors: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""
    Compute k eigenvalues and eigenvectors of a (general) sparse matrix.

    .. math::

        A v_i = \lambda_i v_i,\qquad i = 1\ldots k

    For batched tensors, computes for each batch element. Symmetric
    matrices (and ``which`` in ``{"LA","SA"}``) are routed to the more
    efficient :meth:`eigsh`.

    Algorithm
    ---------
    A subspace iteration (**LOBPCG** for symmetric / Hermitian operators;
    see :func:`_lobpcg_core`) projects ``A`` onto a small search space and
    repeatedly solves the tiny dense Rayleigh-Ritz problem:

    .. code-block:: text

        X = orthonormal random block, columns ~ k
        repeat until Ritz residual small:
            AX = A @ X                       # k sparse matvecs
            H  = X^H A X                     # small (m x m) Gram matrix
            (theta, C) = eigh(H)             # dense Rayleigh-Ritz
            X  = orthonormalize([X | (AX - X theta) | P])  # X | residual | conj-dir
        return theta[:k], X[:k]

    Complexity
    ----------
    Time :math:`O(m\,k\,nnz)` (``m`` outer iterations, ``k`` matvecs each);
    space :math:`O(k n + nnz)` for the block of ``k`` Ritz vectors plus the
    matrix.

    Backward
    --------
    Differentiable via the **adjoint method** with
    :math:`\partial L/\partial A = \sum_i (\partial L/\partial\lambda_i)\,
    v_i v_i^{H}` (eigenvalue part); :math:`O(1)` extra graph nodes,
    independent of the iteration count. For complex eigenvalues only the
    real part is differentiable.

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
        Shape [k] for non-batched, ``[*batch_shape, k]`` for batched.
    eigenvectors : torch.Tensor or None
        Shape [M, k] for non-batched, ``[*batch_shape, M, k]`` for batched.
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
    >>> import torch
    >>> from torch_sla import SparseTensor
    >>> # A = diag(1, 2, 3, 4): eigenvalues are the diagonal entries
    >>> d = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    >>> idx = torch.arange(4)
    >>> A = SparseTensor(d, idx, idx, (4, 4))
    >>> evals, evecs = A.eigs(k=2, which="LM")  # two largest-magnitude
    >>> evals.detach().sort().values
    tensor([3., 4.])
    >>> evals.real.sum().backward()  # differentiable w.r.t. A.values
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
    r"""
    Compute k eigenvalues for symmetric / Hermitian matrices.

    .. math::

        A v_i = \lambda_i v_i,\quad A = A^{H},\qquad i = 1\ldots k

    More efficient than :meth:`eigs` for symmetric matrices: it exploits
    :math:`A=A^{H}` so all Ritz values are real and a single ``eigh`` per
    iteration suffices.

    Algorithm
    ---------
    **LOBPCG** (Knyazev 2001; see :func:`_lobpcg_core`) -- locally optimal
    block preconditioned conjugate gradient. Maintains a 3-block subspace
    ``[X | residual | conjugate-direction]`` and minimizes the Rayleigh
    quotient over it each step:

    .. code-block:: text

        X = orthonormal random block (m >= k columns)
        repeat until ||A x_i - lambda_i x_i|| <= tol*|lambda_i|:
            theta, X = rayleigh_ritz(A, [X | R | P])  # dense eigh on Gram
            R = A @ X - X * theta                     # block residual
            P = conjugate direction (Knyazev eq. 7)
        return theta[:k], X[:k]

    Complexity
    ----------
    Time :math:`O(m\,k\,nnz)`; space :math:`O(k n + nnz)`.

    Backward
    --------
    Adjoint method:
    :math:`\partial L/\partial A = \sum_i (\partial L/\partial\lambda_i)\,
    v_i v_i^{\top}`; :math:`O(1)` extra graph nodes regardless of the
    iteration count.

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
        Shape [k] for non-batched, ``[*batch_shape, k]`` for batched.
    eigenvectors : torch.Tensor or None
        Shape [M, k] for non-batched, ``[*batch_shape, M, k]`` for batched.
    
    Notes
    -----
    **Device support:** CPU and CUDA are first-class. MPS is not
    recommended: PyTorch's MPS backend forces float32 (caps Ritz
    residual at ~1e-4..1e-3 on PDE-like operators) and is missing
    native ``linalg.eigh`` / efficient tall-skinny ``linalg.qr``
    kernels, so the core internally round-trips both to CPU as a
    workaround. The fallback works but most of the LOBPCG work
    actually happens on CPU; a ``RuntimeWarning`` fires on every
    MPS call to point users at ``device='cpu'`` or ``'cuda'``.

    **Gradient Support:**

    - Both CPU and CUDA: Fully differentiable via adjoint method
    - Uses O(1) graph nodes regardless of iteration count
    - Gradient computed as: ∂L/∂A = Σ_i (∂L/∂λ_i) * v_i @ v_i.T
    
    Examples
    --------
    >>> import torch
    >>> from torch_sla import SparseTensor
    >>> # Symmetric A = diag(1, 2, 3, 4)
    >>> d = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    >>> idx = torch.arange(4)
    >>> A = SparseTensor(d, idx, idx, (4, 4))
    >>> evals, evecs = A.eigsh(k=2, which="SA")  # two smallest algebraic
    >>> evals.detach().sort().values
    tensor([1., 2.])
    >>> evals.sum().backward()   # computes d(loss)/d(A.values)
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
    r"""
    Compute truncated (rank-``k``) singular value decomposition.

    .. math::

        A \approx U \Sigma V^{\top},\qquad
        \Sigma = \mathrm{diag}(\sigma_1 \ge \cdots \ge \sigma_k \ge 0)

    Algorithm
    ---------
    A **Lanczos bidiagonalization** (``scipy.sparse.linalg.svds`` / ARPACK
    on CPU) builds a Krylov subspace from sparse matvecs against ``A`` and
    ``Aᵀ`` and extracts the ``k`` dominant singular triplets from the small
    bidiagonal factor:

    .. code-block:: text

        build Krylov basis via alternating  v -> A v,  u -> A^T u
        B = bidiagonal projection of A onto that basis
        (U_b, S, V_b) = dense_svd(B)         # small problem
        U = Q_left @ U_b;  V = Q_right @ V_b
        return U[:, :k], S[:k], V[:, :k]^T

    Complexity
    ----------
    Time :math:`O(m\,k\,nnz)` (``m`` Lanczos steps); space
    :math:`O(k n + nnz)`.

    Backward
    --------
    Adjoint method: the singular-value gradient flows through the stored
    pattern as
    :math:`\partial L/\partial A_{rc} = \sum_i (\partial L/\partial\sigma_i)\,
    U_{r i} V_{c i}`; :math:`O(1)` extra graph nodes.

    Parameters
    ----------
    k : int, optional
        Number of singular values to compute. Default: 6.

    Returns
    -------
    U : torch.Tensor
        Left singular vectors. Shape [M, k] or ``[*batch_shape, M, k]``.
    S : torch.Tensor
        Singular values. Shape [k] or ``[*batch_shape, k]``.
    Vt : torch.Tensor
        Right singular vectors. Shape [k, N] or ``[*batch_shape, k, N]``.

    Notes
    -----
    **Gradient Support:**

    - CUDA: not currently supported (raises ``NotImplementedError``; move
      to CPU first).
    - CPU: singular values are differentiable via the adjoint above; the
      singular *vectors* break the gradient chain (SciPy forward).

    For fully differentiable SVD on CPU, use ``A.to_dense()`` and
    ``torch.linalg.svd()``.

    Examples
    --------
    >>> import torch
    >>> from torch_sla import SparseTensor
    >>> # A = diag(4, 1, 3): singular values are |diagonal|, sorted desc
    >>> d = torch.tensor([4.0, 1.0, 3.0])
    >>> idx = torch.arange(3)
    >>> A = SparseTensor(d, idx, idx, (3, 3))
    >>> U, S, Vt = A.svd(k=2)
    >>> S.detach().round().sort().values
    tensor([3., 4.])
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
        # No CUDA-native sparse SVD available that's both correct on
        # clustered spectra and not a multi-GB CPU<->GPU round-trip:
        #   * power iteration (was the old default) -- wrong σ on
        #     near-degenerate clusters (Halko-Martinsson undershoot).
        #   * torch.svd_lowrank -- same randomised-undershoot issue.
        #   * cuSOLVER / nvmath-python -- no sparse SVD routine exists.
        # True CUDA-native Lanczos bidiagonalisation (Golub-Kahan with
        # partial reorthogonalisation, PROPACK-style) is tracked as a
        # follow-up. Until then, ask the user to move the matrix to
        # CPU explicitly.
        raise NotImplementedError(
            "SparseTensor.svd() on CUDA is not currently supported. "
            "Workaround: move to CPU first.\n"
            "  U, S, Vt = A.cpu().svd(k=k)\n"
            "  if you need them back on GPU: U = U.cuda(); S = S.cuda(); ...\n"
            "Future work: CUDA-native Lanczos bidiagonalisation."
        )

    if not is_scipy_available():
        raise RuntimeError("SciPy is required for sparse SVD on CPU")

    def _svd_forward(val_det, row, col, shape, kk):
        U, S, Vt = scipy_svds(val_det, row, col, shape, k=kk)
        return (U.to(val_det.device), S.to(val_det.device),
                Vt.to(val_det.device))

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
    r"""
    Estimate the condition number :math:`\kappa(A)`.

    .. math::

        \kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)},\qquad
        \kappa_p(A) = \lVert A\rVert_p\,\lVert A^{-1}\rVert_p

    Algorithm
    ---------
    For ``ord=2`` take the ratio of the extreme singular values from a
    truncated :meth:`svd` (Lanczos). For other orders, estimate
    :math:`\lVert A^{-1}\rVert` from a single linear solve against a
    random unit vector:

    .. code-block:: text

        if ord == 2:
            S = svd(A).singular_values
            return S.max() / S.min()
        else:
            e = random_unit_vector()
            x = A.solve(e)                  # ~ ||A^{-1}|| estimate
            return ||A||_ord * ||x|| / ||e||

    Complexity
    ----------
    Time :math:`O(m\,nnz)` (one Lanczos SVD or one Krylov solve), space
    :math:`O(n + nnz)`.

    Parameters
    ----------
    ord : int, optional
        Norm order for condition number. Default: 2 (spectral).

    Returns
    -------
    torch.Tensor
        Condition number. Shape [] or ``[*batch_shape]``.

    Examples
    --------
    >>> import torch
    >>> from torch_sla import SparseTensor
    >>> # A = diag(1, 2, 4): kappa_2 = sigma_max / sigma_min = 4 / 1
    >>> d = torch.tensor([1.0, 2.0, 4.0])
    >>> idx = torch.arange(3)
    >>> A = SparseTensor(d, idx, idx, (3, 3))
    >>> A.condition_number(ord=2).round()
    tensor(4.)
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
    r"""
    Compute determinant of the sparse matrix with gradient support.

    .. math::

        \det(A) = \prod_i U_{ii}\quad\text{from } PA = LU,
        \qquad U_{ii} = \text{pivots}

    Uses LU decomposition (CPU) or dense conversion (CUDA) to compute
    the determinant efficiently. Supports automatic differentiation via
    the adjoint method.

    Algorithm
    ---------
    Sparse LU factorize :math:`PA = LU` and multiply the pivots (``L`` is
    unit-diagonal), correcting the sign by the permutation parity:

    .. code-block:: text

        P, L, U = sparse_lu(A)
        det = sign(P) * prod(diag(U))

    Complexity
    ----------
    Time :math:`O(n^{1.5})` (2-D fill) to :math:`O(n^{2})` (3-D);
    space :math:`O(n\log n)` to :math:`O(n^{4/3})` for the LU factors.

    Backward
    --------
    Adjoint via Jacobi's formula:
    :math:`\partial\det(A)/\partial A = \det(A)\,(A^{-\top})`, i.e. the
    backward reuses the factorization for one transposed solve;
    :math:`O(1)` extra graph nodes.

    Returns
    -------
    torch.Tensor
        Determinant value. Shape [] for single matrix or ``[*batch_shape]`` for batched.
        
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
    r"""Log-determinant :math:`\log|\det A|` of this matrix. See :mod:`torch_sla.det`.

    .. math::

        \log\lvert\det A\rvert = \sum_i \log\lvert U_{ii}\rvert
        \;=\; \mathrm{tr}\,\log A
        \;\approx\; \frac{1}{p}\sum_{j=1}^{p} z_j^{\top}\log(A)\,z_j

    Numerically stable alternative to :meth:`det` (no overflow): works in
    log-space directly.

    Algorithm
    ---------
    Two regimes. **Direct** (small / general): factorize and sum
    ``log|diag(U)|`` (or ``2*sum log diag(L)`` for Cholesky on SPD).
    **Stochastic** (large SPD, default ``method='auto'``): the
    **Hutchinson** estimator approximates :math:`\mathrm{tr}\log A` with
    matvec-only probes (no factorization, distributed-friendly):

    .. code-block:: text

        # direct
        P, L, U = sparse_lu(A);  return sum(log|diag(U)|)
        # stochastic (Hutchinson + matvec log(A) z)
        for j in 1..p:  acc += z_j^T (log A) z_j   # z_j Rademacher probes
        return acc / p

    Complexity
    ----------
    Direct: time :math:`O(n^{1.5})`–:math:`O(n^{2})`, space
    :math:`O(n\log n)`–:math:`O(n^{4/3})`. Stochastic: dominated by ``p``
    matvec sequences, :math:`O(p\,m\,nnz)` time, :math:`O(n+nnz)` space.

    Backward
    --------
    Adjoint :math:`\partial \log\lvert\det A\rvert/\partial A = A^{-\top}`;
    :math:`O(1)` extra graph nodes.

    Examples
    --------
    >>> import torch
    >>> from torch_sla import SparseTensor
    >>> # A = diag(1, e, e^2): logdet = 0 + 1 + 2 = 3
    >>> import math
    >>> d = torch.tensor([1.0, math.e, math.e ** 2])
    >>> idx = torch.arange(3)
    >>> A = SparseTensor(d, idx, idx, (3, 3))
    >>> A.logdet().round()
    tensor(3.)
    """
    from ..det import logdet as _logdet
    return _logdet(self, **kwargs)

def lu(self) -> "LUFactorization":
    r"""
    Compute the LU decomposition for repeated solves.

    .. math::

        P A Q = L U

    with ``L`` unit-lower-triangular, ``U`` upper-triangular, and ``P``/``Q``
    row/column permutations chosen to limit fill-in.

    Algorithm
    ---------
    Sparse Gaussian elimination with a fill-reducing reordering (SuperLU /
    UMFPACK via SciPy). The returned object caches the factors so each
    subsequent :meth:`~LUFactorization.solve` is just two cheap triangular
    sweeps:

    .. code-block:: text

        Q = fill_reducing_order(A)          # symbolic
        P, L, U = factorize(A[:, Q])        # numeric
        # later, per RHS:
        solve(b): y = L \\ (P b);  x = Q (U \\ y)

    Complexity
    ----------
    Factorization (once): time :math:`O(n^{1.5})`–:math:`O(n^{2})`, space
    :math:`O(n\log n)`–:math:`O(n^{4/3})` for the factors. Each reuse:
    :math:`O(nnz(L)+nnz(U))` per right-hand side.

    Returns
    -------
    LUFactorization
        Factorization object with a ``solve()`` method.

    Examples
    --------
    >>> import torch
    >>> from torch_sla import SparseTensor
    >>> d = torch.tensor([2.0, 4.0])
    >>> idx = torch.arange(2)
    >>> A = SparseTensor(d, idx, idx, (2, 2))
    >>> lu = A.lu()
    >>> lu.solve(torch.tensor([2.0, 4.0]))
    tensor([1., 1.])
    >>> lu.solve(torch.tensor([4.0, 8.0]))   # reuses the factorization
    tensor([2., 2.])
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


# ====================================================================== #
# LOBPCG (Knyazev 2001) -- shared core                                   #
#                                                                        #
# Used by both the single-device ``_lobpcg_eigsh`` shim (called from     #
# the autograd path) and the distributed ``eigsh_shard`` (called from    #
# DSparseTensor.eigsh). The only thing the caller customises is the     #
# matvec callback -- everything else (3-block subspace, buffer reuse,   #
# CGS2 reorthogonalisation, convergence) is shared.                     #
# ====================================================================== #


def _qr_orthonormalize(Z: torch.Tensor) -> torch.Tensor:
    """Orthonormalise the columns of ``Z`` via a single LAPACK QR.

    Special case for MPS: empirically the Metal ``torch.linalg.qr``
    on tall-skinny ``(n, m)`` blocks scales as O(n²) -- 62 ms at
    n=2000, m=36 -- because the current kernel doesn't do a reduced
    QR. The (n, m) round-trip CPU is 0.9 ms; ~70x faster. Same
    workaround pattern as :func:`_eigh_with_mps_fallback` used by
    the example bench. Upstream PyTorch issue:
    https://github.com/pytorch/pytorch/issues/187567

    Replaces an earlier hand-written Python-loop CGS2 (twice-iterated
    classical Gram-Schmidt). Profiling on CPU SPD problems showed
    that loop dominated the LOBPCG per-iter cost: at n=200, m=18
    the loop took 448 us while ``torch.linalg.qr`` returned in 46 us
    -- a ~10x gap with identical orthonormality (both at machine
    epsilon, ~1e-16). The PR #43 hypothesis that "CGS2 is GPU-
    friendlier" requires a batched (matrix-matrix) CGS2; a Python
    for-loop pays per-column interpreter overhead that erases the
    win on both CPU and GPU.

    ``torch.linalg.qr`` dispatches to LAPACK ``GEQRF`` on CPU and
    cuSOLVER on CUDA, both of which are heavily tuned for tall-skinny
    matrices like the (n, 3m) subspace we feed in here.

    Drops columns whose normalised diagonal of R is below machine
    epsilon -- happens for rank-deficient ``[X | R | P]`` near
    convergence.
    """
    if Z.device.type == "mps":
        Q_cpu, R_cpu = torch.linalg.qr(Z.cpu())
        diag = R_cpu.diagonal().abs()
        eps = torch.finfo(Z.dtype).eps * 100 * diag.max().clamp(min=1)
        keep = diag > eps
        if not bool(keep.all()):
            Q_cpu = Q_cpu[:, keep]
        return Q_cpu.to(Z.device)
    Q, R = torch.linalg.qr(Z)
    diag = R.diagonal().abs()
    eps = torch.finfo(Z.dtype).eps * 100 * diag.max().clamp(min=1)
    keep = diag > eps
    if not bool(keep.all()):
        Q = Q[:, keep]
    return Q


# Kept under the old name for callers that already imported it (none
# in-tree besides ``_lobpcg_core``, but the name shows up in the
# convergence-benchmark example). New code should call
# ``_qr_orthonormalize`` directly.
_cgs2_inplace = _qr_orthonormalize


def _lobpcg_core(
    matvec,
    n: int,
    k: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    largest: bool = True,
    maxiter: int = 1000,
    tol: float = 1e-8,
    T_apply=None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LOBPCG core iteration; matvec callback returns ``A @ Z`` for any
    ``Z`` shape ``[n, m]``.

    Three improvements over the earlier block-steepest-descent
    implementation:

    1. **3-block subspace ``[X | R | P]``**. After iter 0, ``P`` is the
       conjugate direction extracted from the Ritz coordinates of the
       (R, P) blocks (Knyazev 2001 eq. 7). Restores LOBPCG's
       near-cubic-in-the-gap convergence rate.

    2. **Pre-allocated buffers**. ``X``, ``AX``, ``R``, ``P``, ``Z``,
       ``AZ`` allocated once; the hot loop allocates only the small
       (3m x 3m) Hessian.

    3. **LAPACK QR for re-orthonormalisation**. A single
       ``torch.linalg.qr`` call (dispatches to LAPACK ``GEQRF`` on CPU,
       cuSOLVER on CUDA). Earlier code used a Python-loop CGS2 which
       was ~10x slower than QR on CPU at typical block sizes (m ~ 12-36)
       with identical orthonormality, because the per-column interpreter
       overhead erased the level-2-BLAS advantage CGS2 has in theory.

    Internal block size is ``m = min(max(2k, k+2), n)`` (matches
    scipy.sparse.linalg.lobpcg) to give buffer columns that resolve
    closely-clustered extreme eigenvalues; the final return slices
    back to the requested ``k``.

    Convergence is judged on the **Ritz residual norm**
    ``||A x_i - lambda_i x_i||`` vs ``tol * |lambda_i|`` for
    i = 1..k. The earlier eigvals-diff test (
    ``|lambda_i^{n+1} - lambda_i^n| < tol``) tripped early on
    clustered or near-degenerate spectra because successive
    Rayleigh-Ritz steps could re-pick the same Ritz coordinate
    without actually reducing the residual -- the eigenvalue
    looked stable while the eigenvector was still wrong by
    1e-3..1e-5. The residual-norm check is the true Ritz quality
    metric.
    """
    if k > n:
        raise ValueError(f"k={k} exceeds matrix dimension n={n}")
    k = max(k, 1)
    m = min(max(2 * k, k + 2), n)

    if device.type == "mps":
        warnings.warn(
            "LOBPCG on MPS is not recommended: PyTorch's MPS backend "
            "forces float32 (Ritz residual caps at ~1e-4..1e-3 for "
            "PDE-like operators) and is missing native kernels for "
            "linalg.eigh and tall-skinny linalg.qr -- we round-trip "
            "both to CPU as a workaround, so most of the work "
            "actually runs on CPU anyway. Prefer device='cpu' or "
            "'cuda'. This warning fires once per call.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Reproducible random init when a seed is given. ``mps`` doesn't
    # support a typed generator; fall back to CPU sampling + transfer.
    gen_device = "cpu" if device.type == "mps" else device
    g = (torch.Generator(device=gen_device).manual_seed(seed)
          if seed is not None else None)
    X = torch.randn(n, m, dtype=dtype, device=gen_device, generator=g).to(device)
    X, _ = torch.linalg.qr(X)
    AX = torch.empty_like(X)
    R = torch.empty_like(X)
    P = torch.zeros_like(X)
    Z = torch.empty(n, 3 * m, dtype=dtype, device=device)
    eigenvalues = torch.empty(m, dtype=dtype, device=device)

    # Initial Rayleigh-Ritz step.  Use conjugate transpose (.mH) so the Gram
    # matrix X^H A X is genuinely Hermitian for complex A -- plain .T gives a
    # non-Hermitian matrix and torch.linalg.eigh's behaviour on that is
    # undefined (LAPACK on CPU and cuSOLVER on GPU disagree -> complex eigsh
    # converged on CPU but NOT on GPU). .mH == .T on real tensors.
    AX.copy_(matvec(X))
    H = X.mH @ AX
    H = 0.5 * (H + H.mH)
    eigvals, V = torch.linalg.eigh(H)
    idx = eigvals.argsort(descending=largest)
    eigvals, V = eigvals[idx], V[:, idx]
    X_new = X @ V
    AX_new = AX @ V
    X.copy_(X_new)
    AX.copy_(AX_new)
    eigenvalues.copy_(eigvals[:m])

    for iteration in range(maxiter):
        # Residual R = AX - X * lambda
        torch.mul(X, eigenvalues.unsqueeze(0), out=R)
        R.neg_()
        R.add_(AX)

        # Convergence on the TRUE Ritz residual (pre-preconditioner).
        # ``T_apply`` rescales the residual to be a useful search
        # direction, but ||T R|| is not the actual eigenpair error --
        # checking it would let preconditioned LOBPCG report bogus
        # convergence whenever T happens to map the residual to a
        # small vector.
        res_norms = R[:, :k].norm(dim=0)
        denom = eigenvalues[:k].abs().clamp(min=1e-10)
        if (res_norms < tol * denom).all():
            break

        if T_apply is not None:
            R = T_apply(R)

        # Subspace Z = [X | R | P]
        ncols = 2 * m if iteration == 0 else 3 * m
        Z[:, :m].copy_(X)
        Z[:, m:2 * m].copy_(R)
        if iteration > 0:
            Z[:, 2 * m:3 * m].copy_(P)

        Z_active = _cgs2_inplace(Z[:, :ncols])
        ncols_eff = Z_active.shape[1]

        AZ_active = matvec(Z_active)
        H = Z_active.mH @ AZ_active       # conjugate transpose: Hermitian Gram
        H = 0.5 * (H + H.mH)
        eigvals, V = torch.linalg.eigh(H)
        idx = eigvals.argsort(descending=largest)
        eigvals, V = eigvals[idx], V[:, idx]

        # fp32 rank-deficiency guard. In low precision the subspace
        # ``[X | R | P]`` can become (near-)rank-deficient -- e.g. R
        # collapses into span(X) near convergence, or clustered Ritz
        # vectors in X turn collinear. ``_qr_orthonormalize`` then drops
        # the offending columns and returns ``ncols_eff < ncols``, which
        # can even fall below the block size ``m``. When that happens the
        # projected problem only yields ``ncols_eff`` Ritz pairs, so
        # ``H``, ``eigvals`` and ``V`` are ``ncols_eff``-sized. The
        # buffers ``X``/``AX``/``P``/``eigenvalues`` are all sized ``m``,
        # so we must clamp every update to the number of pairs that
        # actually survive (``m_eff``); writing an ``ncols_eff``-wide
        # slice into the ``m``-wide buffers (the old unconditional
        # ``X.copy_(X_new)`` etc.) raised a shape-mismatch RuntimeError.
        # In fp64 the block stays full rank, so ``m_eff == m`` and this is
        # a no-op -- behaviour and convergence are unchanged.
        m_eff = min(m, ncols_eff)
        Vk = V[:, :m_eff]

        X_new = Z_active @ Vk
        AX_new = AZ_active @ Vk
        # Conjugate-direction P from the (R, P) coordinates, only when the
        # active subspace is wider than the kept block (otherwise there is
        # no extra direction and P resets to zero).
        if ncols_eff > m_eff:
            P_new = Z_active[:, m_eff:] @ Vk[m_eff:, :]
        else:
            P_new = torch.zeros(n, m_eff, dtype=dtype, device=device)

        # Update only the leading ``m_eff`` columns / entries; the trailing
        # ``m - m_eff`` columns keep their previous (already orthonormal,
        # already-Ritz) values so the buffers stay well-defined and the
        # next residual/Gram step remains consistent.
        X[:, :m_eff].copy_(X_new)
        AX[:, :m_eff].copy_(AX_new)
        P[:, :m_eff].copy_(P_new)
        P[:, m_eff:].zero_()
        eigenvalues[:m_eff].copy_(eigvals[:m_eff])

    return eigenvalues[:k], X[:, :k]

