"""
Adjoint Nonlinear Solve module for PyTorch

This module provides differentiable nonlinear equation solvers using the adjoint method.

For a nonlinear equation F(u, θ) = 0, where:
- u is the solution variable
- θ are parameters (e.g., neural network weights)

The forward pass solves for u* such that F(u*, θ) = 0.
The backward pass computes gradients ∂L/∂θ using the adjoint method:

    ∂L/∂θ = -λᵀ · ∂F/∂θ

where λ is the adjoint variable satisfying:

    (∂F/∂u)ᵀ · λ = (∂L/∂u)ᵀ

This avoids storing intermediate Jacobians and is memory-efficient.

Supported methods:
- Newton-Raphson with line search
- Picard iteration (fixed-point)
- Anderson acceleration

Usage:
------
    from torch_sla import nonlinear_solve
    
    # Define residual function F(u, θ) -> residual
    def residual_fn(u, theta):
        # Your nonlinear equation
        return F(u, theta)
    
    # Solve with adjoint gradients
    u = nonlinear_solve(residual_fn, u0, theta, method='newton')
    
    # Gradients flow through automatically
    loss = loss_fn(u)
    loss.backward()  # Computes ∂L/∂θ via adjoint method
"""

import torch
from torch import Tensor
from torch.autograd import Function
from typing import Callable, Optional, Tuple, Union, Dict, Any
import warnings

from .linear_solve import spsolve


class NonlinearSolveAdjoint(Function):
    """
    Adjoint-based nonlinear solver with automatic differentiation.
    
    Uses implicit differentiation to compute gradients without storing
    intermediate Jacobians. Memory-efficient for large-scale problems.
    """
    
    @staticmethod
    def forward(
        ctx,
        u0: Tensor,
        num_params: int,  # Number of parameter tensors
        *args,  # params tensors followed by config dict
    ) -> Tensor:
        """
        Forward pass: solve F(u, θ) = 0 for u.
        
        Args:
            u0: Initial guess for solution
            num_params: Number of parameter tensors
            *args: First num_params elements are param tensors, last is config dict
            
        Returns:
            u: Solution satisfying F(u, θ) ≈ 0
        """
        # Extract params and config
        params = args[:num_params]
        config = args[num_params]
        
        # Extract config
        residual_fn = config['residual_fn']
        jacobian_fn = config.get('jacobian_fn', None)
        method = config.get('method', 'newton')
        tol = config.get('tol', 1e-6)
        atol = config.get('atol', 1e-10)
        max_iter = config.get('max_iter', 50)
        line_search = config.get('line_search', True)
        verbose = config.get('verbose', False)
        linear_solver = config.get('linear_solver', 'pytorch')
        linear_method = config.get('linear_method', 'cg')
        
        # Detach for forward solve (no gradient tracking during iteration)
        u = u0.detach().clone()
        params_detached = tuple(p.detach() if isinstance(p, Tensor) else p for p in params)
        
        if method == 'newton':
            u, info = _newton_solve(
                u, params_detached, residual_fn, jacobian_fn,
                tol=tol, atol=atol, max_iter=max_iter,
                line_search=line_search, verbose=verbose,
                linear_solver=linear_solver, linear_method=linear_method
            )
        elif method == 'picard':
            u, info = _picard_solve(
                u, params_detached, residual_fn,
                tol=tol, atol=atol, max_iter=max_iter, verbose=verbose
            )
        elif method == 'anderson':
            u, info = _anderson_solve(
                u, params_detached, residual_fn,
                tol=tol, atol=atol, max_iter=max_iter, verbose=verbose
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'newton', 'picard', or 'anderson'")
        
        # Save for backward - save u and all param tensors that require grad
        tensors_to_save = [u]
        param_requires_grad = []
        params_no_grad = []  # Store non-grad params for backward
        
        for p in params:
            if isinstance(p, Tensor):
                param_requires_grad.append(p.requires_grad)
                if p.requires_grad:
                    tensors_to_save.append(p)
                    params_no_grad.append(None)  # placeholder
                else:
                    params_no_grad.append(p.detach())  # save detached copy
            else:
                param_requires_grad.append(False)
                params_no_grad.append(p)  # non-tensor param
        
        ctx.save_for_backward(*tensors_to_save)
        ctx.residual_fn = residual_fn
        ctx.jacobian_fn = jacobian_fn
        ctx.num_params = num_params
        ctx.param_requires_grad = param_requires_grad
        ctx.params_no_grad = params_no_grad  # for reconstructing params in backward
        ctx.linear_solver = linear_solver
        ctx.linear_method = linear_method
        ctx.tol = tol
        ctx.atol = atol
        ctx.info = info
        
        return u
    
    @staticmethod
    def backward(ctx, grad_u: Tensor):
        """
        Backward pass using adjoint method.
        
        Computes ∂L/∂θ = -λᵀ · ∂F/∂θ where (∂F/∂u)ᵀ · λ = grad_u
        
        Returns:
            Tuple of gradients: (grad_u0, grad_num_params, *grad_params, grad_config)
        """
        saved = ctx.saved_tensors
        u = saved[0]
        param_tensors = saved[1:]  # Only tensors that required grad
        
        residual_fn = ctx.residual_fn
        jacobian_fn = ctx.jacobian_fn
        num_params = ctx.num_params
        param_requires_grad = ctx.param_requires_grad
        params_no_grad = ctx.params_no_grad
        
        # Reconstruct params list using saved tensors and cached non-grad params
        param_idx = 0
        params_for_backward = []
        for i, requires_grad in enumerate(param_requires_grad):
            if requires_grad:
                params_for_backward.append(param_tensors[param_idx])
                param_idx += 1
            else:
                params_for_backward.append(params_no_grad[i])  # use cached non-grad param
        
        # Enable gradient computation (backward is called in no_grad context)
        with torch.enable_grad():
            # Step 1: Solve adjoint equation (∂F/∂u)ᵀ · λ = grad_u
            lambda_adj = _solve_adjoint_system(
                u, params_for_backward, residual_fn, jacobian_fn, grad_u,
                linear_solver=ctx.linear_solver, 
                linear_method=ctx.linear_method,
                tol=ctx.tol, atol=ctx.atol
            )
            
            # Step 2: Compute ∂L/∂θ = -λᵀ · ∂F/∂θ for each parameter
            grad_params = []
            
            # Setup variables for gradient computation
            u_var = u.detach().requires_grad_(True)
            params_var = []
            for i, requires_grad in enumerate(param_requires_grad):
                if requires_grad:
                    params_var.append(params_for_backward[i].detach().requires_grad_(True))
                else:
                    params_var.append(params_for_backward[i])
            
            # Compute F(u, θ) with gradient tracking
            F = residual_fn(u_var, *params_var)
            
            # Compute gradients for each parameter
            for i, requires_grad in enumerate(param_requires_grad):
                if requires_grad:
                    # ∂L/∂θᵢ = -λᵀ · ∂F/∂θᵢ
                    grad_p = torch.autograd.grad(
                        F, params_var[i], 
                        grad_outputs=lambda_adj,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    
                    if grad_p is not None:
                        grad_params.append(-grad_p)
                    else:
                        grad_params.append(None)
                else:
                    grad_params.append(None)
        
        # Return: (grad_u0, grad_num_params, *grad_params, grad_config)
        # grad_u0 = None, grad_num_params = None, grad_config = None
        return (None, None) + tuple(grad_params) + (None,)


def _newton_solve(
    u: Tensor,
    params: Tuple,
    residual_fn: Callable,
    jacobian_fn: Optional[Callable],
    tol: float,
    atol: float,
    max_iter: int,
    line_search: bool,
    verbose: bool,
    linear_solver: str,
    linear_method: str,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Newton-Raphson solver with optional line search."""
    
    device = u.device
    dtype = u.dtype
    n = u.numel()
    
    for iteration in range(max_iter):
        # Compute residual
        F = residual_fn(u, *params)
        F_norm = torch.norm(F).item()
        
        if verbose:
            print(f"  Newton iter {iteration}: ||F|| = {F_norm:.2e}")
        
        # Check convergence
        if F_norm < atol:
            if verbose:
                print(f"  Converged (atol) at iteration {iteration}")
            return u, {'converged': True, 'iterations': iteration, 'residual': F_norm}
        
        if iteration > 0 and F_norm < tol * F_norm_0:
            if verbose:
                print(f"  Converged (rtol) at iteration {iteration}")
            return u, {'converged': True, 'iterations': iteration, 'residual': F_norm}
        
        if iteration == 0:
            F_norm_0 = F_norm
        
        # Compute Newton step: J(u) * du = -F
        if jacobian_fn is not None:
            # Explicit Jacobian provided
            val, row, col, shape = jacobian_fn(u, *params)
            du = spsolve(val, row, col, shape, -F, 
                        backend=linear_solver, method=linear_method,
                        tol=tol * 0.1, maxiter=max(100, n // 10))
        else:
            # Use Jacobian-free Newton-Krylov
            du = _jacobian_free_solve(
                u, params, residual_fn, -F,
                linear_solver=linear_solver, linear_method=linear_method,
                tol=tol * 0.1, max_iter=max(100, n // 10)
            )
        
        # Line search
        if line_search:
            alpha = _armijo_line_search(u, du, params, residual_fn, F_norm)
        else:
            alpha = 1.0
        
        # Update
        u = u + alpha * du
    
    warnings.warn(f"Newton did not converge in {max_iter} iterations, ||F|| = {F_norm:.2e}")
    return u, {'converged': False, 'iterations': max_iter, 'residual': F_norm}


def _picard_solve(
    u: Tensor,
    params: Tuple,
    residual_fn: Callable,
    tol: float,
    atol: float,
    max_iter: int,
    verbose: bool,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Picard (fixed-point) iteration solver."""
    
    for iteration in range(max_iter):
        F = residual_fn(u, *params)
        F_norm = torch.norm(F).item()
        
        if verbose:
            print(f"  Picard iter {iteration}: ||F|| = {F_norm:.2e}")
        
        if F_norm < atol:
            return u, {'converged': True, 'iterations': iteration, 'residual': F_norm}
        
        if iteration > 0 and F_norm < tol * F_norm_0:
            return u, {'converged': True, 'iterations': iteration, 'residual': F_norm}
        
        if iteration == 0:
            F_norm_0 = F_norm
        
        # Fixed-point update: u_new = u - F(u)
        # This assumes F(u) = u - g(u) form, so u = g(u)
        u = u - F
    
    warnings.warn(f"Picard did not converge in {max_iter} iterations")
    return u, {'converged': False, 'iterations': max_iter, 'residual': F_norm}


def _anderson_solve(
    u: Tensor,
    params: Tuple,
    residual_fn: Callable,
    tol: float,
    atol: float,
    max_iter: int,
    verbose: bool,
    m: int = 5,  # Anderson depth
) -> Tuple[Tensor, Dict[str, Any]]:
    """Anderson acceleration solver."""
    
    device = u.device
    dtype = u.dtype
    n = u.numel()
    
    # History storage
    X_hist = []  # Previous iterates
    F_hist = []  # Previous residuals
    
    for iteration in range(max_iter):
        F = residual_fn(u, *params)
        F_norm = torch.norm(F).item()
        
        if verbose:
            print(f"  Anderson iter {iteration}: ||F|| = {F_norm:.2e}")
        
        if F_norm < atol:
            return u, {'converged': True, 'iterations': iteration, 'residual': F_norm}
        
        if iteration > 0 and F_norm < tol * F_norm_0:
            return u, {'converged': True, 'iterations': iteration, 'residual': F_norm}
        
        if iteration == 0:
            F_norm_0 = F_norm
        
        # Store history
        X_hist.append(u.clone())
        F_hist.append(F.clone())
        
        # Limit history size
        if len(X_hist) > m + 1:
            X_hist.pop(0)
            F_hist.pop(0)
        
        # Anderson mixing
        if len(F_hist) >= 2:
            # Build matrix of residual differences
            k = len(F_hist) - 1
            dF = torch.stack([F_hist[i+1] - F_hist[i] for i in range(k)], dim=1)  # [n, k]
            
            # Solve least squares: min ||F_k - dF @ alpha||^2
            # (dF^T dF) alpha = dF^T F_k
            gram = dF.T @ dF + 1e-10 * torch.eye(k, device=device, dtype=dtype)
            rhs = dF.T @ F_hist[-1]
            alpha = torch.linalg.solve(gram, rhs)
            
            # Compute new iterate
            u_new = X_hist[-1] - F_hist[-1]  # Simple fixed-point
            for i in range(k):
                u_new = u_new - alpha[i] * (X_hist[i+1] - X_hist[i] - (F_hist[i+1] - F_hist[i]))
            u = u_new
        else:
            # Simple fixed-point for first iteration
            u = u - F
    
    warnings.warn(f"Anderson did not converge in {max_iter} iterations")
    return u, {'converged': False, 'iterations': max_iter, 'residual': F_norm}


def _jacobian_free_solve(
    u: Tensor,
    params: Tuple,
    residual_fn: Callable,
    rhs: Tensor,
    linear_solver: str,
    linear_method: str,
    tol: float,
    max_iter: int,
) -> Tensor:
    """
    Jacobian-free Newton-Krylov solve.
    
    Solves J(u) @ x = rhs using Krylov methods with Jacobian-vector products
    computed via automatic differentiation.
    """
    device = u.device
    dtype = u.dtype
    n = u.numel()
    
    # Detach params - we only need Jacobian w.r.t. u, not params
    params_detached = tuple(
        p.detach() if isinstance(p, Tensor) else p for p in params
    )
    
    def matvec(v):
        """Compute J(u) @ v using autograd (jvp)."""
        # Enable gradient tracking - needed when called from autograd.Function.forward
        with torch.enable_grad():
            # Enable gradient tracking for u only
            u_var = u.detach().clone().requires_grad_(True)
            
            # Compute F(u) with gradient tracking (params detached)
            F = residual_fn(u_var, *params_detached)
            
            # Jacobian-vector product via autograd
            # Jv = ∂F/∂u @ v
            Jv = torch.autograd.grad(
                outputs=F, 
                inputs=u_var, 
                grad_outputs=v,
                create_graph=False,
                retain_graph=False
            )[0]
        return Jv
    
    # Use CG with matvec
    x = torch.zeros_like(rhs)
    r = rhs.clone()  # r = b - A @ x, initially r = b
    p = r.clone()
    rs_old = torch.dot(r.flatten(), r.flatten())
    
    rhs_norm = torch.norm(rhs)
    if rhs_norm < 1e-30:
        return x
    
    for i in range(max_iter):
        Ap = matvec(p)
        pAp = torch.dot(p.flatten(), Ap.flatten())
        
        if abs(pAp) < 1e-30:
            break
        
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        
        rs_new = torch.dot(r.flatten(), r.flatten())
        
        if torch.sqrt(rs_new) < tol * rhs_norm:
            break
        
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    
    return x


def _solve_adjoint_system(
    u: Tensor,
    params: Tuple,
    residual_fn: Callable,
    jacobian_fn: Optional[Callable],
    rhs: Tensor,
    linear_solver: str,
    linear_method: str,
    tol: float,
    atol: float,
) -> Tensor:
    """
    Solve the adjoint system: (∂F/∂u)ᵀ @ λ = rhs
    """
    device = u.device
    dtype = u.dtype
    n = u.numel()

    # ``rhs`` here is dL/du, a gradient, so this solve has to be invariant to
    # its scale -- lambda is linear in it. Both branches below stop on an
    # absolute floor (``atol`` in the CG loop, ``max(atol, rtol*||b||)`` inside
    # ``spsolve``), which does not scale: as the outer optimisation converges
    # ||dL/du|| shrinks past the floor and the solve returns zero before doing
    # any work, silently killing the gradient. Normalise here and undo it at the
    # single return below. See ``linear_solve._adjoint_solve``.
    rhs_scale = rhs.norm()
    rhs = rhs / torch.where(rhs_scale > 0, rhs_scale, torch.ones_like(rhs_scale))
    
    if jacobian_fn is not None:
        # Explicit Jacobian: transpose and solve
        val, row, col, shape = jacobian_fn(u, *params)
        # Transpose: swap row and col
        lambda_adj = spsolve(val, col, row, (shape[1], shape[0]), rhs,
                            backend=linear_solver, method=linear_method,
                            tol=tol, maxiter=max(100, n // 10))
    else:
        # Jacobian-free: use CG with Jᵀv products
        def matvec_transpose(v):
            """Compute Jᵀ @ v using autograd."""
            u_var = u.detach().requires_grad_(True)
            F = residual_fn(u_var, *params)
            
            # For Jᵀv, we use the identity: Jᵀv = ∂(v·F)/∂u
            vF = torch.dot(v.flatten(), F.flatten())
            Jtv = torch.autograd.grad(vF, u_var, retain_graph=False)[0]
            return Jtv
        
        # CG for Jᵀ @ λ = rhs
        lambda_adj = torch.zeros_like(rhs)
        r = rhs.clone()
        p = r.clone()
        rs_old = torch.dot(r.flatten(), r.flatten())
        
        for i in range(max(100, n // 10)):
            Ap = matvec_transpose(p)
            pAp = torch.dot(p.flatten(), Ap.flatten())
            
            if abs(pAp) < 1e-30:
                break
            
            alpha = rs_old / pAp
            lambda_adj = lambda_adj + alpha * p
            r = r - alpha * Ap
            
            rs_new = torch.dot(r.flatten(), r.flatten())
            
            if torch.sqrt(rs_new) < tol * torch.norm(rhs) + atol:
                break
            
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

    # Undo the normalisation applied to ``rhs`` above. A zero rhs stays zero.
    return lambda_adj * rhs_scale


def _armijo_line_search(
    u: Tensor,
    du: Tensor,
    params: Tuple,
    residual_fn: Callable,
    F_norm: float,
    c: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 20,
) -> float:
    """Armijo backtracking line search."""
    
    alpha = 1.0
    for _ in range(max_iter):
        u_new = u + alpha * du
        F_new = residual_fn(u_new, *params)
        F_new_norm = torch.norm(F_new).item()
        
        # Armijo condition: f(x + α*d) ≤ f(x) - c*α*||d||
        if F_new_norm <= (1 - c * alpha) * F_norm:
            return alpha
        
        alpha *= rho
    
    return alpha


# ============================================================================
# High-level API
# ============================================================================

def nonlinear_solve(
    residual_fn: Callable,
    u0: Tensor,
    *params,
    jacobian_fn: Optional[Callable] = None,
    method: str = 'newton',
    tol: float = 1e-6,
    atol: float = 1e-10,
    max_iter: int = 50,
    line_search: bool = True,
    verbose: bool = False,
    linear_solver: str = 'pytorch',
    linear_method: str = 'cg',
) -> Tensor:
    """
    Solve nonlinear equation F(u, θ) = 0 with adjoint-based gradients.
    
    Args:
        residual_fn: Function F(u, *params) -> residual tensor
        u0: Initial guess for solution
        *params: Parameters θ (tensors with requires_grad=True for gradient computation)
        jacobian_fn: Optional function J(u, *params) -> (val, row, col, shape)
                    Returns sparse Jacobian in COO format. If None, uses autograd.
        method: Nonlinear solver method
            - 'newton': Newton-Raphson with optional line search (default)
            - 'picard': Fixed-point iteration
            - 'anderson': Anderson acceleration
        tol: Relative convergence tolerance
        atol: Absolute convergence tolerance
        max_iter: Maximum number of nonlinear iterations
        line_search: Use Armijo line search for Newton (default: True)
        verbose: Print convergence information
        linear_solver: Backend for linear solves ('pytorch', 'scipy', 'cudss')
        linear_method: Method for linear solves ('cg', 'bicgstab', 'lu')
        
    Returns:
        u: Solution tensor satisfying F(u, θ) ≈ 0
        
    Example:
        >>> def residual(u, A_val, b):
        ...     # Nonlinear: A(u) @ u - b where A depends on u
        ...     return torch.sparse.mm(A, u.unsqueeze(1)).squeeze() - b
        ...
        >>> u0 = torch.zeros(n, requires_grad=False)
        >>> A_val = torch.randn(nnz, requires_grad=True)
        >>> b = torch.randn(n, requires_grad=True)
        >>> 
        >>> u = nonlinear_solve(residual, u0, A_val, b, method='newton')
        >>> loss = some_loss(u)
        >>> loss.backward()  # Computes ∂L/∂A_val and ∂L/∂b via adjoint
    """
    config = {
        'residual_fn': residual_fn,
        'jacobian_fn': jacobian_fn,
        'method': method,
        'tol': tol,
        'atol': atol,
        'max_iter': max_iter,
        'line_search': line_search,
        'verbose': verbose,
        'linear_solver': linear_solver,
        'linear_method': linear_method,
    }
    
    # Call apply with: u0, num_params, *params, config
    return NonlinearSolveAdjoint.apply(u0, len(params), *params, config)


# Alias
adjoint_solve = nonlinear_solve

