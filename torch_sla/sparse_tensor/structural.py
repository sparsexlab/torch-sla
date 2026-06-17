"""Structural property queries for SparseTensor."""
from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union
import torch

from .core import SparseTensor  # noqa: E402  # cross-module use


def _check_pair_match(
    self,
    conjugate: bool,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    """Check ``A == A^T`` (``conjugate=False``) or ``A == A^H``
    (``conjugate=True``) via hash-paired index matching + value
    comparison. Shared kernel for :meth:`is_symmetric` and
    :meth:`is_hermitian`.
    """
    if not self.is_square:
        result = torch.tensor(False, device=self.device)
        if self.is_batched:
            result = result.expand(self.batch_shape).clone()
        return result

    row = self.row_indices
    col = self.col_indices
    _, N = self.sparse_shape

    forward_hash = row * N + col
    transpose_hash = col * N + row
    forward_order = forward_hash.argsort()
    transpose_order = transpose_hash.argsort()

    if not torch.equal(forward_hash[forward_order],
                       transpose_hash[transpose_order]):
        result = torch.tensor(False, device=self.device)
        if self.is_batched:
            result = result.expand(self.batch_shape).clone()
        return result

    if self.is_batched:
        B = self.batch_size
        vals_flat = self.values.reshape(B, self.nnz)
        vals_forward = vals_flat[:, forward_order]
        vals_transpose = vals_flat[:, transpose_order]
        if conjugate:
            vals_transpose = vals_transpose.conj()
        diff = (vals_forward - vals_transpose).abs()
        threshold = atol + rtol * vals_forward.abs()
        return (diff <= threshold).all(dim=-1).reshape(self.batch_shape)

    vals_forward = self.values[forward_order]
    vals_transpose = self.values[transpose_order]
    if conjugate:
        vals_transpose = vals_transpose.conj()
    diff = (vals_forward - vals_transpose).abs()
    threshold = atol + rtol * vals_forward.abs()
    return torch.tensor((diff <= threshold).all().item(),
                        device=self.device)

def is_symmetric(
    self,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    force_recompute: bool = False
) -> torch.Tensor:
    """
    Check if the matrix is symmetric (``A == A^T``).

    For batched tensors, checks each matrix independently and returns
    a boolean tensor with shape matching the batch dimensions.

    Parameters
    ----------
    atol : float, optional
        Absolute tolerance for comparison. Default: 1e-8.
    rtol : float, optional
        Relative tolerance for comparison. Default: 1e-5.
    force_recompute : bool, optional
        If True, recompute even if cached. Default: False.

    Returns
    -------
    torch.Tensor
        Boolean tensor with shape ``[]`` (non-batched) or
        ``[*batch_shape]`` (batched).
    """
    if self._is_symmetric_cache is not None and not force_recompute:
        return self._is_symmetric_cache
    result = self._check_pair_match(conjugate=False, atol=atol, rtol=rtol)
    self._is_symmetric_cache = result
    return result

def is_hermitian(
    self,
    atol: float = 1e-8,
    rtol: float = 1e-5,
    force_recompute: bool = False
) -> torch.Tensor:
    """
    Check if the matrix is Hermitian (``A == A^H``).

    For real-valued matrices this is equivalent to
    :meth:`is_symmetric`. For complex matrices it additionally
    requires off-diagonal entries to be conjugate transposes and
    diagonal entries to be real.

    Parameters
    ----------
    atol, rtol : float
        Absolute / relative tolerance, defaults ``1e-8`` / ``1e-5``.
    force_recompute : bool
        If ``True``, bypass the cached result.

    Returns
    -------
    torch.Tensor
        Boolean tensor with shape ``[]`` (non-batched) or
        ``[*batch_shape]`` (batched).
    """
    if self._is_hermitian_cache is not None and not force_recompute:
        return self._is_hermitian_cache
    result = self._check_pair_match(conjugate=True, atol=atol, rtol=rtol)
    self._is_hermitian_cache = result
    return result

def is_positive_definite(
    self, 
    method: Literal["gershgorin", "cholesky", "eigenvalue"] = "gershgorin",
    force_recompute: bool = False
) -> torch.Tensor:
    """
    Check if the matrix is positive definite.
    
    For batched tensors, checks each matrix independently and returns
    a boolean tensor with shape matching the batch dimensions.
    
    Parameters
    ----------
    method : {"gershgorin", "cholesky", "eigenvalue"}, optional
        Method for checking:
        - "gershgorin": Fast check using Gershgorin circles (sufficient but not necessary)
        - "cholesky": Try Cholesky decomposition (necessary and sufficient, slower)
        - "eigenvalue": Check smallest eigenvalues (necessary and sufficient, slowest)
        Default: "gershgorin".
    force_recompute : bool, optional
        If True, recompute even if cached. Default: False.
    
    Returns
    -------
    torch.Tensor
        Boolean tensor with shape:
        - [] (scalar) for non-batched tensors
        - [*batch_shape] for batched tensors
    
    Examples
    --------
    >>> A = SparseTensor(val, row, col, (3, 3))
    >>> A.is_positive_definite()  # tensor(True) or tensor(False)
    >>> A.is_positive_definite(method="cholesky")  # More accurate check
    
    >>> A_batch = SparseTensor(val_batch, row, col, (4, 3, 3))
    >>> A_batch.is_positive_definite()  # tensor([True, True, True, True])
    """
    if self._is_positive_definite_cache is not None and not force_recompute:
        return self._is_positive_definite_cache
    
    if not self.is_square:
        result = torch.tensor(False, device=self.device)
        if self.is_batched:
            result = result.expand(self.batch_shape).clone()
        self._is_positive_definite_cache = result
        return result
    
    row = self.row_indices
    col = self.col_indices
    M, N = self.sparse_shape
    
    if method == "gershgorin":
        result = self._check_pd_gershgorin()
    elif method == "cholesky":
        result = self._check_pd_cholesky()
    else:  # eigenvalue
        result = self._check_pd_eigenvalue()
    
    self._is_positive_definite_cache = result
    return result

def _check_pd_gershgorin(self) -> torch.Tensor:
    """Check positive definiteness using Gershgorin circles."""
    row = self.row_indices
    col = self.col_indices
    M, N = self.sparse_shape
    is_diag = (row == col)
    
    if self.is_batched:
        B = self.batch_size
        vals_flat = self.values.reshape(B, self.nnz)
        
        # Gershgorin works on real magnitudes; for complex (Hermitian)
        # matrices the diagonal is real, so use real parts / |a_ij|.
        real_dtype = self.values.abs().dtype

        # Diagonal elements (real part)
        diag_rows = row[is_diag]
        diag_vals = vals_flat[:, is_diag].real  # [B, num_diag]

        diag = torch.zeros(B, M, dtype=real_dtype, device=self.device)
        diag.scatter_(1, diag_rows.unsqueeze(0).expand(B, -1), diag_vals)

        # Off-diagonal sum
        is_offdiag = ~is_diag
        offdiag_rows = row[is_offdiag]
        offdiag_vals = vals_flat[:, is_offdiag].abs()  # [B, num_offdiag]

        offdiag_sum = torch.zeros(B, M, dtype=real_dtype, device=self.device)
        offdiag_sum.scatter_add_(1, offdiag_rows.unsqueeze(0).expand(B, -1), offdiag_vals)
        
        # Check: diag > offdiag_sum AND diag > 0
        is_pd = ((diag > offdiag_sum) & (diag > 0)).all(dim=-1)
        return is_pd.reshape(self.batch_shape)
    else:
        real_dtype = self.values.abs().dtype

        diag_rows = row[is_diag]
        diag_vals = self.values[is_diag].real

        diag = torch.zeros(M, dtype=real_dtype, device=self.device)
        diag.scatter_(0, diag_rows, diag_vals)

        is_offdiag = ~is_diag
        offdiag_rows = row[is_offdiag]
        offdiag_vals = self.values[is_offdiag].abs()

        offdiag_sum = torch.zeros(M, dtype=real_dtype, device=self.device)
        offdiag_sum.scatter_add_(0, offdiag_rows, offdiag_vals)
        
        is_pd = ((diag > offdiag_sum) & (diag > 0)).all()
        return torch.tensor(is_pd.item(), device=self.device)

def _check_pd_cholesky(self) -> torch.Tensor:
    """Check positive definiteness using Cholesky decomposition."""
    if self.is_batched:
        results = []
        for idx in self._batch_indices():
            try:
                A_dense = self.to_dense(idx)
                torch.linalg.cholesky(A_dense)
                results.append(True)
            except RuntimeError:
                results.append(False)
        return torch.tensor(results, device=self.device).reshape(self.batch_shape)
    else:
        try:
            A_dense = self.to_dense()
            torch.linalg.cholesky(A_dense)
            return torch.tensor(True, device=self.device)
        except RuntimeError:
            return torch.tensor(False, device=self.device)

def _check_pd_eigenvalue(self) -> torch.Tensor:
    """Check positive definiteness using eigenvalue computation."""
    if self.is_batched:
        results = []
        for idx in self._batch_indices():
            try:
                A_dense = self.to_dense(idx)
                eigenvalues = torch.linalg.eigvalsh(A_dense)
                results.append((eigenvalues > 0).all().item())
            except Exception:
                results.append(False)
        return torch.tensor(results, device=self.device).reshape(self.batch_shape)
    else:
        try:
            A_dense = self.to_dense()
            eigenvalues = torch.linalg.eigvalsh(A_dense)
            return torch.tensor((eigenvalues > 0).all().item(), device=self.device)
        except Exception:
            return torch.tensor(False, device=self.device)

def _batch_indices(self):
    """Generate all batch index tuples."""
    import itertools
    ranges = [range(s) for s in self.batch_shape]
    return itertools.product(*ranges)

