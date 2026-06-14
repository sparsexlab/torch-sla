"""Matrix-multiplication paths for SparseTensor."""
from __future__ import annotations
from typing import Union
import torch

from .core import SparseTensor  # noqa: E402  # cross-module use


def _spmv_coo(self, x: torch.Tensor) -> torch.Tensor:
    """
    Sparse matrix-vector/matrix multiply using COO format with scatter_add.
    
    Computes A @ x where A is this sparse tensor and x is dense.
    Works on any device without explicit CSR conversion.
    
    Parameters
    ----------
    x : torch.Tensor
        Dense tensor to multiply. Shape depends on batching:
        - Non-batched: [N] or [N, K]
        - Batched: [B, N] or [B, N, K]
    
    Returns
    -------
    torch.Tensor
        Result of A @ x.
    """
    row = self.row_indices
    col = self.col_indices
    M, N = self.sparse_shape
    
    if self.is_batched:
        batch_shape = self.batch_shape
        B = self.batch_size
        vals_flat = self.values.reshape(B, self.nnz)
        # Determine output dtype via type promotion
        out_dtype = torch.result_type(self.values, x)
        
        if x.dim() == 1:
            # x: [N] - same for all batches -> result [B, M]
            x_gathered = x[col]
            products = vals_flat * x_gathered
            result = torch.zeros(B, M, dtype=out_dtype, device=self.device)
            row_expanded = row.unsqueeze(0).expand(B, -1)
            result.scatter_add_(1, row_expanded, products)
            return result.reshape(*batch_shape, M)
        
        elif x.dim() == len(batch_shape) + 1:
            # x: [...batch, N] -> result [...batch, M]
            x_flat = x.reshape(B, N)
            x_gathered = x_flat[:, col]
            products = vals_flat * x_gathered
            result = torch.zeros(B, M, dtype=out_dtype, device=self.device)
            row_expanded = row.unsqueeze(0).expand(B, -1)
            result.scatter_add_(1, row_expanded, products)
            return result.reshape(*batch_shape, M)
        else:
            # x: [...batch, N, K] -> result [...batch, M, K]
            K = x.size(-1)
            x_flat = x.reshape(B, N, K)
            x_gathered = x_flat[:, col, :]
            products = vals_flat.unsqueeze(-1) * x_gathered
            result = torch.zeros(B, M, K, dtype=out_dtype, device=self.device)
            row_expanded = row.unsqueeze(0).unsqueeze(-1).expand(B, -1, K)
            result.scatter_add_(1, row_expanded, products)
            return result.reshape(*batch_shape, M, K)
    else:
        # Determine output dtype via type promotion (handles float32 @ float64, etc.)
        out_dtype = torch.result_type(self.values, x)
        
        if x.dim() == 1:
            x_gathered = x[col]
            products = self.values * x_gathered
            result = torch.zeros(M, dtype=out_dtype, device=self.device)
            result.scatter_add_(0, row, products)
            return result
        else:
            K = x.size(1)
            x_gathered = x[col]
            products = self.values.unsqueeze(1) * x_gathered
            result = torch.zeros(M, K, dtype=out_dtype, device=self.device)
            row_expanded = row.unsqueeze(1).expand(-1, K)
            result.scatter_add_(0, row_expanded, products)
            return result

def _dense_sparse_mm(self, X: torch.Tensor) -> torch.Tensor:
    """
    Dense @ Sparse: X @ A where X is [..., M] or [..., K, M], A is [..., M, N].
    
    Parameters
    ----------
    X : torch.Tensor
        Dense tensor.
    
    Returns
    -------
    torch.Tensor
        Result of X @ A.
    """
    row = self.row_indices
    col = self.col_indices
    M, N = self.sparse_shape
    
    if self.is_batched:
        batch_shape = self.batch_shape
        B = self.batch_size
        vals_flat = self.values.reshape(B, self.nnz)
        # Determine output dtype via type promotion
        out_dtype = torch.result_type(self.values, X)
        
        if X.dim() == 1:
            X_gathered = X[row]
            products = vals_flat * X_gathered
            result = torch.zeros(B, N, dtype=out_dtype, device=self.device)
            col_expanded = col.unsqueeze(0).expand(B, -1)
            result.scatter_add_(1, col_expanded, products)
            return result.reshape(*batch_shape, N)
        
        elif X.dim() == len(batch_shape) + 1:
            X_flat = X.reshape(B, M)
            X_gathered = X_flat[:, row]
            products = vals_flat * X_gathered
            result = torch.zeros(B, N, dtype=out_dtype, device=self.device)
            col_expanded = col.unsqueeze(0).expand(B, -1)
            result.scatter_add_(1, col_expanded, products)
            return result.reshape(*batch_shape, N)
        
        else:
            K = X.size(-2)
            X_flat = X.reshape(B, K, M)
            X_gathered = X_flat[:, :, row]
            products = vals_flat.unsqueeze(1) * X_gathered
            result = torch.zeros(B, K, N, dtype=out_dtype, device=self.device)
            col_expanded = col.unsqueeze(0).unsqueeze(0).expand(B, K, -1)
            result.scatter_add_(2, col_expanded, products)
            return result.reshape(*batch_shape, K, N)
    else:
        # Determine output dtype via type promotion
        out_dtype = torch.result_type(self.values, X)
        
        if X.dim() == 1:
            X_gathered = X[row]
            products = self.values * X_gathered
            result = torch.zeros(N, dtype=out_dtype, device=self.device)
            result.scatter_add_(0, col, products)
            return result
        else:
            K = X.size(0)
            X_gathered = X[:, row]
            products = self.values.unsqueeze(0) * X_gathered
            result = torch.zeros(K, N, dtype=out_dtype, device=self.device)
            col_expanded = col.unsqueeze(0).expand(K, -1)
            result.scatter_add_(1, col_expanded, products)
            return result

def _spsp_multiply(self, other: "SparseTensor") -> "SparseTensor":
    """
    Sparse-Sparse multiplication: A @ B where both are sparse.
    
    Uses custom autograd function to provide SPARSE gradients.
    Memory usage is O(nnz) not O(M*N).
    
    Parameters
    ----------
    other : SparseTensor
        Right-hand side sparse matrix.
    
    Returns
    -------
    SparseTensor
        Result C = A @ B.
    """
    M, K = self.sparse_shape
    K2, N = other.sparse_shape
    if K != K2:
        raise ValueError(f"Inner dimensions don't match: {K} vs {K2}")
    
    C_values, C_row, C_col, C_shape = _sparse_sparse_matmul_with_sparse_grad(
        self.values, self.row_indices, self.col_indices, (M, K),
        other.values, other.row_indices, other.col_indices, (K, N)
    )
    
    return SparseTensor(C_values, C_row, C_col, C_shape)

def __matmul__(self, other: Union[torch.Tensor, "SparseTensor"]) -> Union[torch.Tensor, "SparseTensor"]:
    """
    Matrix multiplication: A @ other.
    
    Supports:
    - Sparse @ Dense vector: A @ x -> y
    - Sparse @ Dense matrix: A @ X -> Y
    - Sparse @ Sparse: A @ B -> C (with sparse gradients)
    
    Parameters
    ----------
    other : torch.Tensor or SparseTensor
        Right-hand side operand.
    
    Returns
    -------
    torch.Tensor or SparseTensor
        Result of multiplication.
    """
    if isinstance(other, SparseTensor):
        return self._spsp_multiply(other)
    return self._spmv_coo(other)

def __rmatmul__(self, other: torch.Tensor) -> torch.Tensor:
    """
    Dense @ Sparse multiplication: other @ A.
    
    Parameters
    ----------
    other : torch.Tensor
        Left-hand side dense tensor.
    
    Returns
    -------
    torch.Tensor
        Result of multiplication.
    """
    return self._dense_sparse_mm(other)

