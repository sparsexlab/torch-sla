"""``SparseTensorList`` for batched-with-different-pattern collections."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union, Literal

import torch

from .core import SparseTensor


class SparseTensorList:
    """
    A list of SparseTensors with different structures.
    
    Provides a unified interface for batch operations on matrices
    with different sparsity patterns. Unlike batched SparseTensor
    (which requires same structure), SparseTensorList allows
    each matrix to have different shape and sparsity pattern.
        
        Parameters
        ----------
    tensors : List[SparseTensor]
        List of SparseTensor objects.
    
    Attributes
    ----------
    shapes : List[Tuple[int, ...]]
        List of shapes for each tensor.
    device : torch.device
        Device (from first tensor).
    dtype : torch.dtype
        Data type (from first tensor).
    
    Examples
    --------
    >>> # Create matrices with different sizes
    >>> A1 = SparseTensor(val1, row1, col1, (10, 10))
    >>> A2 = SparseTensor(val2, row2, col2, (20, 20))
    >>> A3 = SparseTensor(val3, row3, col3, (30, 30))
    
    >>> # Create list
    >>> matrices = SparseTensorList([A1, A2, A3])
    >>> print(matrices.shapes)  # [(10, 10), (20, 20), (30, 30)]
    
    >>> # Batch solve
    >>> x_list = matrices.solve([b1, b2, b3])
    
    >>> # Check properties for all
    >>> is_sym = matrices.is_symmetric()  # [tensor(True), tensor(True), tensor(True)]
    """
    
    def __init__(self, tensors: List["SparseTensor"]):
        if not tensors:
            raise ValueError("SparseTensorList cannot be empty")
        self._tensors = list(tensors)
    
    @classmethod
    def from_coo_list(
        cls,
        matrices: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]],
    ) -> "SparseTensorList":
        """
        Create from list of COO data tuples.
        
        Parameters
        ----------
        matrices : List[Tuple]
            List of (values, row_indices, col_indices, shape) tuples.
        
        Returns
        -------
        SparseTensorList
            List of SparseTensors.
        
        Examples
        --------
        >>> data = [
        ...     (val1, row1, col1, (10, 10)),
        ...     (val2, row2, col2, (20, 20)),
        ... ]
        >>> matrices = SparseTensorList.from_coo_list(data)
        """
        tensors = [
            SparseTensor(val, row, col, shape)
            for val, row, col, shape in matrices
        ]
        return cls(tensors)
    
    @classmethod
    def from_torch_sparse_list(cls, A_list: List[torch.Tensor]) -> "SparseTensorList":
        """
        Create from list of PyTorch sparse tensors.
        
        Parameters
        ----------
        A_list : List[torch.Tensor]
            List of PyTorch sparse COO tensors.
        
        Returns
        -------
        SparseTensorList
            List of SparseTensors.
        """
        tensors = [SparseTensor.from_torch_sparse(A) for A in A_list]
        return cls(tensors)
    
    @property
    def shapes(self) -> List[Tuple[int, ...]]:
        """List of shapes for each tensor."""
        return [t.shape for t in self._tensors]
    
    @property
    def device(self) -> torch.device:
        """Device of the first tensor."""
        return self._tensors[0].device
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of the first tensor."""
        return self._tensors[0].dtype
    
    def __len__(self) -> int:
        """Number of tensors in the list."""
        return len(self._tensors)
    
    def __getitem__(self, idx: int) -> "SparseTensor":
        """
        Get tensor by index.
        
        Parameters
        ----------
        idx : int
            Index (supports negative indexing).
        
        Returns
        -------
        SparseTensor
            The tensor at that index.
        """
        if idx < 0:
            idx = len(self._tensors) + idx
        return self._tensors[idx]
    
    def __iter__(self):
        """Iterate over tensors."""
        return iter(self._tensors)
    
    def to(self, device: Union[str, torch.device]) -> "SparseTensorList":
        """
        Move all tensors to device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device.
        
        Returns
        -------
        SparseTensorList
            New list with tensors on target device.
        """
        return SparseTensorList([t.to(device) for t in self._tensors])
    
    def cuda(self) -> "SparseTensorList":
        """Move all tensors to CUDA."""
        return self.to('cuda')
    
    def cpu(self) -> "SparseTensorList":
        """Move all tensors to CPU."""
        return self.to('cpu')
    
    # =========================================================================
    # Arithmetic Operations
    # =========================================================================
    
    def __matmul__(self, x_list: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
        """
        Batch matrix-vector/matrix multiplication.
        
        Parameters
        ----------
        x_list : List[torch.Tensor] or torch.Tensor
            If List: one vector/matrix per sparse tensor, each with compatible shape.
            If Tensor: broadcasted to all matrices (must have compatible shape for all).
        
        Returns
        -------
        List[torch.Tensor]
            List of results [A1 @ x1, A2 @ x2, ...] or [A1 @ x, A2 @ x, ...]
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> # Per-matrix vectors
        >>> y_list = matrices @ [x1, x2, x3]
        >>> # Broadcast same vector
        >>> y_list = matrices @ x  # x applied to all
        """
        if isinstance(x_list, torch.Tensor):
            # Broadcast same tensor to all
            return [t @ x_list for t in self._tensors]
        
        if len(x_list) != len(self._tensors):
            raise ValueError(f"Expected {len(self._tensors)} vectors, got {len(x_list)}")
        return [t @ x for t, x in zip(self._tensors, x_list)]
    
    def __add__(self, other: Union["SparseTensorList", float, int]) -> "SparseTensorList":
        """
        Element-wise addition.
        
        Parameters
        ----------
        other : SparseTensorList or scalar
            If SparseTensorList: add corresponding matrices (must have same length).
            If scalar: add to all matrices.
            
        Returns
        -------
        SparseTensorList
            Result of addition.
        """
        if isinstance(other, SparseTensorList):
            if len(other) != len(self._tensors):
                raise ValueError(f"Length mismatch: {len(self._tensors)} vs {len(other)}")
            return SparseTensorList([a + b for a, b in zip(self._tensors, other._tensors)])
        # Scalar addition - add to values
        return SparseTensorList([
            SparseTensor(t.values + other, t.row_indices, t.col_indices, t.shape)
            for t in self._tensors
        ])
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other: Union["SparseTensorList", float, int]) -> "SparseTensorList":
        """Element-wise subtraction."""
        if isinstance(other, SparseTensorList):
            if len(other) != len(self._tensors):
                raise ValueError(f"Length mismatch: {len(self._tensors)} vs {len(other)}")
            return SparseTensorList([a - b for a, b in zip(self._tensors, other._tensors)])
        return SparseTensorList([
            SparseTensor(t.values - other, t.row_indices, t.col_indices, t.shape)
            for t in self._tensors
        ])
    
    def __rsub__(self, other):
        return SparseTensorList([
            SparseTensor(other - t.values, t.row_indices, t.col_indices, t.shape)
            for t in self._tensors
        ])
    
    def __mul__(self, other: Union["SparseTensorList", float, int, torch.Tensor]) -> "SparseTensorList":
        """
        Element-wise multiplication.
        
        Parameters
        ----------
        other : SparseTensorList, scalar, or Tensor
            If SparseTensorList: multiply corresponding matrices element-wise.
            If scalar/Tensor: multiply all values.
            
        Returns
        -------
        SparseTensorList
            Result of multiplication.
        """
        if isinstance(other, SparseTensorList):
            if len(other) != len(self._tensors):
                raise ValueError(f"Length mismatch: {len(self._tensors)} vs {len(other)}")
            return SparseTensorList([a * b for a, b in zip(self._tensors, other._tensors)])
        return SparseTensorList([t * other for t in self._tensors])
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other: Union[float, int, torch.Tensor]) -> "SparseTensorList":
        """Element-wise division by scalar."""
        return SparseTensorList([t / other for t in self._tensors])
    
    def __neg__(self) -> "SparseTensorList":
        """Negate all values."""
        return SparseTensorList([-t for t in self._tensors])
    
    def sum(self, axis: Optional[int] = None) -> Union[List[torch.Tensor], torch.Tensor]:
        """
        Sum values in each matrix.
        
        Parameters
        ----------
        axis : int, optional
            If None: sum all values in each matrix, return List[scalar].
            If 0: sum over rows for each matrix.
            If 1: sum over columns for each matrix.
            
        Returns
        -------
        List[torch.Tensor] or torch.Tensor
            If axis is None: List of scalar tensors (one per matrix).
            If axis is 0 or 1: List of 1D tensors.
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> totals = matrices.sum()  # [sum(A1), sum(A2), sum(A3)]
        >>> row_sums = matrices.sum(axis=1)  # [A1.sum(1), A2.sum(1), ...]
        """
        return [t.sum(axis=axis) for t in self._tensors]
    
    def mean(self, axis: Optional[int] = None) -> List[torch.Tensor]:
        """
        Mean of values in each matrix.
        
        Parameters
        ----------
        axis : int, optional
            Same as sum().
            
        Returns
        -------
        List[torch.Tensor]
            List of mean values/vectors.
        """
        return [t.mean(axis=axis) for t in self._tensors]
    
    def max(self) -> List[torch.Tensor]:
        """Maximum value in each matrix."""
        return [t.max() for t in self._tensors]
    
    def min(self) -> List[torch.Tensor]:
        """Minimum value in each matrix."""
        return [t.min() for t in self._tensors]
    
    def abs(self) -> "SparseTensorList":
        """Absolute value of all elements."""
        return SparseTensorList([t.abs() for t in self._tensors])
    
    def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> "SparseTensorList":
        """Clamp values in all matrices."""
        return SparseTensorList([t.clamp(min=min, max=max) for t in self._tensors])
    
    def pow(self, exponent: float) -> "SparseTensorList":
        """Element-wise power."""
        return SparseTensorList([t.pow(exponent) for t in self._tensors])
    
    def sqrt(self) -> "SparseTensorList":
        """Element-wise square root."""
        return SparseTensorList([t.sqrt() for t in self._tensors])
    
    def exp(self) -> "SparseTensorList":
        """Element-wise exponential."""
        return SparseTensorList([t.exp() for t in self._tensors])
    
    def log(self) -> "SparseTensorList":
        """Element-wise natural logarithm."""
        return SparseTensorList([t.log() for t in self._tensors])
    
    # =========================================================================
    # Linear Algebra
    # =========================================================================
    
    def solve(self, b_list: List[torch.Tensor], **kwargs) -> List[torch.Tensor]:
        """
        Solve linear systems for all matrices.
        
        Parameters
        ----------
        b_list : List[torch.Tensor]
            List of right-hand side vectors, one per matrix.
        **kwargs
            Additional arguments passed to SparseTensor.solve().
        
        Returns
        -------
        List[torch.Tensor]
            List of solutions.
        
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> x_list = matrices.solve([b1, b2, b3])
        """
        if len(b_list) != len(self._tensors):
            raise ValueError(f"Expected {len(self._tensors)} RHS vectors, got {len(b_list)}")
        return [t.solve(b, **kwargs) for t, b in zip(self._tensors, b_list)]
    
    def is_symmetric(self, **kwargs) -> List[torch.Tensor]:
        """
        Check symmetry for all matrices.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to SparseTensor.is_symmetric().
        
        Returns
        -------
        List[torch.Tensor]
            List of boolean tensors.
        """
        return [t.is_symmetric(**kwargs) for t in self._tensors]
    
    def is_positive_definite(self, **kwargs) -> List[torch.Tensor]:
        """
        Check positive definiteness for all matrices.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to SparseTensor.is_positive_definite().
        
        Returns
        -------
        List[torch.Tensor]
            List of boolean tensors.
        """
        return [t.is_positive_definite(**kwargs) for t in self._tensors]
    
    def norm(self, ord: Literal['fro', 1, 2] = 'fro') -> List[torch.Tensor]:
        """
        Compute norms for all matrices.
        
        Parameters
        ----------
        ord : {'fro', 1, 2}
            Norm type.
        
        Returns
        -------
        List[torch.Tensor]
            List of norm values.
        """
        return [t.norm(ord=ord) for t in self._tensors]
    
    def eigs(self, k: int = 6, **kwargs) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Compute eigenvalues for all matrices.
        
        Parameters
        ----------
        k : int
            Number of eigenvalues.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
            List of (eigenvalues, eigenvectors) tuples.
        """
        return [t.eigs(k=k, **kwargs) for t in self._tensors]
    
    def eigsh(self, k: int = 6, **kwargs) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Compute eigenvalues for symmetric matrices.
        
        Parameters
        ----------
        k : int
            Number of eigenvalues.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        List[Tuple[torch.Tensor, Optional[torch.Tensor]]]
            List of (eigenvalues, eigenvectors) tuples.
        """
        return [t.eigsh(k=k, **kwargs) for t in self._tensors]
    
    def svd(self, k: int = 6) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Compute SVD for all matrices.
        
        Parameters
        ----------
        k : int
            Number of singular values.
        
        Returns
        -------
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            List of (U, S, Vt) tuples.
        """
        return [t.svd(k=k) for t in self._tensors]
    
    def condition_number(self, ord: int = 2) -> List[torch.Tensor]:
        """
        Compute condition numbers for all matrices.
        
        Parameters
        ----------
        ord : int
            Norm order.
        
        Returns
        -------
        List[torch.Tensor]
            List of condition numbers.
        """
        return [t.condition_number(ord=ord) for t in self._tensors]
    
    def det(self) -> List[torch.Tensor]:
        """
        Compute determinants for all matrices.
        
        Returns
        -------
        List[torch.Tensor]
            List of determinant values.
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3])
        >>> dets = matrices.det()
        >>> print([d.item() for d in dets])
        """
        return [t.det() for t in self._tensors]
    
    def spy(
        self,
        indices: Optional[List[int]] = None,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        """
        Visualize sparsity patterns for multiple matrices in a grid.
        
        Parameters
        ----------
        indices : List[int], optional
            Which matrices to visualize. Default: all.
        ncols : int, optional
            Number of columns in subplot grid. Default: 3.
        figsize : Tuple[float, float], optional
            Figure size. Auto-computed if None.
        **kwargs
            Additional arguments passed to SparseTensor.spy().
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
            
        Examples
        --------
        >>> matrices = SparseTensorList([A1, A2, A3, A4])
        >>> matrices.spy()  # Visualize all in grid
        >>> matrices.spy(indices=[0, 2])  # Visualize specific ones
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for spy(). Install with: pip install matplotlib")
        
        if indices is None:
            indices = list(range(len(self._tensors)))
        
        n = len(indices)
        nrows = (n + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (4 * ncols, 4 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        
        for i, idx in enumerate(indices):
            row, col = i // ncols, i % ncols
            ax = axes[row, col]
            self._tensors[idx].spy(ax=ax, show_colorbar=False, **kwargs)
            M, N = self._tensors[idx].sparse_shape
            ax.set_title(f'[{idx}] {M}×{N}, nnz={self._tensors[idx].nnz:,}', fontsize=10)
        
        # Hide unused subplots
        for i in range(n, nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        return fig
    
    # =========================================================================
    # Conversion Methods
    # =========================================================================
    
    def to_block_diagonal(self) -> "SparseTensor":
        """
        Merge all matrices into a single block-diagonal SparseTensor.
        
        Creates a sparse matrix where each input matrix appears as a block
        on the diagonal: diag(A1, A2, ..., An).
        
        Returns
        -------
        SparseTensor
            Block-diagonal matrix with shape (sum(M_i), sum(N_i)).
            
        Notes
        -----
        The resulting matrix has the structure:
        
        ```
        [A1  0  0  ...]
        [ 0 A2  0  ...]
        [ 0  0 A3  ...]
        [... ... ... ]
        ```
        
        Examples
        --------
        >>> A1 = SparseTensor(val1, row1, col1, (10, 10))
        >>> A2 = SparseTensor(val2, row2, col2, (20, 20))
        >>> stl = SparseTensorList([A1, A2])
        >>> A_block = stl.to_block_diagonal()  # Shape (30, 30)
        """
        if len(self._tensors) == 0:
            raise ValueError("Cannot convert empty SparseTensorList to block diagonal")
        
        if len(self._tensors) == 1:
            return self._tensors[0]
        
        # Compute offsets
        row_offsets = [0]
        col_offsets = [0]
        
        for t in self._tensors:
            M, N = t.sparse_shape
            row_offsets.append(row_offsets[-1] + M)
            col_offsets.append(col_offsets[-1] + N)
        
        total_rows = row_offsets[-1]
        total_cols = col_offsets[-1]
        
        # Concatenate all COO data with offsets
        all_values = []
        all_rows = []
        all_cols = []
        
        for i, t in enumerate(self._tensors):
            all_values.append(t.values)
            all_rows.append(t.row_indices + row_offsets[i])
            all_cols.append(t.col_indices + col_offsets[i])
        
        values = torch.cat(all_values)
        rows = torch.cat(all_rows)
        cols = torch.cat(all_cols)
        
        return SparseTensor(values, rows, cols, (total_rows, total_cols))
    
    @classmethod
    def from_block_diagonal(
        cls,
        sparse: "SparseTensor",
        sizes: List[Tuple[int, int]]
    ) -> "SparseTensorList":
        """
        Split a block-diagonal SparseTensor into a list of matrices.
        
        Parameters
        ----------
        sparse : SparseTensor
            Block-diagonal matrix to split.
        sizes : List[Tuple[int, int]]
            List of (rows, cols) for each block. Must sum to sparse.shape.
            
        Returns
        -------
        SparseTensorList
            List of extracted blocks.
            
        Examples
        --------
        >>> A_block = SparseTensor(val, row, col, (30, 30))
        >>> stl = SparseTensorList.from_block_diagonal(A_block, [(10, 10), (20, 20)])
        >>> print(len(stl))  # 2
        """
        if sparse.is_batched:
            raise NotImplementedError("from_block_diagonal not supported for batched tensors")
        
        # Validate sizes
        total_rows = sum(s[0] for s in sizes)
        total_cols = sum(s[1] for s in sizes)
        
        if (total_rows, total_cols) != sparse.sparse_shape:
            raise ValueError(
                f"Sizes sum to ({total_rows}, {total_cols}) but sparse has shape {sparse.sparse_shape}"
            )
        
        # Compute offsets
        row_offsets = [0]
        col_offsets = [0]
        for m, n in sizes:
            row_offsets.append(row_offsets[-1] + m)
            col_offsets.append(col_offsets[-1] + n)
        
        tensors = []
        row = sparse.row_indices
        col = sparse.col_indices
        val = sparse.values
        
        for i, (m, n) in enumerate(sizes):
            r_start, r_end = row_offsets[i], row_offsets[i + 1]
            c_start, c_end = col_offsets[i], col_offsets[i + 1]
            
            # Find entries in this block
            mask = (row >= r_start) & (row < r_end) & (col >= c_start) & (col < c_end)
            
            block_row = row[mask] - r_start
            block_col = col[mask] - c_start
            block_val = val[mask]
            
            tensors.append(SparseTensor(block_val, block_row, block_col, (m, n)))
        
        return cls(tensors)
    
    @property
    def block_sizes(self) -> List[Tuple[int, int]]:
        """
        Get the (rows, cols) size of each matrix.
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (M, N) tuples.
        """
        return [t.sparse_shape for t in self._tensors]
    
    @property
    def total_nnz(self) -> int:
        """Total number of non-zeros across all matrices."""
        return sum(t.nnz for t in self._tensors)
    
    @property
    def total_shape(self) -> Tuple[int, int]:
        """Shape of the block-diagonal representation."""
        total_rows = sum(t.sparse_shape[0] for t in self._tensors)
        total_cols = sum(t.sparse_shape[1] for t in self._tensors)
        return (total_rows, total_cols)
    
    def __repr__(self) -> str:
        return f"SparseTensorList(n={len(self._tensors)}, device={self.device})"
