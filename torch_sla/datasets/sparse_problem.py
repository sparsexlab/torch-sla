"""The :class:`SparseProblem` container -- the schema shared by every dataset
generator and the SuiteSparse loader.

Kept separate from the problem *generators* (:mod:`torch_sla.datasets.problems`)
and the loaders (:mod:`torch_sla.datasets.suitesparse`) so the data schema has a
single, obvious home.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

__all__ = ["SparseProblem"]


@dataclass
class SparseProblem:
    """A single sparse benchmark / test problem in COO form.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. ``"poisson_2d(m=31)"``).
    val, row, col : Tensor
        COO values (float64 / complex128), row indices (long), col indices
        (long).
    shape : tuple
        ``(n, n)``.
    rhs : Tensor or None
        A right-hand side ``b`` if the problem defines one.
    exact : Tensor or None
        Exact / manufactured solution ``u*`` if known.
    properties : dict
        Boolean-ish flags such as ``{'spd': True, 'symmetric': True,
        'complex': False}``.
    meta : dict
        Free-form metadata, e.g. ``{'dof': n, 'source': 'pde',
        'pde': 'poisson2d'}``.
    """

    name: str
    val: Tensor
    row: Tensor
    col: Tensor
    shape: tuple
    rhs: Optional[Tensor] = None
    exact: Optional[Tensor] = None
    properties: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def coo(self):
        """Return ``(val, row, col, shape)``."""
        return self.val, self.row, self.col, self.shape

    def nnz(self):
        """Number of stored entries."""
        return self.val.numel()

    def to_dense(self) -> Tensor:
        """Materialise a dense matrix (mostly for the RHS / reference work)."""
        n, m = self.shape
        A = torch.zeros(n, m, dtype=self.val.dtype)
        A.index_put_((self.row, self.col), self.val, accumulate=True)
        return A

    def matvec(self, x: Tensor) -> Tensor:
        """Compute ``A @ x`` from the COO triple without densifying."""
        n, _ = self.shape
        out = torch.zeros(n, dtype=torch.result_type(self.val, x))
        out.index_add_(0, self.row, self.val * x[self.col])
        return out
