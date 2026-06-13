"""Element-wise + arithmetic operations for SparseTensor."""
from __future__ import annotations
from typing import Union
import torch

from .core import SparseTensor  # noqa: E402  # cross-module use


def _apply_elementwise(self, func, *args, **kwargs) -> "SparseTensor":
    """Apply element-wise function to values.
    
    Returns the same type as self to support subclassing.
    Subclasses should ensure their __init__ accepts (values, row_indices, col_indices, shape)
    or override this method.
    """
    new_values = func(self.values, *args, **kwargs)
    # Use type(self) to preserve subclass type
    try:
        return type(self)(
            new_values, self.row_indices, self.col_indices, self._shape
        )
    except TypeError:
        # Fallback for subclasses with incompatible __init__
        return SparseTensor(
            new_values, self.row_indices, self.col_indices,
            self._shape, sparse_dim=self._sparse_dim
        )

# Arithmetic operations
def __add__(self, other: Union[torch.Tensor, "SparseTensor", float, int]) -> "SparseTensor":
    """Element-wise addition. For SparseTensor + SparseTensor, patterns must match."""
    if isinstance(other, SparseTensor):
        if not torch.equal(self.row_indices, other.row_indices) or \
           not torch.equal(self.col_indices, other.col_indices):
            raise ValueError("SparseTensor addition requires matching sparsity patterns")
        return self._apply_elementwise(lambda v: v + other.values)
    return self._apply_elementwise(lambda v: v + other)

def __radd__(self, other):
    return self.__add__(other)

def __sub__(self, other: Union[torch.Tensor, "SparseTensor", float, int]) -> "SparseTensor":
    if isinstance(other, SparseTensor):
        if not torch.equal(self.row_indices, other.row_indices) or \
           not torch.equal(self.col_indices, other.col_indices):
            raise ValueError("SparseTensor subtraction requires matching sparsity patterns")
        return self._apply_elementwise(lambda v: v - other.values)
    return self._apply_elementwise(lambda v: v - other)

def __rsub__(self, other):
    return self._apply_elementwise(lambda v: other - v)

def __mul__(self, other: Union[torch.Tensor, "SparseTensor", float, int]) -> "SparseTensor":
    """Element-wise multiplication (Hadamard product for sparse tensors)."""
    if isinstance(other, SparseTensor):
        if not torch.equal(self.row_indices, other.row_indices) or \
           not torch.equal(self.col_indices, other.col_indices):
            raise ValueError("SparseTensor multiplication requires matching sparsity patterns")
        return self._apply_elementwise(lambda v: v * other.values)
    return self._apply_elementwise(lambda v: v * other)

def __rmul__(self, other):
    return self.__mul__(other)

def __truediv__(self, other: Union[torch.Tensor, "SparseTensor", float, int]) -> "SparseTensor":
    if isinstance(other, SparseTensor):
        if not torch.equal(self.row_indices, other.row_indices) or \
           not torch.equal(self.col_indices, other.col_indices):
            raise ValueError("SparseTensor division requires matching sparsity patterns")
        return self._apply_elementwise(lambda v: v / other.values)
    return self._apply_elementwise(lambda v: v / other)

def __rtruediv__(self, other):
    return self._apply_elementwise(lambda v: other / v)

def __floordiv__(self, other):
    if isinstance(other, SparseTensor):
        return self._apply_elementwise(lambda v: v // other.values)
    return self._apply_elementwise(lambda v: v // other)

def __pow__(self, exponent: Union[float, int, torch.Tensor]) -> "SparseTensor":
    return self._apply_elementwise(lambda v: v ** exponent)

def __neg__(self) -> "SparseTensor":
    return self._apply_elementwise(lambda v: -v)

def __pos__(self) -> "SparseTensor":
    return self

def __abs__(self) -> "SparseTensor":
    return self._apply_elementwise(torch.abs)

# Math functions - directly delegate to values
def _abs_impl(self) -> "SparseTensor":
    """Element-wise absolute value."""
    return self._apply_elementwise(torch.abs)

def _sqrt_impl(self) -> "SparseTensor":
    """Element-wise square root."""
    return self._apply_elementwise(torch.sqrt)

def _square_impl(self) -> "SparseTensor":
    """Element-wise square."""
    return self._apply_elementwise(torch.square)

def _exp_impl(self) -> "SparseTensor":
    """Element-wise exponential."""
    return self._apply_elementwise(torch.exp)

def _log_impl(self) -> "SparseTensor":
    """Element-wise natural logarithm."""
    return self._apply_elementwise(torch.log)

def log10(self) -> "SparseTensor":
    """Element-wise base-10 logarithm."""
    return self._apply_elementwise(torch.log10)

def log2(self) -> "SparseTensor":
    """Element-wise base-2 logarithm."""
    return self._apply_elementwise(torch.log2)

def sin(self) -> "SparseTensor":
    """Element-wise sine."""
    return self._apply_elementwise(torch.sin)

def cos(self) -> "SparseTensor":
    """Element-wise cosine."""
    return self._apply_elementwise(torch.cos)

def tan(self) -> "SparseTensor":
    """Element-wise tangent."""
    return self._apply_elementwise(torch.tan)

def sinh(self) -> "SparseTensor":
    """Element-wise hyperbolic sine."""
    return self._apply_elementwise(torch.sinh)

def cosh(self) -> "SparseTensor":
    """Element-wise hyperbolic cosine."""
    return self._apply_elementwise(torch.cosh)

def tanh(self) -> "SparseTensor":
    """Element-wise hyperbolic tangent."""
    return self._apply_elementwise(torch.tanh)

def sigmoid(self) -> "SparseTensor":
    """Element-wise sigmoid."""
    return self._apply_elementwise(torch.sigmoid)

def relu(self) -> "SparseTensor":
    """Element-wise ReLU."""
    return self._apply_elementwise(torch.relu)

def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> "SparseTensor":
    """Element-wise clamp."""
    return self._apply_elementwise(lambda v: torch.clamp(v, min=min, max=max))

def sign(self) -> "SparseTensor":
    """Element-wise sign."""
    return self._apply_elementwise(torch.sign)

def floor(self) -> "SparseTensor":
    """Element-wise floor."""
    return self._apply_elementwise(torch.floor)

def ceil(self) -> "SparseTensor":
    """Element-wise ceil."""
    return self._apply_elementwise(torch.ceil)

def round(self) -> "SparseTensor":
    """Element-wise round."""
    return self._apply_elementwise(torch.round)

def reciprocal(self) -> "SparseTensor":
    """Element-wise reciprocal (1/x)."""
    return self._apply_elementwise(torch.reciprocal)

def pow(self, exponent: Union[float, int, torch.Tensor]) -> "SparseTensor":
    """Element-wise power."""
    return self._apply_elementwise(lambda v: torch.pow(v, exponent))

# Comparison operations (return SparseTensor with bool values)
def __eq__(self, other) -> "SparseTensor":
    if isinstance(other, SparseTensor):
        return self._apply_elementwise(lambda v: v == other.values)
    return self._apply_elementwise(lambda v: v == other)

def __ne__(self, other) -> "SparseTensor":
    if isinstance(other, SparseTensor):
        return self._apply_elementwise(lambda v: v != other.values)
    return self._apply_elementwise(lambda v: v != other)

def __lt__(self, other) -> "SparseTensor":
    if isinstance(other, SparseTensor):
        return self._apply_elementwise(lambda v: v < other.values)
    return self._apply_elementwise(lambda v: v < other)

def __le__(self, other) -> "SparseTensor":
    if isinstance(other, SparseTensor):
        return self._apply_elementwise(lambda v: v <= other.values)
    return self._apply_elementwise(lambda v: v <= other)

def __gt__(self, other) -> "SparseTensor":
    if isinstance(other, SparseTensor):
        return self._apply_elementwise(lambda v: v > other.values)
    return self._apply_elementwise(lambda v: v > other)

def __ge__(self, other) -> "SparseTensor":
    if isinstance(other, SparseTensor):
        return self._apply_elementwise(lambda v: v >= other.values)
    return self._apply_elementwise(lambda v: v >= other)

# Boolean operations
def logical_not(self) -> "SparseTensor":
    """Element-wise logical NOT."""
    return self._apply_elementwise(torch.logical_not)

def logical_and(self, other: "SparseTensor") -> "SparseTensor":
    """Element-wise logical AND."""
    return self._apply_elementwise(lambda v: torch.logical_and(v, other.values))

def logical_or(self, other: "SparseTensor") -> "SparseTensor":
    """Element-wise logical OR."""
    return self._apply_elementwise(lambda v: torch.logical_or(v, other.values))

def logical_xor(self, other: "SparseTensor") -> "SparseTensor":
    """Element-wise logical XOR."""
    return self._apply_elementwise(lambda v: torch.logical_xor(v, other.values))

# Type checking
def isnan(self) -> "SparseTensor":
    """Element-wise isnan check."""
    return self._apply_elementwise(torch.isnan)

def isinf(self) -> "SparseTensor":
    """Element-wise isinf check."""
    return self._apply_elementwise(torch.isinf)

def isfinite(self) -> "SparseTensor":
    """Element-wise isfinite check."""
    return self._apply_elementwise(torch.isfinite)

# Gradient-related
def detach(self) -> "SparseTensor":
    """Detach from computation graph. Preserves subclass type."""
    try:
        return type(self)(
            self.values.detach(),
            self.row_indices,
            self.col_indices,
            self._shape
        )
    except TypeError:
        return SparseTensor(
            self.values.detach(),
            self.row_indices,
            self.col_indices,
            self._shape,
            sparse_dim=self._sparse_dim
        )

def requires_grad_(self, requires_grad: bool = True) -> "SparseTensor":
    """Enable/disable gradient tracking."""
    self.values.requires_grad_(requires_grad)
    return self

@property
def requires_grad(self) -> bool:
    """Whether gradient tracking is enabled."""
    return self.values.requires_grad

@property
def grad(self) -> Optional[torch.Tensor]:
    """Gradient of values if available."""
    return self.values.grad

def clone(self) -> "SparseTensor":
    """Create a copy of this SparseTensor. Preserves subclass type."""
    try:
        return type(self)(
            self.values.clone(),
            self.row_indices.clone(),
            self.col_indices.clone(),
            self._shape
        )
    except TypeError:
        return SparseTensor(
            self.values.clone(),
            self.row_indices.clone(),
            self.col_indices.clone(),
            self._shape,
            sparse_dim=self._sparse_dim
        )

def contiguous(self) -> "SparseTensor":
    """Make values contiguous in memory. Preserves subclass type."""
    try:
        return type(self)(
            self.values.contiguous(),
            self.row_indices.contiguous(),
            self.col_indices.contiguous(),
            self._shape
        )
    except TypeError:
        return SparseTensor(
            self.values.contiguous(),
            self.row_indices.contiguous(),
            self.col_indices.contiguous(),
            self._shape,
            sparse_dim=self._sparse_dim
        )

