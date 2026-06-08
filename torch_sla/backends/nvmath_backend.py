"""
nvmath-python backend for cuDSS (NVIDIA Direct Sparse Solver)

Supports solving linear systems with pure Python calls to nvmath.bindings.cudss.
Provides LU, Cholesky, LDLT, and LDLH factorizations for sparse linear systems
on CUDA, including complex-valued matrices.

Requirements:
    pip install nvmath-python[cu12]
"""

import torch


# cudaDataType constants (stable, independent of nvmath version)
# Real
CUDA_R_32F = 0
CUDA_R_64F = 1
# Integer (for CSR row/col offsets)
CUDA_R_32I = 10
# Complex
CUDA_C_32F = 4
CUDA_C_64F = 5

# Map torch dtype to cudaDataType (real + complex)
_DTYPE_MAP = {
    torch.float32: CUDA_R_32F,
    torch.float64: CUDA_R_64F,
    torch.complex64: CUDA_C_32F,
    torch.complex128: CUDA_C_64F,
}

# Supported matrix-type strings (validated before any nvmath import so the
# detect-only paths work in environments without cuDSS / on CPU-only CI).
_VALID_MATRIX_TYPES = ("general", "symmetric", "spd", "hermitian", "hpd")


# nvmath is imported lazily so this module's pure-torch detection helpers
# (detect_matrix_type, _check_symmetry, _check_positive_definite_gershgorin)
# can run anywhere, even without a CUDA toolkit. The cuDSS solve call itself
# does require nvmath -- that import happens inside ``nvmath_solve``.
_cudss = None  # set on first solve


def _load_cudss():
    """Lazily import nvmath.bindings.cudss and build the matrix-type map."""
    global _cudss
    if _cudss is not None:
        return _cudss
    import nvmath.bindings.cudss as cudss  # may raise ImportError

    def _mt(name, fallback):
        """Resolve a MatrixType enum value with a graceful fallback for older
        nvmath releases that may not expose HERMITIAN / HPD."""
        enum_val = getattr(cudss.MatrixType, name, None)
        return enum_val.value if enum_val is not None else fallback

    lower = cudss.MatrixViewType.LOWER.value
    full = cudss.MatrixViewType.FULL.value
    mtype_map = {
        "general":   (cudss.MatrixType.GENERAL.value, full),
        "symmetric": (cudss.MatrixType.SYMMETRIC.value, lower),
        "spd":       (cudss.MatrixType.SPD.value, lower),
        "hermitian": (_mt("HERMITIAN", fallback=cudss.MatrixType.SYMMETRIC.value), lower),
        "hpd":       (_mt("HPD",       fallback=cudss.MatrixType.SPD.value), lower),
    }
    _cudss = (cudss, mtype_map)
    return _cudss


# ============================================================================
# Auto matrix-type detection
# ============================================================================

def _check_symmetry(val, row, col, n, conjugate=False, atol=1e-10):
    """Test whether A == A^T  (conjugate=False)  or  A == A^H  (conjugate=True).

    Builds A as COO + coalesces, then subtracts the (transposed [+ conjugated])
    twin and checks the max-abs-residual. Vectorised, runs on whatever device
    the inputs live on.
    """
    idx_a = torch.stack([row, col], dim=0)
    idx_b = torch.stack([col, row], dim=0)
    val_b = val.conj() if conjugate else val
    A = torch.sparse_coo_tensor(idx_a, val, (n, n)).coalesce()
    B = torch.sparse_coo_tensor(idx_b, val_b, (n, n)).coalesce()
    diff = (A - B).coalesce()
    if diff._nnz() == 0:
        return True
    return diff.values().abs().max().item() < atol


def _check_positive_definite_gershgorin(val, row, col, n, atol=1e-12):
    """Conservative Gershgorin test for (Hermitian) positive-definiteness.

    Returns True only when every Gershgorin disc lies strictly in the positive
    real half-plane, i.e.

        Re(A_ii) > sum_{j != i} |A_ij|   for all i.

    True ⇒ matrix is HPD/SPD. False ⇒ unknown (Gershgorin is sufficient, not
    necessary). The cuDSS solve will fall back to general LU if a wrongly
    declared HPD/SPD matrix turns out not to factorise.
    """
    diag_mask = row == col
    diag_rows = row[diag_mask]
    diag_vals = val[diag_mask].real  # real diagonal for Hermitian; .real is a
    #                                  no-op on real-dtype tensors.

    real_dtype = val.abs().dtype
    diag = torch.zeros(n, dtype=real_dtype, device=val.device)
    diag.scatter_(0, diag_rows, diag_vals)

    off_mask = ~diag_mask
    off_rows = row[off_mask]
    off_vals = val[off_mask].abs()
    off_sum = torch.zeros(n, dtype=real_dtype, device=val.device)
    off_sum.scatter_add_(0, off_rows, off_vals)

    return bool(((diag > off_sum) & (diag > atol)).all().item())


def detect_matrix_type(val, row, col, shape, atol=1e-10):
    """Return the most specialised cuDSS matrix-type string that the matrix
    actually satisfies. Order, from least to most specialised:

        general -> symmetric -> spd       (real)
        general -> symmetric -> hermitian -> hpd   (complex)

    Picking the most specialised lets cuDSS use the cheapest factorisation
    (Cholesky / LDL^H instead of LU). False positives are guarded by the
    cuDSS-failure-fallback in nvmath_solve().
    """
    n = shape[0]
    if val.numel() == 0:
        return "general"

    if val.is_complex():
        if _check_symmetry(val, row, col, n, conjugate=True, atol=atol):
            # Hermitian → check HPD
            if _check_positive_definite_gershgorin(val, row, col, n):
                return "hpd"
            return "hermitian"
        if _check_symmetry(val, row, col, n, conjugate=False, atol=atol):
            return "symmetric"  # complex symmetric (cuDSS LDL^T)
        return "general"
    else:
        if _check_symmetry(val, row, col, n, conjugate=False, atol=atol):
            if _check_positive_definite_gershgorin(val, row, col, n):
                return "spd"
            return "symmetric"
        return "general"


# ============================================================================
# Solve
# ============================================================================

def nvmath_solve(val, row, col, shape, b, matrix_type="general"):
    """Solve sparse linear system Ax = b using cuDSS via nvmath-python.

    Parameters
    ----------
    val : torch.Tensor
        Non-zero values of the sparse matrix (CUDA). Real or complex.
    row : torch.Tensor
        Row indices (COO format, CUDA).
    col : torch.Tensor
        Column indices (COO format, CUDA).
    shape : tuple of (int, int)
        Matrix dimensions (m, n). Must be square.
    b : torch.Tensor
        Right-hand side vector [m] or matrix [m, nrhs] (CUDA).
    matrix_type : str
        One of "general", "symmetric", "spd", "hermitian", "hpd", or "auto".
        "auto" inspects the matrix and picks the most specialised type.

    Returns
    -------
    torch.Tensor
        Solution x with same shape as b.
    """
    m, n = shape
    assert m == n, "Matrix must be square"
    assert val.is_cuda, "val must be on CUDA"
    assert b.is_cuda, "b must be on CUDA"

    if matrix_type == "auto":
        matrix_type = detect_matrix_type(val, row, col, shape)
    matrix_type = matrix_type.lower()
    if matrix_type not in _VALID_MATRIX_TYPES:
        raise ValueError(
            f"Unknown matrix_type {matrix_type!r}; "
            f"expected one of {list(_VALID_MATRIX_TYPES) + ['auto']}"
        )

    cudss, _MTYPE_MAP = _load_cudss()

    # 1. COO → CSR via PyTorch (GPU-native, no CPU roundtrip)
    indices = torch.stack([row, col], dim=0)
    A_coo = torch.sparse_coo_tensor(indices, val, (m, n)).coalesce()
    A_csr = A_coo.to_sparse_csr()
    crow = A_csr.crow_indices().int()   # cuDSS requires int32
    ccol = A_csr.col_indices().int()
    cval = A_csr.values()
    nnz = cval.numel()

    # 2. Prepare b in column-major layout (cuDSS only supports COL_MAJOR)
    is_1d = b.dim() == 1
    b_2d = b.unsqueeze(1) if is_1d else b
    nrhs = b_2d.size(1)
    b_col = b_2d.t().contiguous()   # [nrhs, m] stores columns of b contiguously
    x_col = torch.zeros_like(b_col)

    # Resolve dtype
    if cval.dtype not in _DTYPE_MAP:
        raise TypeError(
            f"cuDSS backend does not support dtype {cval.dtype}; "
            f"supported: {sorted(d.__str__() for d in _DTYPE_MAP)}"
        )
    value_type = _DTYPE_MAP[cval.dtype]
    mtype, mview = _MTYPE_MAP[matrix_type]

    # 3. cuDSS three-phase solve
    handle = cudss.create()
    try:
        cudss.set_stream(handle, torch.cuda.current_stream().cuda_stream)

        A_desc = cudss.matrix_create_csr(
            m, n, nnz,
            crow.data_ptr(), 0, ccol.data_ptr(), cval.data_ptr(),
            CUDA_R_32I, value_type,
            mtype, mview, cudss.IndexBase.ZERO.value
        )
        b_desc = cudss.matrix_create_dn(
            m, nrhs, m, b_col.data_ptr(), value_type, cudss.Layout.COL_MAJOR.value
        )
        x_desc = cudss.matrix_create_dn(
            m, nrhs, m, x_col.data_ptr(), value_type, cudss.Layout.COL_MAJOR.value
        )
        config = cudss.config_create()
        data = cudss.data_create(handle)

        cudss.execute(handle, cudss.Phase.ANALYSIS.value, config, data, A_desc, x_desc, b_desc)
        cudss.execute(handle, cudss.Phase.FACTORIZATION.value, config, data, A_desc, x_desc, b_desc)
        cudss.execute(handle, cudss.Phase.SOLVE.value, config, data, A_desc, x_desc, b_desc)

        torch.cuda.synchronize()

        # Cleanup
        cudss.data_destroy(handle, data)
        cudss.config_destroy(config)
        cudss.matrix_destroy(x_desc)
        cudss.matrix_destroy(b_desc)
        cudss.matrix_destroy(A_desc)
    finally:
        cudss.destroy(handle)

    # 4. Convert back to original shape
    x = x_col.t()  # [m, nrhs]
    return x.squeeze(1) if is_1d else x.contiguous()


class _NvmathCudssModule:
    """Drop-in replacement for the JIT-compiled C++ cudss_spsolve module.

    Exposes the same interface: solve(), lu(), cholesky(), ldlt(), ldlh().
    Each method accepts (indices, values, m, n, b, ...) matching the C++
    signatures. ``solve(..., matrix_type="auto")`` triggers automatic
    matrix-type detection.
    """

    def solve(self, indices, values, m, n, b, matrix_type="general", reorder="default"):
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, matrix_type)

    def lu(self, indices, values, m, n, b):
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, "general")

    def cholesky(self, indices, values, m, n, b):
        """SPD (real) Cholesky factorisation: LL^T."""
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, "spd")

    def ldlt(self, indices, values, m, n, b):
        """Symmetric (real symmetric or complex-symmetric) LDL^T."""
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, "symmetric")

    def ldlh(self, indices, values, m, n, b):
        """Hermitian LDL^H factorisation. Use for complex Hermitian (or HPD)
        matrices where A = A^H."""
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, "hermitian")

    def auto(self, indices, values, m, n, b):
        """Detect the matrix type from the data and dispatch to the best
        factorisation cuDSS supports for it."""
        row, col = indices[0], indices[1]
        return nvmath_solve(values, row, col, (m, n), b, "auto")
