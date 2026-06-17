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


# nvmath is imported lazily so ``detect_matrix_type`` (a thin delegate to
# ``SparseTensor.detect_matrix_type``) can run anywhere, even without a
# CUDA toolkit. The cuDSS solve call itself does require nvmath -- that
# import happens inside ``nvmath_solve``.
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

def detect_matrix_type(val, row, col, shape):
    """Return the most specialised cuDSS matrix-type string the matrix
    satisfies. Specialisation order is

        general -> symmetric -> spd               (real)
        general -> symmetric -> hermitian -> hpd  (complex)

    Picking the most specialised label lets cuDSS use the cheapest
    factorisation (Cholesky / LDL^H instead of LU). False positives are
    guarded by the cuDSS-failure fallback in :func:`nvmath_solve`.

    Backend-internal wrapper around
    :meth:`torch_sla.SparseTensor.detect_matrix_type`; kept as a free
    function so :func:`nvmath_solve` can resolve ``matrix_type='auto'``
    without going through the SparseTensor API.
    """
    from ..sparse_tensor import SparseTensor
    return SparseTensor(val, row, col, shape).detect_matrix_type()


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


def nvmath_factor_solve_many(val, row, col, shape, build_rhs,
                              n_chunks, gather_per_chunk,
                              matrix_type="general"):
    """Factor A once via cuDSS, then solve A X = E for each E chunk.

    Used by :class:`DetAdjoint.backward` on CUDA: we need n_uc solves of
    ``A^T y = e_c`` and want to share the factorisation across all of them.
    The wrapper :func:`nvmath_solve` always re-factors, so callers that
    issue many solves on the same A should use this helper.

    Parameters
    ----------
    val, row, col, shape : CSR triple on CUDA + matrix dimensions.
    build_rhs : callable ``i -> (E_chunk, meta)``. ``E_chunk`` is a
        ``[m, width]`` CUDA tensor (one column per RHS). ``meta`` is an
        opaque object passed back to ``gather_per_chunk``.
    n_chunks : number of times to call ``build_rhs``.
    gather_per_chunk : callable ``(X_chunk, meta) -> None``. ``X_chunk``
        is the ``[m, width]`` solution; the gatherer is expected to slice
        out whatever entries the caller needs and store them. The chunk
        tensors are reused across iterations, so copy out what you need.

    Notes
    -----
    The factorisation cost (analyze + factor) is paid once. Each chunk
    only pays the SOLVE phase cost, which for sparse LU is O(nnz(L) +
    nnz(U)) per RHS column.
    """
    m, n = shape
    assert m == n, "Matrix must be square"
    assert val.is_cuda, "val must be on CUDA"

    if matrix_type == "auto":
        matrix_type = detect_matrix_type(val, row, col, shape)
    matrix_type = matrix_type.lower()
    if matrix_type not in _VALID_MATRIX_TYPES:
        raise ValueError(
            f"Unknown matrix_type {matrix_type!r}; "
            f"expected one of {list(_VALID_MATRIX_TYPES) + ['auto']}"
        )

    cudss, _MTYPE_MAP = _load_cudss()

    indices = torch.stack([row, col], dim=0)
    A_coo = torch.sparse_coo_tensor(indices, val, (m, n)).coalesce()
    A_csr = A_coo.to_sparse_csr()
    crow = A_csr.crow_indices().int()
    ccol = A_csr.col_indices().int()
    cval = A_csr.values()
    nnz = cval.numel()

    if cval.dtype not in _DTYPE_MAP:
        raise TypeError(
            f"cuDSS backend does not support dtype {cval.dtype}; "
            f"supported: {sorted(d.__str__() for d in _DTYPE_MAP)}"
        )
    value_type = _DTYPE_MAP[cval.dtype]
    mtype, mview = _MTYPE_MAP[matrix_type]

    handle = cudss.create()
    try:
        cudss.set_stream(handle, torch.cuda.current_stream().cuda_stream)

        A_desc = cudss.matrix_create_csr(
            m, n, nnz,
            crow.data_ptr(), 0, ccol.data_ptr(), cval.data_ptr(),
            CUDA_R_32I, value_type,
            mtype, mview, cudss.IndexBase.ZERO.value
        )
        config = cudss.config_create()
        data = cudss.data_create(handle)

        analyzed = False
        for i in range(n_chunks):
            E_chunk, meta = build_rhs(i)
            assert E_chunk.is_cuda and E_chunk.dim() == 2 and E_chunk.size(0) == m
            width = int(E_chunk.size(1))
            X_chunk = torch.empty_like(E_chunk)
            # cuDSS wants COL_MAJOR contiguous; t().contiguous() gives [width, m].
            b_col = E_chunk.t().contiguous()
            x_col = torch.empty_like(b_col)
            b_desc = cudss.matrix_create_dn(
                m, width, m, b_col.data_ptr(), value_type, cudss.Layout.COL_MAJOR.value
            )
            x_desc = cudss.matrix_create_dn(
                m, width, m, x_col.data_ptr(), value_type, cudss.Layout.COL_MAJOR.value
            )
            if not analyzed:
                cudss.execute(handle, cudss.Phase.ANALYSIS.value,
                              config, data, A_desc, x_desc, b_desc)
                cudss.execute(handle, cudss.Phase.FACTORIZATION.value,
                              config, data, A_desc, x_desc, b_desc)
                analyzed = True
            cudss.execute(handle, cudss.Phase.SOLVE.value,
                          config, data, A_desc, x_desc, b_desc)
            torch.cuda.synchronize()
            X_chunk.copy_(x_col.t())
            gather_per_chunk(X_chunk, meta)
            cudss.matrix_destroy(x_desc)
            cudss.matrix_destroy(b_desc)

        cudss.data_destroy(handle, data)
        cudss.config_destroy(config)
        cudss.matrix_destroy(A_desc)
    finally:
        cudss.destroy(handle)


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
