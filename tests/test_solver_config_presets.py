"""Unit tests for SolverConfig chainable preset builders + auto-detect.

Pure construction checks -- no actual solve happens, no GPU required.
"""
from __future__ import annotations

import pytest
import torch

import torch_sla
from torch_sla import SolverConfig, SparseTensor


# ---------------------------------------------------------------- spd
def test_spd_gpu_iterative_picks_amgx_pcg_amg():
    cfg = SolverConfig.spd().gpu()
    assert cfg.backend == "amgx"
    assert cfg.method == "pcg"
    assert cfg.preconditioner == "amg"


def test_spd_gpu_direct_picks_cudss_cholesky():
    cfg = SolverConfig.spd().gpu().direct()
    assert cfg.backend == "cudss"
    assert cfg.method == "cholesky"


def test_spd_cpu_iterative_picks_pyamg_amg():
    cfg = SolverConfig.spd().cpu()
    assert cfg.backend == "pyamg"
    assert cfg.method == "amg"


def test_spd_cpu_direct_picks_scipy_cholesky():
    cfg = SolverConfig.spd().cpu().direct()
    assert cfg.backend == "scipy"
    assert cfg.method == "cholesky"


def test_spd_bare_picks_auto_device():
    """``spd()`` alone uses whatever device is the current default."""
    cfg = SolverConfig.spd()
    expected = "amgx" if torch.cuda.is_available() else "pyamg"
    assert cfg.backend == expected


# ---------------------------------------------------------------- general
def test_general_gpu_picks_amgx_pbicgstab_dilu():
    cfg = SolverConfig.general().gpu()
    assert cfg.backend == "amgx"
    assert cfg.method == "pbicgstab"
    assert cfg.preconditioner == "multicolor_dilu"


def test_general_gpu_direct_picks_cudss_lu():
    cfg = SolverConfig.general().gpu().direct()
    assert cfg.backend == "cudss"
    assert cfg.method == "lu"


def test_general_cpu_picks_scipy_bicgstab():
    cfg = SolverConfig.general().cpu()
    assert cfg.backend == "scipy"
    assert cfg.method == "bicgstab"


def test_general_cpu_direct_picks_scipy_lu():
    cfg = SolverConfig.general().cpu().direct()
    assert cfg.backend == "scipy"
    assert cfg.method == "lu"


# ---------------------------------------------------------------- indefinite
def test_indefinite_gpu_picks_amgx_block_jacobi():
    cfg = SolverConfig.indefinite().gpu()
    assert cfg.backend == "amgx"
    assert cfg.preconditioner == "block_jacobi"


def test_indefinite_gpu_direct_picks_cudss_ldlt():
    cfg = SolverConfig.indefinite().gpu().direct()
    assert cfg.backend == "cudss"
    assert cfg.method == "ldlt"


def test_indefinite_cpu_picks_scipy_minres():
    cfg = SolverConfig.indefinite().cpu()
    assert cfg.backend == "scipy"
    assert cfg.method == "minres"


# ---------------------------------------------------------------- conv-diff
def test_convection_diffusion_gpu_picks_fgmres():
    cfg = SolverConfig.convection_diffusion().gpu()
    assert cfg.backend == "amgx"
    assert cfg.method == "fgmres"
    assert cfg.preconditioner == "multicolor_dilu"


def test_convection_diffusion_cpu_picks_scipy_gmres():
    cfg = SolverConfig.convection_diffusion().cpu()
    assert cfg.backend == "scipy"
    assert cfg.method == "gmres"


def test_convection_diffusion_direct_falls_back_to_general_direct():
    """``.direct()`` on conv-diff isn't realistic; we fall back to the
    general direct path (cuDSS / SciPy LU) rather than erroring."""
    cfg = SolverConfig.convection_diffusion().gpu().direct()
    assert cfg.backend == "cudss"
    assert cfg.method == "lu"


# ---------------------------------------------------------------- axis order
def test_axis_order_irrelevant_gpu_then_direct():
    a = SolverConfig.spd().gpu().direct()
    b = SolverConfig.spd().direct().gpu()
    assert a == b
    assert a.backend == "cudss"


def test_iterative_undoes_direct():
    cfg = SolverConfig.spd().gpu().direct().iterative()
    assert cfg.backend == "amgx"
    assert cfg.method == "pcg"


def test_cpu_undoes_gpu():
    cfg = SolverConfig.spd().gpu().direct().cpu()
    assert cfg.backend == "scipy"
    assert cfg.method == "cholesky"


# ---------------------------------------------------------------- modifiers
def test_high_accuracy_chains_and_preserves_axis_choice():
    cfg = SolverConfig.spd().gpu().high_accuracy()
    assert cfg.backend == "amgx"
    assert cfg.method == "pcg"
    assert cfg.preconditioner == "amg"
    assert cfg.atol == 1e-12
    assert cfg.rtol == 1e-10
    assert cfg.maxiter == 5000


def test_high_accuracy_custom_atol():
    cfg = SolverConfig.general().gpu().high_accuracy(atol=1e-13, maxiter=10_000)
    assert cfg.atol == 1e-13
    assert cfg.maxiter == 10_000
    assert cfg.method == "pbicgstab"


def test_replace_overrides_arbitrary_fields():
    cfg = SolverConfig.general().gpu().replace(method="fgmres")
    assert cfg.method == "fgmres"
    assert cfg.backend == "amgx"
    assert cfg.preconditioner == "multicolor_dilu"


def test_chain_axis_then_modifier_then_modifier():
    cfg = (SolverConfig.spd().gpu()
                       .high_accuracy()
                       .replace(maxiter=20_000))
    assert cfg.backend == "amgx"
    assert cfg.atol == 1e-12
    assert cfg.maxiter == 20_000


# ---------------------------------------------------------------- scope
def test_preset_works_as_context_manager():
    """A preset is just a SolverConfig, so it composes with the existing
    context-manager + decorator + LIFO scope machinery."""
    with SolverConfig.spd().cpu().direct() as cfg:
        assert cfg.backend == "scipy"
        assert cfg.method == "cholesky"


def test_private_axis_state_excluded_from_scope_kwargs():
    """The private chain-state fields must NOT leak into the scope-stack
    kwargs dict -- only the public fields participate in default merging."""
    cfg = SolverConfig.spd().gpu()
    kwargs = cfg._kwargs()
    assert "_kind" not in kwargs
    assert "_device" not in kwargs
    assert "_direct" not in kwargs
    assert kwargs.get("backend") == "amgx"


def test_private_axis_state_excluded_from_equality():
    """Two configs whose public fields match should compare equal even
    if they were built via different chain paths."""
    a = SolverConfig.spd().gpu()
    b = SolverConfig(backend="amgx", method="pcg",
                     preconditioner="amg",
                     atol=1e-9, maxiter=500)
    assert a == b


# ---------------------------------------------------------------- auto
def _make_spd_matrix(n: int = 32, device="cpu") -> SparseTensor:
    """Build a strictly diagonally-dominant SPD matrix (diag=4 instead
    of 2 so detect_matrix_type's Gershgorin check trips cleanly)."""
    device = torch.device(device)
    diag_idx = torch.arange(n, device=device)
    row = torch.cat([diag_idx, diag_idx[:-1], diag_idx[1:]])
    col = torch.cat([diag_idx, diag_idx[1:], diag_idx[:-1]])
    val = torch.cat([
        torch.full((n,), 4.0, device=device),
        torch.full((n - 1,), -1.0, device=device),
        torch.full((n - 1,), -1.0, device=device),
    ])
    return SparseTensor(val, row, col, (n, n))


def _make_general_matrix(n: int = 32, device="cpu") -> SparseTensor:
    device = torch.device(device)
    diag_idx = torch.arange(n, device=device)
    row = torch.cat([diag_idx, diag_idx[:-1]])
    col = torch.cat([diag_idx, diag_idx[1:]])
    val = torch.cat([
        torch.full((n,), 3.0, device=device),
        torch.full((n - 1,), -1.0, device=device),
    ])
    return SparseTensor(val, row, col, (n, n))


def test_auto_spd_cpu_routes_to_pyamg_or_scipy():
    A = _make_spd_matrix(n=32, device="cpu")
    cfg = SolverConfig.auto(A)
    assert cfg.backend != "amgx"
    assert cfg.backend in {"scipy", "pyamg"}


def test_auto_general_cpu_routes_to_scipy():
    A = _make_general_matrix(n=32, device="cpu")
    cfg = SolverConfig.auto(A)
    assert cfg.backend == "scipy"


def test_auto_respects_explicit_device_override():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA -- this test verifies the override works")
    A = _make_spd_matrix(n=32, device="cuda")
    cfg = SolverConfig.auto(A, device="cpu")
    assert cfg.backend != "amgx"
    assert cfg.backend != "cudss"


def test_auto_size_hint_drives_direct_vs_iterative():
    A = _make_spd_matrix(n=200, device="cpu")
    cfg_small = SolverConfig.auto(A, size_hint=10)
    cfg_big   = SolverConfig.auto(A, size_hint=10_000_000)
    assert cfg_small.method in {"cholesky", "lu"}
    if cfg_big.backend == "pyamg":
        assert cfg_big.method == "amg"


def test_auto_chains_with_axis_modifiers():
    """``auto(A)`` returns a SolverConfig with axis state recorded, so
    you can re-chain ``.cpu()``/``.direct()`` to override the auto-pick."""
    A = _make_spd_matrix(n=32, device="cpu")
    auto_cfg = SolverConfig.auto(A)
    forced = auto_cfg.cpu().direct()
    assert forced.backend == "scipy"
    assert forced.method == "cholesky"


def test_auto_chains_with_high_accuracy():
    A = _make_spd_matrix(n=16, device="cpu")
    cfg = SolverConfig.auto(A).high_accuracy()
    assert cfg.atol == 1e-12
    assert cfg.maxiter == 5000


def test_auto_chains_with_replace():
    A = _make_spd_matrix(n=16, device="cpu")
    cfg = SolverConfig.auto(A).replace(maxiter=42)
    assert cfg.maxiter == 42
