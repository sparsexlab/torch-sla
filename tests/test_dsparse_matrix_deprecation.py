"""DSparseMatrix.from_global emits DeprecationWarning.

Single-process unit test -- just exercises the public entry point and
confirms the warning fires."""
from __future__ import annotations

import pytest
import torch

from torch_sla.distributed import DSparseMatrix


def _global_data(n: int = 8):
    diag = torch.arange(n, dtype=torch.int64)
    row = torch.cat([diag, diag[:-1], diag[1:]])
    col = torch.cat([diag, diag[1:], diag[:-1]])
    val = torch.cat([
        torch.full((n,), 4.0, dtype=torch.float64),
        torch.full((n - 1,), -1.0, dtype=torch.float64),
        torch.full((n - 1,), -1.0, dtype=torch.float64),
    ])
    return val, row, col, (n, n)


def test_dsparse_matrix_from_global_emits_deprecation_warning():
    val, row, col, shape = _global_data()
    with pytest.warns(DeprecationWarning, match="DSparseMatrix"):
        DSparseMatrix.from_global(
            val, row, col, shape,
            num_partitions=1, my_partition=0,
            verbose=False,
        )


def test_internal_impl_does_not_emit_warning():
    """``_from_global_impl`` is the silent internal path -- torch-sla's
    legacy code routes through it during the B-phase transition so
    user-facing test runs don't get spammed with DeprecationWarnings."""
    import warnings as _warnings

    val, row, col, shape = _global_data()
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", DeprecationWarning)
        mat = DSparseMatrix._from_global_impl(
            val, row, col, shape,
            num_partitions=1, my_partition=0,
            verbose=False,
        )
        assert mat is not None
