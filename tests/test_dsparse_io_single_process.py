"""Single-process loading of a sharded ``DSparseTensor`` archive.

Covers the ``target_world_size=1`` path on ``load_dsparse`` /
``DSparseTensor.load`` -- the path that lets a single Python process
read a sharded save (written by ``save_sparse_sharded`` or by a
distributed ``save_dsparse``) without needing ``torchrun``.

Sibling to the multiprocess round-trip suite, which only exercises
``target_world_size == stored``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from torch_sla import DSparseTensor, SparseTensor
from torch_sla.datasets import Synthetic
from torch_sla.io import (
    load_dsparse,
    load_metadata,
    save_sparse_sharded,
)


def _build_test_sparse(n: int = 64):
    bench = Synthetic["poisson_2d_16"]
    return SparseTensor(bench.val, bench.row, bench.col, bench.shape)


@pytest.mark.parametrize("num_partitions", [2, 4, 8])
def test_single_process_load_round_trip(num_partitions):
    """Save with N partitions, load with ``target_world_size=1``,
    ``.full_tensor()`` should reproduce the original SparseTensor."""
    A = _build_test_sparse()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "shards"
        save_sparse_sharded(A, directory, num_partitions=num_partitions,
                            partition_method="simple")

        metadata = load_metadata(directory)
        assert metadata["num_partitions"] == num_partitions

        D = load_dsparse(directory, target_world_size=1)
        assert isinstance(D, DSparseTensor)

        full = D.full_tensor()
        # COO triples may be in a different order after the round trip;
        # compare dense forms to keep the assertion simple.
        got = torch.zeros(A.sparse_shape, dtype=A.values.dtype)
        got.index_put_((full.row_indices, full.col_indices),
                       full.values, accumulate=True)
        want = torch.zeros(A.sparse_shape, dtype=A.values.dtype)
        want.index_put_((A.row_indices, A.col_indices),
                        A.values, accumulate=True)
        assert torch.allclose(got, want, atol=1e-10), \
            "full_tensor after single-process load doesn't match the original"


def test_dsparse_tensor_load_classmethod_matches_free_function():
    """``DSparseTensor.load`` is a thin wrapper -- both paths should
    return the same result."""
    A = _build_test_sparse()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "shards"
        save_sparse_sharded(A, directory, num_partitions=4)
        D_func = load_dsparse(directory, target_world_size=1)
        D_meth = DSparseTensor.load(directory, target_world_size=1)
        # Same number of local entries -- the cheapest invariant we can
        # check without sharing internal pointers.
        assert D_func._local_tensor.nnz == D_meth._local_tensor.nnz
        assert D_func.shape == D_meth.shape


def test_target_world_size_mismatch_raises():
    """Stored N != target N != 1 must raise NotImplementedError with a
    workaround hint, not silently mis-read shards."""
    A = _build_test_sparse()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "shards"
        save_sparse_sharded(A, directory, num_partitions=4)
        with pytest.raises(NotImplementedError, match="repartition"):
            load_dsparse(directory, target_world_size=2)


def test_invalid_target_world_size_rejected():
    A = _build_test_sparse()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "shards"
        save_sparse_sharded(A, directory, num_partitions=2)
        with pytest.raises(ValueError, match="target_world_size"):
            load_dsparse(directory, target_world_size=0)
