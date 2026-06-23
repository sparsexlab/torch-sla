#!/usr/bin/env python
"""
Correctness tests for the parallel, pure-torch ``connected_components``.

Compares the partition produced by ``SparseTensor.connected_components`` to
``scipy.sparse.csgraph.connected_components`` (ground truth) on several random
symmetric adjacency matrices, and checks the result is on the input device so
the implementation is GPU-ready.

Run with:
    pytest tests/test_graph_connected_components.py -v
"""

import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components as scipy_cc
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_sla import SparseTensor


def _partition_signature(labels):
    """Map a label vector to a canonical signature that is invariant to a
    permutation of the label ids, so two partitions can be compared directly.

    Each node is mapped to the index (in first-appearance order) of its
    component. Returns a tuple usable for equality comparison.
    """
    labels = list(int(x) for x in labels)
    remap = {}
    out = []
    for lab in labels:
        if lab not in remap:
            remap[lab] = len(remap)
        out.append(remap[lab])
    return tuple(out)


def _assert_matches_scipy(row, col, N, device="cpu"):
    """Build a SparseTensor and assert its CC partition matches scipy."""
    val = torch.ones(len(row), dtype=torch.float64)
    r = torch.as_tensor(row, dtype=torch.long)
    c = torch.as_tensor(col, dtype=torch.long)
    A = SparseTensor(val.to(device), r.to(device), c.to(device), (N, N))

    labels, n_comp = A.connected_components()

    # Result must stay on the input device (GPU-ready: no .cpu() round-trip).
    assert labels.device == A.device
    assert labels.dtype == torch.long
    assert labels.shape == (N,)
    assert int(labels.min()) >= 0
    assert int(labels.max()) == n_comp - 1  # contiguous 0..n_comp-1

    # scipy ground truth (undirected).
    data = np.ones(len(row))
    mat = sp.coo_matrix((data, (np.asarray(row), np.asarray(col))), shape=(N, N))
    n_comp_ref, labels_ref = scipy_cc(mat, directed=False)

    assert n_comp == n_comp_ref, f"num_components {n_comp} != scipy {n_comp_ref}"
    assert _partition_signature(labels.cpu().tolist()) == _partition_signature(labels_ref), (
        "partition does not match scipy"
    )
    return labels, n_comp


def _random_symmetric_graph(N, n_edges, seed):
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, N, (n_edges,), generator=g)
    c = torch.randint(0, N, (n_edges,), generator=g)
    # symmetrize
    row = torch.cat([r, c])
    col = torch.cat([c, r])
    return row.tolist(), col.tolist()


def _block_diagonal_graph(block_sizes, seed):
    """Build a block-diagonal graph: each block is internally connected (a
    path), blocks are mutually disconnected -> #components == #non-empty blocks.
    """
    g = torch.Generator().manual_seed(seed)
    rows, cols = [], []
    offset = 0
    for bs in block_sizes:
        nodes = list(range(offset, offset + bs))
        # connect as a path so the block is one component
        for a, b in zip(nodes[:-1], nodes[1:]):
            rows += [a, b]
            cols += [b, a]
        offset += bs
    N = offset
    return rows, cols, N


def test_single_component_path():
    rows = [0, 1, 1, 2, 2, 3]
    cols = [1, 0, 2, 1, 3, 2]
    labels, n = _assert_matches_scipy(rows, cols, 4)
    assert n == 1


def test_isolated_nodes():
    # Only an edge 0-1; nodes 2,3,4 isolated -> 4 components.
    rows = [0, 1]
    cols = [1, 0]
    labels, n = _assert_matches_scipy(rows, cols, 5)
    assert n == 4


def test_empty_graph():
    # No edges at all -> N singleton components.
    A = SparseTensor(
        torch.ones(0, dtype=torch.float64),
        torch.zeros(0, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
        (6, 6),
    )
    labels, n = A.connected_components()
    assert n == 6
    assert _partition_signature(labels.tolist()) == tuple(range(6))


def test_block_diagonal():
    rows, cols, N = _block_diagonal_graph([2, 3, 1, 4, 5], seed=1)
    labels, n = _assert_matches_scipy(rows, cols, N)
    # 5 blocks each internally connected -> 5 components.
    assert n == 5


def test_random_graphs_vs_scipy():
    for seed in range(8):
        N = int(torch.randint(5, 60, (1,), generator=torch.Generator().manual_seed(seed)))
        n_edges = max(1, N // 2)
        rows, cols = _random_symmetric_graph(N, n_edges, seed)
        _assert_matches_scipy(rows, cols, N)


def test_ordering_convention_matches_node0_first():
    # Component containing node 0 must get id 0 (matches old union-find +
    # unique-ascending convention). Block {0,1} then block {2,3,4}.
    val = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    row = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    col = torch.tensor([0, 1, 0, 1, 2, 3, 2, 4, 3, 4])
    A = SparseTensor(val, row, col, (5, 5))
    labels, n = A.connected_components()
    assert n == 2
    assert labels[0] == 0 and labels[1] == 0
    assert labels[2] == 1 and labels[3] == 1 and labels[4] == 1


def test_batched_shares_partition():
    # Batched: same row/col across batch -> identical partition per item.
    # values shape [B, nnz].
    row = torch.tensor([0, 1, 2, 3])
    col = torch.tensor([1, 0, 3, 2])
    B = 4
    nnz = row.numel()
    vals = torch.randn(B, nnz)
    A = SparseTensor(vals, row, col, (B, 4, 4))
    assert A.is_batched
    labels, n = A.connected_components()
    assert n == 2
    assert labels.shape == (B, 4)
    assert labels.device == A.device
    # Each batch row identical and a valid 2-partition {0,1},{2,3}.
    for b in range(B):
        lb = labels[b]
        assert lb[0] == lb[1]
        assert lb[2] == lb[3]
        assert lb[0] != lb[2]

    # Compare to scipy on the shared structure.
    mat = sp.coo_matrix(
        (np.ones(nnz), (row.numpy(), col.numpy())), shape=(4, 4)
    )
    n_ref, lab_ref = scipy_cc(mat, directed=False)
    assert n == n_ref
    assert _partition_signature(labels[0].tolist()) == _partition_signature(lab_ref)


def test_to_connected_components_still_works():
    val = torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    row = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    col = torch.tensor([0, 1, 0, 1, 2, 3, 2, 4, 3, 4])
    A = SparseTensor(val, row, col, (5, 5))
    stl = A.to_connected_components()
    assert len(stl) == 2
    assert stl[0].sparse_shape == (2, 2)
    assert stl[1].sparse_shape == (3, 3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok {name}")
    print("All passed.")
