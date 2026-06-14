"""Unit tests for ``SparseTensor.extract_partition(p)``.

Single-process (no torch.distributed required). Verifies that the
local-subdomain build produces a SparseTensor whose matvec, when
paired with halo-injected ``x``, reproduces the single-process global
matvec for the owned-row slice.
"""
from __future__ import annotations

import pytest
import torch

from torch_sla import SparseTensor
from torch_sla.distributed import (
    Partition,
    partition_simple,
)


def _poisson_2d(n: int, dtype=torch.float64):
    """4-on-diag, -1 on N/S/E/W neighbours -- the canonical SPD bench."""
    N = n * n
    idx = torch.arange(N)
    i = idx // n
    j = idx % n
    rows, cols, vals = [idx], [idx], [torch.full((N,), 4.0, dtype=dtype)]
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        ok = (ni >= 0) & (ni < n) & (nj >= 0) & (nj < n)
        rows.append(idx[ok])
        cols.append((ni[ok] * n + nj[ok]))
        vals.append(torch.full((int(ok.sum()),), -1.0, dtype=dtype))
    return (torch.cat(vals), torch.cat(rows), torch.cat(cols), (N, N))


def _build_partition(rank: int, world_size: int, N: int,
                     row_indices: torch.Tensor,
                     col_indices: torch.Tensor) -> Partition:
    """Build a Partition for a row-shard with ``partition_simple``
    so we can drive ``extract_partition`` end-to-end."""
    pid_for_node = partition_simple(N, world_size)
    owned = (pid_for_node == rank).nonzero().squeeze(1)
    # halo = global indices touched by owned rows that aren't owned
    owned_set = set(owned.tolist())
    halo_set: set = set()
    for r, c in zip(row_indices.tolist(), col_indices.tolist()):
        if r in owned_set and c not in owned_set:
            halo_set.add(c)
    halo = torch.tensor(sorted(halo_set), dtype=torch.int64)

    local_nodes = torch.cat([owned, halo])
    num_local = local_nodes.numel()
    global_to_local = torch.full((N,), -1, dtype=torch.int64)
    global_to_local[local_nodes] = torch.arange(num_local, dtype=torch.int64)

    return Partition(
        partition_id=rank,
        local_nodes=local_nodes,
        owned_nodes=owned,
        halo_nodes=halo,
        neighbor_partitions=[],
        send_indices={},
        recv_indices={},
        global_to_local=global_to_local,
        local_to_global=local_nodes,
    )


def test_extract_partition_shape_consistent():
    """The extracted local SparseTensor has shape (num_local, num_local)."""
    n = 8
    val, row, col, shape = _poisson_2d(n)
    A = SparseTensor(val, row, col, shape)
    ws = 2
    p = _build_partition(0, ws, n * n, row, col)
    A_local = A.extract_partition(p)
    assert A_local.shape == (p.local_nodes.numel(), p.local_nodes.numel())


def test_extract_partition_matvec_matches_global_owned_slice():
    """For every rank in a 2-rank split, the local matvec applied to
    a globally-correct ``x_local`` (owned + halo entries pulled from
    a known global x) must reproduce ``(A @ x)`` on the owned rows."""
    n = 8
    val, row, col, shape = _poisson_2d(n)
    A = SparseTensor(val, row, col, shape)
    A_dense = A.to_dense()
    ws = 2

    torch.manual_seed(0)
    x_global = torch.randn(n * n, dtype=torch.float64)
    y_global = A_dense @ x_global

    for rank in range(ws):
        p = _build_partition(rank, ws, n * n, row, col)
        A_local = A.extract_partition(p)
        # Sample x at local nodes (owned first, then halo) so the local
        # CSR sees the right values for both blocks.
        x_local = x_global[p.local_nodes]
        y_local = (A_local.to_dense() @ x_local)[:p.owned_nodes.numel()]
        y_expected = y_global[p.owned_nodes]
        torch.testing.assert_close(y_local, y_expected,
                                    rtol=1e-10, atol=1e-10)


def test_extract_partition_drops_off_rank_entries():
    """Entries whose row is NOT owned by this rank should be excluded
    from the local triple."""
    n = 4
    val, row, col, shape = _poisson_2d(n)
    A = SparseTensor(val, row, col, shape)
    ws = 2
    for rank in range(ws):
        p = _build_partition(rank, ws, n * n, row, col)
        A_local = A.extract_partition(p)
        # All local rows must be < num_owned (we excluded halo rows)
        assert (A_local.row_indices < p.owned_nodes.numel()).all()


def test_extract_partition_rejects_batched():
    val, row, col, shape = _poisson_2d(4)
    A = SparseTensor(val.unsqueeze(0).expand(2, -1).clone(),
                      row, col,
                      shape=(2, *shape))
    assert A.is_batched
    n = 4
    p = _build_partition(0, 2, n * n,
                          torch.zeros(0, dtype=torch.int64),
                          torch.zeros(0, dtype=torch.int64))
    with pytest.raises(ValueError, match="batched"):
        A.extract_partition(p)
