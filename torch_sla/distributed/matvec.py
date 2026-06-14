"""Distributed sparse matrix-vector multiplication for
:class:`~torch_sla.distributed.DSparseTensor`.

Two entry points exposed on the class:

* ``D @ x``  -> :func:`matmul_spec`        -- DTensor-aware dispatcher.
* ``D._shard_matvec(x_owned)`` -> :func:`shard_matvec`  -- hot inner loop
  used by :mod:`distributed_solve` Krylov routines (no dtype/dispatch
  overhead; caller already in Shard(0) space).

Free functions, mirroring :mod:`distributed_solve` / :mod:`distributed_eigsh`.
"""
from __future__ import annotations

from typing import Any

import torch

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


def halo_exchange(D, x: torch.Tensor, partition) -> torch.Tensor:
    """Exchange ghost-node values with neighbours in-place on ``x``.

    Buffers + send/recv index tensors are cached on ``D`` so the hot
    path does nothing but ``index_select`` (gather), ``batch_isend_irecv``
    (async NCCL) / send-recv (gloo), and ``index_copy_`` (scatter).
    Per-rank caches are keyed by ``(neighbor_id, dtype)``.
    """
    if not _DIST_AVAILABLE or not dist.is_initialized():
        return x

    device = x.device
    dtype = x.dtype

    send_bufs, recv_bufs = {}, {}
    send_idxs, recv_idxs = {}, {}
    for nid in partition.neighbor_partitions:
        key = (nid, dtype)
        entry = D._halo_send_buffers.get(key)
        if entry is None:
            idx = partition.send_indices[nid].to(device=device, dtype=torch.int64)
            buf = torch.empty(int(idx.numel()), dtype=dtype, device=device)
            entry = (buf, idx)
            D._halo_send_buffers[key] = entry
        send_bufs[nid], send_idxs[nid] = entry

        entry = D._halo_recv_buffers.get(key)
        if entry is None:
            ridx = partition.recv_indices[nid].to(device=device, dtype=torch.int64)
            buf = torch.empty(int(ridx.numel()), dtype=dtype, device=device)
            entry = (buf, ridx)
            D._halo_recv_buffers[key] = entry
        recv_bufs[nid], recv_idxs[nid] = entry

    for nid in partition.neighbor_partitions:
        torch.index_select(x, 0, send_idxs[nid], out=send_bufs[nid])

    # batch_isend_irecv on NCCL avoids the rank-id ordering dance and
    # lets the CPU return immediately. gloo lacks the batched API, so
    # fall back to ordered synchronous send/recv (rank-id ordering
    # prevents the legacy two-rank deadlock).
    backend = dist.get_backend()
    my_id = int(partition.partition_id)
    ordered = sorted(partition.neighbor_partitions, key=int)
    if backend == "nccl":
        ops = []
        for nid in ordered:
            ops.append(dist.P2POp(dist.isend, send_bufs[nid], int(nid)))
            ops.append(dist.P2POp(dist.irecv, recv_bufs[nid], int(nid)))
        for req in dist.batch_isend_irecv(ops):
            req.wait()
    else:
        for nid in ordered:
            nid_i = int(nid)
            if my_id < nid_i:
                dist.send(send_bufs[nid], dst=nid_i)
                dist.recv(recv_bufs[nid], src=nid_i)
            else:
                dist.recv(recv_bufs[nid], src=nid_i)
                dist.send(send_bufs[nid], dst=nid_i)

    for nid in partition.neighbor_partitions:
        x.index_copy_(0, recv_idxs[nid], recv_bufs[nid])
    return x


def matmul_row_shard(D, x: Any) -> Any:
    """Row-sharded ``D @ x``. Pad owned -> local, halo exchange, local
    SpMV, slice back to owned-row range. Returns ``DTensor[Shard(0)]``
    if ``x`` is a DTensor, else a plain owned-row tensor."""
    from .core import _is_dtensor, DTensor, Shard

    partition = D._spec.placement.partition
    if partition is None:
        raise RuntimeError("row-shard matvec requires spec.placement.partition")
    num_owned = int(partition.owned_nodes.numel())
    num_local = int(partition.local_to_global.numel())

    x_local = x.to_local() if _is_dtensor(x) else x

    if x_local.shape[0] == num_owned:
        x_padded = torch.zeros(num_local, dtype=x_local.dtype, device=x_local.device)
        x_padded[:num_owned] = x_local
    elif x_local.shape[0] == num_local:
        x_padded = x_local
    else:
        raise ValueError(
            f"x has shape[0]={x_local.shape[0]}, expected num_owned "
            f"({num_owned}) or num_local ({num_local}).")

    halo_exchange(D, x_padded, partition)
    y_owned = (D._local_tensor @ x_padded)[:num_owned]

    if _is_dtensor(x):
        return DTensor.from_local(y_owned, D._spec.mesh, [Shard(0)])
    return y_owned


def matmul_col_shard(D, x: Any) -> Any:
    """Col-partitioned matvec (``SparseShard(axis=1)``). Not implemented."""
    raise NotImplementedError(
        "SparseShard(axis=1) col-partitioned matvec is not yet implemented; "
        "use SparseShard(axis=0)."
    )


def matmul_spec(D, x: Any) -> Any:
    """``D @ x`` dispatcher. Routes on ``spec.placement.axis``."""
    from .core import SparseShard

    if D._local_tensor is None:
        raise RuntimeError(
            "DSparseTensor matvec requires a SparseTensor backing. "
            "Build via .partition(...) / .from_sparse_local(...).")
    placement = D._spec.placement
    if not isinstance(placement, SparseShard):
        raise RuntimeError(
            f"matmul_spec expects SparseShard placement; got {type(placement).__name__}")

    if placement.axis == 0:
        return matmul_row_shard(D, x)
    if placement.axis == 1:
        return matmul_col_shard(D, x)
    raise NotImplementedError(
        f"SparseShard(axis={placement.axis}) matvec dispatch not implemented")


def shard_matvec(D, x_owned: torch.Tensor) -> torch.Tensor:
    """Hot-path Shard(0) matvec used by the Krylov solvers. Pads
    owned -> local, halo-exchanges, runs a cached CSR ``torch.mv``,
    slices back to owned-row range.

    Caches: ``D._local_csr_cache`` (CSR with int32 indices when safe),
    ``D._x_padded_cache`` ((num_local,) scratch), ``D._y_full_cache``
    ((num_local,) output).
    """
    partition = D._spec.placement.partition
    num_owned = int(partition.owned_nodes.numel())
    num_local = int(partition.local_to_global.numel())
    dtype, device = x_owned.dtype, x_owned.device

    if x_owned.shape[0] == num_owned:
        xp = getattr(D, "_x_padded_cache", None)
        if (xp is None or xp.shape[0] != num_local
                or xp.dtype != dtype or xp.device != device):
            xp = torch.zeros(num_local, dtype=dtype, device=device)
            D._x_padded_cache = xp
        xp[:num_owned].copy_(x_owned)
        x_padded = xp
    elif x_owned.shape[0] == num_local:
        x_padded = x_owned
    else:
        raise ValueError(
            f"x shape[0]={x_owned.shape[0]}, expected num_owned={num_owned} or num_local={num_local}")
    halo_exchange(D, x_padded, partition)

    # Cache local CSR. int32 indices when num_local < 2^31 to halve
    # col_indices storage + improve cuSPARSE L1 hit rate.
    csr = getattr(D, "_local_csr_cache", None)
    if csr is None:
        st = D._local_tensor
        indices = torch.stack([st.row_indices.to(torch.int64),
                               st.col_indices.to(torch.int64)])
        coo = torch.sparse_coo_tensor(indices, st.values, tuple(st.shape)).coalesce()
        csr64 = coo.to_sparse_csr()
        idx_dtype = torch.int32 if num_local < 2_147_483_647 else torch.int64
        if idx_dtype is torch.int32:
            csr = torch.sparse_csr_tensor(
                csr64.crow_indices().to(idx_dtype),
                csr64.col_indices().to(idx_dtype),
                csr64.values(), csr64.size(),
            )
        else:
            csr = csr64
        D._local_csr_cache = csr

    # Pre-alloc output (required for any future CUDA Graphs capture).
    yf = getattr(D, "_y_full_cache", None)
    if (yf is None or yf.shape[0] != num_local
            or yf.dtype != dtype or yf.device != device):
        yf = torch.empty(num_local, dtype=dtype, device=device)
        D._y_full_cache = yf
    torch.mv(csr, x_padded, out=yf)
    return yf[:num_owned]


def matmul_batch_shard(D, x):
    """Embarrassingly-parallel matvec for ``BatchShard(axis=k)``.

    Every rank computes ``A_local @ x_local`` on its own batch slice;
    no halo exchange, no cross-rank reduction. ``x`` may be a
    same-spec DSparseTensor (already sharded along the same axis) or
    a plain ``torch.Tensor`` whose batch axis is full-extent (in which
    case we slice it).
    """
    from .core import DSparseTensor, BatchShard
    placement = D._spec.placement
    assert isinstance(placement, BatchShard)
    if isinstance(x, DSparseTensor):
        x_local = x._local_tensor.values
    elif hasattr(x, 'to_local'):  # DTensor input
        x_local = x.to_local()
    else:
        x_local = x.narrow(placement.axis, placement.start,
                           placement.end - placement.start)
    return D._local_tensor @ x_local
