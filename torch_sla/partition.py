"""Partitioning algorithms for distributed sparse matrices.

This module hosts the data-layout side of the distributed stack:
the :class:`Partition` struct (one rank's view of which nodes it
owns + which neighbours it talks to) and the partitioner family that
builds them from a global graph -- METIS, simple 1-D slicing, RCB,
slicing-by-longest-axis, and a vectorised Hilbert space-filling
curve.

The Krylov-solver side lives in :mod:`torch_sla.distributed_solve`;
the :class:`DSparseTensor` itself lives in :mod:`torch_sla.distributed`.

External users typically just call :meth:`SparseTensor.partition_for_rank`
or :meth:`DSparseTensor.partition` and never import from this module
directly.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class Partition:
    """One rank's view of the global graph after partitioning.

    Attributes
    ----------
    partition_id : int
    local_nodes : torch.Tensor
        Global indices of every node visible to this rank (owned + halo).
    owned_nodes : torch.Tensor
        Subset actually owned -- the rank's slice of the row partition.
    halo_nodes : torch.Tensor
        Ghost nodes pulled in from neighbours each halo exchange.
    neighbor_partitions : List[int]
        IDs of partitions this rank exchanges with.
    send_indices : Dict[int, torch.Tensor]
        For each neighbour, which of OUR owned nodes we send them.
    recv_indices : Dict[int, torch.Tensor]
        For each neighbour, which positions in the local layout the
        received data lands in.
    global_to_local : torch.Tensor
        Global → local index map; -1 (or out-of-bounds) for nodes
        not in this partition.
    local_to_global : torch.Tensor
        Inverse map: local index → global node id.
    """
    partition_id: int
    local_nodes: torch.Tensor
    owned_nodes: torch.Tensor
    halo_nodes: torch.Tensor
    neighbor_partitions: List[int]
    send_indices: Dict[int, torch.Tensor]
    recv_indices: Dict[int, torch.Tensor]
    global_to_local: torch.Tensor
    local_to_global: torch.Tensor

    def to(self, device: "torch.device | str") -> "Partition":
        """Return a copy with every tensor field moved to ``device``.

        Index dtypes (``int64``) are preserved. Used by
        :meth:`DSparseTensor.to` so halo exchange tensors live alongside
        the local matrix data.
        """
        return Partition(
            partition_id=self.partition_id,
            local_nodes=self.local_nodes.to(device),
            owned_nodes=self.owned_nodes.to(device),
            halo_nodes=self.halo_nodes.to(device),
            neighbor_partitions=list(self.neighbor_partitions),
            send_indices={k: v.to(device)
                          for k, v in self.send_indices.items()},
            recv_indices={k: v.to(device)
                          for k, v in self.recv_indices.items()},
            global_to_local=self.global_to_local.to(device),
            local_to_global=self.local_to_global.to(device),
        )


def partition_graph_metis(
    row: torch.Tensor,
    col: torch.Tensor,
    num_nodes: int,
    num_parts: int,
) -> torch.Tensor:
    """Partition a graph using METIS (via ``pymetis``).

    Falls back to :func:`partition_simple` if ``pymetis`` isn't
    importable.

    Returns
    -------
    partition_ids : torch.Tensor
        Partition ID per node, shape ``(num_nodes,)``.
    """
    try:
        import pymetis
        adjacency = [[] for _ in range(num_nodes)]
        row_cpu = row.cpu().numpy()
        col_cpu = col.cpu().numpy()
        for r, c in zip(row_cpu, col_cpu):
            if r != c:
                adjacency[r].append(c)
        _, membership = pymetis.part_graph(num_parts, adjacency=adjacency)
        return torch.tensor(membership, dtype=torch.int64)
    except ImportError:
        warnings.warn("pymetis not available, using simple geometric partitioning")
        return partition_simple(num_nodes, num_parts)


def partition_simple(num_nodes: int, num_parts: int) -> torch.Tensor:
    """Contiguous 1-D partitioning: rank ``k`` owns the ``k``-th
    equal slice of ``[0, num_nodes)``. Fast, no quality guarantees."""
    nodes_per_part = (num_nodes + num_parts - 1) // num_parts
    idx = torch.arange(num_nodes, dtype=torch.int64)
    return torch.clamp(idx // nodes_per_part, max=num_parts - 1)


def partition_coordinates(
    coords: torch.Tensor,
    num_parts: int,
    method: str = "rcb",
) -> torch.Tensor:
    """Geometric partitioning by node coordinates.

    Three methods are supported:

    * ``"rcb"`` -- Recursive Coordinate Bisection (median-cut along
      longest axis). Standard CFD/FEM partitioner.
    * ``"slicing"`` -- Sort along longest axis, slice into equal
      chunks. Fast but worse quality than RCB.
    * ``"hilbert"`` -- Sort by Hilbert space-filling-curve index,
      slice into equal chunks. ~10-100x faster than METIS for
      PDE/mesh graphs where geometric locality correlates with
      sparse adjacency. 2-D and 3-D only.

    Parameters
    ----------
    coords : torch.Tensor
        Node coordinates, shape ``(num_nodes, dim)``.
    num_parts : int
        Number of partitions (power of 2 recommended for RCB).
    method : str
        ``"rcb"`` / ``"slicing"`` / ``"hilbert"``.
    """
    num_nodes = coords.size(0)
    partition_ids = torch.zeros(num_nodes, dtype=torch.int64)

    if method == "rcb":
        _rcb_partition(coords, partition_ids, torch.arange(num_nodes), 0, num_parts)
    elif method == "hilbert":
        sorted_idx = _hilbert_sort_indices(coords)
        nodes_per_part = (num_nodes + num_parts - 1) // num_parts
        for i, idx in enumerate(sorted_idx.tolist()):
            partition_ids[idx] = min(i // nodes_per_part, num_parts - 1)
    else:  # slicing
        ranges = coords.max(0).values - coords.min(0).values
        axis = ranges.argmax().item()
        sorted_idx = coords[:, axis].argsort()
        nodes_per_part = (num_nodes + num_parts - 1) // num_parts
        for i, idx in enumerate(sorted_idx):
            partition_ids[idx] = min(i // nodes_per_part, num_parts - 1)

    return partition_ids


def _hilbert_curve_indices(coords: torch.Tensor,
                            order: int = 16) -> torch.Tensor:
    """Map each row of ``coords`` to its position along a 2-D / 3-D
    Hilbert space-filling curve.

    Vectorised Skilling 2003 ``AxesToTranspose`` / ``TransposeToHilbert``
    bit-twiddling: the per-point loop is folded into a torch tensor
    axis so every bit op runs on all ``N`` points at once. The
    remaining loops are over ``d`` (=2/3) and ``order`` (=16), both
    tiny. ``order`` is the bit-depth per axis (default 16 → up to
    ~65k cells per dim before precision becomes the bottleneck).

    Returns the raw Hilbert index per row (shape ``(N,)``, dtype int64).
    """
    if coords.dim() != 2:
        raise ValueError(f"coords must be 2-D, got shape {tuple(coords.shape)}")
    n, d = coords.shape
    if d not in (2, 3):
        raise ValueError(
            f"Hilbert partitioner supports 2-D / 3-D coords, got d={d}")

    if coords.dtype.is_floating_point:
        c_min = coords.min(0).values
        c_max = coords.max(0).values
        extent = (c_max - c_min).clamp_min(1e-30)
        scale = (1 << order) - 1
        x = ((coords - c_min) / extent * scale).clamp_(0, scale).long()
    else:
        x = coords.long().clone()

    # Skilling 2003 "inverse undo" pass
    q = 1 << (order - 1)
    while q > 1:
        p = q - 1
        for i in range(d):
            mask = (x[:, i] & q) != 0
            t = (x[:, 0] ^ x[:, i]) & p
            x_zero_true  = x[:, 0] ^ p
            x_zero_false = x[:, 0] ^ t
            x_i_false    = x[:, i] ^ t
            x[:, 0] = torch.where(mask, x_zero_true, x_zero_false)
            x[:, i] = torch.where(mask, x[:, i],     x_i_false)
        q >>= 1

    # Gray-encode pass
    for i in range(1, d):
        x[:, i] ^= x[:, i - 1]

    # Untangle pass
    t = torch.zeros(n, dtype=torch.int64)
    q = 1 << (order - 1)
    while q > 1:
        mask = (x[:, d - 1] & q) != 0
        t ^= torch.where(mask, torch.full_like(t, q - 1),
                                torch.zeros_like(t))
        q >>= 1
    x ^= t.unsqueeze(1)

    # Bit-interleave to build the Hilbert index, MSB → LSB.
    h = torch.zeros(n, dtype=torch.int64)
    for bit in range(order - 1, -1, -1):
        for i in range(d):
            h = (h << 1) | ((x[:, i] >> bit) & 1)
    return h


def _hilbert_sort_indices(coords: torch.Tensor,
                          order: int = 16) -> torch.Tensor:
    """Return node indices sorted by Hilbert-curve position. Sorting by
    Hilbert index gives blocks with strong geometric locality -- exactly
    what sparse matvec wants for cheap halo exchange on PDE matrices."""
    return _hilbert_curve_indices(coords, order=order).argsort()


def _rcb_partition(
    coords: torch.Tensor,
    partition_ids: torch.Tensor,
    node_indices: torch.Tensor,
    part_offset: int,
    num_parts: int,
) -> None:
    """Recursive Coordinate Bisection helper. Mutates ``partition_ids``
    in place."""
    if num_parts == 1 or len(node_indices) == 0:
        partition_ids[node_indices] = part_offset
        return

    local_coords = coords[node_indices]
    ranges = local_coords.max(0).values - local_coords.min(0).values
    axis = ranges.argmax().item()

    axis_vals = local_coords[:, axis]
    median = axis_vals.median()

    left_mask = axis_vals <= median
    right_mask = ~left_mask

    left_nodes = node_indices[left_mask]
    right_nodes = node_indices[right_mask]

    left_parts = num_parts // 2
    right_parts = num_parts - left_parts

    _rcb_partition(coords, partition_ids, left_nodes, part_offset, left_parts)
    _rcb_partition(coords, partition_ids, right_nodes, part_offset + left_parts, right_parts)


def resolve_partition_ids(
    row: torch.Tensor,
    col: torch.Tensor,
    num_nodes: int,
    num_parts: int,
    method: str,
    coords: torch.Tensor = None,
) -> torch.Tensor:
    """Dispatch to the right partitioner by string name and return the
    per-node partition-id tensor of shape ``(num_nodes,)``.

    ``method`` may be ``"auto"`` (uses ``"rcb"`` if ``coords`` is given,
    else ``"simple"`` -- deterministic for distributed setups) /
    ``"simple"`` / ``"metis"`` / ``"rcb"`` / ``"slicing"`` /
    ``"hilbert"``. Geometric methods need ``coords``.
    """
    if method == "auto":
        method = "rcb" if coords is not None else "simple"

    if method == "simple":
        return partition_simple(num_nodes, num_parts)
    if method == "metis":
        return partition_graph_metis(row, col, num_nodes, num_parts)
    if method in ("rcb", "slicing", "hilbert"):
        if coords is None:
            raise ValueError(f"partition method '{method}' requires coords")
        return partition_coordinates(coords, num_parts, method=method)
    raise ValueError(
        f"Unknown partition method {method!r}; expected one of "
        "auto / simple / metis / rcb / slicing / hilbert.")


def build_partition(
    row: torch.Tensor,
    col: torch.Tensor,
    num_nodes: int,
    partition_ids: torch.Tensor,
    my_partition: int,
) -> Partition:
    """Build a full :class:`Partition` struct for one rank from a
    partition-id assignment over the global graph.

    Discovers owned / halo nodes, computes the global↔local index map,
    and lays out send / recv buffers for halo exchange. Used by
    :meth:`DSparseTensor.partition` and :meth:`DSparseTensor.from_global_distributed`
    to scatter a global graph across the mesh without ever
    materialising the dense layout.
    """
    owned_mask = partition_ids == my_partition
    owned_nodes = owned_mask.nonzero().squeeze(-1)
    halo_nodes, send_map = find_halo_nodes(row, col, partition_ids, my_partition)

    local_nodes = torch.cat([owned_nodes, halo_nodes])
    num_local = len(local_nodes)

    # Global → local index map (vectorised). ``-1`` marks "not on this rank".
    global_to_local = torch.full((num_nodes,), -1, dtype=torch.int64)
    global_to_local[local_nodes] = torch.arange(num_local, dtype=torch.int64)

    # Map each neighbour's send back to local recv positions.
    halo_offset = len(owned_nodes)
    halo_to_local = torch.full((num_nodes,), -1, dtype=torch.int64)
    halo_to_local[halo_nodes] = torch.arange(len(halo_nodes), dtype=torch.int64) + halo_offset

    recv_indices: Dict[int, torch.Tensor] = {}
    for neighbor_id in send_map.keys():
        neighbor_owned = (partition_ids == neighbor_id).nonzero().squeeze(-1)
        local_idx = halo_to_local[neighbor_owned]
        recv_indices[neighbor_id] = local_idx[local_idx >= 0]

    # send_map stores global node ids; halo_exchange wants local indices.
    send_indices_local: Dict[int, torch.Tensor] = {}
    for neighbor_id, global_nodes in send_map.items():
        send_indices_local[neighbor_id] = global_to_local[global_nodes]

    return Partition(
        partition_id=my_partition,
        local_nodes=local_nodes,
        owned_nodes=owned_nodes,
        halo_nodes=halo_nodes,
        neighbor_partitions=list(send_map.keys()),
        send_indices=send_indices_local,
        recv_indices=recv_indices,
        global_to_local=global_to_local,
        local_to_global=local_nodes.clone(),
    )


def find_halo_nodes(
    row: torch.Tensor,
    col: torch.Tensor,
    partition_ids: torch.Tensor,
    partition_id: int,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Find halo (ghost) nodes for a partition.

    A halo node is owned by some other partition but referenced by
    one of our owned rows -- we need its value to do local SpMV.

    Returns
    -------
    halo_nodes : torch.Tensor
        Global indices of nodes this rank pulls in each exchange.
    send_map : Dict[int, torch.Tensor]
        For each neighbour partition, the global indices of our
        owned nodes that they need from us.
    """
    owned_mask = partition_ids == partition_id
    row_cpu = row.cpu()
    col_cpu = col.cpu()

    row_owned = owned_mask[row_cpu]
    col_owned = owned_mask[col_cpu]

    # Case 1: row owned, col not owned -> col is halo
    mask1 = row_owned & ~col_owned
    halo_from_col = col_cpu[mask1]
    send_to_neighbor_col = row_cpu[mask1]
    neighbor_ids_col = partition_ids[halo_from_col]

    # Case 2: col owned, row not owned -> row is halo
    mask2 = col_owned & ~row_owned
    halo_from_row = row_cpu[mask2]
    send_to_neighbor_row = col_cpu[mask2]
    neighbor_ids_row = partition_ids[halo_from_row]

    all_halo = torch.cat([halo_from_col, halo_from_row])
    halo_nodes = torch.unique(all_halo, sorted=True)

    all_neighbors = torch.cat([neighbor_ids_col, neighbor_ids_row])
    all_send_nodes = torch.cat([send_to_neighbor_col, send_to_neighbor_row])

    send_map: Dict[int, torch.Tensor] = {}
    unique_neighbors = torch.unique(all_neighbors)
    for neighbor_id in unique_neighbors.tolist():
        mask = all_neighbors == neighbor_id
        send_map[neighbor_id] = torch.unique(all_send_nodes[mask], sorted=True)

    return halo_nodes, send_map
