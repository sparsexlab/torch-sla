"""Graph / connected components for SparseTensor."""
from __future__ import annotations
from typing import Tuple, List
import torch

from .core import SparseTensor  # noqa: E402  # cross-module use
from .list import SparseTensorList


def _connected_components_labels(
    row: torch.Tensor,
    col: torch.Tensor,
    N: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Parallel, pure-torch connected components via label propagation.

    Shiloach-Vishkin style iterative ``scatter_reduce(amin)`` + pointer
    jumping. Fully vectorized: no Python per-edge loop, no ``.cpu()``
    round-trip, so it stays on ``device`` and runs on GPU. Converges in
    O(log N) rounds.

    Parameters
    ----------
    row, col : torch.Tensor
        Edge endpoint indices, shape [nnz], on ``device``.
    N : int
        Number of nodes.
    device : torch.device
        Device to keep all tensors on.

    Returns
    -------
    labels : torch.Tensor
        Contiguous component id per node in ``[0, num_components)``, long,
        on ``device``. Component ids follow ``torch.unique`` ordering of the
        propagated min-labels, i.e. the component containing the smallest
        node index gets id 0, matching the original union-find convention.
    num_components : int
    """
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=device), 0

    row = row.to(device=device, dtype=torch.long)
    col = col.to(device=device, dtype=torch.long)

    # Treat as undirected: consider both edge directions. Drop self-loops
    # (they never connect distinct nodes and just waste work).
    non_self = row != col
    src = torch.cat([row[non_self], col[non_self]])
    dst = torch.cat([col[non_self], row[non_self]])

    labels = torch.arange(N, device=device, dtype=torch.long)

    if src.numel() == 0:
        # No edges -> every node is its own component.
        unique, inverse = torch.unique(labels, return_inverse=True)
        return inverse.to(torch.long), unique.numel()

    # FastSV / Shiloach-Vishkin style connected components.
    #
    # Each round does (1) one min-label hooking step over every undirected
    # edge, then (2) FULL pointer jumping (path compression) so every node
    # points straight at the current root of its tree. Compressing all the
    # way to the root every round contracts components in O(log N) rounds
    # regardless of graph DIAMETER -- the old code jumped pointers only once
    # per round, so a long path / big grid needed O(diameter) rounds.
    #
    # Convergence is detected with a single scalar reduction (the sum of all
    # labels can only decrease and is constant once stable) instead of a full
    # ``torch.equal`` scan over N nodes every round.
    def _compress_to_roots(lab: torch.Tensor) -> torch.Tensor:
        # Repeated pointer jumping: lab[i] = lab[lab[i]] until every node
        # points at a root (lab[i] == lab[lab[i]] for all i). Each jump at
        # least halves the remaining tree height, so this is O(log N) jumps.
        while True:
            parent = lab[lab]
            if torch.equal(parent, lab):
                break
            lab = parent
        return lab

    labels = _compress_to_roots(labels)
    prev_checksum = labels.sum()
    while True:
        # Hooking: propagate the minimum root-label across every edge.
        new_labels = labels.clone()
        new_labels.scatter_reduce_(
            0, dst, labels[src], reduce="amin", include_self=True
        )
        # Full path compression so the next round's hooking sees roots.
        labels = _compress_to_roots(new_labels)
        checksum = labels.sum()
        # The total of all labels is monotonically non-increasing and stops
        # changing exactly when the partition is stable -> cheap O(1)-reduce
        # convergence test instead of an O(N) equality scan.
        if checksum == prev_checksum:
            break
        prev_checksum = checksum

    # Relabel to contiguous 0..num_components-1 (unique sorts ascending, so
    # the component with the smallest node index becomes id 0).
    unique, inverse = torch.unique(labels, return_inverse=True)
    return inverse.to(torch.long), unique.numel()


def connected_components(self) -> Tuple[torch.Tensor, int]:
    """
    Find connected components of the graph represented by this sparse matrix.

    Uses a parallel, pure-torch label-propagation / pointer-jumping
    (Shiloach-Vishkin style) algorithm: device-agnostic, GPU-ready, with no
    Python per-edge loop and no ``.cpu()`` round-trip. Treats the matrix as
    an undirected graph adjacency matrix.

    Returns
    -------
    labels : torch.Tensor
        Component label for each node, shape [N] (or [*batch, N] for batched
        input). Labels are in range [0, num_components), long, on
        ``self.device``.
    num_components : int
        Number of connected components. For batched input every batch item
        shares the same sparsity pattern, so the partition is identical
        across the batch and a single int is returned.

    Notes
    -----
    - Matrix is treated as undirected (edges in either direction count)
    - Self-loops are ignored for connectivity
    - Batched: all batch items share row/col indices (same structure), so
      the component partition is the same for every batch item. The returned
      ``labels`` is broadcast to shape [*batch, N].

    Examples
    --------
    >>> # Block diagonal matrix with 3 components
    >>> A = SparseTensor(val, row, col, (100, 100))
    >>> labels, num_comp = A.connected_components()
    >>> print(f"Found {num_comp} components")
    """
    M, N = self.sparse_shape
    if M != N:
        raise ValueError("connected_components requires square matrix")

    labels, num_components = _connected_components_labels(
        self.row_indices, self.col_indices, N, self.device
    )

    if self.is_batched:
        # Same structure across the batch -> same partition. Broadcast to
        # [*batch, N] so the output has a per-batch row.
        batch_shape = self.batch_shape
        labels = labels.reshape(*([1] * len(batch_shape)), N).expand(*batch_shape, N).contiguous()

    return labels, num_components

def has_isolated_components(self) -> bool:
    """
    Check if the matrix has multiple connected components.
    
    Returns
    -------
    bool
        True if matrix has more than one connected component.
        
    Examples
    --------
    >>> A = SparseTensor(val, row, col, (100, 100))
    >>> if A.has_isolated_components():
    ...     components = A.to_connected_components()
    """
    _, num_components = self.connected_components()
    return num_components > 1

def to_connected_components(self) -> "SparseTensorList":
    """
    Split the matrix into a list of connected component subgraphs.
    
    Each component becomes a separate SparseTensor with reindexed nodes.
    
    Returns
    -------
    SparseTensorList
        List of SparseTensors, one per connected component.
        
    Notes
    -----
    - Each component's nodes are reindexed from 0
    - Original node indices can be recovered from the mapping
    
    Examples
    --------
    >>> A = SparseTensor(val, row, col, (100, 100))
    >>> components = A.to_connected_components()
    >>> print(f"Split into {len(components)} components")
    >>> for i, comp in enumerate(components):
    ...     print(f"  Component {i}: {comp.shape}")
    """
    if self.is_batched:
        raise NotImplementedError("to_connected_components not supported for batched tensors")
    
    labels, num_components = self.connected_components()
    
    if num_components == 1:
        # Single component, return list with self
        return SparseTensorList([self])
    
    # Split into components
    components = []
    row = self.row_indices
    col = self.col_indices
    val = self.values
    
    for comp_id in range(num_components):
        # Find nodes in this component
        node_mask = (labels == comp_id)
        comp_nodes = torch.where(node_mask)[0]
        num_comp_nodes = len(comp_nodes)
        
        # Create mapping from old to new indices
        old_to_new = torch.full((self.sparse_shape[0],), -1, dtype=torch.long, device=self.device)
        old_to_new[comp_nodes] = torch.arange(num_comp_nodes, device=self.device)
        
        # Find edges within this component
        row_in_comp = node_mask[row]
        col_in_comp = node_mask[col]
        edge_mask = row_in_comp & col_in_comp
        
        # Extract and remap edges
        comp_row = old_to_new[row[edge_mask]]
        comp_col = old_to_new[col[edge_mask]]
        comp_val = val[edge_mask]
        
        comp_sparse = SparseTensor(
            comp_val, comp_row, comp_col,
            (num_comp_nodes, num_comp_nodes)
        )
        components.append(comp_sparse)
    
    return SparseTensorList(components)

