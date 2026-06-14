"""Graph / connected components for SparseTensor."""
from __future__ import annotations
from typing import Tuple, List
import torch

from .core import SparseTensor  # noqa: E402  # cross-module use


def connected_components(self) -> Tuple[torch.Tensor, int]:
    """
    Find connected components of the graph represented by this sparse matrix.
    
    Uses union-find algorithm for efficiency. Treats the matrix as an
    undirected graph adjacency matrix.
    
    Returns
    -------
    labels : torch.Tensor
        Component label for each node, shape [N]. Labels are in range [0, num_components).
    num_components : int
        Number of connected components.
        
    Notes
    -----
    - Only works for non-batched 2D matrices
    - Matrix is treated as undirected (edges in either direction count)
    - Self-loops are ignored for connectivity
    
    Examples
    --------
    >>> # Block diagonal matrix with 3 components
    >>> A = SparseTensor(val, row, col, (100, 100))
    >>> labels, num_comp = A.connected_components()
    >>> print(f"Found {num_comp} components")
    """
    if self.is_batched:
        raise NotImplementedError("connected_components not supported for batched tensors")
    
    M, N = self.sparse_shape
    if M != N:
        raise ValueError("connected_components requires square matrix")
    
    # Union-Find with path compression and union by rank
    parent = torch.arange(N, device=self.device, dtype=torch.long)
    rank = torch.zeros(N, device=self.device, dtype=torch.long)
    
    def find(x: int) -> int:
        """Find root with path compression."""
        root = x
        while parent[root].item() != root:
            root = parent[root].item()
        # Path compression
        while parent[x].item() != root:
            next_x = parent[x].item()
            parent[x] = root
            x = next_x
        return root
    
    def union(x: int, y: int):
        """Union by rank."""
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
    
    # Process all edges
    row = self.row_indices.cpu()
    col = self.col_indices.cpu()
    
    for i in range(len(row)):
        r, c = row[i].item(), col[i].item()
        if r != c:  # Skip self-loops
            union(r, c)
    
    # Find all roots and relabel
    labels = torch.zeros(N, dtype=torch.long, device=self.device)
    for i in range(N):
        labels[i] = find(i)
    
    # Relabel to consecutive integers starting from 0
    unique_labels = labels.unique()
    num_components = len(unique_labels)
    
    label_map = torch.zeros(N, dtype=torch.long, device=self.device)
    for new_label, old_label in enumerate(unique_labels):
        label_map[labels == old_label] = new_label
    
    return label_map, num_components

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

