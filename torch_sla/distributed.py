"""
Distributed Sparse Matrix for large-scale CFD/FEM computations.

Provides domain decomposition with halo exchange, following the standard
approach used in Ansys, OpenFOAM, and other industrial CFD/FEM solvers.

Key Features:
- Graph-based partitioning (METIS or simple geometric methods)
- Halo/ghost node exchange for parallel computations
- Support for both CPU and CUDA devices
- Same API as SparseTensor for easy migration

Example
-------
>>> from torch_sla import DSparseMatrix
>>> 
>>> # Create from global matrix
>>> A_global = SparseTensor(val, row, col, shape)
>>> A_dist = DSparseMatrix.from_global(A_global, num_partitions=4)
>>> 
>>> # Distributed solve
>>> x_dist = A_dist.solve(b_dist)
>>> 
>>> # Halo exchange for iterative methods
>>> A_dist.halo_exchange(local_x)
"""

import os
import torch
from typing import Any, Tuple, List, Dict, Optional, Union, Literal
from dataclasses import dataclass
import warnings

from .backends import (
    is_scipy_available,
    is_eigen_available,
    is_cupy_available,
    is_cudss_available,
    select_backend,
    select_method,
    BackendType,
    MethodType,
)

try:
    import torch.distributed as dist
    DIST_AVAILABLE = True
except ImportError:
    DIST_AVAILABLE = False

# DTensor support (PyTorch 2.0+)
try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Shard, Replicate
    DTENSOR_AVAILABLE = True
except ImportError:
    try:
        # Older import path (PyTorch 2.0-2.1)
        from torch.distributed._tensor import DTensor
        from torch.distributed._tensor.placement_types import Shard, Replicate
        DTENSOR_AVAILABLE = True
    except ImportError:
        DTENSOR_AVAILABLE = False
        DTensor = None
        Shard = None
        Replicate = None


def _is_dtensor(x) -> bool:
    """Check if x is a DTensor instance."""
    if not DTENSOR_AVAILABLE or DTensor is None:
        return False
    return isinstance(x, DTensor)


@dataclass
class Partition:
    """Represents a single partition/subdomain"""
    partition_id: int
    local_nodes: torch.Tensor      # Global indices of local nodes
    owned_nodes: torch.Tensor      # Nodes owned by this partition (not halo)
    halo_nodes: torch.Tensor       # Ghost/halo nodes from neighbors
    neighbor_partitions: List[int] # Neighboring partition IDs
    send_indices: Dict[int, torch.Tensor]  # Nodes to send to each neighbor
    recv_indices: Dict[int, torch.Tensor]  # Where to place received data
    global_to_local: torch.Tensor  # Mapping from global to local indices
    local_to_global: torch.Tensor  # Mapping from local to global indices
    

def partition_graph_metis(
    row: torch.Tensor,
    col: torch.Tensor,
    num_nodes: int,
    num_parts: int
) -> torch.Tensor:
    """
    Partition graph using METIS (if available) or fallback to simple method.
    
    Returns
    -------
    partition_ids : torch.Tensor
        Partition ID for each node [num_nodes]
    """
    try:
        import pymetis
        # Build adjacency list
        adjacency = [[] for _ in range(num_nodes)]
        row_cpu = row.cpu().numpy()
        col_cpu = col.cpu().numpy()
        
        for r, c in zip(row_cpu, col_cpu):
            if r != c:  # Skip diagonal
                adjacency[r].append(c)
        
        # Run METIS
        _, membership = pymetis.part_graph(num_parts, adjacency=adjacency)
        return torch.tensor(membership, dtype=torch.int64)
    
    except ImportError:
        warnings.warn("pymetis not available, using simple geometric partitioning")
        return partition_simple(num_nodes, num_parts)


def partition_simple(num_nodes: int, num_parts: int) -> torch.Tensor:
    """Simple 1D partitioning (fallback when METIS not available) - vectorized."""
    nodes_per_part = (num_nodes + num_parts - 1) // num_parts
    idx = torch.arange(num_nodes, dtype=torch.int64)
    partition_ids = torch.clamp(idx // nodes_per_part, max=num_parts - 1)
    return partition_ids


def partition_coordinates(
    coords: torch.Tensor,
    num_parts: int,
    method: str = 'rcb'
) -> torch.Tensor:
    """
    Partition based on node coordinates.

    Three methods supported:

    * ``"rcb"``      Recursive Coordinate Bisection (median-cut along
                     longest axis). Standard CFD/FEM partitioner.
    * ``"slicing"``  Sort along longest axis, slice into equal chunks.
                     Fast but worse quality than RCB.
    * ``"hilbert"``  Sort by Hilbert space-filling-curve index, slice
                     into equal chunks. **~10-100x faster than METIS**
                     for PDE/mesh-like graphs where geometric locality
                     correlates with sparse adjacency. Supports 2-D and
                     3-D coordinates (Skilling 2003 algorithm).

    Parameters
    ----------
    coords : torch.Tensor
        Node coordinates ``[num_nodes, dim]``.
    num_parts : int
        Number of partitions (should be power of 2 for RCB).
    method : str
        ``"rcb"`` / ``"slicing"`` / ``"hilbert"``.
    """
    num_nodes = coords.size(0)
    partition_ids = torch.zeros(num_nodes, dtype=torch.int64)

    if method == 'rcb':
        _rcb_partition(coords, partition_ids, torch.arange(num_nodes), 0, num_parts)
    elif method == 'hilbert':
        sorted_idx = _hilbert_sort_indices(coords)
        nodes_per_part = (num_nodes + num_parts - 1) // num_parts
        for i, idx in enumerate(sorted_idx.tolist()):
            partition_ids[idx] = min(i // nodes_per_part, num_parts - 1)
    else:  # slicing
        # Find longest axis
        ranges = coords.max(0).values - coords.min(0).values
        axis = ranges.argmax().item()

        # Sort by that axis
        sorted_idx = coords[:, axis].argsort()
        nodes_per_part = (num_nodes + num_parts - 1) // num_parts

        for i, idx in enumerate(sorted_idx):
            partition_ids[idx] = min(i // nodes_per_part, num_parts - 1)

    return partition_ids


def _hilbert_sort_indices(coords: torch.Tensor,
                          order: int = 16) -> torch.Tensor:
    """Sort node indices by their Hilbert space-filling-curve position.

    Each row of ``coords`` is mapped to its position along a 2-D / 3-D
    Hilbert curve; the returned tensor lists node indices in curve
    order. Sorting by Hilbert index gives blocks that have **strong
    geometric locality** -- exactly what sparse matvec wants for cheap
    halo exchange on PDE-discretised matrices.

    Implementation uses Skilling 2003's ``AxestoTranspose`` /
    ``TransposeToHilbert`` bit-twiddling algorithm; pure-Python, no
    binding required. ``order`` is the number of bits per axis
    (default 16, supports up to ~65k cells per dim before precision
    becomes the bottleneck).
    """
    if coords.dim() != 2:
        raise ValueError(f"coords must be 2-D, got shape {tuple(coords.shape)}")
    n, d = coords.shape
    if d not in (2, 3):
        raise ValueError(
            f"Hilbert partitioner supports 2-D / 3-D coords, got d={d}")

    # Quantise coords to ``order``-bit non-negative integers per axis.
    c_min = coords.min(0).values
    c_max = coords.max(0).values
    extent = (c_max - c_min).clamp_min(1e-30)
    scale = (1 << order) - 1
    qcoords = ((coords - c_min) / extent * scale).clamp_(0, scale).long()

    # Compute Hilbert index per node.
    hilbert_indices = torch.empty(n, dtype=torch.int64)
    qarr = qcoords.tolist()
    for i in range(n):
        hilbert_indices[i] = _hilbert_index_from_axes(qarr[i], order)
    return hilbert_indices.argsort()


def _hilbert_index_from_axes(axes: List[int], order: int) -> int:
    """Skilling 2003: convert per-axis bit-array to a single
    Hilbert-curve index. ``axes`` is one quantised coordinate per
    dimension; ``order`` is the bit-depth per axis."""
    n = len(axes)
    x = list(axes)
    # Skilling's algorithm: transformations to make the bits Gray-code-
    # like before bit-interleaving.
    m = 1 << (order - 1)
    # Inverse undo
    q = m
    while q > 1:
        p = q - 1
        for i in range(n):
            if x[i] & q:
                x[0] ^= p
            else:
                t = (x[0] ^ x[i]) & p
                x[0] ^= t
                x[i] ^= t
        q >>= 1
    # Gray encode
    for i in range(1, n):
        x[i] ^= x[i - 1]
    t = 0
    q = m
    while q > 1:
        if x[n - 1] & q:
            t ^= q - 1
        q >>= 1
    for i in range(n):
        x[i] ^= t

    # Interleave the bits: build the Hilbert index by taking one bit
    # from each axis at each level, MSB-to-LSB.
    h = 0
    for bit in range(order - 1, -1, -1):
        for i in range(n):
            h = (h << 1) | ((x[i] >> bit) & 1)
    return h


def _rcb_partition(
    coords: torch.Tensor,
    partition_ids: torch.Tensor,
    node_indices: torch.Tensor,
    part_offset: int,
    num_parts: int
):
    """Recursive Coordinate Bisection helper"""
    if num_parts == 1 or len(node_indices) == 0:
        partition_ids[node_indices] = part_offset
        return
    
    # Find longest axis
    local_coords = coords[node_indices]
    ranges = local_coords.max(0).values - local_coords.min(0).values
    axis = ranges.argmax().item()
    
    # Find median
    axis_vals = local_coords[:, axis]
    median = axis_vals.median()
    
    # Split
    left_mask = axis_vals <= median
    right_mask = ~left_mask
    
    left_nodes = node_indices[left_mask]
    right_nodes = node_indices[right_mask]
    
    # Handle uneven splits
    left_parts = num_parts // 2
    right_parts = num_parts - left_parts
    
    _rcb_partition(coords, partition_ids, left_nodes, part_offset, left_parts)
    _rcb_partition(coords, partition_ids, right_nodes, part_offset + left_parts, right_parts)


def find_halo_nodes(
    row: torch.Tensor,
    col: torch.Tensor,
    partition_ids: torch.Tensor,
    partition_id: int
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """
    Find halo/ghost nodes for a partition (vectorized version).
    
    Halo nodes are nodes owned by other partitions but connected to this partition's nodes.
    
    Returns
    -------
    halo_nodes : torch.Tensor
        Global indices of halo nodes
    send_map : Dict[int, torch.Tensor]
        For each neighbor, which of our owned nodes to send
    """
    # Vectorized ownership check
    owned_mask = partition_ids == partition_id
    
    row_cpu = row.cpu()
    col_cpu = col.cpu()
    
    row_owned = owned_mask[row_cpu]
    col_owned = owned_mask[col_cpu]
    
    # Case 1: row owned, col not owned -> col is halo
    mask1 = row_owned & ~col_owned
    halo_from_col = col_cpu[mask1]
    send_to_neighbor_col = row_cpu[mask1]  # owned nodes to send
    neighbor_ids_col = partition_ids[halo_from_col]
    
    # Case 2: col owned, row not owned -> row is halo
    mask2 = col_owned & ~row_owned
    halo_from_row = row_cpu[mask2]
    send_to_neighbor_row = col_cpu[mask2]  # owned nodes to send
    neighbor_ids_row = partition_ids[halo_from_row]
    
    # Combine halo nodes
    all_halo = torch.cat([halo_from_col, halo_from_row])
    halo_nodes = torch.unique(all_halo, sorted=True)
    
    # Build send_map: for each neighbor, which owned nodes to send
    all_neighbors = torch.cat([neighbor_ids_col, neighbor_ids_row])
    all_send_nodes = torch.cat([send_to_neighbor_col, send_to_neighbor_row])
    
    send_map = {}
    unique_neighbors = torch.unique(all_neighbors)
    for neighbor_id in unique_neighbors.tolist():
        mask = all_neighbors == neighbor_id
        nodes_to_send = torch.unique(all_send_nodes[mask], sorted=True)
        send_map[neighbor_id] = nodes_to_send
    
    return halo_nodes, send_map


class DSparseMatrix:
    """
    Distributed Sparse Matrix with halo exchange support.
    
    Designed for large-scale CFD/FEM computations following industrial
    practices from Ansys, OpenFOAM, etc.
    
    The matrix is partitioned across multiple processes/GPUs, with automatic
    halo (ghost) node management for parallel iterative solvers.
    
    Supports both CPU and CUDA devices.
    
    Attributes
    ----------
    partition : Partition
        Local partition information
    local_values : torch.Tensor
        Non-zero values for local portion of matrix
    local_row : torch.Tensor
        Local row indices
    local_col : torch.Tensor
        Local column indices
    local_shape : Tuple[int, int]
        Shape of local matrix (including halo)
    global_shape : Tuple[int, int]
        Shape of global matrix
    device : torch.device
        Device where the matrix data resides (cpu or cuda)
    
    Example
    -------
    >>> # Create distributed matrix on CPU
    >>> A = DSparseMatrix.from_global(val, row, col, shape, num_parts=4, my_part=0, device='cpu')
    >>> 
    >>> # Create distributed matrix on CUDA
    >>> A_cuda = DSparseMatrix.from_global(val, row, col, shape, num_parts=4, my_part=0, device='cuda')
    >>> 
    >>> # Distributed matrix-vector product with halo exchange
    >>> y = A.matvec(x)  # Automatically handles halo exchange
    >>> 
    >>> # Explicit halo exchange
    >>> A.halo_exchange(x)  # Update halo values in x
    """
    
    def __init__(
        self,
        partition: Partition,
        local_values: torch.Tensor,
        local_row: torch.Tensor,
        local_col: torch.Tensor,
        local_shape: Tuple[int, int],
        global_shape: Tuple[int, int],
        num_partitions: int,
        device: Union[str, torch.device] = 'cpu',
        verbose: bool = True
    ):
        # Convert device to torch.device
        if isinstance(device, str):
            device = torch.device(device)
        
        self.partition = partition
        self.local_values = local_values.to(device)
        self.local_row = local_row.to(device)
        self.local_col = local_col.to(device)
        self.local_shape = local_shape
        self.global_shape = global_shape
        self.num_partitions = num_partitions
        self.device = device
        self._verbose = verbose
        
        # Move partition tensors to device
        self._partition_to_device()
        
        # For display
        if verbose:
            self._print_partition_info()
    
    def _partition_to_device(self):
        """Move partition tensors to the target device"""
        # Note: We keep some partition info on CPU for indexing
        # Only move what's needed for computation
        pass
    
    def _print_partition_info(self):
        """Print partition info for user awareness"""
        owned = len(self.partition.owned_nodes)
        halo = len(self.partition.halo_nodes)
        total = self.local_shape[0]
        neighbors = len(self.partition.neighbor_partitions)
        
        print(f"[Partition {self.partition.partition_id}/{self.num_partitions}] "
              f"Nodes: {owned} owned + {halo} halo = {total} local | "
              f"Neighbors: {neighbors} | "
              f"Global: {self.global_shape[0]}x{self.global_shape[1]} | "
              f"Device: {self.device}")
    
    def to(self, device: Union[str, torch.device]) -> "DSparseMatrix":
        """
        Move the distributed matrix to a different device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device ('cpu', 'cuda', 'cuda:0', etc.)
            
        Returns
        -------
        DSparseMatrix
            New distributed matrix on the target device
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        return DSparseMatrix(
            partition=self.partition,
            local_values=self.local_values.to(device),
            local_row=self.local_row.to(device),
            local_col=self.local_col.to(device),
            local_shape=self.local_shape,
            global_shape=self.global_shape,
            num_partitions=self.num_partitions,
            device=device,
            verbose=False  # Don't print again when moving
        )
    
    def cuda(self, device: Optional[int] = None) -> "DSparseMatrix":
        """Move to CUDA device"""
        if device is not None:
            return self.to(f'cuda:{device}')
        return self.to('cuda')
    
    def cpu(self) -> "DSparseMatrix":
        """Move to CPU"""
        return self.to('cpu')
    
    @property
    def is_cuda(self) -> bool:
        """Check if matrix is on CUDA"""
        return self.device.type == 'cuda'
    
    @classmethod
    def from_global(
        cls,
        values: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        shape: Tuple[int, int],
        num_partitions: int,
        my_partition: int,
        partition_ids: Optional[torch.Tensor] = None,
        coords: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = 'cpu',
        verbose: bool = True
    ) -> "DSparseMatrix":
        """
        Create distributed matrix from global COO data.
        
        Parameters
        ----------
        values, row, col : torch.Tensor
            Global COO sparse matrix data
        shape : Tuple[int, int]
            Global matrix shape
        num_partitions : int
            Number of partitions
        my_partition : int
            This process's partition ID (0 to num_partitions-1)
        partition_ids : torch.Tensor, optional
            Pre-computed partition assignments. If None, computed automatically.
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning [num_nodes, dim]
        device : str or torch.device
            Device for local data ('cpu', 'cuda', 'cuda:0', etc.)
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseMatrix
            Local portion of the distributed matrix
        """
        num_nodes = shape[0]
        
        # Compute partitioning if not provided
        if partition_ids is None:
            if coords is not None:
                partition_ids = partition_coordinates(coords, num_partitions)
            else:
                partition_ids = partition_graph_metis(row, col, num_nodes, num_partitions)
        
        # Find owned and halo nodes
        owned_mask = partition_ids == my_partition
        owned_nodes = owned_mask.nonzero().squeeze(-1)
        halo_nodes, send_map = find_halo_nodes(row, col, partition_ids, my_partition)
        
        # All local nodes (owned + halo)
        local_nodes = torch.cat([owned_nodes, halo_nodes])
        num_local = len(local_nodes)
        
        # Build global-to-local mapping (vectorized)
        global_to_local = torch.full((num_nodes,), -1, dtype=torch.int64)
        global_to_local[local_nodes] = torch.arange(num_local, dtype=torch.int64)
        
        # Extract local matrix entries (vectorized)
        row_cpu = row.cpu()
        col_cpu = col.cpu()
        val_cpu = values.cpu()
        
        # Map global indices to local
        local_row_mapped = global_to_local[row_cpu]
        local_col_mapped = global_to_local[col_cpu]
        
        # Filter to entries where both row and col are local
        valid_mask = (local_row_mapped >= 0) & (local_col_mapped >= 0)
        local_row = local_row_mapped[valid_mask]
        local_col = local_col_mapped[valid_mask]
        local_values = val_cpu[valid_mask]
        
        # Build recv_indices (vectorized)
        recv_indices = {}
        halo_offset = len(owned_nodes)
        
        # Create halo node to local index mapping
        halo_to_local = torch.full((num_nodes,), -1, dtype=torch.int64)
        halo_to_local[halo_nodes] = torch.arange(len(halo_nodes), dtype=torch.int64) + halo_offset
        
        for neighbor_id in send_map.keys():
            neighbor_owned = (partition_ids == neighbor_id).nonzero().squeeze(-1)
            # Find which of neighbor's owned nodes are in our halo
            local_idx = halo_to_local[neighbor_owned]
            recv_indices[neighbor_id] = local_idx[local_idx >= 0]
        
        # Convert send_map from global node IDs to local indices
        # send_map currently contains global node IDs, but halo_exchange needs local indices
        send_indices_local = {}
        for neighbor_id, global_nodes in send_map.items():
            local_idx = global_to_local[global_nodes]
            send_indices_local[neighbor_id] = local_idx
        
        partition = Partition(
            partition_id=my_partition,
            local_nodes=local_nodes,
            owned_nodes=owned_nodes,
            halo_nodes=halo_nodes,
            neighbor_partitions=list(send_map.keys()),
            send_indices=send_indices_local,  # Use local indices instead of global
            recv_indices=recv_indices,
            global_to_local=global_to_local,
            local_to_global=local_nodes.clone()
        )
        
        return cls(
            partition=partition,
            local_values=local_values,
            local_row=local_row,
            local_col=local_col,
            local_shape=(num_local, num_local),
            global_shape=shape,
            num_partitions=num_partitions,
            device=device,
            verbose=verbose
        )
    
    @property
    def num_owned(self) -> int:
        """Number of owned (non-halo) nodes"""
        return len(self.partition.owned_nodes)
    
    @property
    def num_halo(self) -> int:
        """Number of halo/ghost nodes"""
        return len(self.partition.halo_nodes)
    
    @property
    def num_local(self) -> int:
        """Total local nodes (owned + halo)"""
        return self.local_shape[0]
    
    @property
    def nnz(self) -> int:
        """Number of non-zeros in local matrix"""
        return len(self.local_values)
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of matrix values"""
        return self.local_values.dtype
    
    def halo_exchange(
        self,
        x: torch.Tensor,
        async_op: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Exchange halo/ghost values with neighbors.
        
        This is the core operation for parallel iterative methods.
        Updates the halo portion of x with values from neighboring partitions.
        
        Parameters
        ----------
        x : torch.Tensor
            Local vector [num_local] with owned values filled in.
            Halo values will be updated.
        async_op : bool
            If True, return immediately and return a future.
            
        Returns
        -------
        x : torch.Tensor
            Vector with updated halo values (same tensor, modified in-place)
            
        Example
        -------
        >>> # During iterative solve
        >>> for iteration in range(max_iter):
        >>>     # Compute local update
        >>>     x_new = local_gauss_seidel_step(A_local, x, b)
        >>>     
        >>>     # Exchange boundary values
        >>>     A.halo_exchange(x_new)
        >>>     
        >>>     # Check convergence using owned nodes only
        >>>     residual = compute_residual(A_local, x_new, b)
        """
        if not DIST_AVAILABLE or not dist.is_initialized():
            # Single-process fallback: just return (no exchange needed)
            return x
        
        # Use cached send/recv indices and buffers for efficiency
        send_buffers = self._get_send_buffers(x.dtype)
        recv_buffers = self._get_recv_buffers(x.dtype)
        
        # Fill send buffers (vectorized gather)
        for neighbor_id in self.partition.neighbor_partitions:
            send_idx = self._send_indices_cached.get(neighbor_id)
            if send_idx is None:
                send_idx = self.partition.send_indices[neighbor_id].to(self.device)
                self._send_indices_cached[neighbor_id] = send_idx
            send_buffers[neighbor_id].copy_(x[send_idx])
        
        # Use send/recv for p2p communication
        # Note: For NCCL, we use synchronous send/recv
        backend = dist.get_backend() if dist.is_initialized() else 'gloo'
        
        if backend == 'nccl':
            # NCCL: use synchronous send/recv pairs
            for neighbor_id in sorted(self.partition.neighbor_partitions):
                if self.partition.partition_id < neighbor_id:
                    # Lower rank sends first, then receives
                    dist.send(send_buffers[neighbor_id], dst=neighbor_id)
                    dist.recv(recv_buffers[neighbor_id], src=neighbor_id)
                else:
                    # Higher rank receives first, then sends
                    dist.recv(recv_buffers[neighbor_id], src=neighbor_id)
                    dist.send(send_buffers[neighbor_id], dst=neighbor_id)
        else:
            # Gloo: use non-blocking isend/irecv
            requests = []
            for neighbor_id in self.partition.neighbor_partitions:
                req = dist.isend(send_buffers[neighbor_id], dst=neighbor_id)
                requests.append(req)
                req = dist.irecv(recv_buffers[neighbor_id], src=neighbor_id)
                requests.append(req)
            
            if async_op:
                return requests
            
            for req in requests:
                req.wait()
        
        # Update halo values (vectorized scatter)
        for neighbor_id in self.partition.neighbor_partitions:
            recv_idx = self._recv_indices_cached.get(neighbor_id)
            if recv_idx is None:
                recv_idx = self.partition.recv_indices[neighbor_id].to(self.device)
                self._recv_indices_cached[neighbor_id] = recv_idx
            x[recv_idx] = recv_buffers[neighbor_id]
        
        return x
    
    def halo_exchange_local(
        self,
        x_list: List[torch.Tensor]
    ) -> None:
        """
        Local halo exchange for single-process multi-partition simulation.
        
        Useful for testing/debugging without actual distributed setup.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition
        """
        if not hasattr(self, '_all_partitions'):
            return
        
        # Build mapping from global to local for each partition
        for part_id in range(len(x_list)):
            partition = self._all_partitions[part_id]
            x = x_list[part_id]
            
            # For each halo node, find which neighbor owns it and get the value
            halo_offset = len(partition.owned_nodes)
            
            for halo_idx, global_node in enumerate(partition.halo_nodes.tolist()):
                local_halo_idx = halo_offset + halo_idx
                
                # Find which partition owns this node
                for neighbor_id in partition.neighbor_partitions:
                    neighbor_partition = self._all_partitions[neighbor_id]
                    neighbor_g2l = neighbor_partition.global_to_local
                    
                    if global_node < len(neighbor_g2l):
                        local_idx_in_neighbor = neighbor_g2l[global_node].item()
                        if local_idx_in_neighbor >= 0 and local_idx_in_neighbor < len(neighbor_partition.owned_nodes):
                            # This neighbor owns the node
                            x[local_halo_idx] = x_list[neighbor_id][local_idx_in_neighbor]
                            break
    
    def matvec(self, x: torch.Tensor, exchange_halo: bool = True) -> torch.Tensor:
        """
        Local matrix-vector product y = A_local @ x.
        
        Parameters
        ----------
        x : torch.Tensor
            Local vector [num_local]
        exchange_halo : bool
            If True, perform halo exchange before multiplication
            
        Returns
        -------
        y : torch.Tensor
            Result vector [num_local]
        """
        if exchange_halo:
            self.halo_exchange(x)
        
        # Use cached CSR for efficiency
        return torch.mv(self._get_csr(), x)
    
    def matvec_overlap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Matrix-vector product with optional communication-computation overlap.

        The overlap path issues an async halo exchange, runs the interior SpMV
        (rows independent of halo) while comm is in flight, then waits and runs
        the boundary SpMV. Whether overlap actually wins depends on the ratio
        of comm time to host-side P2P setup overhead, which varies wildly with
        interconnect (NVLink vs PCIe vs Ethernet), problem size, and number of
        ranks. To stay honest we **auto-calibrate** on first use: time both the
        overlap path and the plain halo+matvec sequential path a few times,
        cache the faster choice, and dispatch to it for subsequent calls.

        Parameters
        ----------
        x : torch.Tensor
            Local vector [num_local]

        Returns
        -------
        y : torch.Tensor
            Result vector [num_local]
        """
        # In single-process mode, overlap has overhead with no benefit
        if not DIST_AVAILABLE or not dist.is_initialized():
            return self._matvec_sequential(x)

        # Build interior/boundary decomposition if not cached
        if not hasattr(self, '_interior_csr') or self._interior_csr is None:
            self._build_interior_boundary_decomposition()

        # Not enough interior work to justify overlap overhead — static fallback
        if self._overlap_stats.get('interior_ratio', 0) < 0.1:
            return self._matvec_sequential(x)

        # Dynamic fallback: time both paths once, then use the faster one. The
        # decision is cached per-matrix; if matrix values change call
        # _invalidate_cache() (which also clears _overlap_decision).
        if getattr(self, '_overlap_decision', None) is None:
            self._calibrate_overlap_vs_sequential(x)

        if self._overlap_decision == 'sequential':
            return self._matvec_sequential(x)
        return self._matvec_overlap_path(x)

    def _matvec_sequential(self, x: torch.Tensor) -> torch.Tensor:
        """Plain halo_exchange + local matvec (no overlap)."""
        self.halo_exchange(x)
        return self.matvec(x, exchange_halo=False)

    def _matvec_overlap_path(self, x: torch.Tensor) -> torch.Tensor:
        """Async halo + interior SpMV + wait + boundary SpMV (overlap path)."""
        # NVTX ranges make the phases visible in Nsight Systems so we can
        # verify whether async halo P2P actually runs concurrent with the
        # interior SpMV on the GPU timeline.
        nvtx = torch.cuda.nvtx

        nvtx.range_push("matvec_overlap.halo_async_start")
        comm_handle = self.halo_exchange_async(x)
        nvtx.range_pop()

        nvtx.range_push("matvec_overlap.interior_spmv")
        y = torch.zeros(self.num_local, dtype=x.dtype, device=self.device)
        if self._interior_csr is not None and self._interior_csr._nnz() > 0:
            y.add_(torch.mv(self._interior_csr, x))
        nvtx.range_pop()

        nvtx.range_push("matvec_overlap.halo_wait")
        if comm_handle is not None:
            self._wait_halo_exchange(comm_handle, x)
        nvtx.range_pop()

        nvtx.range_push("matvec_overlap.boundary_spmv")
        if self._boundary_csr is not None and self._boundary_csr._nnz() > 0:
            y.add_(torch.mv(self._boundary_csr, x))
        nvtx.range_pop()

        return y

    def _calibrate_overlap_vs_sequential(self, x: torch.Tensor, n_iters: int = 5) -> None:
        """
        Time the overlap and sequential matvec paths on this matrix and cache
        the faster one in ``self._overlap_decision`` ('overlap' or 'sequential').

        We measure wall-clock per path with a sync+barrier on each side so the
        timing reflects the full GPU+host cost. Profiling on A100 PCIe showed
        ``dist.batch_isend_irecv`` carries ~2.5 ms host overhead per call,
        which can dwarf the actual halo transfer; on faster interconnects
        (NVLink, IB) the overlap path can win by a meaningful margin. The
        right answer is therefore hardware-dependent and best measured.
        """
        import time
        # Warm both paths once so we don't bill JIT / lazy-init into the timing
        _ = self._matvec_overlap_path(x)
        _ = self._matvec_sequential(x)
        if x.is_cuda:
            torch.cuda.synchronize(self.device)
        if dist.is_initialized():
            dist.barrier()

        def _time_path(fn):
            if x.is_cuda:
                torch.cuda.synchronize(self.device)
            if dist.is_initialized():
                dist.barrier()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = fn(x)
            if x.is_cuda:
                torch.cuda.synchronize(self.device)
            if dist.is_initialized():
                dist.barrier()
            return (time.perf_counter() - t0) / n_iters

        t_overlap = _time_path(self._matvec_overlap_path)
        t_seq = _time_path(self._matvec_sequential)

        self._overlap_decision = 'overlap' if t_overlap < t_seq else 'sequential'
        # Stash the timings so callers / debuggers can inspect why
        if not hasattr(self, '_overlap_stats') or self._overlap_stats is None:
            self._overlap_stats = {}
        self._overlap_stats['calibrated_overlap_ms'] = t_overlap * 1e3
        self._overlap_stats['calibrated_sequential_ms'] = t_seq * 1e3
        self._overlap_stats['calibrated_decision'] = self._overlap_decision
    
    def _build_interior_boundary_decomposition(self):
        """
        Decompose matrix into interior and boundary parts.
        
        Interior: All entries in rows that only reference owned nodes (col < num_owned)
        Boundary: All entries in rows that reference at least one halo node (col >= num_owned)
        
        This allows computing interior rows while halo exchange is in progress.
        """
        num_owned = self.num_owned
        
        # For each entry, check if it references a halo node
        entry_uses_halo = self.local_col >= num_owned
        
        # For each row, count how many entries use halo
        # Use scatter_add to count halo references per row
        row_halo_count = torch.zeros(self.num_local, dtype=torch.int32, device=self.device)
        ones = torch.ones_like(self.local_row, dtype=torch.int32)
        row_halo_count.scatter_add_(0, self.local_row[entry_uses_halo], ones[entry_uses_halo])
        
        # A row is "interior" if it has zero halo references
        row_is_interior = row_halo_count == 0
        
        # Mark entries by their row type
        interior_mask = row_is_interior[self.local_row]
        boundary_mask = ~interior_mask
        
        # Only consider owned rows for interior (halo rows don't need computation)
        interior_mask = interior_mask & (self.local_row < num_owned)
        boundary_mask = boundary_mask & (self.local_row < num_owned)
        
        # Build interior CSR
        if interior_mask.any():
            interior_coo = torch.sparse_coo_tensor(
                torch.stack([self.local_row[interior_mask], self.local_col[interior_mask]]),
                self.local_values[interior_mask],
                self.local_shape,
                device=self.device
            )
            self._interior_csr = interior_coo.to_sparse_csr()
        else:
            self._interior_csr = None
        
        # Build boundary CSR
        if boundary_mask.any():
            boundary_coo = torch.sparse_coo_tensor(
                torch.stack([self.local_row[boundary_mask], self.local_col[boundary_mask]]),
                self.local_values[boundary_mask],
                self.local_shape,
                device=self.device
            )
            self._boundary_csr = boundary_coo.to_sparse_csr()
        else:
            self._boundary_csr = None
        
        # Cache statistics
        total_nnz_owned = (self.local_row < num_owned).sum().item()
        interior_nnz_count = interior_mask.sum().item()
        boundary_nnz_count = boundary_mask.sum().item()
        self._overlap_stats = {
            'interior_nnz': interior_nnz_count,
            'boundary_nnz': boundary_nnz_count,
            'total_nnz_owned': total_nnz_owned,
            'interior_ratio': interior_nnz_count / total_nnz_owned if total_nnz_owned > 0 else 0,
            'interior_rows': row_is_interior[:num_owned].sum().item(),
            'boundary_rows': (~row_is_interior[:num_owned]).sum().item(),
        }
    
    def halo_exchange_async(self, x: torch.Tensor):
        """
        Start asynchronous halo exchange.
        
        Returns a handle that can be passed to _wait_halo_exchange().
        """
        if not DIST_AVAILABLE or not dist.is_initialized():
            return None
        
        # Use async point-to-point (isend/irecv) for ALL backends, including
        # NCCL. The previous NCCL branch (_halo_exchange_cuda_async) issued
        # *blocking* dist.send/dist.recv on a side CUDA stream -- those block
        # the host thread until the transfer completes, so they do NOT overlap
        # with the interior compute (the whole point of matvec_overlap). NCCL
        # supports isend/irecv as genuine async work handles, so the unified
        # path below actually overlaps communication with computation.
        return self._halo_exchange_async_p2p(x)
    
    def _halo_exchange_cuda_async(self, x: torch.Tensor):
        """Async halo exchange using CUDA streams."""
        # Create communication stream if not exists
        if not hasattr(self, '_comm_stream'):
            self._comm_stream = torch.cuda.Stream(device=self.device)
        
        send_buffers = self._get_send_buffers(x.dtype)
        recv_buffers = self._get_recv_buffers(x.dtype)
        
        # Record current stream
        current_stream = torch.cuda.current_stream(self.device)
        
        # Fill send buffers on current stream
        for neighbor_id in self.partition.neighbor_partitions:
            send_idx = self._send_indices_cached.get(neighbor_id)
            if send_idx is None:
                send_idx = self.partition.send_indices[neighbor_id].to(self.device)
                self._send_indices_cached[neighbor_id] = send_idx
            send_buffers[neighbor_id].copy_(x[send_idx])
        
        # Synchronize before switching streams
        self._comm_stream.wait_stream(current_stream)
        
        # Do communication on comm stream
        with torch.cuda.stream(self._comm_stream):
            for neighbor_id in sorted(self.partition.neighbor_partitions):
                if self.partition.partition_id < neighbor_id:
                    dist.send(send_buffers[neighbor_id], dst=neighbor_id)
                    dist.recv(recv_buffers[neighbor_id], src=neighbor_id)
                else:
                    dist.recv(recv_buffers[neighbor_id], src=neighbor_id)
                    dist.send(send_buffers[neighbor_id], dst=neighbor_id)
        
        return {'type': 'cuda', 'stream': self._comm_stream, 'recv_buffers': recv_buffers}
    
    def _halo_exchange_async_p2p(self, x: torch.Tensor):
        """Async halo exchange via non-blocking P2P (gloo and NCCL).

        Uses ``dist.batch_isend_irecv`` so NCCL groups the sends and receives
        into one P2P transaction. A naive isend/irecv loop deadlocks on NCCL
        once payloads grow past a few KB: both ranks post isend first and
        then irecv, so the NCCL stream is blocked waiting on the peer's irecv
        which has not yet been issued. batch_isend_irecv lets NCCL pair the
        ops correctly and is also the only form that genuinely overlaps with
        compute on NCCL.
        """
        send_buffers = self._get_send_buffers(x.dtype)
        recv_buffers = self._get_recv_buffers(x.dtype)

        # Fill send buffers
        for neighbor_id in self.partition.neighbor_partitions:
            send_idx = self._send_indices_cached.get(neighbor_id)
            if send_idx is None:
                send_idx = self.partition.send_indices[neighbor_id].to(self.device)
                self._send_indices_cached[neighbor_id] = send_idx
            send_buffers[neighbor_id].copy_(x[send_idx])

        # Build a single grouped P2P transaction for all neighbors
        ops = []
        for neighbor_id in self.partition.neighbor_partitions:
            ops.append(dist.P2POp(dist.isend, send_buffers[neighbor_id], neighbor_id))
            ops.append(dist.P2POp(dist.irecv, recv_buffers[neighbor_id], neighbor_id))

        requests = dist.batch_isend_irecv(ops) if ops else []

        return {'type': 'gloo', 'requests': requests, 'recv_buffers': recv_buffers}
    
    def _wait_halo_exchange(self, handle, x: torch.Tensor):
        """Wait for async halo exchange to complete and update x."""
        if handle is None:
            return
        
        if handle['type'] == 'cuda':
            # Synchronize with comm stream
            torch.cuda.current_stream(self.device).wait_stream(handle['stream'])
        elif handle['type'] == 'gloo':
            # Wait for all requests
            for req in handle['requests']:
                req.wait()
        
        # Update halo values
        recv_buffers = handle['recv_buffers']
        for neighbor_id in self.partition.neighbor_partitions:
            recv_idx = self._recv_indices_cached.get(neighbor_id)
            if recv_idx is None:
                recv_idx = self.partition.recv_indices[neighbor_id].to(self.device)
                self._recv_indices_cached[neighbor_id] = recv_idx
            x[recv_idx] = recv_buffers[neighbor_id]
    
    def _get_csr(self) -> torch.Tensor:
        """Get cached CSR matrix (lazy initialization)."""
        if not hasattr(self, '_csr_cache') or self._csr_cache is None:
            A_coo = torch.sparse_coo_tensor(
            torch.stack([self.local_row, self.local_col]),
            self.local_values,
                self.local_shape,
                device=self.device
            )
            self._csr_cache = A_coo.to_sparse_csr()
        return self._csr_cache
    
    def _invalidate_cache(self):
        """Invalidate CSR cache (call if matrix values change)."""
        self._csr_cache = None
        self._diag_cache = None
        self._diag_inv_cache = None
        self._owned_block_csr = None
        self._interior_csr = None
        self._boundary_csr = None
        self._overlap_decision = None
        self._send_buffers_cache = {}
        self._recv_buffers_cache = {}
        self._send_indices_cached = {}
        self._recv_indices_cached = {}
    
    def _get_send_buffers(self, dtype: torch.dtype) -> Dict[int, torch.Tensor]:
        """Get or create cached send buffers."""
        if not hasattr(self, '_send_buffers_cache'):
            self._send_buffers_cache = {}
        if not hasattr(self, '_send_indices_cached'):
            self._send_indices_cached = {}
        
        cache_key = dtype
        if cache_key not in self._send_buffers_cache:
            buffers = {}
            for neighbor_id in self.partition.neighbor_partitions:
                send_idx = self.partition.send_indices[neighbor_id]
                buffers[neighbor_id] = torch.empty(
                    len(send_idx), dtype=dtype, device=self.device
                )
            self._send_buffers_cache[cache_key] = buffers
        
        return self._send_buffers_cache[cache_key]
    
    def _get_recv_buffers(self, dtype: torch.dtype) -> Dict[int, torch.Tensor]:
        """Get or create cached receive buffers."""
        if not hasattr(self, '_recv_buffers_cache'):
            self._recv_buffers_cache = {}
        if not hasattr(self, '_recv_indices_cached'):
            self._recv_indices_cached = {}
        
        cache_key = dtype
        if cache_key not in self._recv_buffers_cache:
            buffers = {}
            for neighbor_id in self.partition.neighbor_partitions:
                recv_idx = self.partition.recv_indices[neighbor_id]
                buffers[neighbor_id] = torch.empty(
                    len(recv_idx), dtype=dtype, device=self.device
                )
            self._recv_buffers_cache[cache_key] = buffers
        
        return self._recv_buffers_cache[cache_key]
    
    def _get_diagonal(self) -> torch.Tensor:
        """Get cached diagonal elements."""
        if not hasattr(self, '_diag_cache') or self._diag_cache is None:
            diag_mask = self.local_row == self.local_col
            diag_indices = self.local_row[diag_mask]
            diag_values = self.local_values[diag_mask]
            self._diag_cache = torch.zeros(self.num_local, dtype=self.dtype, device=self.device)
            self._diag_cache[diag_indices] = diag_values
        return self._diag_cache
    
    def _get_diagonal_inv(self) -> torch.Tensor:
        """Get cached inverse diagonal (for Jacobi preconditioner)."""
        if not hasattr(self, '_diag_inv_cache') or self._diag_inv_cache is None:
            diag = self._get_diagonal()
            self._diag_inv_cache = torch.where(
                diag.abs() > 1e-14,
                1.0 / diag,
                torch.zeros_like(diag)
            )
        return self._diag_inv_cache

    def _get_owned_block_csr(self) -> Optional[torch.Tensor]:
        """
        Cached CSR of the owned diagonal block A_oo (owned x owned).

        This is the local subdomain operator (no halo coupling) used by the
        block-Jacobi / additive-Schwarz preconditioner. Returns None if the
        owned block has no entries.
        """
        if not hasattr(self, '_owned_block_csr') or self._owned_block_csr is None:
            no = self.num_owned
            mask = (self.local_row < no) & (self.local_col < no)
            if not bool(mask.any()):
                self._owned_block_csr = None
                return None
            idx = torch.stack([self.local_row[mask], self.local_col[mask]], dim=0)
            coo = torch.sparse_coo_tensor(
                idx, self.local_values[mask], (no, no),
                device=self.device, dtype=self.local_values.dtype,
            )
            self._owned_block_csr = coo.to_sparse_csr()
        return self._owned_block_csr
    
    def solve(
        self,
        b: torch.Tensor,
        method: str = 'cg',
        preconditioner: str = 'jacobi',
        atol: float = 1e-10,
        rtol: float = 1e-6,
        maxiter: int = 1000,
        verbose: bool = False,
        distributed: bool = True,
        overlap: bool = False,
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Solve linear system Ax = b.
        
        Optimizations enabled by default:
        - CSR cache: Avoids repeated COO->CSR conversion (use_cache=True)
        - Jacobi preconditioner: ~5% speedup for Poisson-like problems
        
        Parameters
        ----------
        b : torch.Tensor
            Right-hand side. Shape [num_owned] for owned nodes only.
        method : str
            Solver method: 'cg' (default), 'jacobi', 'gauss_seidel'
        preconditioner : str
            Preconditioner for CG: 'none', 'jacobi' (default), 'ssor', 'ic0', 'polynomial'
        atol : float
            Absolute tolerance for convergence
        rtol : float
            Relative tolerance for convergence (|r| < rtol * |b|)
        maxiter : int
            Maximum iterations
        verbose : bool
            Print convergence info (rank 0 only for distributed)
        distributed : bool, default=True
            If True (default): Solve the GLOBAL system using distributed
            algorithms with all_reduce for global dot products.
            If False: Solve only the LOCAL subdomain problem (useful as
            preconditioner in domain decomposition methods).
        overlap : bool, default=False
            If True: Overlap communication with computation.
            Note: Only beneficial for slow interconnects (InfiniBand, Ethernet).
            For NVLink, synchronous communication is faster.
        use_cache : bool, default=True
            If True (default): Cache CSR format and diagonal for reuse.
            Provides ~2% speedup and ~27% memory reduction.
            
        Returns
        -------
        x : torch.Tensor
            Solution for owned nodes, shape [num_owned]
            
        Examples
        --------
        >>> # Distributed solve (default) - all ranks cooperate
        >>> x = local_matrix.solve(b_owned)
        
        >>> # Local subdomain solve - no global communication
        >>> x = local_matrix.solve(b_owned, distributed=False)
        
        >>> # With different preconditioner
        >>> x = local_matrix.solve(b_owned, preconditioner='ssor')
        
        >>> # Disable caching (for memory-constrained cases)
        >>> x = local_matrix.solve(b_owned, use_cache=False)
        """
        # Invalidate cache if not using it
        if not use_cache:
            self._invalidate_cache()
        
        if distributed:
            return self._solve_distributed_pcg(b, preconditioner, atol, rtol, maxiter, verbose, overlap)
        else:
            return self._solve_local(b, method, atol, maxiter, verbose)
    
    def _solve_local(
        self,
        b: torch.Tensor,
        method: str,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> torch.Tensor:
        """Local subdomain solve (no global communication)."""
        # Handle b size
        if b.shape[0] == self.num_owned:
            b_full = torch.zeros(self.num_local, dtype=b.dtype, device=self.device)
            b_full[:self.num_owned] = b
            b = b_full
        elif b.shape[0] != self.num_local:
            raise ValueError(f"b must have size num_owned={self.num_owned} or num_local={self.num_local}")
        
        x = torch.zeros(self.num_local, dtype=b.dtype, device=self.device)
        
        if method == 'jacobi':
            x = self._solve_jacobi(x, b, atol, maxiter, verbose)
        elif method == 'gauss_seidel':
            x = self._solve_gauss_seidel(x, b, atol, maxiter, verbose)
        else:  # CG
            x = self._solve_cg(x, b, atol, maxiter, verbose)
        
        return x[:self.num_owned]
    
    def _solve_cg(self, x, b, atol, maxiter, verbose):
        """
        Local CG solver for subdomain problems.
        
        This solves only the local subdomain problem without global reductions.
        Useful as a preconditioner or subdomain solver in domain decomposition.
        """
        r = b - self.matvec(x)
        p = r.clone()
        rs_old = torch.dot(r[:self.num_owned], r[:self.num_owned])
        
        for i in range(maxiter):
            Ap = self.matvec(p)
            pAp = torch.dot(p[:self.num_owned], Ap[:self.num_owned])
            
            if pAp.abs() < 1e-30:
                break
                
            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            
            rs_new = torch.dot(r[:self.num_owned], r[:self.num_owned])
            
            if verbose and i % 100 == 0:
                print(f"  CG iter {i}: residual = {rs_new.sqrt():.2e}")
            
            if rs_new.sqrt() < atol:
                if verbose:
                    print(f"  CG converged at iter {i}")
                break
            
            if rs_old.abs() < 1e-30:
                break
                
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        
        return x
    
    def _solve_jacobi(self, x, b, atol, maxiter, verbose):
        """Optimized Jacobi iteration with cached diagonal."""
        D_inv = self._get_diagonal_inv()
        D = self._get_diagonal()
        
        for i in range(maxiter):
            # Halo exchange
            self.halo_exchange(x)
            
            # x_new = D^{-1} @ (b - (A - D) @ x) = D^{-1} @ (b - A @ x + D @ x)
            Ax = self.matvec(x, exchange_halo=False)
            x_new = D_inv * (b - Ax + D * x)
            
            # Convergence check on owned nodes only
            diff = (x_new[:self.num_owned] - x[:self.num_owned]).norm()
            x = x_new
            
            if verbose and i % 100 == 0:
                print(f"  Jacobi iter {i}: diff = {diff:.2e}")
            
            if diff < atol:
                if verbose:
                    print(f"  Jacobi converged at iter {i}")
                break
        
        return x
    
    def _solve_gauss_seidel(self, x, b, atol, maxiter, verbose):
        """
        Gauss-Seidel iteration with halo exchange.
        
        Note: True GS requires sequential updates, which is slow on GPU.
        This implementation uses a hybrid approach:
        - On CPU: Use sparse triangular solve (faster than Python loop)
        - On GPU: Fall back to damped Jacobi (parallel, similar convergence)
        """
        if self.device.type == 'cuda':
            # GPU: Use damped Jacobi as approximation (parallel)
            return self._solve_damped_jacobi(x, b, atol, maxiter, verbose, omega=0.8)
        
        # CPU: Use SciPy's efficient sparse triangular solve
        D_inv = self._get_diagonal_inv()
        D = self._get_diagonal()
        
        # Get CSR for efficient access
        A_csr = self._get_csr()
        
        for iteration in range(maxiter):
            x_old = x.clone()
            
            # Exchange halo before sweep
            self.halo_exchange(x)
            
            # Compute residual and apply diagonal scaling
            # This is symmetric GS approximation
            Ax = self.matvec(x, exchange_halo=False)
            r = b - Ax
            x = x + D_inv * r
            
            diff = (x[:self.num_owned] - x_old[:self.num_owned]).norm()
            
            if verbose and iteration % 100 == 0:
                print(f"  GS iter {iteration}: diff = {diff:.2e}")
            
            if diff < atol:
                if verbose:
                    print(f"  GS converged at iter {iteration}")
                break
        
        return x
    
    def _solve_damped_jacobi(self, x, b, atol, maxiter, verbose, omega=0.8):
        """Damped Jacobi iteration (parallel-friendly for GPU)."""
        D_inv = self._get_diagonal_inv()
        D = self._get_diagonal()
        
        for i in range(maxiter):
            self.halo_exchange(x)
            Ax = self.matvec(x, exchange_halo=False)
            
            # x_new = x + omega * D^{-1} @ (b - A @ x)
            x_new = x + omega * D_inv * (b - Ax)
            
            diff = (x_new[:self.num_owned] - x[:self.num_owned]).norm()
            x = x_new
            
            if verbose and i % 100 == 0:
                print(f"  Damped Jacobi iter {i}: diff = {diff:.2e}")
            
            if diff < atol:
                if verbose:
                    print(f"  Damped Jacobi converged at iter {i}")
                break
        
        return x
    
    def _solve_distributed_cg(
        self,
        b_owned: torch.Tensor,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> torch.Tensor:
        """Legacy CG solver - use _solve_distributed_pcg instead."""
        return self._solve_distributed_pcg(b_owned, 'none', atol, 1e-6, maxiter, verbose, overlap=True)
    
    def _solve_distributed_pcg(
        self,
        b_owned: torch.Tensor,
        preconditioner: str,
        atol: float,
        rtol: float,
        maxiter: int,
        verbose: bool,
        overlap: bool = True
    ) -> torch.Tensor:
        """
        Distributed Preconditioned Conjugate Gradient solver.
        
        Optimizations over basic CG:
        1. Cached CSR format for matvec
        2. Jacobi/block-Jacobi preconditioning
        3. Relative tolerance support
        4. Reduced memory allocations
        5. Communication-computation overlap (when overlap=True)
        """
        num_owned = self.num_owned
        num_local = self.num_local
        dtype = b_owned.dtype
        device = self.device
        rank = self.partition.partition_id
        
        # Initialize x_local = 0 (owned + halo)
        x_local = torch.zeros(num_local, dtype=dtype, device=device)
        
        # Extend b to local size (halo part is 0)
        b_local = torch.zeros(num_local, dtype=dtype, device=device)
        b_local[:num_owned] = b_owned
        
        # Compute initial |b| for relative tolerance
        b_norm_local = torch.dot(b_owned, b_owned)
        b_norm = self._global_reduce_sum(b_norm_local).sqrt()
        tol = max(atol, rtol * b_norm)
        
        # r = b - A @ x (no halo exchange needed for x=0)
        r_local = b_local.clone()
        
        # Apply preconditioner: z = M^{-1} @ r
        z_local = self._apply_preconditioner(r_local, preconditioner)
        
        # p = z
        p_local = z_local.clone()
        
        # rz_old = r^T @ z (global reduction, only owned nodes)
        rz_local = torch.dot(r_local[:num_owned], z_local[:num_owned])
        rz_old = self._global_reduce_sum(rz_local)
        
        # For convergence check
        rs_local = torch.dot(r_local[:num_owned], r_local[:num_owned])
        rs_old = self._global_reduce_sum(rs_local)
        
        # Print overlap info on first call
        if verbose and rank == 0 and overlap:
            if hasattr(self, '_overlap_stats'):
                stats = self._overlap_stats
                print(f"  Overlap enabled: interior_ratio = {stats['interior_ratio']:.1%}")
        
        for i in range(maxiter):
            # Ap = A @ p with optional overlap
            if overlap:
                Ap_local = self.matvec_overlap(p_local)
            else:
                self.halo_exchange(p_local)
                Ap_local = self.matvec(p_local, exchange_halo=False)
            
            # pAp = p^T @ A @ p (global reduction)
            pAp_local = torch.dot(p_local[:num_owned], Ap_local[:num_owned])
            pAp = self._global_reduce_sum(pAp_local)
            
            if pAp.abs() < 1e-30:
                break
            
            alpha = rz_old / pAp
            
            # Update x and r (in-place for efficiency)
            x_local.add_(p_local, alpha=alpha)
            r_local.add_(Ap_local, alpha=-alpha)
            
            # Compute residual norm for convergence check
            rs_local = torch.dot(r_local[:num_owned], r_local[:num_owned])
            rs_new = self._global_reduce_sum(rs_local)
            residual = rs_new.sqrt()
            
            if verbose and rank == 0 and i % 50 == 0:
                print(f"  PCG iter {i}: residual = {residual:.2e}, tol = {tol:.2e}")
            
            if residual < tol:
                if verbose and rank == 0:
                    print(f"  PCG converged at iter {i}, residual = {residual:.2e}")
                break
            
            # Apply preconditioner: z = M^{-1} @ r
            z_local = self._apply_preconditioner(r_local, preconditioner)
            
            # rz_new = r^T @ z
            rz_local = torch.dot(r_local[:num_owned], z_local[:num_owned])
            rz_new = self._global_reduce_sum(rz_local)
            
            beta = rz_new / rz_old
            
            # p = z + beta * p (in-place)
            p_local.mul_(beta).add_(z_local)
            rz_old = rz_new
        
        # Return only owned part
        return x_local[:num_owned]
    
    def _apply_preconditioner(
        self,
        r: torch.Tensor,
        preconditioner: str
    ) -> torch.Tensor:
        """
        Apply preconditioner M^{-1} @ r.
        
        Parameters
        ----------
        r : torch.Tensor
            Residual vector [num_local]
        preconditioner : str
            'none', 'jacobi', 'block_jacobi', 'ssor', 'ic0', 'polynomial'
            
        Returns
        -------
        z : torch.Tensor
            Preconditioned residual [num_local]
        """
        if preconditioner == 'none':
            return r.clone()
        
        elif preconditioner == 'jacobi':
            # z = D^{-1} @ r
            D_inv = self._get_diagonal_inv()
            return D_inv * r
        
        elif preconditioner == 'block_jacobi':
            # Solve local subdomain (few iterations of local CG or direct)
            z = torch.zeros_like(r)
            z[:self.num_owned] = self._local_solve_approx(
                r[:self.num_owned], maxiter=5
            )
            return z
        
        elif preconditioner == 'ssor':
            # Symmetric SOR: (D + ωL) D^{-1} (D + ωU)
            omega = 1.5
            return self._apply_ssor(r, omega)
        
        elif preconditioner == 'ic0':
            # Incomplete Cholesky (GPU-friendly iterative version)
            return self._apply_ic0(r, num_sweeps=2)
        
        elif preconditioner == 'polynomial':
            # Neumann series polynomial preconditioner
            return self._apply_polynomial(r, degree=3)
        
        else:
            warnings.warn(f"Unknown preconditioner '{preconditioner}', using none")
            return r.clone()
    
    def _local_solve_approx(
        self,
        b_owned: torch.Tensor,
        maxiter: int = 5,
        omega: float = 0.8,
    ) -> torch.Tensor:
        """
        Approximate local solve A_oo x ~= b_owned for the block-Jacobi
        (restricted additive Schwarz) preconditioner.

        Runs ``maxiter`` damped-Jacobi sweeps on the *owned diagonal block*:

            x_{k+1} = x_k + omega * D_oo^{-1} (b_owned - A_oo @ x_k)

        This is a fixed *linear* operator (so it is a valid CG preconditioner,
        unlike inner CG), needs no halo exchange (owned block only), and uses
        the local off-diagonal coupling -- so it is genuinely stronger than a
        single diagonal Jacobi step. Pure PyTorch, works on CPU and CUDA.
        """
        D_inv = self._get_diagonal_inv()[:self.num_owned]
        A_oo = self._get_owned_block_csr()

        # First sweep from x0 = 0 is just the damped diagonal solve.
        x = omega * D_inv * b_owned
        if A_oo is None:
            return x

        for _ in range(maxiter - 1):
            # residual on the owned block, then damped diagonal correction
            r = b_owned - torch.mv(A_oo, x)
            x = x + omega * D_inv * r

        return x
    
    def _apply_ssor(self, r: torch.Tensor, omega: float = 1.5) -> torch.Tensor:
        """
        Apply SSOR preconditioner (GPU-friendly scaled Jacobi approximation).
        
        True SSOR requires sequential sweeps, slow on GPU.
        This uses a scaled Jacobi that approximates SSOR behavior.
        """
        import math
        D_inv = self._get_diagonal_inv()
        scale = math.sqrt(omega * (2 - omega))
        return scale * D_inv * r
    
    def _apply_ic0(self, r: torch.Tensor, num_sweeps: int = 2) -> torch.Tensor:
        """
        Apply Incomplete Cholesky (IC0) preconditioner using Jacobi iterations.
        
        GPU-friendly approximation of (D + L)^{-1} D (D + L^T)^{-1}.
        Uses parallel Jacobi sweeps for triangular solves.
        """
        # Get or build L/U matrices
        if not hasattr(self, '_ic0_L_csr') or self._ic0_L_csr is None:
            self._build_ic0_factors()
        
        D_inv = self._get_diagonal_inv()
        diag = self._get_diagonal()
        
        if self._ic0_L_csr is None:
            # No off-diagonal elements, just Jacobi
            return D_inv * r
        
        # Forward sweep: solve (D + L) y = r approximately
        # y^{k+1} = D^{-1} (r - L y^k)
        y = D_inv * r
        for _ in range(num_sweeps):
            Ly = torch.mv(self._ic0_L_csr, y)
            y = D_inv * (r - Ly)
        
        # Middle: scale by D
        z = diag * y
        
        # Backward sweep: solve (D + L^T) x = z approximately  
        # x^{k+1} = D^{-1} (z - L^T x^k)
        x = D_inv * z
        for _ in range(num_sweeps):
            Ux = torch.mv(self._ic0_U_csr, x)
            x = D_inv * (z - Ux)
        
        return x
    
    def _build_ic0_factors(self):
        """Build L and U factors for IC0 preconditioner."""
        n = self.num_local
        
        # Get strictly lower triangular part
        lower_mask = self.local_row > self.local_col
        L_row = self.local_row[lower_mask]
        L_col = self.local_col[lower_mask]
        L_val = self.local_values[lower_mask]
        
        if len(L_val) > 0:
            L_indices = torch.stack([L_row, L_col], dim=0)
            L_coo = torch.sparse_coo_tensor(
                L_indices, L_val, (n, n),
                device=self.device, dtype=self.local_values.dtype
            )
            self._ic0_L_csr = L_coo.to_sparse_csr()
            
            # Upper triangular (transpose of L)
            U_indices = torch.stack([L_col, L_row], dim=0)
            U_coo = torch.sparse_coo_tensor(
                U_indices, L_val, (n, n),
                device=self.device, dtype=self.local_values.dtype
            )
            self._ic0_U_csr = U_coo.to_sparse_csr()
        else:
            self._ic0_L_csr = None
            self._ic0_U_csr = None
    
    def _apply_polynomial(self, r: torch.Tensor, degree: int = 3) -> torch.Tensor:
        """
        Apply Neumann series polynomial preconditioner.
        
        Uses M^{-1} ≈ D^{-1} (I + N + N^2 + ...) where N = I - D^{-1}A
        
        This is stable and parallelizes well on GPU.
        """
        D_inv = self._get_diagonal_inv()
        
        # z = D^{-1} @ r (degree=0 term)
        z = D_inv * r
        
        if degree == 0:
            return z
        
        # Neumann series: sum_{k=0}^{degree} (I - D^{-1}A)^k @ (D^{-1} @ r)
        y = r.clone()
        for _ in range(degree):
            # y = (I - D^{-1}A) @ y
            Ay = self._matvec_local(y)
            y = y - D_inv * Ay
            z = z + D_inv * y
        
        return z
    
    def _matvec_local(self, x: torch.Tensor) -> torch.Tensor:
        """Local matrix-vector product without halo exchange."""
        csr = self._get_csr()
        return torch.mv(csr, x)
    
    def _global_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        """Perform global all_reduce sum."""
        if not DIST_AVAILABLE or not dist.is_initialized():
            return value
        
        # Ensure tensor is on the correct device for the backend
        backend = dist.get_backend()
        if backend == 'nccl' and not value.is_cuda:
            # NCCL requires CUDA tensors
            value = value.to(self.device)
        
        result = value.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result
    
    def eigsh(
        self,
        k: int = 6,
        which: str = "LM",
        maxiter: int = 200,
        tol: float = 1e-8,
        verbose: bool = False,
        distributed: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute k eigenvalues of symmetric matrix.
        
        Parameters
        ----------
        k : int
            Number of eigenvalues to compute
        which : str
            Which eigenvalues: "LM" (largest magnitude), "SM" (smallest magnitude)
        maxiter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print convergence info (rank 0 only)
        distributed : bool, default=True
            If True (default): Use distributed LOBPCG with global reductions.
            If False: Gather to single SparseTensor and compute locally
            (not recommended for large matrices).
            
        Returns
        -------
        eigenvalues : torch.Tensor
            k eigenvalues, shape [k]
        eigenvectors_owned : torch.Tensor
            Eigenvectors for owned nodes only, shape [num_owned, k]
        """
        if not distributed:
            # Gather to single node (not recommended)
            import warnings
            warnings.warn("distributed=False gathers entire matrix to one node. "
                         "Use distributed=True for large-scale problems.")
            st = self.to_sparse_tensor()
            eigenvalues, eigenvectors = st.eigsh(k=k, which=which)
            # Extract local portion
            owned_nodes = self.partition.owned_nodes
            return eigenvalues, eigenvectors[owned_nodes]
        n = self.global_shape[0]
        num_owned = self.num_owned
        rank = self.partition.partition_id
        dtype = self.local_values.dtype
        device = self.device
        
        # Initialize random subspace
        torch.manual_seed(42 + rank)  # Different per rank for diversity
        m = min(2 * k, n)
        
        # Each rank has its local portion of X
        X_owned = torch.randn(num_owned, m, dtype=dtype, device=device)
        
        # Orthogonalize globally
        X_owned = self._global_orthogonalize(X_owned)
        
        eigenvalues_prev = None
        
        for iteration in range(maxiter):
            # Distributed matvec: AX
            AX_owned = self._global_matvec_batch(X_owned)
            
            # Rayleigh-Ritz: H = X^T @ AX (global reduction)
            # Local contribution
            H_local = X_owned.T @ AX_owned
            H = self._global_reduce_sum(H_local)
            
            # Solve small eigenvalue problem (same on all ranks)
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            
            # Sort eigenvalues
            if which == "LM":
                idx_sort = eigenvalues.abs().argsort(descending=True)
            else:
                idx_sort = eigenvalues.abs().argsort()
            eigenvalues = eigenvalues[idx_sort]
            eigenvectors = eigenvectors[:, idx_sort]
            
            # Update X = X @ V (local)
            X_owned = X_owned @ eigenvectors
            
            # Check convergence
            if eigenvalues_prev is not None:
                diff = (eigenvalues[:k] - eigenvalues_prev[:k]).abs()
                if (diff < tol * eigenvalues[:k].abs().clamp(min=1e-10)).all():
                    if verbose and rank == 0:
                        print(f"  Distributed LOBPCG converged at iteration {iteration}")
                    break
            eigenvalues_prev = eigenvalues.clone()
            
            if verbose and rank == 0 and iteration % 20 == 0:
                print(f"  Distributed LOBPCG iter {iteration}: λ_0 = {eigenvalues[0]:.6f}")
            
            # Expand subspace with residual
            if iteration < maxiter - 1:
                AX_new = self._global_matvec_batch(X_owned)
                residual = AX_new - X_owned * eigenvalues.unsqueeze(0)
                
                # Combine and orthogonalize
                combined = torch.cat([X_owned[:, :k], residual[:, :k]], dim=1)
                X_owned = self._global_orthogonalize(combined)
                
                # Ensure correct size
                if X_owned.size(1) < m:
                    extra = torch.randn(num_owned, m - X_owned.size(1), dtype=dtype, device=device)
                    X_owned = torch.cat([X_owned, extra], dim=1)
                    X_owned = self._global_orthogonalize(X_owned)
        
        return eigenvalues[:k], X_owned[:, :k]
    
    def _global_matvec_batch(self, X_owned: torch.Tensor) -> torch.Tensor:
        """
        Distributed matvec for a batch of vectors.
        
        Each rank computes A @ X for its local portion.
        """
        num_owned = self.num_owned
        num_local = self.num_local
        m = X_owned.size(1)
        dtype = X_owned.dtype
        device = self.device
        
        # Extend to local size (owned + halo)
        X_local = torch.zeros(num_local, m, dtype=dtype, device=device)
        X_local[:num_owned] = X_owned
        
        # Gather global X for halo (simplified - in production use p2p)
        X_global = self._gather_all_vectors(X_owned)
        
        # Fill halo from global
        halo_nodes = self.partition.halo_nodes
        if len(halo_nodes) > 0:
            X_local[num_owned:] = X_global[halo_nodes]
        
        # Local matvec for each column
        Y_local = torch.zeros(num_local, m, dtype=dtype, device=device)
        for j in range(m):
            Y_local[:, j] = self.matvec(X_local[:, j], exchange_halo=False)
        
        return Y_local[:num_owned]
    
    def _gather_all_vectors(self, X_owned: torch.Tensor) -> torch.Tensor:
        """Gather vectors from all ranks to build global vector."""
        n = self.global_shape[0]
        m = X_owned.size(1)
        dtype = X_owned.dtype
        device = self.device
        
        X_global = torch.zeros(n, m, dtype=dtype, device=device)
        owned_nodes = self.partition.owned_nodes
        X_global[owned_nodes] = X_owned
        
        # All-reduce to combine
        self._global_reduce_sum_inplace(X_global)
        
        return X_global
    
    def _global_reduce_sum_inplace(self, tensor: torch.Tensor) -> None:
        """In-place global all_reduce sum."""
        if DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    def _global_orthogonalize(self, X_owned: torch.Tensor) -> torch.Tensor:
        """
        Globally orthogonalize a distributed matrix using TSQR.
        
        Simplified version: gather, QR, scatter.
        Production version would use TSQR for better scalability.
        """
        # Gather global X
        X_global = self._gather_all_vectors(X_owned)
        
        # QR on global (same result on all ranks)
        Q, _ = torch.linalg.qr(X_global)
        
        # Extract local portion
        owned_nodes = self.partition.owned_nodes
        return Q[owned_nodes]
    
    def gather_global(self, x_local: torch.Tensor) -> torch.Tensor:
        """Gather local vectors into the global vector on **every** rank.

        Semantics match MPI ``Allgather``: after the call every rank
        holds an identical ``[global_shape[0]]``-sized tensor whose
        entry at global index ``i`` equals the owning rank's
        ``x_local[local_index_of(i)]``.

        Use this for I/O and debugging (write the full solution to
        disk, compare against scipy single-process). Don't call it in
        a Krylov hot loop -- distributed CG / GMRES should keep
        vectors local and use ``dist.all_reduce`` for inner products.

        Parameters
        ----------
        x_local : torch.Tensor
            Local vector [num_owned] (halo entries beyond ``num_owned``
            are ignored).

        Returns
        -------
        x_global : torch.Tensor
            Global vector [global_shape[0]] on every rank.
        """
        if not DIST_AVAILABLE or not dist.is_initialized():
            # Single process: just expand to global
            x_global = torch.zeros(self.global_shape[0], dtype=x_local.dtype, device=x_local.device)
            x_global[self.partition.owned_nodes] = x_local[:self.num_owned]
            return x_global

        # Distributed all-gather. We need two things from every rank
        # to reconstruct the global vector: the owned values *and*
        # the global indices each rank owns. The previous code did a
        # one-to-rank-0 gather and forgot to apply the gathered values
        # (inline TODO admitted as much). Switch to ``dist.all_gather``
        # so every rank ends up with an identical x_global -- matches
        # the symmetric MPI Allgather semantics.
        world_size = dist.get_world_size()

        owned_vals  = x_local[:self.num_owned].contiguous()
        owned_nodes = self.partition.owned_nodes.to(
            device=self.device, dtype=torch.int64).contiguous()

        # Each rank's num_owned can differ, so we exchange sizes first
        # to size the all_gather buffers correctly.
        local_size = torch.tensor([self.num_owned],
                                  dtype=torch.int64, device=self.device)
        sizes = [torch.zeros(1, dtype=torch.int64, device=self.device)
                 for _ in range(world_size)]
        dist.all_gather(sizes, local_size)

        nodes_list = [torch.zeros(int(s.item()), dtype=torch.int64,
                                  device=self.device)
                      for s in sizes]
        vals_list  = [torch.zeros(int(s.item()), dtype=x_local.dtype,
                                  device=self.device)
                      for s in sizes]
        dist.all_gather(nodes_list, owned_nodes)
        dist.all_gather(vals_list,  owned_vals)

        x_global = torch.zeros(self.global_shape[0],
                               dtype=x_local.dtype, device=self.device)
        for nodes, vals in zip(nodes_list, vals_list):
            x_global[nodes] = vals
        return x_global
    
    def det(self) -> torch.Tensor:
        """
        Compute determinant of the distributed sparse matrix.
        
        NOTE: DSparseMatrix represents a single partition. To compute the
        determinant of the full global matrix, you need to use DSparseTensor
        which manages all partitions, or manually gather all partitions.
        
        This method raises an error to guide users to the correct approach.
        
        Raises
        ------
        NotImplementedError
            DSparseMatrix is a single partition. Use DSparseTensor.det() instead.
            
        Examples
        --------
        >>> # Correct way: Use DSparseTensor
        >>> from torch_sla import DSparseTensor
        >>> D = DSparseTensor(val, row, col, shape, num_partitions=4)
        >>> det = D.det()  # This works
        >>>
        >>> # If you have individual DSparseMatrix partitions, you need to
        >>> # reconstruct the global matrix first
        """
        raise NotImplementedError(
            "DSparseMatrix represents a single partition of a distributed matrix. "
            "To compute the determinant of the full global matrix, use DSparseTensor.det() instead, "
            "which manages all partitions and can gather the full matrix for determinant computation.\n\n"
            "Example:\n"
            "  from torch_sla import DSparseTensor\n"
            "  D = DSparseTensor(val, row, col, shape, num_partitions=4)\n"
            "  det = D.det()  # Gathers all partitions and computes determinant"
        )
    
    def __repr__(self) -> str:
        return (f"DSparseMatrix(partition={self.partition.partition_id}/{self.num_partitions}, "
                f"local={self.num_local} ({self.num_owned}+{self.num_halo}), "
                f"global={self.global_shape}, nnz={self.nnz}, device={self.device})")
    
    # =========================================================================
    # Persistence (I/O)
    # =========================================================================
    
    @classmethod
    def load(
        cls,
        directory: Union[str, "os.PathLike"],
        rank: int,
        world_size: Optional[int] = None,
        device: Union[str, torch.device] = "cpu"
    ) -> "DSparseMatrix":
        """
        Load a partition from disk for the given rank.
        
        Each rank should call this with its own rank to load only its partition.
        
        Parameters
        ----------
        directory : str or PathLike
            Directory containing partitioned data.
        rank : int
            Rank of this process.
        world_size : int, optional
            Total number of processes (must match num_partitions).
        device : str or torch.device
            Device to load tensors to.
        
        Returns
        -------
        DSparseMatrix
            The partition for this rank.
        
        Example
        -------
        >>> rank = dist.get_rank()
        >>> world_size = dist.get_world_size()
        >>> partition = DSparseMatrix.load("matrix_dist", rank, world_size, "cuda")
        """
        from .io import load_partition
        return load_partition(directory, rank, world_size, device)


def create_distributed_matrices(
    values: torch.Tensor,
    row: torch.Tensor,
    col: torch.Tensor,
    shape: Tuple[int, int],
    num_partitions: int,
    coords: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = 'cpu'
) -> List[DSparseMatrix]:
    """
    Create all distributed matrix partitions for local simulation.
    
    .. deprecated::
        Use DSparseTensor instead for a more Pythonic interface.
    
    Useful for testing/debugging without actual distributed setup.
    
    Parameters
    ----------
    values, row, col : torch.Tensor
        Global COO sparse matrix data
    shape : Tuple[int, int]
        Global matrix shape
    num_partitions : int
        Number of partitions
    coords : torch.Tensor, optional
        Node coordinates for geometric partitioning
    device : str or torch.device
        Device for all partitions ('cpu', 'cuda', 'cuda:0', etc.)
    
    Returns
    -------
    List[DSparseMatrix]
        List of DSparseMatrix, one per partition
    """
    warnings.warn(
        "create_distributed_matrices is deprecated. Use DSparseTensor instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    matrices = []
    
    # Compute partition IDs once
    if coords is not None:
        partition_ids = partition_coordinates(coords, num_partitions)
    else:
        partition_ids = partition_graph_metis(row, col, shape[0], num_partitions)
    
    for i in range(num_partitions):
        mat = DSparseMatrix.from_global(
            values, row, col, shape, num_partitions, i,
            partition_ids=partition_ids, device=device
        )
        matrices.append(mat)
    
    # Store reference to all partitions for local halo exchange
    for mat in matrices:
        mat._all_partitions = [m.partition for m in matrices]
    
    return matrices


# ====================================================================== #
# DTensor-mirror placement vocabulary for sparse tensors.
#
# Adapted from ``torch.distributed.tensor.placement_types``:
#
# * :class:`Replicated`        --  every rank holds the full matrix
#                                  (analogous to ``Replicate()``).
# * :class:`RowPartitioned`    --  rows are split across ranks via an
#                                  irregular METIS / RCB / simple map
#                                  (the sparse analog of ``Shard(0)``;
#                                  *not* the uniform DTensor shard).
#
# A :class:`DSparseSpec` bundles a placement with the device mesh and
# global shape so the rest of the API can mirror DTensor's
# ``DTensor._spec``.
# ====================================================================== #
@dataclass(frozen=True)
class Replicated:
    """DSparseTensor placement: every rank holds the entire matrix."""
    pass


@dataclass(frozen=True)
class SparseShard:
    """DSparseTensor placement: shard one of the **sparse** axes of the
    underlying :class:`SparseTensor` across the mesh.

    Mirrors ``torch.distributed.tensor.Shard(dim)`` -- one generic
    placement parameterised by axis number, not separate classes per
    direction. For a 2-D sparse matrix ``(M, N)`` the canonical axes
    are 0 (rows) and 1 (cols); for a batched / block-sparse tensor
    ``(*batch, M, N, *block)`` the row axis is ``axis=len(batch_shape)``
    and the col axis is ``axis=len(batch_shape) + 1``.

    Unlike DTensor's ``Shard(dim)`` -- which assumes uniform chunks of
    size ``N/world_size`` -- a sparse shard can carry an irregular
    partition map (METIS / hypergraph / RCB). The map lives on each
    rank's :class:`DSparseMatrix` for now; this placement class is a
    cheap marker carrying only ``axis``. Phase B of the
    ``DSparseMatrix`` dissolution will fold the partition metadata
    in here.
    """
    axis: int = 0


def row_shard(axis_offset: int = 0) -> SparseShard:
    """Convenience: row sharding for a ``(*batch, M, N, *block)`` tensor
    where there are ``axis_offset`` leading batch dims. Defaults to a
    plain 2-D matrix (``axis_offset=0``)."""
    return SparseShard(axis=axis_offset)


def col_shard(axis_offset: int = 0) -> SparseShard:
    """Convenience: col sharding for a ``(*batch, M, N, *block)`` tensor.
    Defaults to a plain 2-D matrix (``axis_offset=0``)."""
    return SparseShard(axis=axis_offset + 1)


# ---------------------------------------------------------------------- #
# Backward-compatibility aliases.
# Pre-rename: ``RowPartitioned()`` -- empty marker for "row-sharded".
# Post-rename: ``SparseShard(axis=0)`` is the canonical name.
# Kept as a deprecated alias so external callers keep working.
# ---------------------------------------------------------------------- #
def RowPartitioned() -> SparseShard:  # noqa: N802 -- legacy capital
    """Deprecated alias for :func:`row_shard()` / ``SparseShard(axis=0)``."""
    return SparseShard(axis=0)


SparsePlacement = Union[Replicated, SparseShard]


@dataclass(frozen=True)
class DSparseSpec:
    """The sparse analog of :class:`torch.distributed.tensor.DTensorSpec`.

    Bundles the placement, the device mesh, and the global shape so we
    can dispatch operations purely off the spec and treat
    :class:`DSparseTensor` like any other distributed tensor.
    """
    placement: SparsePlacement
    mesh: Any                            # torch.distributed.DeviceMesh
    global_shape: Tuple[int, int]


class DSparseTensor:
    """
    Distributed Sparse Tensor with automatic partitioning and halo exchange.

    A Pythonic wrapper that provides a unified interface for distributed
    sparse matrix operations. Supports indexing to access individual partitions.
    
    Parameters
    ----------
    values : torch.Tensor
        Non-zero values [nnz]
    row_indices : torch.Tensor
        Row indices [nnz]
    col_indices : torch.Tensor
        Column indices [nnz]
    shape : Tuple[int, int]
        Matrix shape (m, n)
    num_partitions : int
        Number of partitions to create
    coords : torch.Tensor, optional
        Node coordinates for geometric partitioning [num_nodes, dim]
    partition_method : str
        Partitioning method: 'metis', 'rcb', 'slicing', 'simple'
    device : str or torch.device
        Device for the matrix data
    verbose : bool
        Whether to print partition info
    
    Example
    -------
    >>> import torch
    >>> from torch_sla import DSparseTensor
    >>> 
    >>> # Create distributed tensor with 4 partitions
    >>> A = DSparseTensor(val, row, col, shape, num_partitions=4)
    >>> 
    >>> # Access individual partitions
    >>> A0 = A[0]  # First partition
    >>> A1 = A[1]  # Second partition
    >>> 
    >>> # Iterate over partitions
    >>> for partition in A:
    >>>     x = partition.solve(b_local)
    >>> 
    >>> # Properties
    >>> print(A.num_partitions)  # 4
    >>> print(A.shape)           # Global shape
    >>> print(len(A))            # 4
    >>> 
    >>> # Move to CUDA
    >>> A_cuda = A.cuda()
    >>> 
    >>> # Local halo exchange (for testing)
    >>> x_list = [torch.zeros(A[i].num_local) for i in range(4)]
    >>> A.halo_exchange_local(x_list)
    """
    
    def __init__(
        self,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, int],
        num_partitions: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ):
        self._values = values
        self._row_indices = row_indices
        self._col_indices = col_indices
        self._shape = shape
        self._num_partitions = num_partitions
        self._coords = coords
        self._partition_method = partition_method
        self._verbose = verbose
        
        # Infer device from input tensor if not explicitly specified
        if device is None:
            device = values.device
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        
        # Compute partition IDs
        # NOTE: In distributed mode, this should be computed on rank 0 and broadcast
        # to ensure consistency. See _compute_partitions_distributed() for distributed-safe version.
        self._partition_ids = self._compute_partitions(partition_method, coords)
        
        # Create all partitions
        self._partitions: List[DSparseMatrix] = []
        self._create_partitions()

        # DTensor-mirror state. Populated by :meth:`from_local` and
        # related classmethods that wrap an already-distributed local
        # matrix; ``None`` for the legacy single-process simulator
        # constructor so existing call sites keep behaving identically.
        self._local_matrix: Optional[DSparseMatrix] = None
        self._spec: Optional[DSparseSpec] = None
    
    def _compute_partitions(
        self, 
        method: str, 
        coords: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute partition assignments for each node."""
        if method == 'auto':
            if coords is not None:
                method = 'rcb'
            else:
                method = 'metis'
        
        if method == 'metis':
            return partition_graph_metis(
                self._row_indices, self._col_indices, 
                self._shape[0], self._num_partitions
            )
        elif method in ['rcb', 'slicing']:
            if coords is None:
                raise ValueError(f"Partition method '{method}' requires coords")
            return partition_coordinates(coords, self._num_partitions, method=method)
        elif method == 'simple':
            return partition_simple(self._shape[0], self._num_partitions)
        else:
            raise ValueError(f"Unknown partition method: {method}")
    
    def _create_partitions(self):
        """Create all partition matrices."""
        for i in range(self._num_partitions):
            mat = DSparseMatrix.from_global(
                self._values, self._row_indices, self._col_indices,
                self._shape, self._num_partitions, i,
                partition_ids=self._partition_ids,
                device=self._device,
                verbose=self._verbose
            )
            self._partitions.append(mat)
        
        # Store reference to all partitions for local halo exchange
        for mat in self._partitions:
            mat._all_partitions = [m.partition for m in self._partitions]
    
    # ====================================================================== #
    # DTensor-mirror API: from_local / to_local / full_tensor / redistribute.
    #
    # These methods give DSparseTensor the same shape of API as
    # ``torch.distributed.tensor.DTensor``: every call resolves through
    # a private :class:`DSparseSpec` that bundles the placement, the
    # device mesh, and the global shape. Vectors crossing the API stay
    # as ``DTensor[Shard(0)]`` so the rest of the PyTorch distributed
    # ecosystem (FSDP, TP, DCP) composes for free.
    #
    # The legacy single-process simulator entry points
    # (``DSparseTensor(values, row, col, shape, num_partitions=...)``,
    # ``from_sparse_tensor``) leave ``_local_matrix`` / ``_spec`` as
    # ``None`` -- callers that don't want the DTensor mirror keep
    # working byte-for-byte.
    # ====================================================================== #
    @classmethod
    def from_local(
        cls,
        local_matrix: "DSparseMatrix",
        mesh: Any,
        *,
        placement: SparsePlacement = None,
        global_shape: Optional[Tuple[int, int]] = None,
    ) -> "DSparseTensor":
        """Wrap an already-distributed :class:`DSparseMatrix` chunk as a
        DTensor-mirror :class:`DSparseTensor`.

        Mirrors :meth:`torch.distributed.tensor.DTensor.from_local`. Each
        rank passes the local subdomain it received from
        :meth:`DSparseTensor.from_global_distributed` /
        :meth:`from_device_mesh`; the placement / mesh metadata is
        carried in :class:`DSparseSpec` for downstream dispatch.

        Parameters
        ----------
        local_matrix : DSparseMatrix
            This rank's owned + halo subdomain.
        mesh : DeviceMesh
            The PyTorch device mesh the matrix is distributed over.
        placement : SparsePlacement, optional
            Defaults to :class:`RowPartitioned` (rows split across the
            mesh, which is what ``from_global_distributed`` produces).
            Pass :class:`Replicated` for the full-matrix-on-every-rank
            layout.
        global_shape : Tuple[int, int], optional
            Override the deduced global shape. Defaults to
            ``local_matrix.global_shape``.
        """
        if placement is None:
            placement = RowPartitioned()
        if global_shape is None:
            global_shape = tuple(local_matrix.global_shape)

        # We deliberately bypass __init__ -- the legacy constructor
        # builds an in-process Python list of every partition, which is
        # the opposite of what we want here. Build a minimal instance
        # and stamp the DTensor-mirror state on directly.
        self = cls.__new__(cls)
        self._values = None
        self._row_indices = None
        self._col_indices = None
        self._shape = global_shape
        self._num_partitions = mesh.size() if mesh is not None else 1
        self._coords = None
        self._partition_method = None
        self._verbose = False
        self._device = local_matrix.device
        self._partition_ids = None
        self._partitions = []
        self._local_matrix = local_matrix
        self._spec = DSparseSpec(placement=placement, mesh=mesh,
                                 global_shape=global_shape)
        return self

    @classmethod
    def partition(
        cls,
        A: "SparseTensor",
        mesh: Any,
        *,
        partition_method: str = "simple",
        coords: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> "DSparseTensor":
        """One-shot constructor: take a global :class:`SparseTensor` +
        :class:`DeviceMesh`, partition rows across the mesh, return a
        ready-to-use distributed tensor with :class:`RowPartitioned`
        placement.

        Equivalent to::

            local = A.partition_for_rank(rank, world_size,
                                          partition_method=partition_method,
                                          coords=coords)
            D = DSparseTensor.from_local(local, mesh,
                                          placement=RowPartitioned())

        but in one line. This is the recommended way to build a
        distributed sparse tensor from a global :class:`SparseTensor`
        for both unit tests and small-to-medium production runs (where
        every rank can afford to hold the global ``A`` briefly).

        For memory-tight scenarios where only rank 0 should ever
        materialise the global matrix, use
        :meth:`from_global_distributed` (which broadcasts only the
        partition IDs from rank 0) and chain :meth:`from_local`
        manually.

        Parameters
        ----------
        A : SparseTensor
            Global sparse matrix; every rank should hold an identical
            copy at the time of the call.
        mesh : DeviceMesh
            Target device mesh. ``mesh.size()`` becomes the world size
            and ``dist.get_rank()`` picks this rank's chunk.
        partition_method : str
            Partitioning algorithm passed through to
            :meth:`SparseTensor.partition_for_rank`: ``"simple"`` /
            ``"metis"`` / ``"rcb"`` / ``"slicing"``.
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning (RCB/slicing).
        verbose : bool
            Print partition info on each rank.
        """
        if DIST_AVAILABLE and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        world_size = mesh.size() if mesh is not None else 1

        local_matrix = A.partition_for_rank(
            rank, world_size,
            partition_method=partition_method,
            coords=coords,
            verbose=verbose,
        )
        return cls.from_local(
            local_matrix, mesh,
            placement=RowPartitioned(),
            global_shape=tuple(A.shape),
        )

    @property
    def spec(self) -> Optional[DSparseSpec]:
        """The :class:`DSparseSpec` for this tensor (placement + mesh +
        global shape), or ``None`` if this instance was built via the
        legacy single-process simulator constructor."""
        return self._spec

    def to_local(self) -> "DSparseMatrix":
        """Return this rank's local :class:`DSparseMatrix` chunk.

        Mirrors :meth:`DTensor.to_local`. For the legacy single-process
        simulator instance the call falls back to the gathered global
        matrix (logically equivalent: the "local" tensor IS the global
        tensor when there's only one rank)."""
        if self._local_matrix is not None:
            return self._local_matrix
        if not self._partitions:
            raise RuntimeError(
                "DSparseTensor has neither a DTensor-mirror local "
                "matrix nor a simulated partition list -- nothing to "
                "hand back as 'local'."
            )
        # Single-process simulator path: rank 0 owns everything.
        return self._partitions[0]

    def full_tensor(self) -> "SparseTensor":
        """Materialise the full global matrix on every rank.

        Mirrors :meth:`DTensor.full_tensor` (the symmetric ``Allgather``
        return -- every rank ends up with the same result). Returns a
        plain :class:`SparseTensor` so callers can pipe it into the
        non-distributed code path.

        Cheap when the placement is already :class:`Replicated` (just a
        format conversion); for :class:`RowPartitioned` we Allgather
        the global COO triples across the mesh.
        """
        from .sparse_tensor import SparseTensor

        if self._spec is None:
            # Single-process simulator: the original COO triple still
            # lives on the instance; convert directly.
            return self.to_sparse_tensor()

        if isinstance(self._spec.placement, Replicated):
            # Every rank already has the full matrix; rebuild a global
            # SparseTensor from the local one's COO state.
            local = self._local_matrix
            return SparseTensor(local.local_values, local.local_row,
                                local.local_col, self._spec.global_shape)

        if not (DIST_AVAILABLE and dist.is_initialized()):
            raise RuntimeError(
                "DSparseTensor.full_tensor() with RowPartitioned "
                "placement requires torch.distributed to be initialised."
            )
        return self._all_gather_global_matrix()

    def redistribute(
        self,
        placement: SparsePlacement,
        *,
        mesh: Any = None,
    ) -> "DSparseTensor":
        """Convert this :class:`DSparseTensor` to a new placement.

        Mirrors :meth:`DTensor.redistribute`. Only two transitions are
        currently meaningful for sparse matrices: ``RowPartitioned ->
        Replicated`` (Allgather every rank's COO triples and rebuild)
        and ``Replicated -> RowPartitioned`` (drop the rows this rank
        doesn't own).
        """
        if self._spec is None:
            raise RuntimeError(
                "redistribute() requires a DTensor-mirror DSparseTensor;"
                " build one via DSparseTensor.from_local(...) first."
            )
        target_mesh = mesh or self._spec.mesh

        if type(self._spec.placement) is type(placement):
            # Same placement: no-op besides optionally re-stamping mesh.
            if target_mesh is self._spec.mesh:
                return self
            return DSparseTensor.from_local(
                self._local_matrix, target_mesh,
                placement=placement,
                global_shape=self._spec.global_shape,
            )

        if isinstance(self._spec.placement, SparseShard) \
                and isinstance(placement, Replicated):
            full = self._all_gather_global_matrix()
            # ``DSparseMatrix.from_global`` is a static helper that
            # builds a single-partition local chunk -- pass ``num_part=1``
            # so the whole matrix is owned by this rank.
            full_local = DSparseMatrix.from_global(
                SparseTensor(full.val, full.row, full.col,
                             self._spec.global_shape),
                num_partitions=1,
                partition_id=0,
                device=self._local_matrix.device,
                verbose=False,
            )
            return DSparseTensor.from_local(
                full_local, target_mesh,
                placement=Replicated(),
                global_shape=self._spec.global_shape,
            )

        raise NotImplementedError(
            f"redistribute {type(self._spec.placement).__name__} -> "
            f"{type(placement).__name__} not yet supported"
        )

    def _all_gather_global_matrix(self) -> "SparseTensor":
        """Allgather the per-rank COO triples and rebuild a global
        :class:`SparseTensor`. Used by :meth:`full_tensor` and
        :meth:`redistribute`."""
        from .sparse_tensor import SparseTensor

        local = self._local_matrix
        device = local.device
        world_size = self._spec.mesh.size()

        # Local row/col indices live in the LOCAL coordinate system; we
        # need them in GLOBAL coordinates before stitching. Use the
        # partition's ``local_nodes`` map (local idx -> global idx).
        l2g = local.partition.local_nodes.to(device)
        # Only owned-row entries -- duplicates across ranks would
        # otherwise show up after the all_gather.
        owned_mask = local.local_row < local.num_owned
        rows_g = l2g[local.local_row[owned_mask]].contiguous()
        cols_g = l2g[local.local_col[owned_mask]].contiguous()
        vals   = local.local_values[owned_mask].contiguous()

        # Allgather variable-length COO triples. Same shape pattern as
        # gather_global: exchange sizes, allocate per-rank buffers,
        # all_gather each field.
        owned_nnz = torch.tensor([int(vals.numel())],
                                 dtype=torch.int64, device=device)
        sizes = [torch.zeros(1, dtype=torch.int64, device=device)
                 for _ in range(world_size)]
        dist.all_gather(sizes, owned_nnz)
        ns = [int(s.item()) for s in sizes]

        rows_list = [torch.zeros(n, dtype=torch.int64, device=device)
                     for n in ns]
        cols_list = [torch.zeros(n, dtype=torch.int64, device=device)
                     for n in ns]
        vals_list = [torch.zeros(n, dtype=vals.dtype, device=device)
                     for n in ns]
        dist.all_gather(rows_list, rows_g)
        dist.all_gather(cols_list, cols_g)
        dist.all_gather(vals_list, vals)

        return SparseTensor(
            torch.cat(vals_list),
            torch.cat(rows_list),
            torch.cat(cols_list),
            self._spec.global_shape,
        )

    @classmethod
    def from_sparse_tensor(
        cls,
        sparse_tensor: "SparseTensor",
        num_partitions: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ) -> "DSparseTensor":
        """
        Create DSparseTensor from a SparseTensor.
        
        Parameters
        ----------
        sparse_tensor : SparseTensor
            Input sparse tensor (must be 2D, not batched)
        num_partitions : int
            Number of partitions
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning
        partition_method : str
            Partitioning method
        device : str or torch.device, optional
            Target device (defaults to sparse_tensor's device)
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseTensor
            Distributed sparse tensor
        """
        # Avoid circular import
        from .sparse_tensor import SparseTensor
        
        if sparse_tensor.is_batched:
            raise ValueError("DSparseTensor does not support batched SparseTensor. "
                           "Use a 2D SparseTensor.")
        
        if device is None:
            device = sparse_tensor.device
        
        # Use sparse_shape for the matrix dimensions
        sparse_shape = sparse_tensor.sparse_shape
        
        return cls(
            sparse_tensor.values,
            sparse_tensor.row_indices,
            sparse_tensor.col_indices,
            sparse_shape,
            num_partitions=num_partitions,
            coords=coords,
            partition_method=partition_method,
            device=device,
            verbose=verbose
        )
    
    @classmethod
    def from_torch_sparse(
        cls,
        A: torch.Tensor,
        num_partitions: int,
        **kwargs
    ) -> "DSparseTensor":
        """Create DSparseTensor from PyTorch sparse tensor."""
        if A.layout == torch.sparse_csr:
            A = A.to_sparse_coo()
        
        indices = A._indices()
        values = A._values()
        
        return cls(
            values, indices[0], indices[1], tuple(A.shape),
            num_partitions=num_partitions, **kwargs
        )
    
    @classmethod
    def from_global_distributed(
        cls,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, int],
        rank: int,
        world_size: int,
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True
    ) -> "DSparseMatrix":
        """
        Create local partition in a distributed-safe manner.
        
        This method ensures that all ranks compute the same partition assignment
        by having rank 0 compute the partition IDs and broadcasting to all ranks.
        
        Parameters
        ----------
        values : torch.Tensor
            Global non-zero values [nnz]
        row_indices : torch.Tensor
            Global row indices [nnz]
        col_indices : torch.Tensor
            Global column indices [nnz]
        shape : Tuple[int, int]
            Global matrix shape (M, N)
        rank : int
            Current process rank
        world_size : int
            Total number of processes
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning [num_nodes, dim]
        partition_method : str
            Partitioning method: 'metis', 'rcb', 'slicing', 'simple'
        device : str or torch.device, optional
            Target device
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseMatrix
            Local partition matrix for this rank
            
        Example
        -------
        >>> import torch.distributed as dist
        >>> 
        >>> # In each process:
        >>> rank = dist.get_rank()
        >>> world_size = dist.get_world_size()
        >>> 
        >>> local_matrix = DSparseTensor.from_global_distributed(
        ...     val, row, col, shape, 
        ...     rank=rank, world_size=world_size
        ... )
        """
        import torch.distributed as dist
        
        if device is None:
            device = values.device
        if isinstance(device, str):
            device = torch.device(device)
        
        # Compute partition IDs on rank 0 and broadcast
        if rank == 0:
            # Create temporary DSparseTensor to compute partitions
            # Use 'simple' method if METIS might be non-deterministic
            if partition_method == 'auto':
                if coords is not None:
                    actual_method = 'rcb'
                else:
                    # Use simple partitioning by default in distributed mode
                    # to ensure determinism across ranks
                    actual_method = 'simple'
            else:
                actual_method = partition_method
            
            num_nodes = shape[0]
            if actual_method == 'simple':
                partition_ids = partition_simple(num_nodes, world_size)
            elif actual_method == 'metis':
                partition_ids = partition_graph_metis(
                    row_indices, col_indices, num_nodes, world_size
                )
            elif actual_method in ['rcb', 'slicing', 'hilbert']:
                if coords is None:
                    raise ValueError(f"Method '{actual_method}' requires coords")
                partition_ids = partition_coordinates(coords, world_size, method=actual_method)
            else:
                raise ValueError(f"Unknown method: {actual_method}")
            
            partition_ids = partition_ids.to(device)
        else:
            # Create empty tensor to receive broadcast
            partition_ids = torch.zeros(shape[0], dtype=torch.int64, device=device)
        
        # Broadcast partition IDs from rank 0 to all ranks
        dist.broadcast(partition_ids, src=0)
        
        # Now create local partition using the consistent partition IDs
        local_matrix = DSparseMatrix.from_global(
            values, row_indices, col_indices, shape,
            world_size, rank,
            partition_ids=partition_ids,
            device=device,
            verbose=verbose and rank == 0  # Only print on rank 0
        )
        
        return local_matrix
    
    @classmethod
    def from_device_mesh(
        cls,
        values: torch.Tensor,
        row_indices: torch.Tensor,
        col_indices: torch.Tensor,
        shape: Tuple[int, int],
        device_mesh: "DeviceMesh",
        coords: Optional[torch.Tensor] = None,
        partition_method: str = 'simple',
        placement: str = 'shard_rows',
        verbose: bool = False
    ) -> "DSparseMatrix":
        """
        Create local partition using PyTorch DeviceMesh.
        
        This is the recommended method for distributed training with PyTorch's
        DTensor ecosystem. Each rank receives only its local partition.
        
        Parameters
        ----------
        values : torch.Tensor
            Global non-zero values [nnz] (same on all ranks)
        row_indices : torch.Tensor
            Global row indices [nnz]
        col_indices : torch.Tensor
            Global column indices [nnz]
        shape : Tuple[int, int]
            Global matrix shape (M, N)
        device_mesh : DeviceMesh
            PyTorch DeviceMesh specifying device topology
        coords : torch.Tensor, optional
            Node coordinates for geometric partitioning
        partition_method : str
            Partitioning method: 'metis', 'rcb', 'simple'
            Default is 'simple' for determinism in distributed setting
        placement : str
            How to distribute: 'shard_rows', 'shard_cols', 'replicate'
        verbose : bool
            Whether to print partition info
            
        Returns
        -------
        DSparseMatrix
            Local partition for this rank
            
        Example
        -------
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>> from torch_sla import DSparseTensor
        >>> 
        >>> # Initialize 4-GPU device mesh
        >>> mesh = init_device_mesh("cuda", (4,), mesh_dim_names=("dp",))
        >>> 
        >>> # Create distributed sparse tensor (each rank gets its partition)
        >>> local_matrix = DSparseTensor.from_device_mesh(
        ...     val, row, col, shape,
        ...     device_mesh=mesh,
        ...     partition_method='simple'
        ... )
        >>> 
        >>> # Local operations
        >>> y_local = local_matrix.matvec(x_local)
        >>> x_local = local_matrix.solve(b_local)
        """
        try:
            from torch.distributed.device_mesh import DeviceMesh
        except ImportError:
            raise ImportError("DeviceMesh requires PyTorch 2.0+. "
                            "Use from_global_distributed() instead.")
        
        if not DIST_AVAILABLE or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized. "
                             "Call dist.init_process_group() first.")
        
        # Get rank info from device mesh
        rank = device_mesh.get_local_rank()
        world_size = device_mesh.size()
        device_type = device_mesh.device_type
        
        # Determine target device
        if device_type == "cuda":
            device = torch.device(f"cuda:{rank}")
        else:
            device = torch.device(device_type)
        
        # Use the distributed-safe factory method
        return cls.from_global_distributed(
            values, row_indices, col_indices, shape,
            rank=rank, world_size=world_size,
            coords=coords,
            partition_method=partition_method,
            device=device,
            verbose=verbose
        )
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Global matrix shape."""
        return self._shape
    
    @property
    def num_partitions(self) -> int:
        """Number of partitions."""
        return self._num_partitions
    
    @property
    def device(self) -> torch.device:
        """Device of the matrix data."""
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        """Data type of matrix values."""
        return self._values.dtype
    
    @property
    def nnz(self) -> int:
        """Total number of non-zeros."""
        return self._values.size(0)
    
    @property
    def partition_ids(self) -> torch.Tensor:
        """Partition assignment for each node."""
        return self._partition_ids
    
    @property
    def is_cuda(self) -> bool:
        """Check if matrix is on CUDA."""
        return self._device.type == 'cuda'
    
    # =========================================================================
    # Indexing and Iteration
    # =========================================================================
    
    def __len__(self) -> int:
        """Number of partitions."""
        return self._num_partitions
    
    def __getitem__(self, idx: int) -> DSparseMatrix:
        """Get a specific partition."""
        if idx < 0:
            idx = self._num_partitions + idx
        if idx < 0 or idx >= self._num_partitions:
            raise IndexError(f"Partition index {idx} out of range [0, {self._num_partitions})")
        return self._partitions[idx]
    
    def __iter__(self):
        """Iterate over partitions."""
        return iter(self._partitions)
    
    # =========================================================================
    # Device Management
    # =========================================================================
    
    def to(self, device: Union[str, torch.device]) -> "DSparseTensor":
        """
        Move all partitions to a different device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device
            
        Returns
        -------
        DSparseTensor
            New distributed tensor on target device
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        new_tensor = DSparseTensor.__new__(DSparseTensor)
        new_tensor._values = self._values.to(device)
        new_tensor._row_indices = self._row_indices.to(device)
        new_tensor._col_indices = self._col_indices.to(device)
        new_tensor._shape = self._shape
        new_tensor._num_partitions = self._num_partitions
        new_tensor._coords = self._coords
        new_tensor._partition_method = self._partition_method
        new_tensor._verbose = False  # Don't print again
        new_tensor._device = device
        new_tensor._partition_ids = self._partition_ids
        
        # Move partitions
        new_tensor._partitions = [p.to(device) for p in self._partitions]
        
        # Update references
        for mat in new_tensor._partitions:
            mat._all_partitions = [m.partition for m in new_tensor._partitions]
        
        return new_tensor
    
    def cuda(self, device: Optional[int] = None) -> "DSparseTensor":
        """Move to CUDA device."""
        if device is not None:
            return self.to(f'cuda:{device}')
        return self.to('cuda')
    
    def cpu(self) -> "DSparseTensor":
        """Move to CPU."""
        return self.to('cpu')
    
    # =========================================================================
    # Distributed Operations
    # =========================================================================
    
    def halo_exchange_local(self, x_list: List[torch.Tensor]) -> None:
        """
        Local halo exchange for single-process simulation.
        
        Exchanges halo values between all partitions locally.
        Useful for testing without actual distributed setup.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition. Each vector is
            modified in-place to update halo values.
        """
        if len(x_list) != self._num_partitions:
            raise ValueError(f"Expected {self._num_partitions} vectors, got {len(x_list)}")
        
        for part_id in range(self._num_partitions):
            partition = self._partitions[part_id].partition
            x = x_list[part_id]
            
            halo_offset = len(partition.owned_nodes)
            
            for halo_idx, global_node in enumerate(partition.halo_nodes.tolist()):
                local_halo_idx = halo_offset + halo_idx
                
                for neighbor_id in partition.neighbor_partitions:
                    neighbor_partition = self._partitions[neighbor_id].partition
                    neighbor_g2l = neighbor_partition.global_to_local
                    
                    if global_node < len(neighbor_g2l):
                        local_idx_in_neighbor = neighbor_g2l[global_node].item()
                        if local_idx_in_neighbor >= 0 and local_idx_in_neighbor < len(neighbor_partition.owned_nodes):
                            x[local_halo_idx] = x_list[neighbor_id][local_idx_in_neighbor]
                            break
    
    def matvec_all(
        self,
        x_list: List[torch.Tensor],
        exchange_halo: bool = True
    ) -> List[torch.Tensor]:
        """
        Matrix-vector multiply on all partitions.
        
        Performs y = A @ x for each partition, with optional halo exchange.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition. Each vector should have
            size = num_owned + num_halo for that partition.
        exchange_halo : bool
            Whether to perform halo exchange before multiplication.
            Default True.
            
        Returns
        -------
        List[torch.Tensor]
            List of result vectors, one per partition. Each result has
            size = num_owned (only owned nodes have valid results).
            
        Example
        -------
        >>> D = SparseTensor(val, row, col, shape).partition(4)
        >>> x_local = D.scatter_local(x_global)
        >>> y_local = D.matvec_all(x_local)
        >>> y_global = D.gather_global(y_local)
        """
        return [self._partitions[i].matvec(x_list[i], exchange_halo=exchange_halo)
                for i in range(self._num_partitions)]
    
    def solve_all(
        self,
        b_list: List[torch.Tensor],
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Solve on all partitions (subdomain solves).
        
        NOTE: This performs LOCAL subdomain solves, NOT a global distributed solve.
        Each partition solves its own local system independently.
        For a true distributed solve, use `solve_distributed()`.
        
        Parameters
        ----------
        b_list : List[torch.Tensor]
            List of local RHS vectors, one per partition
        **kwargs
            Additional arguments passed to each partition's solve method
            
        Returns
        -------
        List[torch.Tensor]
            List of solution vectors, one per partition
        """
        return [self._partitions[i].solve(b_list[i], **kwargs) 
                for i in range(self._num_partitions)]
    
    def solve_distributed(
        self,
        b_global: Union[torch.Tensor, "DTensor"],
        method: str = 'cg',
        atol: float = 1e-10,
        maxiter: int = 1000,
        verbose: bool = False
    ) -> Union[torch.Tensor, "DTensor"]:
        """
        Distributed solve: find x such that A @ x = b using all partitions.
        
        This performs a TRUE distributed solve where all partitions collaborate
        to solve the global system. Uses distributed CG with global reductions.
        
        Parameters
        ----------
        b_global : torch.Tensor or DTensor
            Global RHS vector [N].
            - If torch.Tensor: treated as global vector
            - If DTensor: automatically handles distributed input/output
        method : str
            Solver method: 'cg' (Conjugate Gradient)
        atol : float
            Absolute tolerance for convergence
        maxiter : int
            Maximum iterations
        verbose : bool
            Print convergence info
            
        Returns
        -------
        torch.Tensor or DTensor
            Global solution vector [N].
            Returns DTensor if input is DTensor, otherwise torch.Tensor.
            
        Example
        -------
        >>> D = A.partition(num_partitions=4)
        >>> x = D.solve_distributed(b)  # Distributed CG solve
        >>> residual = torch.norm(A @ x - b)
        
        >>> # With DTensor input
        >>> from torch.distributed.tensor import DTensor, Replicate
        >>> b_dt = DTensor.from_local(b_local, mesh, [Replicate()])
        >>> x_dt = D.solve_distributed(b_dt)  # Returns DTensor
        """
        # Check for DTensor input
        if _is_dtensor(b_global):
            return self._solve_distributed_dtensor(b_global, method, atol, maxiter, verbose)
        
        N = self._shape[0]
        dtype = b_global.dtype
        device = self._device
        
        # Initialize x = 0
        x_global = torch.zeros(N, dtype=dtype, device=device)
        
        # Scatter b to local
        b_local = self.scatter_local(b_global)
        
        # Distributed CG
        if method == 'cg':
            x_global = self._distributed_cg(x_global, b_global, atol, maxiter, verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Supported: 'cg'")
        
        return x_global
    
    def _solve_distributed_dtensor(
        self,
        b_dtensor: "DTensor",
        method: str,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> "DTensor":
        """
        Distributed solve with DTensor input.
        
        Handles DTensor layout conversion and result wrapping.
        
        Parameters
        ----------
        b_dtensor : DTensor
            Right-hand side as DTensor
        method : str
            Solver method
        atol : float
            Absolute tolerance
        maxiter : int
            Maximum iterations
        verbose : bool
            Print convergence info
            
        Returns
        -------
        DTensor
            Solution as DTensor with same placement as input
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        # Get DTensor metadata
        device_mesh = b_dtensor.device_mesh
        placements = b_dtensor.placements
        original_placements = tuple(placements)
        
        # Check if input is replicated
        is_replicated = all(isinstance(p, Replicate) for p in placements)
        
        if is_replicated:
            # Input is replicated - extract and solve
            b_local = b_dtensor.to_local()
            x_local = self._solve_distributed_tensor(b_local, method, atol, maxiter, verbose)
            # Wrap result as replicated DTensor
            return DTensor.from_local(x_local, device_mesh, [Replicate()])
        
        # Input is sharded - redistribute to replicated for solve
        replicate_placements = [Replicate() for _ in placements]
        b_replicated = b_dtensor.redistribute(device_mesh, replicate_placements)
        b_full = b_replicated.to_local()
        
        # Solve with full vector
        x_full = self._solve_distributed_tensor(b_full, method, atol, maxiter, verbose)
        
        # Wrap as replicated DTensor
        x_replicated = DTensor.from_local(x_full, device_mesh, [Replicate()])
        
        # Redistribute back to original placement if it was sharded
        if not is_replicated:
            output_placements = []
            for p in original_placements:
                if isinstance(p, Shard):
                    output_placements.append(Shard(p.dim))
                else:
                    output_placements.append(Replicate())
            
            return x_replicated.redistribute(device_mesh, output_placements)
        
        return x_replicated
    
    def _solve_distributed_tensor(
        self,
        b_global: torch.Tensor,
        method: str,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> torch.Tensor:
        """
        Internal solve implementation for torch.Tensor input.
        
        Separated from solve_distributed to allow DTensor wrapper to call it.
        """
        N = self._shape[0]
        dtype = b_global.dtype
        device = self._device
        
        # Initialize x = 0
        x_global = torch.zeros(N, dtype=dtype, device=device)
        
        # Scatter b to local
        b_local = self.scatter_local(b_global)
        
        # Distributed CG
        if method == 'cg':
            x_global = self._distributed_cg(x_global, b_global, atol, maxiter, verbose)
        else:
            raise ValueError(f"Unknown method: {method}. Supported: 'cg'")
        
        return x_global
    
    def _distributed_cg(
        self,
        x: torch.Tensor,
        b: torch.Tensor,
        atol: float,
        maxiter: int,
        verbose: bool
    ) -> torch.Tensor:
        """
        Distributed Conjugate Gradient.
        
        All partitions work together, with global reductions for inner products.
        """
        N = self._shape[0]
        dtype = b.dtype
        device = self._device
        
        # r = b - A @ x
        Ax = self @ x  # Uses __matmul__ which does scatter -> matvec_all -> gather
        r = b - Ax
        
        # p = r
        p = r.clone()
        
        # rs_old = r^T @ r (global)
        rs_old = torch.dot(r, r)
        
        for i in range(maxiter):
            # Ap = A @ p
            Ap = self @ p
            
            # pAp = p^T @ A @ p (global)
            pAp = torch.dot(p, Ap)
            
            if pAp.abs() < 1e-30:
                if verbose:
                    print(f"  Distributed CG: pAp too small at iter {i}")
                break
            
            # alpha = rs_old / pAp
            alpha = rs_old / pAp
            
            # x = x + alpha * p
            x = x + alpha * p
            
            # r = r - alpha * Ap
            r = r - alpha * Ap
            
            # rs_new = r^T @ r (global)
            rs_new = torch.dot(r, r)
            
            residual = rs_new.sqrt()
            
            if verbose and i % 100 == 0:
                print(f"  Distributed CG iter {i}: residual = {residual:.2e}")
            
            if residual < atol:
                if verbose:
                    print(f"  Distributed CG converged at iter {i}, residual = {residual:.2e}")
                break
            
            if rs_old.abs() < 1e-30:
                break
            
            # beta = rs_new / rs_old
            beta = rs_new / rs_old
            
            # p = r + beta * p
            p = r + beta * p
            
            rs_old = rs_new
        
        return x
    
    def gather_global(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Gather local vectors to global vector.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of local vectors, one per partition
            
        Returns
        -------
        torch.Tensor
            Global vector
        """
        x_global = torch.zeros(self._shape[0], dtype=x_list[0].dtype, device=self._device)
        
        for i in range(self._num_partitions):
            partition = self._partitions[i].partition
            owned_nodes = partition.owned_nodes
            num_owned = len(owned_nodes)
            x_global[owned_nodes] = x_list[i][:num_owned].to(self._device)
        
        return x_global
    
    def scatter_local(self, x_global: torch.Tensor) -> List[torch.Tensor]:
        """
        Scatter global vector to local vectors.
        
        Parameters
        ----------
        x_global : torch.Tensor
            Global vector
            
        Returns
        -------
        List[torch.Tensor]
            List of local vectors (with halo values filled)
        """
        x_list = []
        
        for i in range(self._num_partitions):
            partition = self._partitions[i].partition
            local_nodes = partition.local_nodes
            x_local = x_global[local_nodes].to(self._partitions[i].device)
            x_list.append(x_local)
        
        return x_list
    
    def to_sparse_tensor(self) -> "SparseTensor":
        """
        Gather all partitions into a single SparseTensor.
        
        This creates a global SparseTensor from the distributed data.
        Useful for verification, debugging, or when you need to perform
        operations that require the full matrix.
        
        Returns
        -------
        SparseTensor
            Global sparse tensor containing all data
            
        Example
        -------
        >>> D = DSparseTensor(val, row, col, shape, num_partitions=4)
        >>> A = D.to_sparse_tensor()  # Gather to global SparseTensor
        >>> x = A.solve(b)  # Solve on the full matrix
        """
        from .sparse_tensor import SparseTensor
        
        # Return the original global data as SparseTensor
        return SparseTensor(
            self._values.to(self._device),
            self._row_indices.to(self._device),
            self._col_indices.to(self._device),
            self._shape
        )
    
    # Alias for convenience
    gather = to_sparse_tensor
    
    def to_list(self) -> "DSparseTensorList":
        """
        Split into DSparseTensorList based on connected components.
        
        If the matrix has isolated subgraphs (block-diagonal structure),
        splits it into separate distributed matrices, one per component.
        
        Returns
        -------
        DSparseTensorList
            List of distributed matrices, one per connected component.
            
        Notes
        -----
        This is useful when you have a block-diagonal matrix representing
        multiple independent graphs and want to process them separately.
        
        Examples
        --------
        >>> D = DSparseTensor(val, row, col, shape, num_partitions=4)
        >>> if D.has_isolated_components():
        ...     dstl = D.to_list()  # Split into components
        """
        # Get connected components from global data
        sparse = self.to_sparse_tensor()
        sparse_list = sparse.to_connected_components()
        
        # Partition each component
        return DSparseTensorList.from_sparse_tensor_list(
            sparse_list,
            num_partitions=self._num_partitions,
            threshold=1000,  # Default threshold
            device=self._device,
            verbose=False
        )
    
    def has_isolated_components(self) -> bool:
        """
        Check if the matrix has multiple connected components.
        
        Returns
        -------
        bool
            True if matrix has more than one connected component.
        """
        sparse = self.to_sparse_tensor()
        return sparse.has_isolated_components()
    
    @classmethod
    def from_list(
        cls,
        dstl: "DSparseTensorList",
        verbose: bool = False
    ) -> "DSparseTensor":
        """
        Merge DSparseTensorList into a single block-diagonal DSparseTensor.
        
        Parameters
        ----------
        dstl : DSparseTensorList
            List of distributed matrices to merge.
        verbose : bool
            Print info.
            
        Returns
        -------
        DSparseTensor
            Block-diagonal distributed matrix.
            
        Examples
        --------
        >>> dstl = DSparseTensorList.from_sparse_tensor_list(stl, 4)
        >>> D = DSparseTensor.from_list(dstl)  # Merge to block-diagonal
        """
        return dstl.to_block_diagonal()
    
    # =========================================================================
    # DTensor Utilities
    # =========================================================================
    
    def scatter_to_dtensor(
        self,
        x_global: torch.Tensor,
        device_mesh: "DeviceMesh",
        shard_dim: int = 0
    ) -> "DTensor":
        """
        Convert a global tensor to a sharded DTensor aligned with matrix partitioning.
        
        This creates a DTensor where each rank holds the portion of the vector
        corresponding to its owned nodes in the matrix partitioning.
        
        Parameters
        ----------
        x_global : torch.Tensor
            Global vector of shape [N]
        device_mesh : DeviceMesh
            PyTorch DeviceMesh for distribution
        shard_dim : int
            Dimension to shard (default 0 for vectors)
            
        Returns
        -------
        DTensor
            Sharded DTensor with local data for this rank
            
        Example
        -------
        >>> mesh = init_device_mesh("cuda", (4,))
        >>> x_global = torch.randn(N)
        >>> x_dt = D.scatter_to_dtensor(x_global, mesh)
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        # Create sharded DTensor
        # Each rank gets the portion corresponding to its partition
        placements = [Shard(shard_dim)]
        return DTensor.from_local(
            x_global,  # Will be redistributed by DTensor
            device_mesh,
            placements
        )
    
    def gather_from_dtensor(
        self,
        x_dtensor: "DTensor"
    ) -> torch.Tensor:
        """
        Convert a DTensor to a global tensor.
        
        Parameters
        ----------
        x_dtensor : DTensor
            Distributed tensor
            
        Returns
        -------
        torch.Tensor
            Full global tensor
            
        Example
        -------
        >>> x_global = D.gather_from_dtensor(x_dt)
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        return x_dtensor.full_tensor()
    
    def to_dtensor(
        self,
        x: torch.Tensor,
        device_mesh: "DeviceMesh",
        replicate: bool = True
    ) -> "DTensor":
        """
        Convert a tensor to DTensor with specified placement.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        device_mesh : DeviceMesh
            PyTorch DeviceMesh
        replicate : bool
            If True, create a replicated DTensor (same data on all ranks).
            If False, create a sharded DTensor (data is split).
            
        Returns
        -------
        DTensor
            Resulting DTensor
            
        Example
        -------
        >>> mesh = init_device_mesh("cuda", (4,))
        >>> x_dt = D.to_dtensor(x, mesh, replicate=True)
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        if replicate:
            placements = [Replicate()]
        else:
            placements = [Shard(0)]
        
        return DTensor.from_local(x, device_mesh, placements)
    
    @property
    def supports_dtensor(self) -> bool:
        """Check if DTensor operations are available."""
        return DTENSOR_AVAILABLE
    
    # =========================================================================
    # Distributed Algorithms (True Distributed, No Gather)
    # =========================================================================
    
    def _global_matvec_with_grad(self, x: torch.Tensor) -> torch.Tensor:
        """
        Global matrix-vector multiplication that preserves gradients.
        
        Uses the original COO data to maintain gradient flow.
        For true distributed MPI execution, use _distributed_matvec instead.
        
        This method is used for gradient-enabled operations like eigsh, solve.
        """
        # Use original global COO data for gradient support
        # y[i] = sum_j A[i,j] * x[j]
        # y = scatter_add(values * x[col], row)
        y = torch.zeros(self._shape[0], dtype=x.dtype, device=x.device)
        vals = self._values.to(x.device)
        rows = self._row_indices.to(x.device)
        cols = self._col_indices.to(x.device)
        
        # y[row] += values * x[col]
        contributions = vals * x[cols]
        y = y.scatter_add(0, rows, contributions)
        return y
    
    def _distributed_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """
        Distributed matrix-vector multiplication with gradient support.
        
        For single-node simulation with gradient support, uses _global_matvec_with_grad.
        For true distributed MPI execution, uses scatter -> local matvec -> gather.
        """
        # Check if we need gradients
        if self._values.requires_grad or x.requires_grad:
            # Use global matvec that preserves gradients
            return self._global_matvec_with_grad(x)
        
        # Otherwise use true distributed pattern
        x_local = self.scatter_local(x)
        y_local = self.matvec_all(x_local)
        return self.gather_global(y_local)
    
    def _distributed_lobpcg(
        self,
        k: int,
        largest: bool = True,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Distributed LOBPCG eigenvalue solver.
        
        Uses distributed matvec with global QR and Rayleigh-Ritz.
        No data gather required - only needs global reductions.
        """
        N = self._shape[0]
        dtype = self._values.dtype
        device = self._device
        
        # Initialize random subspace (global vectors)
        m = min(2 * k, N)
        X = torch.randn(N, m, dtype=dtype, device=device)
        
        # Global QR decomposition
        X, _ = torch.linalg.qr(X)
        
        eigenvalues_prev = None
        
        for iteration in range(maxiter):
            # Distributed matvec: AX = D @ X (column by column or batched)
            AX = torch.zeros_like(X)
            for j in range(X.shape[1]):
                AX[:, j] = self._distributed_matvec(X[:, j])
            
            # Rayleigh-Ritz: project onto subspace
            # H = X^T @ AX (global reduction)
            H = X.T @ AX
            
            # Solve small eigenvalue problem
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            
            # Sort eigenvalues
            if largest:
                idx = eigenvalues.argsort(descending=True)
            else:
                idx = eigenvalues.argsort()
            
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Update X = X @ V
            X = X @ eigenvectors
            
            # Check convergence
            if eigenvalues_prev is not None:
                diff = (eigenvalues[:k] - eigenvalues_prev[:k]).abs()
                if (diff < tol * eigenvalues[:k].abs().clamp(min=1e-10)).all():
                    break
            eigenvalues_prev = eigenvalues.clone()
            
            # Expand subspace with residual
            if iteration < maxiter - 1:
                # Compute residual: R = AX - X @ diag(eigenvalues)
                AX_new = torch.zeros_like(X)
                for j in range(X.shape[1]):
                    AX_new[:, j] = self._distributed_matvec(X[:, j])
                
                residual = AX_new - X * eigenvalues.unsqueeze(0)
                
                # Orthogonalize and expand
                combined = torch.cat([X[:, :k], residual[:, :k]], dim=1)
                X, _ = torch.linalg.qr(combined)
                
                # Pad if needed
                if X.size(1) < m:
                    extra = torch.randn(N, m - X.size(1), dtype=dtype, device=device)
                    X = torch.cat([X, extra], dim=1)
                    X, _ = torch.linalg.qr(X)
        
        return eigenvalues[:k], X[:, :k]
    
    def eigsh(
        self,
        k: int = 6,
        which: str = "LM",
        sigma: Optional[float] = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute k eigenvalues for symmetric matrices using distributed LOBPCG.
        
        This is a TRUE distributed algorithm - no data gather required.
        Uses distributed matvec with global QR decomposition.
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default: 6.
        which : {"LM", "SM", "LA", "SA"}, optional
            Which eigenvalues to find:
            - "LM"/"LA": Largest (default)
            - "SM"/"SA": Smallest
        sigma : float, optional
            Find eigenvalues near sigma (not yet supported).
        return_eigenvectors : bool, optional
            Whether to return eigenvectors. Default: True.
        maxiter : int, optional
            Maximum LOBPCG iterations. Default: 1000.
        tol : float, optional
            Convergence tolerance. Default: 1e-8.
            
        Returns
        -------
        eigenvalues : torch.Tensor
            Shape [k].
        eigenvectors : torch.Tensor or None
            Shape [N, k] if return_eigenvectors is True.
        
        Notes
        -----
        **Distributed Algorithm:**
        
        - Uses distributed LOBPCG (Locally Optimal Block PCG)
        - Only requires distributed matvec + global reductions
        - Memory: O(N * k) per node for eigenvectors
        - Communication: O(k^2) per iteration for Rayleigh-Ritz
        
        **Gradient Support:**
        
        - Gradients flow through the distributed matvec operations
        - O(iterations) graph nodes (not O(1) like adjoint)
        """
        if sigma is not None:
            warnings.warn("sigma (shift-invert) not yet supported for distributed eigsh. Ignoring.")
        
        largest = which in ('LM', 'LA')
        eigenvalues, eigenvectors = self._distributed_lobpcg(k, largest=largest, maxiter=maxiter, tol=tol)
        
        if return_eigenvectors:
            return eigenvalues, eigenvectors
        return eigenvalues, None
    
    def eigs(
        self,
        k: int = 6,
        which: str = "LM",
        sigma: Optional[float] = None,
        return_eigenvectors: bool = True,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute k eigenvalues using distributed LOBPCG.
        
        For symmetric matrices, equivalent to eigsh().
        For non-symmetric, currently falls back to eigsh() (symmetric assumption).
        
        Parameters
        ----------
        k : int, optional
            Number of eigenvalues to compute. Default: 6.
        which : str, optional
            Which eigenvalues to find.
        sigma : float, optional
            Find eigenvalues near sigma.
        return_eigenvectors : bool, optional
            Whether to return eigenvectors. Default: True.
        maxiter : int, optional
            Maximum iterations. Default: 1000.
        tol : float, optional
            Convergence tolerance. Default: 1e-8.
            
        Returns
        -------
        eigenvalues : torch.Tensor
            Shape [k].
        eigenvectors : torch.Tensor or None
            Shape [N, k] if return_eigenvectors is True.
        """
        # For now, use eigsh (assumes symmetric)
        # TODO: Implement Arnoldi for non-symmetric
        return self.eigsh(k=k, which=which, sigma=sigma, 
                         return_eigenvectors=return_eigenvectors,
                         maxiter=maxiter, tol=tol)
    
    def svd(
        self, 
        k: int = 6,
        maxiter: int = 1000,
        tol: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute truncated SVD using distributed power iteration.
        
        Uses A^T @ A for eigenvalues, then recovers U from A @ V.
        
        Parameters
        ----------
        k : int, optional
            Number of singular values to compute. Default: 6.
        maxiter : int, optional
            Maximum iterations. Default: 1000.
        tol : float, optional
            Convergence tolerance. Default: 1e-8.
            
        Returns
        -------
        U : torch.Tensor
            Left singular vectors. Shape [M, k].
        S : torch.Tensor
            Singular values. Shape [k].
        Vt : torch.Tensor
            Right singular vectors. Shape [k, N].
        
        Notes
        -----
        **Distributed Algorithm:**
        
        - Computes eigenvalues of A^T @ A using distributed LOBPCG
        - No data gather required
        """
        M, N = self._shape
        dtype = self._values.dtype
        device = self._device
        
        # For SVD, we need A^T @ A which requires transpose
        # Create A^T as a DSparseTensor
        A_T = self.T()
        
        # Power iteration for A^T @ A
        # Initialize random vectors
        V = torch.randn(N, k, dtype=dtype, device=device)
        V, _ = torch.linalg.qr(V)
        
        for iteration in range(maxiter):
            # AV = A @ V
            AV = torch.zeros(M, k, dtype=dtype, device=device)
            for j in range(k):
                AV[:, j] = self._distributed_matvec(V[:, j])
            
            # AtAV = A^T @ (A @ V)
            AtAV = torch.zeros(N, k, dtype=dtype, device=device)
            for j in range(k):
                AtAV[:, j] = A_T._distributed_matvec(AV[:, j])
            
            # QR decomposition
            V_new, R = torch.linalg.qr(AtAV)
            
            # Check convergence
            diff = (V_new - V).norm()
            V = V_new
            
            if diff < tol:
                break
        
        # Compute singular values and U
        # AV = A @ V, then normalize to get U
        AV = torch.zeros(M, k, dtype=dtype, device=device)
        for j in range(k):
            AV[:, j] = self._distributed_matvec(V[:, j])
        
        # S = ||AV[:, j]||
        S = AV.norm(dim=0)
        
        # U = AV / S
        U = AV / S.unsqueeze(0).clamp(min=1e-10)
        
        return U, S, V.T
    
    def norm(self, ord: Literal['fro', 1, 2] = 'fro') -> torch.Tensor:
        """
        Compute matrix norm (distributed).
        
        For Frobenius norm, computed locally and aggregated.
        For spectral norm, uses distributed SVD.
        
        Parameters
        ----------
        ord : {'fro', 1, 2}
            Type of norm:
            - 'fro': Frobenius norm (distributed sum)
            - 1: Maximum column sum
            - 2: Spectral norm (largest singular value via distributed SVD)
            
        Returns
        -------
        torch.Tensor
            Scalar tensor containing the norm value.
        """
        if ord == 'fro':
            # Frobenius norm: sqrt(sum(values^2))
            # This is truly distributed - each partition has its own values
            return torch.sqrt((self._values ** 2).sum())
        elif ord == 2:
            # Spectral norm: largest singular value
            _, S, _ = self.svd(k=1, maxiter=100)
            return S[0]
        elif ord == 1:
            # Maximum column sum - need to gather
            warnings.warn("1-norm requires data gather. Using to_sparse_tensor().")
            return self.to_sparse_tensor().norm(ord=1)
        else:
            raise ValueError(f"Unknown norm order: {ord}")
    
    def condition_number(self, ord: int = 2) -> torch.Tensor:
        """
        Estimate condition number using distributed SVD.
        
        Parameters
        ----------
        ord : int, optional
            Norm order. Default: 2 (spectral).
            
        Returns
        -------
        torch.Tensor
            Condition number estimate (σ_max / σ_min).
        """
        if ord == 2:
            # Need largest and smallest singular values
            # Compute k=6 singular values
            _, S, _ = self.svd(k=6, maxiter=200)
            return S[0] / S[-1].clamp(min=1e-10)
        else:
            warnings.warn(f"ord={ord} requires data gather. Using to_sparse_tensor().")
            return self.to_sparse_tensor().condition_number(ord=ord)
    
    def det(self) -> torch.Tensor:
        """
        Compute determinant of the distributed sparse matrix.
        
        WARNING: This operation requires gathering the full matrix to compute
        the determinant, as determinant is a global property that cannot be
        computed in a truly distributed manner without full matrix information.
        
        The determinant is computed by:
        1. Gathering all partitions into a global SparseTensor
        2. Computing the determinant using LU decomposition (CPU) or 
           torch.linalg.det (CUDA)
        
        Returns
        -------
        torch.Tensor
            Determinant value (scalar tensor).
            
        Raises
        ------
        ValueError
            If matrix is not square
            
        Notes
        -----
        - Only square matrices have determinants
        - This method gathers all data, so use with caution for large matrices
        - Supports gradient computation via autograd
        - For very large matrices, consider using log-determinant or other
          approximations instead
        
        Examples
        --------
        >>> import torch
        >>> from torch_sla import DSparseTensor
        >>> 
        >>> # Create distributed sparse matrix
        >>> val = torch.tensor([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0])
        >>> row = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        >>> col = torch.tensor([0, 1, 0, 1, 2, 1, 2])
        >>> D = DSparseTensor(val, row, col, (3, 3), num_partitions=2)
        >>> 
        >>> # Compute determinant (gathers to single node)
        >>> det = D.det()
        >>> print(det)
        >>>
        >>> # With gradient support
        >>> val = val.requires_grad_(True)
        >>> D = DSparseTensor(val, row, col, (3, 3), num_partitions=2)
        >>> det = D.det()
        >>> det.backward()
        >>> print(val.grad)  # Gradient w.r.t. matrix values
        """
        M, N = self._shape
        
        if M != N:
            raise ValueError(f"Matrix must be square for determinant, got shape ({M}, {N})")
        
        # Warn user about data gather
        warnings.warn(
            "det() requires gathering all partitions to compute the determinant. "
            "This is a global operation that cannot be computed in a truly distributed manner. "
            "For large matrices, this may be memory-intensive."
        )
        
        # Gather to global SparseTensor and compute determinant
        A_global = self.to_sparse_tensor()
        return A_global.det()
    
    def T(self) -> "DSparseTensor":
        """
        Transpose the distributed sparse tensor.
        
        Returns a new DSparseTensor with swapped row/column indices.
        
        Returns
        -------
        DSparseTensor
            Transposed matrix.
        """
        # Swap row and column indices
        return DSparseTensor(
            self._values,
            self._col_indices,  # swap
            self._row_indices,  # swap
            (self._shape[1], self._shape[0]),
            num_partitions=self._num_partitions,
            coords=self._coords,
            partition_method=self._partition_method,
            device=self._device,
            verbose=False
        )
    
    # =========================================================================
    # Methods that require data gather (with warnings)
    # =========================================================================
    
    def to_dense(self) -> torch.Tensor:
        """
        Convert to dense tensor.
        
        WARNING: This gathers all data to a single node.
        Only use for small matrices or debugging.
        
        Returns
        -------
        torch.Tensor
            Dense matrix of shape (M, N).
        """
        warnings.warn("to_dense() gathers all data to a single node. "
                     "Only use for debugging or small matrices.")
        return self.to_sparse_tensor().to_dense()
    
    def is_symmetric(self, atol: float = 1e-8, rtol: float = 1e-5) -> torch.Tensor:
        """
        Check if matrix is symmetric.
        
        Can be done distributedly by comparing values with transpose.
        
        Parameters
        ----------
        atol : float
            Absolute tolerance for symmetry check.
        rtol : float
            Relative tolerance for symmetry check.
            
        Returns
        -------
        torch.Tensor
            Boolean scalar tensor.
        """
        # This can be done without gather by checking local values
        # For now, use simple implementation
        return self.to_sparse_tensor().is_symmetric(atol=atol, rtol=rtol)
    
    def is_positive_definite(self) -> torch.Tensor:
        """
        Check if matrix is positive definite.
        
        Uses distributed eigenvalue computation.
        
        Returns
        -------
        torch.Tensor
            Boolean scalar tensor.
        """
        # Check smallest eigenvalue > 0
        eigenvalues, _ = self.eigsh(k=1, which='SA', return_eigenvectors=False, maxiter=200)
        return eigenvalues[0] > 0
    
    def lu(self):
        """
        Compute LU decomposition.
        
        WARNING: LU is inherently not distributed-friendly.
        This gathers data to a single node.
        
        For distributed solves, use solve_distributed() with iterative methods.
        
        Returns
        -------
        LUFactorization
            Factorization object with solve() method.
        """
        warnings.warn("LU decomposition is not distributed. "
                     "Use solve_distributed() for distributed solves.")
        return self.to_sparse_tensor().lu()
    
    def spy(self, **kwargs):
        """
        Visualize sparsity pattern.
        
        Gathers data for visualization.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to SparseTensor.spy().
        """
        return self.to_sparse_tensor().spy(**kwargs)
    
    def nonlinear_solve(
        self,
        residual_fn,
        u0: torch.Tensor,
        *params,
        method: str = 'newton',
        tol: float = 1e-6,
        atol: float = 1e-10,
        max_iter: int = 50,
        line_search: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Solve nonlinear equation F(u, D, *params) = 0 using distributed Newton-Krylov.
        
        Uses Jacobian-free Newton-Krylov with distributed CG for linear solves.
        
        Parameters
        ----------
        residual_fn : callable
            Function F(u, D, *params) -> residual tensor.
            D is this DSparseTensor.
        u0 : torch.Tensor
            Initial guess (global vector).
        *params : torch.Tensor
            Additional parameters.
        method : str
            'newton': Newton-Krylov with distributed CG
            'picard': Fixed-point iteration
        tol : float
            Relative tolerance.
        atol : float
            Absolute tolerance.
        max_iter : int
            Maximum outer iterations.
        line_search : bool
            Use Armijo line search.
        verbose : bool
            Print convergence info.
            
        Returns
        -------
        torch.Tensor
            Solution u such that F(u, D, *params) ≈ 0.
        
        Notes
        -----
        **Distributed Algorithm:**
        
        - Uses Jacobian-free Newton-Krylov (JFNK)
        - Linear solves use distributed CG
        - Jacobian-vector products computed via finite differences
        """
        u = u0.clone()
        N = u.shape[0]
        dtype = u.dtype
        device = u.device
        
        for outer_iter in range(max_iter):
            # Compute residual
            F = residual_fn(u, self, *params)
            F_norm = F.norm()
            
            if verbose:
                print(f"  Newton iter {outer_iter}: ||F|| = {F_norm:.2e}")
            
            if F_norm < atol:
                if verbose:
                    print(f"  Converged (atol) at iteration {outer_iter}")
                break
            
            if outer_iter > 0 and F_norm < tol * F_norm_init:
                if verbose:
                    print(f"  Converged (rtol) at iteration {outer_iter}")
                break
            
            if outer_iter == 0:
                F_norm_init = F_norm
            
            if method == 'picard':
                # Simple fixed-point: u = u - F (assuming F = Au - b form)
                u = u - F
            else:
                # Newton-Krylov: solve J @ du = -F using CG with Jacobian-vector products
                # J @ v ≈ (F(u + eps*v) - F(u)) / eps
                eps = 1e-7 * max(u.norm(), 1.0)
                
                def matvec(v):
                    """Jacobian-vector product via finite differences."""
                    F_plus = residual_fn(u + eps * v, self, *params)
                    return (F_plus - F) / eps
                
                # Distributed CG for J @ du = -F
                du = torch.zeros_like(u)
                r = -F - matvec(du)  # r = -F - J @ 0 = -F
                p = r.clone()
                rs_old = torch.dot(r, r)
                
                for cg_iter in range(min(100, N)):
                    Jp = matvec(p)
                    pJp = torch.dot(p, Jp)
                    
                    if pJp.abs() < 1e-30:
                        break
                    
                    alpha = rs_old / pJp
                    du = du + alpha * p
                    r = r - alpha * Jp
                    rs_new = torch.dot(r, r)
                    
                    if rs_new.sqrt() < 1e-10:
                        break
                    
                    beta = rs_new / rs_old
                    p = r + beta * p
                    rs_old = rs_new
                
                # Line search
                if line_search:
                    alpha = 1.0
                    F_new_norm = residual_fn(u + alpha * du, self, *params).norm()
                    while F_new_norm > F_norm and alpha > 1e-8:
                        alpha *= 0.5
                        F_new_norm = residual_fn(u + alpha * du, self, *params).norm()
                    u = u + alpha * du
                else:
                    u = u + du
        
        return u
    
    # =========================================================================
    # Matrix Operations
    # =========================================================================
    
    def __matmul__(self, x: Union[torch.Tensor, "DTensor"]) -> Union[torch.Tensor, "DTensor"]:
        """
        Distributed matrix-vector multiplication: y = D @ x
        
        Automatically handles scatter, distributed matvec, and gather.
        Supports gradient computation when values have requires_grad=True.
        
        Parameters
        ----------
        x : torch.Tensor or DTensor
            Global vector of shape (N,) where N = shape[1].
            - If torch.Tensor: treated as global vector (same on all ranks or single-node)
            - If DTensor: automatically handles distributed input/output
            
        Returns
        -------
        torch.Tensor or DTensor
            Global result vector of shape (M,) where M = shape[0].
            Returns DTensor if input is DTensor, otherwise torch.Tensor.
            
        Example
        -------
        >>> D = A.partition(num_partitions=4)
        >>> y = D @ x  # Equivalent to A @ x
        
        >>> # With DTensor input
        >>> from torch.distributed.tensor import DTensor, Replicate
        >>> x_dt = DTensor.from_local(x_local, mesh, [Replicate()])
        >>> y_dt = D @ x_dt  # Returns DTensor
        
        Notes
        -----
        **Gradient Support:**
        
        For single-node simulation with gradient support, uses global COO matvec.
        For true MPI distributed execution without gradients, uses partition-based matvec.
        
        **DTensor Support:**
        
        When input is a DTensor:
        - Replicated DTensor: extracts local tensor and computes as global
        - Sharded DTensor: redistributes to Replicate, computes, then reshards
        """
        # DTensor-mirror fast path: when this DSparseTensor carries a
        # real distributed spec (built via ``from_local`` on a multi-
        # rank job), the matvec stays entirely in the ``Shard(0)``
        # space -- halo exchange in, local matvec, wrap result as
        # DTensor with the same mesh + Shard(0) placement. No gather.
        if self._spec is not None and isinstance(self._spec.placement,
                                                  SparseShard):
            return self._matmul_spec(x)

        # Check for DTensor input (legacy "replicate-and-densify" path
        # for the single-process simulator).
        if _is_dtensor(x):
            return self._matmul_dtensor(x)

        return self._distributed_matvec(x)

    def _matmul_spec(self, x: Any) -> Any:
        """DTensor-mirror matvec: ``y_Shard0 = D_RowPart @ x_Shard0``.

        Input contract (DTensor's ``Shard(0)`` style):

        * ``x`` is a ``DTensor`` -- its ``num_owned``-sized local shard
          is extracted via ``to_local()``.
        * Or a raw ``torch.Tensor`` sized either ``num_owned`` (clean
          Shard(0) input; halo is internally zero-padded then refilled
          by ``halo_exchange``) or ``num_local`` (legacy callers that
          pre-fill halo themselves; supported for backward compat).

        Output:

        * If the input was a ``DTensor``, returns ``DTensor.from_local(
          y_owned, mesh, [Shard(0)])`` -- pure Shard(0) → Shard(0).
        * Otherwise the raw ``y_owned`` tensor.
        """
        local_matrix = self._local_matrix
        assert local_matrix is not None, "spec-mode requires _local_matrix"

        if _is_dtensor(x):
            x_local = x.to_local()
        else:
            x_local = x

        # Pad x_owned → x_local (num_local) before matvec; halo entries
        # will be overwritten in-place by halo_exchange.
        x_padded = self._pad_owned_to_local(x_local)
        y_full = local_matrix.matvec(x_padded, exchange_halo=True)
        y_owned = y_full[:local_matrix.num_owned]

        if _is_dtensor(x):
            from torch.distributed.tensor import DTensor as _DTensor, Shard
            return _DTensor.from_local(
                y_owned, self._spec.mesh, [Shard(0)])
        return y_owned

    def _pad_owned_to_local(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a Shard(0)-sized ``[num_owned]`` tensor into a full
        ``[num_local]`` tensor with halo entries zero-padded (they get
        overwritten by the next ``halo_exchange``). Pass-through when
        ``x`` is already num_local-sized."""
        local_matrix = self._local_matrix
        nl = local_matrix.num_local
        no = local_matrix.num_owned
        if x.shape[0] == nl:
            return x
        if x.shape[0] == no:
            padded = torch.zeros(nl, dtype=x.dtype, device=x.device)
            padded[:no] = x
            return padded
        raise ValueError(
            f"x has shape[0]={x.shape[0]}, expected num_owned={no} or "
            f"num_local={nl}."
        )

    # ====================================================================== #
    # Shard(0)-space distributed CG.
    #
    # The classic ``_distributed_cg`` (single-process simulator path)
    # operates on global N-sized vectors. That's only correct in the
    # legacy single-process simulator -- on multiple ranks every CG
    # iteration would Allgather the whole vector. The Shard(0)
    # implementation below keeps every vector local (size ``num_owned``)
    # and uses ``dist.all_reduce`` for the inner products that CG
    # needs. Matvec routes through ``_pad_owned_to_local`` so halo
    # entries are filled by ``halo_exchange`` per iteration.
    # ====================================================================== #
    def solve_distributed_shard(
        self,
        b: Any,
        *,
        method: Any = None,
        preconditioner: Any = None,
        atol: Any = None,
        rtol: Any = None,
        maxiter: Any = None,
        restart: Any = None,
        verbose: Any = None,
    ) -> Any:
        """Distributed Krylov solve entirely in Shard(0) space.

        Requires this :class:`DSparseTensor` to carry a real spec
        (built via :meth:`from_local`). The right-hand side ``b`` may
        be a ``DTensor[Shard(0)]`` (most common) or a raw
        ``torch.Tensor`` sized ``num_owned`` for the calling rank;
        the return value mirrors the input's wrapper.

        Methods (all live in Shard(0) space):

        * ``"cg"``       Saad §6.7 conjugate gradient -- SPD systems
        * ``"bicgstab"`` Saad §7.4 BiCGStab -- non-symmetric, no restart
        * ``"gmres"``    Saad §6.5 restarted GMRES(m) -- general
        * ``"fgmres"``   Saad §9.4 flexible GMRES(m) -- variable preconditioner
        * ``"minres"``   Paige-Saunders MINRES -- symmetric indefinite

        Inner products go through ``dist.all_reduce`` (sum), matvec
        through ``halo_exchange`` -- no rank ever sees a global vector.

        SolverConfig integration
        ------------------------
        Every kwarg defaults to ``None``, meaning "look at the active
        :class:`SolverConfig` scope on this thread, then fall back to
        the hard-coded default". The precedence chain matches
        :func:`solve` -- explicit kwarg → innermost scope → outer
        scopes (LIFO) → hard-coded default.

        >>> with SolverConfig(method="bicgstab", atol=1e-8):
        ...     x = D.solve_distributed_shard(b)        # picks BiCGStab + 1e-8
        ...     y = D.solve_distributed_shard(b, atol=1e-12)  # kwarg wins
        """
        if self._spec is None or not isinstance(
                self._spec.placement, SparseShard):
            raise RuntimeError(
                "solve_distributed_shard() requires a DSparseTensor "
                "with SparseShard placement -- build one via "
                "DSparseTensor.from_local(local, mesh, ...) or "
                "DSparseTensor.partition(A, mesh, ...)."
            )
        if not (DIST_AVAILABLE and dist.is_initialized()):
            raise RuntimeError(
                "solve_distributed_shard() requires torch.distributed "
                "to be initialised."
            )

        # Merge with active SolverConfig scope. Explicit kwargs (non-
        # ``None``) win; otherwise we walk the scope stack via
        # ``solve._active_defaults`` and fall back to hard-coded.
        from .solve import _active_defaults
        defaults = _active_defaults()
        def _pick(value, name, hardcoded):
            if value is not None:
                return value
            if name in defaults:
                return defaults[name]
            return hardcoded
        method  = _pick(method,  "method",  "cg")
        atol    = _pick(atol,    "atol",    1e-10)
        rtol    = _pick(rtol,    "rtol",    0.0)
        maxiter = _pick(maxiter, "maxiter", 1000)
        restart = restart if restart is not None else 30  # not in SolverConfig
        verbose = _pick(verbose, "verbose", False)
        # ``preconditioner`` is special-cased in SolverConfig because
        # ``None`` is a legitimate "no preconditioning" choice (the
        # _UNSET sentinel distinguishes that from "not set"). Mirror
        # that here: explicit ``None`` means identity precond.
        if preconditioner is None and "preconditioner" in defaults:
            preconditioner = defaults["preconditioner"]
        M_apply = self._make_preconditioner(preconditioner)

        if _is_dtensor(b):
            b_owned = b.to_local()
            wrap_output = True
        else:
            b_owned = b
            wrap_output = False

        method_l = method.lower()
        if method_l in ("cg", "pcg"):
            x_owned = self._distributed_cg_shard(
                b_owned, M_apply=M_apply, atol=atol, rtol=rtol,
                maxiter=maxiter, verbose=verbose)
        elif method_l == "bicgstab":
            x_owned = self._distributed_bicgstab_shard(
                b_owned, M_apply=M_apply, atol=atol, rtol=rtol,
                maxiter=maxiter, verbose=verbose)
        elif method_l in ("gmres", "fgmres"):
            x_owned = self._distributed_gmres_shard(
                b_owned, M_apply=M_apply, atol=atol, rtol=rtol,
                maxiter=maxiter, restart=restart,
                flexible=(method_l == "fgmres"), verbose=verbose)
        elif method_l == "minres":
            x_owned = self._distributed_minres_shard(
                b_owned, M_apply=M_apply, atol=atol, rtol=rtol,
                maxiter=maxiter, verbose=verbose)
        else:
            raise ValueError(
                f"Unknown distributed solve method {method!r}; expected "
                "one of cg, bicgstab, gmres, fgmres, minres."
            )

        if wrap_output:
            from torch.distributed.tensor import DTensor as _DTensor, Shard
            return _DTensor.from_local(
                x_owned, self._spec.mesh, [Shard(0)])
        return x_owned

    # ------------------------------------------------------------------ #
    # Shard(0)-space primitives reused by every Krylov method.
    # ------------------------------------------------------------------ #
    def _shard_dot(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Global inner product over Shard(0) vectors: local ``torch.dot``
        then ``dist.all_reduce(SUM)`` across the mesh."""
        d = torch.dot(u, v)
        dist.all_reduce(d, op=dist.ReduceOp.SUM)
        return d

    def _shard_norm(self, u: torch.Tensor) -> torch.Tensor:
        return self._shard_dot(u, u).sqrt()

    def _shard_matvec(self, x_owned: torch.Tensor) -> torch.Tensor:
        """Apply ``A`` to a Shard(0) vector. Pads with zeros for halo,
        runs ``halo_exchange`` + local SpMV, slices result to num_owned."""
        local_matrix = self._local_matrix
        x_padded = self._pad_owned_to_local(x_owned)
        y_full = local_matrix.matvec(x_padded, exchange_halo=True)
        return y_full[:local_matrix.num_owned]

    # ------------------------------------------------------------------ #
    # Shard(0)-space preconditioner factory.
    #
    # Each preconditioner is a callable ``apply(r_owned) -> z_owned`` that
    # maps a Shard(0) residual to an "approximate inverse" Shard(0)
    # update. Costs are independent per rank for the local ones
    # (Jacobi / block-Jacobi / SSOR); the polynomial / Neumann variant
    # internally uses ``_shard_matvec`` and so triggers halo exchange
    # per inner iteration.
    # ------------------------------------------------------------------ #
    def _make_preconditioner(self,
                              kind: Any,
                              *,
                              omega: float = 1.0,
                              degree: int = 2):
        """Build a ``apply(r_owned) -> z_owned`` callable.

        Accepts:

        * ``None`` / ``"none"``               -- identity (no precond)
        * ``"jacobi"`` / ``"jacobi_l1"``      -- diagonal inverse on
          owned rows (l1 variant scales by row-sum-of-abs instead of
          diag(A), more robust for indefinite)
        * ``"block_jacobi"``                  -- one-time dense LU on
          the owned-by-owned block; ``apply`` is two triangular solves
        * ``"ssor"``                           -- forward + backward
          symmetric SOR sweep on the owned-by-owned block, ``omega``-
          damped
        * ``"polynomial"`` / ``"neumann"``    -- Neumann series M⁻¹ ≈
          τ⁻¹ Σ_{k=0..degree-1} (I - A/τ)^k. Uses ``_shard_matvec`` so
          halo exchange happens inside the preconditioner -- costly
          but fully distributed
        * A callable ``f(r_owned) -> z_owned`` is passed through
          unchanged (for user-supplied custom preconditioners).
        """
        if callable(kind):
            return kind
        if kind is None or (isinstance(kind, str) and
                            kind.lower() == "none"):
            return lambda r: r

        local_matrix = self._local_matrix
        no = local_matrix.num_owned
        device = local_matrix.device

        # We need the owned-by-owned dense block for everything other
        # than polynomial. Build it once on first call and cache.
        def _owned_block() -> torch.Tensor:
            if getattr(self, "_owned_block_cache", None) is not None:
                return self._owned_block_cache
            csr = local_matrix._get_csr()  # num_local x num_local sparse CSR
            dense = csr.to_dense() if csr.is_sparse_csr or csr.is_sparse \
                else csr
            block = dense[:no, :no].contiguous()
            self._owned_block_cache = block
            return block

        kind_l = str(kind).lower()
        if kind_l in ("jacobi", "jacobi_l1"):
            block = _owned_block()
            if kind_l == "jacobi_l1":
                # Row-l1 norm (Anzt et al. 2015). Robust to small
                # diagonals + indefinite spectra.
                scale = block.abs().sum(dim=1).clamp_min(1e-30)
            else:
                diag = block.diagonal().clamp_min(1e-30)
                scale = diag
            inv_scale = 1.0 / scale
            return lambda r: inv_scale * r

        if kind_l == "block_jacobi":
            block = _owned_block()
            # ``torch.linalg.lu_factor`` returns (LU, pivots).
            LU, pivots = torch.linalg.lu_factor(block)
            def apply_block_jacobi(r):
                z = torch.linalg.lu_solve(
                    LU, pivots, r.unsqueeze(-1)).squeeze(-1)
                return z
            return apply_block_jacobi

        if kind_l == "ssor":
            block = _owned_block()
            # Decompose block = L + D + U; ``omega``-SSOR is
            # (D/ω + L) (D/ω)⁻¹ (D/ω + U) z = r .
            D = torch.diag(block.diagonal().clamp_min(1e-30))
            L = torch.tril(block, diagonal=-1)
            U = torch.triu(block, diagonal=1)
            D_over_om = D / omega
            M1 = D_over_om + L
            M2 = D_over_om + U
            def apply_ssor(r):
                y = torch.linalg.solve_triangular(
                    M1, r.unsqueeze(-1), upper=False).squeeze(-1)
                w = D_over_om @ y
                z = torch.linalg.solve_triangular(
                    M2, w.unsqueeze(-1), upper=True).squeeze(-1)
                return z
            return apply_ssor

        if kind_l in ("polynomial", "neumann"):
            # Estimate τ ≈ ||A||_∞ on owned rows. The row-sum is local;
            # no all_reduce needed because each rank covers its own
            # owned rows and we just need an upper-bound spectral
            # estimate. Take ``max`` across ranks for safety.
            block = _owned_block()
            tau_local = block.abs().sum(dim=1).max()
            if DIST_AVAILABLE and dist.is_initialized():
                dist.all_reduce(tau_local, op=dist.ReduceOp.MAX)
            tau = float(tau_local.item()) if float(tau_local.item()) > 0 else 1.0
            d = int(degree)

            def apply_neumann(r):
                # z = τ⁻¹ Σ_{k=0..d-1} (I - A/τ)^k r
                #   = τ⁻¹ ( r + (I-A/τ)(r + (I-A/τ)(...)) ) -- Horner
                z = r / tau
                for _ in range(d - 1):
                    Az = self._shard_matvec(z)
                    z = (r + tau * z - Az) / tau
                return z
            return apply_neumann

        raise ValueError(
            f"Unknown preconditioner {kind!r}; expected one of None / "
            "'none' / 'jacobi' / 'jacobi_l1' / 'block_jacobi' / "
            "'ssor' / 'polynomial' / 'neumann', or a callable."
        )

    def _invalidate_precond_cache(self) -> None:
        """Drop cached preconditioner factors. Call after the matrix's
        local values change (the cache is keyed implicitly on the
        DSparseTensor instance, not on a hash of the values)."""
        self._owned_block_cache = None

    def _distributed_cg_shard(
        self,
        b_owned: torch.Tensor,
        *,
        M_apply: Any,
        atol: float,
        rtol: float,
        maxiter: int,
        verbose: bool,
    ) -> torch.Tensor:
        """Preconditioned conjugate-gradient solve in Shard(0) space.

        Saad §9.2 PCG: ``rho_k = <r_k, z_k>`` with ``z_k = M^{-1} r_k``,
        ``p_k = z_k + beta_{k-1} p_{k-1}``. Identity preconditioner
        recovers plain CG. ``M_apply`` is a callable Shard(0) → Shard(0)
        produced by :meth:`_make_preconditioner`."""
        local_matrix = self._local_matrix
        no = local_matrix.num_owned

        if b_owned.shape[0] != no:
            raise ValueError(
                f"b_owned size {b_owned.shape[0]} != num_owned {no}; "
                "Shard(0) CG requires b to be the local owned slice."
            )

        x = torch.zeros_like(b_owned)
        r = b_owned - self._shard_matvec(x)
        z = M_apply(r)
        p = z.clone()
        rs_old = self._shard_dot(r, z)             # <r, z>, not <r, r>

        b_norm = float(self._shard_norm(b_owned).item())
        tol = max(atol, rtol * b_norm)

        for k in range(maxiter):
            Ap = self._shard_matvec(p)
            pAp = self._shard_dot(p, Ap)
            if float(pAp.abs().item()) < 1e-30:
                if verbose:
                    print(f"[shard-PCG] iter {k}: pAp ~= 0, stop")
                break
            alpha = rs_old / pAp
            x = x + alpha * p
            r = r - alpha * Ap

            r_norm = float(self._shard_norm(r).item())
            if verbose and (k % 100 == 0 or k < 5):
                print(f"[shard-PCG] iter {k}: ||r||={r_norm:.3e}")
            if r_norm < tol:
                if verbose:
                    print(f"[shard-PCG] converged at iter {k}, "
                          f"||r||={r_norm:.3e}")
                break
            z = M_apply(r)
            rs_new = self._shard_dot(r, z)
            beta = rs_new / rs_old
            p = z + beta * p
            rs_old = rs_new

        return x

    # ------------------------------------------------------------------ #
    # BiCGStab (Saad §7.4.2). Handles non-symmetric A; no restart.
    # ------------------------------------------------------------------ #
    def _distributed_bicgstab_shard(
        self,
        b_owned: torch.Tensor,
        *,
        M_apply: Any,
        atol: float,
        rtol: float,
        maxiter: int,
        verbose: bool,
    ) -> torch.Tensor:
        """Preconditioned BiCGStab (Saad §9.3): apply ``M^{-1}`` to
        the search directions ``p`` and ``s`` before the matvec.
        Identity ``M`` recovers plain BiCGStab."""
        no = self._local_matrix.num_owned
        if b_owned.shape[0] != no:
            raise ValueError(
                f"b_owned size {b_owned.shape[0]} != num_owned {no}")

        x = torch.zeros_like(b_owned)
        r = b_owned - self._shard_matvec(x)
        r_hat = r.clone()                 # shadow residual, fixed
        rho = alpha = omega = torch.tensor(1.0, dtype=b_owned.dtype,
                                            device=b_owned.device)
        p = torch.zeros_like(b_owned)
        v = torch.zeros_like(b_owned)

        b_norm = float(self._shard_norm(b_owned).item())
        tol = max(atol, rtol * b_norm)

        for k in range(maxiter):
            rho_new = self._shard_dot(r_hat, r)
            if float(rho_new.abs().item()) < 1e-30:
                if verbose:
                    print(f"[shard-PBiCGStab] iter {k}: rho ~= 0, breakdown")
                break
            if k == 0:
                p = r.clone()
            else:
                beta = (rho_new / rho) * (alpha / omega)
                p = r + beta * (p - omega * v)
            p_hat = M_apply(p)                       # preconditioned p
            v = self._shard_matvec(p_hat)
            denom = self._shard_dot(r_hat, v)
            if float(denom.abs().item()) < 1e-30:
                if verbose:
                    print(f"[shard-PBiCGStab] iter {k}: r_hat.v ~= 0, breakdown")
                break
            alpha = rho_new / denom
            s = r - alpha * v
            s_norm = float(self._shard_norm(s).item())
            if s_norm < tol:
                x = x + alpha * p_hat
                if verbose:
                    print(f"[shard-PBiCGStab] half-iter {k}: ||s||={s_norm:.3e}")
                break

            s_hat = M_apply(s)                       # preconditioned s
            t = self._shard_matvec(s_hat)
            tt = self._shard_dot(t, t)
            ts = self._shard_dot(t, s)
            if float(tt.abs().item()) < 1e-30:
                omega = torch.tensor(0.0, dtype=b_owned.dtype,
                                     device=b_owned.device)
            else:
                omega = ts / tt
            x = x + alpha * p_hat + omega * s_hat
            r = s - omega * t
            r_norm = float(self._shard_norm(r).item())
            if verbose and (k % 100 == 0 or k < 5):
                print(f"[shard-PBiCGStab] iter {k}: ||r||={r_norm:.3e}")
            if r_norm < tol:
                if verbose:
                    print(f"[shard-PBiCGStab] converged at iter {k}, "
                          f"||r||={r_norm:.3e}")
                break
            rho = rho_new

        return x

    # ------------------------------------------------------------------ #
    # Restarted GMRES(m) (Saad §6.5.1). Optionally flexible (FGMRES).
    # ------------------------------------------------------------------ #
    def _distributed_gmres_shard(
        self,
        b_owned: torch.Tensor,
        *,
        M_apply: Any,
        atol: float,
        rtol: float,
        maxiter: int,
        restart: int,
        flexible: bool = False,
        verbose: bool = False,
    ) -> torch.Tensor:
        no = self._local_matrix.num_owned
        if b_owned.shape[0] != no:
            raise ValueError(
                f"b_owned size {b_owned.shape[0]} != num_owned {no}")

        dtype = b_owned.dtype
        device = b_owned.device
        m = restart

        x = torch.zeros_like(b_owned)
        b_norm = float(self._shard_norm(b_owned).item())
        if b_norm == 0.0:
            return x
        tol = max(atol, rtol * b_norm)

        total_iters = 0
        for cycle in range((maxiter + m - 1) // m):
            r = b_owned - self._shard_matvec(x)
            beta = float(self._shard_norm(r).item())
            if beta < tol:
                if verbose:
                    print(f"[shard-{'F' if flexible else ''}GMRES] "
                          f"converged cycle {cycle}, ||r||={beta:.3e}")
                return x

            V = torch.zeros((m + 1, no), dtype=dtype, device=device)
            V[0] = r / beta
            # Right-preconditioned GMRES (Saad §9.3.2): we apply A M^{-1}
            # at every Arnoldi step. For FGMRES the preconditioner may
            # vary per step, so we have to remember the preconditioned
            # vectors ``Z[:, j] = M_j^{-1} V[j]`` for the final update.
            # Plain GMRES could reuse ``V`` (since M is fixed), but
            # storing ``Z`` lets both branches share one code path.
            Z = torch.zeros((m, no), dtype=dtype, device=device)
            H = torch.zeros((m + 1, m), dtype=dtype, device=device)
            g = torch.zeros(m + 1, dtype=dtype, device=device)
            g[0] = beta
            cs = torch.zeros(m, dtype=dtype, device=device)
            sn = torch.zeros(m, dtype=dtype, device=device)

            j_max = 0
            for j in range(m):
                Z[j] = M_apply(V[j])
                w = self._shard_matvec(Z[j])
                # Modified Gram-Schmidt (Saad's stable variant for GMRES).
                for i in range(j + 1):
                    H[i, j] = self._shard_dot(V[i], w)
                    w = w - H[i, j] * V[i]
                H[j + 1, j] = self._shard_norm(w)
                if float(H[j + 1, j].abs().item()) > 1e-30:
                    V[j + 1] = w / H[j + 1, j]
                else:
                    V[j + 1] = w  # lucky breakdown -- next iter will catch

                # Apply previous Givens rotations to column j of H.
                # ``.clone()`` is mandatory: ``H[i, j]`` returns a 0-d
                # view; writing to ``H[i, j]`` in the first assignment
                # would otherwise leak into the second one through the
                # ``h_ij`` alias.
                for i in range(j):
                    h_ij  = H[i, j].clone()
                    h_ipj = H[i + 1, j].clone()
                    H[i,     j] =  cs[i] * h_ij + sn[i] * h_ipj
                    H[i + 1, j] = -sn[i] * h_ij + cs[i] * h_ipj
                # New Givens rotation eliminating H[j+1, j].
                denom = (H[j, j] * H[j, j] + H[j + 1, j] * H[j + 1, j]).sqrt()
                if float(denom.abs().item()) < 1e-30:
                    cs[j] = torch.tensor(1.0, dtype=dtype, device=device)
                    sn[j] = torch.tensor(0.0, dtype=dtype, device=device)
                else:
                    cs[j] = H[j, j] / denom
                    sn[j] = H[j + 1, j] / denom
                H[j, j]     = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
                H[j + 1, j] = torch.tensor(0.0, dtype=dtype, device=device)
                # ``.clone()`` is essential: g[j] is a 0-d view; the
                # next two writes would otherwise see the partially-
                # updated value through the alias and produce a wrong
                # off-diagonal entry.
                g_j  = g[j].clone()
                g[j]     =  cs[j] * g_j
                g[j + 1] = -sn[j] * g_j

                total_iters += 1
                j_max = j + 1
                if float(g[j + 1].abs().item()) < tol:
                    break
                if total_iters >= maxiter:
                    break

            # Back-solve H[:j_max, :j_max] y = g[:j_max].
            y = torch.zeros(j_max, dtype=dtype, device=device)
            for i in range(j_max - 1, -1, -1):
                s = g[i].clone()
                for k in range(i + 1, j_max):
                    s = s - H[i, k] * y[k]
                y[i] = s / H[i, i]
            # Right-preconditioned update: x += Z y (the preconditioned
            # Arnoldi vectors), NOT V y. This is the right-precond /
            # FGMRES update; the two cases coincide because M is fixed
            # per-call here.
            for i in range(j_max):
                x = x + y[i] * Z[i]

            if verbose:
                r_norm = float(self._shard_norm(b_owned - self._shard_matvec(x)).item())
                print(f"[shard-{'F' if flexible else ''}GMRES] "
                      f"cycle {cycle}: ||r||={r_norm:.3e}")
            if total_iters >= maxiter:
                break

        return x

    # ------------------------------------------------------------------ #
    # MINRES (Paige-Saunders 1975). Symmetric indefinite, short recurrence.
    # ------------------------------------------------------------------ #
    def _distributed_minres_shard(
        self,
        b_owned: torch.Tensor,
        *,
        M_apply: Any,
        atol: float,
        rtol: float,
        maxiter: int,
        verbose: bool,
    ) -> torch.Tensor:
        """Distributed MINRES in Shard(0) space (Paige & Saunders 1975).

        Faithful port of SciPy's ``scipy.sparse.linalg.minres`` Lanczos
        + 2-step Givens recurrence -- same variable names so the math
        is reviewable against the SciPy source. Every inner product
        ``dot(.,.)`` runs through ``_shard_dot`` (local dot +
        all_reduce SUM); matvecs route through ``_shard_matvec`` which
        does halo exchange + local SpMV. M_apply is a Shard(0)
        preconditioner; identity recovers plain MINRES.
        """
        no = self._local_matrix.num_owned
        if b_owned.shape[0] != no:
            raise ValueError(
                f"b_owned size {b_owned.shape[0]} != num_owned {no}")

        eps_floor = 1e-30
        x = torch.zeros_like(b_owned)

        # Initial residual r = b - A x = b (since x = 0)
        r1 = b_owned.clone()
        y = M_apply(r1)
        beta1_sq = float(self._shard_dot(r1, y).item())
        if beta1_sq <= 0.0:
            return x
        beta1 = beta1_sq ** 0.5
        tol = max(atol, rtol * beta1)

        # Lanczos / Givens scalars (track previous two)
        oldb = 0.0
        beta = beta1
        dbar = 0.0
        epsln = 0.0
        phibar = beta1
        cs = -1.0
        sn = 0.0
        w  = torch.zeros_like(b_owned)
        w2 = torch.zeros_like(b_owned)
        r2 = r1.clone()

        for k in range(maxiter):
            s_scale = 1.0 / max(beta, eps_floor)
            v = s_scale * y
            y = self._shard_matvec(v)
            if k > 0:
                y = y - (beta / oldb) * r1

            alfa = float(self._shard_dot(v, y).item())
            y = y - (alfa / beta) * r2
            r1 = r2
            r2 = y
            y = M_apply(r2)
            oldb = beta
            beta_sq = float(self._shard_dot(r2, y).item())
            beta = (beta_sq if beta_sq > 0.0 else 0.0) ** 0.5

            # Apply previous rotation Q_{k-1}.
            oldeps = epsln
            delta  = cs * dbar + sn * alfa
            gbar   = sn * dbar - cs * alfa
            epsln  = sn * beta
            dbar   = -cs * beta

            # New rotation Q_k.
            gamma = max((gbar * gbar + beta * beta) ** 0.5, eps_floor)
            cs    = gbar / gamma
            sn    = beta / gamma
            phi   = cs * phibar
            phibar = sn * phibar

            # Update solution and search direction.
            denom = 1.0 / gamma
            w1 = w2
            w2 = w
            w  = (v - oldeps * w1 - delta * w2) * denom
            x  = x + phi * w

            rnorm = abs(phibar)
            if verbose and (k % 100 == 0 or k < 5):
                print(f"[shard-MINRES] iter {k}: ||r||~={rnorm:.3e}")
            if rnorm < tol:
                if verbose:
                    print(f"[shard-MINRES] converged at iter {k}, "
                          f"||r||~={rnorm:.3e}")
                break

        return x

    def _matmul_dtensor(self, x: "DTensor") -> "DTensor":
        """
        Matrix-vector multiplication with DTensor input.
        
        Handles DTensor layout conversion and result wrapping.
        
        Parameters
        ----------
        x : DTensor
            Distributed tensor input
            
        Returns
        -------
        DTensor
            Result as DTensor with same placement as input
        """
        if not DTENSOR_AVAILABLE:
            raise RuntimeError("DTensor support requires PyTorch 2.0+")
        
        # Get DTensor metadata
        device_mesh = x.device_mesh
        placements = x.placements
        
        # Store original placement for output
        original_placements = tuple(placements)
        
        # Check if input is replicated (easiest case)
        is_replicated = all(isinstance(p, Replicate) for p in placements)
        
        if is_replicated:
            # Input is replicated on all ranks - just extract and compute
            x_local = x.to_local()
            y_local = self._distributed_matvec(x_local)
            # Wrap result as replicated DTensor
            return DTensor.from_local(y_local, device_mesh, [Replicate()])
        
        # Input is sharded - need to handle redistribution
        # For sparse matvec, we typically need the full vector on each rank
        # (because sparse matrix rows may reference any column)
        
        # Redistribute to replicated
        replicate_placements = [Replicate() for _ in placements]
        x_replicated = x.redistribute(device_mesh, replicate_placements)
        x_full = x_replicated.to_local()
        
        # Compute matvec with full vector
        y_full = self._distributed_matvec(x_full)
        
        # Wrap as replicated DTensor first
        y_replicated = DTensor.from_local(y_full, device_mesh, [Replicate()])
        
        # Redistribute back to original placement if it was sharded
        if not is_replicated:
            # For output, we shard along the row dimension (dim 0)
            # which corresponds to the matrix row partitioning
            output_placements = []
            for p in original_placements:
                if isinstance(p, Shard):
                    # Preserve shard dimension for output
                    output_placements.append(Shard(p.dim))
                else:
                    output_placements.append(Replicate())
            
            return y_replicated.redistribute(device_mesh, output_placements)
        
        return y_replicated
    
    # =========================================================================
    # Representation
    # =========================================================================
    
    def __repr__(self) -> str:
        return (f"DSparseTensor(shape={self._shape}, num_partitions={self._num_partitions}, "
                f"nnz={self.nnz}, device={self._device})")
    
    # =========================================================================
    # Persistence (I/O)
    # =========================================================================
    
    def save(
        self,
        directory: Union[str, "os.PathLike"],
        verbose: bool = False
    ) -> None:
        """
        Save DSparseTensor to disk.
        
        Creates a directory with metadata and per-partition files.
        
        Parameters
        ----------
        directory : str or PathLike
            Output directory.
        verbose : bool
            Print progress.
        
        Example
        -------
        >>> D = A.partition(num_partitions=4)
        >>> D.save("matrix_dist")
        """
        from .io import save_dsparse
        save_dsparse(self, directory, verbose)
    
    @classmethod
    def load(
        cls,
        directory: Union[str, "os.PathLike"],
        device: Union[str, torch.device] = "cpu"
    ) -> "DSparseTensor":
        """
        Load a complete DSparseTensor from disk.
        
        Parameters
        ----------
        directory : str or PathLike
            Directory containing saved data.
        device : str or torch.device
            Device to load to.
        
        Returns
        -------
        DSparseTensor
            The loaded distributed sparse tensor.
        
        Example
        -------
        >>> D = DSparseTensor.load("matrix_dist", device="cuda")
        """
        from .io import load_dsparse
        return load_dsparse(directory, device)


# =============================================================================
# DSparseTensorList Class
# =============================================================================

class DSparseTensorList:
    """
    Distributed Sparse Tensor List for batched graph operations.
    
    Holds a collection of graphs where:
    - Small graphs are assigned whole to individual ranks
    - Large graphs are partitioned across ranks using METIS/RCB
    
    This is ideal for molecular property prediction and other batched
    graph learning tasks where graphs have varying sizes.
    
    Parameters
    ----------
    local_matrices : List[DSparseMatrix]
        List of local partitions/graphs for this rank.
    graph_ids : List[int]
        Global graph ID for each local matrix.
    graph_sizes : List[int]
        Number of nodes in each global graph.
    is_partitioned : List[bool]
        Whether each graph is partitioned across ranks.
    device : torch.device
        Device for computations.
    
    Examples
    --------
    >>> # Create from SparseTensorList
    >>> stl = SparseTensorList([A1, A2, A3, ...])
    >>> dstl = stl.partition(num_partitions=4)
    >>> 
    >>> # Distributed operations
    >>> y_list = dstl @ x_list  # matmul
    >>> x_list = dstl.solve(b_list)  # solve
    >>> 
    >>> # Gather back
    >>> stl_result = dstl.gather()
    """
    
    def __init__(
        self,
        local_matrices: List[DSparseMatrix],
        graph_ids: List[int],
        graph_sizes: List[int],
        is_partitioned: List[bool],
        rank: int = 0,
        world_size: int = 1,
        device: Optional[Union[str, torch.device]] = None
    ):
        self._local_matrices = local_matrices
        self._graph_ids = graph_ids
        self._graph_sizes = graph_sizes
        self._is_partitioned = is_partitioned
        self._rank = rank
        self._world_size = world_size
        
        if device is None:
            device = local_matrices[0].device if local_matrices else torch.device('cpu')
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
    
    @classmethod
    def from_sparse_tensor_list(
        cls,
        sparse_list: "SparseTensorList",
        num_partitions: int,
        threshold: int = 1000,
        partition_method: str = 'auto',
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = False
    ) -> "DSparseTensorList":
        """
        Create DSparseTensorList from SparseTensorList.
        
        Parameters
        ----------
        sparse_list : SparseTensorList
            Input list of sparse matrices.
        num_partitions : int
            Number of partitions (typically = world_size).
        threshold : int
            Graphs with nodes >= threshold are partitioned.
            Smaller graphs are assigned whole to ranks.
        partition_method : str
            Partitioning method for large graphs: 'metis', 'simple', 'auto'.
        device : torch.device, optional
            Target device.
        verbose : bool
            Print partition info.
            
        Returns
        -------
        DSparseTensorList
            Distributed list ready for parallel operations.
            
        Notes
        -----
        **Partition Strategy:**
        
        - Small graphs (nodes < threshold): Assigned whole to ranks
          using round-robin. No edge cuts, minimal communication.
        - Large graphs (nodes >= threshold): Partitioned across ranks
          using METIS/RCB. Requires halo exchange for operations.
        
        This hybrid strategy is optimal for datasets with mixed graph sizes
        (e.g., molecular datasets with varying molecule sizes).
        
        Examples
        --------
        >>> stl = SparseTensorList([A1, A2, A3, ...])  # Many small graphs
        >>> dstl = DSparseTensorList.from_sparse_tensor_list(
        ...     stl, num_partitions=4, threshold=1000
        ... )
        """
        from .sparse_tensor import SparseTensorList
        
        if device is None:
            device = sparse_list.device
        if isinstance(device, str):
            device = torch.device(device)
        
        n_graphs = len(sparse_list)
        graph_sizes = [t.sparse_shape[0] for t in sparse_list]
        
        # Classify graphs
        small_graph_ids = []
        large_graph_ids = []
        
        for i, size in enumerate(graph_sizes):
            if size >= threshold:
                large_graph_ids.append(i)
            else:
                small_graph_ids.append(i)
        
        if verbose:
            print(f"DSparseTensorList: {n_graphs} graphs")
            print(f"  Small (<{threshold} nodes): {len(small_graph_ids)}")
            print(f"  Large (>={threshold} nodes): {len(large_graph_ids)}")
        
        # For single-node simulation, create all partitions
        # In true distributed mode, each rank would only create its portion
        all_partitions = [[] for _ in range(num_partitions)]
        all_graph_ids = [[] for _ in range(num_partitions)]
        all_is_partitioned = [[] for _ in range(num_partitions)]
        
        # Assign small graphs round-robin
        for idx, graph_id in enumerate(small_graph_ids):
            target_rank = idx % num_partitions
            tensor = sparse_list[graph_id]
            
            # Create DSparseMatrix for whole graph (single partition)
            mat = DSparseMatrix.from_global(
                tensor.values, tensor.row_indices, tensor.col_indices,
                tensor.sparse_shape,
                num_partitions=1, my_partition=0,
                device=device, verbose=False
            )
            all_partitions[target_rank].append(mat)
            all_graph_ids[target_rank].append(graph_id)
            all_is_partitioned[target_rank].append(False)
        
        # Partition large graphs across ranks
        for graph_id in large_graph_ids:
            tensor = sparse_list[graph_id]
            
            # Create partitioned matrix
            for part_id in range(num_partitions):
                mat = DSparseMatrix.from_global(
                    tensor.values, tensor.row_indices, tensor.col_indices,
                    tensor.sparse_shape,
                    num_partitions=num_partitions, my_partition=part_id,
                    device=device, verbose=False
                )
                all_partitions[part_id].append(mat)
                all_graph_ids[part_id].append(graph_id)
                all_is_partitioned[part_id].append(True)
        
        if verbose:
            for rank in range(num_partitions):
                n_local = len(all_partitions[rank])
                n_whole = sum(1 for p in all_is_partitioned[rank] if not p)
                print(f"  Rank {rank}: {n_local} local matrices ({n_whole} whole graphs)")
        
        # Return combined structure (for single-node, rank=0 gets all info)
        # In true distributed, each rank would only have its portion
        return cls(
            local_matrices=all_partitions[0],  # For single-node simulation
            graph_ids=all_graph_ids[0],
            graph_sizes=graph_sizes,
            is_partitioned=all_is_partitioned[0],
            rank=0,
            world_size=num_partitions,
            device=device
        )
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def device(self) -> torch.device:
        """Device of the matrices."""
        return self._device
    
    @property
    def rank(self) -> int:
        """Current rank."""
        return self._rank
    
    @property
    def world_size(self) -> int:
        """Total number of ranks."""
        return self._world_size
    
    @property
    def num_local_graphs(self) -> int:
        """Number of local matrices on this rank."""
        return len(self._local_matrices)
    
    @property
    def num_total_graphs(self) -> int:
        """Total number of unique graphs (across all ranks)."""
        return len(set(self._graph_ids))
    
    def __len__(self) -> int:
        """Number of local matrices."""
        return len(self._local_matrices)
    
    def __getitem__(self, idx: int) -> DSparseMatrix:
        """Get local matrix by index."""
        return self._local_matrices[idx]
    
    def __iter__(self):
        """Iterate over local matrices."""
        return iter(self._local_matrices)
    
    # =========================================================================
    # Operations
    # =========================================================================
    
    def __matmul__(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Distributed matrix-vector multiplication for all local graphs.
        
        Parameters
        ----------
        x_list : List[torch.Tensor]
            List of input vectors, one per local matrix.
            
        Returns
        -------
        List[torch.Tensor]
            List of output vectors.
        """
        if len(x_list) != len(self._local_matrices):
            raise ValueError(f"Expected {len(self._local_matrices)} vectors, got {len(x_list)}")
        
        results = []
        for mat, x in zip(self._local_matrices, x_list):
            y = mat.matvec(x)
            results.append(y)
        
        return results
    
    def matvec_all(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Alias for __matmul__."""
        return self @ x_list
    
    def solve_all(
        self,
        b_list: List[torch.Tensor],
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Solve linear systems for all local graphs.
        
        Parameters
        ----------
        b_list : List[torch.Tensor]
            List of RHS vectors, one per local matrix.
        **kwargs
            Arguments passed to DSparseMatrix.solve().
            
        Returns
        -------
        List[torch.Tensor]
            List of solution vectors.
        """
        if len(b_list) != len(self._local_matrices):
            raise ValueError(f"Expected {len(self._local_matrices)} vectors, got {len(b_list)}")
        
        results = []
        for mat, b in zip(self._local_matrices, b_list):
            x = mat.solve(b, **kwargs)
            results.append(x)
        
        return results
    
    # =========================================================================
    # Conversion
    # =========================================================================
    
    def gather(self) -> "SparseTensorList":
        """
        Gather all graphs back to a single SparseTensorList.
        
        In distributed mode, this collects data from all ranks.
        For partitioned graphs, it reassembles the full graph.
        
        Returns
        -------
        SparseTensorList
            Gathered list of sparse tensors.
        """
        from .sparse_tensor import SparseTensor, SparseTensorList
        
        # For single-node simulation, reconstruct from local data
        # In true distributed, this would involve all_gather
        
        tensors = []
        for mat in self._local_matrices:
            # Get global data from partition
            partition = mat.partition
            
            # Reconstruct global indices
            global_row = partition.local_to_global[mat.local_row]
            global_col = partition.local_to_global[mat.local_col]
            
            sparse = SparseTensor(
                mat.local_values,
                global_row,
                global_col,
                mat.global_shape
            )
            tensors.append(sparse)
        
        return SparseTensorList(tensors)
    
    def to_block_diagonal(self) -> DSparseTensor:
        """
        Convert to a single distributed block-diagonal matrix.
        
        Merges all graphs into one block-diagonal DSparseTensor.
        
        Returns
        -------
        DSparseTensor
            Block-diagonal distributed matrix.
        """
        # First gather to SparseTensorList
        stl = self.gather()
        
        # Convert to block diagonal
        block_diag = stl.to_block_diagonal()
        
        # Create DSparseTensor
        return DSparseTensor(
            block_diag.values,
            block_diag.row_indices,
            block_diag.col_indices,
            block_diag.sparse_shape,
            num_partitions=self._world_size,
            device=self._device,
            verbose=False
        )
    
    # =========================================================================
    # Device Management
    # =========================================================================
    
    def to(self, device: Union[str, torch.device]) -> "DSparseTensorList":
        """Move all matrices to device."""
        if isinstance(device, str):
            device = torch.device(device)
        
        new_matrices = [m.to(device) for m in self._local_matrices]
        return DSparseTensorList(
            new_matrices,
            self._graph_ids.copy(),
            self._graph_sizes.copy(),
            self._is_partitioned.copy(),
            self._rank,
            self._world_size,
            device
        )
    
    def cuda(self) -> "DSparseTensorList":
        """Move to CUDA."""
        return self.to('cuda')
    
    def cpu(self) -> "DSparseTensorList":
        """Move to CPU."""
        return self.to('cpu')
    
    def __repr__(self) -> str:
        n_whole = sum(1 for p in self._is_partitioned if not p)
        n_part = sum(1 for p in self._is_partitioned if p)
        return (f"DSparseTensorList(local={len(self)}, "
                f"whole_graphs={n_whole}, partitioned={n_part}, "
                f"rank={self._rank}/{self._world_size}, device={self._device})")
