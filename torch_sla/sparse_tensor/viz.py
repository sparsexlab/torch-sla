"""Visualization for :class:`SparseTensor` (spy plot).

Matplotlib is imported lazily so torch_sla without it installed keeps
working for non-viz code paths.
"""
from __future__ import annotations

from typing import Optional, Tuple


def spy(
    self,
    batch_idx: Optional[Tuple[int, ...]] = None,
    ax=None,
    title: Optional[str] = None,
    cmap: str = 'viridis',
    show_grid: bool = True,
    grid_color: str = '#cccccc',
    grid_linewidth: float = 0.5,
    show_colorbar: bool = True,
    figsize: Tuple[float, float] = (8, 8),
    save_path: Optional[str] = None,
    dpi: int = 150,
):
    """Render the sparsity pattern as a pixel-perfect image. Each stored
    entry is a pixel coloured by ``|value|`` (normalised), zeros are white.
    Returns the matplotlib Axes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for spy(). pip install matplotlib")

    if self.is_batched:
        if batch_idx is None:
            raise ValueError("batch_idx is required for batched tensors")
        flat_idx = 0
        for i, (idx, s) in enumerate(zip(batch_idx, self.batch_shape)):
            flat_idx = flat_idx * s + idx
        vals = self.values.reshape(-1, self.nnz)[flat_idx]
    else:
        vals = self.values

    row = self.row_indices.cpu().numpy()
    col = self.col_indices.cpu().numpy()
    vals_np = vals.abs().cpu().numpy()
    M, N = self.sparse_shape

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.get_figure()

    vals_norm = vals_np / vals_np.max() if vals_np.max() > 0 else vals_np

    import numpy as np
    image = np.full((M, N), np.nan, dtype=np.float32)
    image[row, col] = vals_norm

    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color='white')

    im = ax.imshow(image, cmap=cmap_obj, aspect='equal',
                   interpolation='nearest', vmin=0, vmax=1, origin='upper')

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('|value| (normalized)', fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(1)

    if show_grid and max(M, N) <= 30:
        ax.set_xticks([i - 0.5 for i in range(N + 1)], minor=True)
        ax.set_yticks([i - 0.5 for i in range(M + 1)], minor=True)
        ax.grid(which='minor', color=grid_color, linewidth=grid_linewidth)
        ax.tick_params(which='minor', length=0)

    if title is None:
        nnz = len(vals_np)
        sparsity = 1 - nnz / (M * N)
        title = f'Sparse Matrix: {M}×{N}, nnz={nnz:,}, sparsity={sparsity:.1%}'
    ax.set_title(title, fontsize=12, fontweight='bold')

    if created_fig:
        plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return ax
