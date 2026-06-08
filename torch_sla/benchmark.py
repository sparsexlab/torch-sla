"""Sparse-solve benchmarking utilities.

Exposes the :class:`Benchmark` container that bundles a sparse matrix
``A`` together with a list of ``(x_ref, b)`` test cases where
``A @ x_ref == b`` by construction. Indexing yields a single case;
``Benchmark.evaluate(solver, indices, metric)`` runs a user-supplied
solver and returns the error against the stored reference.

Datasets such as :class:`torch_sla.datasets.SuiteSparse` return
``Benchmark`` instances so plotting / regression code can iterate them
uniformly without re-implementing case generation.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

# A solver takes the matrix CSR-or-COO triple + RHS and returns x.
SolverFn = Callable[[Tensor, Tensor, Tensor, Tuple[int, int], Tensor], Tensor]

_Metric = Union[str, Callable[[Tensor, Tensor], float]]


def _rel_l2(x_hat: Tensor, x_ref: Tensor) -> float:
    denom = x_ref.norm()
    if denom.item() == 0.0:
        return (x_hat - x_ref).norm().item()
    return ((x_hat - x_ref).norm() / denom).item()


def _max_abs(x_hat: Tensor, x_ref: Tensor) -> float:
    return (x_hat - x_ref).abs().max().item()


_METRICS = {"rel_l2": _rel_l2, "max_abs": _max_abs}


class Benchmark:
    """A sparse matrix ``A`` plus a list of ``(x_ref, b)`` reference cases.

    Parameters
    ----------
    name : str
        Identifier for the matrix (e.g. ``"Bai/mhd1280b"``). Used in
        diagnostic output.
    val, row, col : torch.Tensor
        COO triple of ``A``.
    shape : (int, int)
        Square matrix shape.
    cases : list of dict, optional
        Pre-computed reference cases, each a dict with keys ``"x"`` and
        ``"b"`` (both 1-D tensors). If omitted, three random cases are
        generated via :meth:`_make_random_cases`.
    n_cases : int, default 3
        Number of random cases to generate when ``cases is None``.
    seed : int, default 0
        Base seed; case ``i`` uses ``seed + i``.
    math_kind, detected_kind : str, optional
        The matrix's true mathematical kind (e.g. ``"hpd"``) and the kind
        the heuristic Gershgorin-based detector is expected to label
        it as (may be more conservative). Both used by tests; users can
        ignore.
    """

    __slots__ = (
        "name", "val", "row", "col", "shape",
        "math_kind", "detected_kind", "_cases",
    )

    def __init__(
        self,
        name: str,
        val: Tensor,
        row: Tensor,
        col: Tensor,
        shape: Tuple[int, int],
        *,
        cases: Optional[Sequence[dict]] = None,
        n_cases: int = 3,
        seed: int = 0,
        math_kind: Optional[str] = None,
        detected_kind: Optional[str] = None,
    ):
        if shape[0] != shape[1]:
            raise ValueError(f"Benchmark expects a square matrix; got {shape}")
        self.name = name
        self.val = val
        self.row = row
        self.col = col
        self.shape = tuple(shape)
        self.math_kind = math_kind
        self.detected_kind = detected_kind
        if cases is None:
            cases = self._make_random_cases(n_cases=n_cases, seed=seed)
        self._cases: List[dict] = list(cases)

    # ------------------------------------------------------------------ #
    # Sequence protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._cases)

    def __getitem__(self, i: int) -> dict:
        return self._cases[i]

    def __iter__(self):
        return iter(self._cases)

    def __repr__(self) -> str:
        n = self.shape[0]
        nnz = self.val.numel()
        kind = self.math_kind or "?"
        return (f"Benchmark({self.name!r}, n={n}, nnz={nnz}, "
                f"dtype={self.val.dtype}, kind={kind}, cases={len(self)})")

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        solver: SolverFn,
        indices: Optional[Union[int, Iterable[int]]] = None,
        *,
        metric: _Metric = "rel_l2",
    ) -> Union[float, List[float]]:
        """Run ``solver`` on the chosen cases, return error(s) vs reference.

        Parameters
        ----------
        solver : callable
            ``solver(val, row, col, shape, b) -> x_hat``. Matches
            torch-sla's :func:`SparseTensor` constructor + ``solve(b)``.
        indices : int or iterable of int, optional
            Case indices to evaluate. ``None`` runs all cases.
            ``int`` runs one and returns a scalar float; an iterable
            returns a list of floats.
        metric : ``'rel_l2'`` | ``'max_abs'`` | callable, default ``'rel_l2'``
            ``'rel_l2'``  is ``||x_hat - x_ref||_2 / ||x_ref||_2``;
            ``'max_abs'`` is ``max |x_hat - x_ref|``; or pass a callable
            ``(x_hat, x_ref) -> float``.
        """
        metric_fn = _METRICS[metric] if isinstance(metric, str) else metric
        scalar = isinstance(indices, int)
        if indices is None:
            idxs = list(range(len(self)))
        elif scalar:
            idxs = [indices]
        else:
            idxs = list(indices)

        out: List[float] = []
        for i in idxs:
            case = self._cases[i]
            x_hat = solver(self.val, self.row, self.col, self.shape, case["b"])
            out.append(metric_fn(x_hat, case["x"]))
        return out[0] if scalar else out

    # ------------------------------------------------------------------ #
    # Reference-case generation
    # ------------------------------------------------------------------ #
    def _make_random_cases(self, *, n_cases: int, seed: int) -> List[dict]:
        """Generate ``n_cases`` reference cases by drawing random ``x_ref``
        and computing ``b = A @ x_ref``.

        Uses ``torch.sparse_coo_tensor`` + ``torch.mv`` so the dtype path
        matches what the solvers actually see (complex matvec, real
        matvec, etc.). All cases live on CPU; move them yourself for GPU
        benchmarks.
        """
        n = self.shape[0]
        dtype = self.val.dtype
        # Coalesce once for stable matvec.
        indices = torch.stack([self.row, self.col], dim=0)
        A = torch.sparse_coo_tensor(indices, self.val, self.shape).coalesce()
        is_complex = dtype.is_complex
        real_dtype = (torch.float32 if dtype == torch.complex64
                      else torch.float64 if dtype == torch.complex128
                      else dtype)

        cases: List[dict] = []
        for i in range(n_cases):
            g = torch.Generator().manual_seed(seed + i)
            if is_complex:
                x = (torch.randn(n, generator=g, dtype=real_dtype)
                     + 1j * torch.randn(n, generator=g, dtype=real_dtype)).to(dtype)
            else:
                x = torch.randn(n, generator=g, dtype=dtype)
            b = torch.mv(A, x)
            cases.append({"x": x, "b": b, "seed": seed + i})
        return cases


# ---------------------------------------------------------------------- #
# Plotting helper (kept separate from Benchmark)
# ---------------------------------------------------------------------- #
def plot_errors(
    errors_by_benchmark: dict,
    *,
    title: str = "Solver error per SuiteSparse benchmark",
    log_y: bool = True,
    ax=None,
):
    """Box-plot error spread for a dict of ``{name: list_of_errors}``.

    A thin convenience wrapper around matplotlib; only imported here so
    matplotlib is not a hard runtime dep. Raises ``ImportError`` with a
    hint if matplotlib is missing.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_errors requires matplotlib. Install with "
            "`pip install matplotlib`."
        ) from e

    if ax is None:
        _fig, ax = plt.subplots(figsize=(max(6, len(errors_by_benchmark) * 1.2), 4))

    names = list(errors_by_benchmark.keys())
    data = [list(errors_by_benchmark[k]) for k in names]
    ax.boxplot(data, labels=names, showfliers=True)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    return ax
