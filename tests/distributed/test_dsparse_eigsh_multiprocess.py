#!/usr/bin/env python
"""``DSparseTensor.eigsh`` -- distributed LOBPCG parity tests.

Phase C7: validates the new distributed eigsh against
``scipy.sparse.linalg.eigsh`` on the same SPD matrix.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _eigsh_worker(rank: int, world_size: int, port: int,
                  k: int, which: str,
                  out_queue: mp.Queue) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo",
                            rank=rank, world_size=world_size)
    try:
        from torch_sla import SparseTensor
        from torch_sla.datasets import Synthetic

        bench = Synthetic["poisson_2d_16"]
        A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)
        D = A.partition_for_rank(rank=rank, world_size=world_size)

        evals, evecs = D.eigsh(k=k, which=which, maxiter=400, tol=1e-9)

        # Sanity check: rank parity -- every rank should land the same
        # spectrum.
        out_queue.put({
            "rank":  rank,
            "evals": evals.tolist(),
            "evecs_shape": tuple(evecs.shape),
            "evec0_norm": float(evecs[:, 0].norm().item()),
        })
    finally:
        dist.destroy_process_group()


def _scipy_reference(A_sparse_tensor, k: int, which: str):
    """Compute reference eigenpairs with scipy."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    val = A_sparse_tensor.values.cpu().numpy()
    row = A_sparse_tensor.row_indices.cpu().numpy()
    col = A_sparse_tensor.col_indices.cpu().numpy()
    A_sp = sp.coo_matrix(
        (val, (row, col)),
        shape=tuple(A_sparse_tensor.shape),
    ).tocsr()
    # scipy.eigsh requires k < N - 1
    return spla.eigsh(A_sp, k=k, which=which)


@pytest.mark.skipif(
    not dist.is_available(),
    reason="torch.distributed not available",
)
@pytest.mark.parametrize("which,port", [
    ("LM", 29670),
    ("SM", 29672),
])
def test_dsparse_eigsh_vs_scipy_2procs(which, port):
    """LM / SM eigenvalues match scipy to ~1e-6 relative."""
    from torch_sla import SparseTensor
    from torch_sla.datasets import Synthetic

    bench = Synthetic["poisson_2d_16"]
    A = SparseTensor(bench.val, bench.row, bench.col, bench.shape)

    k = 4
    ref_evals, ref_evecs = _scipy_reference(A, k=k, which=which)
    # scipy returns ascending; LOBPCG returns the requested order
    ref_evals = np.sort(ref_evals) if which in ("SM", "SA") else \
                np.sort(ref_evals)[::-1]

    world_size = 2
    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_eigsh_worker,
                         args=(rank, world_size, port, k, which, q))
             for rank in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)

    results = []
    while not q.empty():
        results.append(q.get())
    assert len(results) == world_size

    # Every rank should report the same spectrum.
    e0 = results[0]["evals"]
    for r in results:
        for a, b in zip(r["evals"], e0):
            assert abs(a - b) < 1e-10, \
                f"rank {r['rank']}: eigvals diverge across ranks"

    # Spectrum should match scipy.
    sorted_dist = sorted(e0) if which in ("SM", "SA") else \
                  sorted(e0, reverse=True)
    for got, ref in zip(sorted_dist, ref_evals):
        rel = abs(got - ref) / max(1e-12, abs(ref))
        assert rel < 5e-6, \
            f"{which}: got {got}, ref {ref}, rel err {rel:.2e}"

    # Eigenvectors unit-normed.
    for r in results:
        assert abs(r["evec0_norm"] - 1.0) < 1e-6

    print(f"\n[OK] eigsh {which} on 2 procs vs scipy:")
    for got, ref in zip(sorted_dist, ref_evals):
        print(f"  got={got:.10f}  ref={ref:.10f}  rel={abs(got-ref)/max(1e-12,abs(ref)):.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
