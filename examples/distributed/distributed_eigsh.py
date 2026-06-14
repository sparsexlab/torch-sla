#!/usr/bin/env python
"""Distributed eigenvalue computation via ``DSparseTensor.eigsh``.

Distributed LOBPCG on a tridiagonal SPD matrix. Each rank holds the
full Ritz basis ``X`` (replicated); the distributed step is the
column-wise matvec via Shard(0). Spectrum compared against scipy.

Run::

    torchrun --standalone --nproc_per_node=4 distributed_eigsh.py
"""

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"{'=' * 60}\nDistributed eigsh: A v = lambda v  (world={world_size})\n{'=' * 60}")

    from torch.distributed.device_mesh import init_device_mesh
    from torch_sla import DSparseTensor, SparseTensor

    # SPD tridiagonal matrix, known spectrum.
    n = 200
    A = SparseTensor.tridiagonal(n, diag=4.0, off_diag=-1.0)

    mesh = init_device_mesh("cpu", (world_size,))
    D = DSparseTensor.partition(A, mesh, partition_method="simple")

    # Distributed LOBPCG: 5 smallest-magnitude eigenpairs. SM is chosen
    # for the demo because the tridiag(4, -1) spectrum is *very* dense
    # near its upper end (top-5 within ~1e-4 of each other), so LM
    # converges slowly without a shift; SM has well-separated eigenvalues.
    # Every rank participates -- eigsh is collective.
    k = 5
    evals, evecs = D.eigsh(k=k, which="SM", maxiter=400, tol=1e-10)

    print(f"[rank {rank}] evals = {[f'{v:.4f}' for v in evals.tolist()]} "
          f"evec0 norm={evecs[:, 0].norm().item():.6f}")

    # Cross-check against scipy on rank 0 only (post-solve, no comm).
    if rank == 0:
        import numpy as np
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        A_sp = sp.coo_matrix(
            (A.values.numpy(), (A.row_indices.numpy(), A.col_indices.numpy())),
            shape=(n, n),
        ).tocsr()
        ref_vals, _ = spla.eigsh(A_sp, k=k, which="SM")
        ref_vals = np.sort(ref_vals)
        got = sorted(evals.tolist())

        print("\n     rank  distributed       scipy            rel err")
        print("     ----  ----------------  ----------------  --------")
        for i, (g, r) in enumerate(zip(got, ref_vals)):
            rel = abs(g - r) / max(1e-12, abs(r))
            print(f"     {i:>4d}  {g:>16.10f}  {r:>16.10f}  {rel:.2e}")
            # LOBPCG without a preconditioner converges slowly on
            # close-spaced spectra (tridiag(4,-1) has 1e-3 gaps near
            # both endpoints); 1e-3 relative is the realistic bar here.
            assert rel < 1e-3, f"eigenvalue {i} drifted (rel={rel:.2e})"
        print("\nDistributed eigsh converged.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
