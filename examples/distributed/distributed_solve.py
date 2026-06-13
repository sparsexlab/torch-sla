#!/usr/bin/env python
"""Distributed linear solve via ``DSparseTensor`` + unified ``solve``.

Demonstrates the row-sharded CG / BiCGStab / GMRES family on the same
SPD tridiagonal matrix, with Jacobi preconditioning. Verified against a
scipy single-process CG so any drift fails the example.

Run::

    torchrun --standalone --nproc_per_node=4 distributed_solve.py
"""

import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"{'=' * 60}\nDistributed Solve: Ax = b  (world={world_size})\n{'=' * 60}")

    from torch.distributed.device_mesh import init_device_mesh
    from torch_sla import DSparseTensor, SparseTensor, solve, SolverConfig

    # ── SPD tridiagonal A, deterministic b. ──
    n = 256
    idx = torch.arange(n)
    val = torch.cat([
        torch.full((n,), 4.0, dtype=torch.float64),
        torch.full((n - 1,), -1.0, dtype=torch.float64),
        torch.full((n - 1,), -1.0, dtype=torch.float64),
    ])
    row = torch.cat([idx, idx[1:], idx[:-1]])
    col = torch.cat([idx, idx[:-1], idx[1:]])
    A = SparseTensor(val, row, col, shape=(n, n))

    mesh = init_device_mesh("cpu", (world_size,))
    D = DSparseTensor.partition(A, mesh, partition_method="simple")

    torch.manual_seed(0)
    b_global = torch.randn(n, dtype=torch.float64)
    b_dt = D.scatter(b_global)

    # ── Unified API: solve auto-routes on the DSparseTensor type. ──
    with SolverConfig(method="cg", preconditioner="jacobi",
                       atol=1e-12, rtol=1e-10, maxiter=2000):
        x_dt = solve(D, b_dt)

    # Residual ‖b − A x‖ / ‖b‖, computed without leaving Shard(0).
    r_dt = b_dt - D @ x_dt
    rel_resid = (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item()

    # Cross-check against a scipy CG on the same global system.
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    A_sp = sp.coo_matrix(
        (A.values.numpy(), (A.row_indices.numpy(), A.col_indices.numpy())),
        shape=(n, n),
    ).tocsr()
    x_ref, _ = spla.cg(A_sp, b_global.numpy(), rtol=1e-12, maxiter=2000)
    rel_to_scipy = (
        (torch.from_numpy(x_ref) - x_dt.full_tensor()).norm()
        / (torch.from_numpy(x_ref).norm() + 1e-12)
    ).item()

    print(f"[rank {rank}] rel residual={rel_resid:.2e}  "
          f"rel diff vs scipy CG={rel_to_scipy:.2e}")

    if rank == 0:
        assert rel_resid < 1e-8 and rel_to_scipy < 1e-6, "distributed solve diverged"
        print("\nDistributed CG converged.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
