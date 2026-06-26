#!/usr/bin/env python
"""Distributed linear solve via ``DSparseTensor`` + unified ``solve``.

Demonstrates the row-sharded CG / BiCGStab / GMRES family on the same
SPD tridiagonal matrix, with Jacobi preconditioning. Verified against a
scipy single-process CG so any drift fails the example.

Device-aware: uses NCCL + CUDA (one GPU per ``LOCAL_RANK``) when a GPU
is visible, else falls back to gloo + CPU so it still runs on a laptop.

Run::

    # single node (one box, N procs)
    torchrun --standalone --nproc_per_node=4 distributed_solve.py

    # multiple nodes (run on EVERY node; HEAD_NODE_IP reachable by all)
    torchrun --nnodes=2 --nproc_per_node=4 \
        --rdzv-id=sla --rdzv-backend=c10d \
        --rdzv-endpoint=HEAD_NODE_IP:29500 distributed_solve.py
"""

import os
import torch
import torch.distributed as dist


def main():
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        mesh_device = "cuda"
    else:
        device = torch.device("cpu")
        mesh_device = "cpu"

    if rank == 0:
        print(f"{'=' * 60}\nDistributed Solve: Ax = b  "
              f"(world={world_size}, backend={backend}, device={mesh_device})"
              f"\n{'=' * 60}")

    from torch.distributed.device_mesh import init_device_mesh
    from torch_sla import DSparseTensor, SparseTensor, solve, SolverConfig

    # SPD tridiagonal A, deterministic b.
    n = 256
    A = SparseTensor.tridiagonal(n, diag=4.0, off_diag=-1.0).to(device)

    mesh = init_device_mesh(mesh_device, (world_size,))
    D = DSparseTensor.partition(A, mesh, partition_method="simple")

    torch.manual_seed(0)
    b_global = torch.randn(n, dtype=torch.float64, device=device)
    b_dt = D.scatter(b_global)

    # Unified API: solve auto-routes on the DSparseTensor type.
    with SolverConfig(method="cg", preconditioner="jacobi",
                       atol=1e-12, rtol=1e-10, maxiter=2000):
        x_dt = solve(D, b_dt)

    # Residual ‖b − A x‖ / ‖b‖, computed without leaving Shard(0).
    r_dt = b_dt - D @ x_dt
    rel_resid = (r_dt.full_tensor().norm() / b_dt.full_tensor().norm()).item()

    # Cross-check against a scipy CG on the same global system. scipy is
    # CPU/numpy only, so bring the global solution + b back to CPU first.
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    A_cpu = A.to("cpu")
    A_sp = sp.coo_matrix(
        (A_cpu.values.numpy(), (A_cpu.row_indices.numpy(), A_cpu.col_indices.numpy())),
        shape=(n, n),
    ).tocsr()
    x_ref, _ = spla.cg(A_sp, b_global.cpu().numpy(), rtol=1e-12, maxiter=2000)
    x_ref_t = torch.from_numpy(x_ref)
    x_full = x_dt.full_tensor().cpu()
    rel_to_scipy = (
        (x_ref_t - x_full).norm() / (x_ref_t.norm() + 1e-12)
    ).item()

    print(f"[rank {rank}] rel residual={rel_resid:.2e}  "
          f"rel diff vs scipy CG={rel_to_scipy:.2e}")

    if rank == 0:
        assert rel_resid < 1e-8 and rel_to_scipy < 1e-6, "distributed solve diverged"
        print("\nDistributed CG converged.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
