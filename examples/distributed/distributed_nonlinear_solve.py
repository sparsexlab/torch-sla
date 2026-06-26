#!/usr/bin/env python
"""Distributed nonlinear solve via ``DSparseTensor.nonlinear_solve``.

Solves a 1-D Bratu-type semilinear system ``F(u) = A u - lam * exp(u) = 0``
in row-sharded Shard(0) space. ``A`` is the SPD tridiagonal operator and
the nonlinearity is pointwise (diagonal Jacobian shift), exactly the
structure the distributed Newton solver expects. Each Newton step solves
``J du = -F`` with ``J v = A v + d * v``, ``d = -lam * exp(u)``, via
distributed GMRES.

Correctness gate: the converged ``u*`` is gathered to a global vector and
the global residual ``‖F(u*)‖`` is recomputed from the full operator on
rank 0 and asserted small; we also cross-check against a single-process
Newton solve of the same global system.

Device-aware: uses NCCL + CUDA (one GPU per ``LOCAL_RANK``) when a GPU
is visible, else falls back to gloo + CPU so it still runs on a laptop.

Run::

    # single node (one box, N procs)
    torchrun --standalone --nproc_per_node=4 distributed_nonlinear_solve.py

    # multiple nodes (run on EVERY node; HEAD_NODE_IP reachable by all)
    torchrun --nnodes=2 --nproc_per_node=4 \
        --rdzv-id=sla --rdzv-backend=c10d \
        --rdzv-endpoint=HEAD_NODE_IP:29500 distributed_nonlinear_solve.py
"""

import os
import torch
import torch.distributed as dist

# Bratu nonlinearity strength. Small enough that Newton converges from
# u = 0 (the lower branch of the Bratu bifurcation).
LAM = 0.5


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
        print(f"{'=' * 60}\nDistributed nonlinear_solve: A u - lam*exp(u) = 0  "
              f"(world={world_size}, backend={backend}, device={mesh_device})"
              f"\n{'=' * 60}")

    from torch.distributed.device_mesh import init_device_mesh
    from torch_sla import DSparseTensor, SparseTensor
    from torch_sla.distributed import gather_owned_to_global

    n = 256
    A = SparseTensor.tridiagonal(n, diag=4.0, off_diag=-1.0).to(device)

    mesh = init_device_mesh(mesh_device, (world_size,))
    D = DSparseTensor.partition(A, mesh, partition_method="simple")

    # F(u) = A u - lam exp(u)   (owned slice). A u via the distributed
    # matvec (halo exchange happens inside _shard_matvec); the exp term is
    # pointwise on the owned slice.
    def residual_fn(u_owned, Dist):
        return Dist._shard_matvec(u_owned) - LAM * torch.exp(u_owned)

    # Diagonal Jacobian shift d such that J v = A v + d v, i.e.
    # d = -d/du(lam exp(u)) = -lam exp(u).
    def jac_diag_fn(u_owned, Dist):
        return -LAM * torch.exp(u_owned)

    num_owned = int(D._spec.placement.partition.owned_nodes.numel())
    u0 = torch.zeros(num_owned, dtype=torch.float64, device=device)

    # Distributed Newton -- collective; every rank participates.
    u_owned = D.nonlinear_solve(
        residual_fn, u0, jac_diag_fn=jac_diag_fn,
        tol=1e-10, atol=1e-12, max_iter=50)

    # Global residual norm at the converged solution, in Shard(0) space.
    F_owned = residual_fn(u_owned, D)
    global_resid = float(D._shard_norm(F_owned).item())

    # Assemble the global solution for the rank-0 cross-check.
    owned = D._spec.placement.partition.owned_nodes.to(
        device=device, dtype=torch.int64)
    u_global = gather_owned_to_global(owned, u_owned, n).cpu()

    print(f"[rank {rank}] owned={num_owned} "
          f"global ||F(u*)||={global_resid:.2e}")

    if rank == 0:
        # Single-process reference Newton on the full global system using
        # a dense operator (n is small) -- independent of the distributed
        # code path.
        import scipy.sparse as sp
        A_cpu = A.to("cpu")
        A_sp = sp.coo_matrix(
            (A_cpu.values.numpy(),
             (A_cpu.row_indices.numpy(), A_cpu.col_indices.numpy())),
            shape=(n, n),
        ).tocsr()
        A_dense = torch.from_numpy(A_sp.toarray())

        u_ref = torch.zeros(n, dtype=torch.float64)
        for _ in range(50):
            F = A_dense @ u_ref - LAM * torch.exp(u_ref)
            if float(F.norm()) < 1e-12:
                break
            J = A_dense - torch.diag(LAM * torch.exp(u_ref))
            u_ref = u_ref - torch.linalg.solve(J, F)
        ref_resid = float((A_dense @ u_ref - LAM * torch.exp(u_ref)).norm())
        rel_to_ref = float((u_global - u_ref).norm() / (u_ref.norm() + 1e-12))

        print(f"\n     distributed ||F(u*)|| = {global_resid:.2e}")
        print(f"     reference   ||F(u*)|| = {ref_resid:.2e}")
        print(f"     ||u_dist - u_ref|| / ||u_ref|| = {rel_to_ref:.2e}")

        assert global_resid < 1e-8, (
            f"distributed Newton residual too large ({global_resid:.2e})")
        assert rel_to_ref < 1e-6, (
            f"distributed solution disagrees with reference ({rel_to_ref:.2e})")
        print("\nDistributed nonlinear_solve converged.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
