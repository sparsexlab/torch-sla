"""Distributed LOBPCG bench: v1 (pre-fix) vs v2 (this PR) on NCCL.

Probes whether the per-iter QR-vs-Python-CGS2 win compounds when
matvecs go through NCCL allgather. Each rank does the same
arithmetic; the only difference per outer iter is how many
matvec-columns get pushed through the collective.

Designed to run with ``torchrun --nproc-per-node=2`` on a single GPU
machine (e.g. tb16, RTX 5060 8GB). Set:

    export NCCL_P2P_DISABLE=1   # bypass NCCL's same-GPU P2P guard
    export NCCL_SHM_DISABLE=0
    export NCCL_IB_DISABLE=1

These force collectives through shared memory + IPC. **CAVEAT**: this
is NOT representative of true multi-GPU NCCL where collectives go over
NVLink/PCIe -- absolute numbers are biased low. The v1-vs-v2 *ratio*
is still meaningful because both variants pay the same collective
overhead per matvec.

Within one process group lifetime we A/B-test both variants by
monkey-patching ``_cgs2_inplace`` to either the Python-loop CGS2 (v1)
or the LAPACK-QR (v2, the shipped version on this branch).

Usage::

    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_IB_DISABLE=1 \\
        torchrun --nproc-per-node=2 \\
        tests/lobpcg/bench_distributed.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..")))


# --------------------------------------------------------------------- #
# v1 orth: hand-rolled Python-loop CGS2 (what shipped in v0.3.0)
# --------------------------------------------------------------------- #
def _cgs2_python_loop(Z: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(Z.dtype).eps * 100
    for _ in range(2):
        for j in range(Z.shape[1]):
            if j > 0:
                coeff = Z[:, :j].T @ Z[:, j]
                Z[:, j] -= Z[:, :j] @ coeff
            nrm = Z[:, j].norm()
            if nrm > eps:
                Z[:, j] /= nrm
            else:
                Z[:, j].zero_()
    col_norms = Z.norm(dim=0)
    valid = col_norms > 0.5
    if not bool(valid.all()):
        Z = Z[:, valid]
    return Z


def make_banded_spd(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 4.0 + rng.uniform(0, 0.1)
        for offset in [1, 2, 5]:
            if i + offset < n:
                v = -rng.uniform(0.1, 1.0)
                A[i, i + offset] = v
                A[i + offset, i] = v
    A = 0.5 * (A + A.T) + n * np.eye(n) * 0.5
    return torch.from_numpy(A)


def main():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size < 2:
        if rank == 0:
            print("ERROR: launch with torchrun --nproc-per-node>=2",
                  file=sys.stderr)
        return
    # Backend / device pick:
    #   - NCCL + CUDA when both are available (best fidelity to real
    #     multi-GPU distributed)
    #   - gloo + CPU otherwise (Windows PyTorch ships without NCCL;
    #     v1-vs-v2 ratio is the question of interest, and collective
    #     overhead applies equally to both variants either way)
    if dist.is_nccl_available() and torch.cuda.is_available():
        backend = "nccl"
        n_cuda = torch.cuda.device_count()
        dev_idx = local_rank % max(n_cuda, 1)
        torch.cuda.set_device(dev_idx)
        device = torch.device(f"cuda:{dev_idx}")
        mesh_device = "cuda"
    else:
        backend = "gloo"
        device = torch.device("cpu")
        mesh_device = "cpu"
    dist.init_process_group(backend=backend, init_method="env://",
                            rank=rank, world_size=world_size)

    try:
        from torch_sla import SparseTensor
        import torch_sla.sparse_tensor.linalg as linalg_module

        if rank == 0:
            print(f"world_size={world_size}, device={device}, backend={backend}")
            if device.type == "cuda":
                print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            print()
            print(f"{'n':>5s}  {'variant':18s}  {'time_ms':>9s} {'max_err':>10s}")

        # Use partition_for_rank (the same API the existing
        # test_dsparse_eigsh_multiprocess.py uses) so we don't
        # need a DeviceMesh and so we work on backends that don't
        # expose NCCL (Windows native PyTorch).
        sizes = [200, 400, 700, 1000]
        k = 6
        original_orth = linalg_module._cgs2_inplace

        for n in sizes:
            A_dense = make_banded_spd(n)
            idx = A_dense.nonzero().T
            vals = A_dense[idx[0], idx[1]]
            A_st = SparseTensor(vals.to(device).to(torch.float64),
                                idx[0].to(device),
                                idx[1].to(device),
                                shape=(n, n))
            gt = sorted(np.linalg.eigvalsh(A_dense.numpy()), reverse=True)[:k]
            D = A_st.partition_for_rank(rank=rank, world_size=world_size)

            for label, orth_fn in [("v1 (Py CGS2)", _cgs2_python_loop),
                                    ("v2 (LAPACK QR)", original_orth)]:
                linalg_module._cgs2_inplace = orth_fn
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dist.barrier()
                t0 = time.perf_counter()
                vals_d, _ = D.eigsh(k=k, which="LM", maxiter=300, tol=1e-8)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dist.barrier()
                t = time.perf_counter() - t0
                err = max(abs(g - e) for g, e in zip(gt, vals_d.tolist()))
                if rank == 0:
                    print(f"{n:>5d}  {label:18s}  {t*1000:>9.2f} {err:>10.2e}")
            if rank == 0:
                print()

        linalg_module._cgs2_inplace = original_orth  # restore
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
