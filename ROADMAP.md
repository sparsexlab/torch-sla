# Roadmap

Planned directions for torch-sla, roughly ordered by priority.

_Last updated: 2026-06._

## 1. Complex-valued solves (with correct complex adjoint) — ✅ LANDED

Support complex-valued sparse systems — in particular **complex-symmetric** matrices (`A = Aᵀ`, e.g. time-harmonic Helmholtz / PML / impedance) solved via cuDSS's LDLᵀ, and **Hermitian** matrices (`A = Aᴴ`) via LDLᴴ. This is what unblocks complex FEM in TensorMesh (Helmholtz / PML / metamaterial topology optimization).

**Status: done.** Both parts below shipped:
- **(a) cuDSS complex** — `complex64`/`complex128` in `_DTYPE_MAP` and `HERMITIAN`/`HPD` in `_MTYPE_MAP` (`backends/nvmath_backend.py`).
- **(b) complex adjoint** — `backends/.../autograd.py` uses `grad_val = -grad_b[row] * u[col].conj()` and solves `Aᴴ` per matrix type (general/LU, complex-symmetric LDLᵀ via the conjugation trick, Hermitian LDLᴴ). `torch.autograd.gradcheck` passes on complex inputs.
- Backends with complex support: **scipy** (CPU direct + iterative), **cuDSS** (GPU direct), **pytorch-native** (CG / BiCGStab / GMRES / MINRES — verified relerr ~1e-13). The `eigen` C++ backend (real-only) has since been removed.

Remaining: confirm complex paths for `det` / `eigs`. Downstream (TensorMesh repo): the FEM assembly stack still needs a few real-dtype assumptions unblocked before an end-to-end complex Helmholtz example works; see item 2 of the [TensorMesh ROADMAP](https://github.com/camlab-ethz/TensorMesh).

## 2. PETSc backend

Add a [PETSc](https://petsc.org/) backend, primarily for its rich, industrial-grade **preconditioner** ecosystem (GAMG, hypre BoomerAMG, ILU/ICC, block Jacobi, ASM, fieldsplit, …) and its full Krylov suite (GMRES, FGMRES, MINRES, …) — filling the gap left by the pytorch-native backend, which today ships only CG/BiCGStab plus lightweight preconditioners.

Crucially, **PETSc has a complete HIP/ROCm backend** (the `Mat` class was ported to HIP by AMD in 2021–2022, on top of hipBLAS/hipSPARSE/hipSOLVER, with a Kokkos path as well). PETSc's GPU strength is on the **iterative + preconditioned** side — preconditioners such as GAMG and hypre BoomerAMG run on AMD GPUs — so this backend would give AMD users a complete **solver + preconditioner stack on GPU** without us writing a rocSPARSE binding ourselves. (GPU *direct* solves on AMD remain limited — STRUMPACK offers partial GPU offload; MUMPS/SuperLU_DIST are CPU/CUDA-oriented.)

Integration notes:
- `petsc4py` is the fast integration path (Python bindings already exist).
- Key engineering detail: zero-copy hand-off between torch GPU tensors and PETSc `Vec`/`Mat` on the same device (PETSc can wrap external GPU array pointers).
- Build caveat: a ROCm-enabled PETSc + matching `petsc4py` is an HPC-cluster build, not a `pip install`.

References: [PETSc GPU Support Roadmap](https://petsc.org/release/overview/gpu_roadmap/), [PETSc/TAO for Exascale](https://arxiv.org/html/2406.08646v2).

## 3. Multi-GPU linear solvers + TensorMesh multi-GPU assembly

Wire up torch-sla's multi-GPU / distributed linear solvers (see `torch_sla/distributed/`, `DSparseTensor`) with TensorMesh's multi-GPU assembly, so a domain-decomposed FEM problem can be assembled and solved across multiple GPUs end-to-end.

**Status: torch-sla side done; TensorMesh end-to-end integration remaining.** Landed on the torch-sla side: distributed Krylov (`cg` / `bicgstab` / `gmres` / `fgmres` / `minres`, multiprocess tests match scipy), per-rank block-Jacobi + AMG preconditioners (PyAMG / torch-amgx), true comm/compute overlap via async P2P (`batch_isend_irecv` on NCCL, send/recv on gloo), benchmarked to **400M DOF on 4 GPUs**. The remaining work is the cross-repo piece: wiring this into TensorMesh's multi-GPU FEM assembly for an end-to-end domain-decomposed assemble+solve.

## 4. Ginkgo backend

Add a [Ginkgo](https://ginkgo-project.github.io/) backend as a **lighter-weight** alternative to PETSc, primarily for its **batched** solver support. Ginkgo is a single-source, performance-portable C++ sparse linear algebra library spanning CUDA (NVIDIA), HIP (AMD), and SYCL (Intel) — one backend covering all three vendors, which directly serves torch-sla's GPU-agnostic positioning.

Its niche relative to PETSc is complementary, not competing, and aligns especially well with torch-sla's identity:
- **Batched solvers** (`batch::Csr`, batched Krylov across CUDA/HIP/SYCL) — the most mature library-level batched-solver story, matching the "many small systems" regime of ML/FEM workflows. This is the main draw.
- **Cross-vendor portability** from one codebase (AMD + Intel + NVIDIA).
- [pyGinkgo](https://arxiv.org/pdf/2510.08230) lowers the integration cost.

References: [Ginkgo project](https://ginkgo-project.github.io/), [pyGinkgo](https://arxiv.org/pdf/2510.08230).
