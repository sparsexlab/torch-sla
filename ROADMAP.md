# Roadmap

Planned directions for torch-sla, roughly ordered by priority. Checkboxes track
what has actually landed — `[x]` done, `[ ]` pending.

_Last updated: 2026-06._

## 1. Complex-valued solves (with correct complex adjoint) — ✅ done

Support complex-valued sparse systems — in particular **complex-symmetric** matrices (`A = Aᵀ`, e.g. time-harmonic Helmholtz / PML / impedance) solved via cuDSS's LDLᵀ, and **Hermitian** matrices (`A = Aᴴ`) via LDLᴴ. This is what unblocks complex FEM in TensorMesh (Helmholtz / PML / metamaterial topology optimization).

- [x] **cuDSS complex** — `complex64`/`complex128` in `_DTYPE_MAP`, `HERMITIAN`/`HPD` in `_MTYPE_MAP` (`backends/nvmath_backend.py`).
- [x] **complex adjoint** — `autograd.py` uses `grad_val = -grad_b[row] * u[col].conj()` and solves `Aᴴ` per matrix type (general/LU; complex-symmetric LDLᵀ via the conjugation trick; Hermitian LDLᴴ). `torch.autograd.gradcheck` passes on complex inputs.
- [x] **complex across backends** — scipy (CPU direct + iterative), cuDSS (GPU direct), pytorch-native (CG / BiCGStab / GMRES / MINRES, relerr ~1e-13).
- [x] **complex `det` / `eigsh`** — missing-conjugate bugs in the det & eigsh adjoints fixed; verified fwd+bwd on CPU + CUDA + ROCm.
- [ ] **TensorMesh end-to-end complex Helmholtz example** — downstream; the FEM assembly stack still needs a few real-dtype assumptions unblocked (see item 2 of the [TensorMesh ROADMAP](https://github.com/camlab-ethz/TensorMesh)).

## 2. PETSc backend — ⏸ deprioritised (superseded by STRUMPACK + AMG)

Originally planned for PETSc's rich **preconditioner** ecosystem (GAMG, hypre BoomerAMG, ILU/ICC, block Jacobi, ASM, fieldsplit, …) and full Krylov suite, plus its complete **HIP/ROCm** backend (the `Mat` class was ported to HIP by AMD in 2021–2022 on hipBLAS/hipSPARSE/hipSOLVER).

Deprioritised after assessment: `petsc4py` is Linux/MPI-only and a ROCm-enabled PETSc + matching `petsc4py` is an HPC-cluster build, not a `pip install` — at odds with torch-sla's cross-platform, pip-installable positioning. Most of the original need is now covered another way (see what landed below). Revisit only if the GPU preconditioner ecosystem (GAMG / hypre BoomerAMG on AMD) is specifically required.

- [x] **STRUMPACK backend** (the portable GPU *direct* path PETSc was partly wanted for) — multifrontal LU / Cholesky / LDLt, real + complex, **CPU / CUDA / ROCm**; autograd `Aᴴ` adjoint. Gives AMD users a GPU direct solve without a hand-written rocSPARSE binding.
- [x] **pytorch-native preconditioners beyond Jacobi** — ILU(0), additive Schwarz (ASM), block-Jacobi, SSOR, polynomial, IC0 — closing much of the preconditioner gap on-device.
- [ ] **PETSc backend itself** — not pursued for now (rationale above).

## 3. Multi-GPU linear solvers + TensorMesh multi-GPU assembly

Wire up torch-sla's multi-GPU / distributed linear solvers (`torch_sla/distributed/`, `DSparseTensor`) with TensorMesh's multi-GPU assembly, so a domain-decomposed FEM problem can be assembled and solved across multiple GPUs end-to-end.

- [x] **distributed Krylov** — `cg` / `bicgstab` / `gmres` / `fgmres` / `minres`; multiprocess tests match scipy.
- [x] **per-rank preconditioners** — block-Jacobi + AMG (PyAMG / torch-amgx).
- [x] **comm/compute overlap** — async P2P (`batch_isend_irecv` on NCCL, send/recv on gloo).
- [x] **scale** — benchmarked to **400M DOF on 4 GPUs**.
- [ ] **TensorMesh end-to-end** — cross-repo: wire into TensorMesh's multi-GPU FEM assembly for an end-to-end domain-decomposed assemble + solve.

## 4. Ginkgo backend — 🔭 planned

Add a [Ginkgo](https://ginkgo-project.github.io/) backend as a **lighter-weight** alternative to PETSc, primarily for its **batched** solver support. Ginkgo is a single-source, performance-portable C++ library spanning CUDA (NVIDIA), HIP (AMD), and SYCL (Intel) — one backend covering all three vendors, matching torch-sla's GPU-agnostic positioning.

- [ ] **batched solvers** (`batch::Csr`, batched Krylov across CUDA/HIP/SYCL) — the main draw, for the "many small systems" regime.
- [ ] **cross-vendor portability** from one codebase (AMD + Intel + NVIDIA).
- [ ] integration via [pyGinkgo](https://arxiv.org/pdf/2510.08230).

Note: a first-party **batched same-pattern CG** (vectorized scatter matvec, device-agnostic CPU/CUDA/ROCm) already landed in the pytorch backend, so the immediate batched-SPD need is partly met without Ginkgo; Ginkgo would broaden this to more methods + Intel.

References: [Ginkgo](https://ginkgo-project.github.io/), [pyGinkgo](https://arxiv.org/pdf/2510.08230).

## ✅ Delivered since this roadmap was written (not originally listed)

- [x] **STRUMPACK backend** — portable sparse direct (CPU/CUDA/ROCm), real + complex (see item 2).
- [x] **GMRES + MINRES** (pytorch-native) — restarted right-preconditioned GMRES; Paige–Saunders MINRES with scipy-style stopping.
- [x] **LSQR + LSMR** least-squares (pytorch-native, device-agnostic) — verified vs scipy on CPU + CUDA.
- [x] **differentiable nonlinear solve** — Newton + implicit-function-theorem adjoint; validated against the analytical 1D Bratu solution and `scipy.optimize.root`.
- [x] **batched same-pattern CG** — truly vectorized over a batch of matrices sharing a sparsity pattern (no Python loop); device-agnostic.
- [x] **GPU graph ops** — `connected_components` reimplemented as parallel pure-torch (device-staying, batched).
- [x] **removed `eigen` C++ backend** (redundant with scipy) and **removed `cupy` backend** (lu→cudss, iterative+lstsq→pytorch).
