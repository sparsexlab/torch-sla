# Roadmap

Planned directions for torch-sla, roughly ordered by priority.

_Last updated: 2026-05._

## 1. Complex-valued solves (with correct complex adjoint)

Support complex-valued sparse systems — in particular **complex-symmetric** matrices (`A = Aᵀ`, e.g. time-harmonic Helmholtz / PML / impedance) solved via cuDSS's LDLᵀ, and **Hermitian** matrices (`A = Aᴴ`) via LDLᴴ. cuDSS supports both natively; torch-sla just hasn't exposed the path yet. Top priority — it is what unblocks complex FEM in TensorMesh (Helmholtz / PML / metamaterial topology optimization). Two parts:

Status today: the **scipy** backend already solves complex systems on CPU (direct `lu` + iterative, verified to machine precision) — a usable reference/validation path. The cuDSS work below is the high-performance **GPU** path. The `pytorch` (CG/BiCGStab use `torch.dot`, no conjugate; `<` ops break on complex) and `eigen` (C++ hard-codes `double`) backends don't support complex yet — separate gaps.

**(a) Expose complex in the cuDSS backend** — small:
- Add `complex64`/`complex128` to `_DTYPE_MAP` in `backends/nvmath_backend.py`.
- Add `HERMITIAN` / `HPD` entries to `_MTYPE_MAP` (only `general/symmetric/spd` today).

**(b) Fix the complex adjoint in the autograd backward** — small but **mathematically essential** (skipping it gives a correct forward solve but *silently wrong gradients*). `spsolve` is a custom `autograd.Function`, so PyTorch's complex-autograd rules stop at its `backward` boundary — the adjoint is hand-written and currently real-only. For `u = A⁻¹b` with a real loss, the complex adjoint (PyTorch's conjugate-gradient convention) is `λ = A⁻ᴴ ḡ`, `grad_b = λ`, `grad_A|ᵢⱼ = -λᵢ conj(uⱼ)`. Two corrections, both backward-compatible with the real path (`.conj()` is a no-op on real tensors):
- Conjugate `u` in the outer product: `gradval = -gradb[row] * u[col].conj()`.
- Use `Aᴴ` (not `Aᵀ`/`A`) for the adjoint solve, per matrix type:
  - **general / LU**: `Aᴴ = conj(A)ᵀ` → transpose indices **and** `val.conj()`.
  - **complex-symmetric (LDLᵀ)**: `Aᴴ = conj(A) ≠ A` → reuse the forward factorization via the conjugation trick `gradb = ldlt(indices, val, m, n, gradu.conj()).conj()`.
  - **Hermitian (LDLᴴ)**: `Aᴴ = A` → reuse the factorization directly with `gradu`.
- Validate with `torch.autograd.gradcheck` (supports complex inputs).

Downstream (TensorMesh repo): the FEM assembly stack needs a few real-dtype assumptions unblocked before an end-to-end complex Helmholtz example works; see item 2 of the [TensorMesh ROADMAP](https://github.com/camlab-ethz/TensorMesh).

## 2. PETSc backend

Add a [PETSc](https://petsc.org/) backend, primarily for its rich, industrial-grade **preconditioner** ecosystem (GAMG, hypre BoomerAMG, ILU/ICC, block Jacobi, ASM, fieldsplit, …) and its full Krylov suite (GMRES, FGMRES, MINRES, …) — filling the gap left by the pytorch-native backend, which today ships only CG/BiCGStab plus lightweight preconditioners.

Crucially, **PETSc has a complete HIP/ROCm backend** (the `Mat` class was ported to HIP by AMD in 2021–2022, on top of hipBLAS/hipSPARSE/hipSOLVER, with a Kokkos path as well). PETSc's GPU strength is on the **iterative + preconditioned** side — preconditioners such as GAMG and hypre BoomerAMG run on AMD GPUs — so this backend would give AMD users a complete **solver + preconditioner stack on GPU** without us writing a rocSPARSE binding ourselves. (GPU *direct* solves on AMD remain limited — STRUMPACK offers partial GPU offload; MUMPS/SuperLU_DIST are CPU/CUDA-oriented.)

Integration notes:
- `petsc4py` is the fast integration path (Python bindings already exist).
- Key engineering detail: zero-copy hand-off between torch GPU tensors and PETSc `Vec`/`Mat` on the same device (PETSc can wrap external GPU array pointers).
- Build caveat: a ROCm-enabled PETSc + matching `petsc4py` is an HPC-cluster build, not a `pip install`.

References: [PETSc GPU Support Roadmap](https://petsc.org/release/overview/gpu_roadmap/), [PETSc/TAO for Exascale](https://arxiv.org/html/2406.08646v2).

## 3. Multi-GPU linear solvers + TensorMesh multi-GPU assembly

Wire up torch-sla's multi-GPU / distributed linear solvers (see `torch_sla/distributed.py`, `DSparseTensor`) with TensorMesh's multi-GPU assembly, so a domain-decomposed FEM problem can be assembled and solved across multiple GPUs end-to-end. **Status: still debugging.**

## 4. Ginkgo backend

Add a [Ginkgo](https://ginkgo-project.github.io/) backend as a **lighter-weight** alternative to PETSc, primarily for its **batched** solver support. Ginkgo is a single-source, performance-portable C++ sparse linear algebra library spanning CUDA (NVIDIA), HIP (AMD), and SYCL (Intel) — one backend covering all three vendors, which directly serves torch-sla's GPU-agnostic positioning.

Its niche relative to PETSc is complementary, not competing, and aligns especially well with torch-sla's identity:
- **Batched solvers** (`batch::Csr`, batched Krylov across CUDA/HIP/SYCL) — the most mature library-level batched-solver story, matching the "many small systems" regime of ML/FEM workflows. This is the main draw.
- **Cross-vendor portability** from one codebase (AMD + Intel + NVIDIA).
- [pyGinkgo](https://arxiv.org/pdf/2510.08230) lowers the integration cost.

References: [Ginkgo project](https://ginkgo-project.github.io/), [pyGinkgo](https://arxiv.org/pdf/2510.08230).
