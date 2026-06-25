#!/usr/bin/env python
"""Scaling benchmark: batched linear solve (4 SPD systems via pytorch/cg).

Builds a [batch, M, N] SparseTensor (shared sparsity, per-batch values) and
solves a [batch, M] RHS. DOF on the x-axis is the per-system DOF (M).

Run::

    python benchmarks/scaling/ops/solve_batch.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_MID, SWEEP_MID_QUICK, main_for  # noqa: E402
from setups import setup_solve_batch, _solve_batch_check  # noqa: E402

SPEC = OpSpec(
    name="batched linear solve (4x, CG)",
    setup=setup_solve_batch,
    backend="pytorch/cg (batch=4)",
    png_name="solve_batch",
    reps=2,
    sweep=SWEEP_MID,
    sweep_quick=SWEEP_MID_QUICK,
    verify=_solve_batch_check,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
