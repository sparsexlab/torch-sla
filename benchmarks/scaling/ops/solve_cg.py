#!/usr/bin/env python
"""Scaling benchmark: linear solve via conjugate gradient (pytorch/cg).

Emits ``cg_scaling.png`` (the name the docs reference).

Run::

    python benchmarks/scaling/ops/solve_cg.py
    python benchmarks/scaling/ops/solve_cg.py --device cuda
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_MID, SWEEP_MID_QUICK, main_for  # noqa: E402
from setups import setup_solve_cg, setup_solve_backward, verify_solve  # noqa: E402

SPEC = OpSpec(
    name="linear solve (conjugate gradient)",
    setup=setup_solve_cg,
    backend="pytorch/cg",
    png_name="cg",
    reps=2,
    sweep=SWEEP_MID,
    sweep_quick=SWEEP_MID_QUICK,
    verify=verify_solve("pytorch", method="cg", is_spd=True, tol=1e-8, maxiter=20000),
    backward_setup=setup_solve_backward,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
