#!/usr/bin/env python
"""Scaling benchmark: nonlinear_solve (Newton, A @ u + u^3 = f).

Run::

    python benchmarks/scaling/ops/nonlinear_solve.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_MID, SWEEP_MID_QUICK, main_for  # noqa: E402
from setups import (  # noqa: E402
    setup_nonlinear_solve, setup_nonlinear_solve_backward, _nonlinear_residual_norm,
)

SPEC = OpSpec(
    name="nonlinear solve (Newton)",
    setup=setup_nonlinear_solve,
    backend="newton + pytorch/cg",
    png_name="nonlinear_solve",
    reps=2,
    sweep=SWEEP_MID,
    sweep_quick=SWEEP_MID_QUICK,
    verify=_nonlinear_residual_norm,
    backward_setup=setup_nonlinear_solve_backward,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
