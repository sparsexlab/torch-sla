#!/usr/bin/env python
"""Scaling benchmark: direct LU linear solve (scipy/lu).

Emits ``lu_scaling.png`` (the name the docs reference).

Run::

    python benchmarks/scaling/ops/lu.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_DIRECT, SWEEP_DIRECT_QUICK, main_for  # noqa: E402
from setups import setup_solve_lu, verify_solve  # noqa: E402

SPEC = OpSpec(
    name="linear solve (LU)",
    setup=setup_solve_lu,
    backend="scipy/lu",
    png_name="lu",
    reps=2,
    sweep=SWEEP_DIRECT,
    sweep_quick=SWEEP_DIRECT_QUICK,
    verify=verify_solve("scipy", method="lu"),
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
