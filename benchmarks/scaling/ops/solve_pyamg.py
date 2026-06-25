#!/usr/bin/env python
"""Scaling benchmark: linear solve via PyAMG (Ruge-Stuben AMG).

Skipped gracefully if PyAMG is not available.

Run::

    python benchmarks/scaling/ops/solve_pyamg.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_MID, SWEEP_MID_QUICK, main_for  # noqa: E402
from setups import (  # noqa: E402
    setup_solve_pyamg, verify_solve, is_pyamg_available,
)

SPEC = OpSpec(
    name="linear solve (PyAMG)",
    setup=setup_solve_pyamg,
    backend="pyamg/ruge_stuben",
    png_name="pyamg",
    reps=2,
    avail=lambda dev: is_pyamg_available(),
    sweep=SWEEP_MID,
    sweep_quick=SWEEP_MID_QUICK,
    verify=verify_solve("pyamg", method="ruge_stuben"),
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
