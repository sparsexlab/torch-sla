#!/usr/bin/env python
"""Scaling benchmark: determinant backward / adjoint (A.det().backward()).

Run::

    python benchmarks/scaling/ops/det_backward.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_DIRECT, SWEEP_DIRECT_QUICK, main_for  # noqa: E402
from setups import setup_det_backward  # noqa: E402

SPEC = OpSpec(
    name="det backward (adjoint)",
    setup=setup_det_backward,
    backend="adjoint",
    png_name="det_backward",
    reps=2,
    sweep=SWEEP_DIRECT,
    sweep_quick=SWEEP_DIRECT_QUICK,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
