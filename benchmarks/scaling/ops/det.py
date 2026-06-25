#!/usr/bin/env python
"""Scaling benchmark: determinant (A.det()).

Run::

    python benchmarks/scaling/ops/det.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_DIRECT, SWEEP_DIRECT_QUICK, main_for  # noqa: E402
from setups import setup_det  # noqa: E402

SPEC = OpSpec(
    name="determinant",
    setup=setup_det,
    backend="scipy",
    png_name="det",
    reps=2,
    sweep=SWEEP_DIRECT,
    sweep_quick=SWEEP_DIRECT_QUICK,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
