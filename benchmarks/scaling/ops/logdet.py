#!/usr/bin/env python
"""Scaling benchmark: log-determinant (Hutchinson stochastic estimator).

Run::

    python benchmarks/scaling/ops/logdet.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_MID, SWEEP_MID_QUICK, main_for  # noqa: E402
from setups import setup_logdet  # noqa: E402

SPEC = OpSpec(
    name="log-determinant (Hutchinson)",
    setup=setup_logdet,
    backend="hutchinson",
    png_name="logdet",
    reps=2,
    sweep=SWEEP_MID,
    sweep_quick=SWEEP_MID_QUICK,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
