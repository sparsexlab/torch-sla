#!/usr/bin/env python
"""Scaling benchmark: spectral condition number (A.condition_number(ord=2)).

Uses sparse SVD internally -> CPU only (gated like svd).

Run::

    python benchmarks/scaling/ops/condition_number.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_EIG, SWEEP_EIG_QUICK, main_for  # noqa: E402
from setups import setup_condition_number  # noqa: E402

SPEC = OpSpec(
    name="condition number (spectral)",
    setup=setup_condition_number,
    backend="scipy/svds",
    png_name="condition_number",
    reps=2,
    avail=lambda dev: dev == "cpu",  # relies on sparse SVD (CPU only)
    sweep=SWEEP_EIG,
    sweep_quick=SWEEP_EIG_QUICK,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
