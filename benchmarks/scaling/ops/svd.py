#!/usr/bin/env python
"""Scaling benchmark: truncated SVD (A.svd(k=6)).

CPU only -- SparseTensor.svd() on CUDA raises NotImplementedError (no CUDA-native
sparse SVD), so this op is gated to device == 'cpu'.

Run::

    python benchmarks/scaling/ops/svd.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import OpSpec, SWEEP_EIG, SWEEP_EIG_QUICK, main_for  # noqa: E402
from setups import setup_svd, setup_svd_backward, _svd_check  # noqa: E402

SPEC = OpSpec(
    name="truncated SVD (k=6)",
    setup=setup_svd,
    backend="scipy/svds",
    png_name="svd",
    reps=2,
    avail=lambda dev: dev == "cpu",  # CUDA sparse SVD not implemented
    sweep=SWEEP_EIG,
    sweep_quick=SWEEP_EIG_QUICK,
    verify=_svd_check,
    backward_setup=setup_svd_backward,
)

main = main_for(SPEC)

if __name__ == "__main__":
    main()
