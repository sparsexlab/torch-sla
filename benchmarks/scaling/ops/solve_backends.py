#!/usr/bin/env python
"""Multi-backend comparison + precision plots for the linear solve Ax=b.

Answers two of the user's complaints: there was only ONE backend per plot and
only a speed figure, no accuracy figure. This runner overlays every linear-solve
backend available on the current device on a SHARED DOF sweep and emits TWO
figures:

* ``solve_backends_scaling.png`` -- time vs DOF, one line per backend.
* ``solve_precision.png``        -- relative residual ||Ax-b||/||b|| vs DOF, one
  line per backend, so the direct (~1e-12..1e-14) vs iterative (~1e-6) accuracy
  gap is VISIBLE.

Backends:
  scipy/lu        direct, CPU (values moved to CPU internally)
  cudss           direct, CUDA only (NVIDIA cuDSS)
  strumpack       direct (multifrontal), if STRUMPACK is built
  pytorch/cg      iterative, device-portable (SPD Laplacian)
  pyamg           algebraic multigrid, if PyAMG installed

Each backend is gated by its availability predicate; absent ones are skipped and
reported as not-run (never faked). A JSON sidecar records every (dof, time,
residual) and the device label.

Run::

    python benchmarks/scaling/ops/solve_backends.py
    python benchmarks/scaling/ops/solve_backends.py --device cuda
    python benchmarks/scaling/ops/solve_backends.py --quick --time-cap 8
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from harness import (  # noqa: E402
    ASSETS, BackendCase, _REPO, SWEEP_MID, SWEEP_MID_QUICK,
    add_common_args, build, device_info, plot_backend_precision,
    plot_backend_time, resolve_device, sweep_backends,
)
from setups import (  # noqa: E402
    setup_solve_cg, setup_solve_lu, setup_solve_strumpack, setup_solve_cudss,
    setup_solve_pyamg, verify_solve, is_strumpack_available, is_cudss_available,
    is_pyamg_available,
)

# DOF sweep shared by every backend (so the curves are directly comparable). The
# direct sweep is the binding one (fill-in heavy); CG/AMG go further but the cap
# keeps everything bounded.
SOLVE_SIDES = [16, 32, 48, 64, 96, 128, 192]
SOLVE_SIDES_QUICK = [16, 32, 48, 64]

CASES = [
    BackendCase(
        label="scipy/lu (direct)",
        setup=setup_solve_lu,
        residual=verify_solve("scipy", method="lu"),
    ),
    BackendCase(
        label="strumpack (direct)",
        setup=setup_solve_strumpack,
        residual=verify_solve("strumpack"),
        avail=lambda dev: is_strumpack_available(),
    ),
    BackendCase(
        label="cudss (direct)",
        setup=setup_solve_cudss,
        residual=verify_solve("cudss"),
        avail=lambda dev: dev == "cuda" and is_cudss_available(),
    ),
    BackendCase(
        label="pytorch/cg (iterative)",
        setup=setup_solve_cg,
        residual=verify_solve("pytorch", method="cg", is_spd=True,
                              tol=1e-8, maxiter=20000),
    ),
    BackendCase(
        label="pyamg (multigrid)",
        setup=setup_solve_pyamg,
        residual=verify_solve("pyamg", method="ruge_stuben"),
        avail=lambda dev: is_pyamg_available(),
    ),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args = ap.parse_args()

    device = resolve_device(args.device)
    info = device_info(device)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _REPO / out_dir
    sides = SOLVE_SIDES_QUICK if args.quick else SOLVE_SIDES

    print(f"device label: {info['device_kind']}: {info['processor']} | "
          f"{info['dtype']} | torch {info['torch']} | {info['compile']}",
          flush=True)
    results = sweep_backends(CASES, device, sides, time_cap=args.time_cap)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"device_label": info, "backends": results}
    jpath = out_dir / "solve_backends_scaling.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  [json] wrote {jpath}", flush=True)

    plot_backend_time(results, info, out_dir)
    plot_backend_precision(results, info, out_dir)

    print("\nbackends run:", ", ".join(results.keys()) or "(none)")


if __name__ == "__main__":
    main()
