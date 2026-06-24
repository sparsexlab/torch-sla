#!/usr/bin/env python3
"""Inverse coefficient learning on the variable-coefficient Poisson equation.

We discretize  -div(kappa(x) grad u) = f  on (0,1)^2 with u = 0 on the
boundary using a 5-point finite-volume / cell-centered stencil with
arithmetic-mean face conductivities. The matrix A(kappa) is assembled
as a torch-sla SparseTensor whose values depend differentiably on the
nodal kappa field.

Pipeline:
  1. Build kappa* = 1 + 0.5 sin(2 pi x) sin(2 pi y) on an n x n grid.
  2. Forward-solve A(kappa*) u_obs = f to obtain noisy-free observations.
  3. Initialize kappa = ones(n, n) as a torch.nn.Parameter and run Adam
     on || u_pred - u_obs ||^2, where u_pred = A(kappa).solve(f).
  4. Plot loss curve, kappa*, recovered kappa, pointwise error.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torch_sla import SparseTensor

warnings.filterwarnings("ignore", message="PCG did not converge")
warnings.filterwarnings("ignore", message="Sparse CSR tensor support")


def make_kappa_star(n, device, dtype):
    xs = torch.linspace(0, 1, n, device=device, dtype=dtype)
    ys = torch.linspace(0, 1, n, device=device, dtype=dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return 1.0 + 0.5 * torch.sin(2 * torch.pi * X) * torch.sin(2 * torch.pi * Y)


def precompute_indices(n, device):
    """Pre-compute COO row/col index pattern (independent of kappa values)."""
    m = n - 2
    ii, jj = torch.meshgrid(
        torch.arange(m, device=device),
        torch.arange(m, device=device),
        indexing="ij",
    )
    flat = (ii * m + jj).flatten()

    east_mask = (jj < m - 1).flatten()
    south_mask = (ii < m - 1).flatten()
    east_self = flat[east_mask]
    east_neigh = flat[east_mask] + 1
    south_self = flat[south_mask]
    south_neigh = flat[south_mask] + m

    rows = torch.cat([flat, east_self, east_neigh, south_self, south_neigh])
    cols = torch.cat([flat, east_neigh, east_self, south_neigh, south_self])
    n_diag = flat.numel()
    n_east = east_self.numel()
    n_south = south_self.numel()
    return {
        "rows": rows.long(),
        "cols": cols.long(),
        "east_mask": east_mask,
        "south_mask": south_mask,
        "n_diag": n_diag,
        "n_east": n_east,
        "n_south": n_south,
        "m": m,
        "shape": (m * m, m * m),
    }


def assemble_values(kappa, h, idx):
    """Vectorized assembly of A(kappa) values with autograd.

    Stencil: face conductivity = arithmetic mean of adjacent nodal kappas.
    """
    inv_h2 = 1.0 / (h * h)
    k_C = kappa[1:-1, 1:-1]
    k_E_face = (k_C + kappa[1:-1, 2:]) * 0.5
    k_W_face = (k_C + kappa[1:-1, :-2]) * 0.5
    k_N_face = (k_C + kappa[:-2, 1:-1]) * 0.5
    k_S_face = (k_C + kappa[2:, 1:-1]) * 0.5

    diag_vals = (k_E_face + k_W_face + k_N_face + k_S_face) * inv_h2
    east_off = (-k_E_face * inv_h2).flatten()[idx["east_mask"]]
    south_off = (-k_S_face * inv_h2).flatten()[idx["south_mask"]]

    vals = torch.cat([
        diag_vals.flatten(),
        east_off, east_off,    # east + symmetric west
        south_off, south_off,  # south + symmetric north
    ])
    return vals


def solve_with_kappa(kappa, f, h, idx, backend, method, atol, maxiter):
    val = assemble_values(kappa, h, idx)
    A = SparseTensor(val, idx["rows"], idx["cols"], idx["shape"])
    return A.solve(f, backend=backend, method=method,
                   atol=atol, maxiter=maxiter)


def relative_l2(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64,
                        help="Grid side length; problem is (n-2)^2 unknowns")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--backend", type=str, default="pytorch",
                        help="solve backend (auto/scipy/cudss/pytorch)")
    parser.add_argument("--method", type=str, default="cg")
    parser.add_argument("--atol", type=float, default=1e-10,
                        help="Forward-solve tolerance during training")
    parser.add_argument("--maxiter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reg", type=float, default=1e-4,
                        help="Tikhonov coefficient on the discrete gradient "
                             "of kappa (smoothness regularizer)")
    parser.add_argument("--parameterization", type=str, default="softplus",
                        choices=["direct", "softplus"],
                        help="'direct': kappa = theta. 'softplus': "
                             "kappa = softplus(theta) so kappa > 0 always.")
    parser.add_argument("--out", type=str,
                        default="results/benchmark_inverse_coefficient")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"float64": torch.float64, "float32": torch.float32}[args.dtype]
    n = args.n
    h = 1.0 / (n - 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] n={n} (m={n-2}, unknowns={(n-2)**2}) "
          f"device={device} dtype={dtype} backend={args.backend} "
          f"method={args.method}", flush=True)

    idx = precompute_indices(n, device)
    f = torch.ones(idx["m"] * idx["m"], device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Generate observed data with ground-truth kappa*
    # ------------------------------------------------------------------
    kappa_star = make_kappa_star(n, device, dtype)
    print(f"[data] kappa* range: [{float(kappa_star.min()):.3f}, "
          f"{float(kappa_star.max()):.3f}]", flush=True)
    with torch.no_grad():
        u_obs = solve_with_kappa(
            kappa_star, f, h, idx,
            backend=args.backend, method=args.method,
            atol=1e-12, maxiter=args.maxiter,
        )
    u_obs_norm = float(u_obs.norm())
    print(f"[data] |u_obs|_2 = {u_obs_norm:.4e}", flush=True)

    # ------------------------------------------------------------------
    # Initialize parameter and run Adam
    # ------------------------------------------------------------------
    if args.parameterization == "softplus":
        # kappa = softplus(theta); init theta s.t. kappa(0) ≈ 1 → theta ≈ ln(e-1)
        theta_init = torch.full((n, n), float(np.log(np.e - 1.0)),
                                device=device, dtype=dtype)
        theta = nn.Parameter(theta_init.clone())

        def kappa_of():
            return torch.nn.functional.softplus(theta)
    else:
        theta = nn.Parameter(torch.ones(n, n, device=device, dtype=dtype))

        def kappa_of():
            return theta

    optimizer = torch.optim.Adam([theta], lr=args.lr)

    def smooth_reg(k):
        # Sum of squared first differences in x and y (interior contribution)
        dx = k[:, 1:] - k[:, :-1]
        dy = k[1:, :] - k[:-1, :]
        return (dx.pow(2).sum() + dy.pow(2).sum()) / k.numel()

    history = {"step": [], "loss": [], "data_loss": [], "reg_loss": [],
               "kappa_rel_err": [], "u_rel_err": []}
    t0 = time.perf_counter()

    for step in range(args.steps):
        optimizer.zero_grad()
        kappa = kappa_of()
        u_pred = solve_with_kappa(
            kappa, f, h, idx,
            backend=args.backend, method=args.method,
            atol=args.atol, maxiter=args.maxiter,
        )
        data_loss = (u_pred - u_obs).pow(2).sum()
        reg_loss = args.reg * smooth_reg(kappa) if args.reg > 0 else \
            torch.zeros((), device=device, dtype=dtype)
        loss = data_loss + reg_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            kappa_curr = kappa_of()
            k_err = relative_l2(kappa_curr, kappa_star)
            u_err = relative_l2(u_pred, u_obs)
        history["step"].append(step)
        history["loss"].append(float(loss.detach()))
        history["data_loss"].append(float(data_loss.detach()))
        history["reg_loss"].append(float(reg_loss.detach()))
        history["kappa_rel_err"].append(k_err)
        history["u_rel_err"].append(u_err)

        if step % 100 == 0 or step == args.steps - 1:
            print(f"  step {step:5d}  data={float(data_loss):.4e}  "
                  f"reg={float(reg_loss):.4e}  "
                  f"kappa_rel_err={k_err:.4e}  u_rel_err={u_err:.4e}",
                  flush=True)

    wall = time.perf_counter() - t0
    print(f"[train] {args.steps} Adam steps in {wall:.2f} s "
          f"({wall*1000/args.steps:.2f} ms/step)", flush=True)

    # ------------------------------------------------------------------
    # Final evaluation with tight tolerance
    # ------------------------------------------------------------------
    with torch.no_grad():
        kappa_final = kappa_of().detach()
        u_final = solve_with_kappa(
            kappa_of(), f, h, idx,
            backend=args.backend, method=args.method,
            atol=1e-12, maxiter=args.maxiter,
        )
    final = {
        "kappa_rel_err": relative_l2(kappa_final, kappa_star),
        "u_rel_err": relative_l2(u_final, u_obs),
        "kappa_min": float(kappa_final.min()),
        "kappa_max": float(kappa_final.max()),
        "wall_seconds": wall,
        "ms_per_step": wall * 1000 / args.steps,
    }
    print(f"[final] kappa_rel_err={final['kappa_rel_err']:.4e}  "
          f"u_rel_err={final['u_rel_err']:.4e}  "
          f"kappa_range=[{final['kappa_min']:.3f}, {final['kappa_max']:.3f}]",
          flush=True)

    # Save numbers
    payload = {
        "config": vars(args),
        "kappa_star_range": [float(kappa_star.min()), float(kappa_star.max())],
        "history": history,
        "final": final,
    }
    with open(out_dir / "results.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    # Save kappa fields for reproducibility
    np.save(out_dir / "kappa_star.npy", kappa_star.cpu().numpy())
    np.save(out_dir / "kappa_recovered.npy", kappa_final.cpu().numpy())

    # ------------------------------------------------------------------
    # Plot 4-panel figure
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kappa_np = kappa_final.cpu().numpy()
        kappa_star_np = kappa_star.cpu().numpy()
        err_np = np.abs(kappa_np - kappa_star_np)

        fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))

        ax = axes[0]
        ax.semilogy(history["step"], history["loss"], color="#1f77b4",
                    label="loss")
        ax.semilogy(history["step"], history["kappa_rel_err"],
                    color="#d62728", label=r"$\|\kappa-\kappa^*\|/\|\kappa^*\|$")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("value (log)")
        ax.set_title("Convergence")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

        vmin = min(kappa_np.min(), kappa_star_np.min())
        vmax = max(kappa_np.max(), kappa_star_np.max())
        for ax_, data, title in [
            (axes[1], kappa_star_np, r"Ground truth $\kappa^*$"),
            (axes[2], kappa_np, r"Recovered $\kappa$"),
        ]:
            im = ax_.imshow(data, origin="lower", extent=[0, 1, 0, 1],
                            vmin=vmin, vmax=vmax, cmap="viridis")
            ax_.set_title(title)
            ax_.set_xlabel("x")
            ax_.set_ylabel("y")
            plt.colorbar(im, ax=ax_, fraction=0.046, pad=0.04)

        im = axes[3].imshow(err_np, origin="lower", extent=[0, 1, 0, 1],
                            cmap="magma")
        axes[3].set_title(r"$|\kappa - \kappa^*|$")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("y")
        plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(out_dir / "inverse_problem.png", dpi=150)
        fig.savefig(out_dir / "inverse_problem.pdf")
        print(f"[plot] -> {out_dir/'inverse_problem.png'}", flush=True)
    except ImportError:
        print("[plot] matplotlib not available; skipping figure", flush=True)

    print(f"\n[done] -> {out_dir/'results.json'}", flush=True)


if __name__ == "__main__":
    main()
