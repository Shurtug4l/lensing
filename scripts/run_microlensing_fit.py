"""CLI entry point: simulate a microlensing event and fit it.

Usage::

    python scripts/run_microlensing_fit.py \\
        --f 7 --y0 0.1 --t0 183 --tE 20 --sigma 0.5 \\
        --epochs 4000 --out fit_results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch

import lensing as gl


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--f", type=float, default=7.0)
    p.add_argument("--y0", type=float, default=0.1)
    p.add_argument("--t0", type=float, default=183.0)
    p.add_argument("--tE", type=float, default=20.0)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--ndays", type=int, default=365)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    gl.config.setup(seed=args.seed, device="cpu")

    truth = gl.lens.PaczynskiLightcurve(
        f=args.f, y0=args.y0, t0=args.t0, tE=args.tE,
    )
    t = torch.arange(0.0, float(args.ndays), 1.0)
    _, mag = gl.data.simulate_lightcurve(truth, t, noise_sigma=args.sigma, seed=args.seed)

    fit_model = gl.lens.PaczynskiLightcurve(
        f=2.0, y0=0.5, t0=float(args.ndays) / 3.0, tE=10.0,
    )
    loss_fn = gl.inference.ReducedChiSquared(sigma=args.sigma, n_params=4)
    result = gl.inference.fit(
        fit_model, t, mag, loss_fn,
        lr=args.lr, epochs=args.epochs,
        scheduler=gl.inference.optimize.reduce_lr_on_plateau(),
        lbfgs_polish=True, log_every=max(1, args.epochs // 10),
    )

    print(f"\nfinal chi2/dof = {result.best_loss:.4f} in {result.duration_s:.2f}s")
    print("recovered parameters:")
    for k, v in result.parameters.items():
        truth_v = getattr(truth, k).detach().item()
        print(f"  {k:>4s}: fit = {v:+.4f}   truth = {truth_v:+.4f}")

    if args.out:
        import pandas as pd

        df = pd.DataFrame([{**result.parameters, "best_loss": result.best_loss}])
        df.to_csv(args.out, index=False)
        print(f"results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
