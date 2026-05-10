"""CLI: fit a Sérsic profile to a noisy synthetic galaxy image.

Usage::

    python scripts/run_sersic_fit.py --npix 200 --deltapix 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
from torch import nn

import lensing as gl


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--npix", type=int, default=128)
    p.add_argument("--deltapix", type=float, default=0.05)
    p.add_argument("--psf-fwhm", type=float, default=0.10)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    gl.config.setup(seed=args.seed, device="cpu")

    xy = gl.data.coordinate_grid(npix=args.npix, deltapix=args.deltapix)
    truth = dict(Ie=5.0, Re=1.0, n=4.0, x0=0.1, y0=-0.2, e1=0.20, e2=-0.10)
    galaxy = gl.light.Sersic(**truth)
    _, image = gl.data.simulate_image(
        galaxy, xy,
        psf_fwhm=args.psf_fwhm, deltapix=args.deltapix, psf_size=21,
        noise_sigma=args.sigma, seed=args.seed,
    )

    class SersicPSF(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = gl.light.Sersic(Ie=2.0, Re=1.5, n=2.5, x0=0.0, y0=0.0, e1=0.0, e2=0.0)

        def forward(self, xy):
            k = gl.light.gaussian_psf_kernel(args.psf_fwhm, args.deltapix, size=21)
            return gl.light.convolve_psf(self.g(xy), k)

    model = SersicPSF()
    loss_fn = gl.inference.ReducedChiSquared(sigma=args.sigma, n_params=7)
    result = gl.inference.fit(
        model, xy, image, loss_fn,
        lr=args.lr, epochs=args.epochs,
        scheduler=gl.inference.optimize.reduce_lr_on_plateau(patience=200, factor=0.7),
        grad_clip=10.0, lbfgs_polish=True,
        log_every=max(1, args.epochs // 10),
    )

    print(f"\nfinal chi2/dof = {result.best_loss:.4f} in {result.duration_s:.2f}s")
    for k, v in truth.items():
        fit_v = float(getattr(model.g, k))
        print(f"  {k:>3s}: truth = {v:+.4f}   fit = {fit_v:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
