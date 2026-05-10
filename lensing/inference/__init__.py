"""Inference pipeline: gradient-based fits, NUTS posteriors, weak-lensing helpers.

* :func:`fit` + :class:`OptimizeResult` — the workhorse training loop
  used by every parametric notebook: Adam with optional ReduceLROnPlateau
  scheduler, gradient clipping, automatic constraint projection on each
  step, and an opt-in L-BFGS polish at the end (1–2 orders of magnitude
  tighter loss for free on smooth problems).
* :class:`ReducedChiSquared`, :class:`GaussianNLL`, :func:`log_image_mse`
  — three loss functions covering the regimes encountered in lensing
  data: known per-pixel sigma, Gaussian likelihood, and high-dynamic-
  range galaxy images.
* :func:`run_nuts` — Pyro NUTS wrapper that returns a
  :class:`pandas.DataFrame` of posterior samples ready to feed into
  :mod:`lensing.viz.corner_plot`.
* :func:`fit_ellipticity`, :func:`kaiser_squires_estimator` — two
  weak-lensing shape estimators (parametric Sérsic + PSF vs.
  quadrupole-moment baseline).
"""
from .ellipticity import fit_ellipticity, kaiser_squires_estimator
from .losses import GaussianNLL, ReducedChiSquared, log_image_mse
from .mcmc import run_nuts
from .optimize import OptimizeResult, fit

__all__ = [
    "fit",
    "OptimizeResult",
    "ReducedChiSquared",
    "GaussianNLL",
    "log_image_mse",
    "run_nuts",
    "fit_ellipticity",
    "kaiser_squires_estimator",
]
