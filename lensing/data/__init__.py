"""Synthetic-data utilities: coordinate grids, noise, image / light-curve simulators.

This sub-package collects everything needed to build a noisy
observation from a forward model:

* :func:`coordinate_grid` — square ``(2, npix, npix)`` arcsec grid
  centred on the optical axis; the *single* place every notebook gets
  its image-plane coordinates from.
* :func:`add_gaussian_noise`, :func:`add_poisson_noise` — i.i.d.
  Gaussian (background-dominated) and Poisson (source-dominated)
  noise, both reproducibility-friendly via a ``torch.Generator``.
* :func:`simulate_image`, :func:`simulate_lightcurve` — high-level
  one-call simulators: take a model, return ``(clean, noisy)`` tensors
  with optional PSF convolution.

For survey-scale (≥ 10⁴-sample) datasets, see :mod:`lensing.bigdata`,
which uses these primitives internally and persists the result as HDF5.
"""
from .grid import coordinate_grid
from .noise import add_gaussian_noise, add_poisson_noise
from .simulate import simulate_image, simulate_lightcurve

__all__ = [
    "coordinate_grid",
    "add_gaussian_noise",
    "add_poisson_noise",
    "simulate_image",
    "simulate_lightcurve",
]
