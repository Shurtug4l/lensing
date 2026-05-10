"""Noise models for synthetic lensing images.

Two regimes matter for ground-/space-based imaging:

* **Background-dominated** pixels (sky, read noise, dark current): the
  per-pixel error is approximately Gaussian with a constant std. Use
  :func:`add_gaussian_noise`.
* **Source-dominated** pixels (galaxy / arc cores, bright stars):
  Poisson statistics on the photon count dominate. Use
  :func:`add_poisson_noise`, optionally chained with
  :func:`add_gaussian_noise` to model both regimes simultaneously.

For typical optical / IR exposures the two noise sources combine into
the well-known *CCD equation* (Howell 2006); the helpers here let you
build that combination explicitly. Both functions accept an optional
``torch.Generator`` for **bit-for-bit reproducibility** — pass a
seeded generator if you need the same noise realisation across runs
(used heavily by :mod:`lensing.bigdata` and :mod:`lensing.ml.datasets`).
"""
from __future__ import annotations

import torch


def add_gaussian_noise(
    image: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add i.i.d. Gaussian noise of given std (in the flux units of ``image``).

    Parameters
    ----------
    image : tensor of any shape; noise is broadcast.
    sigma : standard deviation of the noise (same units as ``image``).
    generator : optional ``torch.Generator`` for reproducible noise.

    Returns
    -------
    A new tensor (no in-place modification) with the noise added.
    """
    noise = torch.randn(
        image.shape, generator=generator, dtype=image.dtype, device=image.device
    ) * sigma
    return image + noise


def add_poisson_noise(image: torch.Tensor, exposure: float = 1.0) -> torch.Tensor:
    """Add Poisson (shot) noise consistent with the given exposure factor.

    The implementation interprets ``image`` as a flux density, multiplies
    by ``exposure`` to get photon counts, applies Poisson, and divides
    back by ``exposure`` so units are preserved. Negative pixels are
    clipped to zero before the Poisson draw (no physical meaning of a
    negative count).

    Parameters
    ----------
    image : tensor in flux units.
    exposure : multiplier giving the equivalent photon count per pixel.
        Larger ``exposure`` → smaller relative noise.
    """
    counts = torch.clamp(image * exposure, min=0.0)
    return torch.poisson(counts) / exposure
