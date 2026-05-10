"""High-level simulators for images and light curves.

Wraps the model -> PSF -> noise pipeline that was duplicated at the start of
every thesis notebook into a couple of reusable functions.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from ..light.psf import convolve_psf, gaussian_psf_kernel
from .noise import add_gaussian_noise


def simulate_image(
    model: nn.Module,
    xy: torch.Tensor,
    *,
    psf_fwhm: Optional[float] = None,
    deltapix: Optional[float] = None,
    psf_size: int = 21,
    noise_sigma: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate ``(clean, noisy_with_psf)`` images from a light/lens model.

    Parameters
    ----------
    model : nn.Module that returns a 2D flux map given ``xy``
    xy : (2, H, W) coordinate tensor (e.g. from ``coordinate_grid``)
    psf_fwhm, deltapix : if both provided, convolve with a Gaussian PSF of the
        given FWHM (arcsec) at the given pixel scale (arcsec/pixel)
    noise_sigma : Gaussian noise std added to the convolved image
    seed : if not None, sets the local RNG for reproducibility
    """
    with torch.no_grad():
        clean = model(xy)

    if seed is not None:
        gen = torch.Generator(device=clean.device)
        gen.manual_seed(seed)
    else:
        gen = None

    out = clean
    if psf_fwhm is not None and deltapix is not None:
        kernel = gaussian_psf_kernel(psf_fwhm, deltapix, size=psf_size).to(clean.device, clean.dtype)
        out = convolve_psf(out, kernel)
    if noise_sigma is not None:
        out = add_gaussian_noise(out, sigma=noise_sigma, generator=gen)
    return clean, out


def simulate_lightcurve(
    model: nn.Module,
    t: torch.Tensor,
    *,
    noise_sigma: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate a noisy ``(clean, noisy)`` magnification light curve."""
    with torch.no_grad():
        clean = model(t)
    if seed is not None:
        gen = torch.Generator(device=clean.device)
        gen.manual_seed(seed)
    else:
        gen = None
    if noise_sigma == 0.0:
        return clean, clean.clone()
    return clean, add_gaussian_noise(clean, sigma=noise_sigma, generator=gen)
