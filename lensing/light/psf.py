"""Point spread functions and 2D convolution helpers.

Replaces the duplicated ``psf_kernel`` and ``Gaussian2DKernel`` calls scattered
across the thesis notebooks (sersic_final, weak, coresersic_torch).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


_FWHM2SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def gaussian_psf_kernel(
    fwhm_arcsec,
    deltapix: float,
    size: int = 31,
) -> torch.Tensor:
    """Return a normalized Gaussian PSF kernel of shape (1, 1, size, size).

    Differentiable in ``fwhm_arcsec`` so the PSF width can be optimized end to
    end (this is the trick used in ``THESIS/sersic/weak.ipynb``).
    """
    fwhm = torch.as_tensor(fwhm_arcsec, dtype=torch.get_default_dtype())
    sigma_pix = (fwhm / deltapix) * _FWHM2SIGMA
    half = size // 2
    grid = torch.arange(-half, half + 1, dtype=torch.get_default_dtype())
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma_pix ** 2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def moffat_psf_kernel(
    fwhm_arcsec: float,
    deltapix: float,
    beta: float = 2.5,
    size: int = 31,
) -> torch.Tensor:
    """Moffat PSF (closer to optical/IR PSFs than a pure Gaussian)."""
    fwhm = torch.as_tensor(fwhm_arcsec, dtype=torch.get_default_dtype())
    alpha = (fwhm / deltapix) / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    half = size // 2
    grid = torch.arange(-half, half + 1, dtype=torch.get_default_dtype())
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    r2 = x ** 2 + y ** 2
    kernel = (1.0 + r2 / alpha ** 2) ** (-beta)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def convolve_psf(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """2D convolution with reflective padding so flux is preserved at edges.

    ``image`` may be (H, W) or (B, 1, H, W); the helper handles both.
    """
    if image.ndim == 2:
        x = image.unsqueeze(0).unsqueeze(0)
        squeeze = True
    elif image.ndim == 3:
        x = image.unsqueeze(0)
        squeeze = True
    else:
        x = image
        squeeze = False

    k = kernel
    pad = (k.shape[-2] // 2, k.shape[-1] // 2)
    x = F.pad(x, (pad[1], pad[1], pad[0], pad[0]), mode="reflect")
    out = F.conv2d(x, k)
    return out.squeeze(0).squeeze(0) if squeeze else out
