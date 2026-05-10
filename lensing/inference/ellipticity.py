"""Weak-lensing ellipticity estimators.

Two ways to recover the ellipticity from a galaxy image:

1. **Parametric fit**: fit a Sérsic + PSF and read off ``e1, e2``. This is
   what the thesis ``weak.ipynb`` does. We expose it as ``fit_ellipticity``.

2. **Quadrupole moments (Kaiser-Squires-like)**: weight the second moment of
   the image with a Gaussian aperture and form the polarization. Cheap,
   non-parametric, but biased on truncated profiles. We provide it as a
   sanity-check baseline.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from ..light.psf import convolve_psf, gaussian_psf_kernel
from ..light.sersic import Sersic
from ..utils.parameters import e1e2_to_q_pa
from .losses import ReducedChiSquared
from .optimize import OptimizeResult, fit, reduce_lr_on_plateau


class _SersicPSF(nn.Module):
    """A Sérsic light model convolved with a Gaussian PSF on every forward.

    Convolution-aware fitting (PSF is part of the loss) gives correct
    ellipticities even for marginally-resolved galaxies.
    """

    def __init__(self, sersic: Sersic, psf_fwhm: float, deltapix: float, psf_size: int = 21):
        super().__init__()
        self.sersic = sersic
        self.deltapix = deltapix
        self.psf_size = psf_size
        # Make psf_fwhm a parameter only if the user wants to fit it; here we
        # keep it fixed by default (it is a property of the instrument).
        self.register_buffer("_psf_fwhm", torch.as_tensor(psf_fwhm, dtype=torch.get_default_dtype()))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        clean = self.sersic(xy)
        kernel = gaussian_psf_kernel(self._psf_fwhm, self.deltapix, size=self.psf_size).to(clean.device, clean.dtype)
        return convolve_psf(clean, kernel)


def fit_ellipticity(
    image: torch.Tensor,
    xy: torch.Tensor,
    *,
    psf_fwhm: float,
    deltapix: float,
    sigma: float,
    init: Optional[Dict[str, float]] = None,
    epochs: int = 5000,
    lr: float = 0.05,
    psf_size: int = 21,
) -> Tuple[OptimizeResult, Dict[str, float]]:
    """Fit a single-Sérsic galaxy with PSF and return ``(fit_result, summary)``.

    The summary contains derived quantities ``q``, ``pa`` (radians),
    ellipticity magnitude ``|e|`` and ``g1, g2`` shear-style components.
    """
    init_params = {
        "Ie": 1.0,
        "Re": 1.0,
        "n": 4.0,
        "x0": float(xy[0].mean()),
        "y0": float(xy[1].mean()),
        "e1": 0.0,
        "e2": 0.0,
    }
    if init is not None:
        init_params.update(init)

    sersic = Sersic(**init_params)
    model = _SersicPSF(sersic=sersic, psf_fwhm=psf_fwhm, deltapix=deltapix, psf_size=psf_size)

    loss_fn = ReducedChiSquared(sigma=sigma, n_params=sum(1 for _ in model.parameters()))
    result = fit(
        model,
        forward_args=xy,
        target=image,
        loss_fn=loss_fn,
        lr=lr,
        epochs=epochs,
        scheduler=reduce_lr_on_plateau(),
        lbfgs_polish=True,
    )

    e1 = sersic.e1.detach()
    e2 = sersic.e2.detach()
    q, pa = e1e2_to_q_pa(e1, e2)
    summary = {
        "Ie": float(sersic.Ie.detach()),
        "Re": float(sersic.Re.detach()),
        "n": float(sersic.n.detach()),
        "x0": float(sersic.x0.detach()),
        "y0": float(sersic.y0.detach()),
        "e1": float(e1),
        "e2": float(e2),
        "q": float(q),
        "pa_rad": float(pa),
        "pa_deg": float(pa * 180.0 / torch.pi),
        "|e|": float(torch.sqrt(e1 ** 2 + e2 ** 2)),
    }
    return result, summary


def kaiser_squires_estimator(
    image: torch.Tensor,
    xy: torch.Tensor,
    weight_sigma: Optional[float] = None,
) -> Dict[str, float]:
    """Quadrupole-moment ellipticity (KSB-style, *no* PSF correction).

    Useful as a quick non-parametric baseline. Definitions follow Bartelmann &
    Schneider 2001, Sec. 4.2.
    """
    flux = torch.clamp(image, min=0.0)
    if weight_sigma is not None:
        x_c = (flux * xy[0]).sum() / flux.sum()
        y_c = (flux * xy[1]).sum() / flux.sum()
        r2 = (xy[0] - x_c) ** 2 + (xy[1] - y_c) ** 2
        w = torch.exp(-r2 / (2.0 * weight_sigma ** 2))
        flux = flux * w

    norm = flux.sum()
    x_c = (flux * xy[0]).sum() / norm
    y_c = (flux * xy[1]).sum() / norm
    dx, dy = xy[0] - x_c, xy[1] - y_c
    Q11 = (flux * dx * dx).sum() / norm
    Q22 = (flux * dy * dy).sum() / norm
    Q12 = (flux * dx * dy).sum() / norm
    denom = Q11 + Q22 + 2.0 * torch.sqrt(Q11 * Q22 - Q12 ** 2)
    e1 = (Q11 - Q22) / denom
    e2 = (2.0 * Q12) / denom
    q, pa = e1e2_to_q_pa(e1, e2)
    return {
        "x0": float(x_c),
        "y0": float(y_c),
        "e1": float(e1),
        "e2": float(e2),
        "|e|": float(torch.sqrt(e1 ** 2 + e2 ** 2)),
        "q": float(q),
        "pa_rad": float(pa),
        "pa_deg": float(pa * 180.0 / torch.pi),
    }
