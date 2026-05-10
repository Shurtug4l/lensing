"""Navarro-Frenk-White (NFW) lens.

The NFW profile (Navarro, Frenk & White 1997) describes the dark-matter
density of a relaxed halo, and is the standard mass model for galaxy
clusters in strong lensing. The 3D density is

    rho(r) = rho_s / [ (r/r_s) (1 + r/r_s)^2 ]

with characteristic density ``rho_s`` and scale radius ``r_s``. The
projected (2D) convergence and deflection have closed-form expressions
(Wright & Brainerd 2000) which we implement below in dimensionless
form, then scale to arcsec with theta_s = r_s / D_L.

This is the simplest cluster-scale lens model in the package. Used in
notebook 09 (galaxy-cluster mass mapping).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .._helpers import _as_param


def _F(x: torch.Tensor) -> torch.Tensor:
    """Helper function used in the NFW deflection / convergence.

    F(x) = arctanh(sqrt(1-x^2)) / sqrt(1-x^2),  x < 1
    F(x) = 1,                                   x = 1
    F(x) = arctan(sqrt(x^2-1)) / sqrt(x^2-1),   x > 1
    """
    eps = 1e-8
    safe = torch.where(torch.abs(x - 1.0) < eps, x + 2.0 * eps, x)
    out = torch.where(
        safe < 1.0,
        torch.atanh(torch.sqrt(torch.clamp(1.0 - safe ** 2, min=0.0))) /
            torch.sqrt(torch.clamp(1.0 - safe ** 2, min=eps)),
        torch.atan(torch.sqrt(torch.clamp(safe ** 2 - 1.0, min=0.0))) /
            torch.sqrt(torch.clamp(safe ** 2 - 1.0, min=eps)),
    )
    out = torch.where(torch.abs(x - 1.0) < eps, torch.ones_like(out), out)
    return out


class NFW(nn.Module):
    r"""Spherical NFW lens (Wright & Brainerd 2000).

    Parameters
    ----------
    theta_s : scale radius in arcsec
    kappa_s : characteristic convergence (== rho_s * r_s / Sigma_crit, dimensionless)
    center_x, center_y : sky position of the cluster center (arcsec)
    """

    def __init__(
        self,
        theta_s: float = 30.0,
        kappa_s: float = 0.2,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ):
        super().__init__()
        self.theta_s = _as_param(theta_s, "theta_s")
        self.kappa_s = _as_param(kappa_s, "kappa_s")
        self.center_x = _as_param(center_x, "center_x")
        self.center_y = _as_param(center_y, "center_y")

    def _x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dx = x - self.center_x
        dy = y - self.center_y
        return torch.sqrt(dx * dx + dy * dy + 1e-12) / self.theta_s

    def kappa(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Convergence profile."""
        xn = self._x(x, y)
        return 2.0 * self.kappa_s * (1.0 - _F(xn)) / (xn ** 2 - 1.0 + 1e-12)

    def deflection(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deflection angle in arcsec (Wright & Brainerd 2000 Eq. 11)."""
        dx = x - self.center_x
        dy = y - self.center_y
        r = torch.sqrt(dx * dx + dy * dy + 1e-12)
        xn = r / self.theta_s
        # alpha(r) = 4 kappa_s theta_s g(x) / x with g(x) = ln(x/2) + F(x)
        gx = torch.log(xn / 2.0) + _F(xn)
        amp = 4.0 * self.kappa_s * self.theta_s * gx / xn
        return amp * dx / r, amp * dy / r

    def ray_trace(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a1, a2 = self.deflection(x, y)
        return x - a1, y - a2

    @torch.no_grad()
    def enforce_constraints(self):
        self.theta_s.data.clamp_(min=1e-3)
        self.kappa_s.data.clamp_(min=1e-4, max=10.0)
