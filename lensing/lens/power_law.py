"""Power-law lens models.

The (axisymmetric) power-law lens has surface mass density / convergence

    kappa(x) = (3 - n) / 2 * x^(1 - n)

so that the deflection angle is alpha(x) = x^(2 - n) and the lensing
potential goes as Psi(x) ~ x^(3 - n) / (3 - n) (in dimensionless form).

The case n = 2 is the Singular Isothermal Sphere (SIS); n = 1 is a uniform
sheet (kappa = 1); n -> 3 is the point mass.

The elliptical generalisation (Tessore & Metcalf 2015, Keeton 2003) keeps
the same radial profile in the elliptical radius, with closed-form
deflections expressible via the hypergeometric ``2F1`` -- here we expose
the **axisymmetric** version (analytic) and a numerical-deflection
version for the elliptical case where we ray-trace the kernel via
direct integration. For most pedagogical purposes the axisymmetric
PowerLaw + ExternalShear is all you need.

References
----------
* Meneghetti, *Lensing Gravitazionale* (UNIBO MSc lecture notes), Ch. 5.2.
* Tessore & Metcalf 2015, *A&A* 580, A79.
* Keeton 2003, *ApJ* 584, 664.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .._helpers import _as_param


class PowerLawSpherical(nn.Module):
    r"""Axisymmetric power-law lens of slope ``n`` and Einstein radius ``theta_E``.

    Convergence (in units of theta_E):

    .. math::
       \kappa(x) = \frac{3-n}{2}\, x^{1-n}, \qquad n \in (1, 3)

    Setting ``n = 2`` recovers the Singular Isothermal Sphere (SIS).

    Parameters
    ----------
    theta_E : Einstein radius (arcsec)
    n : radial slope; ``n in (1, 3)``
    center_x, center_y : sky-plane center (arcsec)
    """

    def __init__(
        self,
        theta_E: float = 1.0,
        n: float = 2.0,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ):
        super().__init__()
        self.theta_E = _as_param(theta_E, "theta_E")
        self.n = _as_param(n, "n")
        self.center_x = _as_param(center_x, "center_x")
        self.center_y = _as_param(center_y, "center_y")

    def _x(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dx = x - self.center_x
        dy = y - self.center_y
        return torch.sqrt(dx * dx + dy * dy + 1e-12) / self.theta_E

    def kappa(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xn = self._x(x, y)
        return 0.5 * (3.0 - self.n) * xn ** (1.0 - self.n)

    def deflection(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dx = x - self.center_x
        dy = y - self.center_y
        r = torch.sqrt(dx * dx + dy * dy + 1e-12)
        xn = r / self.theta_E
        # alpha(x) = x^(2-n); in dimensional units = theta_E * (r/theta_E)^(2-n) e_r
        amp = self.theta_E * xn ** (2.0 - self.n) / xn  # times r-hat
        return amp * dx / self.theta_E, amp * dy / self.theta_E

    def potential(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Lensing potential Psi(x) = theta_E^2 * x^(3-n) / (3-n)."""
        xn = self._x(x, y)
        return self.theta_E ** 2 * xn ** (3.0 - self.n) / (3.0 - self.n)

    def ray_trace(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ax, ay = self.deflection(x, y)
        return x - ax, y - ay

    @torch.no_grad()
    def enforce_constraints(self):
        self.theta_E.data.clamp_(min=1e-3)
        # n must stay in (1, 3) for the canonical power-law lens regime.
        self.n.data.clamp_(min=1.01, max=2.99)


class SIS(PowerLawSpherical):
    """Singular Isothermal Sphere = PowerLaw with n = 2.

    The deflection is constant in magnitude (= theta_E), independent of r.
    """

    def __init__(self, theta_E: float = 1.0, center_x: float = 0.0, center_y: float = 0.0):
        super().__init__(theta_E=theta_E, n=2.0, center_x=center_x, center_y=center_y)
        # Keep n fixed at 2.
        self.n.requires_grad_(False)
