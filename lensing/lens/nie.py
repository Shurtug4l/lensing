"""Non-singular Isothermal Ellipsoid (NIE / softened SIE).

Surface density (Kormann, Schneider & Bartelmann 1994):

.. math::
   \\Sigma(\\xi) = \\frac{\\sigma^2}{2 G}
                  \\frac{\\sqrt{f}}{\\sqrt{\\xi_1^2 + f^2 \\xi_2^2 + \\xi_c^2}},

where ``f = q`` is the axis ratio of the *isodensity contours* and
``xi_c`` is a core radius that smooths out the central singularity.

Setting ``xi_c = 0`` recovers the SIE; setting both ``xi_c = 0`` and
``q = 1`` recovers the SIS.

The corresponding convergence in dimensionless units (xi -> x = xi/theta_E):

.. math::
   \\kappa(x) = \\frac{\\sqrt{q}}{2 \\sqrt{x_1^2 + q^2 x_2^2 + x_c^2}}.

The deflection has a closed form involving ``arcsinh`` / ``arctan``
functions that we evaluate directly. For ``x_c = 0`` the formulas
reduce to the SIE expressions implemented in :class:`lensing.lens.SIE`.

The interesting feature of NIE compared to SIE is the **caustic
topology**: depending on ``x_c / q^(3/2)`` the lens has tangential and
radial caustics with characteristic cusps; see Kormann+ 1994 Sec. 3 and
Meneghetti lecture notes Ch. 5.4.2 for the regime diagram.

References
----------
* Meneghetti, *Lensing Gravitazionale* (UNIBO MSc lecture notes), Ch. 5.4.
* Kormann, Schneider & Bartelmann 1994, *A&A* 284, 285.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .._helpers import _as_param


class NIE(nn.Module):
    """Non-singular Isothermal Ellipsoid.

    Parameters
    ----------
    theta_E : Einstein radius scale (arcsec)
    q : axis ratio of the isodensity contours, in (0, 1]
    pa : position angle (radians)
    core : core radius xi_c in arcsec; ``core=0`` -> SIE
    center_x, center_y : sky-plane center (arcsec)
    """

    def __init__(
        self,
        theta_E: float = 1.0,
        q: float = 0.7,
        pa: float = 0.0,
        core: float = 0.05,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ):
        super().__init__()
        self.theta_E = _as_param(theta_E, "theta_E")
        self.q = _as_param(q, "q")
        self.pa = _as_param(pa, "pa")
        self.core = _as_param(core, "core")
        self.center_x = _as_param(center_x, "center_x")
        self.center_y = _as_param(center_y, "center_y")

    def _rotate(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = torch.cos(self.pa), torch.sin(self.pa)
        dx = x - self.center_x
        dy = y - self.center_y
        return cos * dx + sin * dy, -sin * dx + cos * dy

    def kappa(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x1, x2 = self._rotate(x, y)
        # In units of theta_E. Add core via xi_c in the same units.
        denom = torch.sqrt(x1 ** 2 + self.q ** 2 * x2 ** 2 + self.core ** 2 + 1e-12)
        return torch.sqrt(self.q) * self.theta_E / (2.0 * denom)

    def deflection(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cartesian deflection in arcsec.

        Closed form (Kormann+94 Eq. 17a-b, with sign conventions matching
        :class:`lensing.lens.SIE`); the core regularizes the cusp at the
        origin.
        """
        x1, x2 = self._rotate(x, y)
        qprime = torch.sqrt(torch.clamp(1.0 - self.q ** 2, min=1e-8))
        psi = torch.sqrt(self.q ** 2 * (self.core ** 2 + x1 ** 2) + x2 ** 2 + 1e-12)
        a1 = (self.theta_E * torch.sqrt(self.q) / qprime) * torch.atan(
            qprime * x1 / (psi + self.core)
        )
        a2 = (self.theta_E * torch.sqrt(self.q) / qprime) * torch.atanh(
            qprime * x2 / (psi + self.q ** 2 * self.core)
        )
        cos, sin = torch.cos(self.pa), torch.sin(self.pa)
        return cos * a1 - sin * a2, sin * a1 + cos * a2

    def ray_trace(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ax, ay = self.deflection(x, y)
        return x - ax, y - ay

    @torch.no_grad()
    def enforce_constraints(self):
        self.theta_E.data.clamp_(min=1e-3)
        self.q.data.clamp_(min=1e-3, max=1.0 - 1e-5)
        self.core.data.clamp_(min=1e-4)
