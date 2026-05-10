"""Common interface for surface brightness models.

The thesis defined `Sersic`, `SersicCore` and `SersicDouble` as independent
``nn.Module`` classes with copy-pasted ellipse geometry.  Here we lift the
common pieces (centre/rotation/axis-ratio) into ``LightModel`` and let the
specific profiles override only ``profile(R)``.

Adopted parameterization
------------------------
We use ``(x0, y0, e1, e2)`` for the geometry and never store ``q`` / ``PA``
directly.  ``e1`` / ``e2`` are unconstrained and continuous, which is what
makes Adam/L-BFGS converge cleanly; ``q``, ``PA`` are recovered post-fit via
``lensing.utils.e1e2_to_q_pa``.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn

from .._helpers import _as_param


class LightModel(nn.Module):
    """Base class for elliptical surface brightness profiles.

    Subclasses need to implement ``profile(R)`` returning the surface
    brightness on the elliptical radius grid ``R``.  Optional ``forward(xy)``
    is provided here.
    """

    def __init__(self, x0=0.0, y0=0.0, e1=0.0, e2=0.0):
        super().__init__()
        self.x0 = _as_param(x0, "x0")
        self.y0 = _as_param(y0, "y0")
        self.e1 = _as_param(e1, "e1")
        self.e2 = _as_param(e2, "e2")

    # ------------------------------------------------------------ geometry
    def elliptical_radius(self, xy: torch.Tensor) -> torch.Tensor:
        """Return the ellipse-aligned radius for every point in ``xy``.

        ``xy`` has shape (2, ...) where xy[0] is x and xy[1] is y.

        Why this formulation: the canonical ``R = sqrt(xt1**2 + (xt2/q)**2)``
        used in the thesis notebooks goes through ``q = (1-|e|)/(1+|e|)`` and
        ``pa = 0.5*atan2(e2, e1)``, both of which produce **NaN gradients at
        (e1, e2) = (0, 0)** - a perfectly common initial value for an
        optimizer. After expanding the rotation analytically and absorbing
        the smooth scaling ``(1-|e|)`` into the effective radius, the
        elliptical radius squared reduces to a polynomial in (e1, e2):

            R'^2 = (1 + e^2 - 2 e1) dx^2 + (1 + e^2 + 2 e1) dy^2 - 4 e2 dx dy

        which is differentiable everywhere. The convention is that ``Re`` in
        this package corresponds to the (1-|e|)-scaled thesis effective
        radius, i.e. ``Re_lensing = Re_thesis * (1 - |e|)``. For
        circularly symmetric profiles (``e1=e2=0``) the two definitions
        coincide.
        """
        dx = xy[0] - self.x0
        dy = xy[1] - self.y0
        e_sq = self.e1 * self.e1 + self.e2 * self.e2
        A = 1.0 + e_sq - 2.0 * self.e1
        B = 1.0 + e_sq + 2.0 * self.e1
        C = -4.0 * self.e2
        R_sq = A * dx * dx + B * dy * dy + C * dx * dy
        return torch.sqrt(torch.clamp(R_sq, min=1e-10))

    # ----------------------------------------------------------- interface
    def profile(self, R: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.profile(self.elliptical_radius(xy))

    # ----------------------------------------------------------- helpers
    def report(self) -> dict:
        """Return a flat dict of named parameters for printing/logging."""
        return {name: float(p.detach()) for name, p in self.named_parameters()}


class MultiLight(nn.Module):
    """Sum of arbitrary ``LightModel`` components (replaces ``SersicDouble``).

    Examples
    --------
    >>> bulge = Sersic(Ie=20., Re=1.0, n=4.0, x0=0., y0=0., e1=0.1, e2=0.0)
    >>> disk  = Sersic(Ie=2.,  Re=4.0, n=1.0, x0=0., y0=0., e1=0.05, e2=0.0)
    >>> galaxy = MultiLight([bulge, disk])
    """

    def __init__(self, components: List[LightModel]):
        super().__init__()
        self.components = nn.ModuleList(components)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return sum(c(xy) for c in self.components)
