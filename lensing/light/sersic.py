"""Sérsic, core-Sérsic and multi-Sérsic surface brightness profiles.

Replaces the duplicated classes in
- THESIS/sersic/sersic_final.ipynb
- THESIS/core_sersic/coresersic_torch.ipynb
- THESIS/double_sersic/double_sersic_final.ipynb
"""
from __future__ import annotations

import torch

from .._helpers import _as_param
from ..utils.parameters import sersic_bn
from .base import LightModel, MultiLight


class Sersic(LightModel):
    r"""Sérsic (1968) surface brightness profile.

    .. math::
       I(R) = I_e \exp\!\Big[ -b_n \big( (R/R_e)^{1/n} - 1 \big) \Big]

    where ``b_n`` is given by :func:`lensing.utils.sersic_bn`.
    """

    def __init__(
        self,
        Ie: float,
        Re: float,
        n: float,
        x0: float = 0.0,
        y0: float = 0.0,
        e1: float = 0.0,
        e2: float = 0.0,
    ):
        super().__init__(x0=x0, y0=y0, e1=e1, e2=e2)
        self.Ie = _as_param(Ie, "Ie")
        self.Re = _as_param(Re, "Re")
        self.n = _as_param(n, "n")

    def profile(self, R: torch.Tensor) -> torch.Tensor:
        bn = sersic_bn(self.n)
        return self.Ie * torch.exp(-bn * ((R / self.Re) ** (1.0 / self.n) - 1.0))

    @torch.no_grad()
    def enforce_constraints(self):
        # Why: Adam can drive Re/Ie/n into invalid ranges after a single step;
        # we project the parameter back into the feasible region after every
        # update. The clamp values are far below any astrophysically meaningful
        # number so they never interfere with a well-posed fit.
        self.Re.data.clamp_(min=1e-3)
        self.Ie.data.clamp_(min=0.0)
        self.n.data.clamp_(min=0.3, max=12.0)


class CoreSersic(LightModel):
    r"""Core-Sérsic (Graham et al. 2003) profile.

    Two-power profile: an outer Sérsic + a steeper inner power law, joined by a
    transition controlled by ``alpha`` at the break radius ``Rb``::

        I(R) = I' (1 + (Rb/R)^alpha)^{gamma/alpha}
               * exp[ - b_n * ((R^alpha + Rb^alpha) / Re^alpha)^{1/(alpha n)} ]

    The thesis ``SersicCore`` notebook implemented this exactly; we keep the
    same numerical recipe but reuse the shared ellipse geometry from the base.
    """

    def __init__(
        self,
        Ib: float,
        Re: float,
        Rb: float,
        n: float,
        gamma: float,
        alpha: float,
        x0: float = 0.0,
        y0: float = 0.0,
        e1: float = 0.0,
        e2: float = 0.0,
    ):
        super().__init__(x0=x0, y0=y0, e1=e1, e2=e2)
        self.Ib = _as_param(Ib, "Ib")
        self.Re = _as_param(Re, "Re")
        self.Rb = _as_param(Rb, "Rb")
        self.n = _as_param(n, "n")
        self.gamma = _as_param(gamma, "gamma")
        self.alpha = _as_param(alpha, "alpha")

    def _I_prime(self) -> torch.Tensor:
        bn = sersic_bn(self.n)
        return (
            self.Ib
            * 2.0 ** (-self.gamma / self.alpha)
            * torch.exp(bn * (2.0 ** (1.0 / self.alpha) * self.Rb / self.Re) ** (1.0 / self.n))
        )

    def profile(self, R: torch.Tensor) -> torch.Tensor:
        bn = sersic_bn(self.n)
        Iprime = self._I_prime()
        inner = (1.0 + (self.Rb / R) ** self.alpha) ** (self.gamma / self.alpha)
        outer = torch.exp(
            -bn * ((R ** self.alpha + self.Rb ** self.alpha) / self.Re ** self.alpha) ** (1.0 / (self.alpha * self.n))
        )
        return Iprime * inner * outer

    @torch.no_grad()
    def enforce_constraints(self):
        self.Ib.data.clamp_(min=0.0)
        self.Re.data.clamp_(min=1e-3)
        self.Rb.data.clamp_(min=1e-3)
        self.n.data.clamp_(min=0.3, max=12.0)
        self.gamma.data.clamp_(min=0.0, max=15.0)
        self.alpha.data.clamp_(min=0.1, max=20.0)


def DoubleSersic(
    *,
    component1: dict,
    component2: dict,
) -> MultiLight:
    """Convenience constructor returning a 2-component ``MultiLight``.

    Each ``componentN`` dict is forwarded to ``Sersic(**componentN)``.

    Examples
    --------
    >>> galaxy = DoubleSersic(
    ...     component1=dict(Ie=20., Re=1.0,  n=3.5, x0=-15., y0=9.,  e1=0.1, e2=0.1),
    ...     component2=dict(Ie=22., Re=1.2,  n=5.5, x0=16.,  y0=-13., e1=0.2, e2=-0.1),
    ... )
    """
    return MultiLight([Sersic(**component1), Sersic(**component2)])
