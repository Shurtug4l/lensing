"""Parameter transforms used across light and lens models.

Two parameterizations of an ellipse are used in the literature and across the
thesis:

* (q, PA): axis ratio in [0, 1] and position angle in [0, pi).  Intuitive for
  reporting results, but PA suffers from a 0 / pi wrap discontinuity that hurts
  gradient-based optimization.

* (e1, e2) = ((1-q)/(1+q)) * (cos 2*PA, sin 2*PA): the two ellipticity components
  used by lenstronomy and weak-lensing literature.  They are continuous in PA
  and unconstrained, which is friendlier to autodiff optimizers.

We always keep (e1, e2) as the **optimization** parameters; (q, PA) are derived
for reporting.

The ``sersic_bn`` polynomial expansion is the Ciotti & Bertin (1999) formula
truncated at order n^-4, accurate to ~1e-4 for n >= 0.5.
"""
from __future__ import annotations

import math

import torch


def e1e2_to_q_pa(e1: torch.Tensor, e2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (e1, e2) ellipticity components to (q, PA in radians).

    Used for **reporting** the fit results in human-friendly units, never
    inside an autograd loop (the elliptical radius is computed directly in
    (e1, e2); see :class:`lensing.light.base.LightModel`).

    The radial magnitude ``e = (1-q)/(1+q)`` is clamped to ``< 1`` to avoid
    the degenerate q -> 0 limit. A tiny epsilon under the sqrt ensures
    ``e1 = e2 = 0`` does not produce a NaN here either.
    """
    e = torch.sqrt(e1 ** 2 + e2 ** 2 + 1e-30)
    e = torch.clamp(e, max=1.0 - 1e-5)
    q = (1.0 - e) / (1.0 + e)
    pa = 0.5 * torch.atan2(e2, e1)
    return q, pa


def q_pa_to_e1e2(q: torch.Tensor, pa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (q, PA) to (e1, e2)."""
    e = (1.0 - q) / (1.0 + q)
    return e * torch.cos(2.0 * pa), e * torch.sin(2.0 * pa)


def sersic_bn(n: torch.Tensor) -> torch.Tensor:
    """Ciotti & Bertin (1999) approximation for the Sérsic ``b_n`` constant.

    ``b_n`` is defined implicitly by the requirement that ``R_e`` enclose half
    of the total luminosity. The polynomial expansion below is accurate to ~1e-4
    for n >= 0.36 and avoids solving an incomplete-gamma equation inside the
    autograd graph.
    """
    n = torch.as_tensor(n)
    bn = (
        2.0 * n
        - 1.0 / 3.0
        + 4.0 / (405.0 * n)
        + 46.0 / (25515.0 * n ** 2)
        + 131.0 / (1148175.0 * n ** 3)
        - 2194697.0 / (30690717750.0 * n ** 4)
    )
    return torch.clamp(bn, min=1e-5)


def softplus_inverse(y: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: returns ``x`` such that ``softplus(x) = y``."""
    return y + torch.log1p(-torch.exp(-y))


def wrap_angle(pa: torch.Tensor, period: float = math.pi) -> torch.Tensor:
    """Wrap an angle into [0, period). Useful for reporting PA after a fit."""
    return pa - period * torch.floor(pa / period)
