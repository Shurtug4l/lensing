"""Point-mass microlens light curve.

Replaces the ``Micro`` / ``MicroGood`` classes in
``THESIS/micro/microlens_torch.ipynb``.

The thesis demonstrated explicitly that the *physical* parameterization
``(M, v_rel, D_L, D_S)`` is degenerate at the level of the observable light
curve, while the *minimal* parameterization ``(f_S, t_0, y_0, t_E)`` is not.
We expose both:

* :class:`PointMassMicrolens` keeps the physical parameters and computes
  ``theta_E`` and ``t_E`` from astrophysical inputs — useful for forward
  modelling and didactic plots.

* :class:`PaczynskiLightcurve` is the minimal model, the one to fit data with.

Units used by this module
-------------------------
* ``f``        : arbitrary flux units (matches the data)
* ``mass``     : solar masses (M_⊙)
* ``y0``       : impact parameter, dimensionless (units of θ_E)
* ``vel``      : km/s — *transverse relative* velocity in the lens plane
* ``t0``       : days — peak-magnification time
* ``dl, ds``   : kpc — angular-diameter distances (galactic micro: flat sky)
* ``tE``       : days — Einstein crossing time
* output ``θ_E`` (``einstein_radius``) : arcsec
* output ``t_E`` (``einstein_time``)   : days

Reference: Paczynski 1986; Meneghetti, *Lensing Gravitazionale* Ch. 4.
"""
from __future__ import annotations

import math

import torch
from astropy import constants as const
from torch import nn

from .._helpers import _as_param

_KPC_TO_KM = (1.0 * const.kpc.to("km")).value  # ~3.085677581e+16
_C_KMS = const.c.to("km/s").value  # ~2.998e5
_MSUN_KG = const.M_sun.to("kg").value
_G_KM3_PER_KG_S2 = const.G.to("km**3 / (kg * s**2)").value
_RAD_TO_ARCSEC = (180.0 / math.pi) * 3600.0


class PaczynskiLightcurve(nn.Module):
    r"""Minimal Paczynski (1986) point-mass microlensing light curve.

    .. math::
        \mu(t) = f_S \cdot \frac{u^2 + 2}{u\sqrt{u^2 + 4}},\quad
        u(t) = \sqrt{((t-t_0)/t_E)^2 + y_0^2}

    Parameters
    ----------
    f : flux of the source star (arbitrary units, in practice baseline flux)
    y0 : impact parameter (dimensionless, units of theta_E)
    t0 : time of peak magnification (days)
    tE : Einstein crossing time (days)
    """

    def __init__(self, f: float, y0: float, t0: float, tE: float):
        super().__init__()
        self.f = _as_param(f, "f")
        self.y0 = _as_param(y0, "y0")
        self.t0 = _as_param(t0, "t0")
        self.tE = _as_param(tE, "tE")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        u1 = (t - self.t0) / self.tE
        u = torch.sqrt(u1 ** 2 + self.y0 ** 2 + 1e-12)
        return self.f * (u ** 2 + 2.0) / (u * torch.sqrt(u ** 2 + 4.0))

    @torch.no_grad()
    def enforce_constraints(self):
        self.f.data.clamp_(min=0.0)
        self.tE.data.clamp_(min=1e-3)


class PointMassMicrolens(nn.Module):
    """Point-mass microlens with astrophysical parameters.

    Computes ``theta_E`` and ``t_E`` from ``(M, D_L, D_S, v_rel)`` and then
    delegates the magnification computation to the Paczynski formula.

    Distances ``dl``, ``ds`` are in kpc; ``vel`` in km/s; ``mass`` in solar
    masses.
    """

    def __init__(
        self,
        f: float,
        mass: float,
        y0: float,
        vel: float,
        t0: float,
        dl: float,
        ds: float,
    ):
        super().__init__()
        self.f = _as_param(f, "f")
        self.mass = _as_param(mass, "mass")
        self.y0 = _as_param(y0, "y0")
        self.vel = _as_param(vel, "vel")
        self.t0 = _as_param(t0, "t0")
        self.dl = _as_param(dl, "dl")
        self.ds = _as_param(ds, "ds")

    # Astropy-derived constants kept on the same device/dtype as parameters.
    def _const(self, value):
        return torch.as_tensor(value, dtype=self.f.dtype, device=self.f.device)

    def einstein_radius(self) -> torch.Tensor:
        """Angular Einstein radius in arcseconds."""
        c = self._const(_C_KMS)
        G = self._const(_G_KM3_PER_KG_S2)
        msun = self._const(_MSUN_KG)
        kpc_km = self._const(_KPC_TO_KM)
        rad2as = self._const(_RAD_TO_ARCSEC)
        m_kg = self.mass * msun
        # D_LS / (D_L D_S) for a flat geometry with distances in kpc
        dist_mod = ((self.ds - self.dl) / (self.dl * self.ds)) / kpc_km
        rad = torch.sqrt(4.0 * G * m_kg / c ** 2 * dist_mod)
        return rad * rad2as

    def einstein_time(self) -> torch.Tensor:
        """Einstein crossing time in days."""
        kpc_km = self._const(_KPC_TO_KM)
        rad2as = self._const(_RAD_TO_ARCSEC)
        thetaE = self.einstein_radius()  # arcsec
        # convert thetaE from arcsec to radians, then to physical km in lens plane
        rE_km = self.dl * (thetaE / rad2as) * kpc_km
        tE_s = rE_km / self.vel
        return tE_s / 86400.0

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        u1 = (t - self.t0) / self.einstein_time()
        u = torch.sqrt(u1 ** 2 + self.y0 ** 2 + 1e-12)
        return self.f * (u ** 2 + 2.0) / (u * torch.sqrt(u ** 2 + 4.0))

    @torch.no_grad()
    def enforce_constraints(self):
        self.f.data.clamp_(min=0.0)
        self.mass.data.clamp_(min=1e-4)
        self.vel.data.clamp_(min=1.0)
        # D_S > D_L > 0 keeps the angular-distance ratio finite.
        self.dl.data.clamp_(min=1e-3)
        self.ds.data.clamp_(min=self.dl.item() + 1e-3)
