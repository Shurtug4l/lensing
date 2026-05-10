"""Time-delay surfaces and Refsdal-style cosmography.

The Fermat potential of a thin lens is

.. math::
   \\tau(\\vec{\\theta}, \\vec{\\beta}) = \\tfrac{1}{2}|\\vec{\\theta} -
   \\vec{\\beta}|^2 - \\hat\\Psi(\\vec{\\theta}),

with two contributions:

* **Geometric**: ½ |θ - β|² is the extra path length traversed by the
  deflected ray relative to the undeflected line of sight.
* **Gravitational**: -Ψ̂(θ) is the Shapiro delay accumulated as the
  photon climbs out of the lens potential.

The physical time delay between two images at θ₁ and θ₂ of the same
source at β is

.. math::
   \\Delta t_{12} = \\frac{1 + z_L}{c}
                    \\frac{D_L D_S}{D_{LS}}\\,
                    [\\tau(\\theta_1,\\beta) - \\tau(\\theta_2,\\beta)].

The dimensional prefactor

.. math:: D_{\\Delta t} \\equiv (1 + z_L)\\, D_L D_S / D_{LS}

is the **time-delay distance**. Inverting Δt to recover D_{Δt} (and
hence H₀) is the principle of Refsdal (1964) cosmography, today applied
to the H0LiCOW / TDCOSMO program.

Reference: Meneghetti, *Lensing Gravitazionale* (UNIBO MSc lecture
notes), Ch. 3.6; Refsdal 1964; Suyu et al. 2017 (H0LiCOW).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from astropy import constants as const
from astropy import units as u

from ..cosmology import Cosmology, DEFAULT_COSMOLOGY


# Conversion: arcsec^2 -> rad^2; (1 + z_L) D_L D_S / D_LS in seconds, with
# distances in Mpc.
_MPC_TO_KM = u.Mpc.to(u.km)
_C_KMS = const.c.to(u.km / u.s).value


def time_delay_distance(zl: float, zs: float, cosmo: Cosmology = DEFAULT_COSMOLOGY) -> float:
    """``D_{Δt} = (1 + z_L) D_L D_S / D_LS`` in Mpc."""
    dl, ds, dls = cosmo.lens_distances(zl, zs)
    return (1.0 + zl) * float(dl) * float(ds) / float(dls)


def fermat_potential(
    theta: torch.Tensor,
    beta: torch.Tensor,
    psi: torch.Tensor,
) -> torch.Tensor:
    """Fermat potential ``τ = ½ |θ - β|² - Ψ̂``.

    Parameters
    ----------
    theta : (..., 2) image-plane positions in arcsec
    beta : (..., 2) source position in arcsec (broadcast)
    psi : (...) lensing potential evaluated at theta, in arcsec²
    """
    diff = theta - beta
    return 0.5 * (diff ** 2).sum(dim=-1) - psi


def time_delay_seconds(
    fermat_diff_arcsec2: torch.Tensor,
    zl: float,
    zs: float,
    cosmo: Cosmology = DEFAULT_COSMOLOGY,
) -> torch.Tensor:
    """Convert a Fermat-potential difference (arcsec²) into a time delay (s).

    .. math:: \\Delta t = \\frac{D_{\\Delta t}}{c}\\, \\Delta\\tau

    with the arcsec² -> rad² conversion factor folded in.
    """
    rad2_per_arcsec2 = (np.pi / 180.0 / 3600.0) ** 2
    Ddt_km = time_delay_distance(zl, zs, cosmo) * _MPC_TO_KM
    return fermat_diff_arcsec2 * rad2_per_arcsec2 * Ddt_km / _C_KMS


def refsdal_H0(
    delta_t_observed_days: float,
    fermat_diff_arcsec2: float,
    zl: float,
    zs: float,
    Om0: float = 0.3,
) -> float:
    """Recover H_0 (km/s/Mpc) from a measured Δt and a model Δτ.

    Inverting :func:`time_delay_seconds`:

    .. math::
       H_0 = h \\cdot 100, \\quad
       h = \\frac{D_{\\Delta t}^{(h=1)} \\cdot \\Delta\\tau_{\\rm rad^2}}
                  {c \\cdot \\Delta t_{\\rm s}}

    where :math:`D_{\\Delta t}^{(h=1)}` is computed at H0 = 100 km/s/Mpc
    (linear in 1/h); the ratio model/observed gives the actual h.
    """
    rad2_per_arcsec2 = (np.pi / 180.0 / 3600.0) ** 2
    Ddt_h1 = time_delay_distance(zl, zs, Cosmology(H0=100.0, Om0=Om0))  # Mpc/h
    Ddt_h1_km = Ddt_h1 * _MPC_TO_KM
    delta_t_s = float(delta_t_observed_days) * 86400.0
    h = (Ddt_h1_km * fermat_diff_arcsec2 * rad2_per_arcsec2) / (_C_KMS * delta_t_s)
    return float(h * 100.0)
