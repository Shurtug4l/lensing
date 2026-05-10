"""Cosmological distance helpers wrapping astropy.

The thesis SIE/microlensing notebooks called ``astropy.cosmology`` directly and
juggled unit-stripping by hand.  This module exposes a thin wrapper that returns
plain torch tensors in physical Mpc / kpc / km, consistent across the package.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


@dataclass(frozen=True)
class Cosmology:
    """Lightweight FlatLambdaCDM wrapper that returns torch tensors.

    Examples
    --------
    >>> cosmo = Cosmology(H0=70.0, Om0=0.3)
    >>> dl, ds, dls = cosmo.lens_distances(zl=0.3, zs=2.0)
    """

    H0: float = 70.0
    Om0: float = 0.3
    Ob0: float = 0.05

    def _astropy(self) -> FlatLambdaCDM:
        return FlatLambdaCDM(H0=self.H0, Om0=self.Om0, Ob0=self.Ob0)

    def angular_diameter_distance(self, z: float) -> torch.Tensor:
        """Angular diameter distance in Mpc."""
        d = self._astropy().angular_diameter_distance(z).to(u.Mpc).value
        return torch.tensor(float(d))

    def angular_diameter_distance_z1z2(self, z1: float, z2: float) -> torch.Tensor:
        """Angular diameter distance between two redshifts in Mpc."""
        d = self._astropy().angular_diameter_distance_z1z2(z1, z2).to(u.Mpc).value
        return torch.tensor(float(d))

    def lens_distances(self, zl: float, zs: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(D_L, D_S, D_LS)`` for a thin-lens system in Mpc."""
        return (
            self.angular_diameter_distance(zl),
            self.angular_diameter_distance(zs),
            self.angular_diameter_distance_z1z2(zl, zs),
        )

    def einstein_radius_sie(self, zl: float, zs: float, sigma_v_kms: float) -> torch.Tensor:
        """SIE Einstein radius in arcseconds for velocity dispersion ``sigma_v``.

        theta_E = 4 pi (sigma_v / c)**2 * D_LS / D_S, converted from radians
        to arcseconds.
        """
        _, ds, dls = self.lens_distances(zl, zs)
        c_kms = const.c.to(u.km / u.s).value
        beta = (float(sigma_v_kms) / c_kms) ** 2
        rad = 4.0 * np.pi * beta * (dls.item() / ds.item())
        arcsec = rad * (180.0 / np.pi) * 3600.0
        return torch.tensor(float(arcsec))


# Default Planck-ish flat cosmology used when nothing is specified.
DEFAULT_COSMOLOGY = Cosmology(H0=70.0, Om0=0.3)
