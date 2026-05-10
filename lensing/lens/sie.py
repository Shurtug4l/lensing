"""Singular Isothermal Ellipsoid (SIE) lens model.

Replaces / cleans up the ``SieLens`` class scattered across
``2_sie/sielens.ipynb``, ``2_sie/inversion.ipynb``, ``2_sie/caustics.ipynb``.

All quantities are expressed in dimensionless units of ``theta_E`` so the
formulas are exactly those of Kormann, Schneider & Bartelmann (1994). When
needed, ``theta_E`` is computed from the velocity dispersion via
:func:`lensing.cosmology.Cosmology.einstein_radius_sie`.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .._helpers import _as_param
from ..cosmology import Cosmology, DEFAULT_COSMOLOGY


class SIE(nn.Module):
    r"""Singular Isothermal Ellipsoid lens (Kormann+ 1994).

    Parameters in the *fit* parameterisation are the Einstein radius
    ``theta_E`` (arcsec), the axis ratio ``q in (0, 1]`` and the position
    angle ``pa`` (radians). Cosmology / redshifts / sigma_v are kept around so
    we can convert between the velocity-dispersion and the angular
    Einstein-radius parameterisations.

    The deflection angle, magnification, caustics and critical curves all
    follow the closed-form expressions for the SIE.
    """

    def __init__(
        self,
        theta_E: float,
        q: float,
        pa: float,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ):
        super().__init__()
        self.theta_E = _as_param(theta_E, "theta_E")
        self.q = _as_param(q, "q")
        self.pa = _as_param(pa, "pa")
        self.center_x = _as_param(center_x, "center_x")
        self.center_y = _as_param(center_y, "center_y")

    # -------------------------------------------------------------- factory
    @classmethod
    def from_velocity_dispersion(
        cls,
        sigma_v_kms: float,
        q: float,
        pa: float,
        zl: float,
        zs: float,
        cosmo: Cosmology = DEFAULT_COSMOLOGY,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> "SIE":
        thetaE = cosmo.einstein_radius_sie(zl, zs, sigma_v_kms)
        return cls(theta_E=thetaE.item(), q=q, pa=pa, center_x=center_x, center_y=center_y)

    # -------------------------------------------------------------- helpers
    def _rotate_to_lens(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = torch.cos(self.pa), torch.sin(self.pa)
        dx = x - self.center_x
        dy = y - self.center_y
        return cos * dx + sin * dy, -sin * dx + cos * dy

    def _delta(self, phi: torch.Tensor) -> torch.Tensor:
        """``Delta(phi)`` in lens-aligned polar coords."""
        return torch.sqrt(torch.cos(phi) ** 2 + self.q ** 2 * torch.sin(phi) ** 2)

    # ----------------------------------------------------- analytic SIE
    def alpha_polar(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deflection angle (alpha1, alpha2) at lens-aligned polar angle phi.

        The closed-form Kormann+94 result in units of ``theta_E``.
        """
        qprime = torch.sqrt(1.0 - self.q ** 2)
        a1 = (torch.sqrt(self.q) / qprime) * torch.asinh((qprime / self.q) * torch.cos(phi))
        a2 = (torch.sqrt(self.q) / qprime) * torch.asin(qprime * torch.sin(phi))
        return a1, a2

    def deflection(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cartesian deflection angle at sky-aligned (x, y) in arcsec.

        Lens-aligned coords: rotate, evaluate alpha, rotate back, multiply by
        theta_E so the output has the same units as the input coords.
        """
        x_rot, y_rot = self._rotate_to_lens(x, y)
        phi = torch.atan2(y_rot, x_rot)
        a1, a2 = self.alpha_polar(phi)
        a1 = a1 * self.theta_E
        a2 = a2 * self.theta_E
        # Rotate the deflection back to the sky frame.
        cos, sin = torch.cos(self.pa), torch.sin(self.pa)
        return cos * a1 - sin * a2, sin * a1 + cos * a2

    def kappa(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Convergence."""
        x_rot, y_rot = self._rotate_to_lens(x, y)
        r = torch.sqrt(x_rot ** 2 + y_rot ** 2)
        phi = torch.atan2(y_rot, x_rot)
        return torch.sqrt(self.q) * self.theta_E / (2.0 * r * self._delta(phi))

    def shear(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_rot, y_rot = self._rotate_to_lens(x, y)
        r = torch.sqrt(x_rot ** 2 + y_rot ** 2)
        phi = torch.atan2(y_rot, x_rot)
        kappa = torch.sqrt(self.q) * self.theta_E / (2.0 * r * self._delta(phi))
        # Shear has the opposite sign convention vs. convergence here, which
        # follows the Kormann+ 94 derivation used in the thesis.
        g1 = -kappa * torch.cos(2.0 * phi)
        g2 = -kappa * torch.sin(2.0 * phi)
        return g1, g2

    def magnification(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        kappa = self.kappa(x, y)
        g1, g2 = self.shear(x, y)
        gamma = torch.sqrt(g1 ** 2 + g2 ** 2)
        return 1.0 / ((1.0 - kappa - gamma) * (1.0 - kappa + gamma))

    # ----------------------------------------------------- caustics & curves
    @torch.no_grad()
    def tangential_caustic(self, n: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """(y1, y2) coordinates of the tangential caustic in the source plane.

        ``@torch.no_grad`` because these are visualization curves: returning
        graph-attached tensors would break ``plt.plot(x, y)`` (which calls
        ``np.asarray`` on the inputs).
        """
        phi = torch.linspace(0.0, 2.0 * torch.pi, n)
        delta = self._delta(phi)
        a1, a2 = self.alpha_polar(phi)
        y1_ = (torch.sqrt(self.q) / delta) * torch.cos(phi) - a1
        y2_ = (torch.sqrt(self.q) / delta) * torch.sin(phi) - a2
        cos, sin = torch.cos(self.pa), torch.sin(self.pa)
        return self.theta_E * (cos * y1_ - sin * y2_), self.theta_E * (sin * y1_ + cos * y2_)

    @torch.no_grad()
    def tangential_critical(self, n: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """(x1, x2) coordinates of the tangential critical line in the image plane."""
        phi = torch.linspace(0.0, 2.0 * torch.pi, n)
        delta = self._delta(phi)
        x1_ = (torch.sqrt(self.q) / delta) * torch.cos(phi)
        x2_ = (torch.sqrt(self.q) / delta) * torch.sin(phi)
        cos, sin = torch.cos(self.pa), torch.sin(self.pa)
        return self.theta_E * (cos * x1_ - sin * x2_), self.theta_E * (sin * x1_ + cos * x2_)

    @torch.no_grad()
    def cut(self, n: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cut curve (the trace of alpha along the unit circle)."""
        phi = torch.linspace(0.0, 2.0 * torch.pi, n)
        a1, a2 = self.alpha_polar(phi)
        cos, sin = torch.cos(self.pa), torch.sin(self.pa)
        y1 = -(cos * a1 - sin * a2)
        y2 = -(sin * a1 + cos * a2)
        return self.theta_E * y1, self.theta_E * y2

    # -------------------------------------------------- multi-image solver
    @torch.no_grad()
    def solve_image_positions(
        self,
        beta_x: torch.Tensor,
        beta_y: torch.Tensor,
        n_grid: int = 2000,
        tol: float = 1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve the lens equation for a point source at ``(beta_x, beta_y)``.

        Implements the same bisection+root-bracketing strategy as the thesis
        ``xphi_image`` method but vectorized: we evaluate the ``F(phi)`` of
        Kormann+94 on a dense grid of polar angles, find sign changes, and
        refine each root with bisection. Returns sky-frame ``(x_image,
        y_image)`` arrays in the same units as ``theta_E``.
        """
        # Source position in lens-aligned units of theta_E.
        bx, by = self._rotate_to_lens(beta_x, beta_y)
        bx, by = bx / self.theta_E, by / self.theta_E

        def F(phi):
            a1, a2 = self.alpha_polar(phi)
            return (bx + a1) * torch.sin(phi) - (by + a2) * torch.cos(phi)

        def x_of_phi(phi):
            return bx * torch.cos(phi) + by * torch.sin(phi) + (
                # |alpha| projection is exactly the integral psi_tilde
                # at the unit circle for SIE, equal to alpha . r_hat.
                self.alpha_polar(phi)[0] * torch.cos(phi)
                + self.alpha_polar(phi)[1] * torch.sin(phi)
            )

        phis = torch.linspace(0.0, 2.0 * torch.pi, n_grid)
        Fvals = F(phis)
        sign_change = (Fvals[:-1] * Fvals[1:] < 0).nonzero(as_tuple=False).flatten()

        roots_phi, roots_x = [], []
        for i in sign_change.tolist():
            a, b = phis[i].clone(), phis[i + 1].clone()
            fa, fb = Fvals[i].clone(), Fvals[i + 1].clone()
            for _ in range(60):  # ~tol = 2*pi / 2^60, plenty
                mid = 0.5 * (a + b)
                fm = F(mid)
                if (fa * fm) < 0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
                if (b - a) < tol:
                    break
            phi = 0.5 * (a + b)
            x = x_of_phi(phi)
            if x > 0:  # valid radial position in lens-aligned polar coords
                roots_phi.append(phi)
                roots_x.append(x)

        if not roots_phi:
            return (
                torch.empty(0, dtype=phis.dtype),
                torch.empty(0, dtype=phis.dtype),
            )

        phi_t = torch.stack(roots_phi)
        x_t = torch.stack(roots_x) * self.theta_E
        # Convert (x, phi) lens-aligned -> sky.
        x1_lens = x_t * torch.cos(phi_t)
        x2_lens = x_t * torch.sin(phi_t)
        cos, sin = torch.cos(self.pa), torch.sin(self.pa)
        return cos * x1_lens - sin * x2_lens, sin * x1_lens + cos * x2_lens

    # ----------------------------------------------------- ray tracing
    def ray_trace(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map image-plane (x, y) to source-plane (beta_x, beta_y)."""
        a1, a2 = self.deflection(x, y)
        return x - a1, y - a2

    @torch.no_grad()
    def enforce_constraints(self):
        """Project parameters back into the physically meaningful region."""
        self.theta_E.data.clamp_(min=1e-3)
        self.q.data.clamp_(min=1e-3, max=1.0 - 1e-5)
