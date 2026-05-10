"""Binary point-mass microlens.

When a microlensing source crosses a *caustic* of a binary lens, the light
curve develops sharp magnification spikes whose timing constrains the
companion's mass and separation -- this is the workhorse of exoplanet
detection in microlensing surveys (OGLE, KMTNet, Roman).

The reduced lens equation for two coplanar point masses with mass fractions
``m_1`` and ``m_2 = 1 - m_1`` separated by ``d`` (in units of the system
Einstein radius) is

    beta = theta - sum_i m_i (theta - theta_i) / |theta - theta_i|^2

with theta_1 = (-d/2, 0), theta_2 = (+d/2, 0).

We implement the forward map ``ray_trace`` and a magnification-via-image-
plane-grid trick (inverse ray-shooting) to produce light curves. For exact
image solving the lens equation reduces to a 5th-order complex polynomial -
we leave that as future work; the magnification-map approach is sufficient
to reproduce caustic-crossing light curves at the resolution we care about.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .._helpers import _as_param


class BinaryPointMass(nn.Module):
    """Two-point-mass microlens with separation ``d`` and mass ratio ``q_m``.

    Conventions:

    * lengths are in units of the total-mass Einstein radius ``theta_E``;
    * the binary lies along the x-axis; ``q_m = m_2/m_1 in (0, 1]``;
    * source and lens are on the same plane (no rotation in time of the
      orbit, which is fine for short caustic-crossing events).
    """

    def __init__(self, d: float = 1.0, q_m: float = 1.0, pa: float = 0.0):
        super().__init__()
        self.d = _as_param(d, "d")
        self.q_m = _as_param(q_m, "q_m")
        self.pa = _as_param(pa, "pa")

    def _masses(self):
        m1 = 1.0 / (1.0 + self.q_m)
        m2 = self.q_m / (1.0 + self.q_m)
        return m1, m2

    def _component_positions(self):
        cos, sin = torch.cos(self.pa), torch.sin(self.pa)
        x1 = -0.5 * self.d * cos
        y1 = -0.5 * self.d * sin
        x2 = +0.5 * self.d * cos
        y2 = +0.5 * self.d * sin
        return (x1, y1), (x2, y2)

    def deflection(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m1, m2 = self._masses()
        (x1, y1), (x2, y2) = self._component_positions()
        dx1, dy1 = x - x1, y - y1
        dx2, dy2 = x - x2, y - y2
        r1_sq = dx1 ** 2 + dy1 ** 2 + 1e-12
        r2_sq = dx2 ** 2 + dy2 ** 2 + 1e-12
        ax = m1 * dx1 / r1_sq + m2 * dx2 / r2_sq
        ay = m1 * dy1 / r1_sq + m2 * dy2 / r2_sq
        return ax, ay

    def ray_trace(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ax, ay = self.deflection(x, y)
        return x - ax, y - ay

    @torch.no_grad()
    def magnification_map(
        self,
        npix: int = 401,
        halfwidth: float = 2.0,
        oversample: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Magnification map ``μ(β)`` by inverse ray-shooting.

        Method (Kayser, Refsdal & Stabell 1986; the canonical recipe in
        microlensing): shoot a uniform grid of ``(npix·oversample)^2``
        image-plane rays back to the source plane and count how many
        rays land in each source-plane bin. Without a lens, the count
        per source bin is ``oversample^2`` exactly (the image-plane
        ray density is uniform and the lens equation is the identity);
        with a lens the ratio of observed to unlensed count *is* the
        magnification:

        .. math::
           \mu(\beta_i) =
              \frac{N_{\rm rays}(\beta_i)}{\langle N_{\rm rays}\rangle_{\rm no\;lens}}.

        Oversampling (default 3×) reduces the Poisson noise on the
        ratio by a factor ``oversample``.

        Units
        -----
        * ``halfwidth`` : units of θ_E (the **system** Einstein radius
          for the binary as a whole).
        * returned ``axis`` : same units (θ_E).
        * returned ``mu`` : dimensionless; ``mu = 1`` outside caustics,
          large (formally infinite on the critical curve).
        """
        n_im = npix * oversample
        ax_im = torch.linspace(-halfwidth, halfwidth, n_im)
        x_im, y_im = torch.meshgrid(ax_im, ax_im, indexing="xy")
        bx, by = self.ray_trace(x_im.flatten(), y_im.flatten())

        # Bin the source-plane positions on the requested resolution.
        bins = torch.linspace(-halfwidth, halfwidth, npix + 1)
        ix = torch.bucketize(bx, bins) - 1
        iy = torch.bucketize(by, bins) - 1
        valid = (ix >= 0) & (ix < npix) & (iy >= 0) & (iy < npix)
        counts = torch.zeros((npix, npix))
        counts.index_put_(
            (iy[valid], ix[valid]),
            torch.ones(int(valid.sum())),
            accumulate=True,
        )
        # Expected count per source bin without lensing = oversample**2.
        unlensed_count = float(oversample) ** 2
        # Centre-of-bin axis for plotting (instead of the bin edges).
        axis = bins[:-1] + 0.5 * (bins[1] - bins[0])
        return axis, counts / unlensed_count

    @torch.no_grad()
    def critical_curves(self, n: int = 1500, halfwidth: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Locate the critical curve numerically by zero-crossings of detA.

        For visualization only; the analytic 5th-order polynomial roots would
        be more efficient if we needed them in a tight loop.
        """
        ax = torch.linspace(-halfwidth, halfwidth, n)
        x_im, y_im = torch.meshgrid(ax, ax, indexing="xy")
        # Numerical det A via finite differences on the deflection.
        bx, by = self.ray_trace(x_im, y_im)
        dy_x = torch.zeros_like(bx)
        dy_y = torch.zeros_like(bx)
        dx_x = torch.zeros_like(bx)
        dx_y = torch.zeros_like(bx)
        # central differences
        dx_x[:, 1:-1] = (bx[:, 2:] - bx[:, :-2]) / (2 * (ax[1] - ax[0]))
        dx_y[1:-1, :] = (bx[2:, :] - bx[:-2, :]) / (2 * (ax[1] - ax[0]))
        dy_x[:, 1:-1] = (by[:, 2:] - by[:, :-2]) / (2 * (ax[1] - ax[0]))
        dy_y[1:-1, :] = (by[2:, :] - by[:-2, :]) / (2 * (ax[1] - ax[0]))
        detA = dx_x * dy_y - dx_y * dy_x
        return ax, detA
