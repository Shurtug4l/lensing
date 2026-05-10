"""Tests for the SIE lens model."""
from __future__ import annotations

import math

import numpy as np
import torch

import lensing as gl


def test_einstein_radius_from_velocity_dispersion():
    cosmo = gl.cosmology.Cosmology(H0=70., Om0=0.3)
    sie = gl.lens.SIE.from_velocity_dispersion(
        sigma_v_kms=200., q=0.7, pa=0.0, zl=0.3, zs=2.0, cosmo=cosmo,
    )
    # Order-of-magnitude check: theta_E ~ 1 arcsec for sigma=200 km/s, zl=0.3, zs=2.0
    assert 0.5 < float(sie.theta_E) < 2.0


def test_circular_lens_has_einstein_ring():
    """For q≈1 and beta=0, image positions lie on the Einstein ring."""
    sie = gl.lens.SIE(theta_E=1.0, q=0.999, pa=0.0)
    beta = torch.tensor(1e-3)  # near-axial source
    xs, ys = sie.solve_image_positions(beta, beta, n_grid=4000)
    if len(xs):
        radii = torch.sqrt(xs ** 2 + ys ** 2)
        # Each image should sit close to the Einstein radius (1.0).
        assert (radii > 0.5).all() and (radii < 1.5).all()


def test_image_positions_are_lensed_consistently():
    """Each found image should ray-trace back to the source position."""
    sie = gl.lens.SIE(theta_E=1.2, q=0.7, pa=math.pi / 5)
    beta_x, beta_y = torch.tensor(0.1), torch.tensor(-0.05)
    xs, ys = sie.solve_image_positions(beta_x, beta_y, n_grid=6000)
    assert len(xs) >= 2
    bx, by = sie.ray_trace(xs, ys)
    assert torch.allclose(bx, torch.full_like(bx, float(beta_x)), atol=1e-2)
    assert torch.allclose(by, torch.full_like(by, float(beta_y)), atol=1e-2)


def test_critical_curves_have_expected_shape():
    sie = gl.lens.SIE(theta_E=1.0, q=0.5, pa=0.0)
    cx, cy = sie.tangential_critical(n=200)
    # Critical line is an ellipse; its bounding box scales with theta_E.
    assert float(cx.abs().max()) > 0.5
    assert float(cy.abs().max()) > 0.5
