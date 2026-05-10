"""Smoke tests for the advanced (PowerLaw, NIE) lens models and time delays."""
from __future__ import annotations

import math

import torch

import lensing as gl


def test_sis_deflection_is_constant():
    """SIS has alpha = theta_E independent of r."""
    sis = gl.lens.SIS(theta_E=1.5)
    rs = torch.tensor([0.5, 1.0, 2.0, 5.0])
    ax, ay = sis.deflection(rs, torch.zeros_like(rs))
    # Magnitude should be theta_E everywhere.
    mag = torch.sqrt(ax ** 2 + ay ** 2)
    assert torch.allclose(mag, torch.full_like(mag, 1.5), atol=1e-3)


def test_powerlaw_einstein_radius():
    """At r = theta_E the deflection magnitude must equal theta_E for any slope."""
    for n in [1.5, 1.8, 2.0, 2.3, 2.7]:
        pl = gl.lens.PowerLawSpherical(theta_E=1.0, n=n)
        x = torch.tensor([1.0])
        y = torch.tensor([0.0])
        ax, _ = pl.deflection(x, y)
        assert abs(float(ax) - 1.0) < 1e-3, n


def test_nie_deflection_finite_at_origin():
    """NIE with finite core should be finite at the lens centre.

    This is the *whole point* of the NIE: regularise the SIE singularity.
    A purely qualitative test — the magnitudes here cannot be compared to
    SIE because the two derivations use different angular conventions
    (polar vs Cartesian Kormann+94 forms).
    """
    nie = gl.lens.NIE(theta_E=1.0, q=0.7, pa=0.0, core=0.1)
    ax, ay = nie.deflection(torch.tensor(0.0), torch.tensor(0.0))
    assert torch.isfinite(ax) and torch.isfinite(ay)
    assert abs(float(ax)) < 1.0
    assert abs(float(ay)) < 1.0


def test_nie_kappa_smooth_at_origin():
    """NIE convergence should NOT diverge at the lens centre."""
    nie = gl.lens.NIE(theta_E=1.0, q=0.7, pa=0.0, core=0.1)
    k0 = nie.kappa(torch.tensor(0.0), torch.tensor(0.0))
    assert torch.isfinite(k0)
    # SIE kappa diverges as 1/r at r=0; NIE kappa(0) ~ 1/(2 core / sqrt(q))
    expected = float(torch.sqrt(nie.q)) * 1.0 / (2.0 * 0.1)
    assert 0.5 * expected < float(k0) < 2.0 * expected


def test_time_delay_distance_positive():
    cosmo = gl.cosmology.Cosmology(H0=70., Om0=0.3)
    Ddt = gl.lens.timedelay.time_delay_distance(zl=0.3, zs=2.0, cosmo=cosmo)
    assert 1500 < Ddt < 5000  # Mpc, reasonable range for these redshifts


def test_refsdal_h0_inverse_of_forward():
    """Round-trip: predict Δt from H₀=70, then invert it to recover H₀."""
    cosmo = gl.cosmology.Cosmology(H0=70., Om0=0.3)
    delta_tau_arcsec2 = 0.05
    # Forward: predict Δt
    delta_t_s = float(gl.lens.timedelay.time_delay_seconds(
        torch.tensor(delta_tau_arcsec2), zl=0.5, zs=2.0, cosmo=cosmo,
    ))
    delta_t_days = delta_t_s / 86400.0
    # Inverse: recover H₀
    H0_rec = gl.lens.timedelay.refsdal_H0(
        delta_t_days, delta_tau_arcsec2, zl=0.5, zs=2.0,
    )
    assert abs(H0_rec - 70.0) < 0.5
