"""Tests for the microlensing models."""
from __future__ import annotations

import torch

import lensing as gl


def test_paczynski_peaks_at_t0():
    p = gl.lens.PaczynskiLightcurve(f=10.0, y0=0.1, t0=183.0, tE=20.0)
    t = torch.linspace(150., 220., 200)
    mu = p(t)
    peak_t = float(t[torch.argmax(mu)])
    assert abs(peak_t - 183.0) < 1.0


def test_pointmass_consistency_with_paczynski():
    """The minimal model should reproduce the physical-parameter model exactly."""
    phys = gl.lens.PointMassMicrolens(
        f=7.0, mass=0.3, y0=0.1, vel=200., t0=183., dl=4., ds=8.,
    )
    paczy = gl.lens.PaczynskiLightcurve(
        f=7.0, y0=0.1, t0=183., tE=float(phys.einstein_time()),
    )
    t = torch.linspace(150., 220., 50)
    assert torch.allclose(phys(t), paczy(t), atol=1e-4)


def test_paczynski_fit_recovers_truth():
    gl.config.setup(seed=0, device="cpu")
    truth = dict(f=7.0, y0=0.1, t0=183.0, tE=20.0)
    p = gl.lens.PaczynskiLightcurve(**truth)
    t = torch.arange(0., 365., 1.)
    sigma = 0.5
    _, mag = gl.data.simulate_lightcurve(p, t, noise_sigma=sigma, seed=1)

    fit_model = gl.lens.PaczynskiLightcurve(f=2.0, y0=0.5, t0=120., tE=10.)
    loss = gl.inference.ReducedChiSquared(sigma=sigma, n_params=4)
    res = gl.inference.fit(
        fit_model, t, mag, loss,
        lr=0.1, epochs=2000,
        scheduler=gl.inference.optimize.reduce_lr_on_plateau(),
        lbfgs_polish=True,
    )

    assert 0.5 < res.best_loss < 1.5  # reduced chi-squared near 1
    for k, v in truth.items():
        assert abs(float(getattr(fit_model, k)) - v) / abs(v) < 0.02, k
