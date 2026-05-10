"""Smoke tests for the light models."""
from __future__ import annotations

import math

import torch

import lensing as gl


def _grid(npix=33, dx=0.1):
    return gl.data.coordinate_grid(npix=npix, deltapix=dx)


def test_sersic_circular_consistency():
    """For e1 = e2 = 0, R'^2 reduces to dx^2 + dy^2."""
    xy = _grid()
    s = gl.light.Sersic(Ie=1.0, Re=1.0, n=4.0, x0=0.0, y0=0.0, e1=0.0, e2=0.0)
    R = s.elliptical_radius(xy)
    expected = torch.sqrt(xy[0] ** 2 + xy[1] ** 2 + 1e-10)
    assert torch.allclose(R, expected, atol=1e-4)


def test_sersic_gradients_at_zero_ellipticity():
    """No NaN gradients at e1=e2=0 (this was the original bug)."""
    xy = _grid()
    s = gl.light.Sersic(Ie=2.0, Re=1.0, n=4.0, x0=0.0, y0=0.0, e1=0.0, e2=0.0)
    out = s(xy).sum()
    out.backward()
    for n, p in s.named_parameters():
        assert p.grad is not None, f"no grad for {n}"
        assert not torch.isnan(p.grad).any(), f"NaN in grad of {n}"


def test_sersic_recovers_truth():
    """Adam + L-BFGS recovers the input parameters."""
    gl.config.setup(seed=0, device="cpu")
    xy = _grid(npix=64, dx=0.05)
    truth = dict(Ie=5.0, Re=1.0, n=4.0, x0=0.1, y0=-0.2, e1=0.10, e2=-0.05)
    galaxy = gl.light.Sersic(**truth)
    sigma_n = 0.05
    _, image = gl.data.simulate_image(galaxy, xy, noise_sigma=sigma_n, seed=1)

    fit_model = gl.light.Sersic(Ie=2.0, Re=1.5, n=2.5, x0=0.0, y0=0.0, e1=0.0, e2=0.0)
    loss = gl.inference.ReducedChiSquared(sigma=sigma_n, n_params=7)
    res = gl.inference.fit(
        fit_model, xy, image, loss,
        lr=0.05, epochs=1500,
        scheduler=gl.inference.optimize.reduce_lr_on_plateau(patience=200, factor=0.7),
        grad_clip=10.0, lbfgs_polish=True,
    )

    assert res.best_loss < 5.0
    for k, v in truth.items():
        assert abs(float(getattr(fit_model, k)) - v) < 0.1, k


def test_double_sersic_sum_equals_components():
    xy = _grid()
    a = dict(Ie=5., Re=1.0, n=4., x0=-1.0, y0=0.0, e1=0.0, e2=0.0)
    b = dict(Ie=2., Re=2.0, n=1., x0=+1.0, y0=0.0, e1=0.0, e2=0.0)
    multi = gl.light.DoubleSersic(component1=a, component2=b)
    independent = gl.light.Sersic(**a)(xy) + gl.light.Sersic(**b)(xy)
    assert torch.allclose(multi(xy), independent, atol=1e-6)


def test_psf_kernel_normalized():
    k = gl.light.gaussian_psf_kernel(0.1, 0.05, size=11)
    assert math.isclose(float(k.sum()), 1.0, abs_tol=1e-6)
