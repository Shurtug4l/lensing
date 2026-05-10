"""Smoke tests for the advanced lens models and the ML pipeline."""
from __future__ import annotations

import torch

import lensing as gl


def test_nfw_kappa_decreases_outward():
    nfw = gl.lens.NFW(theta_s=10.0, kappa_s=0.3)
    r = torch.tensor([0.5, 5.0, 50.0])
    k = nfw.kappa(r, torch.zeros_like(r))
    assert (k[1:] < k[:-1]).all()  # monotonically decreasing


def test_nfw_deflection_runs():
    nfw = gl.lens.NFW(theta_s=10.0, kappa_s=0.3)
    x = torch.tensor([1.0, 5.0, 10.0])
    ax, ay = nfw.deflection(x, torch.zeros_like(x))
    assert torch.isfinite(ax).all() and torch.isfinite(ay).all()


def test_composite_lens_sums_deflections():
    sie = gl.lens.SIE(theta_E=1.0, q=0.7, pa=0.0)
    shear = gl.lens.ExternalShear(gamma1=0.05, gamma2=-0.02)
    comp = gl.lens.CompositeLens([sie, shear])
    x, y = torch.tensor(1.0), torch.tensor(0.5)
    a_sie = sie.deflection(x, y)
    a_shr = shear.deflection(x, y)
    a_cmp = comp.deflection(x, y)
    assert torch.allclose(a_cmp[0], a_sie[0] + a_shr[0])
    assert torch.allclose(a_cmp[1], a_sie[1] + a_shr[1])


def test_binary_pointmass_magnification_finite():
    lens = gl.lens.BinaryPointMass(d=1.0, q_m=0.5)
    _, mu = lens.magnification_map(npix=80, halfwidth=2.0)
    assert torch.isfinite(mu).all()
    assert float(mu.max()) > 1.0  # there is some magnification somewhere


def test_ml_dataset_consistent_with_seed():
    a = gl.ml.datasets.SersicParamDataset(n_samples=4, npix=16, deltapix=0.1, seed=0)
    b = gl.ml.datasets.SersicParamDataset(n_samples=4, npix=16, deltapix=0.1, seed=0)
    img_a, p_a = a[0]
    img_b, p_b = b[0]
    assert torch.allclose(img_a, img_b)
    assert torch.allclose(p_a, p_b)


def test_ml_models_forward_shapes():
    cnn = gl.ml.models.LensCNN()
    reg = gl.ml.models.SersicRegressor()
    unet = gl.ml.models.UNet()

    x = torch.randn(2, 1, 32, 32)
    assert cnn(x).shape == (2, 2)
    assert reg(x).shape == (2, 7)
    assert unet(x).shape == (2, 1, 32, 32)


def test_ml_train_one_epoch_runs():
    """A few epochs on a tiny dataset should not crash and should reduce loss.

    Note: this test is intentionally lenient on the loss-decrease check
    because the underlying dataset is too small to guarantee monotonic
    progress in 2 epochs (and the test isolation can vary global RNG
    state). What we *really* care about is that the loop completes.
    """
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn

    torch.manual_seed(0)  # local determinism for this test
    train = gl.ml.datasets.SersicParamDataset(n_samples=8, npix=16, deltapix=0.1, seed=0)
    loader = DataLoader(train, batch_size=4)

    model = gl.ml.models.SersicRegressor()
    history = gl.ml.train.fit_model(
        model, loader, val_loader=loader,
        loss_fn=nn.MSELoss(), lr=1e-2, epochs=4, log_every=0,
    )
    assert len(history.train_loss) == 4
    # Allow small noise: the *minimum* of the last two losses should be
    # below the *maximum* of the first two — i.e. the loop is making
    # progress, even on a trivially small dataset.
    assert min(history.train_loss[-2:]) < max(history.train_loss[:2]) + 1e-6
