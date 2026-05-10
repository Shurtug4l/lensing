"""Smoke tests for the statistics / validation helpers."""
from __future__ import annotations

import numpy as np

import lensing as gl


def test_chi2_dof_central_value():
    """For data == model the chi2 is exactly 0."""
    d = np.zeros(100); m = np.zeros(100)
    assert gl.stats.chi2_per_dof(d, m, sigma=1.0, n_params=0) == 0.0


def test_aic_bic_monotone_in_params():
    """Adding parameters should *increase* AIC and BIC at fixed loss."""
    a1 = gl.stats.aic(neg_loglike=10.0, n_params=2)
    a2 = gl.stats.aic(neg_loglike=10.0, n_params=5)
    b1 = gl.stats.bic(neg_loglike=10.0, n_params=2, n_samples=100)
    b2 = gl.stats.bic(neg_loglike=10.0, n_params=5, n_samples=100)
    assert a2 > a1
    assert b2 > b1
    # BIC penalises extra parameters more than AIC for N > e^2 ≈ 7.4.
    assert (b2 - b1) > (a2 - a1)


def test_classification_perfect():
    rep = gl.stats.classification_report(np.array([0, 1, 0, 1]),
                                          np.array([0, 1, 0, 1]))
    assert rep.accuracy == 1.0
    assert rep.precision == 1.0
    assert rep.recall == 1.0


def test_roc_auc_perfect_classifier():
    probs = np.linspace(0, 1, 100)
    labels = (probs > 0.5).astype(int)
    _, _, auc = gl.stats.roc_curve(probs, labels)
    assert auc > 0.99


def test_psnr_identity_is_inf():
    a = np.random.rand(16, 16)
    assert gl.stats.psnr(a, a) == float('inf')


def test_ssim_identity_is_one():
    a = np.random.rand(16, 16)
    assert abs(gl.stats.ssim_simple(a, a) - 1.0) < 1e-6


def test_kfold_indices_partition():
    """Every index appears in exactly one validation fold."""
    folds = list(gl.stats.kfold_indices(100, n_folds=5, shuffle=False))
    assert len(folds) == 5
    seen = np.concatenate([v for _, v in folds])
    assert sorted(seen.tolist()) == list(range(100))


def test_bootstrap_ci_brackets_truth():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=2.0, scale=1.0, size=500)
    point, lo, hi = gl.stats.bootstrap_ci(samples, statistic=np.mean,
                                            n_boot=500, seed=42)
    assert lo < 2.0 < hi
    assert lo < point < hi


def test_format_summary_returns_string():
    s = gl.viz.diagnostics.format_summary({'a': 0.5, 'b': 'text'}, title='Test')
    assert 'Test' in s
    assert 'a' in s and 'b' in s


def test_residual_diagnostics_runs():
    """End-to-end smoke test of the 4-panel residual diagnostic plot."""
    import matplotlib
    matplotlib.use('Agg')
    a = np.random.randn(32, 32)
    b = a + 0.1 * np.random.randn(32, 32)
    fig, axes = gl.viz.diagnostics.plot_residual_diagnostics(a, b, sigma=0.1)
    assert len(fig.axes) >= 4   # 4 panels + colorbars
