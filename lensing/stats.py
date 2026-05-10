"""Statistical analysis, model-comparison and validation helpers.

A focused collection of utilities used by the notebooks to validate fits
and compare models. The functions are deliberately self-contained
(no scikit-learn dependency) so the package keeps a small footprint.

Topics covered
--------------
1. **Goodness of fit**: ``chi2_per_dof``, ``aic``, ``bic`` (Schwarz 1978).
2. **Bootstrap confidence intervals**: ``bootstrap_ci``.
3. **Residual diagnostics**: ``standardized_residuals``,
   ``anderson_darling_normality``, ``radial_residual_profile``.
4. **Classification metrics**: ``classification_report``,
   ``roc_curve``, ``pr_curve``, ``auc_trapezoid``.
5. **Probability calibration**: ``calibration_curve``,
   ``expected_calibration_error``.
6. **Image-regression quality**: ``psnr``, ``ssim_simple``.
7. **MCMC diagnostics**: ``gelman_rubin_rhat``,
   ``effective_sample_size``.
8. **Cross-validation**: ``kfold_indices``.

All functions accept either ``numpy`` arrays or ``torch.Tensor`` and
return plain ``numpy`` (or scalar) values for downstream plotting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


def _np(x) -> np.ndarray:
    """Coerce torch.Tensor / list / scalar to numpy array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 1. Goodness of fit
# ---------------------------------------------------------------------------
def chi2_per_dof(
    data: np.ndarray, model: np.ndarray, sigma: float | np.ndarray, n_params: int
) -> float:
    """Reduced chi-squared ``χ²/dof`` (dof = N - p).

    A χ²/dof ≈ 1 indicates a fit consistent with the noise model.
    χ²/dof >> 1 means the model is too rigid or σ is underestimated;
    χ²/dof << 1 means σ is overestimated (or the model is overfit).
    """
    d = _np(data); m = _np(model); s = _np(sigma)
    n = d.size - int(n_params)
    return float(np.sum(((d - m) / s) ** 2) / max(n, 1))


def aic(neg_loglike: float, n_params: int) -> float:
    """Akaike Information Criterion: AIC = 2k − 2 ln L (Akaike 1974)."""
    return 2.0 * n_params + 2.0 * neg_loglike


def bic(neg_loglike: float, n_params: int, n_samples: int) -> float:
    """Bayesian Information Criterion: BIC = k ln N − 2 ln L (Schwarz 1978)."""
    return n_params * np.log(max(n_samples, 1)) + 2.0 * neg_loglike


def gaussian_neg_loglike(
    data: np.ndarray, model: np.ndarray, sigma: float | np.ndarray
) -> float:
    """``-ln L`` for an i.i.d. Gaussian likelihood with known σ.

    Useful as the input to :func:`aic` / :func:`bic`. The constant
    ``N/2 · ln(2π σ²)`` term is included so the value is comparable
    across models with the same σ (and cancels in differences).
    """
    d = _np(data); m = _np(model); s = _np(sigma)
    if np.ndim(s) == 0:
        s_arr = float(s) * np.ones_like(d)
    else:
        s_arr = s
    return float(
        0.5 * np.sum(((d - m) / s_arr) ** 2)
        + np.sum(np.log(s_arr))
        + 0.5 * d.size * np.log(2.0 * np.pi)
    )


# ---------------------------------------------------------------------------
# 2. Bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(
    samples: np.ndarray,
    statistic=np.mean,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> Tuple[float, float, float]:
    """Bootstrap (resample-with-replacement) confidence interval.

    Returns ``(point_estimate, lower, upper)`` for the chosen statistic
    at the given confidence level. Default is a 95% percentile CI.
    """
    s = _np(samples)
    rng = np.random.default_rng(seed)
    n = s.size
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = statistic(s[idx])
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return float(statistic(s)), float(lo), float(hi)


# ---------------------------------------------------------------------------
# 3. Residual diagnostics
# ---------------------------------------------------------------------------
def standardized_residuals(
    data: np.ndarray, model: np.ndarray, sigma: float | np.ndarray
) -> np.ndarray:
    """Per-pixel (data − model) / σ. Should be ~ N(0, 1) for a good fit."""
    return (_np(data) - _np(model)) / _np(sigma)


def anderson_darling_normality(residuals: np.ndarray) -> Tuple[float, str]:
    """Anderson-Darling test for normality of the standardized residuals.

    Returns ``(A², verdict)`` where ``verdict ∈ {'normal','non-normal'}``
    using the standard 5%-significance critical value (≈ 0.752 for the
    Anderson-Darling statistic with unknown mean/variance).
    """
    try:
        from scipy import stats as scstats
        A2 = float(scstats.anderson(_np(residuals).ravel(), dist="norm").statistic)
    except ImportError:
        # Pure-numpy fallback: not as accurate for large samples but
        # adequate for the docstring's pass/fail verdict.
        x = np.sort(_np(residuals).ravel())
        n = x.size
        # Use sample mean & std (Stephens 1974 modification).
        z = (x - x.mean()) / (x.std(ddof=1) + 1e-30)
        from math import log
        s = sum((2 * i + 1) * (log(_phi(z[i]) + 1e-30)
                              + log(1.0 - _phi(z[n - 1 - i]) + 1e-30))
                for i in range(n))
        A2 = -n - s / n
    verdict = "normal" if A2 < 0.752 else "non-normal"
    return A2, verdict


def _phi(x: float) -> float:
    """Std-normal CDF without scipy."""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def radial_residual_profile(
    residuals_2d: np.ndarray,
    center: Tuple[float, float] | None = None,
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin a 2-D residual map into radial annuli.

    Returns ``(r_centers, mean_residual, std_residual)`` — useful to spot
    systematic radial trends (under/over-fitting at small/large R).
    """
    r2d = _np(residuals_2d)
    H, W = r2d.shape
    cy, cx = (H / 2 - 0.5, W / 2 - 0.5) if center is None else center
    yy, xx = np.indices(r2d.shape)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    edges = np.linspace(0, r.max(), n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    mean = np.full(n_bins, np.nan)
    std = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (r >= edges[i]) & (r < edges[i + 1])
        if mask.any():
            mean[i] = r2d[mask].mean()
            std[i] = r2d[mask].std()
    return centers, mean, std


# ---------------------------------------------------------------------------
# 4. Classification metrics
# ---------------------------------------------------------------------------
@dataclass
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray   # shape (2, 2): [pred_neg/pos][truth_neg/pos]


def classification_report(
    preds: np.ndarray, labels: np.ndarray
) -> ClassificationReport:
    """Binary classification metrics.

    Conventions:
    * positive class = label 1
    * confusion[i, j] = count where pred==i and truth==j
    """
    p = _np(preds).astype(int).ravel()
    y = _np(labels).astype(int).ravel()
    cm = np.array([[((p == i) & (y == j)).sum() for j in (0, 1)] for i in (0, 1)])
    tn, fn = cm[0, 0], cm[0, 1]
    fp, tp = cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(cm.sum(), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-30)
    return ClassificationReport(
        accuracy=float(acc),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        confusion=cm,
    )


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """Area under a curve sorted by x, via the trapezoidal rule."""
    x_, y_ = _np(x), _np(y)
    order = np.argsort(x_)
    # `np.trapezoid` was added in NumPy 2.0; fall back to `np.trapz` on
    # older versions where the new spelling is missing.
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(y_[order], x_[order]))


def roc_curve(
    probabilities: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """ROC curve. Returns ``(fpr, tpr, auc)``."""
    p = _np(probabilities).ravel()
    y = _np(labels).astype(int).ravel()
    order = np.argsort(-p)
    y_sorted = y[order]
    P = y_sorted.sum()
    N = len(y_sorted) - P
    tp_curve = np.concatenate([[0], np.cumsum(y_sorted == 1)]) / max(P, 1)
    fp_curve = np.concatenate([[0], np.cumsum(y_sorted == 0)]) / max(N, 1)
    return fp_curve, tp_curve, auc_trapezoid(fp_curve, tp_curve)


def pr_curve(
    probabilities: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Precision-recall curve. Returns ``(recall, precision, average_precision)``."""
    p = _np(probabilities).ravel()
    y = _np(labels).astype(int).ravel()
    order = np.argsort(-p)
    y_sorted = y[order]
    P = max(y_sorted.sum(), 1)
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    rec = tp / P
    prec = tp / np.maximum(tp + fp, 1)
    # Average precision = sum_n (R_n - R_{n-1}) * P_n
    ap = float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))
    return rec, prec, ap


# ---------------------------------------------------------------------------
# 5. Calibration
# ---------------------------------------------------------------------------
def calibration_curve(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability diagram for a probabilistic classifier.

    Returns ``(bin_centers, predicted_freq, observed_freq)``. A perfectly
    calibrated model has ``predicted_freq == observed_freq``.
    """
    p = _np(probabilities).ravel()
    y = _np(labels).astype(int).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    pred_freq = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.any():
            pred_freq[i] = p[mask].mean()
            obs_freq[i] = y[mask].mean()
    return centers, pred_freq, obs_freq


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (Naeini+ 2015)."""
    p = _np(probabilities).ravel()
    y = _np(labels).astype(int).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(p)
    for i in range(n_bins):
        mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.any():
            ece += (mask.sum() / n_total) * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# 6. Image-regression quality
# ---------------------------------------------------------------------------
def psnr(
    truth: np.ndarray, prediction: np.ndarray, data_range: float | None = None
) -> float:
    """Peak Signal-to-Noise Ratio in dB.

    PSNR = 20 log10 (data_range / RMSE). Higher is better; > 30 dB is
    typical for "visually indistinguishable" image regression. If
    ``data_range`` is None it is inferred from ``truth.max() - truth.min()``.
    """
    t = _np(truth); p = _np(prediction)
    rmse = float(np.sqrt(np.mean((t - p) ** 2)))
    if data_range is None:
        data_range = float(t.max() - t.min())
    if rmse == 0 or data_range == 0:
        return float("inf")
    return float(20.0 * np.log10(data_range / rmse))


def ssim_simple(
    truth: np.ndarray, prediction: np.ndarray, data_range: float | None = None
) -> float:
    """Single-window SSIM (Wang+ 2004 with the standard K1 = 0.01, K2 = 0.03).

    The full multi-scale SSIM lives in scikit-image; this is a
    dependency-free version sufficient for the U-Net QC metric in
    notebook 12.
    """
    t = _np(truth).astype(float); p = _np(prediction).astype(float)
    if data_range is None:
        data_range = float(t.max() - t.min())
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu_t, mu_p = t.mean(), p.mean()
    var_t, var_p = t.var(), p.var()
    cov = np.mean((t - mu_t) * (p - mu_p))
    num = (2 * mu_t * mu_p + C1) * (2 * cov + C2)
    den = (mu_t ** 2 + mu_p ** 2 + C1) * (var_t + var_p + C2)
    return float(num / max(den, 1e-30))


# ---------------------------------------------------------------------------
# 7. MCMC diagnostics
# ---------------------------------------------------------------------------
def gelman_rubin_rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin R-hat for a (n_chains, n_samples) array.

    Values close to 1 indicate convergence; > 1.1 typically means the
    chains have not mixed (Gelman & Rubin 1992).
    """
    c = _np(chains)
    if c.ndim != 2 or c.shape[0] < 2:
        raise ValueError("Need at least 2 chains as a (M, N) array")
    _, N = c.shape
    chain_means = c.mean(axis=1)
    chain_vars = c.var(axis=1, ddof=1)
    B = N * np.var(chain_means, ddof=1)        # between-chain variance
    W = chain_vars.mean()                       # within-chain variance
    var_hat = ((N - 1) / N) * W + B / N
    return float(np.sqrt(var_hat / max(W, 1e-30)))


def effective_sample_size(samples: np.ndarray) -> float:
    """Effective sample size from autocorrelation (Geyer 1992 IPS).

    Cheap version: integrate autocorrelation up to the first negative
    lag. Adequate for posterior-quality diagnostics in the notebooks;
    use ArviZ for production work.
    """
    x = _np(samples).ravel().astype(float)
    n = x.size
    x = x - x.mean()
    f = np.fft.fft(np.concatenate([x, np.zeros_like(x)]))
    acf = np.real(np.fft.ifft(f * np.conj(f))[:n])
    acf = acf / acf[0]
    # Sum until first negative.
    s = 1.0
    for k in range(1, n):
        if acf[k] < 0:
            break
        s += 2 * acf[k]
    return float(n / max(s, 1.0))


# ---------------------------------------------------------------------------
# 8. Cross-validation
# ---------------------------------------------------------------------------
def kfold_indices(
    n_samples: int, n_folds: int = 5, shuffle: bool = True, seed: int = 0
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Generator of ``(train_idx, val_idx)`` pairs for k-fold CV."""
    idx = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    for k in range(n_folds):
        val = folds[k]
        train = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        yield train, val
