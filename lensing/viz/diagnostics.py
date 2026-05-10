"""Rich diagnostic plots for fits and machine-learning results.

These wrappers combine 2-4 panels each, so a single function call
gives the same level of detail as a typical paper figure. They follow
the visual conventions of the rest of the package: ``inferno`` for
scalar maps, ``RdBu_r`` for residuals, log-stretched scalar plots
where appropriate.

Functions
---------
* :func:`plot_residual_diagnostics` — histogram + Q-Q + radial profile +
  2-D residual map of a fit's standardized residuals.
* :func:`plot_classification_diagnostics` — confusion matrix, ROC,
  precision-recall and reliability diagram on a single 2x2 grid.
* :func:`plot_regression_diagnostics` — predicted-vs-truth scatter,
  residual histogram, robust σ_residual annotation, per-bin error trend.
* :func:`plot_image_quality` — side-by-side PSNR / SSIM / pixelwise
  difference for U-Net-style image-to-image regression.
* :func:`format_summary` — pretty-print a dict of metrics as a
  Markdown-aligned table for use inside Jupyter ``print`` calls.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from ..stats import (
    calibration_curve,
    classification_report,
    expected_calibration_error,
    pr_curve,
    psnr,
    radial_residual_profile,
    roc_curve,
    ssim_simple,
    standardized_residuals,
)


# ---------------------------------------------------------------------------
# 1. Residual diagnostics for a 2-D fit
# ---------------------------------------------------------------------------
def plot_residual_diagnostics(
    data,
    model,
    sigma,
    *,
    title: str = "Fit residuals",
    extent=None,
):
    """Four-panel residual diagnostic.

    Panel A: 2-D map of standardized residuals (data − model)/σ.
    Panel B: histogram of std residuals + N(0,1) overlay.
    Panel C: Q-Q plot vs. standard normal.
    Panel D: radial residual profile (mean ± 1σ inside annular bins).
    """
    import matplotlib.pyplot as plt

    res = standardized_residuals(data, model, sigma)
    rh = np.asarray(res, dtype=float)
    if rh.ndim != 2:
        raise ValueError("plot_residual_diagnostics expects 2-D inputs")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    # A: residual map
    vmax = float(np.percentile(np.abs(rh), 99))
    im = axes[0, 0].imshow(
        rh, origin="lower", cmap="RdBu_r", extent=extent, vmin=-vmax, vmax=vmax
    )
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 0].set_title("(data − model) / σ")

    # B: histogram + standard normal overlay
    axes[0, 1].hist(rh.ravel(), bins=60, density=True, color="C0", alpha=0.5)
    x = np.linspace(-5, 5, 400)
    axes[0, 1].plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi),
                    "k-", lw=1.5, label="N(0, 1)")
    axes[0, 1].set(xlabel="standardized residual", ylabel="density",
                    title=f"distribution (mean={rh.mean():+.3f}, std={rh.std():.3f})",
                    xlim=(-5, 5))
    axes[0, 1].legend()

    # C: Q-Q plot vs Gaussian
    sorted_r = np.sort(rh.ravel())
    q_theory = _normal_quantiles(len(sorted_r))
    axes[1, 0].plot(q_theory, sorted_r, ".", ms=2, color="C0")
    lo, hi = q_theory.min(), q_theory.max()
    axes[1, 0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[1, 0].set(xlabel="theoretical N(0,1) quantile",
                    ylabel="sample quantile", title="Q-Q plot")
    axes[1, 0].grid(alpha=0.3)

    # D: radial residual profile
    r, mean, std = radial_residual_profile(rh, n_bins=15)
    axes[1, 1].errorbar(r, mean, yerr=std, marker="o", color="C0", capsize=3)
    axes[1, 1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1, 1].set(xlabel="r [pix]", ylabel="mean residual +/- 1 sigma",
                   title="radial residual profile")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes


def _normal_quantiles(n: int) -> np.ndarray:
    """Quantiles of a standard normal at probabilities (i+0.5)/n.

    Uses a Beasley-Springer-Moro-style approximation that does not
    require scipy.
    """
    p = (np.arange(n) + 0.5) / n
    # Acklam's approximation of the inverse standard-normal CDF.
    a = [-3.969683028665376e+1, 2.209460984245205e+2, -2.759285104469687e+2,
         1.383577518672690e+2, -3.066479806614716e+1, 2.506628277459239]
    b = [-5.447609879822406e+1, 1.615858368580409e+2, -1.556989798598866e+2,
         6.680131188771972e+1, -1.328068155288572e+1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
         3.754408661907416]
    p_low, p_high = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    lower = p < p_low
    upper = p > p_high
    middle = ~(lower | upper)
    q = np.sqrt(-2.0 * np.log(p[lower]))
    out[lower] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p[middle] - 0.5
    r = q * q
    out[middle] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                  (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    q = np.sqrt(-2.0 * np.log(1 - p[upper]))
    out[upper] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                  ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    return out


# ---------------------------------------------------------------------------
# 2. Classifier diagnostics
# ---------------------------------------------------------------------------
def plot_classification_diagnostics(
    probabilities,
    labels,
    *,
    threshold: float = 0.5,
    title: str = "Classifier diagnostics",
):
    """Confusion matrix + ROC + PR + reliability diagram on a 2x2 grid."""
    import matplotlib.pyplot as plt

    probs = np.asarray(probabilities).ravel()
    y = np.asarray(labels).astype(int).ravel()
    preds = (probs >= threshold).astype(int)
    rep = classification_report(preds, y)
    fpr, tpr, auc = roc_curve(probs, y)
    rec, prec, ap = pr_curve(probs, y)
    bin_c, pred_freq, obs_freq = calibration_curve(probs, y, n_bins=10)
    ece = expected_calibration_error(probs, y, n_bins=10)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(title, fontsize=14)

    # A: confusion matrix
    ax = axes[0, 0]
    im = ax.imshow(rep.confusion, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(rep.confusion[i, j]),
                    ha="center", va="center",
                    color="white" if rep.confusion[i, j] > rep.confusion.max()/2 else "black",
                    fontsize=14)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["truth=0", "truth=1"],
           yticklabels=["pred=0", "pred=1"],
           title=f"confusion matrix (acc={rep.accuracy:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # B: ROC curve
    ax = axes[0, 1]
    ax.plot(fpr, tpr, "C0-", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    ax.set(xlabel="false-positive rate", ylabel="true-positive rate",
           title="ROC")
    ax.legend(); ax.grid(alpha=0.3)

    # C: precision-recall
    ax = axes[1, 0]
    ax.plot(rec, prec, "C2-", lw=2, label=f"AP = {ap:.3f}")
    ax.set(xlabel="recall", ylabel="precision", title="precision-recall")
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    # D: reliability diagram
    ax = axes[1, 1]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.plot(pred_freq, obs_freq, "C3o-", lw=1.5,
            label=f"model (ECE={ece:.3f})")
    ax.set(xlabel="predicted probability",
           ylabel="observed frequency",
           title="reliability diagram",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes, dict(report=rep, auc=auc, ap=ap, ece=ece)


# ---------------------------------------------------------------------------
# 3. Regression diagnostics
# ---------------------------------------------------------------------------
def plot_regression_diagnostics(
    truths,
    predictions,
    *,
    param_names: Optional[Iterable[str]] = None,
    title: str = "Regression diagnostics",
):
    """Scatter (truth vs pred) + residual histogram for each output dim."""
    import matplotlib.pyplot as plt

    t = np.atleast_2d(np.asarray(truths))
    p = np.atleast_2d(np.asarray(predictions))
    if t.shape[1] != p.shape[1]:
        t = t.T; p = p.T
    n_out = t.shape[1]
    names = list(param_names) if param_names is not None else [f"y_{i}" for i in range(n_out)]

    n_cols = 2
    n_rows = n_out
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.5 * n_rows),
                              squeeze=False)
    fig.suptitle(title, fontsize=14)

    summary = {}
    for i, name in enumerate(names):
        ti = t[:, i]; pi = p[:, i]
        resid = pi - ti
        # robust σ_residual: 1.4826 × median absolute deviation
        mad = np.median(np.abs(resid - np.median(resid)))
        sigma_res = 1.4826 * mad
        bias = float(np.mean(resid))
        # scatter
        ax = axes[i, 0]
        ax.scatter(ti, pi, s=8, alpha=0.4, color="C0")
        lo, hi = float(ti.min()), float(ti.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1")
        r = np.corrcoef(ti, pi)[0, 1]
        ax.set(xlabel=f"true {name}", ylabel=f"predicted {name}",
               title=f"{name}:  r={r:.3f},  σ_res={sigma_res:.3f},  bias={bias:+.3f}")
        ax.legend(); ax.grid(alpha=0.3)
        # histogram
        ax = axes[i, 1]
        ax.hist(resid, bins=40, density=True, color="C0", alpha=0.6)
        ax.axvline(0, color="k", ls="--", lw=0.8)
        ax.axvline(bias, color="r", ls=":", lw=1.0, label=f"bias={bias:+.3f}")
        ax.set(xlabel=f"residual ({name})", ylabel="density",
               title=f"residual histogram, σ={sigma_res:.3f}")
        ax.legend(); ax.grid(alpha=0.3)
        summary[name] = dict(r=float(r), bias=bias, sigma=float(sigma_res))

    fig.tight_layout()
    return fig, axes, summary


# ---------------------------------------------------------------------------
# 4. Image-quality grid (for U-Net etc.)
# ---------------------------------------------------------------------------
def plot_image_quality(
    truths,
    predictions,
    *,
    n_show: int = 6,
    title: str = "Image regression",
):
    """Top: truth, middle: prediction, bottom: |truth − pred|.

    PSNR and SSIM are annotated per column.
    """
    import matplotlib.pyplot as plt

    t = np.asarray(truths)
    p = np.asarray(predictions)
    n_show = min(n_show, t.shape[0])

    fig, axes = plt.subplots(3, n_show, figsize=(2.5 * n_show, 7))
    fig.suptitle(title, fontsize=14)
    if n_show == 1:
        axes = axes[:, None]

    psnrs = []
    ssims = []
    for j in range(n_show):
        ti = np.squeeze(t[j])
        pi = np.squeeze(p[j])
        diff = np.abs(ti - pi)
        ps = psnr(ti, pi)
        ss = ssim_simple(ti, pi)
        psnrs.append(ps); ssims.append(ss)
        for ax, im, label in zip(
            axes[:, j],
            [ti, pi, diff],
            [f"truth", f"pred  PSNR={ps:.1f}dB  SSIM={ss:.2f}", "|diff|"],
        ):
            ax.imshow(im, origin="lower", cmap="inferno")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label.split()[0], fontsize=10)
            ax.set_title(label if j == n_show // 2 else "", fontsize=8)

    fig.tight_layout()
    summary = dict(
        psnr_mean=float(np.mean(psnrs)),
        psnr_std=float(np.std(psnrs)),
        ssim_mean=float(np.mean(ssims)),
        ssim_std=float(np.std(ssims)),
    )
    return fig, axes, summary


# ---------------------------------------------------------------------------
# 5. Pretty printing
# ---------------------------------------------------------------------------
def format_summary(metrics: Dict[str, float], title: Optional[str] = None) -> str:
    """Pretty-print a metric dict as a Markdown-aligned table.

    Returns a string with one row per ``metric: value``, suitable for
    ``print(format_summary(...))`` in a notebook cell.
    """
    width_k = max(len(k) for k in metrics)
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k.ljust(width_k)} : {v:>+12.4g}")
        else:
            lines.append(f"  {k.ljust(width_k)} : {v}")
    return "\n".join(lines)
