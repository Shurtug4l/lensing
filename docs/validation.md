# Validation methodology

This note describes the validation strategy applied throughout the
notebooks: which metric is appropriate for which kind of fit, what a
"good" value looks like for each metric, and how to read the output of
the helpers in `lensing.stats` and `lensing.viz.diagnostics`.

## 1. Parametric fits (Adam / L-BFGS / NUTS)

The standard pipeline for the notebooks 01–04, 06, 07, 14, 15.

### 1.1 Goodness of fit

| Metric                       | Helper                              | What "good" means                                |
|------------------------------|-------------------------------------|--------------------------------------------------|
| reduced χ²                   | `gl.stats.chi2_per_dof`             | ≈ 1 ± few/√N — **the** primary indicator         |
| neg-log-likelihood           | `gl.stats.gaussian_neg_loglike`     | reported alongside AIC/BIC                       |
| AIC                          | `gl.stats.aic`                      | lower is better; differences > 10 strongly favour the smaller value |
| BIC                          | `gl.stats.bic`                      | as AIC, with a stronger penalty for parameters   |

### 1.2 Residual diagnostics

Standardized residuals `(d − m) / σ` should look ~ N(0, 1). Three
plots inspect this from different angles:

* **Histogram** with the N(0, 1) overlay: looks for the right
  width / mean.
* **Q-Q plot** against the standard normal: a tight diagonal means
  Gaussian; deviations in the tails indicate outliers or non-Gaussian
  noise (often unmodelled cosmic rays, bad pixels, neighbours).
* **Radial profile** (annulus-binned mean ± std): exposes systematic
  trends with radius — over-fit cores, under-fit outskirts, halo light
  not modelled.

The single function `gl.viz.diagnostics.plot_residual_diagnostics`
produces all three plots plus the 2-D residual map in a single
2 × 2 figure.

The Anderson-Darling test (`gl.stats.anderson_darling_normality`)
returns A² and a verdict (`'normal'` or `'non-normal'`) at the 5 %
significance level (critical value ≈ 0.752).

### 1.3 Posterior diagnostics (NUTS)

When using `gl.inference.run_nuts`:

* `gl.stats.gelman_rubin_rhat(chains)` (need ≥ 2 chains): R̂ < 1.05
  is the de-facto convergence threshold.
* `gl.stats.effective_sample_size(samples)` for autocorrelation-aware
  ESS; ESS / N_total > 0.1 is normally adequate.

## 2. Binary classification (notebook 10, 16)

The full validation set:

| Metric                | Helper                              | "Good" range                                     |
|-----------------------|-------------------------------------|--------------------------------------------------|
| accuracy              | `gl.stats.classification_report`    | depends on class balance                         |
| precision             | "                                   | high if false-positive cost is high              |
| recall                | "                                   | high if false-negative cost is high              |
| F1                    | "                                   | balanced precision/recall                        |
| ROC-AUC               | `gl.stats.roc_curve`                | ≳ 0.9 for usable lens-finders                    |
| Average Precision     | `gl.stats.pr_curve`                 | useful when the positive class is rare           |
| Expected Cal. Err.    | `gl.stats.expected_calibration_error`| ≲ 0.05 for trustable probabilities              |

The combined plot `plot_classification_diagnostics` draws all four
panels (confusion, ROC, PR, reliability) on a 2 × 2 grid.

**K-fold cross-validation** (`gl.stats.kfold_indices`) replaces a
single train/test split with K independent splits and reports the
mean ± std of any metric across folds. For tiny datasets, also
combine with bootstrap CIs on the K fold AUCs
(`gl.stats.bootstrap_ci`).

## 3. Regression (notebook 11, 17)

Per output dimension:

| Metric          | What it measures                                                |
|-----------------|-----------------------------------------------------------------|
| Pearson r       | linear correlation truth ↔ prediction (1 = perfect, 0 = random) |
| σ_residual      | robust 1-σ scatter (1.4826 × MAD); insensitive to outliers      |
| bias            | systematic offset; |bias| ≲ σ_residual / √N is consistent with zero |

`gl.viz.diagnostics.plot_regression_diagnostics` returns both the
plots and a dict with the three indicators per parameter for tabular
reporting.

**Rule of thumb**: σ_residual ≪ prior width means the regressor
learned something; σ_residual ≈ prior width means it predicts the
prior mean. For Sérsic-parameter regression at npix = 48 we observe
σ_Re ~ 0.05 arcsec and σ_n ~ 0.3 — well below the prior widths
(`Re ∈ [0.4, 1.5]`, `n ∈ [1, 6]`).

## 4. Image regression (notebook 12, U-Net)

Two image-quality metrics, computed per test sample and reported as
mean ± std:

| Metric          | Helper                  | "Good"                                           |
|-----------------|-------------------------|--------------------------------------------------|
| PSNR (dB)       | `gl.stats.psnr`         | > 30 dB visually indistinguishable; 25–35 dB typical for source reconstruction |
| SSIM ∈ [-1, 1]  | `gl.stats.ssim_simple`  | > 0.9 morphology preserved; > 0.95 excellent     |

`plot_image_quality` shows the truth / prediction / |difference|
column-by-column for `n_show` test samples plus the global summary.

## 5. Cross-checked extraction (notebook 18, LLM)

For metadata extraction we compute **field-level precision / recall**:
each (paper × field) cell is a binary task — was the value present in
the abstract correctly extracted? FN means the LLM missed a value
that *was* there; FP means the LLM produced a number that wasn't.

Cross-validation is against the embedded `slacs_table()` reference
catalog: for every system whose `theta_E` and `sigma_v` are *also* in
the catalog, the extracted value should match within the published
uncertainty (typically ± 0.01 arcsec for `theta_E` and ± 5 km/s for
`sigma_v`).

## 6. Bootstrap confidence intervals

Whenever a metric is summarised over a small sample (folds of CV,
images per test set, etc.), use `gl.stats.bootstrap_ci(samples,
statistic=np.mean)` to attach a percentile CI. This is
distribution-free and works for any statistic, at the cost of being
sensitive to the sample being representative of the population.

## How to read the printed summaries

Every notebook section that runs validation ends with a call to
`gl.viz.diagnostics.format_summary(metrics_dict, title=...)`. The
output is a Markdown-aligned table with one row per metric:

```
=== CNN classifier — test-set summary ===
  accuracy  : +0.93
  precision : +0.91
  recall    : +0.95
  F1        : +0.93
  ROC-AUC   : +0.96
  PR-AP     : +0.94
  ECE (10b) : +0.04
```

This is the canonical "publishable table" produced by every notebook,
to which the diagnostic plots add the visual companion.
