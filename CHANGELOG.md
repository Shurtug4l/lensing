# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Detailed `docs/usage.md` covering every notebook (01–18) and script
  (`run_microlensing_fit.py`, `run_sersic_fit.py`, `_make_notebooks.py`).
- **`lensing.stats` module** with goodness-of-fit (χ²/dof, AIC, BIC),
  bootstrap CIs, residual diagnostics (Anderson-Darling normality,
  radial residual profile), classification metrics
  (precision/recall/F1, ROC/PR/AUC), probability calibration
  (reliability curve, ECE), image-regression quality (PSNR, SSIM),
  MCMC diagnostics (Gelman-Rubin R̂, effective sample size) and
  k-fold CV index generator.
- **`lensing.viz.diagnostics` module** with rich multi-panel figures:
  `plot_residual_diagnostics` (4-panel), `plot_classification_diagnostics`
  (confusion + ROC + PR + reliability), `plot_regression_diagnostics`
  (per-parameter scatter + residual hist), `plot_image_quality`
  (truth/pred/diff with PSNR & SSIM annotations), and
  `format_summary` for Markdown-aligned metric tables.
- **Validation sections** added to notebooks 02 (Sérsic), 06 (SIE
  inversion), 10 (CNN), 11 (DNN regressor), 12 (U-Net), 16 (lens
  finder + k-fold CV), 17 (parameter recovery), 18 (LLM extraction).
  Each section ends with a printed summary table of metrics.
- New `docs/validation.md` describing the validation strategy
  end-to-end (which metric for which fit, what "good" looks like).
- 10 new pytest smoke tests for `lensing.stats` (40/40 total).
- **Section 0 "Conventions and units"** at the top of
  `docs/background.md` — single source of truth for the unit system
  (arcsec, Mpc, M_⊙, km/s, days), the coordinate convention, the
  shear-sign convention, and the default cosmology.
- Per-module unit tables in the docstrings of `cosmology`,
  `microlens`, `sie`, `nfw`, `power_law` and `nie`.

### Changed
- **Default device is now ``None`` (auto-detect)** rather than
  ``"cpu"`` — `gl.config.setup()` returns the best available
  accelerator (MPS → CUDA → CPU). Pass ``device="cpu"`` explicitly to
  force the CPU path.
- :func:`lensing.ml.train.fit_model` moves the model and every batch
  to the chosen device automatically; ML notebooks (10-12, 16-17) now
  transparently train on MPS where available, with the inference cells
  shuttling tensors back to CPU only for plotting.

### Fixed
- **NFW convergence at the scale radius** (`x = 1`): previously the
  formula returned ~0 due to a 0/0 in `(1 - F(x))/(x² - 1)`; now the
  analytic limit κ(1) = 2 κ_s / 3 (Wright & Brainerd 2000 Eq. 11
  middle line) is used inside a small ε-ball.
- **Binary-microlens magnification map**: the previous version
  returned the *count* of image-plane rays per source-plane bin
  (proportional to but not equal to μ); now it divides by the expected
  unlensed count (= ``oversample²``) and oversamples the image plane
  by 3× by default to suppress shot noise. The output is now a true
  dimensionless μ field.
- Noise functions, dataset / archive / LLM modules, and every package
  `__init__.py` got proper module-level docstrings explaining the
  *why* of each function and the units involved.

## [0.2.0] — 2024-Q4

### Added
- **Big-data module** (`lensing.bigdata`) with HDF5 dataset generator and
  lazy DataLoader-compatible reader.
- **Public-archive downloader** (`lensing.archive`) for HSC PDR3 cutouts and
  an embedded SLACS-lite catalog.
- **LLM metadata extractor** (`lensing.llm`) with both a regex-based mock
  backend and an Anthropic-API backend (Claude Haiku 4.5 with prompt cache).
- **CPU vs MPS benchmarks** module (`lensing.benchmarks`) with proper device
  synchronisation and pre-built workloads.
- **Power-law and NIE lens models** (`lensing.lens.PowerLawSpherical`,
  `lensing.lens.SIS`, `lensing.lens.NIE`).
- **Time-delay machinery** (`lensing.lens.timedelay`): Fermat potential,
  time-delay distance, Refsdal H₀ inversion.
- Notebooks 13–18:
  - `13_cpu_vs_mps_benchmarks.ipynb`
  - `14_power_law_and_NIE_lenses.ipynb`
  - `15_time_delay_cosmography.ipynb`
  - `16_large_scale_lens_finder.ipynb`
  - `17_param_recovery_at_scale.ipynb`
  - `18_LLM_metadata_extraction.ipynb`
- Documentation: `docs/background.md` (theory), `docs/astrophysics.md`
  (applications), `docs/benchmarks.md` (measured numbers).
- 30 pytest smoke tests across all modules.

### Changed
- Package directory promoted from `good_lensing/good_lensing/` to top-level
  `lensing/lensing/`; flattened layout, single namespace.
- Notebook bootstrap simplified to `import lensing as gl`.

## [0.1.0] — 2024-Q4

### Added
- Initial reorganization of the MSc thesis code into a tested Python package.
- Lens models: SIE, NFW, PointMass / Paczynski microlens, BinaryPointMass,
  ExternalShear, CompositeLens.
- Light models: Sérsic, CoreSersic, MultiLight (DoubleSersic).
- Inference: Adam + L-BFGS fit loop, NUTS posterior wrapper, ellipticity
  estimators (parametric + KSB-style).
- ML module: synthetic datasets + LensCNN / SersicRegressor / U-Net.
- Visualization helpers: `imshow_log`, `corner_plot`, residuals, side-by-side.
- Notebooks 01–12 reproducing thesis results and extending them with cluster
  / binary / deep-learning case studies.
- 12 smoke tests; full notebook execution validated via `nbconvert`.
