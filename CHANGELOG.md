# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Detailed `docs/usage.md` covering every notebook (01–18) and script
  (`run_microlensing_fit.py`, `run_sersic_fit.py`, `_make_notebooks.py`).

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
