# Usage guide — running every notebook and script

This document describes, **for each artifact in the repository**, what
it does, how to run it, what parameters you can tune and what outputs
to expect. Sections are ordered the same way the notebooks are
numbered in `notebooks/`.

## Quick start

```bash
# 1. Install dependencies (~ 3 min)
cd lensing/
pip install -r requirements.txt
# OR via conda:
conda env create -f environment.yml && conda activate lensing

# 2. Verify the install
pytest tests/        # 30/30 should pass in ~10 s

# 3. Open the notebooks
jupyter lab notebooks/
```

Every notebook bootstraps itself with this top cell, which prepends
the repository root to `sys.path`:

```python
import sys
from pathlib import Path
repo = Path.cwd().resolve().parent
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import lensing as gl
device, dtype = gl.config.setup(seed=42, device="cpu")
```

You can therefore run notebooks without `pip install`-ing the package.
Set `device="mps"` for Apple GPU, `"cuda"` for NVIDIA, `None` for
auto-detection.

---

## Common controls

| Knob              | Where                                 | Effect                                           |
|-------------------|---------------------------------------|--------------------------------------------------|
| `seed`            | `gl.config.setup(seed=…)`             | Reproducibility for NumPy + torch RNGs           |
| `device`          | `gl.config.setup(device=…)`           | "cpu" / "mps" / "cuda" / `None` (auto, default)  |
| `npix, deltapix`  | `gl.data.coordinate_grid(...)`        | Image-plane grid size and pixel scale (arcsec)   |
| `psf_fwhm`        | `gl.light.gaussian_psf_kernel(...)`   | PSF width (arcsec)                               |
| `noise_sigma`     | `gl.data.simulate_image(...)`         | Gaussian noise std (flux units)                  |
| `lr`, `epochs`    | `gl.inference.fit(...)`               | Adam learning rate, epoch budget                 |
| `lbfgs_polish`    | `gl.inference.fit(...)`               | Append L-BFGS polish (recommended for fits)      |
| `RUN_NUTS`        | inside posterior cells                | `False` → read cached CSV; `True` → resample     |

### Device-agnostic by default

Since v0.3 the bootstrap cell calls

```python
device, dtype = gl.config.setup(seed=42)   # device defaults to None
```

which **auto-detects the best available accelerator** (preference
order: ``mps`` → ``cuda`` → ``cpu``). To force the CPU path (e.g. for
operators with no MPS kernel yet, or for strict reproducibility),
pass ``device="cpu"``. ML notebooks (10–12, 16–17) automatically move
their model and batch tensors to the chosen device through
:func:`lensing.ml.train.fit_model`.

## Units and conventions (cheat-sheet)

A single source of truth is `docs/background.md` (Sec. 0). The most
important entries:

| Symbol               | Unit                            |
|----------------------|---------------------------------|
| angles `θ, β, α`     | arcsec                          |
| distances `D_*`      | Mpc                             |
| velocity dispersion  | km/s                            |
| time `t, t_E, Δt`    | days                            |
| mass (microlensing)  | M_⊙ (solar masses)              |
| `Ψ`, `τ`             | arcsec²                         |
| `κ, γ, μ, q, e1, e2` | dimensionless                   |

Sky-aligned right-handed Cartesian; positive `x` East, positive `y`
North. Sign convention for the SIE shear is Kormann+1994.

---

# Notebook reference

> Every notebook is self-contained: bootstrap → simulate → fit →
> inspect. Cells are numbered in the actual files, but here we group
> them by purpose.

## 01 — `01_microlensing_lightcurve.ipynb`

Microlensing light-curve fit (Paczynski model).

**Steps**:
1. Build a physical-parameter `gl.lens.PointMassMicrolens` (M, v_rel,
   D_L, D_S) and use `gl.data.simulate_lightcurve` to produce noisy
   data.
2. Fit a minimal `gl.lens.PaczynskiLightcurve` with
   `gl.inference.fit(..., lbfgs_polish=True)`.
3. (Optional) Run NUTS posterior — set `RUN_NUTS = True`. ~30 s; cached
   in `notebooks/cache/posterior_microlens.csv`.

**Tunable**: `phys` parameters; `t = torch.arange(0., 365., 1.)` time
grid; `sigma` measurement noise; `epochs` for Adam.

**Expected output**: chi²/dof ≈ 1; recovered `(f_S, t_0, y_0, t_E)`
within 0.1% of truth in ≲ 5 s on CPU. Corner plot if `RUN_NUTS=True`.

## 02 — `02_sersic_fit.ipynb`

Sérsic galaxy fit (single component, with PSF and noise).

**Steps**:
1. Generate a noisy `gl.light.Sersic` image at `npix=128`, `deltapix=0.05`.
2. Fit a `SersicPSF` wrapper (Sérsic + Gaussian PSF) with Adam +
   L-BFGS.
3. Plot data / best fit / residual.
4. Optional NUTS posterior with corner plot.

**Tunable**: `true` parameter dict; PSF FWHM (`0.10` arcsec) and pixel
scale (`0.05` arcsec); `lr`, `epochs` (default 2500).

**Expected output**: chi²/dof ≈ 1, parameters recovered within ~0.1%.

## 03 — `03_core_sersic_fit.ipynb`

Core-Sérsic profile fit (Graham+ 2003) — for massive ellipticals with
a depleted central core.

**Steps**:
1. Simulate a Core-Sérsic with **moderate central core** to keep the
   `(R_b/R)^α` term well-conditioned (default `Rb=0.4`, `α=4`).
2. Plot the profile cross-section; compare to a pure Sérsic.
3. Fit using `gl.inference.log_image_mse` (compresses the ~10-decade
   dynamic range of galaxy brightness).

**Tunable**: dictionary `true` of all 10 Core-Sérsic params; loss
choice (`log_image_mse` vs `GaussianNLL`); `grad_clip=1.0` is
deliberately low because the gradients can spike when the optimizer
moves into the inner-power-law regime.

**Expected output**: best-fit parameters within a few % of truth.

## 04 — `04_double_sersic_fit.ipynb`

Bulge + disk decomposition (14 free parameters).

**Steps**:
1. Build a `gl.light.DoubleSersic(component1=..., component2=...)`.
2. Fit with Adam + L-BFGS.
3. Print per-component truth vs fit.

**Tunable**: `true_a`, `true_b` dicts, init dicts (initial guesses
matter — start the centres near where the bulge / disk obviously sit
in the image).

**Expected output**: ~10–15% per-parameter residuals after 3000
epochs; tighter with more epochs.

## 05 — `05_weak_lensing_ellipticity.ipynb`

Two ellipticity estimators on a Sérsic galaxy:

1. Parametric forward fit via `gl.inference.fit_ellipticity`.
2. Quadrupole-moment KSB-style estimator
   `gl.inference.kaiser_squires_estimator`.

**Tunable**: `weight_sigma` for the Gaussian aperture in the
non-parametric estimator; PSF FWHM and noise_sigma for the simulation.

**Expected output**: parametric `(e1, e2)` recovered to ~0.005,
quadrupole estimator biased low by ~10–20% (no PSF deconvolution).

## 06 — `06_strong_lensing_SIE.ipynb`

SIE caustics, critical curves, image solver and lens inversion.

**Steps**:
1. Build a `gl.lens.SIE.from_velocity_dispersion(...)`.
2. Plot the tangential caustic / critical / cut curves and overlay
   the multiple images of a point source.
3. Plot the convergence and magnification maps.
4. Recover the lens parameters from observed image positions
   (lens inversion).

**Tunable**: `sigma_v_kms`, `q`, `pa`, `zl`, `zs`; source position
`(beta_x, beta_y)`; image-plane grid resolution.

**Expected output**: 4 images for the worked example, `(theta_E, q,
pa)` recovered to 1% by the inversion fit.

## 07 — `07_real_galaxy_F150W.ipynb`

Single-Sérsic + PSF fit on a **real JWST NIRCam F150W postage stamp**.

**Steps**:
1. Read `data/raw/TEST_F150W_NIRCAM.fits`, median-sky-subtract, crop a
   128×128 patch around the central source.
2. Forward-fit a Sérsic + Gaussian PSF.
3. Plot data / best fit / residual.

**Tunable**: `FITS_PATH` to point at your own postage stamp;
`DELTAPIX = 0.031` (NIRCam SW pixel scale); `PSF_FWHM = 0.05` (F150W
diffraction limit); `half_size` controls the crop.

**Expected output**: chi²/dof close to 1 if the postage stamp contains
a single isolated galaxy; visible residuals if the source is
multi-component or contaminated by neighbours.

## 08 — `08_binary_microlens.ipynb`

Binary point-mass microlens — caustic topology and light curves.

**Steps**:
1. Compute magnification maps for `(d, q_m)` in the close /
   intermediate / wide regimes.
2. Overlay numerical critical curves (det A = 0).
3. Sample a magnification along a straight source trajectory →
   caustic-crossing light curve.

**Tunable**: lens separation `d`, mass ratio `q_m`, source-trajectory
parameters (start/end `beta_x`, fixed `beta_y` impact parameter).

**Expected output**: visibly different caustic topologies in the three
regimes; sharp magnification spikes at caustic crossings.

## 09 — `09_galaxy_cluster_NFW.ipynb`

Cluster-scale lensing (NFW + member SIEs + external shear) and a
real-data reference (MACS J1206 HST stack).

**Steps**:
1. Build a `gl.lens.CompositeLens([NFW, SIE_a, SIE_b, ExternalShear])`.
2. Plot the convergence map (NFW dominated).
3. Compute critical curves numerically (det A = 0 via central
   finite-differences on the lens map).
4. Side-by-side with `data/raw/macs1206_stack.fits`.

**Tunable**: NFW `theta_s`, `kappa_s`; member-galaxy positions and
Einstein radii; external-shear amplitude. The MACS J1206 cell
gracefully skips if the FITS is missing.

**Expected output**: a roughly elliptical primary critical curve
spanning ~30 arcsec, with secondary "wings" at the member-galaxy
positions; the HST stack shows the giant-arc system.

## 10 — `10_CNN_lens_classifier.ipynb`

Train a small CNN to classify lens vs. no-lens images.

**Steps**:
1. Construct `gl.ml.datasets.LensClassifierDataset` (on-the-fly,
   400 train / 100 val).
2. Train `gl.ml.models.LensCNN` for 8 epochs at lr=1e-3.
3. Plot loss / accuracy curves; print confusion matrix on a fresh
   200-sample test set.

**Tunable**: `n_samples`, `npix`, `lr`, `epochs`; `seed` for
reproducibility.

**Expected output**: ≳ 80% test accuracy on this small toy problem;
notebook 16 scales the same idea to 5,000 samples.

## 11 — `11_DNN_param_regression.ipynb`

DNN regressor for Sérsic parameters.

**Steps**:
1. Build `gl.ml.datasets.SersicParamDataset` (image → 7-vector).
2. Train `gl.ml.models.SersicRegressor` for 10 epochs.
3. Per-parameter scatter plots (`pred` vs `truth`) with Pearson r
   reported.
4. Wall-time comparison vs. classical Adam fit on a single image.

**Tunable**: dataset size, model depth, `epochs`.

**Expected output**: r ≳ 0.9 for `(Ie, Re, n, x0, y0)`, r ≳ 0.8 for
`(e1, e2)`; the DNN is ~50× faster than 500-epoch Adam at inference.

## 12 — `12_unet_source_reconstruction.ipynb`

U-Net mapping a lensed image to its source-plane reconstruction.

**Steps**:
1. Generate `gl.ml.datasets.LensSourcePairDataset` (observed →
   unlensed source).
2. Train `gl.ml.models.UNet(base=16)` for 8 epochs.
3. Show 8 reconstructions side by side: observed / U-Net / truth.

**Tunable**: dataset size; U-Net `base` channels (16 → 32 doubles
parameter count); `epochs`.

**Expected output**: visually correct source-plane recovery on most
test samples after 8 epochs; some failures on configurations where the
arc is very close to the caustic.

## 13 — `13_cpu_vs_mps_benchmarks.ipynb`

Quantify the CPU vs MPS performance trade-off.

**Steps**:
1. Print available devices and PyTorch / platform info.
2. Sweep image size (32 → 512) for forward and backward, comparing
   wall time on every device. Plot log-log.
3. End-to-end Adam fit timing at `npix=128, epochs=500`.
4. U-Net training (one epoch) — the conv-heavy benchmark where MPS
   wins.

**Tunable**: `sizes` list, `n_repeats`, `n_warmup`. The same numbers
underpin `docs/benchmarks.md`.

**Expected output**: forward-pass crossover at `npix ≈ 256` on
M-series; for small Sérsic fits CPU wins ~5×; for U-Net training MPS
wins ~1.5–4×.

## 14 — `14_power_law_and_NIE_lenses.ipynb`

Theoretical deep-dive into power-law and NIE lenses.

**Steps**:
1. Plot κ(r) and α(r) for `n ∈ {1.5, 2.0, 2.5}` power laws.
2. NIE caustic-topology regimes: increasing core radius
   `xc ∈ {0.02, 0.10, 0.30, 0.6}` produces image-plane critical
   curves and source-plane caustics that become disconnected, then
   disappear.

**Tunable**: power-law slopes `slopes`; NIE `q`, list `cores`.

**Expected output**: clear visual demonstration of the
"radial-caustic-disappears" transition described in
Kormann+ 1994 / Meneghetti Ch. 5.4.2.

**Dependencies**: requires `scikit-image` (for `find_contours`) — pip
installs it automatically.

## 15 — `15_time_delay_cosmography.ipynb`

Refsdal-style H₀ inversion from time delays.

**Steps**:
1. Plot the Fermat potential `τ(θ; β)` of an SIS — image positions are
   the saddle points.
2. Pick a 4-image SIE; solve image positions; integrate the lensing
   potential along the rays; compute `Δτ` between images.
3. Convert `Δτ` → `Δt` (days) at H₀ = 70 km/s/Mpc.
4. **Invert**: pretend Δt is measured with 0%, 2%, 5% noise; recover
   H₀ via `gl.lens.timedelay.refsdal_H0` and compare to the truth.

**Tunable**: source position; SIE parameters; noise levels in the
inversion loop.

**Expected output**: H₀ recovered exactly at zero noise; ±5% spread
at 5% Δt noise — illustrating how Δt-precision propagates to H₀.

## 16 — `16_large_scale_lens_finder.ipynb`

Survey-scale CNN classifier on a 5,000-sample HDF5 dataset.

**Steps**:
1. Generate the HDF5 dataset (~3 min on CPU, 83 MB on disk).
   Re-runs reuse the cached file at `notebooks/cache/lens_dataset_5k.h5`.
2. 70/15/15 train/val/test split.
3. Train `LensCNN` with 5 epochs, plot loss/accuracy curves.
4. Confusion matrix + ROC + AUC.
5. Visual inspection of the misclassified subset.

**Tunable**: `N_TOTAL` (default 5000), `npix` (default 48); set
`REGENERATE = True` to force a fresh dataset; `epochs`.

**Expected output**: ≳ 90% test accuracy; AUC ≳ 0.95; the FP/FN panel
is the most informative — typical FP are tidal-tail galaxies, typical
FN are very faint arcs.

## 17 — `17_param_recovery_at_scale.ipynb`

Sérsic-parameter regression on the same HDF5 dataset.

**Prerequisite**: notebook 16 must have generated
`notebooks/cache/lens_dataset_5k.h5` (the assert at cell 2 will
remind you).

**Steps**:
1. Same train/val/test split as notebook 16.
2. Train `SersicRegressor` for 8 epochs.
3. Reliability diagrams per parameter, with robust σ_residual reported.
4. Wall-time comparison vs. classical Adam fit (50 images).

**Tunable**: epochs, batch size, `n_compare` for the timing panel.

**Expected output**: σ_residual on `Re ≲ 0.05 arcsec`, `n ≲ 0.3`; DNN
inference 50–100× faster than per-image Adam.

## 18 — `18_LLM_metadata_extraction.ipynb`

Extract structured lens metadata from paper-style abstracts.

**Steps**:
1. Five demonstration abstracts in `corpus`.
2. Build a `gl.llm.MetadataExtractor(backend='mock')` (regex,
   deterministic, offline).
3. Extract → DataFrame.
4. Cross-validate against `gl.archive.slacs_table()` for the SLACS
   systems present in the corpus.

**Switch to real Claude API**: change `BACKEND = 'anthropic'`. Set
`ANTHROPIC_API_KEY` in the environment first. The extractor uses the
official `anthropic` SDK with **prompt caching** on the system prompt
so a 100-paper batch costs ≪ $1 with Haiku.

**Tunable**: model name (`claude-haiku-4-5` ↔ `claude-sonnet-4-6`);
the system prompt in `lensing/llm/extractor.py:_SYSTEM` controls the
output schema.

**Expected output**: DataFrame with one row per abstract; for SLACS
systems the `theta_E` and `sigma_v` should match the embedded catalog
within their published uncertainties.

---

# Script reference

The `scripts/` directory contains four entry points.

## `scripts/run_microlensing_fit.py`

Simulate a microlensing event and fit it from the command line.

```bash
python scripts/run_microlensing_fit.py \
    --f 7 --y0 0.1 --t0 183 --tE 20 --sigma 0.5 \
    --epochs 4000 --seed 0 --out fit_results.csv
```

**Args**:
| Flag           | Default | Meaning                                    |
|----------------|---------|--------------------------------------------|
| `--f`          | 7.0     | Source flux                                |
| `--y0`         | 0.1     | Impact parameter                           |
| `--t0`         | 183     | Peak time (days)                           |
| `--tE`         | 20      | Einstein crossing time (days)              |
| `--sigma`      | 0.5     | Per-point Gaussian noise                   |
| `--ndays`      | 365     | Light-curve duration                       |
| `--epochs`     | 4000    | Adam epochs                                |
| `--lr`         | 0.1     | Adam learning rate                         |
| `--seed`       | 0       | RNG seed                                   |
| `--out`        | None    | If given, write a one-row CSV with fit + best loss |

**Stdout**: progress every `epochs/10`, then a summary table of
`(fit, truth)` for every parameter.

## `scripts/run_sersic_fit.py`

End-to-end Sérsic + PSF fit from the command line.

```bash
python scripts/run_sersic_fit.py --npix 128 --deltapix 0.05 --epochs 3000
```

**Args**:
| Flag           | Default | Meaning                                    |
|----------------|---------|--------------------------------------------|
| `--npix`       | 128     | Grid side                                  |
| `--deltapix`   | 0.05    | Pixel scale (arcsec)                       |
| `--psf-fwhm`   | 0.10    | PSF FWHM (arcsec)                          |
| `--sigma`      | 0.05    | Pixel noise std                            |
| `--epochs`     | 3000    | Adam epochs                                |
| `--lr`         | 0.05    | Adam learning rate                         |
| `--seed`       | 0       | RNG seed                                   |

**Stdout**: progress every `epochs/10`, then `(fit, truth)` summary.

## `scripts/_make_notebooks.py`

Regenerate every `.ipynb` in `notebooks/` from the cell definitions in
this script. Use after edits to the cell bodies; never hand-edit the
generated `.ipynb` JSON.

```bash
python scripts/_make_notebooks.py
```

No arguments. The script writes 18 notebooks to `notebooks/`.

---

# How to extend

* **Add a new lens model**: create `lensing/lens/<name>.py` with a
  `nn.Module` exposing `deflection(x, y) -> (ax, ay)` and optionally
  `kappa`, `enforce_constraints`. Import it in
  `lensing/lens/__init__.py` and add a smoke test in
  `tests/test_advanced_lenses.py`.
* **Add a notebook**: write a `notebook_xxx()` factory in
  `scripts/_make_notebooks.py` returning a `[md(...), code(...), ...]`
  list, then register it in `main()`.
* **Add a benchmark**: subclass the workload pattern in
  `lensing/benchmarks.py` (a factory `factory(device) -> callable`).
* **Add a real-data archive**: extend `lensing/archive.py` with an
  HSC- or MAST-like downloader following the `fetch_hsc_cutout`
  pattern (URL → local cache → return numpy array).

---

# Common pitfalls

| Symptom                                        | Cause and fix                                    |
|------------------------------------------------|--------------------------------------------------|
| `NaN` parameters during Sérsic fit             | Likely too-aggressive `lr`. Lower to 0.01 and add `grad_clip=10.0`. The package already calls `enforce_constraints()` after each step, so positivity is preserved. |
| `LogNorm` ValueError in plots                  | Image contains zeros or negatives. Use `gl.viz.imshow_log` (falls back to linear automatically). |
| Notebook 06 caustic plot crashes               | A pre-existing tensor still has `requires_grad=True`. The package wraps SIE plotting helpers with `@torch.no_grad()` since v0.2; pull the latest. |
| `RUN_NUTS=True` runs forever                   | NUTS scales with image size; reduce `npix` or `num_samples`. The cached CSV in `notebooks/cache/` was produced at `npix=128, num_samples=2000`. |
| Notebook 16 `assert DATA_PATH.exists()` fails  | Run notebook 16 first to populate the cache (or set `REGENERATE = True` there). |
| `RuntimeError: failed to fetch ...`            | HSC public-archive endpoint is unreachable. Use the cached SLACS-lite catalog (`gl.archive.slacs_table()`) instead. |
| `ANTHROPIC_API_KEY not set`                    | Notebook 18 falls back to `BACKEND='mock'` — set the variable only if you want live Claude calls. |
