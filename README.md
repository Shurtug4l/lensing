# lensing

A clean, reproducible Python package built around the code developed for the
MSc thesis **["Applications of Automatic Differentiation in Gravitational
Lensing"](../Applications-of-Automatic-Differentiation-in-Gravitational-Lensing-main/thesis.pdf)**
(Simone La Porta, *Astrophysics & Cosmology*, Università di Bologna, 2024).

The package consolidates the original notebooks (formerly `bad_lensing/`)
into a tested, importable library and twelve narrative notebooks that
reproduce each thesis application end-to-end and extend the work to
**galaxy clusters (NFW), binary microlenses, and deep-learning pipelines
(CNN classifier, parametric regressor, U-Net source reconstruction)**.

## What's inside

```
lensing/
├── lensing/
│   ├── config.py           setup unico (device, RNG, plot style)
│   ├── cosmology.py        FlatLambdaCDM wrapper -> torch tensors
│   ├── light/              brillanze parametriche (Sersic, CoreSersic, MultiLight, PSF)
│   ├── lens/               PointMass, SIE, NIE, NFW, PowerLaw/SIS,
│   │                       BinaryPointMass, ExternalShear, CompositeLens,
│   │                       timedelay (Fermat potential / Refsdal)
│   ├── data/               coordinate grids, noise, simulator unificati
│   ├── inference/          Adam+L-BFGS, NUTS, weak-lensing ellipticities
│   ├── ml/                 datasets sintetici, CNN/DNN/U-Net, training loop
│   ├── bigdata/            HDF5-backed dataset generator + lazy Dataset
│   ├── archive.py          downloader HSC public + SLACS-lite catalog
│   ├── llm/                lens-metadata extractor (Anthropic + mock)
│   ├── benchmarks.py       Stopwatch + compare_devices (CPU/MPS/CUDA)
│   ├── stats.py            chi^2/AIC/BIC, classification + calibration,
│   │                       PSNR/SSIM, MCMC R-hat, k-fold CV, bootstrap
│   ├── viz/                imshow log-stretch + diagnostics multi-panel
│   └── utils/              transforms parametri (e1/e2 ↔ q,PA), bn(n)
├── notebooks/              dodici case-study eseguibili (vedi sotto)
├── scripts/                CLI (run_microlensing, run_sersic) + nbgen
├── tests/                  pytest smoke-tests (12 verdi in ~3.5s)
├── data/raw/               FITS reali (TEST_F150W_NIRCAM, MACS J1206, kappa_gl)
├── docs/background.md      teoria condensata dalla tesi e dai testi di riferimento
└── pyproject.toml, requirements.txt, environment.yml
```

## Notebook guide

| #   | Notebook                                  | Topic                                                     | Tools         |
|-----|-------------------------------------------|-----------------------------------------------------------|---------------|
| 01  | `01_microlensing_lightcurve.ipynb`        | Paczynski curve fit + NUTS posterior                      | Adam, NUTS    |
| 02  | `02_sersic_fit.ipynb`                     | Sérsic galaxy fit (Adam + L-BFGS) + NUTS posterior        | Adam, NUTS    |
| 03  | `03_core_sersic_fit.ipynb`                | Core-Sérsic profile (Graham+ 2003) for core ellipticals   | Adam, log-MSE |
| 04  | `04_double_sersic_fit.ipynb`              | Bulge + disk decomposition (14 free parameters)           | Adam          |
| 05  | `05_weak_lensing_ellipticity.ipynb`       | Parametric vs quadrupole-moment shape measurement         | Adam          |
| 06  | `06_strong_lensing_SIE.ipynb`             | SIE caustics, magnification map, image solver, inversion  | Adam          |
| 07  | `07_real_galaxy_F150W.ipynb`              | Real JWST/NIRCam F150W galaxy fit                         | Adam          |
| 08  | `08_binary_microlens.ipynb`               | Binary microlensing: caustics + light-curve crossings     | numerical     |
| 09  | `09_galaxy_cluster_NFW.ipynb`             | Cluster mass model (NFW + members + shear); MACS J1206    | numerical     |
| 10  | `10_CNN_lens_classifier.ipynb`            | CNN classifier — lens vs no-lens                          | PyTorch CNN   |
| 11  | `11_DNN_param_regression.ipynb`           | One-shot DNN regressor for Sérsic parameters              | PyTorch CNN   |
| 12  | `12_unet_source_reconstruction.ipynb`     | U-Net for source-plane reconstruction                     | PyTorch U-Net |
| 13  | `13_cpu_vs_mps_benchmarks.ipynb`          | CPU vs MPS wall-time scaling (forward / backward / fit)   | benchmark     |
| 14  | `14_power_law_and_NIE_lenses.ipynb`       | PowerLaw / SIS profiles + NIE caustic-topology regimes    | numerical     |
| 15  | `15_time_delay_cosmography.ipynb`         | Fermat potential, Δt, Refsdal H₀ inversion                | analytic      |
| 16  | `16_large_scale_lens_finder.ipynb`        | Survey-scale CNN trained on 5k HDF5-streamed lenses       | PyTorch CNN   |
| 17  | `17_param_recovery_at_scale.ipynb`        | DNN regressor: per-param accuracy + Adam speed comparison | PyTorch DNN   |
| 18  | `18_LLM_metadata_extraction.ipynb`        | Claude / regex backend mining lens metadata from papers   | Anthropic API |

The first seven reproduce the thesis chapters; the rest extend the work to
cluster-scale and binary lensing, three deep-learning pipelines
(CNN, regressor, U-Net) and several theoretical deep-dives inspired by the
*Lensing Gravitazionale* graduate course (Prof. M. Meneghetti, UNIBO MSc
in Astrophysics & Cosmology) — power-law / NIE lenses, time-delay
cosmography, and a CPU-vs-MPS performance study.

## Installation

```bash
pip install -r requirements.txt
# OR
conda env create -f environment.yml && conda activate lensing
```

Notebooks bootstrap themselves (prepend the repo to `sys.path`); you can
launch `jupyter lab notebooks/` directly without installing the package.

## Quickstart

```python
import torch
import lensing as gl

device, dtype = gl.config.setup(seed=42)

# 1. simulate a noisy galaxy
xy = gl.data.coordinate_grid(npix=128, deltapix=0.05)
truth  = gl.light.Sersic(Ie=5., Re=1., n=4., x0=0., y0=0., e1=0.2, e2=-0.1)
clean, image = gl.data.simulate_image(
    truth, xy, psf_fwhm=0.10, deltapix=0.05, noise_sigma=0.05, seed=0,
)

# 2. fit it back with Adam + L-BFGS
import torch.nn as nn

class SersicPSF(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = gl.light.Sersic(Ie=2., Re=1.5, n=2.5, x0=0., y0=0., e1=0., e2=0.)
    def forward(self, xy):
        k = gl.light.gaussian_psf_kernel(0.10, 0.05, size=21)
        return gl.light.convolve_psf(self.g(xy), k)

result = gl.inference.fit(
    SersicPSF(), xy, image,
    gl.inference.ReducedChiSquared(sigma=0.05, n_params=7),
    lr=0.05, epochs=2000, lbfgs_polish=True,
    scheduler=gl.inference.optimize.reduce_lr_on_plateau(),
)
print(result.parameters, result.best_loss)
```

For the deep-learning notebooks:

```python
from torch.utils.data import DataLoader

train = gl.ml.datasets.LensClassifierDataset(n_samples=400, npix=48)
val   = gl.ml.datasets.LensClassifierDataset(n_samples=100, npix=48, seed=1000)

model = gl.ml.models.LensCNN()
hist  = gl.ml.train.fit_model(
    model,
    DataLoader(train, batch_size=32, shuffle=True),
    DataLoader(val, batch_size=32),
    loss_fn=torch.nn.CrossEntropyLoss(),
    metrics={'acc': gl.ml.train.accuracy},
    epochs=8,
)
```

## Background documentation

`docs/background.md` distills the gravitational-lensing theory used in the
package, with a column mapping each piece of theory to the implementing
module. The exposition follows closely the *Introduction to Gravitational
Lensing* lecture notes by **M. Meneghetti** (UNIBO graduate course
*Lensing Gravitazionale*, MSc in Astrophysics & Cosmology) which were the
primary reference for the thesis. Other reference texts: Bartelmann &
Schneider (2001) *Phys. Rep.*; Schneider, Kochanek & Wambsganss (2006)
*Saas-Fee 33*; Congdon & Keeton (2018) *Principles of Gravitational
Lensing*; Wright & Brainerd (2000) *ApJ* (NFW); Kormann, Schneider &
Bartelmann (1994) *A&A* (SIE/NIE).

`docs/astrophysics.md` covers the **astrophysical applications**: H0
cosmography from time delays, dark-matter substructure detection,
exoplanet hunting via microlensing, primordial-black-hole constraints,
weak-lensing cosmic-shear surveys, and clusters as cosmic telescopes.

`docs/benchmarks.md` reports the CPU vs MPS performance numbers from
notebook 13.

`docs/validation.md` is the **validation cookbook** — for each kind of
fit (parametric, classification, regression, image-regression,
literature-mining) it lists the appropriate metrics, the helper that
computes them, and what a "good" value looks like.

`docs/usage.md` is the **detailed how-to-run guide** — for every
notebook (01–18) and every script it lists the steps, the tunable
parameters, the expected outputs, and the most common pitfalls.

## Notable improvements over the original code

1. **Smooth ellipse geometry** in (e1, e2). The thesis form
   `q = (1-|e|)/(1+|e|)`, `pa = ½ atan2(e2, e1)` produces NaN gradients at
   `(e1, e2) = (0, 0)`. We expand the rotation algebraically and absorb
   `(1-|e|)²` into `Re`, yielding a polynomial radius
   `R'^2 = (1+e²-2e1) dx² + (1+e²+2e1) dy² - 4 e2 dx dy`
   that is differentiable everywhere. See `light/base.py`.

2. **Parameter projection** (`enforce_constraints()`) called after every
   optimizer step keeps `Re > 0`, `n ∈ [0.3, 12]`, `q ∈ (0, 1)` etc. without
   needing softplus reparameterization.

3. **L-BFGS polish** after Adam for 1–2 orders of magnitude tighter loss
   without tuning (`fit(..., lbfgs_polish=True)`).

4. **Robust visualization**: `imshow_log` falls back to a linear scale when
   the dynamic range is too small for `LogNorm`. SIE caustics/critical
   curves are wrapped with `@torch.no_grad()` so they plot cleanly.

5. **Deep-learning module** (`lensing.ml`) with synthetic dataset
   generators that stream samples on the fly so memory stays flat.

## Tests

```bash
pytest tests/
```

12 smoke tests (~3.5 s) covering the light models, microlensing, and SIE.

## License

MIT — drop in your preferred LICENSE file.
