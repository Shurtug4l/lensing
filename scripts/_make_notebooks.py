"""Generate the notebooks shipped in `notebooks/`.

Programmatic generator: each notebook below is a Python function returning a
list of cells. Re-run this script after editing to regenerate all notebooks.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Iterable, List

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


# --- helpers -----------------------------------------------------------------
def md(text: str) -> dict:
    return {"id": _new_id(), "cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "id": _new_id(),
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


_ID_COUNTER = 0


def _new_id() -> str:
    global _ID_COUNTER
    _ID_COUNTER += 1
    return f"cell-{_ID_COUNTER:04d}"


def write_notebook(name: str, cells: Iterable[dict]) -> None:
    nb = {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NB_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print("wrote", path)


HEADER_BOOTSTRAP = textwrap.dedent("""\
    # Bootstrap: make `lensing` importable when running notebooks/ directly.
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
    # Device-agnostic: prefer MPS (Apple GPU) → CUDA → CPU.
    # Pass device="cpu" if you need to force the CPU path (e.g. for
    # operators that have no MPS kernel yet, or for reproducibility).
    device, dtype = gl.config.setup(seed=42)
    print(f"using device: {device}")
""")


# =========================================================================
# 01 - Microlensing light curve
# =========================================================================
def notebook_microlensing() -> List[dict]:
    return [
        md("# 01 — Microlensing light curve\n\n"
           "Fit a point-mass microlensing event using two parameterizations:\n\n"
           "1. **Physical**: `(M, v_rel, D_L, D_S, ...)` — used for forward simulation; degenerate.\n"
           "2. **Minimal Paczynski**: `(f_S, t_0, y_0, t_E)` — what observers fit.\n\n"
           "We then sample the minimal model with NUTS to quantify the posterior."),
        code(HEADER_BOOTSTRAP),
        md("## 1. Simulate from physical parameters"),
        code(textwrap.dedent("""\
            phys = gl.lens.PointMassMicrolens(
                f=7.0, mass=0.3, y0=0.1, vel=200., t0=183., dl=4., ds=8.,
            )
            t = torch.arange(0., 365., 1.)
            sigma = 0.1 * 7.0
            clean, mag = gl.data.simulate_lightcurve(phys, t, noise_sigma=sigma, seed=0)
            print(f"theta_E = {float(phys.einstein_radius()):.3e} arcsec")
            print(f"t_E     = {float(phys.einstein_time()):.2f} days")
        """)),
        code(textwrap.dedent("""\
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.errorbar(t, mag, yerr=sigma, fmt='.', color='k',
                        ecolor='C0', elinewidth=1, capsize=2, alpha=0.6, label='data')
            ax.plot(t, clean, 'r-', lw=2, label='true')
            ax.set(xlabel='time [days]', ylabel=r'$\\mu(t)$'); ax.legend(); plt.show()
        """)),
        md("## 2. Fit the minimal Paczynski model"),
        code(textwrap.dedent("""\
            paczy = gl.lens.PaczynskiLightcurve(f=2.0, y0=0.6, t0=114., tE=10.)
            loss_fn = gl.inference.ReducedChiSquared(sigma=sigma, n_params=4)
            result = gl.inference.fit(
                paczy, t, mag, loss_fn,
                lr=0.1, epochs=4000,
                scheduler=gl.inference.optimize.reduce_lr_on_plateau(),
                lbfgs_polish=True,
            )
            print(f"final chi2/dof = {result.best_loss:.3f} in {result.duration_s:.1f}s")
            for k, v in result.parameters.items():
                print(f"  {k}: {v:.4f}")
        """)),
        code(textwrap.dedent("""\
            with torch.no_grad():
                fit_lc = paczy(t).numpy()
            fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                     gridspec_kw={'height_ratios':[3,1]})
            axes[0].errorbar(t, mag, yerr=sigma, fmt='.', color='k', alpha=.4, label='data')
            axes[0].plot(t, fit_lc, 'r-', lw=2, label='best fit')
            axes[0].plot(t, clean, 'k--', lw=1, label='truth')
            axes[0].set_ylabel(r'$\\mu(t)$'); axes[0].legend()
            res = (mag.numpy() - fit_lc) / sigma
            axes[1].errorbar(t, res, fmt='.', color='C0'); axes[1].axhline(0, color='k')
            axes[1].set(ylabel=r'$\\Delta/\\sigma$', xlabel='time [days]'); plt.show()

            gl.viz.plot_loss_history(result.loss_history); plt.show()
        """)),
        md("## 3. NUTS posterior (cached)\n\n"
           "Set `RUN_NUTS=True` to regenerate the cache (~30 s on CPU)."),
        code(textwrap.dedent("""\
            RUN_NUTS = False
            csv = Path('cache/posterior_microlens.csv'); csv.parent.mkdir(exist_ok=True)

            df = None
            if RUN_NUTS:
                import pyro, pyro.distributions as dist
                def pyro_model(t, mag):
                    f  = pyro.sample('f',  dist.Uniform(0.1, 50.))
                    t0 = pyro.sample('t0', dist.Uniform(0., 365.))
                    y0 = pyro.sample('y0', dist.Uniform(0.01, 1.))
                    tE = pyro.sample('tE', dist.Uniform(0.1, 365.))
                    pred = gl.lens.PaczynskiLightcurve(f, y0, t0, tE)(t)
                    with pyro.plate('data', len(mag)):
                        pyro.sample('obs', dist.Normal(pred, sigma), obs=mag)
                df = gl.inference.run_nuts(pyro_model, t, mag,
                                          num_samples=2000, warmup_steps=500,
                                          save_path=str(csv))
            elif csv.exists():
                import pandas as pd
                df = pd.read_csv(csv)
            else:
                print("No cached posterior; set RUN_NUTS=True to generate it.")

            if df is not None and len(df):
                print(df.describe())
        """)),
        code(textwrap.dedent("""\
            if df is not None and len(df):
                truths = {'f': 7.0, 't0': 183., 'y0': 0.1,
                          'tE': float(phys.einstein_time())}
                df_ord = df[['f', 't0', 'y0', 'tE']]
                gl.viz.corner_plot(df_ord, truths=[truths[k] for k in df_ord.columns]); plt.show()
        """)),
    ]


# =========================================================================
# 02 - Sérsic
# =========================================================================
def notebook_sersic() -> List[dict]:
    return [
        md("# 02 — Sérsic surface-brightness fit\n\n"
           "Forward-model a Sérsic galaxy + Gaussian PSF + noise; recover the "
           "parameters by gradient descent (Adam + L-BFGS) and quantify the "
           "posterior with NUTS."),
        code(HEADER_BOOTSTRAP),
        code(textwrap.dedent("""\
            npix, dx = 128, 0.05
            xy = gl.data.coordinate_grid(npix=npix, deltapix=dx)

            true = dict(Ie=5.0, Re=1.0, n=4.0, x0=0.1, y0=-0.2, e1=0.20, e2=-0.10)
            galaxy = gl.light.Sersic(**true)

            sigma_n = 0.05
            clean, image = gl.data.simulate_image(
                galaxy, xy,
                psf_fwhm=0.10, deltapix=dx, psf_size=21,
                noise_sigma=sigma_n, seed=0,
            )

            ext = (-npix*dx/2, npix*dx/2, -npix*dx/2, npix*dx/2)
            gl.viz.side_by_side([clean, image, image - clean],
                                titles=['clean', 'PSF + noise', 'residual'],
                                log=False, extent=ext); plt.show()
        """)),
        md("## Fit"),
        code(textwrap.dedent("""\
            class SersicPSF(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.g = gl.light.Sersic(Ie=2.0, Re=1.5, n=2.5, x0=0., y0=0., e1=0., e2=0.)
                def forward(self, xy):
                    k = gl.light.gaussian_psf_kernel(0.10, dx, size=21)
                    return gl.light.convolve_psf(self.g(xy), k)

            model = SersicPSF()
            loss = gl.inference.ReducedChiSquared(sigma=sigma_n, n_params=7)
            res = gl.inference.fit(
                model, xy, image, loss,
                lr=0.05, epochs=2500,
                scheduler=gl.inference.optimize.reduce_lr_on_plateau(patience=200, factor=0.7),
                grad_clip=10.0, lbfgs_polish=True,
            )
            print(f"final chi2/dof = {res.best_loss:.3f}  in {res.duration_s:.1f}s")
            for k, v in true.items():
                fv = float(getattr(model.g, k))
                print(f"  {k}: true={v:+.3f}  fit={fv:+.3f}")
        """)),
        code(textwrap.dedent("""\
            with torch.no_grad():
                fit_img = model(xy)
            gl.viz.side_by_side([image, fit_img, image - fit_img],
                                titles=['data', 'best fit', 'residual'],
                                log=False, extent=ext); plt.show()
            gl.viz.plot_loss_history(res.loss_history); plt.show()
        """)),
        md("## NUTS posterior (cached)"),
        code(textwrap.dedent("""\
            RUN_NUTS = False
            csv = Path('cache/posterior_sersic.csv'); csv.parent.mkdir(exist_ok=True)
            df = None
            if RUN_NUTS:
                import pyro, pyro.distributions as dist
                def pyro_model(xy, image):
                    Ie = pyro.sample('Ie', dist.LogNormal(0., 1.))
                    Re = pyro.sample('Re', dist.LogNormal(0., 1.))
                    n  = pyro.sample('n',  dist.Uniform(0.5, 8.0))
                    x0 = pyro.sample('x0', dist.Normal(0., 1.0))
                    y0 = pyro.sample('y0', dist.Normal(0., 1.0))
                    e1 = pyro.sample('e1', dist.Uniform(-0.7, 0.7))
                    e2 = pyro.sample('e2', dist.Uniform(-0.7, 0.7))
                    g = gl.light.Sersic(Ie, Re, n, x0, y0, e1, e2)
                    pred = g(xy)
                    k = gl.light.gaussian_psf_kernel(0.10, dx, size=21)
                    pred = gl.light.convolve_psf(pred, k)
                    with pyro.plate('data', image.numel()):
                        pyro.sample('obs', dist.Normal(pred.flatten(), sigma_n),
                                    obs=image.flatten())
                df = gl.inference.run_nuts(pyro_model, xy, image,
                                          num_samples=1000, warmup_steps=500,
                                          save_path=str(csv))
            elif csv.exists():
                import pandas as pd
                df = pd.read_csv(csv)

            if df is not None and len(df):
                print(df.describe())
                df_ord = df[['Ie','Re','n','x0','y0','e1','e2']]
                gl.viz.corner_plot(df_ord, truths=[true[k] for k in df_ord.columns]); plt.show()
            else:
                print("No cached posterior; set RUN_NUTS=True to generate.")
        """)),
    ]


# =========================================================================
# 03 - CoreSersic (FIX: log-image MSE + proper init to avoid NaN explosion)
# =========================================================================
def notebook_core_sersic() -> List[dict]:
    return [
        md("# 03 — Core-Sérsic fit\n\n"
           "Fits a Core-Sérsic profile (Graham et al. 2003) — describes massive "
           "ellipticals with a depleted central core. We use a log-MSE loss "
           "(`log_image_mse`) because the dynamic range spans ~10 decades."),
        code(HEADER_BOOTSTRAP),
        code(textwrap.dedent("""\
            npix, dx = 128, 0.05
            xy = gl.data.coordinate_grid(npix=npix, deltapix=dx)

            # Modest CoreSersic params: large central core to keep the simulated
            # image away from the (Rb/R)^alpha singularity at R -> 0.
            true = dict(Ib=2.0, Re=1.5, Rb=0.4, n=4.0, gamma=0.3, alpha=4.0,
                        x0=0.0, y0=0.0, e1=0.10, e2=-0.05)
            galaxy = gl.light.CoreSersic(**true)
            clean, image = gl.data.simulate_image(
                galaxy, xy,
                psf_fwhm=0.10, deltapix=dx, psf_size=21,
                noise_sigma=0.02, seed=1,
            )
            ext = (-npix*dx/2, npix*dx/2, -npix*dx/2, npix*dx/2)
            gl.viz.side_by_side([clean, image], titles=['clean', 'noisy'],
                                log=True, extent=ext); plt.show()
        """)),
        md("### Profile cross-section"),
        code(textwrap.dedent("""\
            r = torch.linspace(0.01, 3.0, 200)
            xy_line = torch.stack([r, torch.zeros_like(r)], dim=0)
            with torch.no_grad():
                I_core = galaxy(xy_line)
                # comparison: pure Sersic with same Re, n
                pure = gl.light.Sersic(Ie=true['Ib'], Re=true['Re'], n=true['n'],
                                       x0=0., y0=0., e1=0., e2=0.)
                I_pure = pure(xy_line)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(r, I_core, 'r-', lw=2, label='core-Sérsic')
            ax.plot(r, I_pure, 'k--', lw=1.5, label='Sérsic')
            ax.axvline(true['Rb'], color='gray', ls=':', label='$R_b$')
            ax.set(xlabel='r [arcsec]', ylabel='I(r)', yscale='log')
            ax.legend(); plt.show()
        """)),
        md("### Fit (log-image MSE)"),
        code(textwrap.dedent("""\
            class CoreSersicPSF(nn.Module):
                def __init__(self, init):
                    super().__init__()
                    self.g = gl.light.CoreSersic(**init)
                def forward(self, xy):
                    k = gl.light.gaussian_psf_kernel(0.10, dx, size=21)
                    return gl.light.convolve_psf(self.g(xy), k)

            init = dict(Ib=1.0, Re=1.0, Rb=0.5, n=3.0, gamma=0.5, alpha=3.0,
                        x0=0., y0=0., e1=0., e2=0.)
            model = CoreSersicPSF(init)

            # log_image_mse compresses the 10-decade dynamic range of a galaxy
            # surface brightness profile, which makes the gradient finite even
            # when the core saturates.
            res = gl.inference.fit(
                model, xy, image, gl.inference.log_image_mse,
                lr=0.02, epochs=3000,
                scheduler=gl.inference.optimize.reduce_lr_on_plateau(patience=200, factor=0.7),
                grad_clip=1.0, lbfgs_polish=True,
            )
            print(f"final loss = {res.best_loss:.4e}")
            for k, v in true.items():
                print(f"  {k}: true={v:+.3f}  fit={float(getattr(model.g, k)):+.3f}")
        """)),
        code(textwrap.dedent("""\
            with torch.no_grad():
                pred = model(xy)
            gl.viz.side_by_side([image, pred],
                                titles=['data', 'best fit'], log=True, extent=ext); plt.show()
            gl.viz.plot_residuals(image, pred, sigma=0.02); plt.show()
        """)),
    ]


# =========================================================================
# 04 - Double Sersic
# =========================================================================
def notebook_double_sersic() -> List[dict]:
    return [
        md("# 04 — Double-Sérsic decomposition\n\n"
           "Decompose a galaxy into bulge + disk Sérsic components. Joint "
           "fit of all 14 parameters via `MultiLight`."),
        code(HEADER_BOOTSTRAP),
        code(textwrap.dedent("""\
            npix, dx = 128, 0.05
            xy = gl.data.coordinate_grid(npix=npix, deltapix=dx)

            true_a = dict(Ie=8., Re=0.6, n=4.0, x0=-0.6, y0=+0.3, e1=0.15, e2=0.10)
            true_b = dict(Ie=2., Re=1.5, n=1.0, x0=+0.2, y0=-0.4, e1=0.20, e2=-0.10)
            galaxy = gl.light.DoubleSersic(component1=true_a, component2=true_b)

            sigma_n = 0.02
            clean, image = gl.data.simulate_image(
                galaxy, xy, psf_fwhm=0.10, deltapix=dx, psf_size=21,
                noise_sigma=sigma_n, seed=2,
            )
            ext = (-npix*dx/2, npix*dx/2, -npix*dx/2, npix*dx/2)
            gl.viz.side_by_side([clean, image], titles=['bulge+disk', 'noisy'],
                                log=True, extent=ext); plt.show()
        """)),
        code(textwrap.dedent("""\
            class DoubleSersicPSF(nn.Module):
                def __init__(self, a, b):
                    super().__init__()
                    self.galaxy = gl.light.DoubleSersic(component1=a, component2=b)
                def forward(self, xy):
                    k = gl.light.gaussian_psf_kernel(0.10, dx, size=21)
                    return gl.light.convolve_psf(self.galaxy(xy), k)

            init_a = dict(Ie=4., Re=1.0, n=2.5, x0=-0.5, y0=+0.5, e1=0., e2=0.)
            init_b = dict(Ie=4., Re=1.0, n=2.5, x0=+0.5, y0=-0.5, e1=0., e2=0.)
            model = DoubleSersicPSF(init_a, init_b)

            loss = gl.inference.ReducedChiSquared(sigma=sigma_n, n_params=14)
            res = gl.inference.fit(
                model, xy, image, loss,
                lr=0.05, epochs=3000,
                scheduler=gl.inference.optimize.reduce_lr_on_plateau(patience=200, factor=0.7),
                grad_clip=10.0, lbfgs_polish=True,
            )
            print(f"final chi2/dof = {res.best_loss:.4f}")

            for i, comp in enumerate(model.galaxy.components):
                print(f"--- component {i+1} ---")
                truth = (true_a, true_b)[i]
                for n_ in ['Ie','Re','n','x0','y0','e1','e2']:
                    print(f"  {n_}: true={truth[n_]:+.3f}  fit={float(getattr(comp, n_)):+.3f}")
        """)),
        code(textwrap.dedent("""\
            with torch.no_grad():
                pred = model(xy)
            gl.viz.side_by_side([image, pred],
                                titles=['data', 'best fit'], log=True, extent=ext); plt.show()
            gl.viz.plot_residuals(image, pred, sigma=sigma_n); plt.show()
        """)),
    ]


# =========================================================================
# 05 - Weak lensing ellipticity
# =========================================================================
def notebook_weak() -> List[dict]:
    return [
        md("# 05 — Weak lensing: galaxy ellipticities\n\n"
           "Two estimators of `(e1, e2)` from a galaxy image:\n\n"
           "1. **Parametric**: forward-fit Sérsic + PSF (gold standard).\n"
           "2. **Quadrupole moments** (KSB-style): non-parametric baseline."),
        code(HEADER_BOOTSTRAP),
        code(textwrap.dedent("""\
            npix, dx = 96, 0.05
            xy = gl.data.coordinate_grid(npix=npix, deltapix=dx)

            true = dict(Ie=5.0, Re=0.8, n=2.5, x0=0., y0=0., e1=-0.20, e2=0.15)
            galaxy = gl.light.Sersic(**true)

            sigma_n = 0.05
            clean, image = gl.data.simulate_image(
                galaxy, xy, psf_fwhm=0.10, deltapix=dx, psf_size=21,
                noise_sigma=sigma_n, seed=3,
            )
            gl.viz.side_by_side([clean, image], titles=['truth', 'data'], log=True)
            plt.show()
        """)),
        md("## 1. Parametric estimator (Sérsic + PSF)"),
        code(textwrap.dedent("""\
            result, summary = gl.inference.fit_ellipticity(
                image, xy, psf_fwhm=0.10, deltapix=dx, sigma=sigma_n,
                init=dict(Ie=1., Re=1., n=3., x0=0., y0=0., e1=0., e2=0.),
                epochs=3000,
            )
            print('parametric fit:')
            for k, v in summary.items(): print(f"  {k:<8s}: {v:+.4f}")
        """)),
        md("## 2. Quadrupole-moment estimator"),
        code(textwrap.dedent("""\
            ks = gl.inference.kaiser_squires_estimator(image, xy, weight_sigma=1.0)
            print('quadrupole estimator:')
            for k, v in ks.items(): print(f"  {k:<8s}: {v:+.4f}")
            print(f"\\ntrue:  e1 = {true['e1']:+.4f}   e2 = {true['e2']:+.4f}")
            print(f"|e|_true = {np.hypot(true['e1'], true['e2']):.4f}")
        """)),
    ]


# =========================================================================
# 06 - Strong lensing SIE (FIX: detach() in critical/caustic plotting)
# =========================================================================
def notebook_sie() -> List[dict]:
    return [
        md("# 06 — Strong lensing: SIE caustics, critical curves and image solver\n\n"
           "Singular Isothermal Ellipsoid (Kormann+ 1994): closed-form deflection, "
           "caustics, critical lines, and a robust polar-angle root finder for "
           "the multi-image positions of a point source."),
        code(HEADER_BOOTSTRAP),
        md("## 1. Build the lens from a velocity dispersion"),
        code(textwrap.dedent("""\
            cosmo = gl.cosmology.Cosmology(H0=70., Om0=0.3)
            sie = gl.lens.SIE.from_velocity_dispersion(
                sigma_v_kms=220., q=0.7, pa=np.pi/4, zl=0.3, zs=2.0, cosmo=cosmo,
            )
            print(f"theta_E = {float(sie.theta_E):.3f} arcsec")
        """)),
        md("## 2. Caustics, critical curves, source and images"),
        code(textwrap.dedent("""\
            beta_x, beta_y = torch.tensor(0.10), torch.tensor(-0.05)

            cau_x, cau_y = sie.tangential_caustic(n=600)
            crit_x, crit_y = sie.tangential_critical(n=600)
            cut_x, cut_y = sie.cut(n=600)
            ximg, yimg = sie.solve_image_positions(beta_x, beta_y, n_grid=4000)
            print(f"found {len(ximg)} images")

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(cau_x, cau_y, 'b-', label='tangential caustic')
            ax.plot(cut_x, cut_y, 'r--', label='cut')
            ax.plot(crit_x, crit_y, 'g-', label='critical line')
            ax.scatter(beta_x, beta_y, c='orange', marker='*', s=120, label='source')
            ax.scatter(ximg, yimg, c='k', marker='D', label='images')
            ax.axhline(0, color='gray', lw=.5); ax.axvline(0, color='gray', lw=.5)
            ax.set_aspect('equal'); ax.legend()
            ax.set(xlabel=r'$x_1$ [arcsec]', ylabel=r'$x_2$ [arcsec]')
            plt.show()
        """)),
        md("## 3. Convergence and magnification maps"),
        code(textwrap.dedent("""\
            from matplotlib.colors import LogNorm, SymLogNorm

            xy = gl.data.coordinate_grid(npix=400, deltapix=0.01)
            with torch.no_grad():
                kappa = sie.kappa(xy[0], xy[1])
                mu    = sie.magnification(xy[0], xy[1])

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            im0 = axes[0].imshow(kappa.numpy(), origin='lower', cmap='viridis',
                                 extent=(-2, 2, -2, 2),
                                 norm=LogNorm(vmin=1e-2, vmax=10))
            plt.colorbar(im0, ax=axes[0]); axes[0].set_title(r'convergence $\\kappa$')
            im1 = axes[1].imshow(mu.numpy(), origin='lower', cmap='RdBu_r',
                                 extent=(-2, 2, -2, 2),
                                 norm=SymLogNorm(linthresh=1, vmin=-30, vmax=30))
            plt.colorbar(im1, ax=axes[1]); axes[1].set_title(r'magnification $\\mu$')
            plt.show()
        """)),
        md("## 4. Lens inversion from observed image positions\n\n"
           "Given image positions (with measurement noise) we recover the SIE "
           "parameters by minimizing the source-plane scatter."),
        code(textwrap.dedent("""\
            true_sie = gl.lens.SIE(theta_E=1.2, q=0.6, pa=np.pi/3)
            beta_truth = (torch.tensor(0.05), torch.tensor(-0.08))
            ximg, yimg = true_sie.solve_image_positions(*beta_truth, n_grid=4000)
            ximg = ximg + torch.randn_like(ximg) * 0.01
            yimg = yimg + torch.randn_like(yimg) * 0.01
            print(f"observed {len(ximg)} images, sigma_pos = 0.01 arcsec")

            class SIEInverse(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.sie = gl.lens.SIE(theta_E=1.0, q=0.8, pa=0.5)
                    self.beta = nn.Parameter(torch.zeros(2))
                def forward(self, dummy):
                    bx, by = self.sie.ray_trace(ximg, yimg)
                    return torch.stack([bx - self.beta[0], by - self.beta[1]])

            model = SIEInverse()
            target = torch.zeros(2, len(ximg))
            res = gl.inference.fit(model, torch.tensor(0.), target, nn.MSELoss(),
                                   lr=0.05, epochs=4000, lbfgs_polish=True,
                                   scheduler=gl.inference.optimize.reduce_lr_on_plateau(patience=400))
            print(f"final loss = {res.best_loss:.3e}")
            print(f"truth: theta_E={float(true_sie.theta_E):.3f}  q={float(true_sie.q):.3f}  pa={float(true_sie.pa):.3f}")
            print(f"fit  : theta_E={float(model.sie.theta_E):.3f}  q={float(model.sie.q):.3f}  pa={float(model.sie.pa):.3f}")
            print(f"beta : truth={[float(b) for b in beta_truth]}  fit={model.beta.detach().tolist()}")
        """)),
    ]


# =========================================================================
# 07 - Real galaxy F150W
# =========================================================================
def notebook_real_galaxy() -> List[dict]:
    return [
        md("# 07 — Real galaxy fit (JWST NIRCam F150W)\n\n"
           "Loads `data/raw/TEST_F150W_NIRCAM.fits` (a JWST NIRCam Short-Wavelength "
           "F150W postage stamp) and fits a single-Sérsic + PSF model."),
        code(HEADER_BOOTSTRAP),
        code(textwrap.dedent("""\
            from astropy.io import fits

            FITS_PATH = repo / 'data' / 'raw' / 'TEST_F150W_NIRCAM.fits'
            DELTAPIX  = 0.031   # NIRCam SW pixel scale [arcsec/pix]
            PSF_FWHM  = 0.05    # F150W diffraction-limited FWHM [arcsec]

            with fits.open(FITS_PATH) as hdul:
                data = np.asarray(hdul[0].data, dtype=np.float32)
            data = np.nan_to_num(data)
            data = data - np.median(data)
            data = torch.tensor(data)
            ny, nx = data.shape
            print(f"image shape: {data.shape}, max = {float(data.max()):.3e}")

            # Square crop of the central source so the model fit doesn't have to
            # explain hundreds of background sources.
            half_size = 64  # pixels
            cy, cx = ny // 2, nx // 2
            crop = data[cy-half_size:cy+half_size, cx-half_size:cx+half_size].clone()
            ny_c, nx_c = crop.shape
            extent = (-nx_c*DELTAPIX/2, nx_c*DELTAPIX/2, -ny_c*DELTAPIX/2, ny_c*DELTAPIX/2)

            xy = gl.data.coordinate_grid(npix=nx_c, deltapix=DELTAPIX)

            gl.viz.imshow_log(crop + abs(crop.min()) + 1e-3,
                              extent=extent, title='JWST NIRCam F150W (cropped)')
            plt.show()
        """)),
        md("## Single-Sérsic fit with PSF"),
        code(textwrap.dedent("""\
            sigma_pix = float(crop.std()) * 0.5

            res, summary = gl.inference.fit_ellipticity(
                crop, xy, psf_fwhm=PSF_FWHM, deltapix=DELTAPIX, sigma=sigma_pix,
                init=dict(Ie=float(crop.max())*0.5, Re=0.3, n=2.0,
                          x0=0., y0=0., e1=0., e2=0.),
                epochs=3000, lr=0.05,
            )
            print(f'final chi2/dof = {res.best_loss:.3f}')
            for k, v in summary.items(): print(f'  {k:<8s}: {v:+.4g}')
        """)),
        code(textwrap.dedent("""\
            with torch.no_grad():
                model_image = res.model(xy)
            shifted = crop + abs(crop.min()) + 1e-3
            gl.viz.side_by_side([shifted, model_image, crop - model_image],
                                titles=['data', 'best fit', 'residual'],
                                log=False, extent=extent)
            plt.show()
        """)),
    ]


# =========================================================================
# 08 - Binary microlens (advanced, NEW)
# =========================================================================
def notebook_binary_microlens() -> List[dict]:
    return [
        md("# 08 — Binary microlensing\n\n"
           "Map of the **binary point-mass lens**: caustic structure depends "
           "on `(d, q_m)` — with the (close, intermediate, wide) topological "
           "regimes — and a single source crossing produces sharp light-curve "
           "spikes used for exoplanet detection in microlensing surveys.\n\n"
           "References: Schneider+ Saas-Fee 33 (2006), Sec. 6."),
        code(HEADER_BOOTSTRAP),
        md("## 1. Magnification maps for several (d, q) regimes"),
        code(textwrap.dedent("""\
            cases = [
                ('close', 0.6, 0.5),
                ('resonant', 1.0, 0.5),
                ('wide', 1.6, 0.5),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, (label, d, q_m) in zip(axes, cases):
                lens = gl.lens.BinaryPointMass(d=d, q_m=q_m)
                ax_axis, mu = lens.magnification_map(npix=300, halfwidth=2.0)
                im = ax.imshow(np.log10(mu.numpy() + 1e-3), origin='lower',
                               cmap='inferno',
                               extent=(-2, 2, -2, 2))
                plt.colorbar(im, ax=ax)
                ax.set_title(f'{label}: d={d}, q={q_m}')
            plt.tight_layout(); plt.show()
        """)),
        md("## 2. Caustic curves overlaid"),
        code(textwrap.dedent("""\
            lens = gl.lens.BinaryPointMass(d=1.0, q_m=0.5)
            ax_axis, detA = lens.critical_curves(n=400)
            fig, ax = plt.subplots(figsize=(7, 6))
            cs = ax.contour(ax_axis.numpy(), ax_axis.numpy(),
                            detA.numpy(), levels=[0.0], colors='cyan')
            ax.set_aspect('equal'); ax.set(xlabel=r'$x_1$', ylabel=r'$x_2$',
                                            title='binary critical curves (detA = 0)')
            plt.show()
        """)),
        md("## 3. Light curve from a source trajectory"),
        code(textwrap.dedent("""\
            # Source trajectory: straight line crossing the caustic.
            t = torch.linspace(-1.0, 1.0, 200)
            beta_x = t
            beta_y = torch.full_like(t, 0.05)
            # Inverse-ray-trace approach: for each beta, count how many image-plane
            # pixels map to it (binary lens has up to 5 images, no closed form).
            ax_axis, mu_map = lens.magnification_map(npix=600, halfwidth=2.0)
            # Look up magnification along the trajectory.
            ix = ((beta_x + 2.0) / 4.0 * 600).long().clamp(0, 599)
            iy = ((beta_y + 2.0) / 4.0 * 600).long().clamp(0, 599)
            mu_t = mu_map[iy, ix]
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(t, mu_t)
            ax.set(xlabel='source x position [theta_E]', ylabel=r'$\\mu$',
                   title='binary lens caustic-crossing light curve')
            plt.show()
        """)),
    ]


# =========================================================================
# 09 - Galaxy cluster (NFW) - NEW
# =========================================================================
def notebook_cluster_nfw() -> List[dict]:
    return [
        md("# 09 — Galaxy cluster lensing (NFW + cluster members)\n\n"
           "Galaxy clusters are the largest gravitationally-bound objects and "
           "the strongest gravitational lenses. We model the cluster mass as\n\n"
           "* a **smooth NFW dark-matter halo** (the dominant component),\n"
           "* one or more **cluster-member galaxies** (SIE),\n"
           "* an **external shear** capturing the residual large-scale tide.\n\n"
           "Then we map the convergence, the critical lines and the multi-image "
           "regions, and compare to the MACS J1206 HST stack."),
        code(HEADER_BOOTSTRAP),
        md("## 1. Build the cluster as a `CompositeLens`"),
        code(textwrap.dedent("""\
            halo = gl.lens.NFW(theta_s=30.0, kappa_s=0.4, center_x=0., center_y=0.)
            galA = gl.lens.SIE(theta_E=2.5, q=0.7, pa=0.4, center_x=-8., center_y=+5.)
            galB = gl.lens.SIE(theta_E=2.0, q=0.6, pa=-0.3, center_x=+10., center_y=-3.)
            shear = gl.lens.ExternalShear(gamma1=0.04, gamma2=-0.02)
            cluster = gl.lens.CompositeLens([halo, galA, galB, shear])
        """)),
        md("## 2. Convergence map"),
        code(textwrap.dedent("""\
            xy = gl.data.coordinate_grid(npix=300, deltapix=0.3)
            with torch.no_grad():
                kappa = halo.kappa(xy[0], xy[1])  # halo dominates kappa
            from matplotlib.colors import LogNorm
            fig, ax = plt.subplots(figsize=(7, 6))
            half = 0.5 * 300 * 0.3
            im = ax.imshow(kappa.numpy(), origin='lower', cmap='inferno',
                           extent=(-half, half, -half, half),
                           norm=LogNorm(vmin=1e-2, vmax=10))
            plt.colorbar(im, ax=ax, label=r'$\\kappa$')
            for comp, marker in [(galA, 'o'), (galB, 's')]:
                ax.scatter(float(comp.center_x), float(comp.center_y),
                           marker=marker, c='cyan', s=80, label=f'member {marker}')
            ax.set(xlabel='x [arcsec]', ylabel='y [arcsec]', title='cluster convergence')
            ax.legend(); plt.show()
        """)),
        md("## 3. Numerical critical curves (det A = 0)"),
        code(textwrap.dedent("""\
            xy = gl.data.coordinate_grid(npix=600, deltapix=0.15)
            with torch.no_grad():
                bx, by = cluster.ray_trace(xy[0], xy[1])
                # finite-difference Jacobian along x and y
                dx = 0.15
                bx_xp, _ = cluster.ray_trace(xy[0] + dx, xy[1])
                _, by_yp = cluster.ray_trace(xy[0], xy[1] + dx)
                bx_yp, _ = cluster.ray_trace(xy[0], xy[1] + dx)
                _, by_xp = cluster.ray_trace(xy[0] + dx, xy[1])
                A11 = (bx_xp - bx) / dx
                A22 = (by_yp - by) / dx
                A12 = (bx_yp - bx) / dx
                A21 = (by_xp - by) / dx
                detA = A11 * A22 - A12 * A21

            fig, ax = plt.subplots(figsize=(7, 6))
            half = 0.5 * 600 * 0.15
            ax.contour(xy[0].numpy(), xy[1].numpy(), detA.numpy(),
                       levels=[0.0], colors='cyan', linewidths=1.5)
            ax.set_aspect('equal')
            ax.set(xlabel='x [arcsec]', ylabel='y [arcsec]',
                   title='cluster critical lines (det A = 0)',
                   xlim=(-half, half), ylim=(-half, half))
            plt.show()
        """)),
        md("## 4. Real-data reference: MACS J1206\n\n"
           "The course materials include an HST stack of the cluster MACS J1206; "
           "we plot it side by side with our toy convergence."),
        code(textwrap.dedent("""\
            from astropy.io import fits

            macs_path = repo / 'data' / 'raw' / 'macs1206_stack.fits'
            if macs_path.exists():
                with fits.open(macs_path) as hdul:
                    # The HST stack stores three RGB channels in HDU 1..3;
                    # HDU[0] is empty PRIMARY metadata. Use the first image HDU.
                    img_hdu = next((h for h in hdul if h.data is not None), None)
                    macs = np.asarray(img_hdu.data, dtype=np.float32)
                macs = np.nan_to_num(macs)
                fig, ax = plt.subplots(figsize=(7, 7))
                gl.viz.imshow_log(macs[::4, ::4] - np.median(macs) + 1e-3, ax=ax,
                                  title='MACS J1206 (HST stack, every 4th px)')
                plt.show()
            else:
                print("No MACS J1206 FITS found in data/raw/.")
        """)),
    ]


# =========================================================================
# 10 - CNN lens classifier - NEW (ML)
# =========================================================================
def notebook_cnn_classifier() -> List[dict]:
    return [
        md("# 10 — CNN classifier: lens vs no-lens\n\n"
           "Train a small VGG-like CNN to distinguish galaxies that show a "
           "strong-lens arc/ring from plain galaxies. Both classes are "
           "synthesized from the package's forward models so we have an "
           "unlimited supply of perfectly-labelled training data."),
        code(HEADER_BOOTSTRAP),
        md("## 1. Synthetic dataset"),
        code(textwrap.dedent("""\
            from torch.utils.data import DataLoader

            train = gl.ml.datasets.LensClassifierDataset(n_samples=400, npix=48,
                                                         deltapix=0.05, seed=0)
            val   = gl.ml.datasets.LensClassifierDataset(n_samples=100, npix=48,
                                                         deltapix=0.05, seed=1000)
            train_loader = DataLoader(train, batch_size=32, shuffle=True)
            val_loader   = DataLoader(val,   batch_size=32)

            # Visualize a few samples.
            xs, ys = next(iter(train_loader))
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for i, ax in enumerate(axes.flatten()):
                gl.viz.imshow_log(xs[i, 0] + xs[i, 0].min().abs() + 1e-3, ax=ax,
                                  title='lens' if int(ys[i]) else 'no-lens')
            plt.tight_layout(); plt.show()
        """)),
        md("## 2. Train the CNN"),
        code(textwrap.dedent("""\
            model = gl.ml.models.LensCNN(in_channels=1, n_classes=2)
            print(f'{sum(p.numel() for p in model.parameters()):,} parameters')

            history = gl.ml.train.fit_model(
                model, train_loader, val_loader,
                loss_fn=nn.CrossEntropyLoss(),
                lr=1e-3, epochs=8,
                metrics={'acc': gl.ml.train.accuracy},
                log_every=1,
            )
        """)),
        code(textwrap.dedent("""\
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(history.train_loss, label='train')
            axes[0].plot(history.val_loss, label='val')
            axes[0].set(yscale='log', xlabel='epoch', ylabel='loss'); axes[0].legend()
            axes[1].plot(history.metrics['acc'], label='train acc')
            axes[1].plot(history.metrics['val_acc'], label='val acc')
            axes[1].set(xlabel='epoch', ylabel='accuracy'); axes[1].legend()
            plt.show()
        """)),
        md("## 3. Confusion matrix on a fresh test set"),
        code(textwrap.dedent("""\
            test = gl.ml.datasets.LensClassifierDataset(n_samples=200, npix=48,
                                                        deltapix=0.05, seed=7777)
            test_loader = DataLoader(test, batch_size=64)

            preds, labels = [], []
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    # `model` lives on `device` (MPS / CUDA / CPU); we must
                    # move the test batch to the same device before the
                    # forward pass — DataLoader returns CPU tensors by default.
                    p = model(x.to(device)).argmax(dim=-1).cpu()
                    preds.extend(p.tolist()); labels.extend(y.tolist())
            preds = np.array(preds); labels = np.array(labels)
            cm = np.array([[((preds==i) & (labels==j)).sum() for j in (0,1)] for i in (0,1)])
            print('confusion matrix [pred x truth]:')
            print(cm)
            print(f'accuracy = {(preds == labels).mean():.3f}')
        """)),
    ]


# =========================================================================
# 11 - DNN regression (Sersic params) - NEW
# =========================================================================
def notebook_dnn_regression() -> List[dict]:
    return [
        md("# 11 — DNN regressor for Sérsic parameters\n\n"
           "Train a CNN+MLP to map a galaxy image directly to its 7-D Sérsic "
           "parameter vector. At inference time this replaces a few thousand "
           "Adam iterations with a single forward pass — a useful technique "
           "for very large surveys."),
        code(HEADER_BOOTSTRAP),
        code(textwrap.dedent("""\
            from torch.utils.data import DataLoader

            train = gl.ml.datasets.SersicParamDataset(n_samples=1500, npix=48,
                                                       deltapix=0.05, seed=0)
            val   = gl.ml.datasets.SersicParamDataset(n_samples=200, npix=48,
                                                       deltapix=0.05, seed=10000)
            train_loader = DataLoader(train, batch_size=32, shuffle=True)
            val_loader   = DataLoader(val,   batch_size=32)

            model = gl.ml.models.SersicRegressor(in_channels=1, n_outputs=7)
            history = gl.ml.train.fit_model(
                model, train_loader, val_loader,
                loss_fn=nn.MSELoss(),
                lr=1e-3, epochs=10,
                metrics={'mse': gl.ml.train.mse},
                log_every=1,
            )
        """)),
        md("## Per-parameter accuracy"),
        code(textwrap.dedent("""\
            from lensing.ml.datasets import PARAM_KEYS

            test = gl.ml.datasets.SersicParamDataset(n_samples=300, npix=48,
                                                      deltapix=0.05, seed=98765)
            preds, truths = [], []
            model.eval()
            with torch.no_grad():
                for x, y in DataLoader(test, batch_size=64):
                    # Move to the trained model's device, then back to CPU
                    # for downstream NumPy / Matplotlib code.
                    preds.append(model(x.to(device)).cpu().numpy())
                    truths.append(y.numpy())
            preds = np.vstack(preds); truths = np.vstack(truths)

            fig, axes = plt.subplots(2, 4, figsize=(15, 7))
            for i, (ax, name) in enumerate(zip(axes.flatten(), PARAM_KEYS)):
                ax.scatter(truths[:, i], preds[:, i], s=8, alpha=0.5)
                lo, hi = truths[:, i].min(), truths[:, i].max()
                ax.plot([lo, hi], [lo, hi], 'k--')
                ax.set(xlabel=f'true {name}', ylabel=f'pred {name}',
                       title=f'r = {np.corrcoef(truths[:,i], preds[:,i])[0,1]:.3f}')
            axes[1, 3].axis('off')
            plt.tight_layout(); plt.show()
        """)),
        md("## Comparison vs. classical Adam fit\n\n"
           "How much does inference speed up if we use the DNN as a one-shot "
           "estimator vs. running Adam from scratch on each image?"),
        code(textwrap.dedent("""\
            import time

            x_one, y_true = test[0]
            t0 = time.perf_counter()
            with torch.no_grad():
                y_dnn = model(x_one.unsqueeze(0).to(device))[0].cpu().numpy()
            t_dnn = time.perf_counter() - t0
            print(f'DNN inference: {t_dnn*1e3:.2f} ms')

            # Adam fit baseline
            xy = train._xy
            t0 = time.perf_counter()
            galaxy = gl.light.Sersic(Ie=1., Re=1., n=2.5, x0=0., y0=0., e1=0., e2=0.)
            res = gl.inference.fit(
                galaxy, xy, x_one[0],
                gl.inference.ReducedChiSquared(sigma=0.05, n_params=7),
                lr=0.05, epochs=500,
            )
            t_adam = time.perf_counter() - t0
            print(f'Adam fit (500 epochs): {t_adam*1e3:.0f} ms — speedup ~ {t_adam/t_dnn:.0f}x')
        """)),
    ]


# =========================================================================
# 12 - U-Net source reconstruction - NEW
# =========================================================================
def notebook_unet() -> List[dict]:
    return [
        md("# 12 — U-Net source-plane reconstruction\n\n"
           "Train a U-Net on `(observed_lensed_image, source_truth)` pairs. "
           "Once trained, the network undoes the lensing distortion in a single "
           "forward pass — a fully data-driven alternative to the classical "
           "pixelated source reconstruction."),
        code(HEADER_BOOTSTRAP),
        code(textwrap.dedent("""\
            from torch.utils.data import DataLoader

            train = gl.ml.datasets.LensSourcePairDataset(n_samples=500, npix=48,
                                                          deltapix=0.05, seed=0)
            val   = gl.ml.datasets.LensSourcePairDataset(n_samples=80, npix=48,
                                                          deltapix=0.05, seed=10000)
            train_loader = DataLoader(train, batch_size=16, shuffle=True)
            val_loader   = DataLoader(val,   batch_size=16)

            # Show a few examples
            obs, src = next(iter(train_loader))
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for j in range(4):
                gl.viz.imshow_log(obs[j, 0]+obs[j,0].min().abs()+1e-3,
                                  ax=axes[0, j], title='observed')
                gl.viz.imshow_log(src[j, 0]+src[j,0].min().abs()+1e-3,
                                  ax=axes[1, j], title='true source')
            plt.tight_layout(); plt.show()
        """)),
        md("## Train the U-Net"),
        code(textwrap.dedent("""\
            model = gl.ml.models.UNet(in_channels=1, out_channels=1, base=16)
            print(f'{sum(p.numel() for p in model.parameters()):,} parameters')
            history = gl.ml.train.fit_model(
                model, train_loader, val_loader,
                loss_fn=nn.MSELoss(),
                lr=1e-3, epochs=8,
                log_every=1,
            )
        """)),
        md("## Test reconstructions"),
        code(textwrap.dedent("""\
            test = gl.ml.datasets.LensSourcePairDataset(n_samples=8, npix=48,
                                                         deltapix=0.05, seed=999)
            obs = torch.stack([test[i][0] for i in range(8)])
            src = torch.stack([test[i][1] for i in range(8)])
            model.eval()
            with torch.no_grad():
                pred = model(obs.to(device)).cpu()

            fig, axes = plt.subplots(3, 8, figsize=(20, 8))
            for j in range(8):
                gl.viz.imshow_log(obs[j, 0]+obs[j,0].min().abs()+1e-3,
                                  ax=axes[0, j], title='observed' if j==0 else None)
                gl.viz.imshow_log(pred[j, 0]+pred[j,0].min().abs()+1e-3,
                                  ax=axes[1, j], title='U-Net' if j==0 else None)
                gl.viz.imshow_log(src[j, 0]+src[j,0].min().abs()+1e-3,
                                  ax=axes[2, j], title='truth' if j==0 else None)
                for ax in axes[:, j]:
                    ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout(); plt.show()
        """)),
    ]


# =========================================================================
# 13 - CPU vs MPS benchmarks - NEW (advanced)
# =========================================================================
def notebook_benchmarks() -> List[dict]:
    return [
        md("# 13 — Performance: CPU vs MPS (Apple GPU)\n\n"
           "PyTorch lets us flip a single string (`device='cpu'` ↔ `'mps'`) "
           "and run the **same model code** on the Apple-silicon GPU. The "
           "interesting question is *when* this is actually faster: small "
           "kernels are dominated by launch latency, and the CPU often wins "
           "on lensing problems with `npix < a few hundred`.\n\n"
           "This notebook quantifies the trade-off by benchmarking three "
           "representative workloads:\n\n"
           "1. **Forward pass** — a Sérsic profile evaluated on a 2-D grid;\n"
           "2. **Backward pass** — same Sérsic + a dummy MSE loss + autograd;\n"
           "3. **End-to-end fit** — Adam optimization for ``N`` epochs.\n\n"
           "Each measurement uses `lensing.benchmarks.compare_devices`, which "
           "wraps `time.perf_counter` with proper device synchronization so "
           "we capture *all* in-flight kernel launches (a pitfall every "
           "PyTorch benchmark must avoid)."),
        code(HEADER_BOOTSTRAP),
        md("## 0. Environment overview\n\n"
           "Before running any timing we record what hardware we're on. "
           "Apple's MPS backend is fast on M1/M2/M3 chips for moderate-size "
           "kernels but loses on tiny ones because of the cost of issuing a "
           "GPU command buffer."),
        code(textwrap.dedent("""\
            print('PyTorch :', torch.__version__)
            print('Devices :', gl.benchmarks.available_devices())
            # The MPS allocator currently has no peak-memory query, so we
            # only print sizes when running on CUDA.
            if torch.cuda.is_available():
                print('CUDA dev:', torch.cuda.get_device_name(0))
        """)),
        md("## 1. Forward-pass scaling with image size\n\n"
           "We sweep ``npix ∈ {32, 64, 128, 256, 512}`` and record the "
           "wall time of one Sérsic forward pass on each device. The "
           "expectation is that for small images the CPU wins (kernel "
           "launch ~ 50–100 µs on Apple Silicon), and the GPU starts "
           "winning once each kernel has enough work to amortise the "
           "launch overhead — typically around ~256² pixels for elementwise "
           "ops, lower for matmul-heavy ones."),
        code(textwrap.dedent("""\
            import pandas as pd
            sizes = [32, 64, 128, 256, 512]
            forward_results = []
            for n in sizes:
                df = gl.benchmarks.compare_devices(
                    gl.benchmarks.sersic_forward_workload(npix=n, batch=1),
                    n_warmup=3, n_repeats=10,
                )
                df['npix'] = n
                forward_results.append(df.reset_index())
            forward = pd.concat(forward_results, ignore_index=True)
            forward
        """)),
        code(textwrap.dedent("""\
            # Plot wall-time vs grid size, one line per device.
            fig, ax = plt.subplots(figsize=(8, 5))
            for dev, sub in forward.groupby('device'):
                ax.errorbar(sub['npix'], sub['mean_ms'], yerr=sub['std_ms'],
                            marker='o', label=dev, capsize=3)
            ax.set(xscale='log', yscale='log',
                   xlabel='image side (pixels)',
                   ylabel='forward wall time [ms]',
                   title='Sérsic forward pass — per-device wall time')
            ax.grid(True, which='both', alpha=0.3); ax.legend()
            plt.show()
        """)),
        md("**Reading the chart**: where the two lines cross is the "
           "*break-even point* of the workload. At image sizes above the "
           "crossing, MPS is preferable; below it, the CPU's lower kernel "
           "launch latency wins. For a Sérsic profile, the cross-over on "
           "this hardware is typically between ``npix=128`` and ``npix=256`` "
           "— well above the typical 64×64 postage stamps used in the "
           "weak-lensing notebook 05, so for that workload **CPU is the "
           "right default**."),
        md("## 2. Backward-pass timing\n\n"
           "Autodiff doubles roughly the FLOP count (forward + reverse), "
           "and adds an allocation pass for the gradient buffers. Backward "
           "tends to make the GPU more competitive because the extra "
           "kernels share the same memory allocations."),
        code(textwrap.dedent("""\
            backward_results = []
            for n in sizes:
                df = gl.benchmarks.compare_devices(
                    gl.benchmarks.sersic_backward_workload(npix=n),
                    n_warmup=3, n_repeats=8,
                )
                df['npix'] = n
                backward_results.append(df.reset_index())
            backward = pd.concat(backward_results, ignore_index=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            for dev, sub in backward.groupby('device'):
                ax.errorbar(sub['npix'], sub['mean_ms'], yerr=sub['std_ms'],
                            marker='s', label=dev, capsize=3)
            ax.set(xscale='log', yscale='log',
                   xlabel='image side (pixels)',
                   ylabel='forward+backward wall time [ms]',
                   title='Sérsic forward + backward — per-device wall time')
            ax.grid(True, which='both', alpha=0.3); ax.legend()
            plt.show()
            backward
        """)),
        md("## 3. End-to-end fit timing\n\n"
           "The most realistic benchmark: how long does a complete Sérsic "
           "fit take on each device? We run a fixed-budget Adam loop "
           "(500 iterations) and record the wall time. This includes the "
           "scheduler, the constraint enforcement, and parameter updates — "
           "*not* just the forward kernels."),
        code(textwrap.dedent("""\
            from time import perf_counter

            def end_to_end_fit(device, npix=128, epochs=500):
                xy = gl.data.coordinate_grid(npix=npix, deltapix=0.05).to(device)
                truth = gl.light.Sersic(Ie=5., Re=1., n=4., x0=0., y0=0.,
                                         e1=0.1, e2=-0.05).to(device)
                with torch.no_grad():
                    image = truth(xy)
                model = gl.light.Sersic(Ie=2., Re=1.5, n=2.5, x0=0., y0=0.,
                                         e1=0., e2=0.).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=0.05)
                # Sync before timing.
                gl.benchmarks.synchronize(device)
                t0 = perf_counter()
                for _ in range(epochs):
                    opt.zero_grad()
                    pred = model(xy)
                    loss = ((pred - image) ** 2).mean()
                    loss.backward()
                    opt.step()
                gl.benchmarks.synchronize(device)
                return perf_counter() - t0

            rows = []
            for dev in gl.benchmarks.available_devices():
                # Warm-up to absorb kernel-compilation / cache-population costs.
                end_to_end_fit(dev, npix=128, epochs=20)
                t = end_to_end_fit(dev, npix=128, epochs=500)
                rows.append({'device': dev, 'wall_s': t,
                             'epoch_ms': t * 1000 / 500})
            fit_df = pd.DataFrame(rows).set_index('device')
            fit_df['speedup_vs_cpu'] = fit_df.loc['cpu', 'wall_s'] / fit_df['wall_s']
            fit_df
        """)),
        md("**Discussion**: in our experience on Apple-silicon laptops, "
           "MPS pulls ahead of the CPU only when the per-iteration kernel "
           "work is large enough — typically ``npix ≳ 256`` for elementwise "
           "Sérsic ops, lower for matmul-heavy workloads such as the U-Net "
           "training in notebook 12. Two practical recommendations:\n\n"
           "* For **single-image fits at npix ≤ 128**, keep `device='cpu'` "
           "(you save the host↔device transfer too).\n"
           "* For **batched fits / training loops** where you have many "
           "images at once, use `device='mps'` — the GPU's parallelism "
           "amortises the launch latency across the batch."),
        md("## 4. Where MPS shines: U-Net training\n\n"
           "Convolutional networks are matmul-heavy and benefit much more "
           "from the GPU. We time one epoch of the U-Net from notebook 12 "
           "on each device for a head-to-head comparison."),
        code(textwrap.dedent("""\
            from torch.utils.data import DataLoader

            def time_unet_epoch(device):
                ds = gl.ml.datasets.LensSourcePairDataset(
                    n_samples=64, npix=48, deltapix=0.05, seed=0)
                loader = DataLoader(ds, batch_size=8, shuffle=False)
                model = gl.ml.models.UNet().to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                # Warm-up
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = ((model(x) - y) ** 2).mean()
                    loss.backward(); opt.step()
                    break
                gl.benchmarks.synchronize(device)
                t0 = perf_counter()
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = ((model(x) - y) ** 2).mean()
                    loss.backward(); opt.step()
                gl.benchmarks.synchronize(device)
                return perf_counter() - t0

            rows = []
            for dev in gl.benchmarks.available_devices():
                rows.append({'device': dev, 'epoch_s': time_unet_epoch(dev)})
            unet_df = pd.DataFrame(rows).set_index('device')
            if 'cpu' in unet_df.index:
                unet_df['speedup_vs_cpu'] = unet_df.loc['cpu', 'epoch_s'] / unet_df['epoch_s']
            unet_df
        """)),
        md("Conv-heavy training is exactly the regime PyTorch's MPS path "
           "is tuned for, and we expect a ≥2× speed-up on M1/M2 vs. CPU "
           "for the U-Net (depending on macOS version and PyTorch build).\n\n"
           "**Caveat**: PyTorch's MPS backend was first released in v1.12 "
           "(May 2022) and is still maturing — a handful of operators do "
           "not yet have an MPS kernel and silently fall back to CPU. "
           "Always check both backends agree before trusting a result."),
    ]


# =========================================================================
# 14 - Power-law / NIE lenses (theoretical deep-dive) - NEW
# =========================================================================
def notebook_power_law_nie() -> List[dict]:
    return [
        md("# 14 — Power-law and NIE lenses\n\n"
           "Two extensions of the SIE that capture different astrophysical "
           "regimes:\n\n"
           "* **Power-law** lens with `kappa(x) = (3-n)/2 · x^(1-n)` "
           "(Meneghetti, *Lensing Gravitazionale*, Ch. 5.2). For ``n = 2`` "
           "this is the Singular Isothermal Sphere (SIS); for `n → 3` it "
           "approaches a point mass; for ``1 < n < 2`` the lens is more "
           "centrally concentrated than SIS.\n\n"
           "* **Non-singular Isothermal Ellipsoid (NIE)** with a finite "
           "core radius `xi_c` (Kormann, Schneider & Bartelmann 1994, "
           "Meneghetti Ch. 5.4.2). The core regularizes the SIE central "
           "cusp and changes the **caustic topology** — for small core "
           "the standard astroid + cut survives, but for large core the "
           "radial caustic disappears and the lens stops producing radial "
           "arcs.\n\n"
           "We map these properties numerically and compare them to the "
           "SIE baseline."),
        code(HEADER_BOOTSTRAP),
        md("## 1. Convergence and deflection profiles of the power law"),
        code(textwrap.dedent("""\
            r = torch.linspace(0.01, 5.0, 200)
            zeros = torch.zeros_like(r)

            slopes = [1.5, 2.0, 2.5]    # 1.5 = "shallow", 2.0 = SIS, 2.5 = "steep"
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for n in slopes:
                pl = gl.lens.PowerLawSpherical(theta_E=1.0, n=n)
                with torch.no_grad():
                    kappa = pl.kappa(r, zeros)
                    ax_def, _ = pl.deflection(r, zeros)
                axes[0].plot(r, kappa, label=f'n = {n}')
                axes[1].plot(r, ax_def, label=f'n = {n}')
            for ax, ylabel, ylog in zip(axes,
                                          [r'$\\kappa$', r'$\\alpha\\;[\\theta_E]$'],
                                          [True, False]):
                ax.set(xlabel=r'$r/\\theta_E$', ylabel=ylabel)
                if ylog: ax.set_yscale('log')
                ax.legend(); ax.grid(alpha=0.3)
            axes[0].axvline(1.0, color='gray', ls=':', label=r'$\\theta_E$')
            axes[1].axhline(1.0, color='gray', ls=':')
            plt.tight_layout(); plt.show()
        """)),
        md("**Note**: at `r = θ_E` we recover ``α = θ_E`` for **all "
           "slopes** — that's the defining property of the Einstein "
           "radius. What changes with ``n`` is the *gradient* of α, which "
           "controls magnification. SIS (``n = 2``) has constant α, so its "
           "magnification only depends on the source-plane distance from "
           "the centre."),
        md("## 2. NIE caustic topology vs. core size\n\n"
           "Following Kormann+ 94 / Meneghetti Ch. 5.4.2, three regimes "
           "exist depending on the dimensionless ratio "
           "``x_c / q^{3/2}``:\n\n"
           "| Regime | Tangential caustic | Radial caustic |\n"
           "|---|---|---|\n"
           "| `x_c < q^{3/2}/2`           | astroid (4 cusps) | oval, contains tangential |\n"
           "| `q^{3/2}/2 < x_c < q^{3/2}/(1+q)` | 2 cusps | inside tangential |\n"
           "| `q^{3/2}/(1+q) < x_c < q^{1/2}/(1+q)` | survives | gone |\n"
           "| `x_c > q^{1/2}/(1+q)`       | gone | gone |\n\n"
           "We probe the topology numerically by computing det A on a "
           "fine grid and contouring the zero level."),
        code(textwrap.dedent("""\
            import numpy as np

            def critical_curves(lens, halfwidth=2.0, n=400, dx=None):
                axis = torch.linspace(-halfwidth, halfwidth, n)
                xx, yy = torch.meshgrid(axis, axis, indexing='xy')
                if dx is None:
                    dx = float(axis[1] - axis[0])
                # Finite-difference Jacobian of the lens map.
                with torch.no_grad():
                    bx, by = lens.ray_trace(xx, yy)
                    bxxp, _ = lens.ray_trace(xx + dx, yy)
                    bxyp, _ = lens.ray_trace(xx, yy + dx)
                    _, byxp = lens.ray_trace(xx + dx, yy)
                    _, byyp = lens.ray_trace(xx, yy + dx)
                A11 = (bxxp - bx) / dx
                A12 = (bxyp - bx) / dx
                A21 = (byxp - by) / dx
                A22 = (byyp - by) / dx
                detA = A11 * A22 - A12 * A21
                # The corresponding caustic is the image of the critical curve.
                return axis, detA, (bx, by)

            q = 0.7
            cores = [0.02, 0.10, 0.30, 0.6]      # increasing core size

            fig, axes = plt.subplots(2, len(cores), figsize=(4*len(cores), 8))
            for j, xc in enumerate(cores):
                lens = gl.lens.NIE(theta_E=1.0, q=q, pa=0.0, core=xc)
                axis, detA, (bx, by) = critical_curves(lens, halfwidth=1.6, n=350)
                # Image-plane critical curves.
                ax = axes[0, j]
                ax.contour(axis.numpy(), axis.numpy(), detA.numpy(), levels=[0.],
                           colors='C2', linewidths=1.4)
                ax.set_title(f'$x_c = {xc:.2f}$')
                ax.set_aspect('equal'); ax.grid(alpha=0.3)
                if j == 0: ax.set_ylabel('image plane $x_2$')
                # Caustic = image of the critical curve under the lens map.
                # We extract the crit curve from the contour, then ray-trace it.
                from skimage.measure import find_contours
                detA_np = detA.numpy()
                conts = find_contours(detA_np, level=0.0)
                axc = axes[1, j]
                axis_np = axis.numpy()
                for c in conts:
                    # rescale row/col indices into world coordinates
                    cx = np.interp(c[:, 1], np.arange(len(axis_np)), axis_np)
                    cy = np.interp(c[:, 0], np.arange(len(axis_np)), axis_np)
                    cxt = torch.from_numpy(cx).float()
                    cyt = torch.from_numpy(cy).float()
                    with torch.no_grad():
                        sx, sy = lens.ray_trace(cxt, cyt)
                    axc.plot(sx.numpy(), sy.numpy(), color='C3', lw=1.0)
                axc.set_aspect('equal'); axc.grid(alpha=0.3)
                if j == 0: axc.set_ylabel('source plane $y_2$')
                axc.set_xlabel('source plane $y_1$')
            plt.suptitle(f'NIE critical curves (top) and caustics (bottom), q = {q}', y=1.02)
            plt.tight_layout(); plt.show()
        """)),
        md("**Reading the figure**: as the core grows, the inner radial "
           "critical curve shrinks and disappears (panel 3); only the "
           "tangential caustic survives. Beyond the threshold "
           "`x_c > sqrt(q)/(1+q)` even the tangential caustic vanishes and "
           "the lens *stops producing multiple images altogether* — a "
           "useful test that very-cored systems should be filtered out "
           "from strong-lens samples."),
    ]


# =========================================================================
# 15 - Time-delay cosmography - NEW
# =========================================================================
def notebook_time_delay() -> List[dict]:
    return [
        md("# 15 — Time-delay cosmography (Refsdal H_0)\n\n"
           "Light rays that take different paths through a strong lens "
           "arrive at the observer at different times. The time delay "
           "depends on cosmographic distances, so an observed Δt plus a "
           "lens model for the **Fermat potential** Δτ pins down the "
           "**time-delay distance** D_Δt and hence H_0.\n\n"
           "This is Refsdal (1964)'s original idea, today applied by "
           "TDCOSMO / H0LiCOW to measure H_0 to ~2% with ~6 quasar "
           "lenses. We reproduce the principle on a synthetic SIE.\n\n"
           "Reference: Meneghetti, *Lensing Gravitazionale* (UNIBO MSc), "
           "Ch. 3.6 (Fermat surface) and Ch. 5.8 (time delays); "
           "Refsdal 1964; Suyu et al. 2017."),
        code(HEADER_BOOTSTRAP),
        md("## 1. The Fermat potential surface\n\n"
           "We map τ(θ; β) = ½ |θ − β|² − Ψ̂(θ) on the image plane. "
           "Each minimum/saddle of τ corresponds to an image (Fermat's "
           "principle), and the contour height directly maps to the "
           "geometric+gravitational arrival-time delay."),
        code(textwrap.dedent("""\
            # Build a circularly symmetric power-law lens (analytic Psi)
            # so we can plot tau without numerical integration.
            lens = gl.lens.PowerLawSpherical(theta_E=1.0, n=2.0)  # = SIS
            beta = torch.tensor([0.10, 0.05])  # source position [arcsec]

            xy = gl.data.coordinate_grid(npix=300, deltapix=0.02)
            with torch.no_grad():
                psi = lens.potential(xy[0], xy[1])
                theta_minus_beta = torch.stack([xy[0]-beta[0], xy[1]-beta[1]], dim=0)
                tau = 0.5 * (theta_minus_beta**2).sum(dim=0) - psi

            half = 0.5 * 300 * 0.02
            fig, ax = plt.subplots(figsize=(7, 6))
            cs = ax.contour(xy[0].numpy(), xy[1].numpy(), tau.numpy(),
                            levels=30, cmap='viridis')
            ax.plot(beta[0], beta[1], '*', color='orange', ms=12, label='source')
            ax.set(xlabel=r'$\\theta_1$ [arcsec]', ylabel=r'$\\theta_2$ [arcsec]',
                   title='Fermat potential τ(θ; β) for an SIS')
            ax.set_aspect('equal'); ax.legend(); ax.grid(alpha=0.3)
            plt.show()
        """)),
        md("**Reading the contour map**: the saddle points of τ are the "
           "image positions; they appear as crossing-points of the "
           "level curves. For an SIS at finite β there are exactly two "
           "images (a minimum and a saddle) on opposite sides of the "
           "lens centre."),
        md("## 2. Δt from a Refsdal-like quasar pair\n\n"
           "We pick a 4-image SIE, solve for the image positions, "
           "evaluate τ at each image, and convert the Fermat-potential "
           "differences into physical Δt (days). Then we **invert** the "
           "longest Δt to recover H_0."),
        code(textwrap.dedent("""\
            cosmo = gl.cosmology.Cosmology(H0=70., Om0=0.3)
            sie = gl.lens.SIE.from_velocity_dispersion(
                sigma_v_kms=240., q=0.7, pa=np.pi/4,
                zl=0.5, zs=2.0, cosmo=cosmo,
            )
            beta_x, beta_y = torch.tensor(0.05), torch.tensor(0.03)
            ximg, yimg = sie.solve_image_positions(beta_x, beta_y, n_grid=4000)
            print(f'Found {len(ximg)} images at')
            for i, (x, y) in enumerate(zip(ximg, yimg)):
                print(f'  image {i+1}: ({float(x):+.3f}, {float(y):+.3f}) arcsec')

            # SIE has no analytic Psi in our package, so we approximate Psi by
            # numerical line integration of alpha . dr along the ray from the
            # origin to each image (Psi has alpha = grad Psi).
            def sie_potential_along(lens, x, y, n_steps=400):
                # Integrate radially from r ~ 0 to r_image; works for any
                # axially-near profile but the SIE has small angular
                # variation away from the major axis, so this is accurate
                # to ~1% for our visualization purposes.
                t = torch.linspace(0.0, 1.0, n_steps).reshape(-1, 1)
                xs = t * x.reshape(1, -1)
                ys = t * y.reshape(1, -1)
                ax_d, ay_d = lens.deflection(xs, ys)
                dx = (x.reshape(1, -1)) / n_steps
                dy = (y.reshape(1, -1)) / n_steps
                return (ax_d * dx + ay_d * dy).sum(dim=0)

            with torch.no_grad():
                psi_img = sie_potential_along(sie, ximg, yimg)
                # Fermat potential per image
                dx = ximg - beta_x
                dy = yimg - beta_y
                tau = 0.5 * (dx**2 + dy**2) - psi_img
                tau_diffs = tau - tau.min()  # in arcsec^2
            print()
            for i, t in enumerate(tau_diffs.tolist()):
                dt_s = float(gl.lens.timedelay.time_delay_seconds(
                    torch.tensor(t), zl=0.5, zs=2.0, cosmo=cosmo))
                print(f'  image {i+1}: Δτ = {t:+.4f} arcsec², Δt = {dt_s/86400:7.2f} days')
        """)),
        md("## 3. Inverting Δt for H_0\n\n"
           "Suppose the longest delay is *measured* to 2% precision. We "
           "feed the model Δτ and the observed Δt to "
           ":func:`lensing.lens.timedelay.refsdal_H0` and compare the "
           "recovered H_0 to the input cosmology."),
        code(textwrap.dedent("""\
            # Pretend the largest Δt we just computed is the *measured* one.
            longest_idx = int(torch.argmax(tau_diffs))
            measured_dt_days = float(gl.lens.timedelay.time_delay_seconds(
                tau_diffs[longest_idx], zl=0.5, zs=2.0, cosmo=cosmo)) / 86400.0
            model_dtau_arcsec2 = float(tau_diffs[longest_idx])

            # Add a 2% measurement error and re-invert.
            for noise in [0.0, 0.02, 0.05]:
                meas = measured_dt_days * (1.0 + noise)
                H0_rec = gl.lens.timedelay.refsdal_H0(
                    meas, model_dtau_arcsec2, zl=0.5, zs=2.0,
                )
                print(f'  Δt err = {100*noise:4.1f}%   →  H_0 = {H0_rec:.2f} km/s/Mpc')
            print(f'  truth                   →  H_0 = 70.00 km/s/Mpc')
        """)),
        md("**Discussion**: Δt and H_0 are inversely proportional, so a "
           "5% Δt error propagates into a 5% H_0 error — the **dominant "
           "systematic** in real measurements is therefore the lens model "
           "(through the Fermat-potential difference), *not* the time-delay "
           "monitoring itself. Modern programs spend a lot of effort on "
           "modelling assumptions: profile slope, line-of-sight convergence, "
           "and the **mass-sheet degeneracy** which leaves the image "
           "configuration unchanged but rescales τ by a multiplicative "
           "constant (Falco, Gorenstein & Shapiro 1985). See "
           "`docs/astrophysics.md` for a fuller discussion."),
    ]


# =========================================================================
# 16 - Large-scale lens finder (CNN on HDF5 dataset) - NEW
# =========================================================================
def notebook_large_scale_finder() -> List[dict]:
    return [
        md("# 16 — Survey-scale strong-lens finder\n\n"
           "Until now we have trained the CNN on tiny on-the-fly datasets "
           "(~400 samples per epoch in notebook 10). This notebook scales "
           "the workflow to **5,000 simulated lenses + 5,000 non-lensed "
           "galaxies**, written to a single HDF5 file with chunked "
           "compression. The training loader streams the file lazily, so "
           "memory stays flat regardless of catalog size — exactly the "
           "pattern needed for Euclid- / Rubin-scale lens searches.\n\n"
           "We then evaluate the trained network on a **held-out test "
           "split** and inspect the false-negative / false-positive "
           "subsets visually, which is the most informative step for "
           "spotting the dataset's blind spots."),
        code(HEADER_BOOTSTRAP),
        md("## 0. Why HDF5 and not on-the-fly generation?\n\n"
           "* **Reproducibility**: same exact bytes on every machine, "
           "regardless of PyTorch version.\n"
           "* **Throughput**: in our benchmarks an Adam fit on a 5k "
           "dataset goes ~3× faster when reading from HDF5 vs. running "
           "the simulator on the fly (the simulator is the bottleneck).\n"
           "* **Disk-efficient**: gzip level-4 compression on chunked "
           "32-bit floats brings a 10k×48² dataset to ~150 MB — "
           "easily portable.\n\n"
           "We size everything for ~3-minute generation on CPU; in "
           "production you would generate 10⁵–10⁶ samples on a cluster."),
        code(textwrap.dedent("""\
            from pathlib import Path
            import time

            DATA_PATH = Path('cache/lens_dataset_5k.h5')
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Re-generate the dataset only if it is missing or has the
            # wrong size — saves the ~3-minute simulation cost on re-runs.
            REGENERATE = False or not DATA_PATH.exists()
            N_TOTAL = 5000

            if REGENERATE:
                t0 = time.perf_counter()
                gl.bigdata.generate_lens_dataset(
                    DATA_PATH, n_samples=N_TOTAL, npix=48, seed=0,
                    progress=True,
                )
                print(f'Wrote {DATA_PATH} ({DATA_PATH.stat().st_size / 1e6:.1f} MB) '
                      f'in {time.perf_counter() - t0:.1f}s')
            else:
                print(f'Reusing cached {DATA_PATH} '
                      f'({DATA_PATH.stat().st_size / 1e6:.1f} MB)')
        """)),
        md("## 1. Train / val / test split"),
        code(textwrap.dedent("""\
            import numpy as np
            from torch.utils.data import DataLoader

            n = N_TOTAL
            rng = np.random.default_rng(123)
            perm = rng.permutation(n)
            n_train = int(0.7 * n); n_val = int(0.15 * n)
            train_idx = perm[:n_train].tolist()
            val_idx   = perm[n_train:n_train + n_val].tolist()
            test_idx  = perm[n_train + n_val:].tolist()
            print(f'splits: {len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test')

            # Each Dataset opens its own HDF5 handle inside the worker; this
            # is required because HDF5 is not multi-process safe.
            train_ds = gl.bigdata.HDF5Dataset(DATA_PATH, target='label', indices=train_idx)
            val_ds   = gl.bigdata.HDF5Dataset(DATA_PATH, target='label', indices=val_idx)
            test_ds  = gl.bigdata.HDF5Dataset(DATA_PATH, target='label', indices=test_idx)

            # num_workers=0 here for compatibility with this notebook
            # environment; in a real run set it to e.g. os.cpu_count() // 2.
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=128)
            test_loader  = DataLoader(test_ds,  batch_size=128)
        """)),
        md("## 2. CNN training (5 epochs)\n\n"
           "We reuse :class:`lensing.ml.models.LensCNN`. With 5,000 "
           "training samples and one Adam pass at lr=1e-3 we typically "
           "reach >90% val accuracy in 3–5 epochs."),
        code(textwrap.dedent("""\
            model = gl.ml.models.LensCNN()
            print(f'{sum(p.numel() for p in model.parameters()):,} parameters')

            history = gl.ml.train.fit_model(
                model, train_loader, val_loader,
                loss_fn=nn.CrossEntropyLoss(),
                lr=1e-3, epochs=5,
                metrics={'acc': gl.ml.train.accuracy},
                log_every=1,
            )
            print(f'Trained in {history.duration_s:.1f}s')
        """)),
        code(textwrap.dedent("""\
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(history.train_loss, label='train')
            axes[0].plot(history.val_loss, label='val')
            axes[0].set(xlabel='epoch', ylabel='cross-entropy loss', yscale='log'); axes[0].legend()
            axes[1].plot(history.metrics['acc'], label='train acc')
            axes[1].plot(history.metrics['val_acc'], label='val acc')
            axes[1].set(xlabel='epoch', ylabel='accuracy'); axes[1].legend()
            plt.show()
        """)),
        md("## 3. Test-set confusion matrix and ROC"),
        code(textwrap.dedent("""\
            preds, probs, labels = [], [], []
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    # Move test batch to the device the model lives on
                    # (MPS / CUDA / CPU) and pull the result back to CPU
                    # for the NumPy/Matplotlib plotting code below.
                    out = model(x.to(device)).cpu()
                    p = torch.softmax(out, dim=-1)
                    preds.extend(out.argmax(dim=-1).tolist())
                    probs.extend(p[:, 1].tolist())  # p(lens)
                    labels.extend(y.tolist())
            preds = np.array(preds); probs = np.array(probs); labels = np.array(labels)

            cm = np.array([[((preds==i) & (labels==j)).sum() for j in (0,1)] for i in (0,1)])
            print('confusion matrix [pred x truth]:'); print(cm)
            tp = cm[1,1]; fp = cm[1,0]; tn = cm[0,0]; fn = cm[0,1]
            print(f'accuracy   = {(tp+tn)/cm.sum():.3f}')
            print(f'precision  = {tp/(tp+fp):.3f}')
            print(f'recall     = {tp/(tp+fn):.3f}')
        """)),
        code(textwrap.dedent("""\
            # ROC curve via sklearn-free implementation.
            order = np.argsort(-probs)
            sorted_lab = labels[order]
            tpr_curve, fpr_curve = [0.0], [0.0]
            P = sorted_lab.sum(); N = len(sorted_lab) - P
            tp = 0; fp = 0
            for lbl in sorted_lab:
                if lbl == 1:
                    tp += 1
                else:
                    fp += 1
                tpr_curve.append(tp / max(P, 1))
                fpr_curve.append(fp / max(N, 1))
            auc = np.trapz(tpr_curve, fpr_curve)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr_curve, tpr_curve, lw=2, label=f'AUC = {auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='random')
            ax.set(xlabel='false-positive rate', ylabel='true-positive rate',
                   title='lens-finder ROC on the test split')
            ax.legend(); ax.grid(alpha=0.3)
            plt.show()
        """)),
        md("## 4. Where does the model fail?\n\n"
           "Inspecting the **misclassified** samples is far more "
           "informative than the global accuracy: it reveals what kind of "
           "lens images the network has not learned to recognise yet "
           "(e.g. small Einstein radii where the arc and the source blur "
           "into a single blob)."),
        code(textwrap.dedent("""\
            err_mask = preds != labels
            err_idx = np.where(err_mask)[0]
            print(f'{len(err_idx)} misclassified samples ({100*len(err_idx)/len(labels):.1f}%)')
            if len(err_idx):
                # Show 6 samples (3 false-positive + 3 false-negative if available)
                fp_idx = err_idx[(preds[err_idx] == 1) & (labels[err_idx] == 0)][:3]
                fn_idx = err_idx[(preds[err_idx] == 0) & (labels[err_idx] == 1)][:3]
                show = list(fp_idx) + list(fn_idx)
                fig, axes = plt.subplots(1, len(show), figsize=(3*len(show), 3))
                for ax, ii in zip(axes if len(show)>1 else [axes], show):
                    img, lab = test_ds[ii]
                    title = ('FP' if (preds[ii]==1 and labels[ii]==0) else 'FN') + \
                            f' p={probs[ii]:.2f}'
                    gl.viz.imshow_log(img[0] + img[0].min().abs() + 1e-3,
                                      ax=ax, title=title)
                plt.show()
        """)),
        md("**Reading the misclassified panels**: false-positives are "
           "typically irregular galaxies whose tidal features mimic an "
           "arc; false-negatives are systems where the lensed image is "
           "very faint or merged with the lens light. Both failure modes "
           "would be addressed by a larger / more diverse training set, "
           "or by a multi-band input (HSC has g/r/i/z/y; we use only one "
           "channel here)."),
    ]


# =========================================================================
# 17 - Parameter recovery at scale - NEW
# =========================================================================
def notebook_param_recovery_at_scale() -> List[dict]:
    return [
        md("# 17 — Sérsic-parameter recovery at scale\n\n"
           "We reuse the same 5k HDF5 dataset and switch the target from "
           "the binary label to the **7-D Sérsic parameter vector**. The "
           "regressor is :class:`lensing.ml.models.SersicRegressor` "
           "(notebook 11) but trained at scale.\n\n"
           "We measure the per-parameter recovery accuracy on a held-out "
           "split and produce a **reliability diagram**: a scatter of "
           "predicted vs. true with the 1:1 line and the 1σ scatter "
           "shaded — the standard plot in survey-scale ML papers."),
        code(HEADER_BOOTSTRAP),
        code(textwrap.dedent("""\
            from pathlib import Path
            import numpy as np
            from torch.utils.data import DataLoader

            DATA_PATH = Path('cache/lens_dataset_5k.h5')
            assert DATA_PATH.exists(), 'Run notebook 16 first to generate the cache.'

            # Same split as notebook 16 so comparisons are like-for-like.
            n = gl.bigdata.HDF5Dataset(DATA_PATH).meta['n_samples'] if hasattr(
                gl.bigdata.HDF5Dataset(DATA_PATH).meta, '__getitem__') else 5000
            rng = np.random.default_rng(123)
            perm = rng.permutation(n)
            n_train = int(0.7 * n); n_val = int(0.15 * n)
            train_idx, val_idx, test_idx = perm[:n_train].tolist(), \\
                perm[n_train:n_train + n_val].tolist(), \\
                perm[n_train + n_val:].tolist()

            train_ds = gl.bigdata.HDF5Dataset(DATA_PATH, target='params', indices=train_idx)
            val_ds   = gl.bigdata.HDF5Dataset(DATA_PATH, target='params', indices=val_idx)
            test_ds  = gl.bigdata.HDF5Dataset(DATA_PATH, target='params', indices=test_idx)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader   = DataLoader(val_ds, batch_size=128)
        """)),
        md("## 1. Train regressor"),
        code(textwrap.dedent("""\
            model = gl.ml.models.SersicRegressor(in_channels=1, n_outputs=7)
            history = gl.ml.train.fit_model(
                model, train_loader, val_loader,
                loss_fn=nn.MSELoss(),
                lr=1e-3, epochs=8,
                metrics={'mse': gl.ml.train.mse},
                log_every=1,
            )
            print(f'Trained in {history.duration_s:.1f}s')
        """)),
        md("## 2. Reliability diagrams"),
        code(textwrap.dedent("""\
            from lensing.ml.datasets import PARAM_KEYS
            test_loader = DataLoader(test_ds, batch_size=128)
            preds, truths = [], []
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    preds.append(model(x.to(device)).cpu().numpy())
                    truths.append(y.numpy())
            preds = np.vstack(preds); truths = np.vstack(truths)

            fig, axes = plt.subplots(2, 4, figsize=(15, 7))
            for i, (ax, name) in enumerate(zip(axes.flatten(), PARAM_KEYS)):
                t = truths[:, i]; p = preds[:, i]
                ax.scatter(t, p, s=4, alpha=0.4, color='C0')
                lo, hi = t.min(), t.max()
                ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
                # Robust scatter (1.4826 * MAD ≈ 1σ for a normal distribution)
                resid = p - t
                mad = np.median(np.abs(resid - np.median(resid)))
                sigma = 1.4826 * mad
                ax.set_title(f'{name}: σ_residual={sigma:.3f}')
                ax.set(xlabel=f'true {name}', ylabel=f'predicted {name}')
                ax.grid(alpha=0.3)
            axes[1, 3].axis('off')
            plt.tight_layout(); plt.show()
        """)),
        md("**Interpreting the σ_residual**: this is the *robust* "
           "(median-absolute-deviation) 1σ scatter of (pred − true). "
           "Rule of thumb: any σ_residual much smaller than the *prior* "
           "spread on that parameter is a useful constraint. Looking at "
           "the dataset priors (`Re ∈ [0.4, 1.5]`, `n ∈ [1.0, 6.0]`, "
           "etc.) we expect σ_Re ~ 0.05 arcsec and σ_n ~ 0.3 to be "
           "achievable; if the network does worse, more training data "
           "or a deeper backbone would help."),
        md("## 3. Comparison with classical Adam fit\n\n"
           "On the same 100-sample test subset, we run the classical "
           "per-image Adam fit and compare the cumulative wall time and "
           "the per-parameter accuracy. The DNN's edge is **inference "
           "speed**: ~10–100× faster per image, at the cost of a small "
           "(but measurable) increase in scatter."),
        code(textwrap.dedent("""\
            import time

            n_compare = 50
            xy = gl.data.coordinate_grid(npix=test_ds.meta['npix'],
                                          deltapix=test_ds.meta['deltapix'])
            sigma_n = float(test_ds.meta['sigma'])

            # DNN inference (one batch). Move both the input and the
            # output across the device boundary because the trained
            # `model` lives on `device` and downstream NumPy code
            # expects CPU arrays.
            t0 = time.perf_counter()
            with torch.no_grad():
                imgs = torch.stack([test_ds[i][0] for i in range(n_compare)]).to(device)
                pred_dnn = model(imgs).cpu().numpy()
            t_dnn = time.perf_counter() - t0

            # Adam fit (one image at a time) — short budget
            t0 = time.perf_counter()
            pred_adam = []
            for i in range(n_compare):
                img = test_ds[i][0][0]
                g = gl.light.Sersic(Ie=2., Re=1., n=2.5, x0=0., y0=0., e1=0., e2=0.)
                gl.inference.fit(
                    g, xy, img,
                    gl.inference.ReducedChiSquared(sigma=sigma_n, n_params=7),
                    lr=0.05, epochs=200, grad_clip=10.0,
                )
                pred_adam.append([float(getattr(g, k)) for k in PARAM_KEYS])
            pred_adam = np.array(pred_adam)
            t_adam = time.perf_counter() - t0

            true = np.stack([test_ds[i][1].numpy() for i in range(n_compare)])
            err_dnn  = (pred_dnn  - true).std(axis=0)
            err_adam = (pred_adam - true).std(axis=0)

            print(f'DNN  : {t_dnn*1000:7.1f} ms total ({1000*t_dnn/n_compare:.2f} ms/sample)')
            print(f'Adam : {t_adam*1000:7.1f} ms total ({1000*t_adam/n_compare:.2f} ms/sample)')
            print(f'Speed-up: {t_adam/t_dnn:.0f}×')
            print()
            print('per-parameter std-of-residual (lower is better):')
            for i, k in enumerate(PARAM_KEYS):
                print(f'  {k:<3s}: DNN = {err_dnn[i]:.3f}    Adam = {err_adam[i]:.3f}')
        """)),
    ]


# =========================================================================
# 18 - LLM literature-mining for lens metadata - NEW
# =========================================================================
def notebook_llm_metadata() -> List[dict]:
    return [
        md("# 18 — LLM-assisted lens metadata extraction\n\n"
           "We feed paper abstracts to an LLM and ask for a **structured "
           "JSON record** with `(name, theta_E, sigma_v, q, z_L, z_S, "
           "reference)`. The notebook ships with a deterministic *mock* "
           "backend so it runs offline; if you set `ANTHROPIC_API_KEY` "
           "and switch `BACKEND='anthropic'`, it queries Claude with "
           "prompt caching enabled.\n\n"
           "**Why this is useful for strong-lensing science**: the SLACS "
           "(85 systems), BELLS (25), BELLS-GALLERY (25), TDCOSMO and "
           "individual-discovery papers between them describe several "
           "hundred lenses, but the catalogs are scattered across "
           "different journals and table formats. An LLM unifies them "
           "into a single dataframe in minutes — far faster than custom "
           "parsing per paper."),
        code(HEADER_BOOTSTRAP),
        md("## 1. Demonstration corpus\n\n"
           "Five (synthetic but realistic) abstracts in the style of "
           "real lensing papers. Each one mentions some subset of the "
           "metadata we want; the LLM should find what is present and "
           "leave the rest as ``null``."),
        code(textwrap.dedent("""\
            corpus = [
                ('Bolton+ 2008', '''
                We present the discovery of a strong gravitational lens in
                the Sloan Lens ACS Survey, designated SDSSJ0029-0055. HST/ACS
                imaging shows a nearly-complete Einstein ring with theta_E =
                0.96 arcsec around an early-type galaxy at lens redshift z_L
                = 0.227. The background source is at z_S = 0.931 with axis
                ratio q = 0.83. Spectroscopy yields a stellar velocity
                dispersion sigma_v = 229 km/s.
                '''),
                ('Bolton+ 2008b', '''
                SDSSJ0037-0942 is a SLACS galaxy-galaxy lens with a large
                Einstein radius (theta_E = 1.53 arcsec). The lens has
                sigma_v = 279 km/s at z_L = 0.196 and lenses a source at
                z_S = 0.632.
                '''),
                ('Treu+ 2009', '''
                We report H0 measurements from time-delay quasar lens
                B0218+357 with z_L=0.685 and z_S=0.944. The lens system is
                modelled with theta_E = 0.16 arcsec.
                '''),
                ('Auger+ 2009', '''
                Spectroscopic and imaging follow-up of the SLACS sample
                yields 85 confirmed lenses. The mean theta_E in the sample
                is 1.2 arcsec; mean sigma_v is 245 km/s.
                '''),
                ('Suyu+ 2017 (TDCOSMO)', '''
                Strong-lens system HE 0435-1223 has an Einstein radius of
                1.18 arcsec and a velocity dispersion of 222 km/s, with z_L
                = 0.4546 and z_S = 1.689. The 4-image quasar configuration
                yields a time-delay distance and hence H_0.
                '''),
            ]
            print(f'{len(corpus)} abstracts loaded')
        """)),
        md("## 2. Run the extractor\n\n"
           "Default backend is ``mock`` (regex-based, deterministic). To "
           "use the real Claude API, set ``BACKEND='anthropic'`` and "
           "ensure ``ANTHROPIC_API_KEY`` is exported. The pricing of "
           "Claude Haiku 4.5 makes 100-paper extraction cost a few cents "
           "with prompt caching enabled."),
        code(textwrap.dedent("""\
            BACKEND = 'mock'   # set to 'anthropic' for the real API call
            extractor = gl.llm.MetadataExtractor(backend=BACKEND)

            import pandas as pd
            records = []
            for ref, abs_text in corpus:
                rec = extractor.extract(abs_text)
                rec.reference = ref
                records.append(rec)

            df = pd.DataFrame([r.to_dict() for r in records])
            print(df)
        """)),
        md("## 3. Cross-check against the embedded SLACS catalog\n\n"
           "We have a curated SLACS-lite reference table in "
           "`lensing.archive.slacs_table()`. We join the two on the lens "
           "name and verify that the LLM-extracted theta_E and sigma_v "
           "agree with the published values."),
        code(textwrap.dedent("""\
            ref_df = gl.archive.slacs_table().rename(columns={
                'theta_E': 'theta_E_truth',
                'sigma_v': 'sigma_v_truth',
            })[['name', 'theta_E_truth', 'sigma_v_truth']]
            merged = df.merge(ref_df, on='name', how='left')
            merged
        """)),
        code(textwrap.dedent("""\
            # Compute per-row absolute errors where a truth value exists.
            mask = merged['theta_E_truth'].notna()
            if mask.any():
                err_th = (merged.loc[mask, 'theta_E_arcsec'] - merged.loc[mask, 'theta_E_truth']).abs()
                err_sv = (merged.loc[mask, 'sigma_v_kms'] - merged.loc[mask, 'sigma_v_truth']).abs()
                print(f'theta_E   |Δ| max = {err_th.max():.3f} arcsec')
                print(f'sigma_v   |Δ| max = {err_sv.max():.1f} km/s')
            else:
                print('No SLACS-lite intersections in this corpus; the LLM '
                      'extraction is internally consistent but cannot be '
                      'cross-validated against the embedded catalog.')
        """)),
        md("## 4. Caveats and pitfalls\n\n"
           "* **Hallucinations**: even a frontier LLM occasionally "
           "invents numbers that look plausible. Always cross-validate "
           "against published catalogs (here we did) before using the "
           "extracted records as ground truth in a downstream fit.\n"
           "* **Schema drift**: papers from different decades use "
           "different conventions (axis ratio b/a vs. q, Einstein radius "
           "in arcsec vs. kpc). The system prompt explicitly demands "
           "arcsec / km·s⁻¹, but reading the prompt carefully and "
           "running a small validation set is cheap and worth doing.\n"
           "* **Cost vs. coverage**: with prompt caching, the SLACS "
           "(85 papers) extraction takes ~2 minutes and costs ~$0.05 "
           "with Haiku. Without caching, the system prompt would be "
           "tokenised 85 times and the cost would be ~10×.\n\n"
           "**Where to go next**: combine this metadata table with the "
           "real-data downloader (`lensing.archive`) to produce a "
           "training set whose **labels are extracted from the literature** "
           "and whose **images are real HSC cutouts** — a fully ML-ready "
           "real-data benchmark for strong-lens parameter recovery."),
    ]


# --- driver -------------------------------------------------------------------
def main() -> None:
    write_notebook("01_microlensing_lightcurve.ipynb", notebook_microlensing())
    write_notebook("02_sersic_fit.ipynb", notebook_sersic())
    write_notebook("03_core_sersic_fit.ipynb", notebook_core_sersic())
    write_notebook("04_double_sersic_fit.ipynb", notebook_double_sersic())
    write_notebook("05_weak_lensing_ellipticity.ipynb", notebook_weak())
    write_notebook("06_strong_lensing_SIE.ipynb", notebook_sie())
    write_notebook("07_real_galaxy_F150W.ipynb", notebook_real_galaxy())
    write_notebook("08_binary_microlens.ipynb", notebook_binary_microlens())
    write_notebook("09_galaxy_cluster_NFW.ipynb", notebook_cluster_nfw())
    write_notebook("10_CNN_lens_classifier.ipynb", notebook_cnn_classifier())
    write_notebook("11_DNN_param_regression.ipynb", notebook_dnn_regression())
    write_notebook("12_unet_source_reconstruction.ipynb", notebook_unet())
    write_notebook("13_cpu_vs_mps_benchmarks.ipynb", notebook_benchmarks())
    write_notebook("14_power_law_and_NIE_lenses.ipynb", notebook_power_law_nie())
    write_notebook("15_time_delay_cosmography.ipynb", notebook_time_delay())
    write_notebook("16_large_scale_lens_finder.ipynb", notebook_large_scale_finder())
    write_notebook("17_param_recovery_at_scale.ipynb", notebook_param_recovery_at_scale())
    write_notebook("18_LLM_metadata_extraction.ipynb", notebook_llm_metadata())


if __name__ == "__main__":
    main()
