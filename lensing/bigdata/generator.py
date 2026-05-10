"""Write a synthetic strong-lens dataset to an HDF5 file.

Why HDF5 (and not e.g. WebDataset / parquet)?

* Random access: the U-Net training loop wants to draw indices, not
  read sequentially, so plain TFRecords / WebDataset would be a poor
  fit;
* Multi-process safe: HDF5 with ``swmr=False`` is read-safe across
  processes provided each worker opens its own handle (the dataset
  class below does this);
* Stable on disk: a single file moves cleanly between machines, makes
  ``rsync`` easy, and is read by Julia / Matlab / R if needed.

Layout of the produced file::

    images   : (N, 1, H, W) float32  — observed (lensed/unlensed) PSF + noise
    sources  : (N, 1, H, W) float32  — unlensed source on the same grid
                                       (only for U-Net training; can be skipped)
    params   : (N, 7)       float32  — Sérsic params (Ie,Re,n,x0,y0,e1,e2)
    lens     : (N, 5)       float32  — SIE params (theta_E,q,pa,cx,cy)
    label    : (N,)         int64    — 1 if a strong lens was applied, else 0
    sigma    : ()           float32  — noise std used to generate the dataset
    psf_fwhm : ()           float32  — PSF FWHM in arcsec
    deltapix : ()           float32  — pixel scale in arcsec
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..data.grid import coordinate_grid
from ..data.noise import add_gaussian_noise
from ..lens.sie import SIE
from ..light.psf import convolve_psf, gaussian_psf_kernel
from ..light.sersic import Sersic


@dataclass
class HDF5Generator:
    """Serialise a synthetic strong-lens dataset to HDF5.

    The default settings produce an *equally balanced* (lens / no-lens)
    dataset suitable for the CNN classifier of notebook 16, *and* a
    Sérsic-parameter regression target suitable for notebook 17. The
    same file therefore covers two case studies; we save only one
    HDF5 instead of two.
    """

    n_samples: int = 5_000
    npix: int = 48
    deltapix: float = 0.05
    psf_fwhm: float = 0.10
    psf_size: int = 11
    noise_sigma: float = 0.05
    seed: int = 0
    write_sources: bool = True       # also write the unlensed source planes
    chunk: int = 256                 # HDF5 chunk size along sample axis

    # Internals
    _xy: torch.Tensor = field(init=False, repr=False)
    _psf: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        self._xy = coordinate_grid(npix=self.npix, deltapix=self.deltapix)
        self._psf = gaussian_psf_kernel(self.psf_fwhm, self.deltapix, size=self.psf_size)

    # ------------------------------------------------------------------
    def _draw_one(self, rng: np.random.Generator, gen: torch.Generator):
        """Draw one (image, source, params, lens, label) tuple.

        Half of samples are non-lensed Sérsic galaxies; the other half
        are sources ray-traced through a random SIE. We use the same
        Sérsic for both classes so the classifier cannot pick up shortcut
        features (e.g. galaxy size, n).
        """
        # Source Sérsic
        src_params = (
            float(rng.uniform(2.0, 6.0)),                       # Ie
            float(rng.uniform(0.10, 0.30)),                      # Re
            float(rng.uniform(1.0, 4.0)),                        # n
            float(rng.uniform(-0.10, 0.10)),                     # x0
            float(rng.uniform(-0.10, 0.10)),                     # y0
            float(rng.uniform(-0.20, 0.20)),                     # e1
            float(rng.uniform(-0.20, 0.20)),                     # e2
        )
        is_lens = bool(rng.integers(0, 2))
        # SIE (only used if is_lens; otherwise we still record placeholder)
        if is_lens:
            sie_params = (
                float(rng.uniform(0.6, 1.4)),                    # theta_E
                float(rng.uniform(0.5, 0.95)),                   # q
                float(rng.uniform(0.0, np.pi)),                  # pa
                0.0, 0.0,                                        # center fixed
            )
        else:
            sie_params = (0.0, 1.0, 0.0, 0.0, 0.0)

        with torch.no_grad():
            source = Sersic(
                Ie=src_params[0], Re=src_params[1], n=src_params[2],
                x0=src_params[3], y0=src_params[4],
                e1=src_params[5], e2=src_params[6],
            )
            src_img = source(self._xy)
            if is_lens:
                sie = SIE(theta_E=sie_params[0], q=sie_params[1],
                          pa=sie_params[2])
                bx, by = sie.ray_trace(self._xy[0], self._xy[1])
                obs = source(torch.stack([bx, by], dim=0))
            else:
                obs = src_img
            obs = convolve_psf(obs, self._psf)
            obs = add_gaussian_noise(obs, sigma=self.noise_sigma, generator=gen)

        return (
            obs.numpy().astype(np.float32),
            src_img.numpy().astype(np.float32),
            np.array(src_params, dtype=np.float32),
            np.array(sie_params, dtype=np.float32),
            int(is_lens),
        )

    # ------------------------------------------------------------------
    def write(self, path: str | Path, *, progress: bool = True) -> Path:
        """Generate ``self.n_samples`` rows and write them to ``path``."""
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        H = self.npix
        # Chunk size must not exceed the dataset size in any dimension.
        chunk_n = min(self.chunk, self.n_samples)
        # Open with "w" so re-running overwrites cleanly.
        with h5py.File(path, "w") as f:
            # Datasets — chunked, lz4-style compression for ~3-4× size reduction.
            ds_img = f.create_dataset(
                "images", shape=(self.n_samples, 1, H, H), dtype="f4",
                chunks=(chunk_n, 1, H, H), compression="gzip", compression_opts=4,
            )
            if self.write_sources:
                ds_src = f.create_dataset(
                    "sources", shape=(self.n_samples, 1, H, H), dtype="f4",
                    chunks=(chunk_n, 1, H, H), compression="gzip", compression_opts=4,
                )
            ds_par = f.create_dataset("params", shape=(self.n_samples, 7), dtype="f4")
            ds_lens = f.create_dataset("lens", shape=(self.n_samples, 5), dtype="f4")
            ds_lab = f.create_dataset("label", shape=(self.n_samples,), dtype="i8")

            iterator = range(self.n_samples)
            if progress:
                try:
                    from tqdm.auto import tqdm
                    iterator = tqdm(iterator, desc=f"-> {path.name}")
                except ImportError:
                    pass
            for i in iterator:
                rng = np.random.default_rng(self.seed + i)
                gen = torch.Generator()
                gen.manual_seed(self.seed + i)
                obs, src, par, lp, lab = self._draw_one(rng, gen)
                ds_img[i, 0] = obs
                if self.write_sources:
                    ds_src[i, 0] = src
                ds_par[i] = par
                ds_lens[i] = lp
                ds_lab[i] = lab

            # Scalar metadata so a downstream consumer doesn't have to
            # remember which constants were used.
            f.attrs["sigma"] = float(self.noise_sigma)
            f.attrs["psf_fwhm"] = float(self.psf_fwhm)
            f.attrs["deltapix"] = float(self.deltapix)
            f.attrs["npix"] = int(self.npix)
            f.attrs["seed"] = int(self.seed)
            f.attrs["n_samples"] = int(self.n_samples)

        return path


def generate_lens_dataset(
    path: str | Path,
    n_samples: int = 5000,
    *,
    npix: int = 48,
    seed: int = 0,
    progress: bool = True,
) -> Path:
    """Convenience entry point: generate and write a dataset in one call."""
    g = HDF5Generator(n_samples=n_samples, npix=npix, seed=seed)
    return g.write(path, progress=progress)
