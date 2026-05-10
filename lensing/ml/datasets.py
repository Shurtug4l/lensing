"""Synthetic dataset generators for the ML notebooks.

These generators are deterministic given a seed: a regression model trained on
``SersicParamDataset(seed=0)`` will see the *exact* same images on every run.

The two main datasets:

* :class:`SersicParamDataset`
      ``(image, params)`` pairs, where ``params`` is a 7-vector
      ``(Ie, Re, n, x0, y0, e1, e2)``. Used for the DNN regression notebook.

* :class:`LensClassifierDataset`
      ``(image, label)`` pairs with ``label in {0, 1}`` indicating whether
      the image contains a strong-lens arc/ring. Used for the CNN classifier
      notebook.

* :class:`LensSourcePairDataset`
      ``(observed_image, source_truth)`` pairs for source-plane reconstruction
      (U-Net notebook).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.grid import coordinate_grid
from ..data.noise import add_gaussian_noise
from ..lens.sie import SIE
from ..light.psf import convolve_psf, gaussian_psf_kernel
from ..light.sersic import Sersic


# Note: NumPy generates the random parameters; PyTorch handles the noise.
def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _torch_gen(seed: Optional[int]) -> torch.Generator:
    """Per-sample torch RNG so the dataset is reproducible bit-for-bit."""
    g = torch.Generator()
    g.manual_seed(int(seed) if seed is not None else 0)
    return g


@dataclass
class _GridSpec:
    npix: int
    deltapix: float
    psf_fwhm: float
    psf_size: int = 11

    def grid(self) -> torch.Tensor:
        return coordinate_grid(npix=self.npix, deltapix=self.deltapix)

    def psf(self) -> torch.Tensor:
        return gaussian_psf_kernel(self.psf_fwhm, self.deltapix, size=self.psf_size)


def sample_sersic_params(rng: np.random.Generator) -> dict:
    """Draw a random Sérsic parameter dict on a sensible prior."""
    return dict(
        Ie=float(rng.uniform(2.0, 8.0)),
        Re=float(rng.uniform(0.4, 1.5)),
        n=float(rng.uniform(1.0, 6.0)),
        x0=float(rng.uniform(-0.5, 0.5)),
        y0=float(rng.uniform(-0.5, 0.5)),
        e1=float(rng.uniform(-0.4, 0.4)),
        e2=float(rng.uniform(-0.4, 0.4)),
    )


PARAM_KEYS: tuple[str, ...] = ("Ie", "Re", "n", "x0", "y0", "e1", "e2")


class SersicParamDataset(Dataset):
    """``(image, params)`` pairs with params as a 7-vector.

    Each sample is generated on the fly from a fresh seed so memory stays
    flat regardless of dataset size.
    """

    def __init__(
        self,
        n_samples: int = 2000,
        npix: int = 64,
        deltapix: float = 0.05,
        psf_fwhm: float = 0.10,
        noise_sigma: float = 0.05,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.spec = _GridSpec(npix=npix, deltapix=deltapix, psf_fwhm=psf_fwhm)
        self.noise_sigma = noise_sigma
        self.seed = seed
        # cache the PSF kernel (independent of sample)
        self._psf = self.spec.psf()
        self._xy = self.spec.grid()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = _rng(self.seed + idx)
        params = sample_sersic_params(rng)
        with torch.no_grad():
            galaxy = Sersic(**params)
            img = galaxy(self._xy)
            img = convolve_psf(img, self._psf)
            img = add_gaussian_noise(img, sigma=self.noise_sigma, generator=_torch_gen(self.seed + idx))
        target = torch.tensor([params[k] for k in PARAM_KEYS], dtype=torch.float32)
        return img.unsqueeze(0), target  # (1, H, W), (7,)


class LensClassifierDataset(Dataset):
    """Binary classification: image contains a strong-lens arc or just a galaxy.

    Half the dataset is plain Sérsic galaxies (label 0); the other half is
    the same galaxy whose light is ray-traced through a random SIE that puts
    the source close to the caustic so an arc appears (label 1).
    """

    def __init__(
        self,
        n_samples: int = 2000,
        npix: int = 64,
        deltapix: float = 0.05,
        psf_fwhm: float = 0.10,
        noise_sigma: float = 0.05,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.spec = _GridSpec(npix=npix, deltapix=deltapix, psf_fwhm=psf_fwhm)
        self.noise_sigma = noise_sigma
        self.seed = seed
        self._psf = self.spec.psf()
        self._xy = self.spec.grid()

    def __len__(self) -> int:
        return self.n_samples

    def _draw(self, rng: np.random.Generator, gen: torch.Generator) -> Tuple[torch.Tensor, int]:
        # Source: small Sérsic, near grid center.
        src_params = dict(
            Ie=float(rng.uniform(2.0, 6.0)),
            Re=float(rng.uniform(0.10, 0.30)),
            n=float(rng.uniform(1.0, 4.0)),
            x0=float(rng.uniform(-0.10, 0.10)),
            y0=float(rng.uniform(-0.10, 0.10)),
            e1=float(rng.uniform(-0.2, 0.2)),
            e2=float(rng.uniform(-0.2, 0.2)),
        )
        source = Sersic(**src_params)

        is_lens = bool(rng.integers(0, 2))
        with torch.no_grad():
            if is_lens:
                sie = SIE(
                    theta_E=float(rng.uniform(0.6, 1.4)),
                    q=float(rng.uniform(0.5, 0.95)),
                    pa=float(rng.uniform(0.0, np.pi)),
                )
                bx, by = sie.ray_trace(self._xy[0], self._xy[1])
                xy_src = torch.stack([bx, by], dim=0)
                img = source(xy_src)
            else:
                img = source(self._xy)
            img = convolve_psf(img, self._psf)
            img = add_gaussian_noise(img, sigma=self.noise_sigma, generator=gen)
        return img.unsqueeze(0), int(is_lens)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self._draw(_rng(self.seed + idx), _torch_gen(self.seed + idx))
        return img, torch.tensor(label, dtype=torch.long)


class LensSourcePairDataset(Dataset):
    """``(observed_image, source_image)`` pairs for source-plane reconstruction.

    Both images live on the same pixel grid: the observed image is the
    Sérsic source ray-traced through a random SIE, PSF-convolved and
    noised; the target is the source on the same coordinate grid (unlensed,
    no PSF, no noise). Used for the U-Net notebook.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        npix: int = 64,
        deltapix: float = 0.05,
        psf_fwhm: float = 0.10,
        noise_sigma: float = 0.05,
        seed: int = 0,
    ):
        self.n_samples = n_samples
        self.spec = _GridSpec(npix=npix, deltapix=deltapix, psf_fwhm=psf_fwhm)
        self.noise_sigma = noise_sigma
        self.seed = seed
        self._psf = self.spec.psf()
        self._xy = self.spec.grid()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = _rng(self.seed + idx)
        src_params = dict(
            Ie=float(rng.uniform(2.0, 6.0)),
            Re=float(rng.uniform(0.15, 0.40)),
            n=float(rng.uniform(1.0, 4.0)),
            x0=float(rng.uniform(-0.20, 0.20)),
            y0=float(rng.uniform(-0.20, 0.20)),
            e1=float(rng.uniform(-0.2, 0.2)),
            e2=float(rng.uniform(-0.2, 0.2)),
        )
        sie = SIE(
            theta_E=float(rng.uniform(0.7, 1.3)),
            q=float(rng.uniform(0.5, 0.95)),
            pa=float(rng.uniform(0.0, np.pi)),
        )
        with torch.no_grad():
            source = Sersic(**src_params)
            src_img = source(self._xy)  # unlensed source on the grid
            bx, by = sie.ray_trace(self._xy[0], self._xy[1])
            obs = source(torch.stack([bx, by], dim=0))
            obs = convolve_psf(obs, self._psf)
            obs = add_gaussian_noise(obs, sigma=self.noise_sigma, generator=_torch_gen(self.seed + idx))
        return obs.unsqueeze(0), src_img.unsqueeze(0)
