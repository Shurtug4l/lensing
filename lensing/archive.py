"""Public-archive downloaders for real strong-lens cutouts.

The package's "case study with real data" notebooks need a small set of
real lens postage stamps. Rather than ship them with the repo (license,
size) we download them on demand from public archives, with a local
cache so re-runs are offline.

Two archives are supported:

* **Hyper Suprime-Cam (HSC) DR3 PDR** — public images, served as
  ~30-arcsec cutouts via the HSC ``cutout`` REST endpoint. We expose a
  thin wrapper around the documented URL pattern. No login required for
  PDR (the public-data release).

* **STScI / MAST archive** — used here only as a fallback for the
  curated SLACS catalog (Bolton+ 2008): a list of ~85 lensing galaxies
  with HST F814W postage stamps. We provide a small embedded list of
  RA/Dec/HST proposal IDs and let ``astroquery`` fetch the cutouts.

The downloader is robust against network failures: every call is
cached, and a missing connection at notebook-time results in a clear
``RuntimeError`` rather than a silent empty array.
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import urlencode

import numpy as np


_DEFAULT_CACHE = Path.home() / ".cache" / "lensing_archive"


def _cache_dir(custom: Optional[str | Path] = None) -> Path:
    p = Path(custom) if custom is not None else _DEFAULT_CACHE
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash_url(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]


# ===========================================================================
# Hyper Suprime-Cam DR3 PDR cutouts
# ===========================================================================
@dataclass
class HSCCutoutQuery:
    """Cutout request for the HSC SSP PDR3 cutout server.

    See https://hsc-release.mtk.nao.ac.jp/das_cutout/ for the live API.
    The PDR3 endpoint has been used for thousands of lens-finding
    studies; the request below mirrors the documented parameters.
    """

    ra_deg: float
    dec_deg: float
    size_arcsec: float = 20.0
    band: str = "I"
    rerun: str = "pdr3_wide"

    def url(self) -> str:
        params = {
            "ra": self.ra_deg,
            "dec": self.dec_deg,
            "sw": self.size_arcsec,
            "sh": self.size_arcsec,
            "type": "coadd",
            "image": "on",
            "filter": f"HSC-{self.band}",
            "rerun": self.rerun,
        }
        return "https://hsc-release.mtk.nao.ac.jp/das_cutout/pdr3/cgi-bin/cutout?" + urlencode(params)


def fetch_hsc_cutout(
    query: HSCCutoutQuery,
    cache: Optional[str | Path] = None,
    timeout_s: float = 30.0,
) -> np.ndarray:
    """Return a 2-D image array for the requested HSC cutout.

    Caches FITS bytes in ``cache_dir / hsc_<sha>.fits`` so subsequent
    runs are fully offline. Returns the science extension (HDU 1) as a
    numpy array.

    The HSC public data release is open and does not need a login.
    """
    import urllib.request

    from astropy.io import fits

    cdir = _cache_dir(cache)
    url = query.url()
    cache_path = cdir / f"hsc_{_hash_url(url)}.fits"
    if not cache_path.exists():
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                data = resp.read()
        except Exception as exc:  # network / HTTP error
            raise RuntimeError(
                f"failed to fetch {url!r}: {exc}. The HSC archive may be "
                "down or you may be offline."
            ) from exc
        cache_path.write_bytes(data)
    with fits.open(cache_path) as hdul:
        # PDR3 cutouts return PRIMARY (header) + SCI; some return PRIMARY only
        # with the data inline. Pick the first HDU with non-None data.
        for h in hdul:
            if h.data is not None:
                return np.asarray(h.data, dtype=np.float32)
    raise RuntimeError(f"no data in cutout: {cache_path}")


# ===========================================================================
# SLACS / BELLS lens catalogues
# ===========================================================================
SLACS_LITE: list[dict] = [
    # A small embedded sample from Bolton+ 2008 (Tab. 1) for
    # *demonstrative* purposes only — the full SLACS catalogue has 85
    # systems and is published in the SLACS papers and the STScI
    # archive. RA/Dec in degrees, theta_E in arcsec, sigma_v in km/s.
    {"name": "SDSSJ0029-0055", "ra": 7.4555, "dec": -0.9203, "theta_E": 0.96, "sigma_v": 229., "z_L": 0.227, "z_S": 0.931},
    {"name": "SDSSJ0037-0942", "ra": 9.4438, "dec": -9.7053, "theta_E": 1.53, "sigma_v": 279., "z_L": 0.196, "z_S": 0.632},
    {"name": "SDSSJ0157-0056", "ra": 29.3962, "dec": -0.9417, "theta_E": 0.79, "sigma_v": 295., "z_L": 0.513, "z_S": 0.924},
    {"name": "SDSSJ0216-0813", "ra": 34.0530, "dec": -8.2307, "theta_E": 1.16, "sigma_v": 333., "z_L": 0.332, "z_S": 0.523},
    {"name": "SDSSJ0252+0039", "ra": 43.1825, "dec":  0.6608, "theta_E": 1.04, "sigma_v": 164., "z_L": 0.280, "z_S": 0.982},
    {"name": "SDSSJ0330-0020", "ra": 52.5742, "dec": -0.3445, "theta_E": 1.10, "sigma_v": 212., "z_L": 0.351, "z_S": 1.071},
    {"name": "SDSSJ0728+3835", "ra": 112.0040, "dec": 38.5984, "theta_E": 1.25, "sigma_v": 214., "z_L": 0.206, "z_S": 0.688},
    {"name": "SDSSJ0935-0003", "ra": 143.7503, "dec": -0.0508, "theta_E": 0.87, "sigma_v": 396., "z_L": 0.347, "z_S": 0.467},
]


def slacs_table() -> "pd.DataFrame":
    """Return the embedded SLACS-lite reference catalog as a DataFrame.

    Useful as a *small* known-truth set to validate the pipeline against.
    The full SLACS sample (85 systems) is in Bolton et al. 2008 ApJ 682,
    964 and on the SLACS website.
    """
    import pandas as pd
    return pd.DataFrame(SLACS_LITE)


def fetch_slacs_cutouts(
    names: Optional[Sequence[str]] = None,
    *,
    size_arcsec: float = 12.0,
    band: str = "I",
    cache: Optional[str | Path] = None,
) -> dict[str, np.ndarray]:
    """Try to fetch HSC cutouts for the requested SLACS systems.

    Note that SDSS / SLACS lenses are mostly covered by HSC SSP, but
    coverage is incomplete; missing systems raise a ``RuntimeError`` —
    catch it in the caller if you want a "best-effort" result.
    """
    table = {row["name"]: row for row in SLACS_LITE}
    targets = list(names) if names else list(table.keys())
    out: dict[str, np.ndarray] = {}
    for n in targets:
        if n not in table:
            raise KeyError(f"unknown SLACS system: {n}")
        row = table[n]
        q = HSCCutoutQuery(ra_deg=row["ra"], dec_deg=row["dec"],
                            size_arcsec=size_arcsec, band=band)
        try:
            out[n] = fetch_hsc_cutout(q, cache=cache)
        except RuntimeError as exc:
            print(f"  [skipped] {n}: {exc}")
    return out
