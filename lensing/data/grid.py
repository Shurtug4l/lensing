"""Coordinate-grid builders for image-plane simulations.

Replaces this 5-line idiom that appeared in every notebook::

    deltapix = 0.03
    numpix = 3000
    side = int(numpix * deltapix)
    xy = np.meshgrid(np.linspace(-numpix/2*deltapix, numpix/2*deltapix, side),
                     np.linspace(-numpix/2*deltapix, numpix/2*deltapix, side))
    xy = torch.tensor(np.array(xy), dtype=torch.float32)
"""
from __future__ import annotations

from typing import Tuple

import torch


def coordinate_grid(
    npix: int,
    deltapix: float,
    center: Tuple[float, float] = (0.0, 0.0),
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Square ``(2, npix, npix)`` coordinate tensor in arcsec.

    The grid is centred on ``center`` (sky coordinates) and the half-side is
    ``npix * deltapix / 2``.
    """
    dtype = dtype or torch.get_default_dtype()
    half = npix * deltapix / 2.0
    axis = torch.linspace(-half, half, npix, dtype=dtype, device=device)
    x, y = torch.meshgrid(axis + center[0], axis + center[1], indexing="xy")
    return torch.stack([x, y], dim=0)
