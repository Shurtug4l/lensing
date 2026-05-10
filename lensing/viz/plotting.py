"""Image and residual plot helpers."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def imshow_log(
    image,
    *,
    ax=None,
    title: Optional[str] = None,
    extent: Optional[Tuple[float, float, float, float]] = None,
    cmap: str = "inferno",
    floor: float = 1e-6,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """``imshow`` with a log-stretched colour scale.

    Robust against images containing zeros, negative pixels (background-
    subtracted data) and all-equal patches: we clip below ``floor``, drop
    NaN/Inf and fall back to a linear scale if the dynamic range is too small
    for ``LogNorm`` to be meaningful.
    """
    img = _to_numpy(image).astype(float)
    img = np.where(np.isfinite(img), img, floor)
    img = np.clip(img, floor, None)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    lo = float(np.nanmin(img)) if vmin is None else vmin
    hi = float(np.nanmax(img)) if vmax is None else vmax
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate image - fall back to linear scale.
        im = ax.imshow(img, origin="lower", cmap=cmap, extent=extent)
    else:
        im = ax.imshow(img, origin="lower", cmap=cmap, extent=extent,
                       norm=LogNorm(vmin=max(lo, floor), vmax=hi))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        ax.set_title(title)
    return ax


def plot_residuals(
    data,
    model,
    *,
    sigma: Optional[float] = None,
    ax=None,
    title: str = "Residuals",
    extent: Optional[Tuple[float, float, float, float]] = None,
    vmax: Optional[float] = None,
):
    """Plot ``(data - model)`` either standardized (if sigma given) or raw."""
    diff = _to_numpy(data) - _to_numpy(model)
    if sigma is not None:
        diff = diff / sigma
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    if vmax is None:
        vmax = float(np.percentile(np.abs(diff), 99))
    im = ax.imshow(diff, origin="lower", cmap="RdBu_r", extent=extent, vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    return ax


def plot_loss_history(history, *, ax=None, log_y: bool = True, label: Optional[str] = None):
    """Plot a loss history with sensible defaults."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, lw=1.5, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if log_y:
        ax.set_yscale("log")
    if label:
        ax.legend()
    return ax


def side_by_side(
    images: Iterable,
    titles: Optional[List[str]] = None,
    *,
    log: bool = True,
    extent: Optional[Tuple[float, float, float, float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """Plot a row of images, optionally on a log scale."""
    images = list(images)
    n = len(images)
    if figsize is None:
        figsize = (5 * n, 5)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    titles = titles or [None] * n
    for ax, img, title in zip(axes, images, titles):
        if log:
            imshow_log(img, ax=ax, title=title, extent=extent)
        else:
            arr = _to_numpy(img).astype(float)
            arr = np.where(np.isfinite(arr), arr, 0.0)
            im = ax.imshow(arr, origin="lower", cmap="inferno", extent=extent)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if title:
                ax.set_title(title)
    return fig, axes
