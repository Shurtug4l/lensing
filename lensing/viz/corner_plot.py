"""Posterior plotting helpers wrapping ``corner`` and seaborn KDEs."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import corner as _corner
except ImportError:  # pragma: no cover
    _corner = None


def corner_plot(
    samples,
    *,
    labels: Optional[Sequence[str]] = None,
    truths: Optional[Sequence[float]] = None,
    quantiles: Sequence[float] = (0.16, 0.5, 0.84),
    color: str = "C0",
):
    """Wrapper around ``corner.corner`` with project-defaults.

    Accepts a DataFrame, a 2D array of shape (N, K), or a dict-like of named
    arrays.
    """
    if _corner is None:
        raise ImportError("`corner` not installed. `pip install corner`.")

    if isinstance(samples, pd.DataFrame):
        data = samples.values
        if labels is None:
            labels = list(samples.columns)
    elif isinstance(samples, dict):
        df = pd.DataFrame(samples)
        data = df.values
        if labels is None:
            labels = list(df.columns)
    else:
        data = np.asarray(samples)

    return _corner.corner(
        data,
        labels=labels,
        truths=truths,
        truth_color="k",
        quantiles=list(quantiles),
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 12},
        color=color,
        fill_contours=True,
        plot_density=True,
        plot_contours=True,
        hist_kwargs={"density": True, "color": "grey", "alpha": 0.5, "linewidth": 1.5},
    )


def marginals_grid(
    samples: pd.DataFrame,
    truths: Optional[Dict[str, float]] = None,
    ncols: int = 3,
    figsize_per: float = 3.0,
):
    """1D marginal posteriors with mean / std / true-value annotations."""
    cols = list(samples.columns)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_per, nrows * figsize_per))
    axes_flat = np.atleast_1d(axes).flatten()

    for i, name in enumerate(cols):
        ax = axes_flat[i]
        x = samples[name].values
        ax.hist(x, bins=40, density=True, color="grey", alpha=0.5)
        ax.axvline(x.mean(), color="r", ls="--", label=f"mean = {x.mean():.3g}")
        ax.axvline(x.mean() - x.std(), color="g", ls="--", label=f"std = {x.std():.3g}")
        ax.axvline(x.mean() + x.std(), color="g", ls="--")
        if truths is not None and name in truths:
            ax.axvline(truths[name], color="k", lw=1.2, label=f"true = {truths[name]:.3g}")
        ax.set_xlabel(name)
        ax.legend(fontsize=8)

    for j in range(len(cols), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.tight_layout()
    return fig, axes
