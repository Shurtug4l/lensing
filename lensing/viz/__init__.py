"""Plotting helpers tuned for lensing data.

* :func:`imshow_log` — robust log-stretched image with automatic
  fall-back to a linear scale when the dynamic range is degenerate
  (zero / NaN / all-equal pixels). Avoids the ``LogNorm`` ValueError
  that crashes the default Matplotlib path on noisy real images.
* :func:`plot_residuals`, :func:`side_by_side` — opinionated layouts
  for *(data, model, residual)* triplets used throughout the notebooks.
* :func:`plot_loss_history` — log-y loss curves with sensible defaults.
* :func:`corner_plot`, :func:`marginals_grid` — wrappers around the
  ``corner`` library + seaborn KDE marginals for posterior plots.
"""
from .plotting import imshow_log, plot_loss_history, plot_residuals, side_by_side
from .corner_plot import corner_plot, marginals_grid

__all__ = [
    "imshow_log",
    "plot_loss_history",
    "plot_residuals",
    "side_by_side",
    "corner_plot",
    "marginals_grid",
]
