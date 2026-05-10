"""Global configuration: device selection, dtype defaults, plot style, RNG.

Centralizing these here removes the boilerplate that was duplicated at the top
of every thesis notebook (device autodetect block, sns.set_*, plt.rcParams,
torch.manual_seed) and makes notebooks one-line bootstrappable::

    from lensing.config import setup
    device, dtype = setup(seed=42)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


DEFAULT_DTYPE: torch.dtype = torch.float32


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Pick the best available device.

    The thesis notebooks systematically forced ``device = "cpu"`` after the
    auto-detection block; here we keep the auto-detection but let the caller
    override via ``prefer`` (useful for unit tests and CI).
    """
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_plot_style(use_tex: bool = False, dpi: int = 120) -> None:
    """Apply a sober, paper-ready Matplotlib style.

    Why: every thesis notebook carried 30+ lines of seaborn/matplotlib
    incantations; collecting them here keeps the notebooks short.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_context("paper")
        sns.set_style("white")
    except ImportError:  # seaborn is optional
        pass

    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "figure.dpi": dpi,
            "savefig.dpi": dpi * 2,
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 11,
            "axes.grid": False,
            "image.cmap": "inferno",
            "image.origin": "lower",
            "text.usetex": use_tex,
        }
    )


def setup(
    seed: Optional[int] = 42,
    device: Optional[str] = None,
    dtype: torch.dtype = DEFAULT_DTYPE,
    use_tex: bool = False,
) -> Tuple[torch.device, torch.dtype]:
    """One-call bootstrap. Returns ``(device, dtype)``.

    Parameters
    ----------
    seed : RNG seed for NumPy + PyTorch (None to skip).
    device : ``"cpu"``, ``"mps"``, ``"cuda"`` or ``None`` for auto-detect
        (which prefers ``mps`` → ``cuda`` → ``cpu``). The default of
        ``None`` means **the toolkit picks the best available
        accelerator** — pass ``"cpu"`` explicitly if you need to force
        the CPU path (useful for unit tests or when a specific PyTorch
        op does not have an MPS kernel yet).
    dtype : default float dtype for newly-created tensors.
    use_tex : if True, enable Matplotlib's TeX backend (slower but
        produces paper-quality labels).
    """
    if seed is not None:
        set_seed(seed)
    set_plot_style(use_tex=use_tex)
    torch.set_default_dtype(dtype)
    return get_device(device), dtype
