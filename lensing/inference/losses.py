"""Loss functions used by the fitting code.

The thesis used three patterns interchangeably:

1. ``MSELoss`` on log10 of the image (good for high dynamic range, but biases
   toward bright pixels and breaks in the presence of negative noise);

2. ``red_chi2 = sum((obs - model)^2 / sigma^2) / (N - p)`` (the canonical
   astronomy choice, used for microlensing and Sérsic fits);

3. an ad-hoc Gaussian NLL.

We collect all three here and document tradeoffs.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn


class ReducedChiSquared(nn.Module):
    r"""Reduced chi-squared loss.

    .. math::
        \chi^2_\nu = \frac{1}{N - p} \sum_i \frac{(d_i - m_i)^2}{\sigma_i^2}

    Parameters
    ----------
    sigma : per-pixel uncertainty (scalar or tensor matching the data shape)
    n_params : number of free parameters in the model (for the d.o.f. count)
    """

    def __init__(self, sigma, n_params: int):
        super().__init__()
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.get_default_dtype()))
        self.n_params = int(n_params)

    def forward(self, model: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        residuals = (data - model) / self.sigma
        n = data.numel() - self.n_params
        return (residuals ** 2).sum() / max(n, 1)


class GaussianNLL(nn.Module):
    """Gaussian negative log-likelihood (omits the constant additive term).

    Equivalent up to a constant to ``ReducedChiSquared`` for fixed sigma but
    drops the d.o.f. division - useful when comparing models with different
    parameter counts.
    """

    def __init__(self, sigma):
        super().__init__()
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.get_default_dtype()))

    def forward(self, model: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return 0.5 * (((data - model) / self.sigma) ** 2).sum()


def log_image_mse(model: torch.Tensor, data: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``MSE(log(model + eps), log(data + eps))`` — useful for galaxy fits.

    Why: galaxy surface brightness varies over many decades; an MSE on the
    log compresses this dynamic range and treats core and outskirts more
    equally. Adding ``eps`` regularizes the log against tiny / negative pixels
    arising from background subtraction.
    """
    a = torch.clamp(model, min=eps)
    b = torch.clamp(data, min=eps)
    return torch.mean((torch.log(a) - torch.log(b)) ** 2)


# Pretty type alias: a loss callable takes (model_pred, data) -> scalar tensor.
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
