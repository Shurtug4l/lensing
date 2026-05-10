"""External shear (constant tidal field) added on top of a primary lens.

A galaxy- or group-scale lens often sits in the tidal field of a more
extended structure (a galaxy group or filament). To first order this is a
constant-shear convergence-free perturbation, parameterized by

    Psi_ext = (gamma1/2)(x^2 - y^2) + gamma2 x y

with deflection alpha_ext = (gamma1 x + gamma2 y, gamma2 x - gamma1 y).

Use as ``CompositeLens([SIE(...), ExternalShear(g1, g2)])``.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .._helpers import _as_param


class ExternalShear(nn.Module):
    """Constant external shear (gamma1, gamma2)."""

    def __init__(self, gamma1: float = 0.0, gamma2: float = 0.0):
        super().__init__()
        self.gamma1 = _as_param(gamma1, "gamma1")
        self.gamma2 = _as_param(gamma2, "gamma2")

    def deflection(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ax = self.gamma1 * x + self.gamma2 * y
        ay = self.gamma2 * x - self.gamma1 * y
        return ax, ay

    def ray_trace(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ax, ay = self.deflection(x, y)
        return x - ax, y - ay


class CompositeLens(nn.Module):
    """Sum of arbitrary lens components (deflections add linearly)."""

    def __init__(self, components):
        super().__init__()
        self.components = nn.ModuleList(components)

    def deflection(self, x: torch.Tensor, y: torch.Tensor):
        ax_total = torch.zeros_like(x)
        ay_total = torch.zeros_like(y)
        for comp in self.components:
            ax, ay = comp.deflection(x, y)
            ax_total = ax_total + ax
            ay_total = ay_total + ay
        return ax_total, ay_total

    def ray_trace(self, x: torch.Tensor, y: torch.Tensor):
        ax, ay = self.deflection(x, y)
        return x - ax, y - ay
