"""Tiny shared helpers used across modules.

Kept private (leading underscore) so they don't pollute the public API.
"""
from __future__ import annotations

import torch
from torch import nn


def _as_param(value, name: str = "") -> nn.Parameter:
    """Wrap ``value`` as ``nn.Parameter`` (idempotent)."""
    if isinstance(value, nn.Parameter):
        return value
    t = torch.as_tensor(value, dtype=torch.get_default_dtype())
    return nn.Parameter(t.clone().detach())
