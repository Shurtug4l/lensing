"""Generic training loop for the ML notebooks."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainHistory:
    """Per-epoch loss / metric history."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    duration_s: float = 0.0


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    *,
    device: str = "cpu",
    metrics: Optional[Dict[str, Callable]] = None,
) -> Tuple[float, Dict[str, float]]:
    model.train()
    total_loss, total_n = 0.0, 0
    metric_sums: Dict[str, float] = {k: 0.0 for k in (metrics or {})}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += float(loss.detach()) * bs
        total_n += bs
        if metrics:
            for k, fn in metrics.items():
                metric_sums[k] += float(fn(out.detach(), y)) * bs
    return total_loss / max(total_n, 1), {k: v / max(total_n, 1) for k, v in metric_sums.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    *,
    device: str = "cpu",
    metrics: Optional[Dict[str, Callable]] = None,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss, total_n = 0.0, 0
    metric_sums: Dict[str, float] = {k: 0.0 for k in (metrics or {})}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        bs = x.size(0)
        total_loss += float(loss) * bs
        total_n += bs
        if metrics:
            for k, fn in metrics.items():
                metric_sums[k] += float(fn(out, y)) * bs
    return total_loss / max(total_n, 1), {k: v / max(total_n, 1) for k, v in metric_sums.items()}


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    *,
    loss_fn: Callable = nn.MSELoss(),
    optimizer_cls: type = torch.optim.Adam,
    lr: float = 1e-3,
    epochs: int = 10,
    device: str = "cpu",
    metrics: Optional[Dict[str, Callable]] = None,
    log_every: int = 1,
) -> TrainHistory:
    """Train ``model`` for ``epochs`` epochs and return its history."""
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    history = TrainHistory()
    history.metrics = {k: [] for k in (metrics or {})}
    val_keys = [f"val_{k}" for k in (metrics or {})]
    for k in val_keys:
        history.metrics[k] = []

    start = perf_counter()
    for epoch in range(epochs):
        train_loss, train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device=device, metrics=metrics
        )
        history.train_loss.append(train_loss)
        for k, v in train_metrics.items():
            history.metrics[k].append(v)

        val_loss = float("nan")
        if val_loader is not None:
            val_loss, val_metrics = evaluate(model, val_loader, loss_fn, device=device, metrics=metrics)
            for k, v in val_metrics.items():
                history.metrics[f"val_{k}"].append(v)
        history.val_loss.append(val_loss)

        if log_every and epoch % log_every == 0:
            metric_str = " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            print(
                f"epoch {epoch+1:>3d}/{epochs}  "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  {metric_str}"
            )

    history.duration_s = perf_counter() - start
    return history


# --- Common metric helpers ---------------------------------------------------
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == labels).float().mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Coefficient of determination R^2 (per-batch)."""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum()
    return 1.0 - ss_res / (ss_tot + 1e-12)
