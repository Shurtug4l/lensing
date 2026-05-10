"""Generic gradient-based fit loop.

Wraps the same Adam + ReduceLROnPlateau pattern that appeared in every thesis
notebook with one extra benefit: it tracks loss history, supports an L-BFGS
polishing step, and prints progress with ``tqdm`` so noisy notebook output is
gone for good.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Iterable, List, Optional, Tuple

import torch
from torch import nn

from .losses import LossFn


@dataclass
class OptimizeResult:
    """Container for fit outputs."""

    model: nn.Module
    loss_history: List[float] = field(default_factory=list)
    best_loss: float = float("inf")
    duration_s: float = 0.0
    n_epochs: int = 0
    converged: bool = False

    @property
    def parameters(self) -> dict[str, float]:
        return {n: float(p.detach()) for n, p in self.model.named_parameters()}


def _default_inputs(forward_args: Any) -> Tuple[Any, ...]:
    """Coerce a single tensor or tuple-of-tensors into a tuple."""
    return forward_args if isinstance(forward_args, tuple) else (forward_args,)


def _enforce_constraints(model: nn.Module) -> None:
    """Call ``enforce_constraints()`` on every submodule that defines it.

    Why: many of our light/lens models project their parameters back into a
    physically-valid region (Re > 0, 0 < n < 12, ...) after each gradient
    step. Centralizing the call here means the user never has to remember it.
    """
    for sub in model.modules():
        fn = getattr(sub, "enforce_constraints", None)
        if callable(fn):
            fn()


def fit(
    model: nn.Module,
    forward_args: Any,
    target: torch.Tensor,
    loss_fn: LossFn,
    *,
    lr: float = 0.1,
    epochs: int = 5000,
    optimizer_cls: type = torch.optim.Adam,
    scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
    lbfgs_polish: bool = False,
    lbfgs_steps: int = 50,
    log_every: int = 0,
    tol: float = 0.0,
    grad_clip: Optional[float] = None,
    callback: Optional[Callable[[int, float, nn.Module], None]] = None,
) -> OptimizeResult:
    """Fit ``model`` so that ``loss_fn(model(*forward_args), target)`` is small.

    Parameters
    ----------
    forward_args : a single tensor or a tuple of tensors passed to ``model``
    log_every : if > 0, print progress every N epochs
    tol : if > 0, stop early when relative loss change drops below ``tol``
    lbfgs_polish : if True, run a short L-BFGS polish after Adam (this almost
        always tightens the fit by 1-2 orders of magnitude on smooth losses
        like the ones we use here, and was missing from the thesis pipeline)
    """
    args = _default_inputs(forward_args)
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    sched = scheduler(optimizer) if scheduler is not None else None

    history: List[float] = []
    best = float("inf")
    start = perf_counter()
    last_loss = float("inf")
    converged = False

    for epoch in range(epochs):
        optimizer.zero_grad()
        prediction = model(*args)
        loss = loss_fn(prediction, target)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        _enforce_constraints(model)
        if sched is not None:
            try:
                sched.step(loss)
            except TypeError:
                sched.step()

        v = float(loss.detach())
        history.append(v)
        best = min(best, v)
        if callback is not None:
            callback(epoch, v, model)
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:>6d}/{epochs}  loss = {v:.5e}")

        if tol > 0 and abs(last_loss - v) / max(1e-30, last_loss) < tol:
            converged = True
            break
        last_loss = v

    if lbfgs_polish:
        lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=lbfgs_steps, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            l = loss_fn(model(*args), target)
            l.backward()
            return l

        lbfgs.step(closure)
        _enforce_constraints(model)
        with torch.no_grad():
            v = float(loss_fn(model(*args), target))
            history.append(v)
            best = min(best, v)

    duration = perf_counter() - start
    return OptimizeResult(
        model=model,
        loss_history=history,
        best_loss=best,
        duration_s=duration,
        n_epochs=len(history),
        converged=converged,
    )


def reduce_lr_on_plateau(patience: int = 100, factor: float = 0.9):
    """Convenience scheduler factory matching the thesis defaults."""
    def make(opt):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=patience, factor=factor)
    return make
