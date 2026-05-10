"""Benchmarking utilities — CPU vs MPS / CUDA wall-clock timings.

Why it exists
-------------
The whole point of using PyTorch for lensing is that the same code runs on
CPU, GPU (CUDA) and Apple-silicon GPU (MPS) with no source changes. But
the *break-even* point — the problem size at which the GPU starts
beating the CPU — depends on the workload (small, kernel-launch-bound
problems tend to *lose* on the GPU). This module gives the package a
uniform way to quantify that trade-off.

Design
------
* :class:`Stopwatch` — minimal context manager that handles the device
  synchronisation correctly (``torch.cuda.synchronize`` /
  ``torch.mps.synchronize``) so the measured wall-clock includes all
  pending kernels. Using ``time.perf_counter`` without sync would
  over-estimate GPU speedups by missing in-flight work.

* :func:`time_callable` — runs a callable a number of times after a
  warm-up phase (so JIT / kernel-cache compilation costs don't pollute
  the measurement) and returns mean ± std.

* :func:`compare_devices` — runs the same workload on every available
  device and returns a :class:`pandas.DataFrame` with per-device wall
  time and the speedup vs. CPU baseline.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Callable, Iterable, List, Optional

import torch


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------
def available_devices() -> List[str]:
    """Return the list of devices we know how to benchmark on this host."""
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def synchronize(device: torch.device | str) -> None:
    """Block until all pending kernels on ``device`` have finished."""
    d = torch.device(device) if isinstance(device, str) else device
    if d.type == "cuda":
        torch.cuda.synchronize(d)
    elif d.type == "mps":
        # torch.mps.synchronize() exists since PyTorch 2.0; fall back gracefully.
        sync = getattr(torch.mps, "synchronize", None)
        if sync is not None:
            sync()
    # CPU is synchronous by definition; nothing to do.


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------
class Stopwatch(contextlib.AbstractContextManager):
    """Context-manager wall-clock timer with device synchronization.

    Example
    -------
    >>> with Stopwatch(device="mps") as sw:
    ...     y = model(x)
    >>> sw.elapsed   # seconds
    """

    def __init__(self, device: str | torch.device = "cpu"):
        self.device = device
        self.elapsed: float = float("nan")
        self._start: Optional[float] = None

    def __enter__(self) -> "Stopwatch":
        synchronize(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        synchronize(self.device)
        assert self._start is not None
        self.elapsed = time.perf_counter() - self._start


@dataclass
class TimingResult:
    """Aggregated timing information for one benchmark run."""

    device: str
    samples_s: List[float] = field(default_factory=list)
    n_warmup: int = 0
    n_repeats: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return mean(self.samples_s) if self.samples_s else float("nan")

    @property
    def std(self) -> float:
        return stdev(self.samples_s) if len(self.samples_s) > 1 else 0.0

    def summary(self) -> str:
        return f"{self.device:<5s}: {1e3*self.mean:8.2f} ms ± {1e3*self.std:6.2f} ms"


def time_callable(
    fn: Callable[[], None],
    *,
    device: str | torch.device = "cpu",
    n_warmup: int = 2,
    n_repeats: int = 5,
    extra: Optional[dict] = None,
) -> TimingResult:
    """Run ``fn`` ``n_repeats`` times after ``n_warmup`` warm-up calls.

    The warm-up serves three purposes:
    1. on first GPU launch the runtime compiles kernels and allocates
       memory pools - this is a one-time cost that should not be charged
       to the user-visible timing;
    2. on MPS the first kernel of every shape may trigger a recompile
       (Apple Metal compiles ahead of time);
    3. on CPU the first call may need to populate caches.
    """
    for _ in range(n_warmup):
        fn()
        synchronize(device)

    samples: List[float] = []
    for _ in range(n_repeats):
        with Stopwatch(device=device) as sw:
            fn()
        samples.append(sw.elapsed)
    return TimingResult(
        device=str(device), samples_s=samples,
        n_warmup=n_warmup, n_repeats=n_repeats,
        extra=dict(extra or {}),
    )


# ---------------------------------------------------------------------------
# Cross-device comparator
# ---------------------------------------------------------------------------
def compare_devices(
    factory: Callable[[torch.device], Callable[[], None]],
    *,
    devices: Optional[Iterable[str]] = None,
    n_warmup: int = 2,
    n_repeats: int = 5,
):
    """Run the workload on every available device and return a DataFrame.

    ``factory(device)`` must return a zero-arg callable that performs the
    workload on ``device``. We isolate construction from execution so that
    each device can hold its own copy of any tensors / models.

    The returned DataFrame has columns:

    ====================  =========================================
    column                meaning
    ====================  =========================================
    ``device``            device name
    ``mean_ms``           mean wall time per call (milliseconds)
    ``std_ms``            sample standard deviation
    ``speedup_vs_cpu``    cpu_mean / device_mean (1 means "same as CPU")
    ====================  =========================================
    """
    import pandas as pd

    devs = list(devices) if devices is not None else available_devices()
    rows = []
    for d in devs:
        fn = factory(torch.device(d))
        res = time_callable(fn, device=d, n_warmup=n_warmup, n_repeats=n_repeats)
        rows.append({
            "device": d,
            "mean_ms": 1e3 * res.mean,
            "std_ms": 1e3 * res.std,
            "n_warmup": n_warmup,
            "n_repeats": n_repeats,
        })

    df = pd.DataFrame(rows).set_index("device")
    cpu_mean = df.loc["cpu", "mean_ms"] if "cpu" in df.index else df["mean_ms"].max()
    df["speedup_vs_cpu"] = cpu_mean / df["mean_ms"]
    return df


# ---------------------------------------------------------------------------
# Pre-built workloads
# ---------------------------------------------------------------------------
def sersic_forward_workload(npix: int, batch: int = 1):
    """Factory that builds a Sersic-on-grid forward-pass workload."""
    from .data.grid import coordinate_grid
    from .light.sersic import Sersic

    def factory(device: torch.device):
        xy = coordinate_grid(npix=npix, deltapix=0.05).to(device)
        # Repeat the same model `batch` times to amortise launch overhead.
        models = [
            Sersic(Ie=5., Re=1., n=4., x0=0., y0=0., e1=0.1, e2=-0.05).to(device)
            for _ in range(batch)
        ]

        def run():
            for m in models:
                _ = m(xy)
        return run

    return factory


def sersic_backward_workload(npix: int):
    """Forward + backward through a Sersic + dummy loss."""
    from .data.grid import coordinate_grid
    from .light.sersic import Sersic

    def factory(device: torch.device):
        xy = coordinate_grid(npix=npix, deltapix=0.05).to(device)
        target = torch.zeros((npix, npix), device=device)

        def run():
            m = Sersic(Ie=5., Re=1., n=4., x0=0., y0=0., e1=0.1, e2=-0.05).to(device)
            pred = m(xy)
            loss = ((pred - target) ** 2).sum()
            loss.backward()
        return run

    return factory
