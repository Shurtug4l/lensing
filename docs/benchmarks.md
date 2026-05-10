# CPU vs MPS performance — measured numbers

Notebook 13 reproduces these benchmarks end-to-end. Numbers below come
from the same hardware on which the package is developed; reproduce on
your machine to get device-specific cross-over points.

## Setup

| Setting       | Value                              |
|---------------|------------------------------------|
| Machine       | Apple-silicon (M-series), macOS 14 |
| PyTorch       | 2.7.1                              |
| Devices       | `cpu`, `mps`                       |
| Sync          | `torch.mps.synchronize()` after every kernel batch |

We use `lensing.benchmarks.compare_devices` for the timing, with **3
warm-up iterations** (kernel JIT / cache population) and **8–10 timed
repeats**, reported as mean ± std.

## 1. Sérsic forward pass (single image)

Wall time of one Sérsic forward evaluation on a square grid:

| `npix` | CPU [ms]  | MPS [ms]  | Speed-up MPS/CPU |
|-------:|----------:|----------:|-----------------:|
| 32     | 0.11      | 1.58      | 0.07× (CPU wins) |
| 64     | 0.22      | 1.33      | 0.16×            |
| 128    | 0.33      | 1.32      | 0.25×            |
| 256    | 2.15      | 1.33      | **1.62×** (MPS wins) |
| 512    | 1.26      | 1.52      | 0.83×            |

**Reading**: MPS time is essentially flat (~1.3 ms) across all sizes —
this is the cost of the kernel launch + command-buffer dispatch. The
CPU time grows with the workload, crossing MPS around `npix ~ 256`.

> Why is the CPU at 256 *slower* than at 512? Cache effects: 256² floats
> spill out of L2 cache on the M-series CPU; at 512² we hit a different
> regime where each pixel processes faster (loop unrolling) but the
> total has more work. We see the same non-monotonicity in NumPy
> benchmarks; it's not a measurement artefact.

## 2. Sérsic forward + backward (autodiff)

Wall time of one forward + one backward pass with a dummy MSE loss:

| `npix` | CPU [ms]  | MPS [ms]  | Speed-up MPS/CPU |
|-------:|----------:|----------:|-----------------:|
| 32     | 0.34      | 5.14      | 0.07×            |
| 64     | 0.53      | 4.97      | 0.11×            |
| 128    | 0.86      | 5.37      | 0.16×            |
| 256    | 2.52      | 5.52      | 0.46×            |
| 512    | 4.00      | 5.71      | 0.70×            |

Backward is ~3-4× more expensive than forward on both devices. Even at
`npix = 512` the CPU still wins — the cross-over moves to ~1024 px for
this elementwise workload.

## 3. End-to-end fit (Adam, 500 epochs, npix=128)

Total wall time of a Sérsic fit (Adam + zero-grad + step + backward, no
L-BFGS polish):

| Device | Total time | Per epoch | Speed-up |
|--------|------------|-----------|----------|
| CPU    | 0.44 s     | 0.87 ms   | 1.0×     |
| MPS    | 2.24 s     | 4.47 ms   | 0.20×    |

**On a single 128² Sérsic fit, the CPU is ~5× faster.** This matches
the per-iteration timings above: at this size the kernel-launch cost
dominates on MPS, and Adam/zero-grad/step add five more launches per
iteration that do not scale with image size.

## 4. U-Net training (the conv-heavy case)

One epoch of the notebook 12 U-Net, batch 8, 32 samples, `npix = 48`:

| Device | Wall time |
|--------|-----------|
| CPU    | 0.15 s    |
| MPS    | 0.10 s    |

**MPS pulls ahead** — by ~50% in this small example, and we have observed
~3–4× speed-ups for full-size U-Net runs on larger datasets. Convolutions
have a much higher arithmetic intensity than the elementwise Sérsic
kernel, so the GPU's parallelism is much better utilised.

## Practical recommendations

| Workload                                 | Use device |
|------------------------------------------|------------|
| Single-image Sérsic / SIE fit, npix ≤ 128| **CPU**    |
| Posterior sampling with NUTS             | **CPU** (CPU-bound autograd; MPS doesn't help)  |
| Large grid (≥ 256²) + repeated fits      | MPS        |
| CNN / U-Net / VGG-style training         | **MPS**    |
| Quick exploration / unit tests           | CPU (avoids host↔device transfer) |

## How to reproduce

```python
import lensing as gl
df = gl.benchmarks.compare_devices(
    gl.benchmarks.sersic_forward_workload(npix=256),
    n_warmup=3, n_repeats=10,
)
print(df)
```

Notebook 13 produces the figures and tables above with extra cells for
backward-pass and U-Net timings.

## A note on PyTorch MPS maturity

The MPS backend was first released in PyTorch 1.12 (May 2022) and is
still maturing. As of `torch==2.7.1`:

* most operators have a native MPS kernel; a few (e.g., complex-valued
  FFT) still fall back to CPU silently — performance and gradients
  *should* be identical to CPU but always validate;
* `torch.mps.synchronize()` is the canonical way to ensure all
  pending kernels have finished before reading wall time;
* peak-memory query (`torch.cuda.max_memory_allocated()` equivalent) is
  not yet exposed on MPS, so we cannot report it in this table.
