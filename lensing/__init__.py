"""lensing: PyTorch-based gravitational lensing toolkit.

A clean reorganization of the code developed for the Master's thesis
"Applications of Automatic Differentiation in Gravitational Lensing"
(Simone La Porta, Università di Bologna, 2024), expanded with material
and case studies inspired by the *Lensing Gravitazionale* graduate course
(Prof. M. Meneghetti, UNIBO).

The package is organized into:

* ``lensing.light``      - parametric surface brightness profiles
                           (Sérsic, core-Sérsic, multi-component)
* ``lensing.lens``       - lens models (point-mass / Paczynski microlens,
                           binary microlens, SIE, NIE, NFW, power-law /
                           SIS, external shear, composite lenses) and
                           time-delay utilities
* ``lensing.data``       - synthetic image / light-curve generators,
                           noise models, coordinate grids
* ``lensing.inference``  - losses, Adam + L-BFGS training, NUTS
                           posteriors, weak-lensing ellipticity
* ``lensing.ml``         - PyTorch datasets and CNN/DNN/U-Net models
                           for deep-learning case studies
* ``lensing.benchmarks`` - CPU vs MPS / CUDA benchmark utilities
* ``lensing.viz``        - plotting helpers (images, posteriors)
* ``lensing.utils``      - parameter transforms (e1/e2 <-> q/PA, b_n)
"""

from . import (
    archive,
    benchmarks,
    bigdata,
    config,
    cosmology,
    data,
    inference,
    lens,
    light,
    llm,
    ml,
    utils,
    viz,
)

__all__ = [
    "archive",
    "benchmarks",
    "bigdata",
    "config",
    "cosmology",
    "data",
    "inference",
    "lens",
    "light",
    "llm",
    "ml",
    "utils",
    "viz",
]

__version__ = "0.2.0"
