"""Big-data utilities: HDF5-backed datasets that scale to 10⁴–10⁶ samples.

Why a separate module rather than ``lensing.ml.datasets``?
The existing :class:`lensing.ml.datasets.SersicParamDataset` regenerates
images on the fly from a seed — perfect for unit tests and small toy
experiments where you trust that the simulator is the ground truth.
For survey-scale studies (10k–10⁵ images) two issues arise:

1. **Reproducibility**: re-running with PyTorch upgrades or different
   floating-point libraries can shift pixel values; a paper-quality
   experiment wants the *exact same bytes* every time. Persist to disk.
2. **Throughput**: GPU training spends most of its time waiting for the
   simulator, even if the latter is parallelised. Caching to HDF5 cuts
   training time by 5–10× on our M1 box.

The module exposes:

* :class:`HDF5Generator` — writes a dataset to disk, with a Sérsic+SIE
  pipeline that mirrors :mod:`lensing.ml.datasets` but supports parallel
  workers and chunked writes;
* :class:`HDF5Dataset` — lazy ``torch.utils.data.Dataset`` that maps the
  HDF5 file with the right access pattern (each worker keeps its own
  file handle to avoid cross-process locking).
"""
from .generator import HDF5Generator, generate_lens_dataset
from .dataset import HDF5Dataset

__all__ = ["HDF5Generator", "generate_lens_dataset", "HDF5Dataset"]
