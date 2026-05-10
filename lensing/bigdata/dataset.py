"""HDF5-backed PyTorch Dataset.

Implementation note
-------------------
The cleanest way to share an HDF5 file across DataLoader workers is to
let *each* worker open its own handle on first access. Sharing one
``h5py.File`` between fork()ed processes is a well-known deadlock
source (h5py / HDF5 use thread-local locks that confuse fork). We
therefore lazily open the file in ``__getitem__``, store the handle on
``self``, and let each worker keep its own.

The ``mode='r'`` plus ``swmr=False`` setting is the default; concurrent
*writers* would need ``swmr=True`` and a single-writer many-reader
pattern, which we do not need here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """Lazy ``Dataset`` reading a file produced by :class:`HDF5Generator`.

    Parameters
    ----------
    path : path to the HDF5 file
    target : which target to return alongside the image:
        ``'label'`` -> int (lens/no-lens, for the CNN classifier);
        ``'params'`` -> 7-vector Sérsic params (for the regressor);
        ``'source'`` -> source-plane image (for the U-Net);
        ``'lens'`` -> 5-vector SIE params.
    indices : optional subset of indices to expose (for train/val splits)
    """

    SUPPORTED = {"label", "params", "source", "lens"}

    def __init__(
        self,
        path: str | Path,
        *,
        target: str = "label",
        indices: Optional[Sequence[int]] = None,
    ):
        if target not in self.SUPPORTED:
            raise ValueError(f"target must be one of {self.SUPPORTED}, got {target!r}")
        self.path = str(path)
        self.target = target
        # Inspect the file once to know how many samples it contains.
        import h5py
        with h5py.File(self.path, "r") as f:
            n = int(f["images"].shape[0])
            self._meta = dict(f.attrs)
        self._all_indices = list(range(n)) if indices is None else list(indices)
        self._handle = None  # opened lazily, per-worker

    def __len__(self) -> int:
        return len(self._all_indices)

    @property
    def meta(self) -> dict:
        """Generator metadata (sigma, psf_fwhm, deltapix, ...)."""
        return dict(self._meta)

    def _open(self):
        if self._handle is None:
            import h5py
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, idx: int):
        i = self._all_indices[idx]
        f = self._open()
        img = torch.from_numpy(f["images"][i])  # (1, H, W)
        if self.target == "label":
            tgt = torch.tensor(int(f["label"][i]), dtype=torch.long)
        elif self.target == "params":
            tgt = torch.from_numpy(f["params"][i])
        elif self.target == "source":
            tgt = torch.from_numpy(f["sources"][i])
        else:  # 'lens'
            tgt = torch.from_numpy(f["lens"][i])
        return img, tgt

    def __getstate__(self):
        # Drop the un-picklable HDF5 handle when DataLoader forks workers.
        state = self.__dict__.copy()
        state["_handle"] = None
        return state
