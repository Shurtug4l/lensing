"""Smoke tests for the big-data, archive and LLM modules."""
from __future__ import annotations

import os
from pathlib import Path

import torch

import lensing as gl


def test_hdf5_roundtrip(tmp_path):
    path = tmp_path / "tiny.h5"
    gl.bigdata.generate_lens_dataset(path, n_samples=20, npix=24, seed=0, progress=False)
    assert path.exists()
    ds = gl.bigdata.HDF5Dataset(path, target='label')
    assert len(ds) == 20
    img, lab = ds[0]
    assert img.shape == (1, 24, 24)
    assert isinstance(int(lab), int)


def test_hdf5_targets(tmp_path):
    path = tmp_path / "tiny.h5"
    gl.bigdata.generate_lens_dataset(path, n_samples=4, npix=16, seed=1, progress=False)
    for tgt in ['label', 'params', 'source', 'lens']:
        ds = gl.bigdata.HDF5Dataset(path, target=tgt)
        img, y = ds[0]
        if tgt == 'label':
            assert y.shape == ()
        elif tgt == 'params':
            assert y.shape == (7,)
        elif tgt == 'lens':
            assert y.shape == (5,)
        elif tgt == 'source':
            assert y.shape == (1, 16, 16)


def test_slacs_catalog_has_8_systems():
    df = gl.archive.slacs_table()
    assert len(df) >= 5
    assert {'name', 'ra', 'dec', 'theta_E', 'sigma_v', 'z_L', 'z_S'} <= set(df.columns)


def test_llm_mock_extracts_known_fields():
    abstract = (
        "We report SDSSJ0029-0055, a strong lens with theta_E = 0.96 arcsec, "
        "sigma_v = 229 km/s, lens redshift z_L = 0.227 and source z_S = 0.931."
    )
    rec = gl.llm.extract_lens_metadata(abstract, backend='mock')
    assert rec.name == 'SDSSJ0029-0055'
    assert abs(rec.theta_E_arcsec - 0.96) < 1e-3
    assert abs(rec.sigma_v_kms - 229.0) < 1e-3
    assert abs(rec.z_L - 0.227) < 1e-3
    assert abs(rec.z_S - 0.931) < 1e-3


def test_llm_mock_handles_missing_fields():
    """Empty abstract -> all-null record (no crash, no hallucination)."""
    rec = gl.llm.extract_lens_metadata("", backend='mock')
    assert rec.name is None
    assert rec.theta_E_arcsec is None
    assert rec.sigma_v_kms is None
