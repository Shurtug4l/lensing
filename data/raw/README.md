# Real-data FITS files

Small files used by the notebooks are tracked here directly. Large
files (>10 MB) are gitignored and must be obtained separately.

## Tracked

| File                       | Size  | Notebook | Source                                   |
|----------------------------|-------|----------|------------------------------------------|
| `TEST_F150W_NIRCAM.fits`   | 4.0 MB| 07       | JWST/NIRCam F150W postage stamp          |
| `galaxy_crop.fits`         | 1.3 MB| (legacy) | smaller crop of the same field           |
| `kappa_gl.fits`            | 1.0 MB| 09       | example convergence map                  |

## Not tracked (download on demand)

These are useful for the cluster-lensing notebook (09) but too large
to ship in-repo:

| File                       | Size  | How to obtain                            |
|----------------------------|-------|------------------------------------------|
| `macs1206_stack.fits`      | 72 MB | HST archive (CLASH or HFF programs)      |
| `RGB_macs1206.fits`        | 72 MB | RGB composite of the same cluster        |

The notebook **gracefully skips** the corresponding cell if the file is
missing, so the rest of the case study still runs.
