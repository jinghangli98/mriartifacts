"""Volume helpers shared by training and inference: orientation-to-plane
lookup, the robust intensity window / normalisation used at training time,
and isotropic resampling for the CLI's `--iso` option."""

import nibabel as nib
import nibabel.processing
import numpy as np

PLANE_OF_AXCODE = {"R": "sagittal", "L": "sagittal", "A": "coronal",
                   "P": "coronal", "S": "axial", "I": "axial"}
DEFAULT_AXIS_PLANES = ["sagittal", "coronal", "axial"]


def axis_planes(affine):
    """Plane of a slab thin along each array axis; RAS -> [sagittal, coronal, axial]."""
    return [PLANE_OF_AXCODE[c] for c in nib.aff2axcodes(np.asarray(affine))]


def volume_window(vol, percentiles=(0.5, 99.5), max_voxels=2_000_000):
    """`(lo, hi)` intensities that `normalize_volume` maps to 0 and 1,
    estimated on a fixed random subsample of at most `max_voxels` voxels."""
    vol = np.asarray(vol, dtype=np.float32)
    flat = vol.reshape(-1)
    if flat.size > max_voxels:
        idx = np.random.default_rng(0).integers(0, flat.size, size=max_voxels)
        flat = flat[idx]
    lo, hi = np.percentile(flat, percentiles)
    if hi <= lo:
        lo, hi = float(vol.min()), float(vol.max())
    return float(lo), float(hi)


def normalize_volume(vol, percentiles=(0.5, 99.5), clip=True, max_voxels=2_000_000):
    """Per-volume robust normalisation: `percentiles` -> [0, 1], clipped."""
    vol = np.asarray(vol, dtype=np.float32)
    lo, hi = volume_window(vol, percentiles, max_voxels)
    if hi <= lo:
        return np.zeros_like(vol)
    out = (vol - lo) / (hi - lo)
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


def _float_image(resampled):
    data = np.asarray(resampled.dataobj, dtype=np.float32)
    data[data < 0] = 0.0
    out = nib.Nifti1Image(data, resampled.affine)
    out.set_data_dtype(np.float32)
    return out


def resample_iso(img, voxel_mm):
    """`img` resampled to a `voxel_mm` isotropic grid (linear), as a float32 image."""
    return _float_image(nibabel.processing.resample_to_output(img, voxel_sizes=voxel_mm, order=1))


def resample_like(img, like):
    """`img` resampled (linear) onto the grid and affine of `like`, as a float32 image."""
    return _float_image(nibabel.processing.resample_from_to(img, (like.shape[:3], like.affine), order=1))
