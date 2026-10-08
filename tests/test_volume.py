import nibabel as nib
import numpy as np
import pytest

from mriartifacts import volume as mv
from mriartifacts.slab import save_volume, to_ras


def _aniso_img(shape=(20, 24, 10), vox=(1.0, 1.2, 2.5)):
    rng = np.random.default_rng(0)
    data = rng.random(shape).astype(np.float32) * 1000
    aff = np.diag([vox[0], vox[1], vox[2], 1.0])
    return nib.Nifti1Image(data, aff)


def test_resample_iso_grid_and_dtype():
    img = _aniso_img()
    out = mv.resample_iso(img, 1.0)
    assert np.allclose(np.abs(np.diag(out.affine)[:3]), 1.0)
    assert out.get_data_dtype() == np.float32
    assert out.shape[2] >= 22                     # 10 slices at 2.5 mm span 22.5 mm -> >= 23 at 1 mm
    assert float(np.asarray(out.dataobj).min()) >= 0.0


def test_resample_like_returns_to_input_grid():
    img = _aniso_img()
    iso = mv.resample_iso(img, 1.0)
    back = mv.resample_like(iso, img)
    assert back.shape == img.shape
    assert np.allclose(back.affine, img.affine)
    # smooth data round-trips closely: compare a constant volume exactly
    const = nib.Nifti1Image(np.full((20, 24, 10), 7.0, np.float32), img.affine)
    back_c = mv.resample_like(mv.resample_iso(const, 1.0), const)
    inner = np.asarray(back_c.dataobj)[2:-2, 2:-2, 1:-1]
    assert np.allclose(inner, 7.0, atol=1e-3)


def test_volume_window_and_normalize_agree():
    v = np.random.default_rng(1).random((8, 8, 8)).astype(np.float32) * 50
    lo, hi = mv.volume_window(v)
    n = mv.normalize_volume(v)
    assert n.min() >= 0 and n.max() <= 1
    assert np.allclose(n * (hi - lo) + lo, np.clip(v, lo, hi), atol=1e-3)


def test_axis_planes_ras_and_lps():
    assert mv.axis_planes(np.eye(4)) == ["sagittal", "coronal", "axial"]
    lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    assert mv.axis_planes(lps) == ["sagittal", "coronal", "axial"]
    swapped = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
    assert mv.axis_planes(swapped) == ["coronal", "axial", "sagittal"]


def test_save_volume_with_scaled_int16_header_reads_back_unscaled(tmp_path):
    """Review focus 2: an int16 input with scl_slope/inter must not leak its
    scaling into the float32 output."""
    ref = nib.Nifti1Image(np.ones((6, 6, 6), np.int16), np.eye(4))
    ref.header.set_slope_inter(2.0, 10.0)        # in-memory header, as nib.load returns for a scaled file
    assert ref.header.get_slope_inter() == (2.0, 10.0)
    arr = np.linspace(0, 1, 216, dtype=np.float32).reshape(6, 6, 6)
    out_path = str(tmp_path / "out.nii.gz")
    save_volume(arr, out_path, None, (100.0, 300.0), ref)
    back = nib.load(out_path)
    assert back.get_data_dtype() == np.float32
    assert np.allclose(back.get_fdata(), arr * 200.0 + 100.0, atol=1e-2)


def test_to_ras_roundtrip_orientation(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    lps = nib.Nifti1Image(data, np.diag([-1.0, -1.0, 1.0, 1.0]))
    ras, to_orig = to_ras(lps)
    assert nib.aff2axcodes(ras.affine) == ("R", "A", "S")
    restored = nib.orientations.apply_orientation(np.asarray(ras.dataobj), to_orig)
    assert np.array_equal(restored, data)
