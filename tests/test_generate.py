import nibabel as nib
import numpy as np
import pytest
import torch

from mriartifacts import generate as g
from mriartifacts.model import load_for_inference


@pytest.fixture(scope="module")
def tiny_model(tiny_ckpt):
    fm, cc, _ = load_for_inference(tiny_ckpt, torch.device("cpu"))
    return fm, cc


def _ras_img(shape=(20, 18, 12), seed=0, scale=500.0):
    data = np.random.default_rng(seed).random(shape).astype(np.float32) * scale
    return nib.Nifti1Image(data, np.eye(4))


def test_generate_volume_shape_range_and_window(tiny_model):
    fm, cc = tiny_model
    img = _ras_img()
    out, (lo, hi) = g.generate_volume(img, fm, cc, "motion", 3.0, 1, "axial", 0, "cpu")
    assert out.shape == img.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0 and np.isfinite(out).all()
    assert lo < hi


def test_generate_volume_is_seeded(tiny_model):
    fm, cc = tiny_model
    img = _ras_img()
    a, _ = g.generate_volume(img, fm, cc, "spike", 1.5, 1, "axial", 3, "cpu")
    b, _ = g.generate_volume(img, fm, cc, "spike", 1.5, 1, "axial", 3, "cpu")
    c, _ = g.generate_volume(img, fm, cc, "spike", 1.5, 1, "axial", 4, "cpu")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_generate_volume_thin_axis(tiny_model):
    """Review focus 5: fewer than 7 slices along the slab axis is padded, not fatal."""
    fm, cc = tiny_model
    img = _ras_img(shape=(16, 16, 4))
    out, _ = g.generate_volume(img, fm, cc, "lowres", 3.0, 1, "axial", 0, "cpu")
    assert out.shape == (16, 16, 4) and np.isfinite(out).all()
    out_s, _ = g.generate_volume(_ras_img(shape=(5, 16, 16)), fm, cc, "lowres", 3.0, 1, "sagittal", 0, "cpu")
    assert out_s.shape == (5, 16, 16) and np.isfinite(out_s).all()


def test_generate_volume_constant_input_is_finite(tiny_model):
    """Review focus 5: an all-zero volume has an empty window; output must still be finite."""
    fm, cc = tiny_model
    img = nib.Nifti1Image(np.zeros((16, 16, 8), np.float32), np.eye(4))
    out, (lo, hi) = g.generate_volume(img, fm, cc, "motion", 3.0, 1, "axial", 0, "cpu")
    assert out.shape == (16, 16, 8) and np.isfinite(out).all() and lo == hi == 0.0


def test_generate_volume_rejects_bad_plane_and_family(tiny_model):
    fm, cc = tiny_model
    with pytest.raises(ValueError, match="plane"):
        g.generate_volume(_ras_img(), fm, cc, "motion", 3.0, 1, "oblique", 0, "cpu")
    with pytest.raises(ValueError, match="family"):
        g.generate_volume(_ras_img(), fm, cc, "blur", 3.0, 1, "axial", 0, "cpu")

