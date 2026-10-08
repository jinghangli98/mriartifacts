"""Whole-volume slab inference (adapted from autoflow enhance_flow_3d.py).

A `(X, Y, Z)` volume is permuted so the requested plane's slab axis is last,
X/Y are zero-padded to a multiple of PAD_MULT, then PATCH_Z-thick slabs are
generated with stride `PATCH_Z - non_overlap` and blended with a linear
overlap-and-add window. The `sampler(condition, noise)` callable receives
`(B, 1, X, Y, PATCH_Z)` tensors and returns the generated slabs.
"""

import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .volume import DEFAULT_AXIS_PLANES

PATCH_Z = 7
PAD_MULT = 16
# Permutations moving array axis `a` last (and back), as the training loader's
# np.moveaxis(crop, axis, -1).
AXIS_PERMS = {2: ((0, 1, 2), (0, 1, 2)), 1: ((0, 2, 1), (0, 2, 1)), 0: ((1, 2, 0), (2, 0, 1))}


def to_ras(img):
    orig_ornt = nib.orientations.io_orientation(img.affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    to_orig_ornt = nib.orientations.ornt_transform(ras_ornt, orig_ornt)
    return nib.as_closest_canonical(img), to_orig_ornt


def make_blend_window(patch_z, non_overlap, dtype=torch.float32):
    w = torch.ones(patch_z, dtype=dtype)
    if non_overlap > 0:
        ramp = torch.arange(1, non_overlap + 1, dtype=dtype) / (non_overlap + 1)
        w[:non_overlap] = ramp
        w[-non_overlap:] = ramp.flip(0)
    return w


def pad_to_multiple(volume, mult, min_z=0, z_mult=None):
    def _amount(n, m, min_n=0):
        target = max(n, min_n)
        rem = target % m
        if rem:
            target += m - rem
        total = target - n
        return total // 2, total - total // 2

    z_mult = mult if z_mult is None else z_mult
    x_l, x_r = _amount(volume.shape[0], mult)
    y_l, y_r = _amount(volume.shape[1], mult)
    z_l, z_r = _amount(volume.shape[2], z_mult, min_z)
    padded = F.pad(volume, (z_l, z_r, y_l, y_r, x_l, x_r), mode="constant", value=0.0)
    return padded, ((x_l, x_r), (y_l, y_r), (z_l, z_r)), tuple(volume.shape)


def unpad(volume, pad_info, original_shape):
    (x_l, _), (y_l, _), (z_l, _) = pad_info
    ox, oy, oz = original_shape
    return volume[x_l:x_l + ox, y_l:y_l + oy, z_l:z_l + oz]


def slab_starts(z_dim, patch_z, stride):
    starts, z = [], 0
    while z + patch_z <= z_dim:
        starts.append(z)
        z += stride
    if not starts or starts[-1] + patch_z < z_dim:
        starts.append(z_dim - patch_z)
    return starts


def slab_inference(padded, sampler, non_overlap, seed, batch_size, device, desc="slabs"):
    """Slide PATCH_Z-thick slabs along the last axis of `padded` (X, Y, Z),
    generate each with `sampler`, blend with overlap-and-add. Each slab draws
    its own noise, seeded by (`seed`, slab start): the noise is the artifact
    realisation, and one shared tensor would stack the same ghost pattern
    through the whole volume (streaks in the sagittal/coronal views). Results
    are fixed by `seed` and independent of `batch_size`."""
    stride = PATCH_Z - max(0, non_overlap)
    starts = slab_starts(padded.shape[2], PATCH_Z, stride)
    window = make_blend_window(PATCH_Z, max(0, non_overlap), dtype=padded.dtype).view(1, 1, -1)
    out_sum = torch.zeros_like(padded)
    out_w = torch.zeros_like(padded)
    noise_shape = (1, 1, padded.shape[0], padded.shape[1], PATCH_Z)

    def slab_noise(z):
        gen = torch.Generator(device="cpu").manual_seed(seed * 100_003 + z)
        return torch.randn(noise_shape, generator=gen)

    for b0 in tqdm(range(0, len(starts), max(1, batch_size)), desc=desc, ncols=80):
        batch = starts[b0:b0 + max(1, batch_size)]
        cond = torch.stack([padded[:, :, z:z + PATCH_Z] for z in batch]).unsqueeze(1).to(device)
        noise = torch.cat([slab_noise(z) for z in batch]).to(device)
        out = sampler(cond, noise)[:, 0].float().cpu()
        for i, z in enumerate(batch):
            out_sum[:, :, z:z + PATCH_Z] += out[i] * window
            out_w[:, :, z:z + PATCH_Z] += window
    return out_sum / out_w.clamp(min=1e-8), len(starts)


def run_plane_inference(vol_t, plane, sampler, non_overlap, seed, batch_size, device,
                        axis_planes=None):
    axis = list(axis_planes or DEFAULT_AXIS_PLANES).index(plane)
    perm, inv_perm = AXIS_PERMS[axis]
    permuted = vol_t.permute(*perm).contiguous()
    padded, pad_info, original_shape = pad_to_multiple(permuted, PAD_MULT, min_z=PATCH_Z, z_mult=1)
    out_padded, n = slab_inference(padded, sampler, non_overlap, seed, batch_size, device,
                                   desc=f"{plane} slabs")
    return unpad(out_padded, pad_info, original_shape).permute(*inv_perm).contiguous()


def save_volume(arr, path, to_orig_ornt, rescale, ref_img, dtype=np.float32):
    """Reorient back to the input's storage order, optionally map [0, 1] back
    through the `(lo, hi)` percentile window, and save with the input's
    affine/header."""
    arr = arr.numpy() if torch.is_tensor(arr) else np.asarray(arr)
    if to_orig_ornt is not None:
        arr = nib.orientations.apply_orientation(arr, to_orig_ornt)
    arr = np.asarray(arr, dtype=np.float32)
    if rescale:
        lo, hi = rescale
        arr = arr * (hi - lo) + lo
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out = nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header)
    out.header.set_slope_inter(1.0, 0.0)   # a scaled integer input header must not rescale the float output
    out.set_data_dtype(dtype)
    out.to_filename(path)
