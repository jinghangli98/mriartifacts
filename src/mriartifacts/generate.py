"""Clean RAS volume -> artifact-corrupted volume for one artifact family.

The volume is normalised like training (0.5/99.5 percentiles -> [0, 1]) and
generated in 7-slice slabs along one plane with classifier-free guidance
`v = v_null + strength * (v_family - v_null)`; each slab draws its own noise
from `seed`, so different seeds give different artifact realisations."""

import numpy as np
import torch

from .slab import run_plane_inference
from .volume import axis_planes, normalize_volume, volume_window

PLANES = ("axial", "sagittal", "coronal")


def make_sampler(flow_matcher, labels, null_labels, guidance_scale, num_steps, device):
    """`labels` / `null_labels` are `(1,)` class-label tensors (one family for
    the whole volume), broadcast to each slab batch."""
    def sampler(condition, noise):
        b = condition.shape[0]
        x, _ = flow_matcher.sample(condition, num_steps, device,
                                   context=labels.expand(b, *labels.shape[1:]),
                                   null_context=null_labels.expand(b, *null_labels.shape[1:]),
                                   guidance_scale=guidance_scale, noise=noise)
        return x
    return sampler


def generate_volume(img_ras, flow_matcher, conditioner, family, strength, steps, plane, seed, device,
                    non_overlap=3, batch_size=4):
    """`(volume in [0, 1] as (X, Y, Z) float32, (lo, hi) window)` for one artifact
    `family` on the RAS image `img_ras`. `strength` is the guidance scale,
    `plane` the slab orientation (one of PLANES), `seed` fixes the noise."""
    if plane not in PLANES:
        raise ValueError(f"unknown plane {plane!r}; expected one of {PLANES}")
    data = np.asanyarray(img_ras.dataobj).astype(np.float32)
    window = volume_window(data)
    vol_t = torch.from_numpy(normalize_volume(data))
    ctx = conditioner.encode([family], device)          # raises ValueError for an unknown family
    null = conditioner.null(1, device)
    sampler = make_sampler(flow_matcher, ctx, null, strength, steps, device)
    out = run_plane_inference(vol_t, plane, sampler, non_overlap, seed, batch_size, device,
                              axis_planes=axis_planes(img_ras.affine))
    return out.clamp(0.0, 1.0).numpy().astype(np.float32), window
