---
license: mit
tags:
- medical-imaging
- mri
- data-augmentation
- flow-matching
---

# mriartifacts-flow

Class-conditioned 3D flow-matching model that turns a clean brain MRI volume into an
artifact-corrupted copy. Used by the `mriartifacts` Python package (`pip install mriartifacts`,
command `artifact-gen`).

## Validated artifacts

* `motion`: head-motion artifacts learned from the MR-ART dataset (3T T1w MPRAGE, real
  motion-corrupted rescans). Transfers zero-shot to other contrasts and field strengths
  (7T MPRAGE, 7T T2w SPACE, 7T FLAIR, 3T MPRAGE).
* `spike`: k-space spike artifacts, learned from raw-k-space simulations.
* `lowres`: low-resolution blur from k-space truncation, learned from raw-k-space simulations.

The model was trained with five degradation classes; parallel-imaging (GRAPPA) and
compressed-sensing undersampling are in the class table but are not validated and are
not exposed by the CLI.

## Model

MONAI `DiffusionModelUNet`, channels 128-256-512, 165 M parameters, input = (noisy sample,
clean volume), output = velocity; artifact class added to the timestep embedding;
classifier-free guidance with a null class. Inference: 7-slice axial slabs, 20 Euler steps,
guidance scale 1.5-5. Weights: `motion_cfg_classemb128_100526_best.pt`, a PyTorch state-dict
checkpoint (fp32, EMA weights at step 62,500; keys `model`, `channels`, `families`,
`epoch`, `global_step`), loaded with `torch.load(..., weights_only=True)`.

## Usage

    pip install mriartifacts
    artifact-gen clean_T1w.nii.gz --artifact motion --strength 3

## Licence and citation

Licence: MIT (code and weights). Citation: IEEE ISBI 2027 submission, under review (to be added).
