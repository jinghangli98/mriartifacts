# mriartifacts

Generate realistic MRI artifacts on clean brain volumes.

## Install

```
pip install mriartifacts
```
Or
```bash
git clone https://github.com/jinghangli98/mriartifacts.git
cd mriartifacts 
pip install -e .
```

Two models are available with `--model`: `small` (default, 64-192-320 channels, 68 M
parameters) and `large` (128-256-512 channels, 165 M parameters; a 1 mm whole-brain volume
takes about 2.5 minutes on an A100).


Weights (660 MB) are downloaded from
[huggingface.co/jil202/mriartifacts](https://huggingface.co/jil202/mriartifacts)

## Usage

```bash
artifact-gen clean_T1w.nii.gz --artifact motion            # -> clean_T1w_motion.nii.gz
artifact-gen in.nii.gz out.nii.gz --artifact spike --strength 5 --seed 42
artifact-gen in.nii.gz --artifact lowres --iso 1
artifact-gen -i in.nii.gz -o out.nii.gz --artifact grappa     # -i/-o instead of positionals
```

| Option | Default | Meaning |
| --- | --- | --- |
| `INPUT` / `-i` | required | Clean 3D NIfTI. |
| `OUTPUT` / `-o` | `<input stem>_<artifact>.nii.gz` | Output path; existing files need `--overwrite`. |
| `--artifact` | required | `motion`, `spike`, `grappa` or `lowres`. `grappa` was not validated in the paper. |
| `--strength` (alias `--guidance-scale`) | 3 | Classifier-free-guidance scale: 1 = none, 1.5 mild, 3 moderate, 5 strong. |
| `--steps` | 20 | Euler sampling steps. |
| `--plane` | `axial` | Slab orientation: `axial`, `sagittal`, `coronal`. |
| `--overlap N` | 1 | Slices shared by neighbouring 7-slice slabs and linearly blended (0 = none, up to 6). |
| `--iso MM` | off | Run at an `MM`-mm isotropic grid, resample the output back to the input grid. |
| `--seed` | random | Noise seed; the seed used is printed, so a result can be reproduced. |
| `--model` | `small` | Hub model to use: `small` (68 M params) or `large` (165 M params, slower). Ignored with `--checkpoint`. |
| `--checkpoint PATH` | Hub download | Local weights (`.pt` or `.safetensors`); or set `MRIARTIFACTS_CHECKPOINT`. |
| `--device` | `auto` | `auto`, `cuda`, `cuda:N`, `cpu`. |
| `--fp32` | off | Disable bf16 autocast on CUDA. |
| `--no-compile` | off | Run the UNet eagerly. By default the UNet is wrapped in `torch.compile` on CUDA; the first slab batch pays a one-off compile cost. |

The output keeps the input's grid, orientation, affine and intensity units. Different
seeds give different artifact realisations of the same volume.


## Citation

Paper brewing......

## Licence

MIT. See `LICENSE`. Weights: see the model card on the Hub.
