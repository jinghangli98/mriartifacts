"""`artifact-gen`: clean brain-MRI NIfTI -> artifact-corrupted copy.

    artifact-gen INPUT.nii.gz [OUTPUT.nii.gz] --artifact {motion,spike,grappa,lowres} [--strength 3]
    artifact-gen -i INPUT.nii.gz -o OUTPUT.nii.gz --artifact motion

The output is always on the input's grid, orientation and intensity units,
so the clean/corrupted pair is voxel-aligned for downstream training."""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
import torch

from . import PUBLIC_ARTIFACTS, __version__
from .generate import PLANES, generate_volume
from .model import load_for_inference
from .slab import PATCH_Z, save_volume, to_ras
from .volume import resample_iso, resample_like
from .weights import DEFAULT_MODEL, MODELS, WeightsError, resolve_checkpoint


class CliError(Exception):
    """A user-facing failure: printed as one line, exit code 1."""


def default_output(input_path, artifact):
    directory, base = os.path.split(input_path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return os.path.join(directory, f"{base}_{artifact}.nii.gz")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="artifact-gen",
        description="Generate a realistic MRI artifact on a clean brain volume with a "
                    "class-conditioned flow-matching model. The output keeps the input's grid, "
                    "orientation and intensity units.")
    p.add_argument("input", nargs="?", help="clean 3D NIfTI (.nii or .nii.gz)")
    p.add_argument("output", nargs="?", help="output NIfTI (default: <input stem>_<artifact>.nii.gz)")
    p.add_argument("-i", "--input", dest="input_opt", metavar="INPUT",
                   help="input NIfTI; same as the positional INPUT")
    p.add_argument("-o", "--output", dest="output_opt", metavar="OUTPUT",
                   help="output NIfTI; same as the positional OUTPUT")
    p.add_argument("--artifact", required=True, choices=PUBLIC_ARTIFACTS,
                   help="artifact to generate (motion = MR-ART-style head motion; spike = k-space "
                        "spike; grappa = parallel-imaging aliasing/noise amplification; lowres = "
                        "low-resolution blur)")
    p.add_argument("--strength", "--guidance-scale", dest="strength", type=float, default=3.0,
                   help="classifier-free-guidance scale: 1 = no guidance, 1.5 mild, 3 moderate "
                        "(default), 5 strong")
    p.add_argument("--steps", type=int, default=20, help="Euler sampling steps (default 20)")
    p.add_argument("--plane", choices=PLANES, default="axial",
                   help="slab orientation the volume is generated along (default axial)")
    p.add_argument("--overlap", type=int, default=1, metavar="N",
                   help=f"slices shared by neighbouring {PATCH_Z}-slice slabs and linearly blended "
                        f"(default 1; 0 = none, up to {PATCH_Z - 1})")
    p.add_argument("--iso", type=float, default=None, metavar="MM",
                   help="resample the input to an MM-mm isotropic grid for inference; the output "
                        "is resampled back to the input grid (default: native grid)")
    p.add_argument("--seed", type=int, default=None,
                   help="noise seed (default random; the seed used is printed)")
    p.add_argument("--model", choices=tuple(MODELS), default=DEFAULT_MODEL,
                   help="which Hub model to download: small = 64-192-320 channels, 68M params "
                        "(default); large = 128-256-512 channels, 165M params, slower. Ignored with "
                        "--checkpoint or $MRIARTIFACTS_CHECKPOINT")
    p.add_argument("--checkpoint", default=None,
                   help="local weights (.pt or .safetensors); default: download --model from the "
                        "Hub, or $MRIARTIFACTS_CHECKPOINT")
    p.add_argument("--device", default="auto", help="auto (default), cuda, cuda:N or cpu")
    p.add_argument("--fp32", action="store_true", help="disable bf16 autocast on CUDA")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="run the UNet eagerly instead of through torch.compile (on by default on "
                        "CUDA; the first slab batch pays a one-off compile cost)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)
    return resolve_paths(p, args)


def resolve_paths(parser, args):
    """Merge the positional INPUT/OUTPUT with -i/-o into `args.input` and
    `args.output`. With -i, a single positional is taken as the output
    (`artifact-gen -i in.nii out.nii`); giving the same path twice is an error."""
    positional = [a for a in (args.input, args.output) if a is not None]
    if args.input_opt is not None:
        if len(positional) > 1 or (positional and args.output_opt is not None):
            parser.error("input/output given both positionally and with -i/-o")
        args.input = args.input_opt
        args.output = args.output_opt if args.output_opt is not None else (positional[0] if positional else None)
    elif args.output_opt is not None:
        if args.output is not None:
            parser.error("output given both positionally and with -o")
        args.output = args.output_opt
    if args.input is None:
        parser.error("an input NIfTI is required (positional INPUT or -i/--input)")
    return args


def pick_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("warning: no CUDA device found; running on CPU, which is slow for this "
              "165M-parameter 3D model", file=sys.stderr)
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise CliError("CUDA requested but torch.cuda.is_available() is False; use --device cpu")
    return torch.device(name)


def use_amp(device, fp32):
    """bf16 autocast only on a CUDA device that supports it (pre-Ampere GPUs
    do not; there bf16 convolutions fail or crawl), and never with --fp32."""
    return device.type == "cuda" and not fp32 and torch.cuda.is_bf16_supported()


def compile_model(flow_matcher, device, enabled):
    """Wrap the UNet in torch.compile on CUDA (off with --no-compile, and never
    on CPU, where the compile cost outweighs the gain). Returns whether it is
    compiled; the FlowMatcher is changed in place."""
    if not enabled or device.type != "cuda":
        return False
    if not hasattr(torch, "compile"):
        print("warning: torch.compile needs torch >= 2.0; running eagerly", file=sys.stderr)
        return False
    flow_matcher.model = torch.compile(flow_matcher.model)
    return True


def load_input(path):
    if not os.path.isfile(path):
        raise CliError(f"input not found: {path}")
    try:
        img = nib.load(path)
    except Exception as e:
        raise CliError(f"cannot read {path} as NIfTI: {e}")
    shape = tuple(img.shape)
    if len(shape) > 3 and all(s == 1 for s in shape[3:]):
        data = np.asanyarray(img.dataobj).reshape(shape[:3])
        img = nib.Nifti1Image(data, img.affine, img.header)
    elif len(shape) != 3:
        raise CliError(f"expected a 3D volume, got shape {shape}; split 4D series first")
    return img


def run(args):
    output = args.output or default_output(args.input, args.artifact)
    if os.path.realpath(output) == os.path.realpath(args.input):   # realpath: symlinked inputs too
        raise CliError("output path equals the input path; refusing to overwrite the input")
    if os.path.exists(output) and not args.overwrite:
        raise CliError(f"{output} exists; pass --overwrite to replace it")
    if not 0 <= args.overlap < PATCH_Z:
        raise CliError(f"--overlap must be between 0 and {PATCH_Z - 1}, got {args.overlap}")
    img = load_input(args.input)
    device = pick_device(args.device)
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
    print(f"seed: {seed}")
    try:
        ckpt = resolve_checkpoint(args.checkpoint, model=args.model)
    except WeightsError as e:
        raise CliError(str(e))
    amp = use_amp(device, args.fp32)
    flow_matcher, conditioner, channels = load_for_inference(ckpt, device, amp=amp)
    compiled = compile_model(flow_matcher, device, args.compile)
    print(f"weights: {ckpt} (channels {channels}, device {device}, {'bf16' if amp else 'fp32'}, "
          f"{'torch.compile' if compiled else 'eager'})")
    ras, to_orig = to_ras(img)
    work = resample_iso(ras, args.iso) if args.iso else ras
    out01, window = generate_volume(work, flow_matcher, conditioner, args.artifact, args.strength,
                                    args.steps, args.plane, seed, device,
                                    non_overlap=args.overlap)
    if args.iso:
        out01 = np.asarray(resample_like(nib.Nifti1Image(out01, work.affine), ras).dataobj, dtype=np.float32)
    save_volume(out01, output, to_orig, window, img)
    print(f"saved {output}")
    return output


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
