"""Flow-matching pieces shared by training, generation and evaluation.

Condensed from autoflow/train_flow.py and enhance_flow_3d.py. Lives in the
`mriartifacts` package so the CLI ships without the training code; the repo's
`flow_model.py` re-exports it.

* `ClassConditioner`: the artifact family as a class index `(B,)` for the
  UNet's class embedding (MONAI `num_class_embeds` = NUM_CLASSES =
  len(FAMILIES) + 1): a learned 6 x time_embed_dim table whose row is added to
  the timestep embedding, so the family reaches every ResBlock at every
  resolution. The extra index (NULL_INDEX) is the unconditional signal for
  classifier-free guidance: during training a sample's label is swapped for
  the null label with probability `cfg_dropout_prob`; `null()` returns null
  labels for guided sampling. The conditioner itself has no parameters. The
  UNet has NO cross-attention (`with_conditioning=False`); the deepest level
  keeps self-attention.
* `FlowMatcher`: the UNet takes `cat(x_t, condition)` (2 channels) and
  predicts the velocity (1 output channel; plain MSE loss). Sampling is Euler
  integration from t=0 (noise) to t=1 with CFG
  `v = v_u + scale * (v_c - v_u)`. Checkpoints with a 2-channel head
  (velocity + log-variance) still load: channel 0 is the velocity and the
  sampler then also returns a per-voxel predictive std. Cross-attention
  checkpoints (the text model and the one-hot-token model, everything before
  the class-embedding change of 2026-10-05) do NOT load for inference:
  `load_for_inference` raises ValueError. They do serve as relaxed warm
  starts (`load_checkpoint`): convolutions / resnets load, attention and the
  class table start fresh.
* Thin slabs (Z = 7) are replicate-padded to a multiple of 4 (the UNet's
  total downsampling) inside `_unet` and cropped back.
"""

import json
import os

import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet

from . import FAMILIES, NULL_INDEX, NUM_CLASSES

DEPTH_MULT = 4


def pad_depth(x, multiple=4):
    """Replicate-pad the last (Z) axis of `x` up to a multiple of `multiple`.

    Thin training patches (e.g. Z=7) are not divisible by the UNet's total
    downsampling factor, which breaks the skip concatenations. Returns
    `(padded, original_z)`; `x` itself is returned when no padding is needed.
    """
    z0 = x.shape[-1]
    extra = (-z0) % multiple
    if extra == 0:
        return x, z0
    last = x[..., -1:].expand(*x.shape[:-1], extra)
    return torch.cat([x, last], dim=-1), z0


def crop_depth(y, original_z):
    """Undo `pad_depth`: keep the first `original_z` slices along Z."""
    return y[..., :original_z] if y.shape[-1] != original_z else y


def relaxed_state_dict(src_sd, tgt_sd):
    """Loosely match a checkpoint to a model: for each key present in both,
    copy the block of the checkpoint tensor that overlaps the model tensor
    (`tgt[:s0, :s1, ...] = src[:s0, :s1, ...]`) and keep the model's own
    init elsewhere. Handles the variance head (out 1 -> 2), a wider input
    conv, and channel widening (e.g. 128 -> 256 changes a conv's in- and
    out-dims at once). Tensors of different rank, and keys the model lacks,
    are skipped.

    Returns `(filtered, partial, skipped)`: the state dict to load with
    `strict=False`, a list of `(key, src_shape, tgt_shape, ((dim, copied),
    ...))` for partially copied tensors, and `(key, src_shape, tgt_shape or
    None)` for skipped ones."""
    filtered, partial, skipped = {}, [], []
    for k, v in src_sd.items():
        if k not in tgt_sd:
            skipped.append((k, tuple(v.shape), None))
            continue
        tv = tgt_sd[k]
        if v.shape == tv.shape:
            filtered[k] = v
            continue
        if v.dim() != tv.dim():
            skipped.append((k, tuple(v.shape), tuple(tv.shape)))
            continue
        overlap = tuple(min(a, b) for a, b in zip(v.shape, tv.shape))
        slicer = tuple(slice(0, n) for n in overlap)
        new_v = tv.clone()
        new_v[slicer] = v[slicer].to(new_v.dtype)
        filtered[k] = new_v
        partial.append((k, tuple(v.shape), tuple(tv.shape),
                        tuple((d, overlap[d]) for d in range(v.dim())
                              if v.shape[d] != tv.shape[d])))
    return filtered, partial, skipped


class ClassConditioner(nn.Module):
    """Artifact-family name -> class index for the UNet's class embedding (see
    module doc). `encode` / `null` keep the interface of the earlier text and
    one-hot conditioners, so the trainer, generators and evaluator are
    unchanged callers; they now carry `(B,)` long tensors."""

    def __init__(self, families=FAMILIES):
        super().__init__()
        self.families = tuple(families)
        self.index = {f: i for i, f in enumerate(self.families)}
        self.null_index = len(self.families)
        self.num_classes = self.null_index + 1
        # Tracks the conditioner's device so `null()` lands where the model is.
        self.register_buffer("_anchor", torch.zeros(1, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def encode(self, families, device, cfg_dropout_prob=0.0):
        try:
            idx = torch.tensor([self.index[f] for f in families], dtype=torch.long)
        except KeyError as e:
            raise ValueError(f"unknown artifact family {e.args[0]!r}; "
                             f"expected one of {self.families}") from None
        idx = idx.to(device)
        if cfg_dropout_prob > 0 and idx.numel():
            drop = torch.rand(idx.shape[0], device=idx.device) < cfg_dropout_prob
            idx = torch.where(drop, torch.full_like(idx, self.null_index), idx)
        return idx

    @torch.no_grad()
    def null(self, batch_size, device=None):
        """Null labels `(B,)`; on `device` if given, else on the conditioner's
        device (move the conditioner with `.to(device)`)."""
        dev = self._anchor.device if device is None else device
        return torch.full((batch_size,), self.null_index, dtype=torch.long, device=dev)


class FlowMatcher(nn.Module):
    def __init__(self, model, sigma_min=0.001, logvar_clamp=7.0, amp=False):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.logvar_clamp = logvar_clamp
        self.amp = amp  # bf16 autocast around UNet forwards during sampling

    def _unet(self, model_input, timesteps, context):
        """`context` is the `(B,)` class-label tensor (name kept for the
        callers' keyword arguments)."""
        x, z0 = pad_depth(model_input, DEPTH_MULT)
        return crop_depth(self.model(x, timesteps, class_labels=context), z0)

    def forward(self, x0, x1, condition, t, context=None):
        """Flow-matching MSE on the velocity `x1 - x0` at `x_t`."""
        batch_size = x0.shape[0]
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        t = t.view(-1, 1, 1, 1, 1)
        x_t = t * x1 + (1 - t) * x0 + self.sigma_min * torch.randn_like(x1)
        v_t = x1 - x0
        timesteps = (t.view(-1) * 999).long()
        out = self._unet(torch.cat([x_t, condition], dim=1), timesteps, context)
        return ((out[:, 0:1] - v_t) ** 2).mean()

    @staticmethod
    def _logvar(out):
        return out[:, 1:2].float() if out.shape[1] > 1 else None

    @torch.no_grad()
    def _guided_velocity(self, model_input, timesteps, context, null_context, guidance_scale):
        device_type = "cuda" if model_input.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=self.amp):
            out_c = self._unet(model_input, timesteps, context)
            if guidance_scale == 1.0 or null_context is None:
                return out_c[:, 0:1].float(), self._logvar(out_c)
            out_u = self._unet(model_input, timesteps, null_context)
        v_c, v_u = out_c[:, 0:1].float(), out_u[:, 0:1].float()
        if guidance_scale == 0.0:
            return v_u, self._logvar(out_u)
        return v_u + guidance_scale * (v_c - v_u), self._logvar(out_c)

    @torch.no_grad()
    def sample(self, condition, num_steps, device, context=None, null_context=None,
               guidance_scale=1.0, noise=None):
        """Euler ODE integration; returns `(x, uncertainty)`. `uncertainty` is
        None for 1-channel models; for legacy 2-channel checkpoints it is
        `sqrt(sum_i dt^2 * exp(logvar_i))`. Pass a fixed `noise` to compare
        guidance scales under identical draws."""
        x = torch.randn_like(condition) if noise is None else noise.to(condition.device)
        x = x.float()
        dt = 1.0 / num_steps
        var_accum = None
        for i in range(num_steps):
            t = torch.full((condition.shape[0],), i * dt, device=x.device)
            timesteps = (t * 999).long()
            v, logvar = self._guided_velocity(torch.cat([x, condition], dim=1),
                                              timesteps, context, null_context, guidance_scale)
            if logvar is not None:
                var_accum = (torch.zeros_like(x) if var_accum is None else var_accum) + \
                    (dt ** 2) * torch.exp(logvar.clamp(-self.logvar_clamp, self.logvar_clamp))
            x = torch.clamp(x + v * dt, -10.0, 10.0)
        return x, (None if var_accum is None else var_accum.sqrt())


def build_unet(channels=(128, 256, 512), num_class_embeds=NUM_CLASSES, norm_num_groups=32,
               out_channels=1):
    """Class-conditional UNet: self-attention at the deepest level, no
    cross-attention; the class embedding is added to the timestep embedding."""
    return DiffusionModelUNet(
        spatial_dims=3, in_channels=2, out_channels=out_channels,
        channels=tuple(channels), attention_levels=(False, False, True),
        num_res_blocks=2, num_head_channels=channels[-1],
        norm_num_groups=norm_num_groups,
        with_conditioning=False, num_class_embeds=num_class_embeds,
    )


def clean_state_dict(sd):
    return {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}


def infer_channels(sd):
    """Channel widths from `down_blocks.<i>.resnets.0.conv1.conv.weight`."""
    widths = {}
    for k, v in sd.items():
        if k.startswith("down_blocks.") and k.endswith(".resnets.0.conv1.conv.weight"):
            widths[int(k.split(".")[1])] = int(v.shape[0])
    if not widths:
        raise ValueError("cannot infer UNet channels from checkpoint")
    return tuple(widths[i] for i in range(len(widths)))


def infer_out_channels(sd):
    """Output channels from the final conv `out.2.conv.weight` (1 = velocity
    only, 2 = legacy velocity + log-variance)."""
    w = sd.get("out.2.conv.weight")
    return int(w.shape[0]) if w is not None else 1


def read_checkpoint(path):
    """Checkpoint dict `{"model": state_dict, ...}` from a training `.pt` or a
    Hub `model.safetensors` (with the advisory `config.json` beside it, when
    present, supplying `channels` and `families`)."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        ckpt = {"model": load_file(str(path))}
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            ckpt.update({k: cfg[k] for k in ("channels", "families", "global_step") if k in cfg})
        return ckpt
    return torch.load(path, map_location="cpu", weights_only=True)


def load_for_inference(path, device, amp=False):
    ckpt = read_checkpoint(path)
    sd = clean_state_dict(ckpt["model"])
    if "class_embedding.weight" not in sd:
        raise ValueError(
            f"{path}: no class_embedding table; this is a cross-attention checkpoint (text or "
            "one-hot token, before the class-embedding change of 2026-10-05) and cannot be loaded "
            "for inference (it can still be a relaxed warm start via load_checkpoint)")
    if tuple(sd["class_embedding.weight"].shape)[0] != NUM_CLASSES:
        raise ValueError(f"{path}: class table has {sd['class_embedding.weight'].shape[0]} rows, "
                         f"expected {NUM_CLASSES}")
    if "families" in ckpt and tuple(ckpt["families"]) != tuple(FAMILIES):
        raise ValueError(f"{path}: checkpoint families {tuple(ckpt['families'])} != "
                         f"current FAMILIES {tuple(FAMILIES)}")
    channels = tuple(ckpt["channels"]) if "channels" in ckpt else infer_channels(sd)
    conditioner = ClassConditioner().to(device)
    model = build_unet(channels, num_class_embeds=conditioner.num_classes,
                       out_channels=infer_out_channels(sd))
    model.load_state_dict(sd)
    model.to(device).eval()
    return FlowMatcher(model, amp=amp).eval(), conditioner, channels
