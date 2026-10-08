"""Export a training checkpoint to the files served from the Hugging Face Hub:
`model.safetensors` (fp32, so results match the paper exactly), `config.json`
(channels, families, training step, source checksum) and the model card."""

import hashlib
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from . import FAMILIES, __version__
from .model import clean_state_dict, infer_channels

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def export_checkpoint(ckpt_path, out_dir, card_path=None):
    """Write `out_dir/model.safetensors`, `out_dir/config.json` and, when
    `card_path` exists, `out_dir/README.md`. Returns the config dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {k: v.contiguous() for k, v in clean_state_dict(ckpt["model"]).items()}
    os.makedirs(out_dir, exist_ok=True)
    save_file(sd, os.path.join(out_dir, WEIGHTS_FILE))
    cfg = {
        "channels": [int(c) for c in (ckpt.get("channels") or infer_channels(sd))],
        "families": list(ckpt.get("families", FAMILIES)),
        "global_step": ckpt.get("global_step"),
        "source_sha256": _sha256(ckpt_path),
        "exported_by": f"mriartifacts {__version__}",
    }
    with open(os.path.join(out_dir, CONFIG_FILE), "w") as f:
        json.dump(cfg, f, indent=1)
    if card_path and os.path.exists(card_path):
        shutil.copyfile(card_path, os.path.join(out_dir, "README.md"))
    return cfg
