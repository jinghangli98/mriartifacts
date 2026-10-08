import hashlib
import json
import os

import numpy as np
import pytest
import torch

from mriartifacts import export as ex
from mriartifacts.model import load_for_inference


def test_export_writes_safetensors_and_config(tiny_ckpt, tmp_path):
    card = tmp_path / "card.md"
    card.write_text("# card\n")
    cfg = ex.export_checkpoint(tiny_ckpt, str(tmp_path / "hub"), card_path=str(card))
    assert sorted(os.listdir(tmp_path / "hub")) == ["README.md", "config.json", "model.safetensors"]
    on_disk = json.load(open(tmp_path / "hub" / "config.json"))
    assert on_disk == cfg
    assert cfg["channels"] == [32, 32, 64]
    assert cfg["families"] == ["motion", "spike", "grappa", "lowres", "cs"]
    assert cfg["global_step"] == 1
    assert cfg["source_sha256"] == hashlib.sha256(open(tiny_ckpt, "rb").read()).hexdigest()
    assert cfg["exported_by"].startswith("mriartifacts ")


def test_safetensors_loads_and_matches_pt(tiny_ckpt, tmp_path):
    ex.export_checkpoint(tiny_ckpt, str(tmp_path))
    fm_pt, cc_pt, ch_pt = load_for_inference(tiny_ckpt, "cpu")
    fm_st, cc_st, ch_st = load_for_inference(str(tmp_path / "model.safetensors"), "cpu")
    assert ch_pt == ch_st == (32, 32, 64)
    torch.manual_seed(0)
    cond = torch.rand(1, 1, 16, 16, 7)
    noise = torch.randn_like(cond)
    a, _ = fm_pt.sample(cond, 2, "cpu", context=cc_pt.encode(["motion"], "cpu"), null_context=cc_pt.null(1),
                        guidance_scale=3.0, noise=noise)
    b, _ = fm_st.sample(cond, 2, "cpu", context=cc_st.encode(["motion"], "cpu"), null_context=cc_st.null(1),
                        guidance_scale=3.0, noise=noise)
    assert torch.equal(a, b)


def test_safetensors_without_config_infers_channels(tiny_ckpt, tmp_path):
    ex.export_checkpoint(tiny_ckpt, str(tmp_path))
    os.remove(tmp_path / "config.json")
    _, _, ch = load_for_inference(str(tmp_path / "model.safetensors"), "cpu")
    assert ch == (32, 32, 64)


def test_safetensors_config_families_are_checked(tiny_ckpt, tmp_path):
    ex.export_checkpoint(tiny_ckpt, str(tmp_path))
    cfg = json.load(open(tmp_path / "config.json"))
    cfg["families"] = list(reversed(cfg["families"]))
    json.dump(cfg, open(tmp_path / "config.json", "w"))
    with pytest.raises(ValueError, match="families"):
        load_for_inference(str(tmp_path / "model.safetensors"), "cpu")
