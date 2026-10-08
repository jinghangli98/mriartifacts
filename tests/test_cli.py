import os
import re

import nibabel as nib
import numpy as np
import pytest

from mriartifacts import cli

LPS_ANISO = np.diag([-1.0, -1.2, 2.0, 1.0])


def _write(path, shape=(40, 40, 12), affine=None, dtype=np.float32, seed=0, scale=800.0):
    data = (np.random.default_rng(seed).random(shape) * scale).astype(dtype)
    img = nib.Nifti1Image(data, np.eye(4) if affine is None else affine)
    img.to_filename(str(path))
    return str(path)


def _run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_default_output_name():
    assert cli.default_output("/d/sub1.nii.gz", "motion") == "/d/sub1_motion.nii.gz"
    assert cli.default_output("/d/sub1.nii", "spike") == "/d/sub1_spike.nii.gz"
    assert cli.default_output("sub1.nii.gz", "lowres") == "sub1_lowres.nii.gz"


def test_cli_output_matches_input_grid_lps_anisotropic(tmp_path, tiny_ckpt, capsys):
    """Review focus 1: output on the input's grid, orientation, affine and units."""
    inp = _write(tmp_path / "t1.nii.gz", affine=LPS_ANISO)
    code, out, err = _run(capsys, inp, "--artifact", "motion", "--checkpoint", tiny_ckpt,
                          "--steps", "1", "--device", "cpu", "--seed", "1")
    assert code == 0, err
    res = nib.load(str(tmp_path / "t1_motion.nii.gz"))
    src = nib.load(inp)
    assert res.shape == src.shape and np.allclose(res.affine, src.affine)
    assert res.get_data_dtype() == np.float32
    lo, hi = np.percentile(src.get_fdata(), [0.5, 99.5])
    assert lo - 1 <= res.get_fdata().min() and res.get_fdata().max() <= hi + 1
    assert "seed: 1" in out and "saved" in out


def test_cli_iso_roundtrips_to_input_grid(tmp_path, tiny_ckpt, capsys):
    inp = _write(tmp_path / "t1.nii.gz", affine=LPS_ANISO)
    out_path = str(tmp_path / "o.nii.gz")
    code, _, err = _run(capsys, inp, out_path, "--artifact", "lowres", "--checkpoint", tiny_ckpt,
                        "--steps", "1", "--device", "cpu", "--iso", "2", "--seed", "0")
    assert code == 0, err
    res, src = nib.load(out_path), nib.load(inp)
    assert res.shape == src.shape and np.allclose(res.affine, src.affine)
    assert np.isfinite(res.get_fdata()).all()


def test_cli_random_seed_is_printed_and_reproducible(tmp_path, tiny_ckpt, capsys):
    inp = _write(tmp_path / "t1.nii.gz", shape=(24, 24, 8))
    a = str(tmp_path / "a.nii.gz")
    code, out, _ = _run(capsys, inp, a, "--artifact", "spike", "--checkpoint", tiny_ckpt, "--steps", "1",
                        "--device", "cpu")
    assert code == 0
    seed = re.search(r"seed: (\d+)", out).group(1)
    b = str(tmp_path / "b.nii.gz")
    code, _, _ = _run(capsys, inp, b, "--artifact", "spike", "--checkpoint", tiny_ckpt, "--steps", "1",
                      "--device", "cpu", "--seed", seed)
    assert code == 0
    assert np.array_equal(nib.load(a).get_fdata(), nib.load(b).get_fdata())


def test_cli_rejects_grappa_and_cs(tmp_path, tiny_ckpt):
    inp = _write(tmp_path / "t1.nii.gz", shape=(16, 16, 8))
    for fam in ("grappa", "cs"):
        with pytest.raises(SystemExit) as e:
            cli.main([inp, "--artifact", fam, "--checkpoint", tiny_ckpt])
        assert e.value.code == 2


def test_cli_refuses_existing_output_without_overwrite(tmp_path, tiny_ckpt, capsys):
    inp = _write(tmp_path / "t1.nii.gz", shape=(16, 16, 8))
    out_path = _write(tmp_path / "o.nii.gz", shape=(2, 2, 2))
    code, _, err = _run(capsys, inp, out_path, "--artifact", "motion", "--checkpoint", tiny_ckpt,
                        "--steps", "1", "--device", "cpu")
    assert code == 1 and "--overwrite" in err
    code, _, err = _run(capsys, inp, out_path, "--artifact", "motion", "--checkpoint", tiny_ckpt,
                        "--steps", "1", "--device", "cpu", "--overwrite")
    assert code == 0, err


def test_cli_refuses_output_equal_to_input(tmp_path, tiny_ckpt, capsys):
    """Review focus 3."""
    inp = _write(tmp_path / "t1.nii.gz", shape=(16, 16, 8))
    code, _, err = _run(capsys, inp, inp, "--artifact", "motion", "--checkpoint", tiny_ckpt, "--overwrite")
    assert code == 1 and "overwrite the input" in err


def test_cli_missing_input_and_bad_checkpoint(tmp_path, capsys):
    code, _, err = _run(capsys, str(tmp_path / "none.nii.gz"), "--artifact", "motion",
                        "--checkpoint", str(tmp_path / "none.pt"))
    assert code == 1 and "input not found" in err
    inp = _write(tmp_path / "t1.nii.gz", shape=(16, 16, 8))
    code, _, err = _run(capsys, inp, "--artifact", "motion", "--checkpoint", str(tmp_path / "none.pt"),
                        "--device", "cpu")
    assert code == 1 and "checkpoint not found" in err


def test_cli_refuses_4d(tmp_path, tiny_ckpt, capsys):
    """Review focus 4: a real 4D series is refused."""
    inp = _write(tmp_path / "f.nii.gz", shape=(16, 16, 8, 3))
    code, _, err = _run(capsys, inp, "--artifact", "motion", "--checkpoint", tiny_ckpt, "--device", "cpu")
    assert code == 1 and "split 4D series first" in err


def test_cli_squeezes_trailing_singleton(tmp_path, tiny_ckpt, capsys):
    """Review focus 4: (X, Y, Z, 1) is a 3D volume."""
    inp = _write(tmp_path / "s.nii.gz", shape=(16, 16, 8, 1))
    code, _, err = _run(capsys, inp, "--artifact", "motion", "--checkpoint", tiny_ckpt, "--steps", "1",
                        "--device", "cpu", "--seed", "0")
    assert code == 0, err
    assert nib.load(str(tmp_path / "s_motion.nii.gz")).shape == (16, 16, 8)


def test_cli_cuda_requested_but_unavailable(tmp_path, tiny_ckpt, capsys, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    inp = _write(tmp_path / "t1.nii.gz", shape=(16, 16, 8))
    code, _, err = _run(capsys, inp, "--artifact", "motion", "--checkpoint", tiny_ckpt, "--device", "cuda")
    assert code == 1 and "CUDA" in err


def test_cli_auto_falls_back_to_cpu_with_warning(tmp_path, tiny_ckpt, capsys, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    inp = _write(tmp_path / "t1.nii.gz", shape=(16, 16, 8))
    code, _, err = _run(capsys, inp, "--artifact", "motion", "--checkpoint", tiny_ckpt, "--steps", "1")
    assert code == 0 and "CPU" in err


def test_cli_version_and_strength_alias(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    args = cli.parse_args(["in.nii.gz", "--artifact", "motion", "--guidance-scale", "5"])
    assert args.strength == 5.0
    assert cli.parse_args(["in.nii.gz", "--artifact", "motion"]).strength == 3.0


def test_use_amp_only_on_cuda_with_bf16_support(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert cli.use_amp(torch.device("cuda"), fp32=False) is True
    assert cli.use_amp(torch.device("cuda"), fp32=True) is False
    assert cli.use_amp(torch.device("cpu"), fp32=False) is False
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert cli.use_amp(torch.device("cuda"), fp32=False) is False   # pre-Ampere GPUs: fall back to fp32


def test_cli_refuses_output_equal_to_input_via_symlink(tmp_path, tiny_ckpt, capsys):
    """A symlinked input must not be overwritten through its target path."""
    target = _write(tmp_path / "t1.nii.gz", shape=(16, 16, 8))
    link = tmp_path / "link.nii.gz"
    link.symlink_to(target)
    code, _, err = _run(capsys, str(link), target, "--artifact", "motion", "--checkpoint", tiny_ckpt, "--overwrite")
    assert code == 1 and "overwrite the input" in err
    assert nib.load(target).shape == (16, 16, 8)
