import pytest

from mriartifacts import weights as w


def test_explicit_path_wins(tmp_path, monkeypatch):
    p = tmp_path / "a.pt"
    p.write_bytes(b"x")
    monkeypatch.setenv(w.ENV_VAR, str(tmp_path / "missing.pt"))
    monkeypatch.setattr(w, "_hub_download", lambda repo, filename: pytest.fail("hub must not be called"))
    assert w.resolve_checkpoint(str(p)) == str(p)


def test_explicit_missing_path_errors(tmp_path):
    with pytest.raises(w.WeightsError, match="not found"):
        w.resolve_checkpoint(str(tmp_path / "nope.pt"))


def test_env_var_used_when_no_path(tmp_path, monkeypatch):
    p = tmp_path / "env.safetensors"
    p.write_bytes(b"x")
    monkeypatch.setenv(w.ENV_VAR, str(p))
    monkeypatch.setattr(w, "_hub_download", lambda repo, filename: pytest.fail("hub must not be called"))
    assert w.resolve_checkpoint(None) == str(p)


def test_hub_download_fetches_weights_and_config(tmp_path, monkeypatch):
    monkeypatch.delenv(w.ENV_VAR, raising=False)
    calls = []

    def fake(repo, filename):
        calls.append((repo, filename))
        f = tmp_path / filename
        f.write_bytes(b"x")
        return str(f)
    monkeypatch.setattr(w, "_hub_download", fake)
    assert w.resolve_checkpoint(None, repo="someone/repo") == str(tmp_path / w.WEIGHTS_FILE)
    assert calls == [("someone/repo", w.WEIGHTS_FILE), ("someone/repo", "config.json")]


def test_hub_failure_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(w.ENV_VAR, raising=False)

    def boom(repo, filename):
        raise OSError("offline")
    monkeypatch.setattr(w, "_hub_download", boom)
    with pytest.raises(w.WeightsError) as e:
        w.resolve_checkpoint(None, repo="someone/repo")
    msg = str(e.value)
    assert "someone/repo" in msg and "--checkpoint" in msg and w.ENV_VAR in msg and "offline" in msg


def test_missing_config_is_not_fatal(tmp_path, monkeypatch):
    monkeypatch.delenv(w.ENV_VAR, raising=False)

    def fake(repo, filename):
        if filename == "config.json":
            raise OSError("no config")
        f = tmp_path / filename
        f.write_bytes(b"x")
        return str(f)
    monkeypatch.setattr(w, "_hub_download", fake)
    assert w.resolve_checkpoint(None, repo="r/r") == str(tmp_path / w.WEIGHTS_FILE)


def test_default_repo_points_at_the_published_checkpoint():
    assert w.DEFAULT_REPO == "jil202/mriartifacts"
    assert w.WEIGHTS_FILE == "motion_cfg_classemb64w_100826_best.pt"
    assert w.MODELS == {"large": "motion_cfg_classemb128_100526_best.pt",
                        "small": "motion_cfg_classemb64w_100826_best.pt"}
    assert w.DEFAULT_MODEL == "small"


def test_model_selects_hub_file(tmp_path, monkeypatch):
    monkeypatch.delenv(w.ENV_VAR, raising=False)
    calls = []

    def fake(repo, filename):
        calls.append(filename)
        f = tmp_path / filename
        f.write_bytes(b"x")
        return str(f)
    monkeypatch.setattr(w, "_hub_download", fake)
    assert w.resolve_checkpoint(None, repo="r/r", model="small") == str(tmp_path / w.MODELS["small"])
    assert calls[0] == w.MODELS["small"]
    with pytest.raises(w.WeightsError, match="unknown model"):
        w.resolve_checkpoint(None, repo="r/r", model="huge")


def test_explicit_path_ignores_model(tmp_path, monkeypatch):
    p = tmp_path / "a.pt"
    p.write_bytes(b"x")
    monkeypatch.setattr(w, "_hub_download", lambda repo, filename: pytest.fail("hub must not be called"))
    assert w.resolve_checkpoint(str(p), model="small") == str(p)
