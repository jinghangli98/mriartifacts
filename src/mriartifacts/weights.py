"""Locate the model weights: an explicit path, the MRIARTIFACTS_CHECKPOINT
environment variable, or a cached download from the Hugging Face Hub."""

import os

DEFAULT_REPO = "jil202/mriartifacts"                       # https://huggingface.co/jil202/mriartifacts
# `--model` name -> weights file on the Hub (fp32 .pt state-dict checkpoints, loaded with weights_only=True)
MODELS = {
    "large": "motion_cfg_classemb128_100526_best.pt",   # channels 128-256-512, 165 M params, EMA step 62,500
    "small": "motion_cfg_classemb64w_100826_best.pt",   # channels 64-192-320, 68 M params, EMA step 25,000
}
DEFAULT_MODEL = "small"
WEIGHTS_FILE = MODELS[DEFAULT_MODEL]
CONFIG_FILE = "config.json"
ENV_VAR = "MRIARTIFACTS_CHECKPOINT"


class WeightsError(RuntimeError):
    """Weights could not be located; the message tells the user what to do."""


def _hub_download(repo, filename):
    from huggingface_hub import hf_hub_download   # imported lazily: offline users never need it
    return hf_hub_download(repo_id=repo, filename=filename)


def resolve_checkpoint(path=None, repo=DEFAULT_REPO, model=DEFAULT_MODEL):
    """Local file to load. Order: `path`, then `$MRIARTIFACTS_CHECKPOINT`, then
    the `model` entry of MODELS from `repo` on the Hub (cached under HF_HOME;
    later calls are offline no-ops). `config.json` is fetched beside it when
    available."""
    candidate = path or os.environ.get(ENV_VAR)
    if candidate:
        if not os.path.isfile(candidate):
            raise WeightsError(f"checkpoint not found: {candidate}")
        return candidate
    if model not in MODELS:
        raise WeightsError(f"unknown model {model!r}; expected one of {tuple(MODELS)}")
    filename = MODELS[model]
    try:
        weights = _hub_download(repo, filename)
    except Exception as e:  # network, auth, missing repo: all end the same way for the user
        raise WeightsError(
            f"could not download {filename} from Hub repo {repo} ({type(e).__name__}: {e}). "
            f"Pass --checkpoint PATH or set {ENV_VAR} to a local checkpoint.") from e
    try:
        _hub_download(repo, CONFIG_FILE)
    except Exception:
        pass   # advisory file; channels are inferred from the state dict without it
    return weights
