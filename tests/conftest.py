import pytest


@pytest.fixture(scope="session")
def tiny_ckpt(tmp_path_factory):
    """A randomly initialised 32-32-64 model saved like a training checkpoint."""
    import torch
    from mriartifacts.model import build_unet
    torch.manual_seed(0)
    unet = build_unet(channels=(32, 32, 64))
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    torch.save({"model": unet.state_dict(), "channels": (32, 32, 64),
                "families": ("motion", "spike", "grappa", "lowres", "cs"), "global_step": 1}, path)
    return str(path)
