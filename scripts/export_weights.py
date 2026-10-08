#!/usr/bin/env python
"""Export a training checkpoint for the Hub.

    python scripts/export_weights.py path/to/checkpoint.pt hub_export
    hf upload jil202/mriartifacts hub_export .   # maintainer token

Writes model.safetensors, config.json and README.md (from hub/README.md)."""
import json
import os
import sys

from mriartifacts.export import export_checkpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    cfg = export_checkpoint(sys.argv[1], sys.argv[2], card_path=os.path.join(REPO_ROOT, "hub", "README.md"))
    print(json.dumps(cfg, indent=1))
