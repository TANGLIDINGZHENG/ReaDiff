"""Load the published generator, shared modules and optional motion evaluator."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from .model import ReaDiff
from .shared import SharedModules


ROOT = Path(__file__).resolve().parents[1]


def setup_device(name):
    device = torch.device(name)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return device


def load_weights(module, directory, variant, device):
    path = directory / f"ReaDiff-weights-{variant}.safetensors"
    legacy = directory / f"ReaDiff-{variant}.safetensors"
    if not path.is_file() and legacy.is_file():
        path = legacy
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Download this asset from the ReaDiff release.")
    module.load_state_dict(load_file(str(path)), strict=True, assign=True)
    return module.to(device).eval().requires_grad_(False)


def load_generator(size, weights, device):
    config = json.loads((ROOT / "configs" / f"ReaDiff-{size}.json").read_text())
    model = load_weights(ReaDiff(config["architecture"]), weights, size, device)
    shared = load_weights(SharedModules(), weights, "shared", device)
    return model, shared, config["sampling"]
