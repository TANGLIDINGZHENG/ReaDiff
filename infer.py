"""Generate a 3D human reaction from an RGB video with ReaDiff."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors.torch import load_file

from readiff.model import ReaDiff
from readiff.motion import recover_from_ric
from readiff.sampler import sample
from readiff.shared import SharedModules
from readiff.video import read_video


ROOT = Path(__file__).resolve().parent


def load_weights(module, path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Download this asset from the ReaDiff release.")
    module.load_state_dict(load_file(str(path)), strict=True, assign=True)
    return module.to(device).eval().requires_grad_(False)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("S", "M", "L"), required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=88, help="4 to 200, in multiples of four (20 fps)")
    parser.add_argument("--seed", type=int, default=10107)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video file does not exist: {args.video}")
    if not 4 <= args.frames <= 200 or args.frames % 4:
        parser.error("--frames must be a multiple of four from 4 to 200")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    name = f"ReaDiff-{args.model}"
    config = json.loads((ROOT / "configs" / f"{name}.json").read_text())
    model = load_weights(ReaDiff(config["architecture"]), args.weights / f"{name}.safetensors", device)
    shared = load_weights(SharedModules(), args.weights / "ReaDiff-shared.safetensors", device)
    features = shared.encode(read_video(args.video).unsqueeze(0).to(device))
    lengths = torch.tensor([args.frames // 4], device=device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent = sample(model, features, lengths, config["sampling"], generator)
    normalized = shared.decoder(latent)[0].cpu().numpy()
    motion = normalized * shared.std.cpu().numpy() + shared.mean.cpu().numpy()
    joints = recover_from_ric(torch.from_numpy(motion).float(), 22).numpy()
    if not np.isfinite(motion).all() or not np.isfinite(joints).all():
        raise RuntimeError("Generated non-finite motion")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.video.stem}_{name}_seed{args.seed}_len{args.frames}"
    for kind, data in (("motion", motion), ("joints", joints)):
        path = args.output / f"{stem}_{kind}.npy"
        np.save(path, data)
        print(f"{path}  {data.shape}")


if __name__ == "__main__":
    main()
