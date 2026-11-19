"""Generate ViMo test reactions and compute FID, Diversity and Multimodality."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from readiff.evaluation import MotionEvaluator, ViMoTest, metrics, summarize
from readiff.runtime import ROOT, load_generator, load_weights, setup_device
from readiff.sampler import sample


def prepare(dataset, shared, evaluator, device, workers):
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers,
                        prefetch_factor=1 if workers else None)
    batches, real = [], []
    for images, motion, lengths in loader:
        lengths = lengths.to(device)
        real.append(evaluator(motion.to(device), lengths).cpu().numpy())
        features = shared.encode(images.to(device))
        batches.append((features, lengths))
    return batches, np.concatenate(real)


def evaluate_repeat(model, shared, evaluator, batches, real, sampling, generator, rng):
    predicted, multimodal = [], []
    for index, (features, lengths) in enumerate(batches):
        embeddings = []
        for _ in range(30 if index < 3 else 1):
            latent = sample(model, features, lengths // 4, sampling, generator)
            motion = shared.decoder(latent)
            embeddings.append(evaluator(motion, lengths).cpu().numpy())
        if index < 3:
            multimodal.append(np.stack(embeddings, axis=1))
        predicted.append(embeddings[-1])
    return metrics(real, np.concatenate(predicted), np.concatenate(multimodal), rng)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("S", "M", "L"))
    parser.add_argument("--data-root", required=True, type=Path, help="ViMo directory containing test.txt")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=0, help="Video decoding processes")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 0:
        parser.error("--repeats must be positive and --workers nonnegative")
    device = setup_device(args.device)
    cv2.setNumThreads(1)
    model, shared, sampling = load_generator(args.model, args.weights, device)
    evaluator = load_weights(MotionEvaluator(), args.weights, "evaluator", device)
    dataset = ViMoTest(args.data_root, shared.mean.cpu().numpy(), shared.std.cpu().numpy())
    print(f"Encoding {len(dataset)} ViMo test clips...", flush=True)
    batches, real = prepare(dataset, shared, evaluator, device, args.workers)
    print(f"Generating {args.repeats} evaluation repeat(s) for ReaDiff-{args.model}...", flush=True)
    seed = 10107
    generator = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.RandomState(seed)
    rows = []
    for index in range(args.repeats):
        row = evaluate_repeat(model, shared, evaluator, batches, real, sampling, generator, rng)
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(f"Non-finite metric in repeat {index + 1}")
        rows.append(row)
        print(f"Repeat {index + 1}/{args.repeats}  {json.dumps(row)}", flush=True)
    report = dict(model=f"ReaDiff-{args.model}", samples=len(dataset), seed=seed, repeats=args.repeats,
                  per_repeat=rows, summary=summarize(rows))
    print(json.dumps(report["summary"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
