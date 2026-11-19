"""ViMo test inputs, the frozen HERO motion encoder and motion-embedding metrics."""
import csv
from pathlib import Path

import numpy as np
from scipy import linalg
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset

from .runtime import ROOT
from .video import read_video


class ViMoTest(Dataset):
    def __init__(self, root, mean, std):
        self.root = Path(root)
        split = (self.root / "test.txt").read_text().splitlines()
        pairs = dict(line.split() for line in split if line.strip())
        with (ROOT / "configs/vimo-test-crops.csv").open(newline="") as stream:
            self.crops = [(row["video"], row["motion"], int(row["start"]), int(row["frames"]))
                          for row in csv.DictReader(stream)]
        expected_pairs = {video: motion for video, motion, _, _ in self.crops}
        if len(pairs) != 700 or len(split) != 700 or pairs != expected_pairs:
            raise ValueError("Expected the official 700 video-motion pairs in --data-root/test.txt")
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        video, motion_file, start, frames = self.crops[index]
        path = self.root / motion_file
        motion = np.load(path, allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] != 263 or len(motion) < start + frames:
            raise ValueError(f"Invalid reference motion or crop for {path}")
        padded = np.zeros((200, 263), dtype=np.float32)
        padded[:frames] = (motion[start:start + frames] - self.mean) / self.std
        return read_video(self.root / video), padded, frames


class MovementEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(259, 512, 4, 2, 1), nn.Dropout(0.2, inplace=True), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(512, 512, 4, 2, 1), nn.Dropout(0.2, inplace=True), nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(512, 512)

    def forward(self, inputs):
        return self.out_net(self.main(inputs.permute(0, 2, 1)).permute(0, 2, 1))


class MotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_emb = nn.Linear(512, 1024)
        self.gru = nn.GRU(1024, 1024, batch_first=True, bidirectional=True)
        self.hidden = nn.Parameter(torch.empty(2, 1, 1024))
        self.output_net = nn.Sequential(nn.Linear(2048, 1024), nn.LayerNorm(1024),
                                        nn.LeakyReLU(0.2, inplace=True), nn.Linear(1024, 512))

    def forward(self, inputs, lengths):
        packed = pack_padded_sequence(self.input_emb(inputs), lengths, batch_first=True)
        _, hidden = self.gru(packed, self.hidden.repeat(1, len(inputs), 1))
        return self.output_net(torch.cat([hidden[0], hidden[1]], dim=-1))


class MotionEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.movement_encoder = MovementEncoder()
        self.motion_encoder = MotionEncoder()

    def forward(self, motion, lengths):
        # Preserve HERO's tie order for equal-length examples.
        order = np.argsort(lengths.tolist())[::-1].copy()
        motion = motion[order]
        movement = self.movement_encoder(motion[..., :-4])
        return self.motion_encoder(movement, (lengths[order] // 4).tolist())


def frechet_distance(real, predicted):
    mu_a, mu_b = real.mean(0), predicted.mean(0)
    cov_a, cov_b = np.cov(real, rowvar=False), np.cov(predicted, rowvar=False)
    root = linalg.sqrtm(cov_a @ cov_b)
    if not np.isfinite(root).all():
        offset = np.eye(len(mu_a)) * 1e-6
        root = linalg.sqrtm((cov_a + offset) @ (cov_b + offset))
    if np.iscomplexobj(root):
        if not np.allclose(root.diagonal().imag, 0, atol=1e-3):
            raise ValueError("FID covariance square root has a significant imaginary component")
        root = root.real
    diff = mu_a - mu_b
    return diff.dot(diff) + np.trace(cov_a) + np.trace(cov_b) - 2 * np.trace(root)


def metrics(real, predicted, multimodal, rng):
    # The pair draws share one NumPy stream across metrics and repetitions.
    first = rng.choice(30, 10, replace=False)
    second = rng.choice(30, 10, replace=False)
    mm = linalg.norm(multimodal[:, first] - multimodal[:, second], axis=2).mean()

    def diversity(embeddings):
        a = rng.choice(len(embeddings), 300, replace=False)
        b = rng.choice(len(embeddings), 300, replace=False)
        return linalg.norm(embeddings[a] - embeddings[b], axis=1).mean()

    real_diversity = diversity(real)
    generated_diversity = diversity(predicted)
    return dict(fid=float(frechet_distance(real, predicted)), diversity=float(generated_diversity),
                multimodality=float(mm), diversity_real=float(real_diversity))


def summarize(rows):
    summary = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        summary[key] = dict(mean=float(np.mean(values)),
                            ci95=float(1.96 * np.std(values) / np.sqrt(len(rows))) if len(rows) > 1 else None)
    return summary
