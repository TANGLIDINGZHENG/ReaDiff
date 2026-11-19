"""RGB frames and dense optical flow for TC-CLIP."""
import math

import cv2
import decord
import numpy as np
import torch


def dense_flow(first, second):
    """Forward Farneback flow, (H, W, 2), with x/y displacement channels."""
    if first.shape != second.shape or first.ndim != 3 or first.shape[-1] != 3:
        raise ValueError("Flow requires two equally sized RGB images")
    if first.dtype != np.uint8 or second.dtype != np.uint8:
        raise ValueError("Flow requires unnormalized uint8 RGB images")
    if np.array_equal(first, second):
        return np.zeros((*first.shape[:2], 2), dtype=np.float32)
    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(first, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(second, cv2.COLOR_RGB2GRAY), None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)


def flow_to_rgb(flow, bound=20.0):
    """Direction as hue, fixed-scale magnitude as brightness; zero is black.

    A fixed bound (pixels/frame) preserves speed differences between videos.
    This is a three-channel representation for the RGB-pretrained TC-CLIP,
    not a claim that TC-CLIP was pretrained on optical flow.
    """
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("flow_bound must be finite and positive")
    if flow.ndim != 3 or flow.shape[-1] != 2 or not np.isfinite(flow).all():
        raise ValueError("Flow must be a finite (H, W, 2) array")
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.stack([angle, np.ones_like(magnitude), np.clip(magnitude / bound, 0, 1)], axis=-1)
    return np.rint(cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB) * 255).astype(np.uint8)


def read_video(path):
    """Sample 16 segment midpoints and their successors, including short clips."""
    reader = decord.VideoReader(str(path), num_threads=1)
    if not len(reader):
        raise ValueError(f"Empty video: {path}")
    interval = len(reader) / 16
    indices = (np.arange(16) * interval + interval / 2).astype(int)
    next_indices = np.minimum(indices + 1, len(reader) - 1)
    frames = {
        int(i): cv2.resize(reader[int(i)].asnumpy(), (224, 224),
                           interpolation=cv2.INTER_LINEAR)
        for i in np.unique(np.concatenate([indices, next_indices]))
    }
    rgb = np.stack([frames[int(i)] for i in indices])
    flows = {int(i): flow_to_rgb(
        dense_flow(frames[int(i)], frames[min(int(i) + 1, len(reader) - 1)]), 20.0)
        for i in np.unique(indices)}
    flow_rgb = np.stack([flows[int(i)] for i in indices])
    mean = np.asarray([123.675, 116.28, 103.53], dtype=np.float32).astype(np.float64)
    # Match mmcv.imnormalize_'s float32 OpenCV arithmetic exactly.
    inverse_std = 1 / np.asarray([58.395, 57.12, 57.375], dtype=np.float32).astype(np.float64)

    def normalize(images):
        images = images.astype(np.float32)
        for frame in images:
            cv2.subtract(frame, mean.reshape(1, -1), frame)
            cv2.multiply(frame, inverse_std.reshape(1, -1), frame)
        return images.transpose(0, 3, 1, 2)

    return torch.from_numpy(np.ascontiguousarray(
        np.concatenate([normalize(rgb), normalize(flow_rgb)], axis=1)))
