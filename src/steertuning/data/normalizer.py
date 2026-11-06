"""Action normalization shared by pretraining, fine-tuning and deployment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from steertuning.data.representation import chunk_frames
from steertuning.data.session import NO_WINDOWS_HINT, Session, num_anchors, read_segments, session_directories


@dataclass
class ActionNormalizer:
    """Per-axis map, one scale for all points: y = clip(x, low, high) * scale + offset.

    Fine-tuning and deployment load the stats from pretraining, so the denoiser keeps its
    coordinate scale. Works on arrays of shape (..., 3); `denormalize` takes torch tensors too.
    """

    scale: np.ndarray
    offset: np.ndarray
    clip_low: np.ndarray
    clip_high: np.ndarray

    def __post_init__(self) -> None:
        for name in ("scale", "offset", "clip_low", "clip_high"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.clip_low, self.clip_high) * self.scale + self.offset

    def denormalize(self, y: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(y, torch.Tensor):
            scale, offset = (torch.as_tensor(v, dtype=y.dtype, device=y.device) for v in (self.scale, self.offset))
            return (y - offset) / scale
        return (y - self.offset) / self.scale

    @classmethod
    def fit(cls, points: np.ndarray, quantiles: tuple[float, float] = (0.001, 0.999)) -> ActionNormalizer:
        """Map the quantile range of (N, 3) points to [-1, 1]. Outliers are SLAM jumps, hence the clip."""
        clip_low = np.quantile(points, quantiles[0], axis=0).astype(np.float64)
        clip_high = np.quantile(points, quantiles[1], axis=0).astype(np.float64)
        span = clip_high - clip_low
        constant = span < 1e-4
        scale = 2.0 / np.where(constant, 2.0, span)
        offset = np.where(constant, -clip_low, -1.0 - scale * clip_low)
        return cls(scale, offset, clip_low, clip_high)

    @classmethod
    def from_json(cls, path: str | Path) -> ActionNormalizer:
        with open(path) as f:
            stats = json.load(f)
        return cls(stats["scale"], stats["offset"], stats["clip_low"], stats["clip_high"])

    def to_json(self, path: str | Path) -> None:
        stats = {
            "scale": self.scale.tolist(),
            "offset": self.offset.tolist(),
            "clip_low": self.clip_low.tolist(),
            "clip_high": self.clip_high.tolist(),
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)


def fit_normalizer(root: str | Path) -> ActionNormalizer:
    """Fit the normalizer on the valid action points of every window in the training split.

    A window contributes its anchor and its targets, in its anchor frame. Segment labels play no part.
    """
    pool = []
    for directory in session_directories(root):
        if not read_segments(directory):
            continue
        session = Session(directory)
        for anchor in range(num_anchors(len(session))):
            frames = np.concatenate([[anchor], chunk_frames(anchor)])
            pool.append(session.valid_action_points(frames, session.camera_from_world(anchor)).astype(np.float32))
    if not pool:
        raise RuntimeError(f"No windows under {root} for its training split. {NO_WINDOWS_HINT}")
    return ActionNormalizer.fit(np.concatenate(pool))
