"""Fine-tuning augmentations."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from steertuning.data.representation import transform_points

ANCHOR_TRANSLATION = 0.05  # meters
ANCHOR_ROTATION = 5.0  # degrees
PROPRIO_TRANSLATION = 0.03
PROPRIO_ROTATION = 5.0


def sample_rigid(rng: np.random.RandomState, translation: float, rotation: float) -> np.ndarray:
    """A 4x4 rigid transform: uniform translation per axis (meters), uniform Euler angles (degrees)."""
    delta = np.eye(4)
    delta[:3, 3] = rng.uniform(-translation, translation, size=3)
    angles = rng.uniform(-rotation, rotation, size=3)
    delta[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    return delta


def perturb_fingertips(tips: np.ndarray, wrists: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Move each hand's (5, 3) fingertips by its own rigid transform about the wrist.

    The transform is sampled in the wrist's axes. Targets stay as recorded, so only the
    proprioception sees it. A hand with no wrist pose stays where it is.
    """
    moved = tips.copy()
    for hand in range(2):
        delta = sample_rigid(rng, PROPRIO_TRANSLATION, PROPRIO_ROTATION)
        if np.isfinite(wrists[hand]).all():
            moved[hand] = transform_points(tips[hand], wrists[hand] @ delta @ np.linalg.inv(wrists[hand]))
    return moved


def jitter_rgb(cloud: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Jitter the colors of an (N, 6) cloud, rgb 0-255: brightness, contrast, saturation, hue, noise.

    One draw of each per cloud. Hue is in OpenCV units, where 180 is a full turn.
    """
    rgb = cloud[:, 3:6] / 255.0
    rgb = rgb * rng.uniform(0.6, 1.4)
    contrast = rng.uniform(0.7, 1.3)
    mean = rgb.mean(axis=0, keepdims=True)
    rgb = (rgb - mean) * contrast + mean

    saturation = rng.uniform(0.7, 1.3)
    hue = rng.uniform(-18.0, 18.0)
    hsv = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype(np.uint8)[None], cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0.0, 255.0)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)[0].astype(np.float32) / 255.0

    rgb = rgb + rng.normal(0.0, rng.uniform(0.0, 0.02), size=rgb.shape)
    jittered = cloud.copy()
    jittered[:, 3:6] = np.clip(rgb, 0.0, 1.0) * 255.0
    return jittered
