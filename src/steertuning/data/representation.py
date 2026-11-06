"""Observation and action representation.

An action is 52 points in the RGB camera frame of the anchor frame:

    0-20    left hand, 21 tracked landmarks (0-4 fingertips thumb to pinky, 5 wrist)
    21-23   left wrist axis tips
    24-27   the device's origin and axis tips
    28-48   right hand, 21 tracked landmarks
    49-51   right wrist axis tips

A chunk holds 15 such actions at stride 3. The first target is the frame after the anchor.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

NUM_POINTS = 52
HAND_POINTS = 24
HEAD_POINTS = 4
AXIS_LENGTH = 0.03  # meters
CHUNK_STEPS = 15
CHUNK_STRIDE = 3
FPS = 10

LEFT = slice(0, HAND_POINTS)
HEAD = slice(HAND_POINTS, HAND_POINTS + HEAD_POINTS)
RIGHT = slice(HAND_POINTS + HEAD_POINTS, NUM_POINTS)

# Indices within one hand.
FINGERTIPS = np.arange(5)  # thumb to pinky
WRIST_INDEX = 5
AXIS_TIPS = np.array([21, 22, 23])

# Indices within the action.
LEFT_WRIST = LEFT.start + WRIST_INDEX
RIGHT_WRIST = RIGHT.start + WRIST_INDEX
BOTH_FINGERTIPS = np.concatenate([LEFT.start + FINGERTIPS, RIGHT.start + FINGERTIPS])
# One wrist pose as points, per hand: the wrist and its three axis tips, shape (2, 4).
WRIST_POSE_POINTS = np.array([[LEFT_WRIST, *(LEFT.start + AXIS_TIPS)], [RIGHT_WRIST, *(RIGHT.start + AXIS_TIPS)]])

ELLIPSOID_RADII = (0.85, 1.5, 1.5)  # semi-axes along camera x, y, z, meters
MAX_DEPTH = 1.7  # meters
CLOUD_POINTS = 5000
MASK_SIZE = 255


def chunk_frames(anchor: int | np.ndarray) -> np.ndarray:
    """Target frames of the chunk that follows an anchor frame."""
    return anchor + 1 + CHUNK_STRIDE * np.arange(CHUNK_STEPS)


def camera_from_world(
    head_pose: np.ndarray, T_device_camera: np.ndarray, delta: np.ndarray | None = None
) -> np.ndarray:
    """World-to-camera transform, inv(head @ T_device_camera @ delta).

    `delta` moves the camera in its own axes.
    """
    T_world_camera = head_pose @ T_device_camera
    if delta is not None:
        T_world_camera = T_world_camera @ delta
    return np.linalg.inv(T_world_camera)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (..., 3) points, keeping their float type."""
    points = np.asarray(points)
    homogeneous = np.concatenate([points, np.ones((*points.shape[:-1], 1), points.dtype)], axis=-1)
    return np.einsum("...ij,...j->...i", np.asarray(T, points.dtype), homogeneous)[..., :3]


def transform_poses(poses: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (..., 4, 4) poses."""
    return T @ np.asarray(poses)


def _axis_tips(poses: np.ndarray) -> np.ndarray:
    # (..., 4, 4) -> (..., 3, 3): points AXIS_LENGTH along each local axis.
    return poses[..., None, :3, 3] + AXIS_LENGTH * np.swapaxes(poses[..., :3, :3], -1, -2)


def hand_points(landmarks: np.ndarray, wrist_poses: np.ndarray) -> np.ndarray:
    """21 landmarks plus the three wrist axis tips: (..., 21, 3), (..., 4, 4) -> (..., 24, 3)."""
    return np.concatenate([landmarks, _axis_tips(wrist_poses)], axis=-2)


def head_points(head_poses: np.ndarray) -> np.ndarray:
    """The device's origin plus its three axis tips, in the poses' frame: (..., 4, 4) -> (..., 4, 3).

    A chunk's poses are in the anchor camera frame.
    """
    return np.concatenate([head_poses[..., None, :3, 3], _axis_tips(head_poses)], axis=-2)


def action_points(
    left_lm: np.ndarray,
    right_lm: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    head: np.ndarray,
) -> np.ndarray:
    """The (..., 52, 3) action. Untracked hands are NaN on disk and become zeros here."""
    points = [hand_points(left_lm, left_wrist), head_points(head), hand_points(right_lm, right_wrist)]
    return np.nan_to_num(np.concatenate(points, axis=-2), nan=0.0)


def fingertips(points: np.ndarray) -> np.ndarray:
    """Proprioception from actions: (..., 52, 3) -> (..., 2, 5, 3), left then right."""
    return points[..., BOTH_FINGERTIPS, :].reshape(*points.shape[:-2], 2, 5, 3)


def action_mask(hand_valid: np.ndarray) -> np.ndarray:
    """Per-dimension validity, (..., 2) -> (..., 156). The head is valid if either hand is."""
    hand_valid = np.asarray(hand_valid, dtype=np.float32)
    left, right = hand_valid[..., None, 0], hand_valid[..., None, 1]
    point_valid = np.zeros((*hand_valid.shape[:-1], NUM_POINTS), dtype=np.float32)
    point_valid[..., LEFT] = left
    point_valid[..., HEAD] = np.maximum(left, right)
    point_valid[..., RIGHT] = right
    return np.repeat(point_valid, 3, axis=-1)


def in_ellipsoid(xyz: np.ndarray) -> np.ndarray:
    """Points inside the camera-centered workspace ellipsoid, in front of the camera."""
    xyz = np.asarray(xyz, dtype=np.float64)
    inside = ((xyz / np.array(ELLIPSOID_RADII)) ** 2).sum(axis=-1) <= 1.0
    return inside & (xyz[..., 2] > 1e-5) & np.isfinite(xyz).all(axis=-1)


def crop_and_resample(cloud: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Crop an (N, 6) camera-frame cloud to the workspace and resample it to 5000 points.

    Points repeat only when fewer than 5000 remain. An empty crop gives all zeros.
    """
    kept = cloud[in_ellipsoid(cloud[:, :3])]
    if len(kept) == 0:
        return np.zeros((CLOUD_POINTS, cloud.shape[1]), dtype=cloud.dtype)
    return kept[rng.choice(len(kept), size=CLOUD_POINTS, replace=len(kept) < CLOUD_POINTS)]


def cloud_transform() -> Callable[[dict], dict]:
    """The Utonia preprocessing: coordinates stay in meters at the camera origin, voxels are 1 cm."""
    import utonia

    return utonia.transform.default(scale=1.0, apply_z_positive=False, normalize_coord=False)


def utonia_input(cloud: np.ndarray) -> dict:
    """Input of `cloud_transform` from an (N, 6) float32 cloud with rgb 0-255."""
    return {"coord": cloud[:, :3], "color": cloud[:, 3:6], "normal": np.zeros_like(cloud[:, :3])}
