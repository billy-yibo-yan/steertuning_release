"""A recording as the pipeline reads it.

Two sources give the same view of a recording: a demonstration from the live stream, and a recording
processed offline by `aria2mano` from VRS and MPS. Every array below has one row per kept frame, in
world coordinates. A frame is left out

    from the stream:  when a head pose or one of its three images is missing, or when the hand sample
                      nearest in time is farther than `HAND_MATCH_TOLERANCE_S`, has no head pose or
                      tracks no hand
    offline:          when MPS tracks neither hand

A recording that cannot give correct data raises `UnusableSession`, and processing excludes it: a frozen
head pose on the stream, or no frame left.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from steertuning.data.representation import transform_points

RAW_DIR = "raw"
ARIA_DATA_FILE = "aria_data.pkl"
HAND_MATCH_TOLERANCE_S = 0.15
MAX_FROZEN_FRACTION = 0.15


class UnusableSession(RuntimeError):
    """The recording cannot give correct training data. Record it again."""


def head_pose_problem(poses: np.ndarray) -> str | None:
    """Why a (T, 4, 4) head pose sequence is unusable, or None. A pose that stops changing is a lost SLAM track."""
    poses = poses[np.isfinite(poses).all(axis=(1, 2))]
    frozen = np.all(np.abs(np.diff(poses, axis=0)) < 1e-12, axis=(1, 2))
    if frozen.size and frozen.mean() > MAX_FROZEN_FRACTION:
        return f"head pose frozen on {frozen.mean() * 100:.0f}% of frames"
    return None


@dataclass
class Recording:
    frames: np.ndarray  # (T,) recording frame index of each kept frame
    head_pose: np.ndarray  # (T, 4, 4) device in world, at the RGB capture
    T_world_depth: np.ndarray  # (T, 4, 4)
    T_world_rgb: np.ndarray  # (T, 4, 4)
    left_landmarks: np.ndarray  # (T, 21, 3), NaN when untracked
    right_landmarks: np.ndarray
    left_wrist: np.ndarray  # (T, 4, 4), NaN when untracked
    right_wrist: np.ndarray
    hand_valid: np.ndarray  # (T, 2), left then right
    K_depth: np.ndarray
    K_rgb: np.ndarray
    T_device_rgb: np.ndarray
    rgb_width: int
    rgb_height: int

    def __len__(self) -> int:
        return len(self.frames)

    def rgb(self, t: int) -> np.ndarray:
        """(H, W, 3) uint8 in R, G, B order."""
        raise NotImplementedError

    def stereo_left(self, t: int) -> np.ndarray:
        """(H, W) uint8 rectified left image, the view of the depth map."""
        raise NotImplementedError

    def depth(self, t: int, models) -> np.ndarray:
        """(H, W) uint16 millimeters, zero where the stereo match is not valid."""
        raise NotImplementedError


def _read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return image


class StreamRecording(Recording):
    """A demonstration written by `stream_recorder.py`. Depth comes from S2M2 here."""

    def __init__(self, directory: Path) -> None:
        raw = directory / RAW_DIR
        frames = dict(np.load(raw / "frames.npz"))
        hands = dict(np.load(raw / "hands.npz"))
        calib = json.loads((raw / "calib.json").read_text())
        if problem := head_pose_problem(frames["T_odom_device_rgb"]):
            raise UnusableSession(f"{directory.name}: {problem}")

        self.processed = directory / "processed"
        self.focal_length = float(calib["focal_length"])
        self.baseline = float(calib["baseline"])

        matched = _match_hands(frames, hands)
        T_stereo, T_rgb = frames["T_odom_device_stereo"], frames["T_odom_device_rgb"]
        has_pose = np.isfinite(T_stereo).all(axis=(1, 2)) & np.isfinite(T_rgb).all(axis=(1, 2))
        valid = np.zeros((len(matched), 2))
        sampled = matched >= 0
        if sampled.any():
            sample = matched[sampled]
            posed = np.isfinite(hands["T_odom_device"][sample]).all(axis=(1, 2))
            tracked = np.stack([hands["left_valid"][sample], hands["right_valid"][sample]], axis=1) > 0
            valid[sampled] = tracked & posed[:, None]
        kept = np.flatnonzero(has_pose & (valid.sum(axis=1) > 0) & self._images_exist(len(matched)))
        if len(kept) == 0:
            raise UnusableSession(f"{directory.name}: no frame has a head pose, images and a tracked hand")

        sample, hand_valid = matched[kept], valid[kept]
        T_hands = hands["T_odom_device"][sample]
        left_landmarks, left_wrist = _world_hand(hands, "left", sample, T_hands, hand_valid[:, 0] > 0)
        right_landmarks, right_wrist = _world_hand(hands, "right", sample, T_hands, hand_valid[:, 1] > 0)
        T_device_rgb = np.asarray(calib["T_device_rgb"], dtype=np.float64)
        super().__init__(
            frames=kept,
            head_pose=T_rgb[kept],
            T_world_depth=T_stereo[kept] @ np.asarray(calib["T_device_rectleft"], dtype=np.float64),
            T_world_rgb=T_rgb[kept] @ T_device_rgb,
            left_landmarks=left_landmarks,
            right_landmarks=right_landmarks,
            left_wrist=left_wrist,
            right_wrist=right_wrist,
            hand_valid=hand_valid.astype(np.float32),
            K_depth=np.asarray(calib["K_stereo"], dtype=np.float64),
            K_rgb=np.asarray(calib["K_rgb"], dtype=np.float64),
            T_device_rgb=T_device_rgb,
            rgb_width=int(calib["rgb_w"]),
            rgb_height=int(calib["rgb_h"]),
        )

    def _images_exist(self, count: int) -> np.ndarray:
        paths = [("rgb", ".jpg"), ("stereo_left", ".png"), ("stereo_right", ".jpg")]
        return np.array(
            [all((self.processed / name / f"{i:06d}{ext}").is_file() for name, ext in paths) for i in range(count)]
        )

    def rgb(self, t: int) -> np.ndarray:
        # The publisher gives the camera's RGB array to OpenCV's encoder, so OpenCV decodes it back to R, G, B.
        return _read_image(self.processed / "rgb" / f"{self.frames[t]:06d}.jpg", cv2.IMREAD_COLOR)

    def stereo_left(self, t: int) -> np.ndarray:
        return _read_image(self.processed / "stereo_left" / f"{self.frames[t]:06d}.png", cv2.IMREAD_GRAYSCALE)

    def depth(self, t: int, models) -> np.ndarray:
        right = _read_image(self.processed / "stereo_right" / f"{self.frames[t]:06d}.jpg", cv2.IMREAD_GRAYSCALE)
        depth_mm, valid = models.depth(self.stereo_left(t), right, self.focal_length, self.baseline)
        return np.where(valid > 0, depth_mm, 0)


class AriaRecording(Recording):
    """A recording processed offline by `aria2mano`, with MPS poses and hands and stored depth."""

    def __init__(self, directory: Path) -> None:
        from aria2mano.aria_loader import AriaLoader

        self.loader = AriaLoader(directory)
        self.processed = directory / "processed"
        rows = {key: [] for key in ("kept", "head", "T_depth", "T_rgb", "ll", "rl", "lw", "rw", "valid")}
        for i in range(self.loader.get_num_frames()):
            frame = self.loader.get_frame_data(i)
            tracking = frame.mps_hand_tracking
            left = tracking.left if tracking is not None else None
            right = tracking.right if tracking is not None else None
            if left is None and right is None:
                continue
            T_world_device = np.asarray(frame.T_world_device, dtype=np.float64)
            rows["kept"].append(i)
            rows["head"].append(T_world_device)
            rows["T_depth"].append(T_world_device @ frame.stereo_left_camera.T_device_camera)
            rows["T_rgb"].append(T_world_device @ frame.rgb_camera.T_device_camera)
            for hand, landmarks, wrist in ((left, "ll", "lw"), (right, "rl", "rw")):
                rows[landmarks].append(
                    np.full((21, 3), np.nan)
                    if hand is None
                    else transform_points(np.asarray(hand.landmarks, dtype=np.float64), T_world_device)
                )
                rows[wrist].append(np.full((4, 4), np.nan) if hand is None else T_world_device @ hand.T_device_wrist)
            rows["valid"].append([float(left is not None), float(right is not None)])
        if not rows["kept"]:
            raise UnusableSession(f"{directory.name}: no frame has a tracked hand")

        first = self.loader.get_frame_data(rows["kept"][0])
        super().__init__(
            frames=np.array(rows["kept"]),
            head_pose=np.stack(rows["head"]),
            T_world_depth=np.stack(rows["T_depth"]),
            T_world_rgb=np.stack(rows["T_rgb"]),
            left_landmarks=np.stack(rows["ll"]),
            right_landmarks=np.stack(rows["rl"]),
            left_wrist=np.stack(rows["lw"]),
            right_wrist=np.stack(rows["rw"]),
            hand_valid=np.asarray(rows["valid"], dtype=np.float32),
            K_depth=np.asarray(first.stereo_left_camera.K, dtype=np.float64),
            K_rgb=np.asarray(first.rgb_camera.K, dtype=np.float64),
            T_device_rgb=np.asarray(first.rgb_camera.T_device_camera, dtype=np.float64),
            rgb_width=int(first.rgb_camera.width),
            rgb_height=int(first.rgb_camera.height),
        )

    def rgb(self, t: int) -> np.ndarray:
        return self.loader.get_rgb_image(int(self.frames[t]))

    def stereo_left(self, t: int) -> np.ndarray:
        return _read_image(self.processed / "stereo_left" / f"{self.frames[t]:08d}.png", cv2.IMREAD_GRAYSCALE)

    def depth(self, t: int, models) -> np.ndarray:
        depth_mm = _read_image(self.processed / "depth" / f"{self.frames[t]:08d}.png", cv2.IMREAD_UNCHANGED)
        valid = _read_image(self.processed / "depth_mask" / f"{self.frames[t]:08d}.png", cv2.IMREAD_GRAYSCALE)
        return np.where(valid > 0, depth_mm, 0)


def is_recording(directory: Path) -> bool:
    return (directory / RAW_DIR / "frames.npz").is_file() or (directory / "processed" / ARIA_DATA_FILE).is_file()


def open_recording(directory: str | Path) -> Recording:
    """Either source, by what the directory holds."""
    directory = Path(directory)
    if (directory / RAW_DIR / "frames.npz").is_file():
        return StreamRecording(directory)
    if (directory / "processed" / ARIA_DATA_FILE).is_file():
        return AriaRecording(directory)
    raise FileNotFoundError(f"{directory} holds neither {RAW_DIR}/frames.npz nor processed/{ARIA_DATA_FILE}")


def _match_hands(frames: dict, hands: dict) -> np.ndarray:
    """The nearest hand sample per image by publish time, -1 when none lies within the tolerance."""
    if len(hands.get("t_pub", ())) == 0:
        return np.full(len(frames["t_pub"]), -1)
    image_times, hand_times = frames["t_pub"].astype(np.float64), hands["t_pub"].astype(np.float64)
    order = np.argsort(hand_times, kind="stable")
    sorted_times = hand_times[order]
    after = np.searchsorted(sorted_times, image_times).clip(0, len(order) - 1)
    before = (after - 1).clip(0, len(order) - 1)
    use_before = np.abs(sorted_times[before] - image_times) <= np.abs(sorted_times[after] - image_times)
    nearest = np.where(use_before, before, after)
    return np.where(np.abs(sorted_times[nearest] - image_times) <= HAND_MATCH_TOLERANCE_S, order[nearest], -1)


def _world_hand(
    hands: dict, side: str, sample: np.ndarray, T: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    landmarks = transform_points(hands[f"{side}_landmarks"][sample].astype(np.float64), T[:, None])
    wrist = T @ hands[f"{side}_wrist"][sample].astype(np.float64)
    return np.where(valid[:, None, None], landmarks, np.nan), np.where(valid[:, None, None], wrist, np.nan)
