"""Read one preprocessed session directory."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from steertuning.data.representation import (
    CHUNK_STEPS,
    CHUNK_STRIDE,
    MASK_SIZE,
    action_mask,
    action_points,
    camera_from_world,
    chunk_frames,
    transform_points,
    transform_poses,
)

SPLIT_FILE = "train_test_split.json"
PARAPHRASE_FILE = "text_prompt_augmentations.json"
NO_WINDOWS_HINT = (
    "A window needs 4.3 s after its anchor, so a session of 4.3 s or less has none. "
    "Check the session lengths, and that scripts/process.py has run."
)


@dataclass(frozen=True)
class Segment:
    """One annotated snippet. Frame indices are inclusive. In the JSON file `instruction` is `action`."""

    start: int
    end: int
    instruction: str


def session_name(directory: str | Path, root: str | Path) -> str:
    """The name of a preprocessed directory's session: its folder relative to the root, as a split file has it."""
    return str(Path(directory).parent.relative_to(root))


def read_segments(directory: str | Path) -> list[Segment]:
    """The session's language segments, empty when it has none."""
    path = Path(directory) / "text_prompt_segments.json"
    if not path.is_file():
        return []
    with open(path) as f:
        return [Segment(int(s["start"]), int(s["end"]), str(s["action"])) for s in json.load(f)]


def window_first_targets(segment: Segment, length: int) -> np.ndarray:
    """First target frames of a segment's windows: in the segment, with the whole chunk in the session."""
    stop = min(length - (CHUNK_STEPS - 1) * CHUNK_STRIDE, segment.end + 1)
    return np.arange(max(segment.start, 1), stop)


def num_frames(directory: str | Path) -> int:
    """Session length, from an array header."""
    return len(np.load(Path(directory) / "left_landmarks.npy", mmap_mode="r"))


def num_anchors(length: int) -> int:
    """How many frames of a session can anchor a window: its last target has to be in the session."""
    return max(0, length - int(chunk_frames(0)[-1]))


def read_paraphrases(path: str | Path) -> dict[str, list[str]]:
    """Paraphrases by original instruction, empty when the file is missing.

    The file holds `{"original": ..., "variants": [...]}` or `{"by_original": {original: [...]}}`.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path) as f:
        data = json.load(f)
    by_original = dict(data.get("by_original", {}))
    if "original" in data:
        by_original[data["original"]] = data["variants"]
    return {key.strip(): [v.strip() for v in variants if v.strip()] for key, variants in by_original.items()}


def read_split(root: str | Path) -> tuple[list[str], list[str]]:
    """The `train` and `test` entries of a root's split file, both empty when it has none.

    Session ids are relative to the root. Anything under `test` is out of `train`, so listing a session
    under `test` is enough to hold it out.
    """
    path = Path(root) / SPLIT_FILE
    if not path.is_file():
        return [], []
    split = json.loads(path.read_text())
    test = split.get("test", [])
    held_out = set(test)
    train = [session_id for session_id in split.get("train", []) if session_id not in held_out]
    return train, test


def session_directories(root: str | Path) -> list[Path]:
    """The preprocessed directories that train: those under `train` in the split file, from `read_split`.

    Without a split file, every session one, two or three levels down trains.
    """
    root = Path(root)
    if (root / SPLIT_FILE).is_file():
        directories = [root / session_id / "preprocessed" for session_id in read_split(root)[0]]
        return [directory for directory in directories if directory.is_dir()]
    for pattern in ("*/*/*/preprocessed", "*/*/preprocessed", "*/preprocessed"):
        directories = sorted(path for path in root.glob(pattern) if path.is_dir())
        if directories:
            return directories
    return []


class Session:
    """Memory-mapped arrays of one session. Poses and points are in the world frame on disk."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.point_cloud = self._open("point_cloud")  # (T, N, 6): xyz, rgb 0-255
        self.left_landmarks = self._open("left_landmarks")  # (T, 21, 3), NaN when untracked
        self.right_landmarks = self._open("right_landmarks")
        self.left_wrist = self._open("left_wrist")  # (T, 4, 4)
        self.right_wrist = self._open("right_wrist")
        self.head_pose = self._open("head_pose")  # (T, 4, 4), device in world
        self.hand_valid = self._open("hand_valid")  # (T, 2), left then right
        self.masks = self._open("object_mask_160x120")  # (T, 120, 160[, 3]) uint8
        with np.load(self.directory / "rgb_camera_info.npz") as camera:
            self.T_device_camera = camera["T_device_camera"].astype(np.float64)

    def _open(self, name: str) -> np.ndarray:
        return np.load(self.directory / f"{name}.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.left_landmarks)

    def camera_from_world(self, frame: int, delta: np.ndarray | None = None) -> np.ndarray:
        """World-to-RGB-camera transform at a frame."""
        return camera_from_world(np.asarray(self.head_pose[frame]), self.T_device_camera, delta)

    def action_points(self, frames: np.ndarray, T_camera_world: np.ndarray) -> np.ndarray:
        """The (len(frames), 52, 3) actions at the given frames, in one camera frame."""
        return action_points(
            transform_points(self.left_landmarks[frames], T_camera_world),
            transform_points(self.right_landmarks[frames], T_camera_world),
            transform_poses(self.left_wrist[frames], T_camera_world),
            transform_poses(self.right_wrist[frames], T_camera_world),
            transform_poses(self.head_pose[frames], T_camera_world),
        )

    def valid_action_points(self, frames: np.ndarray, T_camera_world: np.ndarray) -> np.ndarray:
        """The (N, 3) points of those actions that `action_mask` marks valid."""
        valid = action_mask(self.hand_valid[frames])[..., ::3] > 0.5
        return self.action_points(frames, T_camera_world)[valid]

    def wrist_poses(self, frame: int, T_camera_world: np.ndarray) -> np.ndarray:
        """(2, 4, 4) wrist poses at a frame in a camera frame, left then right."""
        return transform_poses(np.stack([self.left_wrist[frame], self.right_wrist[frame]]), T_camera_world)

    def cloud(self, frame: int, T_camera_world: np.ndarray) -> np.ndarray:
        """(N, 6) float32 cloud at a frame: xyz in a camera frame, rgb 0-255.

        A frame with no valid depth is stored as zeros and stays zeros, so the crop reads it as empty.
        """
        cloud = np.array(self.point_cloud[frame], dtype=np.float32)
        if cloud.any():
            cloud[:, :3] = transform_points(cloud[:, :3], T_camera_world)
        return cloud

    def object_mask(self, frame: int) -> np.ndarray:
        """Binary (255, 255) uint8 mask of the manipulated object at a frame."""
        mask = np.asarray(self.masks[frame])
        if mask.ndim == 3:
            mask = mask[..., 0]
        rows = np.arange(MASK_SIZE) * mask.shape[0] // MASK_SIZE
        cols = np.arange(MASK_SIZE) * mask.shape[1] // MASK_SIZE
        return (mask[np.ix_(rows, cols)] > 127).astype(np.uint8)


@lru_cache(maxsize=64)
def open_session(directory: str | Path) -> Session:
    """A bounded cache of open sessions.

    Pages touched through a memmap stay in the process's resident memory until the memmap is dropped.
    """
    return Session(directory)
