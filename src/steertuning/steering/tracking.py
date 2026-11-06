"""Progress through the selected demonstration."""

from __future__ import annotations

import numpy as np

from steertuning.data.representation import AXIS_TIPS, FINGERTIPS, LEFT, RIGHT, WRIST_INDEX, chunk_frames

WINDOW = 15  # frames searched on either side of the cursor
SHAPE_WEIGHT = 4.0

_SHAPE_POINTS = np.concatenate([FINGERTIPS, AXIS_TIPS])


def hand_state(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrists (..., 2, 3) and hand shapes (..., 2, 8, 3) from (..., 52, 3).

    A hand shape is the fingertips and the wrist axis tips, relative to the wrist.
    """
    hands = np.stack([points[..., LEFT, :], points[..., RIGHT, :]], axis=-3)
    wrists = hands[..., WRIST_INDEX, :]
    return wrists, hands[..., _SHAPE_POINTS, :] - wrists[..., None, :]


class ProgressTracker:
    """One cursor for both hands and the head, at the closest corresponding wrist and hand motion.

    The cursor starts at frame 0. Each update moves it to the best frame within `WINDOW` frames on either
    side of it, capped at the trajectory's ends. A frame scores its wrist distance plus 4 times its
    hand-shape distance, summed over the hands, and the later frame wins a tie. Every frame is scored,
    so a demonstration's untracked frames need values: `fill_untracked` supplies them.
    """

    def __init__(self, trajectory_in_camera: np.ndarray) -> None:
        self.trajectory = np.asarray(trajectory_in_camera)
        self.length = len(self.trajectory)
        self.wrists, self.shapes = hand_state(self.trajectory)
        self.cursor = 0

    def update(self, wrists: np.ndarray, hand_shape: np.ndarray) -> int:
        """Move the cursor and return it.

        Args:
            wrists: wrists (2, 3) in the same camera frame as the trajectory.
            hand_shape: hand shapes (2, 8, 3), see `hand_state`.
        """
        latest_first = np.arange(min(self.cursor + WINDOW, self.length - 1), max(self.cursor - WINDOW, 0) - 1, -1)
        self.cursor = int(latest_first[np.argmin(self._score(latest_first, wrists, hand_shape))])
        return self.cursor

    def _score(self, frames: np.ndarray, wrists: np.ndarray, hand_shape: np.ndarray) -> np.ndarray:
        wrist_cost = np.linalg.norm(self.wrists[frames] - wrists, axis=-1)
        shape_cost = np.linalg.norm(self.shapes[frames] - hand_shape, axis=-1).mean(axis=-1)
        return (wrist_cost + SHAPE_WEIGHT * shape_cost).sum(axis=-1)

    def steering_chunk(self) -> np.ndarray:
        """The steering action in meters, (15, 52, 3): frames cursor + 1 + 3 k, holding the last."""
        return self.trajectory[np.minimum(chunk_frames(self.cursor), self.length - 1)]


class BlendedTracker:
    """The weighted average of several demonstrations' steering actions, one cursor per demonstration."""

    def __init__(self, trackers: list[ProgressTracker], weights: np.ndarray) -> None:
        self.trackers = trackers
        self.weights = weights

    def update(self, wrists: np.ndarray, hand_shape: np.ndarray) -> None:
        for tracker in self.trackers:
            tracker.update(wrists, hand_shape)

    def steering_chunk(self) -> np.ndarray:
        return sum(w * tracker.steering_chunk() for w, tracker in zip(self.weights, self.trackers))
