"""Kinematic retargeting of predicted landmarks to the Wuji hands.

Uses Wuji Technology's `wuji_retargeting` with the hand models in assets/robot/wuji_urdf. Each hand's
landmarks are taken relative to its own palm, so the fingers follow the prediction relative to the wrist.
"""

from __future__ import annotations

import copy

import numpy as np

from steertuning.deployment.kinematics import MJCF_PATH, SIDES

URDF_DIR = MJCF_PATH.parent / "wuji_urdf"

# Index of each MediaPipe landmark in the 21 tracked Aria landmarks. MediaPipe point 1 has no direct
# counterpart; `mediapipe_landmarks` places it halfway between the wrist and the first thumb point.
ARIA_TO_MEDIAPIPE = np.array([5, 5, 6, 7, 0, 8, 9, 10, 1, 11, 12, 13, 2, 14, 15, 16, 3, 17, 18, 19, 4])


def mediapipe_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Reorder (21, 3) Aria hand landmarks to the MediaPipe order the retargeter expects."""
    reordered = landmarks[ARIA_TO_MEDIAPIPE].copy()
    reordered[1] = (landmarks[5] + landmarks[6]) / 2.0
    return reordered


class HandRetargeter:
    """Both hands' retargeters. See `retargeting` in configs/deploy.yaml for the settings."""

    def __init__(self, config: dict) -> None:
        from wuji_retargeting import Retargeter

        self.hands = {}
        for side in SIDES:
            hand_config = copy.deepcopy(config)
            hand_config.setdefault("optimizer", {})["urdf_path"] = str(URDF_DIR / f"{side}.urdf")
            self.hands[side] = Retargeter.from_config(hand_config, hand_side=side)

    def reset(self) -> None:
        """Forget the filters and warm starts, at the start of a rollout."""
        for hand in self.hands.values():
            hand.reset()

    def __call__(self, side: str, landmarks: np.ndarray) -> np.ndarray | None:
        """Joint angles (5, 4) for one hand's (21, 3) landmarks, or None when the solve diverged."""
        q = self.hands[side].retarget(mediapipe_landmarks(landmarks))
        if not np.isfinite(q).all():
            self.hands[side].reset()  # a NaN inside the low-pass filter would stay there
            return None
        return q.reshape(5, 4)
