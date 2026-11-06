"""The control loop without hardware: a recorded session as the camera, joints that track instantly.

The head stays at the recording's first camera pose, as a robot's head does during manipulation, and
the arms are placed under it. The robot therefore runs the prediction, its IK and its retargeting
against a real scene while nothing moves. The recording advances with the rows the loop executes,
so a slow first chunk does not skip it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from steertuning.data.representation import FPS
from steertuning.data.session import Session
from steertuning.deployment.observation import CameraFrame
from steertuning.deployment.robot import ARM_HOME, HAND_HOME
from steertuning.steering.surface import level_rotation

NOMINAL_CAMERA_IN_BASE = np.array([0.06, -0.06, 0.73])  # the mount we calibrated, meters, in the base frame


def nominal_camera_from_base(session: Session) -> np.ndarray:
    """A plausible head mount for replays: T_camera_base from the recording's gravity and view direction."""
    right, forward, up = level_rotation(session.camera_from_world(0)[:3, 2])
    rotation = np.stack([-right, -forward, up], axis=1)  # the arms reach along -y of the base
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rotation, -rotation @ NOMINAL_CAMERA_IN_BASE
    return T


class ReplayRobot:
    """Joints that reach every target at once."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.joints = {
            "left_arm": ARM_HOME["left"].copy(),
            "right_arm": ARM_HOME["right"].copy(),
            "left_hand": HAND_HOME.copy(),
            "right_hand": HAND_HOME.copy(),
        }

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in self.joints.items()}

    def command(
        self, arm_targets: dict[str, np.ndarray] | None = None, hand_targets: dict[str, np.ndarray] | None = None
    ) -> None:
        for side, q in (arm_targets or {}).items():
            self.joints[f"{side}_arm"] = np.array(q, dtype=np.float64)
        for side, q in (hand_targets or {}).items():
            self.joints[f"{side}_hand"] = np.array(q, dtype=np.float64)


class ReplayCamera:
    """Plays a preprocessed session, all frames in its first camera frame.

    with_rgb: read the recording's images, which only the object mask needs.
    clock: seconds of motion so far, which pick the recording's frame.
    """

    def __init__(self, session_dir: str | Path, with_rgb: bool) -> None:
        self.session = Session(session_dir)
        self.with_rgb = with_rgb
        self.clock: Callable[[], float] = lambda: 0.0
        self.T_camera_world = self.session.camera_from_world(0)
        self.images = self.session.directory.parent / "processed" / "rgb"
        kept = self.session.directory / "kept_frame_indices.npy"
        self.source_frames = np.load(kept) if kept.is_file() else np.arange(len(self.session))

    def _rgb(self, frame: int) -> np.ndarray:
        path = self.images / f"{self.source_frames[frame]:06d}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"No image at {path}. To replay without images, pass object_prompt=null.")
        return cv2.imread(str(path), cv2.IMREAD_COLOR)  # stored as the publisher encoded it, so this is RGB

    def receive(self) -> int:
        """The index of the frame the motion has reached."""
        return min(round(self.clock() * FPS), len(self.session) - 1)

    def frame(self, message: int) -> CameraFrame:
        rgb = self._rgb(message) if self.with_rgb else None
        return CameraFrame(self.session.cloud(message, self.T_camera_world), rgb)
