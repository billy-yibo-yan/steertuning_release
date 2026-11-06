"""The demonstration bank of one task: trajectories and object layouts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from steertuning.data.representation import LEFT, RIGHT
from steertuning.data.session import Session, session_directories, session_name
from steertuning.steering.layout import ObjectLayout, extract_layout

log = logging.getLogger(__name__)


def fill_untracked(trajectory: np.ndarray, hand_valid: np.ndarray) -> np.ndarray:
    """Interpolate each hand across its untracked frames, holding the ends."""
    filled = trajectory.copy()
    frames = np.arange(len(trajectory))
    for hand, points, side in ((0, LEFT, "left"), (1, RIGHT, "right")):
        valid = hand_valid[:, hand]
        if valid.all():
            continue
        if not valid.any():
            raise ValueError(f"The {side} hand is never tracked, so the robot has no motion of it to follow.")
        block = filled[:, points].reshape(len(frames), -1)
        for dim in range(block.shape[1]):
            block[~valid, dim] = np.interp(frames[~valid], frames[valid], block[valid, dim])
        filled[:, points] = block.reshape(len(frames), -1, 3)
    return filled


@dataclass
class Demonstration:
    """One processed human demonstration.

    name: session name, see `session_name`.
    trajectory: (T, 52, 3) action points in meters, in the demonstration's frame-0 camera frame, with
        untracked frames from `fill_untracked`.
    layout: object layout at frame 0.
    initial_context: mean-pooled context latent at frame 0, used by the VINN selection only.
    """

    name: str
    trajectory: np.ndarray
    layout: ObjectLayout
    initial_context: np.ndarray | None = None


def load_demonstration(directory: Path, name: str) -> Demonstration:
    """Build a demonstration from a preprocessed session directory."""
    session = Session(directory)
    T_camera_world = session.camera_from_world(0)
    trajectory = session.action_points(np.arange(len(session)), T_camera_world).astype(np.float64)
    trajectory = fill_untracked(trajectory, np.asarray(session.hand_valid) > 0.5)
    gravity_up = T_camera_world[:3, :3] @ np.array([0.0, 0.0, 1.0])
    layout = extract_layout(session.cloud(0, T_camera_world)[:, :3], gravity_up)
    return Demonstration(name, trajectory, layout)


def task_sessions(task_dir: str | Path) -> list[Path]:
    """The preprocessed session directories of a task, in numeric order of their names within a folder.

    With a split file these are the sessions it lists under train, otherwise every session.
    """
    directories = session_directories(task_dir)
    return sorted(directories, key=lambda d: (d.parent.parent, len(d.parent.name), d.parent.name))


class DemoBank:
    """The demonstrations a robot can be steered with for one task."""

    def __init__(self, demonstrations: list[Demonstration]) -> None:
        self.demonstrations = demonstrations

    def __len__(self) -> int:
        return len(self.demonstrations)

    def __getitem__(self, index: int) -> Demonstration:
        return self.demonstrations[index]

    def __iter__(self):
        return iter(self.demonstrations)

    def without(self, folder: str) -> DemoBank:
        """The bank without the demonstrations recorded in a session folder of this name, at any depth."""
        return DemoBank([demo for demo in self.demonstrations if Path(demo.name).name != folder])

    @classmethod
    def build(cls, task_dir: str | Path) -> DemoBank:
        """Read a task's sessions. A session with no surface in view or a hand never tracked is left out."""
        demonstrations = []
        for directory in task_sessions(task_dir):
            name = session_name(directory, task_dir)
            try:
                demonstrations.append(load_demonstration(directory, name))
            except ValueError as error:
                log.warning("Leaving %s out of the demonstration bank: %s", name, error)
        if not demonstrations:
            raise ValueError(f"No usable demonstration under {task_dir}.")
        return cls(demonstrations)
