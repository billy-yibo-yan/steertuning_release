"""The robot interface and its YOR implementation.

YOR runs its own driver with an RPC server on the robot computer. This client
sends joint targets to it and to one server per Wuji hand (scripts/wuji_server.py).
"""

from __future__ import annotations

import threading
import time
from typing import Protocol

import numpy as np

from steertuning.remote import connect_rpc

ARM_HOME = {
    "left": np.array([0.256, 1.512, -1.425, 2.011, -0.221, 0.107, -0.091]),
    "right": np.array([-0.256, 1.512, 1.425, 2.011, 0.221, -0.107, -0.091]),
}
HAND_HOME = np.array(  # open hand, thumb to pinky, knuckle to tip
    [
        [0.585, 0.581, 0.101, 0.179],
        [0.174, 0.000, 0.158, 0.024],
        [0.160, 0.002, 0.005, 0.033],
        [0.164, 0.000, 0.159, 0.046],
        [0.182, -0.001, 0.135, 0.170],
    ]
)
RESET_MIN_SECONDS = 2.0
RESET_MAX_SPEED = 1.0  # rad/s
RESET_RATE = 30.0  # commands per second
RESET_ARM_DELAY = 0.15  # fraction of the reset that passes before the arms move
RESET_HAND_FRACTION = 0.3  # fraction of the reset the hands take to open
RESET_SETTLE = 0.5  # seconds between the last command and the check that the arms arrived
HOME_TOLERANCE = 0.3  # radians from ARM_HOME, per joint. YOR reads zeros from an arm that is down.


class Robot(Protocol):
    """What the control loop needs from a robot."""

    def reset(self) -> None:
        """Move to the home pose and return when there. Raises RuntimeError if an arm did not arrive."""

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        """Measured joints: left_arm, right_arm (7,), left_hand, right_hand (5, 4)."""

    def command(
        self, arm_targets: dict[str, np.ndarray] | None = None, hand_targets: dict[str, np.ndarray] | None = None
    ) -> None:
        """Send targets keyed by "left" and "right". Omitted arms and hands keep their last target."""


class YorRobot:
    """YOR arms and two Wuji hands over commlink RPC. Calls from several threads are serialized."""

    def __init__(self, host: str, port: int, hand_host: str | None, left_hand_port: int, right_hand_port: int) -> None:
        """Connects to the YOR driver and the hand servers. `hand_host` None means they run on `host`."""
        self.arms = connect_rpc(host, port)
        self.hands = {
            "left": connect_rpc(hand_host or host, left_hand_port),
            "right": connect_rpc(hand_host or host, right_hand_port),
        }
        self._lock = threading.Lock()
        self.get_joint_positions()  # raises here when a server is down

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {
                "left_arm": np.asarray(self.arms.get_left_joint_positions(), dtype=np.float64),
                "right_arm": np.asarray(self.arms.get_right_joint_positions(), dtype=np.float64),
                "left_hand": np.asarray(self.hands["left"].current_angles, dtype=np.float64).reshape(5, 4),
                "right_hand": np.asarray(self.hands["right"].current_angles, dtype=np.float64).reshape(5, 4),
            }

    def command(
        self, arm_targets: dict[str, np.ndarray] | None = None, hand_targets: dict[str, np.ndarray] | None = None
    ) -> None:
        with self._lock:
            for side, q in (arm_targets or {}).items():
                getattr(self.arms, f"set_{side}_joint_target")(np.asarray(q, dtype=np.float64))
            for side, q in (hand_targets or {}).items():
                self.hands[side].target_angles = np.asarray(q, dtype=np.float64)

    def reset(self) -> None:
        """Open the hands, bring the arms home at no more than 1 rad/s, and check that they arrived."""
        start = self.get_joint_positions()
        travel = max(np.abs(start[f"{side}_arm"] - ARM_HOME[side]).max() for side in ARM_HOME)
        duration = max(RESET_MIN_SECONDS, travel / RESET_MAX_SPEED / (1.0 - RESET_ARM_DELAY))
        steps = int(duration * RESET_RATE)
        for step in range(1, steps + 1):
            progress = step / steps
            hand = min(1.0, progress / RESET_HAND_FRACTION)
            arm = max(0.0, (progress - RESET_ARM_DELAY) / (1.0 - RESET_ARM_DELAY))
            self.command(
                arm_targets={s: (1 - arm) * start[f"{s}_arm"] + arm * ARM_HOME[s] for s in ARM_HOME},
                hand_targets={s: (1 - hand) * start[f"{s}_hand"] + hand * HAND_HOME for s in ARM_HOME},
            )
            time.sleep(duration / steps)
        time.sleep(RESET_SETTLE)
        self._check_home()

    def _check_home(self) -> None:
        joints = self.get_joint_positions()
        for side, home in ARM_HOME.items():
            distance = np.abs(joints[f"{side}_arm"] - home).max()
            if distance > HOME_TOLERANCE:
                raise RuntimeError(
                    f"The {side} arm did not reach home: a joint is {distance:.2f} rad away. "
                    "Check that the arm follows commands, and YOR's gravity compensation."
                )
