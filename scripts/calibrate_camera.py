"""Find the head camera's pose on the robot from an ArUco marker on the left end-effector mount.

    python scripts/calibrate_camera.py --robot-host <ip> --camera-host <ip>

Pose the arm with your own controller and press Enter per sample. This script never moves the arm.
The calibration goes to --output, by default the place configs/deploy.yaml reads it from.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from steertuning import ask
from steertuning.deployment.calibration import detect_marker, marker_points, save_calibration, solve_camera_pose
from steertuning.deployment.observation import AriaCamera
from steertuning.remote import connect_rpc

MAX_DRIFT = 0.004  # rad of joint motion allowed while an image is taken
DEFAULT_OUTPUT = Path(os.environ.get("STEERTUNING_HOME") or "./data") / "calibration" / "camera_from_base.npz"


class Rig:
    """The left arm and the RGB stream."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.arm = connect_rpc(args.robot_host, args.robot_port)
        self.camera = AriaCamera(args.camera_host, args.camera_port)
        self.K = np.asarray(self.camera.calibration.read()["K_rgb"], dtype=np.float64)
        self.marker_id = args.marker_id

    def joints(self) -> np.ndarray:
        return np.asarray(self.arm.get_left_joint_positions(), dtype=np.float64)

    def rgb(self) -> np.ndarray:
        for _ in range(3):  # let frames taken during the motion pass
            message = self.camera.receive()
            time.sleep(0.12)
        return self.camera.rgb(message)

    def sample(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Joints and marker corners of the still arm, None if it moved or the marker is not visible."""
        before = self.joints()
        corners = detect_marker(self.rgb(), self.marker_id)
        after = self.joints()
        if corners is None or np.abs(after - before).max() > MAX_DRIFT:
            return None
        return after, corners


def collect(rig: Rig) -> tuple[list, list]:
    joints, corners = [], []
    print("Pose the arm with the marker in view, stop, press Enter. Type q when done.")
    while ask(f"[{len(joints)} samples] ") != "q":
        sample = rig.sample()
        if sample is None:
            print("Skipped: the arm moved or the marker is not visible.")
            continue
        joints.append(sample[0])
        corners.append(sample[1])
    return joints, corners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-host", required=True)
    parser.add_argument("--robot-port", type=int, default=5557)
    parser.add_argument("--camera-host", required=True)
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--marker-id", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="default: %(default)s")
    args = parser.parse_args()

    rig = Rig(args)
    joints, corners = collect(rig)
    T, errors, inliers = solve_camera_pose(marker_points(np.array(joints)), np.array(corners), rig.K)
    save_calibration(args.output, T, rig.K, errors, inliers)
    mean_error = errors[inliers].mean()
    print(f"Saved {args.output}: {inliers.sum()}/{len(inliers)} poses, mean inlier error {mean_error:.2f} px")


if __name__ == "__main__":
    main()
