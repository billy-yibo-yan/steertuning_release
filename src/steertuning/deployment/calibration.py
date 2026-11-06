"""The head camera's pose on the robot, from an ArUco marker on the end-effector mount.

The marker's corners are known in the base frame through forward kinematics and seen in the rectified
RGB image. Pose-level RANSAC with PnP, then Levenberg-Marquardt on the inliers, gives T_camera_base.
Any movement of the glasses on their mount invalidates the result.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from steertuning.deployment.kinematics import ARM_JOINTS, MJCF_PATH

# In OpenCV's corner order for a decoded marker: top left, top right, bottom right, bottom left.
CORNER_SITES = ["left_R_U", "left_L_U", "left_L_B", "left_R_B"]
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_250
REPROJECTION_THRESHOLD = 3.0  # pixels
MIN_POSES = 6
MIN_INLIER_FRACTION = 0.6
RANSAC_ITERATIONS = 160
MIN_SPREAD = 0.01  # meters, second singular value of the marker centers


def detect_marker(rgb: np.ndarray, marker_id: int) -> np.ndarray | None:
    """Corners (4, 2) of one marker in an RGB image, None unless found exactly once."""
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    matches = [] if ids is None else np.flatnonzero(ids.ravel() == marker_id)
    return corners[int(matches[0])].reshape(4, 2) if len(matches) == 1 else None


def marker_points(joint_rows: np.ndarray) -> np.ndarray:
    """Marker corners (N, 4, 3) in the base frame, for left-arm joints (N, 7)."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    addresses = model.jnt_qposadr[[model.joint(name).id for name in ARM_JOINTS["left"]]]
    sites = [model.site(name).id for name in CORNER_SITES]
    result = np.empty((len(joint_rows), 4, 3))
    for i, q in enumerate(np.asarray(joint_rows, dtype=np.float64)):
        data.qpos[addresses] = q
        mujoco.mj_kinematics(model, data)
        result[i] = data.site_xpos[sites]
    return result


def solve_camera_pose(xyz: np.ndarray, uv: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T_camera_base from corners xyz (N, 4, 3) and uv (N, 4, 2) on a rectified image.

    Returns the transform, each pose's mean reprojection error in pixels, and the inlier mask.
    Raises ValueError for too few or too similar poses, or when under 60% of them agree.
    """
    if len(xyz) < MIN_POSES:
        raise ValueError(f"Need at least {MIN_POSES} marker poses, got {len(xyz)}.")
    centers = xyz.mean(axis=1)
    if np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)[1] < MIN_SPREAD:
        raise ValueError("The marker poses are too similar. Move the arm over a wider area.")
    no_distortion = np.zeros(5)
    needed = max(MIN_POSES, int(np.ceil(MIN_INLIER_FRACTION * len(xyz))))

    def errors(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        projected = cv2.projectPoints(xyz.reshape(-1, 3), rvec, tvec, K, no_distortion)[0].reshape(uv.shape)
        return np.linalg.norm(projected - uv, axis=2).mean(axis=1)

    rng = np.random.default_rng(0)
    best, best_score = None, (-1, -np.inf)
    for _ in range(RANSAC_ITERATIONS):
        chosen = rng.choice(len(xyz), MIN_POSES, replace=False)
        found, rvec, tvec = cv2.solvePnP(
            xyz[chosen].reshape(-1, 3), uv[chosen].reshape(-1, 2), K, no_distortion, flags=cv2.SOLVEPNP_EPNP
        )
        if not found:
            continue
        pose_errors = errors(rvec, tvec)
        inliers = pose_errors < REPROJECTION_THRESHOLD
        score = (int(inliers.sum()), -float(np.median(pose_errors)))
        if score > best_score:
            best, best_score = (rvec, tvec, inliers), score
    if best is None or best_score[0] < needed:
        raise ValueError("No camera pose agrees with 60% of the marker poses.")
    rvec, tvec, inliers = best
    for _ in range(10):
        rvec, tvec = cv2.solvePnPRefineLM(
            xyz[inliers].reshape(-1, 3), uv[inliers].reshape(-1, 2), K, no_distortion, rvec, tvec
        )
        pose_errors = errors(rvec, tvec)
        updated = pose_errors < REPROJECTION_THRESHOLD
        if updated.sum() < needed:
            raise ValueError("The fit lost too many marker poses during refinement.")
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return T, pose_errors, inliers


def save_calibration(
    path: str | Path, T_camera_base: np.ndarray, K: np.ndarray, errors: np.ndarray, inliers: np.ndarray
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, T_camera_base=T_camera_base, K=K, pose_errors_px=errors, inliers=inliers)


def load_calibration(path: str | Path) -> np.ndarray:
    """T_camera_base (4, 4) from `save_calibration`'s file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No camera calibration at {path}. Run scripts/calibrate_camera.py first.")
    with np.load(path) as calibration:
        return calibration["T_camera_base"]
