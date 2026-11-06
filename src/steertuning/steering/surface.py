"""The manipulated-surface frame: gravity along +z, the surface at z = 0."""

from __future__ import annotations

import numpy as np

# Region in front of the camera that can hold the manipulated surface, meters.
WORKSPACE_XY = 0.6
WORKSPACE_Z = (0.15, 1.0)
MIN_WORKSPACE_POINTS = 500

MAX_TILT_DEG = 25.0  # a surface is near-horizontal: its normal lies within this of gravity
MIN_PLANE_POINTS = 150
MIN_FOOTPRINT_STD = 0.05  # meters, along the shorter horizontal axis; a wall band is a thin strip
MAX_CANDIDATES = 8
MIN_PEAK_GAP = 0.04  # meters between height-histogram peaks
MIN_LEVEL_GAP = 0.03  # meters between candidate surfaces
HISTOGRAM_BIN = 0.01  # meters of height; finer than MIN_PEAK_GAP, so that two surfaces show as two peaks
MIN_PEAK_POINTS = 75  # half of MIN_PLANE_POINTS, because a table off gravity spreads over several bins
REFINE_BANDS = (0.015, 0.01, 0.01)  # meters around the plane per refinement; the first allows for the bin width
PLANE_INLIER_BAND = 0.01  # meters around a refined plane, for its point count and footprint
HEIGHT_INLIER_BAND = 0.02  # meters around the plane, for the camera height; its median tolerates a wider band


def workspace_points(xyz_camera: np.ndarray) -> np.ndarray:
    """The points of an (N, 3) camera-frame cloud inside the workspace box, as float64."""
    xyz = np.asarray(xyz_camera, dtype=np.float64)[:, :3]
    keep = (
        (np.abs(xyz[:, 0]) < WORKSPACE_XY)
        & (np.abs(xyz[:, 1]) < WORKSPACE_XY)
        & (xyz[:, 2] > WORKSPACE_Z[0])
        & (xyz[:, 2] < WORKSPACE_Z[1])
    )
    return xyz[keep]


def level_rotation(up: np.ndarray) -> np.ndarray:
    """The rotation from the camera frame to level right, level forward and up, the rows of the result.

    `up` is a unit vector in the camera frame. Level forward is the camera's view direction on the
    ground, so a roll about the view axis leaves these axes where they are in the world. A camera
    looking straight down has no such direction, and there level forward is the camera's up axis.
    """
    forward = np.array([0.0, 0.0, 1.0]) - up[2] * up
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([0.0, -1.0, 0.0]) - up[1] * up
    forward = forward / np.linalg.norm(forward)
    return np.stack([np.cross(forward, up), forward, up])


def _refine_plane(points: np.ndarray, up: np.ndarray, height: float) -> tuple[np.ndarray, float] | None:
    # A table a few degrees off gravity meets a height slice in a strip, so inliers are reselected
    # with each refined normal until the whole surface is in.
    normal, offset = up, -height
    for band in REFINE_BANDS:
        inliers = points[np.abs(points @ normal + offset) < band]
        if len(inliers) < MIN_PLANE_POINTS:
            return None
        center = inliers.mean(axis=0)
        refined = np.linalg.svd((inliers - center).T @ (inliers - center))[0][:, 2]
        refined = refined if refined @ up >= 0 else -refined
        if refined @ up < np.cos(np.radians(MAX_TILT_DEG)):
            break
        normal, offset = refined, -refined @ center
    return normal, offset


def _footprint_std(inliers: np.ndarray, normal: np.ndarray) -> float:
    horizontal = inliers - np.outer(inliers @ normal, normal)
    return float(np.sqrt(max(np.sort(np.linalg.eigvalsh(np.cov(horizontal.T)))[-2], 0.0)))


def fit_support_plane(points: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The near-horizontal plane with the most points, (normal, offset), height = normal @ p + offset.

    With gravity known, a horizontal surface is one number, its height, so the candidates are the
    peaks of the height histogram. Returns None when nothing in the cloud is planar and wide.
    """
    heights = points @ up
    edges = np.arange(heights.min(), heights.max() + HISTOGRAM_BIN, HISTOGRAM_BIN)
    if len(edges) < 2:
        return None
    counts, _ = np.histogram(heights, edges)
    centers = (edges[:-1] + edges[1:]) / 2
    best, best_count, peaks, levels = None, 0, [], []
    for peak in np.argsort(-counts):
        if counts[peak] < MIN_PEAK_POINTS or len(peaks) >= MAX_CANDIDATES:
            break
        if any(abs(centers[peak] - other) < MIN_PEAK_GAP for other in peaks):
            continue
        plane = _refine_plane(points, up, centers[peak])
        if plane is None:
            continue
        inliers = points[np.abs(points @ plane[0] + plane[1]) < PLANE_INLIER_BAND]
        if len(inliers) < MIN_PLANE_POINTS:
            continue
        level = inliers.mean(axis=0) @ up
        if any(abs(level - other) < MIN_LEVEL_GAP for other in levels):
            continue  # two peaks converged on one surface
        if _footprint_std(inliers, plane[0]) < MIN_FOOTPRINT_STD:
            continue
        peaks.append(centers[peak])
        levels.append(level)
        if len(inliers) > best_count:
            best, best_count = plane, len(inliers)
    return best


class SurfaceFrame:
    """Surface coordinates of one scene: to_surface(x) = R x + h e_z.

    R is `level_rotation` of gravity, so +z is up and +y is the camera's view direction on the ground,
    and a head roll does not turn the frame. h is the camera's height above the surface, so the
    horizontal origin is the camera's ground projection. The fitted plane (`normal`, `offset`) sets h
    and decides which points lie above the surface. Coordinates follow gravity, because a fitted
    normal carries more tilt error than the gravity estimate.
    """

    def __init__(self, gravity_up: np.ndarray, height: float, normal: np.ndarray, offset: float) -> None:
        self.gravity_up = np.asarray(gravity_up, dtype=np.float64)
        self.height = float(height)
        self.normal = np.asarray(normal, dtype=np.float64)
        self.offset = float(offset)
        self.rotation = level_rotation(self.gravity_up)

    @classmethod
    def from_points(cls, xyz_camera: np.ndarray, gravity_up: np.ndarray) -> SurfaceFrame | None:
        """Fit the frame to an (N, 3) camera-frame cloud, None when no surface is in view.

        `gravity_up` is the upward gravity direction in the camera frame. For a demonstration it is
        the SLAM world +z, for the robot the vertical of its calibrated base.
        """
        up = np.asarray(gravity_up, dtype=np.float64)
        up = up / np.linalg.norm(up)
        points = workspace_points(xyz_camera)
        if len(points) < MIN_WORKSPACE_POINTS:
            return None
        plane = fit_support_plane(points, up)
        if plane is None:
            return None
        normal, offset = plane
        inliers = points[np.abs(points @ normal + offset) < HEIGHT_INLIER_BAND]
        return cls(up, -np.median(inliers @ up), normal, offset)

    def height_above_plane(self, xyz_camera: np.ndarray) -> np.ndarray:
        """Height of camera-frame points above the fitted plane."""
        return xyz_camera @ self.normal + self.offset

    def to_surface(self, x: np.ndarray) -> np.ndarray:
        """Camera-frame points (..., 3) in surface coordinates."""
        return np.asarray(x) @ self.rotation.T + self.height * np.array([0.0, 0.0, 1.0])

    def surface_from_camera(self) -> np.ndarray:
        """The 4x4 transform behind `to_surface`."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[2, 3] = self.height
        return T

    def camera_from_surface(self) -> np.ndarray:
        """The 4x4 inverse of `surface_from_camera`."""
        T = np.eye(4)
        T[:3, :3] = self.rotation.T
        T[:3, 3] = -self.height * self.rotation[2]
        return T
