"""Object layouts: the objects on the manipulated surface and their 3D centroids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from steertuning.steering.surface import SurfaceFrame, workspace_points

LAYOUT_POINTS = 40000  # clouds are thinned to this, so all densities see the same point-count thresholds
BAND = (0.015, 0.35)  # meters above the surface
VOXEL = 0.02  # meters
MIN_OBJECT_POINTS = 40
UNMATCHED_PENALTY = 0.10  # meters per object without a counterpart


@dataclass
class ObjectLayout:
    """The objects on the surface of one scene.

    frame: the scene's surface frame.
    centroids: (K, 3) object centroids in surface coordinates.
    """

    frame: SurfaceFrame
    centroids: np.ndarray


def _object_clusters(points: np.ndarray, frame: SurfaceFrame) -> list[np.ndarray]:
    """Connected voxel clusters of enough points above the surface."""
    heights = frame.height_above_plane(points)
    band = points[(heights > BAND[0]) & (heights < BAND[1])]
    if len(band) < MIN_OBJECT_POINTS:
        return []
    voxels = np.floor(band / VOXEL).astype(int)
    voxels -= voxels.min(axis=0)
    occupied = np.zeros(voxels.max(axis=0) + 1, dtype=bool)
    occupied[tuple(voxels.T)] = True
    labels, count = ndimage.label(occupied, structure=np.ones((3, 3, 3)))
    point_labels = labels[tuple(voxels.T)]

    clusters = (band[point_labels == label] for label in range(1, count + 1))
    return [cluster for cluster in clusters if len(cluster) >= MIN_OBJECT_POINTS]


def extract_layout(xyz_camera: np.ndarray, gravity_up: np.ndarray) -> ObjectLayout:
    """The object layout of an (N, 3) camera-frame cloud, given gravity up in the camera frame.

    Each connected cluster of enough points above the surface is an object, and its centroid is the
    mean of its points. Raises ValueError when no surface is in view.
    """
    if len(xyz_camera) > LAYOUT_POINTS:
        xyz_camera = xyz_camera[np.random.default_rng(0).choice(len(xyz_camera), LAYOUT_POINTS, replace=False)]
    frame = SurfaceFrame.from_points(xyz_camera, gravity_up)
    if frame is None:
        raise ValueError("No support surface found in the point cloud.")
    clusters = _object_clusters(workspace_points(xyz_camera), frame)
    centroids = np.array([frame.to_surface(c.mean(axis=0)) for c in clusters]).reshape(-1, 3)
    return ObjectLayout(frame, centroids)


def match_objects(a: ObjectLayout, b: ObjectLayout) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian matching of centroids on squared 3D distance: indices into a and into b."""
    differences = a.centroids[:, None] - b.centroids[None]
    return linear_sum_assignment((differences**2).sum(axis=-1))


def layout_distance(a: ObjectLayout, b: ObjectLayout) -> float:
    """Root-mean-square 3D error of the matched objects, plus 0.10 per unmatched object.

    Two layouts with no objects in common to match are 0.10 apart per object they hold.
    """
    rows, cols = match_objects(a, b)
    unmatched = abs(len(a.centroids) - len(b.centroids))
    if len(rows) == 0:
        return UNMATCHED_PENALTY * unmatched
    errors = a.centroids[rows] - b.centroids[cols]
    return float(np.sqrt((errors**2).sum(axis=-1).mean())) + UNMATCHED_PENALTY * unmatched


def surface_transform(demo: ObjectLayout, live: ObjectLayout) -> np.ndarray:
    """The transform T_camera_demo that re-expresses a demonstration through the two surface frames.

    The demonstration goes into its own surface frame and comes out of the live one, so gravity and the
    surface height line up, and the demonstration keeps its heading and horizontal position relative
    to the camera.
    """
    return live.frame.camera_from_surface() @ demo.frame.surface_from_camera()
