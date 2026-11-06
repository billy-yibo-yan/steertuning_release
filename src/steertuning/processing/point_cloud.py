"""Colored scene clouds with the embodiment removed.

The camera-centered ellipsoid crop (semi-axes 0.85, 1.5, 1.5 m in the RGB frame) and the resampling
to 5,000 points happen per sample at read time, after the anchor frame is chosen, in
`steertuning.data.representation.crop_and_resample`. On disk a cloud keeps enough points for that.
"""

from __future__ import annotations

import cv2
import numpy as np

from steertuning.data.representation import MAX_DEPTH, transform_points


def clear_masked_pixels(image: np.ndarray, mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """A copy of a (H, W) image with zeros under a dilated binary mask.

    Applied to depth, it stops masked pixels from ever becoming points.
    """
    kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
    return np.where(cv2.dilate(mask.astype(np.uint8), kernel) > 0, 0, image)


def unproject_depth(depth_mm: np.ndarray, K: np.ndarray, max_depth_m: float) -> np.ndarray:
    """(N, 3) camera-frame points for pixels with depth in (0, max_depth_m), in raster order."""
    height, width = depth_mm.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_mm.astype(np.float32) / 1000.0
    valid = (z > 0.0) & (z < max_depth_m)
    x = (u[valid] - K[0, 2]) * z[valid] / K[0, 0]
    y = (v[valid] - K[1, 2]) * z[valid] / K[1, 1]
    return np.stack([x, y, z[valid]], axis=-1)


def project_into_image(
    points_world: np.ndarray, K: np.ndarray, T_world_camera: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean mask of the points that land inside an image, and their (M, 2) float pixels."""
    camera = transform_points(points_world, np.linalg.inv(T_world_camera))
    projected = camera @ K.T
    uv = projected[:, :2] / np.clip(projected[:, 2:3], 1e-9, None)
    inside = (
        (projected[:, 2] > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width - 1)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height - 1)
    )
    return inside, uv[inside]


def colored_cloud(
    depth_mm: np.ndarray,
    K_depth: np.ndarray,
    T_world_depth: np.ndarray,
    rgb: np.ndarray,
    K_rgb: np.ndarray,
    T_world_rgb: np.ndarray,
) -> np.ndarray:
    """(N, 6) float32 world-frame points with rgb 0-255, from the depth pixels the RGB camera sees."""
    points = transform_points(unproject_depth(depth_mm, K_depth, MAX_DEPTH), T_world_depth)
    inside, uv = project_into_image(points, K_rgb, T_world_rgb, rgb.shape[1], rgb.shape[0])
    colors = rgb[uv[:, 1].astype(np.int32), uv[:, 0].astype(np.int32)]
    return np.concatenate([points[inside].astype(np.float32), colors.astype(np.float32)], axis=1)


def resample_cloud(cloud: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly `num_points` points. Points repeat only when too few exist. An empty cloud stays zeros."""
    if len(cloud) == 0:
        return np.zeros((num_points, 6), dtype=np.float32)
    return cloud[rng.choice(len(cloud), size=num_points, replace=len(cloud) < num_points)]
