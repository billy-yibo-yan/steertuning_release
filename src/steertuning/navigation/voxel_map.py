"""Voxelized SLAM map in which each voxel holds a SAM3 latent.

The world frame is z up with the floor at z = 0. Camera frames are x right, y down, z forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from steertuning.navigation.sam3_latents import LATENT_DIM


def backproject(
    points: np.ndarray,
    world_from_camera: np.ndarray,
    stride: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """World points (N, 3) and their pixels (N, 2) as (col, row), from organized camera points (H, W, 3).

    Points that are not finite, not in front of the camera or deeper than `max_depth` are skipped.
    """
    rows, cols = np.mgrid[stride // 2 : points.shape[0] : stride, stride // 2 : points.shape[1] : stride]
    camera_points = points[rows, cols]
    depth = camera_points[..., 2]
    valid = np.isfinite(camera_points).all(axis=-1) & (depth > 0) & (depth <= max_depth)
    world_points = camera_points[valid] @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    return world_points, np.stack([cols[valid], rows[valid]], axis=1)


@dataclass
class VoxelMap:
    """Sparse voxels with a running mean of the latent over the points they received."""

    voxel_size: float
    keyframes: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index: dict[tuple[int, int, int], int] = {}
        self._cells = np.zeros((1024, 3), dtype=np.int32)
        self._latents = np.zeros((1024, LATENT_DIM), dtype=np.float32)
        self._hits = np.zeros(1024, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._index)

    def integrate(self, points: np.ndarray, latents: np.ndarray) -> None:
        """Add world points (N, 3) with their latents (N, 256)."""
        if len(points) == 0:
            return
        cells = np.floor(points / self.voxel_size).astype(np.int32)
        unique, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((len(unique), latents.shape[1]), dtype=np.float32)
        np.add.at(sums, inverse.ravel(), latents)
        ids = np.array([self._index.setdefault(key, len(self._index)) for key in map(tuple, unique.tolist())])
        self._reserve(len(self._index))
        self._cells[ids] = unique
        total = self._hits[ids] + counts
        self._latents[ids] += (sums - counts[:, None] * self._latents[ids]) / total[:, None]
        self._hits[ids] = total

    def _reserve(self, size: int) -> None:
        if size <= len(self._hits):
            return
        grow = max(size, 2 * len(self._hits)) - len(self._hits)
        self._cells = np.concatenate([self._cells, np.zeros((grow, 3), dtype=np.int32)])
        self._latents = np.concatenate([self._latents, np.zeros((grow, LATENT_DIM), dtype=np.float32)])
        self._hits = np.concatenate([self._hits, np.zeros(grow, dtype=np.int64)])

    def centers(self) -> np.ndarray:
        """Voxel centers (K, 3) in world coordinates."""
        return (self._cells[: len(self)] + 0.5) * self.voxel_size

    def nearest(self, latent: np.ndarray) -> np.ndarray:
        """Center of the voxel with the highest cosine similarity to `latent`."""
        stored = self._latents[: len(self)]
        norms = np.linalg.norm(stored, axis=1)
        similarity = stored @ latent / np.maximum(norms * np.linalg.norm(latent), 1e-8)
        return self.centers()[int(similarity.argmax())]

    def save(self, path: str) -> None:
        count = len(self)
        np.savez(
            path,
            voxel_size=self.voxel_size,
            cells=self._cells[:count],
            latents=self._latents[:count],
            hits=self._hits[:count],
            keyframes=np.stack(self.keyframes) if self.keyframes else np.zeros((0, 1, 1, 3), np.uint8),
        )

    @classmethod
    def load(cls, path: str) -> VoxelMap:
        data = np.load(path)
        voxel_map = cls(voxel_size=float(data["voxel_size"]), keyframes=list(data["keyframes"]))
        voxel_map._cells = data["cells"]
        voxel_map._latents = data["latents"]
        voxel_map._hits = data["hits"]
        voxel_map._index = {tuple(cell): i for i, cell in enumerate(data["cells"].tolist())}
        return voxel_map
