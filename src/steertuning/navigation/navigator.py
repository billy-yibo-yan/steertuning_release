"""Navigate to an object by name.

Following OK-Robot, the robot maps the room as voxels with SAM3 latents, finds the voxel nearest
to the target object's latent, and drives to a stop position within reach of it, facing it.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation

from steertuning.navigation.planner import START_SNAP_METERS, build_grid, plan_approach
from steertuning.navigation.sam3_latents import Sam3Latents
from steertuning.navigation.voxel_map import VoxelMap, backproject
from steertuning.remote import TopicReader, connect_rpc

log = logging.getLogger(__name__)

# ZED world (y up, right-handed) to the map's world (z up), camera (y up, looks along -z) to
# the map's camera convention (y down, looks along +z).
WORLD_FROM_ZED_WORLD = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
ZED_CAMERA_FROM_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
# Within a `zed/pose` message: qx qy qz qw x y z of each pose, then base x y z theta, then the timestamp.
BASE_POSE = slice(0, 7)
CAMERA_POSE = slice(7, 14)
POSE_TIMESTAMP = 18
FRAME_TIMEOUT_S = 2.0
POLL_PERIOD_S = 0.05


@dataclass
class NavigationConfig:
    voxel_size: float = 0.03
    max_depth: float = 5.0
    pixel_stride: int = 4
    body_radius: float = 0.8
    map_rate_hz: float = 1.0
    max_keyframes: int = 64
    min_target_score: float = 0.7
    grid_resolution: float = 0.05
    height_band: tuple[float, float] = (0.3, 1.5)
    robot_radius: float = 0.30
    reach_distance: float = 0.45
    arrival_distance: float = 0.08
    arrival_yaw_deg: float = 5.0
    timeout_s: float = 180.0


@dataclass
class RgbdFrame:
    rgb: np.ndarray  # (H, W, 3) uint8
    points: np.ndarray  # (H, W, 3) meters in the map's camera frame, non-finite where invalid
    world_from_camera: np.ndarray  # (4, 4)


class Sensors(Protocol):
    def read(self) -> RgbdFrame: ...


class Base(Protocol):
    def pose(self) -> np.ndarray:
        """World x, y and yaw, with the robot's forward direction at (cos yaw, sin yaw)."""

    def follow_path(self, waypoints: np.ndarray) -> None: ...

    def move_to(self, xy: np.ndarray, yaw: float) -> None:
        """Drive to `xy` and end at heading `yaw`."""

    def stop(self) -> None: ...


class TargetNotFound(RuntimeError):
    pass


class Navigator:
    def __init__(
        self,
        sensors: Sensors,
        base: Base,
        latents: Sam3Latents,
        config: NavigationConfig,
        voxel_map: VoxelMap | None = None,
    ) -> None:
        self.sensors = sensors
        self.base = base
        self.latents = latents
        self.config = config
        self.map = VoxelMap(config.voxel_size) if voxel_map is None else voxel_map

    def observe(self) -> None:
        """Integrate one sensor frame into the map and keep its image as a keyframe."""
        frame = self.sensors.read()
        config = self.config
        points, pixels = backproject(frame.points, frame.world_from_camera, config.pixel_stride, config.max_depth)
        # The robot's arms and whatever they hold must not become obstacles that move with it.
        keep = np.linalg.norm(points[:, :2] - self.base.pose()[:2], axis=1) > config.body_radius
        pixel_latents = self.latents.pixel_latents(frame.rgb)
        self.map.integrate(points[keep], self.latents.sample(pixel_latents, pixels[keep], frame.points.shape[:2]))
        self.map.keyframes.append(frame.rgb.copy())
        if len(self.map.keyframes) > config.max_keyframes:
            del self.map.keyframes[::2]  # every other one, so the keyframes still span the whole run

    def build_map(self, duration_s: float) -> None:
        """Map at `map_rate_hz` for `duration_s` while the robot is driven around."""
        end = time.monotonic() + duration_s
        self.observe()  # a ZED node that is not running fails here, before the mapping time passes
        while time.monotonic() < end:
            tick = time.monotonic()
            try:
                self.observe()
            except TimeoutError as error:  # a dropped frame must not cost the map built so far
                log.warning("Skipped a frame: %s", error)
            time.sleep(max(0.0, 1.0 / self.config.map_rate_hz - (time.monotonic() - tick)))

    def locate(self, target: str) -> np.ndarray:
        """Center of the voxel nearest to the target's SAM3 latent."""
        found = self.latents.target_latent(self.map.keyframes, target, self.config.min_target_score)
        if found is None or len(self.map) == 0:
            raise TargetNotFound(f'"{target}" is not visible in any of {len(self.map.keyframes)} keyframes.')
        return self.map.nearest(found.latent)

    def go_to(self, target: str) -> bool:
        """Drive to a stop position within reach of `target` on the map, then turn in place to face it.

        Returns False when the base does not get there within `timeout_s`, or has nowhere to start a path.
        """
        config = self.config
        position = self.locate(target)[:2]
        start = self.base.pose()[:2]
        grid = build_grid(
            self.map.centers(),
            np.vstack([start, position]),
            config.grid_resolution,
            tuple(config.height_band),
            config.robot_radius,
        )
        approach = plan_approach(grid, start, position, config.reach_distance)
        if approach is None:
            log.warning("No free cell within %g m of the robot, so no path can start.", START_SNAP_METERS)
            return False
        end = approach.waypoints[-1]
        deadline = time.monotonic() + config.timeout_s
        try:
            self.base.follow_path(approach.waypoints)
            if not self._wait(lambda pose: np.linalg.norm(end - pose[:2]) < config.arrival_distance, deadline):
                return False
            self.base.move_to(self.base.pose()[:2], approach.yaw)
            return self._wait(lambda pose: self._faces(pose, approach.yaw), deadline)
        finally:
            self.base.stop()

    def _wait(self, arrived, deadline: float) -> bool:
        while time.monotonic() < deadline:
            if arrived(self.base.pose()):
                return True
            time.sleep(POLL_PERIOD_S)
        return False

    def _faces(self, pose: np.ndarray, yaw: float) -> bool:
        heading_error = np.angle(np.exp(1j * (yaw - pose[2])))
        return abs(np.degrees(heading_error)) < self.config.arrival_yaw_deg


def _pose_matrix(pose) -> np.ndarray:
    """4 x 4 transform from qx qy qz qw x y z."""
    pose = np.asarray(pose, dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[:4]).as_matrix()
    matrix[:3, 3] = pose[4:]
    return matrix


class YorSensors:
    """Left ZED image, organized point cloud and SLAM pose from YOR's ZED node."""

    def __init__(self, host: str, port: int) -> None:
        self._readers = [TopicReader(host, port, topic) for topic in ("zed/image", "zed/pcd", "zed/pose")]
        self._pool = ThreadPoolExecutor(len(self._readers))
        self._last_timestamp = None

    def _read_frame(self) -> tuple[dict, dict, list]:
        """The three messages of one new ZED frame.

        The node stamps the three alike. A read of the cloud can outlast a frame, so the three
        reads run at once and repeat until their timestamps match. The topics keep their last
        message while ZED tracking is lost, so a repeated timestamp counts as no frame.
        """
        deadline = time.monotonic() + FRAME_TIMEOUT_S
        while time.monotonic() < deadline:
            image, points, pose = self._pool.map(lambda reader: reader.read(), self._readers)
            timestamp = pose[POSE_TIMESTAMP]
            if image["timestamp"] == points["timestamp"] == timestamp != self._last_timestamp:
                self._last_timestamp = timestamp
                return image, points, pose
            time.sleep(0.005)
        raise TimeoutError(
            f"No new ZED frame for {FRAME_TIMEOUT_S:g} s. "
            "Check the ZED node, its tracking, and that it runs commlink==0.4.2."
        )

    def read(self) -> RgbdFrame:
        image, points, pose = self._read_frame()
        zed_world_from_zed_camera = _pose_matrix(pose[CAMERA_POSE])
        return RgbdFrame(
            rgb=image["image"][..., ::-1].copy(),  # the ZED's byte order is BGR
            points=points["points"][..., :3] * np.diag(ZED_CAMERA_FROM_CAMERA)[:3],
            world_from_camera=WORLD_FROM_ZED_WORLD @ zed_world_from_zed_camera @ ZED_CAMERA_FROM_CAMERA,
        )


class YorBase:
    """YOR base over RPC, with its pose from the ZED node.

    YOR works in the ZED's (x, z) ground plane; this class speaks map xy. YOR's `get_pose` RPC
    updates only while the base follows a path or a goal, so the pose comes from `zed/pose`.
    """

    def __init__(self, host: str, port: int, zed_host: str, zed_port: int) -> None:
        self._client = connect_rpc(host, port)
        self._pose = TopicReader(zed_host, zed_port, "zed/pose")
        self._timestamp, self._timestamp_since = None, 0.0

    def pose(self) -> np.ndarray:
        """The base pose. Raises TimeoutError when its timestamp has not changed for FRAME_TIMEOUT_S.

        The topic keeps its last message while ZED tracking is lost, and the base must not drive on it.
        """
        message = self._pose.read()
        now = time.monotonic()
        if message[POSE_TIMESTAMP] != self._timestamp:
            self._timestamp, self._timestamp_since = message[POSE_TIMESTAMP], now
        elif now - self._timestamp_since > FRAME_TIMEOUT_S:
            raise TimeoutError(f"The ZED pose has not changed for {FRAME_TIMEOUT_S:g} s. Check the ZED tracking.")
        zed_world_from_base = _pose_matrix(message[BASE_POSE])
        yaw = float(np.arctan2(-zed_world_from_base[2, 0], zed_world_from_base[0, 0])) % (2 * np.pi)
        return np.array([zed_world_from_base[0, 3], -zed_world_from_base[2, 3], yaw])

    def follow_path(self, waypoints: np.ndarray) -> None:
        self._client.follow_path([(float(x), float(-y)) for x, y in waypoints])

    def move_to(self, xy: np.ndarray, yaw: float) -> None:
        # YOR's goal heading is the angle from +z toward +x of the ZED ground plane.
        heading = float(np.arctan2(np.cos(yaw), -np.sin(yaw)))
        self._client.follow_path(None)
        self._client.move_to((float(xy[0]), float(-xy[1]), heading))

    def stop(self) -> None:
        self._client.follow_path(None)


def yor_navigator(cfg: DictConfig) -> Navigator:
    """A navigator on YOR from `configs/navigation.yaml`, starting from the map at `map_path` when that file exists."""
    config = NavigationConfig(**OmegaConf.to_container(cfg.navigation))
    voxel_map = VoxelMap.load(cfg.map_path) if os.path.exists(cfg.map_path) else None
    latents = Sam3Latents(cfg.device, cfg.sam3_checkpoint)
    base = YorBase(cfg.robot.host, cfg.robot.port, cfg.zed.host, cfg.zed.port)
    return Navigator(YorSensors(cfg.zed.host, cfg.zed.port), base, latents, config, voxel_map)
