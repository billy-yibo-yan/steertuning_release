"""Forward kinematics and best-effort Cartesian inverse kinematics for YOR.

The MJCF world is the robot's base frame: +z is up, the arms reach along -y, the left arm is at +x.
`T_camera_base` maps base coordinates to the head camera, see `steertuning.deployment.calibration`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from steertuning.data.representation import AXIS_TIPS, FINGERTIPS, LEFT, NUM_POINTS, RIGHT, WRIST_INDEX, hand_points

MJCF_PATH = Path(__file__).resolve().parents[3] / "assets" / "robot" / "yor_wuji.xml"
SIDES = ("left", "right")

ARM_JOINTS = {side: [f"{side}_arm_joint{i}" for i in range(1, 8)] for side in SIDES}
HAND_JOINTS = {side: [f"{side}_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)] for side in SIDES}
END_EFFECTOR_SITE = {side: f"{side}_arm_ee" for side in SIDES}
FINGERTIP_SITES = {side: [f"{side}_finger{i}_tip" for i in range(1, 6)] for side in SIDES}

ROBOT_BOX_MARGIN = 0.045  # meters of padding around each link's box
ROBOT_GEOM_GROUPS = (1, 2)  # the arm, mount and hand meshes


def _rotation(axis: str, degrees: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
    return T


def _translation(x: float, y: float, z: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = x, y, z
    return T


HAND_ROOT_FROM_SITE = 0.0375  # meters along the site's y axis, from its mount in assets/robot/yor_wuji.xml

# From the human wrist frame the policy predicts to the end-effector site of the same hand mount.
WRIST_TO_END_EFFECTOR = {
    "left": _rotation("z", -90.0) @ _translation(0.0, -HAND_ROOT_FROM_SITE, 0.0),
    "right": _rotation("z", 90.0) @ _rotation("y", 180.0) @ _translation(0.0, -HAND_ROOT_FROM_SITE, 0.0),
}


def wrist_from_hand_points(points: np.ndarray) -> np.ndarray:
    """The wrist pose (4, 4) from a hand's 24 points: origin at the wrist, axes through the axis tips."""
    origin = points[WRIST_INDEX]
    x_tip, y_tip, z_tip = points[AXIS_TIPS] - origin
    x = x_tip / max(np.linalg.norm(x_tip), 1e-9)
    y = y_tip - (y_tip @ x) * x
    y = y / max(np.linalg.norm(y), 1e-9)
    z = np.cross(x, y)
    if z @ z_tip < 0.0:
        z = -z
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, origin
    return T


class RobotModel:
    """The robot's kinematic chain at given joint positions, read out in the head camera frame."""

    def __init__(self, T_camera_base: np.ndarray) -> None:
        import mujoco

        self._mujoco = mujoco
        self.T_camera_base = np.asarray(T_camera_base, dtype=np.float64)
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.data = mujoco.MjData(self.model)
        self._qpos = {
            f"{side}_{part}": self.model.jnt_qposadr[[self.model.joint(name).id for name in names[side]]]
            for side in SIDES
            for part, names in (("arm", ARM_JOINTS), ("hand", HAND_JOINTS))
        }
        self._geoms = [
            (geom, self._mesh_vertices(geom))
            for geom in range(self.model.ngeom)
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            and self.model.geom_group[geom] in ROBOT_GEOM_GROUPS
        ]

    def _mesh_vertices(self, geom: int) -> np.ndarray:
        mesh = self.model.geom_dataid[geom]
        start = self.model.mesh_vertadr[mesh]
        return self.model.mesh_vert[start : start + self.model.mesh_vertnum[mesh]].copy()

    def update(self, joints: dict[str, np.ndarray]) -> None:
        """Pose the model. `joints` holds left_arm, right_arm (7,) and left_hand, right_hand (5, 4)."""
        for key, addresses in self._qpos.items():
            self.data.qpos[addresses] = np.asarray(joints[key], dtype=np.float64).reshape(-1)
        self._mujoco.mj_forward(self.model, self.data)

    def wrist_poses(self) -> np.ndarray:
        """Policy wrist poses (2, 4, 4) in the camera frame, left then right."""
        poses = []
        for side in SIDES:
            site = self.data.site(END_EFFECTOR_SITE[side])
            T_base_site = np.eye(4)
            T_base_site[:3, :3], T_base_site[:3, 3] = site.xmat.reshape(3, 3), site.xpos
            poses.append(self.T_camera_base @ T_base_site @ np.linalg.inv(WRIST_TO_END_EFFECTOR[side]))
        return np.stack(poses)

    def fingertips(self) -> np.ndarray:
        """The 10 fingertip points (2, 5, 3) in the camera frame, left then right, thumb to pinky."""
        tips = np.stack([[self.data.site(name).xpos for name in FINGERTIP_SITES[side]] for side in SIDES])
        return tips @ self.T_camera_base[:3, :3].T + self.T_camera_base[:3, 3]

    def action_points(self) -> np.ndarray:
        """A (52, 3) action holding the points the robot has: fingertips, wrists and their axis tips."""
        tips, wrists = self.fingertips(), self.wrist_poses()
        points = np.zeros((NUM_POINTS, 3))
        for hand, side_points in enumerate((LEFT, RIGHT)):
            landmarks = np.zeros((21, 3))
            landmarks[FINGERTIPS], landmarks[WRIST_INDEX] = tips[hand], wrists[hand][:3, 3]
            points[side_points] = hand_points(landmarks, wrists[hand])
        return points

    def link_boxes(self) -> np.ndarray:
        """Padded axis-aligned boxes of the robot's links in the camera frame, (G, 2, 3): low, high."""
        rotation, translation = self.T_camera_base[:3, :3], self.T_camera_base[:3, 3]
        boxes = np.empty((len(self._geoms), 2, 3))
        for i, (geom, vertices) in enumerate(self._geoms):
            R = rotation @ self.data.geom_xmat[geom].reshape(3, 3)
            camera = vertices @ R.T + rotation @ self.data.geom_xpos[geom] + translation
            boxes[i] = camera.min(axis=0) - ROBOT_BOX_MARGIN, camera.max(axis=0) + ROBOT_BOX_MARGIN
        return boxes


@dataclass
class IKResult:
    """One solve. q is the arm's joint vector to command, the seed when ok is False.

    ok is False when the target is not a finite pose with a proper rotation, or the closest pose is over
    25 cm or 86 degrees from the target.
    """

    q: np.ndarray
    ok: bool


class ArmIK:
    """Differential IK with mink on one shared model. The target is an end-effector site pose in the base frame."""

    SOLVER_DT = 0.01
    POSITION_COST = 1.0
    ORIENTATION_COST = 0.1
    LM_DAMPING = 1.0
    POSTURE_COST = 1e-3
    SOLVER_DAMPING = 1e-5
    MAX_ITERATIONS = 20
    POSITION_TOLERANCE = 1e-3
    ROTATION_TOLERANCE = 1e-3
    REJECT_POSITION = 0.25
    REJECT_ROTATION = 1.5

    def __init__(self) -> None:
        import mink
        import mujoco

        self._mink = mink
        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.configuration = mink.Configuration(self.model)
        limited = self.model.jnt_limited.astype(bool)
        addresses = self.model.jnt_qposadr[limited]
        q = self.configuration.q.copy()
        q[addresses] = np.clip(q[addresses], *self.model.jnt_range[limited].T)
        self.q = q
        self.configuration.update(self.q)
        self.limits = [mink.ConfigurationLimit(self.model)]
        self.joints = {side: np.array([self.model.joint(n).id for n in ARM_JOINTS[side]]) for side in SIDES}
        self.dofs = {side: self.model.jnt_qposadr[self.joints[side]] for side in SIDES}
        self.frame_tasks, self.posture_tasks = {}, {}
        for side in SIDES:
            self.frame_tasks[side] = mink.FrameTask(
                END_EFFECTOR_SITE[side],
                "site",
                position_cost=self.POSITION_COST,
                orientation_cost=self.ORIENTATION_COST,
                lm_damping=self.LM_DAMPING,
            )
            self.posture_tasks[side] = mink.PostureTask(self.model, cost=np.full(self.model.nv, self.POSTURE_COST))

    def seed(self, side: str, q: np.ndarray) -> None:
        """Solve the next target from these joints, so the solution stays on the arm's current branch."""
        low, high = self.model.jnt_range[self.joints[side]].T
        self.q[self.dofs[side]] = np.clip(q, low + 1e-4, high - 1e-4)

    def _residual(self, side: str) -> tuple[float, float]:
        error = self.frame_tasks[side].compute_error(self.configuration)
        return float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))

    def _iterate(self, side: str, target) -> None:
        tasks = [self.frame_tasks[side], self.posture_tasks[side]]
        self.configuration.update(self.q)
        tasks[1].set_target_from_configuration(self.configuration)
        tasks[0].set_target(target)
        for _ in range(self.MAX_ITERATIONS):
            velocity = self._mink.solve_ik(
                self.configuration,
                tasks,
                self.SOLVER_DT,
                solver="daqp",
                damping=self.SOLVER_DAMPING,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(velocity, self.SOLVER_DT)
            position, rotation = self._residual(side)
            if position <= self.POSITION_TOLERANCE and rotation <= self.ROTATION_TOLERANCE:
                break
        self.q[:] = self.configuration.q

    def solve(self, side: str, T_base_target: np.ndarray) -> IKResult:
        """The closest reachable joints for a target pose, by weighted position and orientation error."""
        dofs, start = self.dofs[side], self.q.copy()
        T = np.asarray(T_base_target, dtype=np.float64)
        rotation = T[:3, :3]
        rigid = np.isfinite(T).all() and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        if not rigid or abs(np.linalg.det(rotation) - 1.0) > 1e-3:
            return IKResult(start[dofs].copy(), False)
        self._iterate(side, self._mink.SE3.from_matrix(T))
        site = self.configuration.get_transform_frame_to_world(END_EFFECTOR_SITE[side], "site")
        position_error = np.linalg.norm(site.translation() - T[:3, 3])
        rotation_error = self._residual(side)[1]
        ok = bool(position_error <= self.REJECT_POSITION and rotation_error <= self.REJECT_ROTATION)
        if not ok:
            self.q[:] = start
        return IKResult(self.q[dofs].copy(), ok)
