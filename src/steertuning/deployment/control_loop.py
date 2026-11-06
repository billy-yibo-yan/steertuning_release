"""The deployment loop: observe, track, steer, retarget, command.

A planner thread samples chunks that start at the row being executed, with real-time chunking over
the plan in progress. An executor thread walks the plan's rows at 0.3 s per row, solves wrists by IK
and fingers by retargeting, and streams interpolated joint targets. The head camera is fixed.
A rollout ends itself after MAX_PLAN_GAP seconds without a chunk or MAX_HELD_ROWS held rows in a row.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from steertuning.data.representation import CHUNK_STEPS, LEFT, RIGHT, WRIST_POSE_POINTS
from steertuning.deployment.chunking import chunking_target, compose_targets
from steertuning.deployment.kinematics import SIDES, WRIST_TO_END_EFFECTOR, wrist_from_hand_points
from steertuning.deployment.observation import Camera, CameraFrame, workspace_scene
from steertuning.deployment.pacing import limit_joint_step, row_schedule, row_ticks
from steertuning.deployment.robot import Robot
from steertuning.policy.policy import bf16_autocast
from steertuning.steering.layout import extract_layout
from steertuning.steering.selection import build_tracker, initial_context, select_demonstrations
from steertuning.steering.tracking import hand_state

log = logging.getLogger(__name__)

FIRST_PLAN_TIMEOUT = 90.0  # seconds without a usable observation before the first chunk
MAX_PLAN_GAP = 5.0  # the same after it. The plan in hand lasts 4.5 s at the default row period.
MAX_HELD_ROWS = 25  # rows in a row that IK or the row-size limit held
FIRST_ROW_SECONDS = 2.4  # the first row moves arms and fingers slowly from where they are
OBSERVATION_RETRY = 0.1  # seconds, so a recording with an empty scene does not spin
HAND_POINTS_BY_SIDE = {"left": LEFT, "right": RIGHT}


@dataclass
class ControlConfig:
    steering_weight: float = 0.3  # uniform over points and rows before real-time chunking
    num_steps: int = 8  # DDIM steps
    row_period: float = 0.3  # seconds per chunk row
    control_period: float = 0.02  # seconds between arm commands, at least
    interpolation_points: int = 12  # arm commands per row
    max_joint_speed: float = 1.0  # rad/s, a row takes longer rather than exceed it
    max_row_move: float = 1.5  # radians, a row asking for more is held
    chunking_blend: int = 8  # rows of the hand-over to a new chunk
    selection: str = "layout"  # how `start` chooses the demonstration: layout, wrist or vinn
    seed: int = 0


@dataclass
class RolloutStats:
    chunks: int = 0
    rows: int = 0
    held_rows: int = 0  # rows that IK or the row-size limit held
    error: BaseException | None = None  # what ended the rollout on its own


class ControlLoop:
    """One rollout. `start` selects the demonstration from the first observation, `stop` ends it."""

    def __init__(
        self,
        policy,
        robot: Robot,
        camera: Camera,
        robot_model,
        ik,
        retargeter,
        batch_builder,
        bank,
        config: ControlConfig,
    ) -> None:
        self.policy, self.robot, self.camera = policy, robot, camera
        self.robot_model, self.ik, self.retargeter = robot_model, ik, retargeter
        self.batch_builder, self.bank, self.config = batch_builder, bank, config
        self.device = next(policy.parameters()).device
        self.T_base_camera = np.linalg.inv(robot_model.T_camera_base)
        self.gravity_up = robot_model.T_camera_base[:3, 2]  # the base's vertical, in the camera frame
        self.lock = threading.Lock()
        self.halt = threading.Event()
        self.threads: list[threading.Thread] = []
        self.warned: set[tuple[str, str]] = set()  # (side, kind) of the problems logged in this rollout

    def start(self) -> None:
        """Prepare a rollout from the current observation and start moving."""
        self.halt.clear()
        self.stats = RolloutStats()
        self.plan: np.ndarray | None = None  # normalized rows (H, 156)
        self.plan_start = 0  # global index of the plan's first row
        self.row = 0  # global index of the row being executed
        self.inference_rows = 1.0  # running estimate of the rows that pass during one cycle
        self.tracker = None
        self.warned.clear()
        self.generator = torch.Generator(self.device).manual_seed(self.config.seed)
        self.retargeter.reset()
        if self.bank is not None:
            self.tracker = self._select_demonstration()
        self.threads = [
            threading.Thread(target=self._guarded, args=(loop,), daemon=True) for loop in (self._plan, self._execute)
        ]
        for thread in self.threads:
            thread.start()

    def stop(self) -> RolloutStats:
        """Stop both threads and wait for the calls they are in. The arms hold their last target.

        Exiting without this wait can abort the process while a thread is still inside torch.
        """
        self.halt.set()
        for thread in self.threads:
            thread.join()
        return self.stats

    @property
    def running(self) -> bool:
        return not self.halt.is_set()

    def _observe(self) -> tuple[CameraFrame, np.ndarray]:
        """The newest frame and its robot-free cloud. Joints are read before depth runs, so boxes match the image."""
        message = self.camera.receive()
        self.robot_model.update(self.robot.get_joint_positions())
        frame = self.camera.frame(message)
        return frame, workspace_scene(frame.cloud, self.robot_model.link_boxes())

    def _select_demonstration(self):
        """Choose the demonstration from the first observation and start tracking it."""
        frame, scene = self._observe()
        layout = extract_layout(scene[:, :3], self.gravity_up)
        context = None
        if self.config.selection == "vinn":
            context = initial_context(self.policy, self.batch_builder(scene, frame.rgb, self._proprio()))
        wrist_points = self.robot_model.action_points()[WRIST_POSE_POINTS]
        indices, weights = select_demonstrations(self.bank, layout, self.config.selection, wrist_points, context)
        chosen = (f"{self.bank[int(i)].name} at weight {w:.2f}" for i, w in zip(indices, weights))
        log.info("Selected %s.", "; ".join(chosen))
        return build_tracker(self.bank, layout, indices, weights)

    def _guarded(self, loop: Callable[[], None]) -> None:
        try:
            loop()
        except Exception as error:
            log.exception("The rollout stopped.")
            if not self.halt.is_set():
                self.stats.error = error
                self.halt.set()

    def _plan(self) -> None:
        last_plan, plan_limit = time.monotonic(), FIRST_PLAN_TIMEOUT
        while not self.halt.is_set():
            with self.lock:
                plan, plan_start, row = self.plan, self.plan_start, self.row
            try:
                frame, scene = self._observe()
                problem = None if len(scene) > 0 else "no points left in the workspace"
            except TimeoutError as error:
                problem = f"no frames from the camera ({error})"
            if self.halt.is_set():
                return
            if problem is not None:
                if time.monotonic() - last_plan > plan_limit:
                    raise RuntimeError(f"No new chunk for {plan_limit:g} s: {problem}")
                log.warning("Skipping this observation: %s", problem)
                self.halt.wait(OBSERVATION_RETRY)
                continue

            batch = self.batch_builder(scene, frame.rgb, self._proprio())
            target = compose_targets(self._chunking(plan, plan_start, row), self._steering())
            with bf16_autocast(self.device):
                chunk = self.policy.sample(
                    batch,
                    **self._steering_arguments(target),
                    num_steps=self.config.num_steps,
                    generator=self.generator,
                )
            rows = chunk[0].float().cpu().numpy()
            with self.lock:
                self.plan, self.plan_start = rows, row
                executed_meanwhile = self.row - row
            self.inference_rows = 0.5 * self.inference_rows + 0.5 * executed_meanwhile
            self.stats.chunks += 1
            last_plan, plan_limit = time.monotonic(), MAX_PLAN_GAP

    def _steering_arguments(self, target: tuple[np.ndarray, np.ndarray] | None) -> dict:
        if target is None:
            return {}
        action, weight = (torch.as_tensor(part, device=self.device) for part in target)
        return {"steering_target": action[None], "steering_weight": weight[None, :, None]}

    def _proprio(self) -> np.ndarray:
        """The normalized fingertips at the joints read with the image, by forward kinematics."""
        return self.policy.normalizer.normalize(self.robot_model.fingertips())

    def _chunking(self, plan: np.ndarray | None, plan_start: int, row: int) -> tuple[np.ndarray, np.ndarray] | None:
        if plan is None:
            return None
        freeze = min(math.ceil(self.inference_rows - 1e-6) + 1, CHUNK_STEPS // 2)
        return chunking_target(plan, plan_start, row, freeze, self.config.chunking_blend)

    def _steering(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The demonstration's next chunk from the cursor, with uniform w, or None without a bank.

        The cursor follows the robot's wrists and hand shape by forward kinematics.
        """
        if self.tracker is None:
            return None
        self.tracker.update(*hand_state(self.robot_model.action_points()))
        target = self.policy.normalizer.normalize(self.tracker.steering_chunk()).reshape(CHUNK_STEPS, -1)
        return target.astype(np.float32), np.full(CHUNK_STEPS, self.config.steering_weight, dtype=np.float32)

    def _execute(self) -> None:
        arms, hands = self._current_targets()
        first = True
        held_rows = 0
        while not self.halt.is_set():
            with self.lock:
                plan, plan_start, row = self.plan, self.plan_start, self.row
            if plan is None or row - plan_start >= len(plan):
                time.sleep(0.005)
                continue
            started = time.monotonic()
            points = self.policy.normalizer.denormalize(plan[max(0, row - plan_start)].reshape(-1, 3))
            arm_targets, hand_targets, hold_reason = self._row_targets(points, arms, hands)
            if hold_reason is None:
                held_rows = 0
            else:
                held_rows += 1
                self.stats.held_rows += 1
                log.warning("Row %d held: %s", row, hold_reason)
                if held_rows >= MAX_HELD_ROWS:
                    raise RuntimeError(f"{held_rows} rows in a row held. The last: {hold_reason}")
                if first:  # the fingers wait for the first row the arms can follow
                    hand_targets = hands
            ramp = first and hold_reason is None
            arms, hands = self._run_row(arm_targets, hand_targets, arms, hands, time.monotonic() - started, ramp)
            first = first and not ramp
            if self.halt.is_set():
                return
            self.stats.rows += 1
            with self.lock:
                self.row = row + 1

    def _current_targets(self) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        joints = self.robot.get_joint_positions()
        return (
            {side: joints[f"{side}_arm"] for side in SIDES},
            {side: joints[f"{side}_hand"] for side in SIDES},
        )

    def _row_targets(
        self, points: np.ndarray, arms: dict[str, np.ndarray], hands: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str | None]:
        """Arm and hand targets of one row, and why no arm can move, None when one can."""
        try:
            arm_targets, hand_targets, arm_solved = self._solve_row(points, arms, hands)
        except Exception as error:
            return arms, hands, f"the solver raised {type(error).__name__}: {error}"
        if not arm_solved:
            return arms, hand_targets, "IK rejected both wrists"
        largest_move = max(np.abs(arm_targets[side] - arms[side]).max() for side in SIDES)
        if largest_move > self.config.max_row_move:
            return arms, hand_targets, f"it asks for a {largest_move:.2f} rad joint move"
        return arm_targets, hand_targets, None

    def _solve_row(
        self, points: np.ndarray, arms: dict[str, np.ndarray], hands: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], bool]:
        """Joint targets of one row: wrists by IK, fingers by retargeting relative to the wrists.

        An arm whose wrist IK rejects holds, and so do fingers whose solve diverged. The last value says
        whether IK accepted at least one wrist.
        """
        arm_targets, hand_targets = dict(arms), dict(hands)
        arm_solved = False
        for side in SIDES:
            hand = points[HAND_POINTS_BY_SIDE[side]]
            if not np.isfinite(hand).all():
                self._warn_once(side, "ik", "%s arm: its wrist target is not finite. It holds.")
                continue
            wrist = wrist_from_hand_points(hand)
            self.ik.seed(side, arms[side])
            result = self.ik.solve(side, self.T_base_camera @ wrist @ WRIST_TO_END_EFFECTOR[side])
            if result.ok:
                arm_targets[side] = result.q
                arm_solved = True
            else:
                self._warn_once(side, "ik", "%s arm: IK rejected its wrist target. It holds until one is reachable.")
            fingers = self.retargeter(side, hand[:21])
            if fingers is not None:
                hand_targets[side] = fingers
            else:
                self._warn_once(side, "fingers", "%s hand: a finger solve diverged. Its fingers hold.")
        return arm_targets, hand_targets, arm_solved

    def _warn_once(self, side: str, kind: str, message: str) -> None:
        """Log `message` the first time in the rollout that `side` has a problem of this `kind`."""
        if (side, kind) not in self.warned:
            self.warned.add((side, kind))
            log.warning(message, side.title())

    def _run_row(
        self,
        arm_targets: dict[str, np.ndarray],
        hand_targets: dict[str, np.ndarray],
        arms: dict[str, np.ndarray],
        hands: dict[str, np.ndarray],
        prep: float,
        ramp: bool,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Send the row's finger targets, then stream its arm motion under the speed limit.

        With `ramp`, the row takes at least FIRST_ROW_SECONDS and the fingers move along with the arms.

        Returns the arm and hand targets it ended on.
        """
        cfg = self.config
        start = {side: arms[side].copy() for side in SIDES}
        largest_move = max(np.abs(arm_targets[side] - start[side]).max() for side in SIDES)
        duration, ticks = row_schedule(
            cfg.row_period,
            prep,
            largest_move,
            cfg.max_joint_speed,
            cfg.interpolation_points,
            cfg.control_period,
            FIRST_ROW_SECONDS if ramp else 0.0,
        )
        if not ramp and not self.halt.is_set():
            self.robot.command(hand_targets=hand_targets)  # the Wuji servers low-pass their targets
        current = dict(start)
        last_command = time.monotonic()
        for fraction in row_ticks(duration, ticks, self.halt):
            now = time.monotonic()
            for side in SIDES:
                waypoint = (1.0 - fraction) * start[side] + fraction * arm_targets[side]
                current[side] = limit_joint_step(current[side], waypoint, cfg.max_joint_speed, now - last_command)
            last_command = now
            fingers = {s: (1.0 - fraction) * hands[s] + fraction * hand_targets[s] for s in SIDES} if ramp else None
            self.robot.command(arm_targets=current, hand_targets=fingers)
        return current, hand_targets
