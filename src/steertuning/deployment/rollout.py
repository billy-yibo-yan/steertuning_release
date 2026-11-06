"""Build a control loop from `configs/deploy.yaml`, wait for a rollout to end, and summarize it."""

from __future__ import annotations

import select
import sys
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from steertuning.deployment.calibration import load_calibration
from steertuning.deployment.control_loop import ControlConfig, ControlLoop, RolloutStats
from steertuning.deployment.kinematics import ArmIK, RobotModel
from steertuning.deployment.observation import AriaCamera, BatchBuilder, ObjectSegmenter
from steertuning.deployment.replay import ReplayCamera, ReplayRobot, nominal_camera_from_base
from steertuning.deployment.retargeting import HandRetargeter
from steertuning.deployment.robot import YorRobot
from steertuning.policy.lora import load_lora
from steertuning.policy.policy import SteertuningPolicy
from steertuning.processing.depth import StereoDepth
from steertuning.steering.bank import DemoBank
from steertuning.steering.selection import attach_initial_contexts


def build_hardware(cfg: DictConfig):
    """The robot, the camera and T_camera_base, real or replayed. Connects before loading the depth model."""
    if cfg.replay.session is not None:
        camera = ReplayCamera(cfg.replay.session, with_rgb=cfg.object_prompt is not None)
        return ReplayRobot(), camera, nominal_camera_from_base(camera.session)

    T_camera_base = load_calibration(cfg.calibration)
    robot = YorRobot(
        cfg.robot.host, cfg.robot.port, cfg.robot.hand_host, cfg.robot.left_hand_port, cfg.robot.right_hand_port
    )
    camera = AriaCamera(cfg.camera.host, cfg.camera.port)
    camera.depth = StereoDepth(cfg.device, cfg.camera.depth_model, cfg.camera.depth_refine, cfg.camera.depth_weights)
    return robot, camera, T_camera_base


def build_bank(cfg: DictConfig) -> DemoBank | None:
    """The task's demonstrations, or None at steering weight 0 to run the policy alone.

    A recording never steers itself.
    """
    if cfg.control.steering_weight <= 0.0:
        return None
    bank = DemoBank.build(cfg.task_dir)
    if cfg.replay.session is None:
        return bank
    name = Path(cfg.replay.session).parent.name
    remaining = bank.without(name)
    if len(remaining) == 0:
        raise ValueError(
            f"{name} is the only demonstration of {cfg.task_dir}, and a recording cannot steer its own replay. "
            "Add another demonstration, or pass control.steering_weight=0 to run the policy alone."
        )
    return remaining


def build_loop(cfg: DictConfig) -> ControlLoop:
    # First, so a wrong address fails before the models load.
    robot, camera, T_camera_base = build_hardware(cfg)
    policy = SteertuningPolicy.from_pretrained(cfg.pretrained, cfg.device)
    if cfg.adapter is not None:
        load_lora(policy, cfg.adapter)
    segmenter = None
    if cfg.object_prompt is not None:
        segmenter = ObjectSegmenter(cfg.object_prompt, cfg.device, cfg.sam3_checkpoint, cfg.object_min_score)
    batch_builder = BatchBuilder(
        cfg.instruction, cfg.device, segmenter, policy.config.text_model, policy.config.text_revision, cfg.control.seed
    )
    bank = build_bank(cfg)
    if bank is not None and cfg.selection == "vinn":
        attach_initial_contexts(bank, policy, cfg.task_dir, cfg.instruction)
    config = ControlConfig(**OmegaConf.to_container(cfg.control), selection=cfg.selection)
    retargeter = HandRetargeter(OmegaConf.to_container(cfg.retargeting))
    loop = ControlLoop(
        policy, robot, camera, RobotModel(T_camera_base), ArmIK(), retargeter, batch_builder, bank, config
    )
    if cfg.replay.session is not None:
        camera.clock = lambda: loop.row * config.row_period
    return loop


def wait_for_stop(loop: ControlLoop, time_limit: float) -> str:
    """Wait for the end of a rollout and return why it ended: "operator", "time_limit" or "failed".

    The time limit counts from the first executed row, so model warm-up does not use it up.
    "failed" means the rollout stopped on its own, with the error in its stats.
    """
    deadline = None
    while loop.running:
        if select.select([sys.stdin], [], [], 0.1)[0]:
            sys.stdin.readline()
            return "operator"
        if deadline is None and loop.stats.rows > 0:
            deadline = time.monotonic() + time_limit
        if deadline is not None and time.monotonic() >= deadline:
            return "time_limit"
    return "failed"


def summary(stats: RolloutStats) -> str:
    """One line for the operator: chunks, rows, held rows and the error that ended the rollout."""
    error = "" if stats.error is None else f". Error: {type(stats.error).__name__}: {stats.error}"
    return f"{stats.chunks} chunks, {stats.rows} rows, {stats.held_rows} held{error}"
