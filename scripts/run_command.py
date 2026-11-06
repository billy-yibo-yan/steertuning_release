r"""Run a long-horizon language command on YOR.

    GEMINI_API_KEY=... python scripts/run_command.py command="Put the white cup into the trash" \
        robot.host=<robot computer> zed.host=<ZED computer> camera.host=<Aria computer>

Gemini ER splits the command into navigate and manipulate steps. Tasks come from the catalog in
configs/orchestrator.yaml. Each manipulate step loads that task's adapter and demonstrations and
runs the control loop of scripts/deploy.py with the other settings of configs/deploy.yaml.
"""

import gc
import os

import hydra
import torch
from hydra import compose
from omegaconf import DictConfig

from steertuning import ask, setup_logging
from steertuning.deployment.rollout import build_loop, summary, wait_for_stop
from steertuning.navigation.navigator import yor_navigator
from steertuning.orchestrator.planner import plan_command


def deploy_config(cfg: DictConfig, task: str) -> DictConfig:
    """The `configs/deploy.yaml` of one task in the catalog."""
    entry = cfg.tasks[task]
    object_prompt = "null" if entry.object_prompt is None else repr(entry.object_prompt)
    sam3_checkpoint = "null" if cfg.sam3_checkpoint is None else repr(cfg.sam3_checkpoint)
    return compose(
        "deploy",
        overrides=[
            f"task={task}",
            f"instruction={entry.instruction!r}",
            f"object_prompt={object_prompt}",
            f"sam3_checkpoint={sam3_checkpoint}",
            f"device={cfg.device}",
            f"robot.host={cfg.robot.host}",
            f"robot.port={cfg.robot.port}",
            f"camera.host={cfg.camera.host}",
            f"camera.port={cfg.camera.port}",
        ],
    )


def manipulator(cfg: DictConfig):
    """The manipulate callable. The arms go home before the first task and then hold between tasks."""
    went_home = False

    def manipulate(task: str, time_limit: float) -> bool:
        """Run one rollout. True only when the time limit ended it without error, so the plan may go on."""
        nonlocal went_home
        try:
            loop = build_loop(deploy_config(cfg, task))
            if not went_home:
                loop.robot.reset()
                went_home = True
            loop.start()
        except Exception as error:  # a model or server that is down, no surface or no image at the start
            print(f"{task}: not started: {type(error).__name__}: {error}")
            return False
        try:
            reason = wait_for_stop(loop, time_limit)
        finally:
            stats = loop.stop()
            print(f"{task}: {summary(stats)}")
        del loop
        gc.collect()
        torch.cuda.empty_cache()
        return reason == "time_limit" and stats.error is None

    return manipulate


@hydra.main(config_path="../configs", config_name="orchestrator", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    if not os.path.exists(cfg.map_path):
        raise FileNotFoundError(f"No map at {cfg.map_path}. Build one with scripts/navigate.py.")
    navigator = yor_navigator(cfg)

    descriptions = {name: entry.description for name, entry in cfg.tasks.items()}
    plan = plan_command(cfg.command, navigator.sensors.read().rgb, descriptions, **cfg.planner)
    for number, step in enumerate(plan.steps, 1):
        print(f"{number}. {step.skill} {step.target}")
    if cfg.confirm and ask("Enter runs this plan, q quits: ") == "q":
        return

    # A manipulate step has no signal that its task succeeded, only that it ran for its time limit.
    # The plan stops at the first step that fails, so the base cannot drive off after one.
    manipulate = manipulator(cfg)
    for step in plan.steps:
        if step.skill == "navigate":
            done = navigator.go_to(step.target)
        else:
            done = manipulate(step.target, cfg.time_limit)
        print(f"{step.skill} {step.target}: {'done' if done else 'failed'}")
        if not done:
            break


if __name__ == "__main__":
    main()
