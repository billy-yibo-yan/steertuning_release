r"""Run a fine-tuned policy on the robot, or on a recorded session without hardware.

    python scripts/deploy.py task=pickup_bread instruction="picking up the bread" object_prompt=bread \
        robot.host=<robot computer> camera.host=<Aria computer>
    python scripts/deploy.py task=pickup_bread instruction="picking up the bread" object_prompt=bread \
        replay.session=$STEERTUNING_HOME/pickup_bread/pickup_bread_1/preprocessed

Type `home` to bring the robot home, Enter starts the rollout, and Enter or the time limit stops it.
"""

import sys

import hydra
from omegaconf import DictConfig

from steertuning import ask, setup_logging
from steertuning.deployment.rollout import build_loop, summary, wait_for_stop


@hydra.main(config_path="../configs", config_name="deploy", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    loop = build_loop(cfg)
    while True:
        answer = ""
        while answer not in ("home", "q"):
            answer = ask("Type home to move the robot home, q quits: ")
        if answer == "q":
            return
        try:
            loop.robot.reset()
        except KeyboardInterrupt:
            print("Stopped on the way home.")
            return
        except RuntimeError as error:  # an arm did not arrive
            print(error)
            sys.exit(1)
        if ask("Enter starts the rollout, q quits: ") == "q":
            return
        try:
            loop.start()
        except (ValueError, TimeoutError) as error:  # no surface or no image in the first observation
            print(f"Not started: {error}")
            continue
        print("Running. Enter stops.")
        try:
            reason = wait_for_stop(loop, cfg.time_limit)
        except KeyboardInterrupt:
            reason = "operator"
        print(f"Stopped ({reason}). {summary(loop.stop())}")


if __name__ == "__main__":
    main()
