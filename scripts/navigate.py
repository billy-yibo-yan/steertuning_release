"""Build a voxel map with SAM3 latents and drive YOR to an object by name.

    python scripts/navigate.py robot.host=<robot computer> zed.host=<ZED computer> map_duration_s=300
    python scripts/navigate.py robot.host=<robot computer> zed.host=<ZED computer> target="white cup"

Drive the robot around with its joystick while the map is built.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from steertuning import setup_logging
from steertuning.navigation.navigator import yor_navigator


@hydra.main(config_path="../configs", config_name="navigation", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    navigator = yor_navigator(cfg)
    if Path(cfg.map_path).exists():
        print(f"Using the map at {cfg.map_path}.")
    else:
        Path(cfg.map_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"Mapping for {cfg.map_duration_s:.0f} s.")
        navigator.build_map(cfg.map_duration_s)
        navigator.map.save(cfg.map_path)
        print(f"{len(navigator.map)} voxels, {len(navigator.map.keyframes)} keyframes: {cfg.map_path}")
    if cfg.target is not None:
        arrived = navigator.go_to(cfg.target)
        print(f"{'Arrived at' if arrived else 'Did not reach'} {cfg.target}.")


if __name__ == "__main__":
    main()
