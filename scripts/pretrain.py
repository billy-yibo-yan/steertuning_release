"""Pretrain the policy on D_wild.

    accelerate launch --num_processes 2 scripts/pretrain.py dataset_root=/path/to/d_wild

Overrides use Hydra syntax. See configs/pretrain.yaml. Rerunning the same command resumes.
"""

import hydra
from omegaconf import DictConfig

from steertuning import setup_logging
from steertuning.training.pretrain import pretrain


@hydra.main(config_path="../configs", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    pretrain(cfg)


if __name__ == "__main__":
    main()
