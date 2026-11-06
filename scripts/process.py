"""Process recordings into training data.

    python scripts/process.py task=pickup_bread description="picking up the bread"
    python scripts/process.py path=/path/to/recordings

Overrides use Hydra syntax. See configs/process.yaml.
"""

import hydra
from omegaconf import DictConfig

from steertuning import setup_logging
from steertuning.processing.directory import process_directory
from steertuning.processing.pipeline import ProcessConfig


@hydra.main(config_path="../configs", config_name="process", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    process_directory(cfg.path, ProcessConfig(**cfg.process), cfg.description)


if __name__ == "__main__":
    main()
