"""Record one task demonstration from the Aria stream. Enter starts and stops it.

    python scripts/record_demo.py task=pickup_bread stream.host=<publisher address>

The publisher runs on the machine connected to the glasses.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from steertuning import setup_logging
from steertuning.processing.stream_recorder import record_demo


@hydra.main(config_path="../configs", config_name="process", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    record_demo(Path(cfg.path), cfg.description, cfg.stream.host, cfg.stream.port)


if __name__ == "__main__":
    main()
