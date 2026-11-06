"""Fine-tune LoRA adapters on a task's demonstrations.

    python scripts/finetune.py task=pickup_bread
    python scripts/finetune.py task=pickup_bread finetune.steps=500 adapter=adapter_500.safetensors

Overrides use Hydra syntax. See configs/finetune.yaml.
"""

import hydra
from omegaconf import DictConfig

from steertuning import setup_logging
from steertuning.training.finetune import FinetuneConfig, finetune_task


@hydra.main(config_path="../configs", config_name="finetune", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    config = FinetuneConfig(**cfg.finetune)
    adapter = finetune_task(cfg.task_dir, cfg.pretrained, config, cfg.device, cfg.adapter)
    print(f"Adapter written to {adapter}")


if __name__ == "__main__":
    main()
