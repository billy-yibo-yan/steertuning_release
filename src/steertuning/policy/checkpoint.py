"""Pretrained policy directory: policy.safetensors, config.yaml, normalizer.json.

Frozen encoders load from their upstream repositories and are not stored.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file
from torch import nn

WEIGHTS_FILE = "policy.safetensors"
CONFIG_FILE = "config.yaml"
NORMALIZER_FILE = "normalizer.json"


def _frozen_names(policy: nn.Module) -> tuple[str, ...]:
    """State-dict name starts of the child modules marked `frozen = True`."""
    return tuple(f"{name}." for name, module in policy.named_children() if getattr(module, "frozen", False))


def stored_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    """State dict without the frozen encoders."""
    frozen = _frozen_names(policy)
    return {name: value.detach().cpu() for name, value in policy.state_dict().items() if not name.startswith(frozen)}


def save_pretrained(policy: nn.Module, path: str | Path) -> None:
    """Write the policy weights, its config and its normalizer to a directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(stored_state_dict(policy), path / WEIGHTS_FILE)
    with open(path / CONFIG_FILE, "w") as f:
        yaml.safe_dump(asdict(policy.config), f, sort_keys=False)
    policy.normalizer.to_json(path / NORMALIZER_FILE)


def read_config(path: str | Path) -> dict:
    return yaml.safe_load((Path(path) / CONFIG_FILE).read_text())


def load_weights(policy: nn.Module, path: str | Path) -> None:
    """Load the stored weights. Only frozen encoders may stay at their upstream values."""
    state = load_file(Path(path) / WEIGHTS_FILE)
    missing, unexpected = policy.load_state_dict(state, strict=False)
    frozen = _frozen_names(policy)
    missing = [name for name in missing if not name.startswith(frozen)]
    if missing or unexpected:
        raise RuntimeError(f"Weights in {path} do not match the config: missing {missing}, unexpected {unexpected}")
