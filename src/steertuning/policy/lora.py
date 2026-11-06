"""LoRA adapters on the denoiser.

The adapters go on the denoiser's attention projections and linear layers. Its time MLP and the
observation transformer stay as pretrained.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """A linear layer plus a low-rank update: base(x) + (alpha / rank) B A x.

    B starts at zero, so the layer starts out equal to its base.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.A = nn.Parameter(0.02 * torch.randn(rank, base.in_features).to(base.weight))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank).to(base.weight))

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + (self.alpha / self.rank) * (x @ self.A.T) @ self.B.T


def _wrap_linears(module: nn.Module, rank: int, alpha: float) -> list[LoRALinear]:
    adapters = []
    for name, child in list(module.named_children()):
        if name == "time_embedding":
            continue
        if isinstance(child, LoRALinear):
            child = child.base
        if isinstance(child, nn.Linear):
            adapter = LoRALinear(child, rank, alpha)
            setattr(module, name, adapter)
            adapters.append(adapter)
        else:
            adapters += _wrap_linears(child, rank, alpha)
    return adapters


def inject_lora(policy: nn.Module, rank: int = 8, alpha: float = 16.0) -> list[nn.Parameter]:
    """Freeze the policy and add adapters to the linear layers of its denoiser.

    On the transformer the adapters cover the attention q, k, v and output projections, the FFNs, the
    AdaLN modulation, the action projection, the final layer and the branch gates, and on the UNet its FiLM
    modulations.
    They replace any earlier adapters.
    Returns the adapter parameters, the only trainable ones.
    """
    policy.requires_grad_(False)
    adapters = _wrap_linears(policy.denoiser, rank, alpha)
    return [parameter for adapter in adapters for parameter in (adapter.A, adapter.B)]


def _adapter_state(policy: nn.Module) -> dict[str, Tensor]:
    state = policy.denoiser.state_dict(prefix="denoiser.")
    return {name: value for name, value in state.items() if name.endswith((".A", ".B"))}


def save_lora(policy: nn.Module, path: str | Path) -> None:
    """Write the adapter weights to a safetensors file, with their rank and alpha."""
    adapter = next(m for m in policy.denoiser.modules() if isinstance(m, LoRALinear))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.cpu() for name, value in _adapter_state(policy).items()}
    save_file(tensors, path, {"rank": str(adapter.rank), "alpha": str(adapter.alpha)})


def load_lora(policy: nn.Module, path: str | Path) -> None:
    """Add the adapters in a file to a policy, all frozen."""
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
    inject_lora(policy, int(metadata["rank"]), float(metadata["alpha"]))
    tensors = load_file(path)
    if tensors.keys() != _adapter_state(policy).keys():
        raise ValueError(f"The adapters in {path} do not match this policy's denoiser.")
    policy.load_state_dict(tensors, strict=False)
    policy.requires_grad_(False)
