"""Fine-tuning on a few demonstrations of a new task.

The pretrained policy stays frozen. By default its observation encoders run once over a pool of
augmented samples, and the LoRA adapters then train on those stored features, so a training step only
runs the denoiser. With `pool_size=None` every step encodes a fresh augmented batch, which long runs
want and which costs one encoder pass per step.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from steertuning.data.dataset import FinetuneDataset, TextTokenizer, WindowDataset, collate
from steertuning.policy.lora import inject_lora, save_lora
from steertuning.policy.policy import SteertuningPolicy, bf16_autocast
from steertuning.policy.transformer import ObservationEncoding

Batch = tuple[ObservationEncoding, Tensor, Tensor]  # encoding, actions, action mask


@dataclass
class FinetuneConfig:
    """Fine-tuning settings. See configs/finetune.yaml."""

    rank: int = 8
    alpha: float = 16.0
    steps: int = 2000
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    pool_size: int | None = 720
    num_workers: int = 8
    seed: int = 0


@dataclass
class FeaturePool:
    """Encoder outputs and targets of the augmented samples, in fp16 to fit on the GPU."""

    tokens: Tensor  # Z, (N, 515, 1024)
    valid: Tensor  # (N, 515)
    contexts: Tensor  # (N, stages, 1024)
    actions: Tensor  # (N, 15, 156)
    action_mask: Tensor  # (N, 15, 156)

    def __len__(self) -> int:
        return len(self.actions)

    def encoding(self, indices: Tensor) -> ObservationEncoding:
        contexts = self.contexts[indices].float().unbind(dim=1)
        return ObservationEncoding(self.tokens[indices].float(), self.valid[indices], list(contexts))


class _AugmentedSamples(Dataset):
    """Cycles through all windows of a task in one shuffled order. Each item's seed fixes its augmentations.

    It holds at least `size` samples, and one pass over the windows when they are more, so no window is left
    out. A snippet's share of the samples therefore follows its number of windows.
    """

    def __init__(self, dataset: WindowDataset, size: int, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.windows = [
            (segment, int(anchor)) for segment, entry in enumerate(dataset.segments) for anchor in entry.anchors
        ]
        self.size = max(size, len(self.windows))
        np.random.default_rng(seed).shuffle(self.windows)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict:
        segment, anchor = self.windows[index % len(self.windows)]
        return self.dataset.sample(segment, anchor=anchor, seed=self.seed + index)


def encoded_batches(policy: SteertuningPolicy, samples: _AugmentedSamples, config: FinetuneConfig) -> Iterator[Batch]:
    """Encode augmented samples batch by batch, with no dropout or proprioception noise."""
    from accelerate.utils import send_to_device

    policy.eval()
    loader = DataLoader(
        samples, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate
    )
    device = next(policy.parameters()).device
    for batch in loader:
        batch = send_to_device(batch, device, non_blocking=True)
        with torch.no_grad(), bf16_autocast(device):
            encoding = policy.encode_observation(batch)
        yield encoding, batch["actions"], batch["action_mask"]


def build_feature_pool(policy: SteertuningPolicy, dataset: WindowDataset, config: FinetuneConfig) -> FeaturePool:
    """Encode at least `config.pool_size` augmented samples once, and one per window when the windows are more."""
    samples = _AugmentedSamples(dataset, config.pool_size, config.seed)
    parts = [
        (encoding.tokens.half(), encoding.valid, torch.stack(encoding.contexts, dim=1).half(), actions, mask)
        for encoding, actions, mask in tqdm(encoded_batches(policy, samples, config), desc="Encoding the feature pool")
    ]
    return FeaturePool(*(torch.cat(columns) for columns in zip(*parts)))


def pool_batches(pool: FeaturePool, batch_size: int, seed: int) -> Iterator[Batch]:
    """Endless random batches from a feature pool."""
    device = pool.actions.device
    generator = torch.Generator(device).manual_seed(seed)
    while True:
        indices = torch.randperm(len(pool), device=device, generator=generator)[:batch_size]
        yield pool.encoding(indices), pool.actions[indices], pool.action_mask[indices]


def train_adapters(policy: SteertuningPolicy, batches: Iterable[Batch], config: FinetuneConfig) -> list[float]:
    """Add LoRA adapters to the policy and train them on `batches`. Returns the loss of each step."""
    policy.eval()
    parameters = inject_lora(policy, config.rank, config.alpha)
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    device = next(policy.parameters()).device
    generator = torch.Generator(device).manual_seed(config.seed)
    losses = []
    progress = tqdm(range(config.steps), desc="Fine-tuning")
    for _, (encoding, actions, mask) in zip(progress, batches):
        with bf16_autocast(device):
            loss = policy.noise_loss(encoding, actions, mask, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
        optimizer.step()
        losses.append(loss.item())
        progress.set_postfix(loss=f"{losses[-1]:.4f}", refresh=False)
    return losses


def finetune_task(
    task_dir: str | Path,
    pretrained: str | Path,
    config: FinetuneConfig | None = None,
    device: str | torch.device = "cuda",
    adapter: str | Path | None = None,
) -> Path:
    """Fine-tune on the demonstrations in a task directory and write the adapter file.

    The action normalizer comes with the pretrained policy. `adapter` defaults to
    <task_dir>/lora/adapter.safetensors.
    """
    config = FinetuneConfig() if config is None else config
    task_dir = Path(task_dir)
    adapter = task_dir / "lora" / "adapter.safetensors" if adapter is None else Path(adapter)
    torch.manual_seed(config.seed)
    policy = SteertuningPolicy.from_pretrained(pretrained, device)
    tokenizer = TextTokenizer(policy.config.text_model, policy.config.text_revision)
    dataset = FinetuneDataset(task_dir, policy.normalizer, tokenizer=tokenizer)
    if config.pool_size is None:
        samples = _AugmentedSamples(dataset, config.steps * config.batch_size, config.seed)
        batches = encoded_batches(policy, samples, config)
    else:
        batches = pool_batches(build_feature_pool(policy, dataset, config), config.batch_size, config.seed)
    train_adapters(policy, batches, config)
    save_lora(policy, adapter)
    return adapter
