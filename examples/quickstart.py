"""Fine-tune and steer a tiny policy on synthetic data. Needs no GPU, weights, or network.

    python examples/quickstart.py

The policy is `configs/policy/tiny.yaml` with stub point and text encoders, and the data is random.
The losses and errors therefore show the code path, and they say nothing about robot performance.
"""

from pathlib import Path

import numpy as np
import torch
import yaml

from steertuning.data.normalizer import ActionNormalizer
from steertuning.data.representation import CHUNK_STEPS, NUM_POINTS, fingertips
from steertuning.policy.policy import PolicyConfig, SteertuningPolicy
from steertuning.policy.stubs import StubPointEncoder, StubTextEncoder
from steertuning.training.finetune import FeaturePool, FinetuneConfig, pool_batches, train_adapters

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "policy" / "tiny.yaml"
NUM_SAMPLES = 64
CLOUD_POINTS = 64
FINETUNE_STEPS = 100
WEIGHTS = (0.0, 0.3, 1.0)


def build_policy() -> SteertuningPolicy:
    config = PolicyConfig(**yaml.safe_load(CONFIG.read_text()))
    point_encoder = StubPointEncoder(config.num_point_tokens, config.point_feature_dim)
    text_encoder = StubTextEncoder(20, config.text_feature_dim)
    return SteertuningPolicy(config, point_encoder, text_encoder).eval()


def synthetic_demonstrations(count: int, rng: np.random.Generator) -> np.ndarray:
    """Action chunks in meters, (count, 15, 52, 3): a fixed hand shape that moves in a straight line."""
    shape = rng.normal(0.0, 0.05, (NUM_POINTS, 3))
    start = rng.uniform(-0.2, 0.2, (count, 1, 1, 3))
    motion = rng.uniform(-0.3, 0.3, (count, 1, 1, 3))
    progress = np.linspace(0.0, 1.0, CHUNK_STEPS)[None, :, None, None]
    return (shape + start + progress * motion + np.array([0.0, 0.0, 0.5])).astype(np.float32)


def observations(actions: np.ndarray, normalizer: ActionNormalizer, rng: np.random.Generator) -> dict:
    """Random observations in the policy's batch format, with fingertips at the first row."""
    count = len(actions)
    coord = rng.normal(0.0, 0.5, (count * CLOUD_POINTS, 3))
    return {
        "point_cloud": {
            "coord": torch.as_tensor(coord, dtype=torch.float32),
            "offset": CLOUD_POINTS * torch.arange(1, count + 1),
        },
        "object_mask": torch.as_tensor(rng.random((count, 1, 255, 255)) > 0.9, dtype=torch.float32),
        "proprio": torch.as_tensor(fingertips(normalizer.normalize(actions[:, 0])), dtype=torch.float32),
        "text_tokens": torch.as_tensor(rng.integers(0, 1000, (count, 20)), dtype=torch.long),
        "text_mask": torch.ones(count, 20, dtype=torch.long),
    }


@torch.no_grad()
def feature_pool(policy: SteertuningPolicy, batch: dict, actions: torch.Tensor) -> FeaturePool:
    encoding = policy.encode_observation(batch)
    return FeaturePool(
        encoding.tokens.half(),
        encoding.valid,
        torch.stack(encoding.contexts, dim=1).half(),
        actions,
        torch.ones_like(actions),
    )


def sample(policy: SteertuningPolicy, batch: dict, target: torch.Tensor | None = None, w: float = 0.0) -> torch.Tensor:
    return policy.sample(batch, target, w, generator=torch.Generator().manual_seed(0))


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    policy = build_policy()
    parameters = sum(p.numel() for p in policy.parameters())
    print(f"Tiny policy with stub encoders: {parameters / 1e6:.2f}M parameters, on CPU")

    meters = synthetic_demonstrations(NUM_SAMPLES, rng)
    normalizer = ActionNormalizer.fit(meters.reshape(-1, 3))
    policy.normalizer = normalizer
    batch = observations(meters, normalizer, rng)
    actions = torch.as_tensor(normalizer.normalize(meters).reshape(NUM_SAMPLES, CHUNK_STEPS, -1), dtype=torch.float32)

    # One demonstration to steer with, and the observation it goes with.
    query = {key: value[:1] for key, value in batch.items() if key != "point_cloud"}
    query["point_cloud"] = {
        "coord": batch["point_cloud"]["coord"][:CLOUD_POINTS],
        "offset": torch.tensor([CLOUD_POINTS]),
    }
    target = actions[:1]

    config = FinetuneConfig(steps=FINETUNE_STEPS, batch_size=8)
    pool = feature_pool(policy, batch, actions)
    losses = train_adapters(policy, pool_batches(pool, config.batch_size, config.seed), config)
    print(f"Fine-tuned {FINETUNE_STEPS} steps: noise loss {np.mean(losses[:10]):.3f} -> {np.mean(losses[-10:]):.3f}")

    print("Steering toward one demonstration, 8 DDIM steps, same seed:")
    print(f"{'w':>6} {'distance to the demonstration (cm)':>36}")
    demonstration = normalizer.denormalize(target.reshape(1, CHUNK_STEPS, NUM_POINTS, 3))
    for w in WEIGHTS:
        chunk = sample(policy, query, target, w).reshape(1, CHUNK_STEPS, NUM_POINTS, 3)
        error = (normalizer.denormalize(chunk) - demonstration).norm(dim=-1).mean().item()
        note = "  (w = 1 replays the demonstration)" if w == 1.0 else ""
        print(f"{w:6.1f} {100.0 * error:36.2f}{note}")


if __name__ == "__main__":
    main()
