"""Steer your own diffusion policy toward a demonstration, with no policy class.

    python examples/steer_your_policy.py

`steered_ddim` needs one callable, which maps a noisy chunk and its timesteps to predicted noise.
"""

import torch

from steertuning.steering.sampler import build_scheduler, steered_ddim

SCHEDULER = build_scheduler()
DEMONSTRATION = torch.ones(1, 15, 156)  # the steering action a*, in the policy's normalized units


def denoise(noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """A toy policy whose clean prediction is always the zero chunk. Your noise predictor goes here."""
    return noisy / (1 - SCHEDULER.alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1)


if __name__ == "__main__":
    for weight in (0.0, 0.3, 1.0):
        generator = torch.Generator().manual_seed(0)
        noise = torch.randn(DEMONSTRATION.shape, generator=generator)
        chunk = steered_ddim(denoise, noise, SCHEDULER, 8, DEMONSTRATION, weight, generator)
        print(f"w = {weight:.1f}: distance to the demonstration {(chunk - DEMONSTRATION).abs().mean():.2f}")
