"""DDIM sampling with action steering."""

from __future__ import annotations

from collections.abc import Callable

import torch
from diffusers import DDIMScheduler


def build_scheduler(num_train_timesteps: int = 100) -> DDIMScheduler:
    """Cosine-schedule DDIM scheduler for an epsilon predictor."""
    return DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        set_alpha_to_one=True,
        steps_offset=0,
        timestep_spacing="leading",
        prediction_type="epsilon",
    )


def gaussian_noise(
    shape: tuple[int, ...], device: torch.device, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Standard normal noise on `device`, drawn from a generator on any device."""
    draw_device = device if generator is None else generator.device
    return torch.randn(shape, generator=generator, device=draw_device).to(device)


def steered_ddim(
    denoise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    scheduler,
    num_steps: int,
    target: torch.Tensor | None = None,
    weight: float | torch.Tensor = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Run deterministic DDIM from `x`, steering every step toward `target`.

    Each DDIM step brings the sample to the next timestep's noise level. There the target is noised
    to that level with a fresh draw and blended into the sample, before sampling continues.
    The noising also applies at the level of t = 0, the last timestep, where alpha_bar is still below 1. The
    blend with the clean target comes after the last step, where alpha_bar is 1. So w = 1 returns the
    target exactly, and w = 0 is plain DDIM. The first timestep's level stays as drawn.

    Args:
        denoise: maps a noisy chunk (B, 15, 156) and timesteps (B,) to predicted noise.
        x: initial noise (B, 15, 156).
        scheduler: scheduler from `build_scheduler`.
        num_steps: number of DDIM steps.
        target: steering action a* in normalized units, (B or 1, 15, 156).
        weight: steering weight w, a float or a per-row tensor (B or 1, 15, 1).
        generator: source of the target noise. Defaults to the global generator.
    """
    scheduler.set_timesteps(num_steps)
    timesteps = scheduler.timesteps
    weight = torch.as_tensor(weight, device=x.device, dtype=x.dtype)
    for i, t in enumerate(timesteps):
        noise_pred = denoise(x, t.repeat(x.shape[0]).to(x.device))
        x = scheduler.step(noise_pred, t, x).prev_sample
        if target is None:
            continue
        noisy_target = target
        if i + 1 < len(timesteps):
            alpha_bar = scheduler.alphas_cumprod[timesteps[i + 1]]
            noise = gaussian_noise(x.shape, x.device, generator)
            noisy_target = alpha_bar.sqrt() * target + (1 - alpha_bar).sqrt() * noise
        x = weight * noisy_target + (1 - weight) * x
    return x
