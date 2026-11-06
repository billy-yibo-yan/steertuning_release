# Adapted from DiT-Policy, data4robotics/models/diffusion_unet.py
# (https://github.com/SudeepDasari/dit-policy), whose U-Net comes from Diffusion Policy
# (Chi et al., https://github.com/real-stanford/diffusion_policy).
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# Licensed under the MIT license; see THIRD_PARTY_NOTICES.md.
"""Temporal UNet for the decoder ablation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from steertuning.policy.transformer import ObservationEncoding, TimeEmbedding


def _conv_block(in_channels: int, channels: int, kernel_size: int, groups: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_channels, channels, kernel_size, padding=kernel_size // 2),
        nn.GroupNorm(groups, channels),
        nn.Mish(),
    )


class ResidualBlock(nn.Module):
    """Two convolutions with FiLM between them: scale and shift from the condition, as AdaLN does."""

    def __init__(self, in_channels: int, channels: int, condition_dim: int, kernel_size: int, groups: int) -> None:
        super().__init__()
        self.conv1 = _conv_block(in_channels, channels, kernel_size, groups)
        self.conv2 = _conv_block(channels, channels, kernel_size, groups)
        self.modulation = nn.Linear(condition_dim, 2 * channels)
        self.residual = nn.Conv1d(in_channels, channels, 1) if in_channels != channels else nn.Identity()

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(F.mish(condition))[:, :, None].chunk(2, dim=1)
        h = scale * self.conv1(x) + shift
        return self.conv2(h) + self.residual(x)


class UNetDecoder(nn.Module):
    """Noise predictor with the interface of `DiffusionTransformer`.

    Every level has a skip. A chunk is 15 x 156 = 2340 numbers and the first downsample holds
    256 x 8 = 2048, so the top skip carries what the rest of the network cannot.

    The observation reaches the UNet through the context latents, with no cross-attention to the tokens. The
    condition is a context plus the time embedding, as in the transformer: level i of the down path uses
    contexts[min(i, last)], and the rest of the network uses the last context.
    """

    def __init__(
        self,
        action_dim: int,
        width: int,
        time_dim: int,
        channels: tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 3,
        groups: int = 8,
    ) -> None:
        super().__init__()

        def block(in_channels: int, out_channels: int) -> ResidualBlock:
            return ResidualBlock(in_channels, out_channels, width, kernel_size, groups)

        self.time_embedding = TimeEmbedding(time_dim, width)
        levels = list(zip([action_dim, *channels[:-1]], channels))
        self.down = nn.ModuleList()
        for level, (in_channels, out_channels) in enumerate(levels):
            is_last = level == len(levels) - 1
            down = nn.Identity() if is_last else nn.Conv1d(out_channels, out_channels, 3, 2, 1)
            self.down.append(nn.ModuleList([block(in_channels, out_channels), block(out_channels, out_channels), down]))
        self.middle = nn.ModuleList([block(channels[-1], channels[-1]) for _ in range(2)])
        self.up = nn.ModuleList(
            nn.ModuleList(
                [
                    block(2 * out_channels, in_channels),
                    block(in_channels, in_channels),
                    nn.ConvTranspose1d(in_channels, in_channels, 4, 2, 1),
                ]
            )
            for in_channels, out_channels in reversed(levels[1:])
        )
        self.final = nn.Sequential(
            _conv_block(2 * channels[0], channels[0], kernel_size, groups),
            nn.Conv1d(channels[0], action_dim, 1),
        )

    def forward(self, noisy_actions: Tensor, timesteps: Tensor, encoding: ObservationEncoding) -> Tensor:
        x = noisy_actions.transpose(1, 2)
        time = self.time_embedding(timesteps)
        contexts = encoding.contexts
        skips = []
        for level, (block1, block2, down) in enumerate(self.down):
            condition = contexts[min(level, len(contexts) - 1)] + time
            x = block2(block1(x, condition), condition)
            skips.append(x)
            x = down(x)
        condition = contexts[-1] + time
        for block in self.middle:
            x = block(x, condition)
        for block1, block2, up in self.up:
            x = torch.cat((x, skips.pop()), dim=1)
            x = up(block2(block1(x, condition), condition))
        # 15 rows upsample to 16, and the extra row is dropped.
        x = torch.cat((x[..., : noisy_actions.shape[1]], skips.pop()), dim=1)
        return self.final(x).transpose(1, 2)
