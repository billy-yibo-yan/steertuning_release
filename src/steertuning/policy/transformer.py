# Adapted from DiT-Policy, data4robotics/models/diffusion.py
# (https://github.com/SudeepDasari/dit-policy).
# Copyright (c) Sudeep Dasari, 2023. Licensed under the MIT license; see THIRD_PARTY_NOTICES.md.
"""Observation transformer and diffusion transformer.

All sequences are batch-first, (B, T, D).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ObservationEncoding:
    """Encoded observation that conditions the denoiser.

    tokens: observation-token sequence Z, before self-attention, (B, N, D).
    valid: False on padded point tokens, (B, N).
    contexts: one context latent per observation-transformer stage, each (B, D).
    """

    tokens: Tensor
    valid: Tensor
    contexts: list[Tensor]


def sinusoidal_positions(start: int, count: int, dim: int) -> Tensor:
    """Sine and cosine positions for indices start, ..., start + count - 1, shape (count, dim)."""
    index = torch.arange(start, start + count, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))
    positions = torch.zeros(count, dim)
    positions[:, 0::2] = torch.sin(index * frequency)
    positions[:, 1::2] = torch.cos(index * frequency)
    return positions


def _masked_mean(tokens: Tensor, valid: Tensor) -> Tensor:
    weights = valid.to(tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def _xavier_init(module: nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)


class Attention(nn.Module):
    """Multi-head attention with per-head LayerNorm on queries and keys.

    The scores keep the usual 1 / sqrt(d) scale, not an L2 normalization with a learned scale.
    """

    def __init__(self, width: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.q_norm = nn.LayerNorm(width // num_heads)
        self.k_norm = nn.LayerNorm(width // num_heads)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, valid: Tensor | None = None) -> Tensor:
        # The softmax over Q K^T is sensitive to rounding: attention stays in fp32 under autocast.
        with torch.autocast(query.device.type, enabled=False):
            q = self.q_norm(self._split_heads(self.q_proj(query.float())))
            k = self.k_norm(self._split_heads(self.k_proj(key.float())))
            v = self._split_heads(self.v_proj(value.float()))
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if valid is None else valid[:, None, None, :],
                dropout_p=self.dropout if self.training else 0.0,
            )
            return self.out_proj(out.transpose(1, 2).flatten(2))

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        return x.view(batch, length, self.num_heads, width // self.num_heads).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(width, hidden_dim)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class ObservationBlock(nn.Module):
    """Pre-norm self-attention block. Positions join queries and keys, not values."""

    def __init__(self, width: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(width)
        self.attn = Attention(width, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = FeedForward(width, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        _xavier_init(self)

    def forward(self, x: Tensor, positions: Tensor, valid: Tensor) -> Tensor:
        normed = self.attn_norm(x)
        query_key = normed + positions
        x = x + self.dropout(self.attn(query_key, query_key, normed, valid))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class ObservationTransformer(nn.Module):
    """Self-attention over Z that yields one context latent per stage.

    The context latent of a stage is the mean of the valid tokens after its observation block,
    through `stage_norm` and then the stage's own learned linear layer. Denoiser block i uses stage
    min(i, number of observation blocks - 1). Sinusoidal positions apply to the three non-point
    tokens only: mask, proprioception, task description.
    """

    def __init__(
        self,
        width: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        dropout: float,
        num_point_tokens: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ObservationBlock(width, num_heads, ffn_dim, dropout) for _ in range(num_blocks))
        # The trained weights have both LayerNorms, shared by all stages.
        self.stage_norm = nn.Sequential(nn.LayerNorm(width), nn.LayerNorm(width))
        self.context_proj = nn.ModuleList(nn.Linear(width, width) for _ in range(num_blocks))
        for proj in self.context_proj:
            nn.init.eye_(proj.weight)
            nn.init.zeros_(proj.bias)
        self.num_point_tokens = num_point_tokens
        self.register_buffer("positions", sinusoidal_positions(num_point_tokens, 3, width), persistent=False)

    def forward(self, tokens: Tensor, valid: Tensor) -> list[Tensor]:
        positions = F.pad(self.positions, (0, 0, self.num_point_tokens, 0))
        contexts = []
        x = tokens
        for block, proj in zip(self.blocks, self.context_proj):
            x = block(x, positions, valid)
            contexts.append(proj(_masked_mean(self.stage_norm(x), valid)))
        return contexts


class TimeEmbedding(nn.Module):
    """Cosine and sine features of the timestep at fixed frequencies, followed by an MLP."""

    def __init__(self, dim: int, width: int) -> None:
        super().__init__()
        half = dim // 2
        frequencies = torch.exp(torch.arange(half) * -(math.log(10000.0) / (half - 1))).float()
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, timesteps: Tensor) -> Tensor:
        phase = timesteps[:, None] * self.frequencies[None]
        return self.mlp(torch.cat((torch.cos(phase), torch.sin(phase)), dim=1))


class AdaLN(nn.Module):
    """Scale and shift the normalized input of a branch from the condition."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Linear(width, width)
        self.shift = nn.Linear(width, width)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        c = F.silu(condition)
        return x * self.scale(c)[:, None] + self.shift(c)[:, None]


class Gate(nn.Module):
    """Scale the output of a branch from the condition. `DenoiserBlock` zeroes it, so a new block is the identity."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.scale = nn.Linear(width, width)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        return x * self.scale(F.silu(condition))[:, None]


class DenoiserBlock(nn.Module):
    """Self-attention, cross-attention to Z, and FFN, each with AdaLN conditioning."""

    def __init__(self, width: int, num_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_mod = AdaLN(width)
        self.self_attn = Attention(width, num_heads, dropout)
        self.self_gate = Gate(width)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_mod = AdaLN(width)
        self.cross_attn = Attention(width, num_heads, dropout)
        self.cross_gate = Gate(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn_mod = AdaLN(width)
        self.ffn = FeedForward(width, ffn_dim, dropout)
        self.ffn_gate = Gate(width)
        self.dropout = nn.Dropout(dropout)

        _xavier_init(self)
        for mod in (self.self_mod, self.cross_mod, self.ffn_mod):
            nn.init.zeros_(mod.scale.bias)
            nn.init.zeros_(mod.shift.bias)
        for gate in (self.self_gate, self.cross_gate, self.ffn_gate):
            nn.init.zeros_(gate.scale.weight)
            nn.init.zeros_(gate.scale.bias)

    def forward(self, x: Tensor, condition: Tensor, tokens: Tensor, valid: Tensor) -> Tensor:
        h = self.self_mod(self.self_norm(x), condition)
        x = x + self.self_gate(self.dropout(self.self_attn(h, h, h)), condition)
        h = self.cross_mod(self.cross_norm(x), condition)
        x = x + self.cross_gate(self.dropout(self.cross_attn(h, tokens, tokens, valid)), condition)
        h = self.ffn_mod(self.ffn_norm(x), condition)
        return x + self.ffn_gate(self.dropout(self.ffn(h)), condition)


class FinalLayer(nn.Module):
    """The final FFN: AdaLN modulation and one linear layer, from action tokens to noise."""

    def __init__(self, width: int, action_dim: int) -> None:
        super().__init__()
        self.modulation = nn.Linear(width, 2 * width)
        self.linear = nn.Linear(width, action_dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(F.silu(condition)).chunk(2, dim=1)
        return self.linear(x * scale[:, None] + shift[:, None])


class DiffusionTransformer(nn.Module):
    """Noise predictor over one token per chunk step.

    Block i is conditioned on contexts[min(i, last)] plus the time embedding. The final layer
    uses the last context. All blocks cross-attend to Z.
    """

    def __init__(
        self,
        action_dim: int,
        chunk_steps: int,
        width: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        dropout: float,
        time_dim: int,
    ) -> None:
        super().__init__()
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, width),
        )
        self.action_positions = nn.Parameter(torch.empty(1, chunk_steps, width))
        nn.init.xavier_uniform_(self.action_positions)
        self.time_embedding = TimeEmbedding(time_dim, width)
        self.blocks = nn.ModuleList(DenoiserBlock(width, num_heads, ffn_dim, dropout) for _ in range(num_blocks))
        self.final_layer = FinalLayer(width, action_dim)

    def forward(self, noisy_actions: Tensor, timesteps: Tensor, encoding: ObservationEncoding) -> Tensor:
        x = self.action_proj(noisy_actions) + self.action_positions
        time = self.time_embedding(timesteps)
        contexts = encoding.contexts
        for i, block in enumerate(self.blocks):
            condition = contexts[min(i, len(contexts) - 1)] + time
            x = block(x, condition, encoding.tokens, encoding.valid)
        return self.final_layer(x, contexts[-1] + time)
