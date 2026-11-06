"""Small stand-ins for the frozen encoders, to run the policy on CPU with no weights or network."""

from __future__ import annotations

import torch
from torch import nn


class StubPointEncoder(nn.Module):
    """Stands in for Utonia: a fixed random projection of the first points of each collated cloud."""

    frozen = True

    def __init__(self, num_tokens: int, feature_dim: int) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.register_buffer("projection", torch.randn(3, feature_dim))

    def forward(self, point_cloud: dict) -> tuple[torch.Tensor, torch.Tensor]:
        coord, offset = point_cloud["coord"], point_cloud["offset"].tolist()
        starts = [0, *offset[:-1]]
        tokens = [coord[start:end][: self.num_tokens] for start, end in zip(starts, offset)]
        features = coord.new_zeros(len(tokens), self.num_tokens, self.projection.shape[1])
        valid = torch.zeros(len(tokens), self.num_tokens, dtype=torch.bool, device=coord.device)
        for i, points in enumerate(tokens):
            features[i, : len(points)] = torch.tanh(points @ self.projection)
            valid[i, : len(points)] = True
        return features, valid


class StubTextEncoder(nn.Module):
    """Stands in for SigLIP: a fixed random projection of the token ids."""

    frozen = True

    def __init__(self, num_tokens: int, feature_dim: int) -> None:
        super().__init__()
        self.register_buffer("projection", torch.randn(num_tokens, feature_dim))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.tanh((tokens * mask).float() / 1000.0 @ self.projection)
