"""Modality encoders: point cloud, object mask, proprioception, task description."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18


class UtoniaEncoder(nn.Module):
    """Frozen Utonia features at a random subset of voxels.

    A voxel's feature is the concatenation of the three coarsest encoder levels (1224) and its
    xyz (3). Each call draws a fresh subset. Samples with fewer voxels are padded and marked
    invalid. Returns features (B, num_tokens, 1227) and valid (B, num_tokens).
    """

    frozen = True

    def __init__(self, num_tokens: int, repo_id: str, revision: str) -> None:
        import utonia
        from huggingface_hub import hf_hub_download

        super().__init__()
        self.num_tokens = num_tokens
        # `utonia.load` by name always takes the repository's newest file, and by path takes this one.
        checkpoint = hf_hub_download(repo_id, "utonia.pth", revision=revision)
        self.model = utonia.load(checkpoint).eval().requires_grad_(False)

    def train(self, mode: bool = True) -> UtoniaEncoder:
        return super().train(False)

    @torch.no_grad()
    def forward(self, point_cloud: dict) -> tuple[torch.Tensor, torch.Tensor]:
        # Its sparse convolutions allocate on the current CUDA device.
        with torch.cuda.device(point_cloud["coord"].device):
            point = self.model(point_cloud)
        point = _unpool_features(point)
        features = torch.cat([point.feat, point.coord], dim=-1)
        return _select_voxels(features, point.offset.tolist(), self.num_tokens)


def _unpool_features(point):
    """Walk from the coarsest level up to the voxels: two concatenating steps, then broadcasts."""
    level = 0
    while "pooling_parent" in point:
        parent = point.pop("pooling_parent")
        upsampled = point.feat[point.pop("pooling_inverse")]
        parent.feat = torch.cat([parent.feat, upsampled], dim=-1) if level < 2 else upsampled
        point = parent
        level += 1
    return point


def _select_voxels(features: torch.Tensor, offsets: list[int], num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    selected = features.new_zeros(len(offsets), num_tokens, features.shape[1])
    valid = torch.zeros(len(offsets), num_tokens, dtype=torch.bool, device=features.device)
    start = 0
    for i, end in enumerate(offsets):
        count = end - start
        if count > num_tokens:
            index = start + torch.randperm(count, device=features.device)[:num_tokens]
        else:
            index = torch.arange(start, end, device=features.device)
        selected[i, : len(index)] = features[index]
        valid[i, : len(index)] = True
        start = end
    return selected, valid


def build_mask_encoder(out_dim: int = 512) -> nn.Module:
    """ResNet-18 for the single-channel object mask, trained from scratch."""
    model = resnet18(num_classes=out_dim)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    return model


class ProprioEncoder(nn.Module):
    """FFN on the fingertip points S (B, 2, 5, 3), with Gaussian noise while training."""

    def __init__(self, hidden_dim: int, out_dim: int, noise_std: float) -> None:
        super().__init__()
        self.noise_std = noise_std
        self.net = nn.Sequential(
            nn.Linear(30, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        if self.training:
            proprio = proprio + self.noise_std * torch.randn_like(proprio)
        return self.net(proprio.flatten(1))


class SigLIPTextEncoder(nn.Module):
    """Frozen SigLIP text model: L2-normalized pooler output, cached per prompt."""

    frozen = True

    def __init__(self, model_name: str, revision: str) -> None:
        from transformers import SiglipTextModel

        super().__init__()
        self.text_model = SiglipTextModel.from_pretrained(model_name, revision=revision).text_model
        self.text_model.eval().requires_grad_(False)
        self._cache: dict[bytes, torch.Tensor] = {}

    def train(self, mode: bool = True) -> SigLIPTextEncoder:
        return super().train(False)

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        keys = [ids.tobytes() + attention.tobytes() for ids, attention in zip(tokens.cpu().numpy(), mask.cpu().numpy())]
        missing = [i for i, key in enumerate(keys) if key not in self._cache]
        if missing:
            output = self.text_model(input_ids=tokens[missing], attention_mask=mask[missing])
            features = F.normalize(output.pooler_output, dim=-1).float().cpu()
            self._cache.update({keys[i]: feature for i, feature in zip(missing, features)})
        return torch.stack([self._cache[key] for key in keys]).to(tokens.device)
