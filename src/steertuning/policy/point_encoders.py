"""Point encoders for the encoder ablation: PointNeXt and PTv3.

Both take the collated cloud that `UtoniaEncoder` takes and return what it returns: features
(B, num_tokens, D) and valid (B, num_tokens), padded to the token budget. They start from random
weights and train during pretraining. LoRA injection freezes them with the rest of the policy.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from steertuning.policy.encoders import _select_voxels


def _split_clouds(values: Tensor, offset: Tensor) -> tuple[Tensor, ...]:
    return values.tensor_split(offset[:-1].tolist())


@torch.no_grad()
def farthest_point_sample(xyz: Tensor, count: int) -> Tensor:
    """Indices (B, count) of well-spread points of a cloud (B, N, 3), from a random start."""
    batch, num_points, _ = xyz.shape
    index = torch.zeros(batch, count, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, num_points), torch.inf, device=xyz.device)
    farthest = torch.randint(0, num_points, (batch,), device=xyz.device)
    rows = torch.arange(batch, device=xyz.device)
    for i in range(count):
        index[:, i] = farthest
        center = xyz[rows, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((xyz - center) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return index


@torch.no_grad()
def nearest_neighbors(query: Tensor, support: Tensor, k: int) -> Tensor:
    """Indices (B, M, k) of the nearest support points. Chunks of 256 queries keep the distance matrix small."""
    k = min(k, support.shape[1])
    indices = [torch.cdist(block, support).topk(k, dim=-1, largest=False).indices for block in query.split(256, dim=1)]
    return torch.cat(indices, dim=1)


def index_points(points: Tensor, index: Tensor) -> Tensor:
    """Gather (B, N, C) by indices (B, ...) into (B, ..., C)."""
    batch, _, channels = points.shape
    gathered = points.gather(1, index.reshape(batch, -1, 1).expand(-1, -1, channels))
    return gathered.view(*index.shape, channels)


class _ChannelNorm(nn.Module):
    """BatchNorm over the last axis of a tensor of any shape."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.reshape(-1, x.shape[-1])).view(x.shape)


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), _ChannelNorm(out_dim), nn.ReLU(inplace=True))


class LocalAggregation(nn.Module):
    """MLP over neighbor features and relative positions, reduced by max-pooling.

    Relative positions are divided by the mean neighbor distance, since PointNeXt normalizes them and
    raw meters are too small for the MLP to use.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = _mlp(in_dim + 3, out_dim)

    def forward(self, center_xyz: Tensor, support_xyz: Tensor, support: Tensor, neighbors: Tensor) -> Tensor:
        relative = index_points(support_xyz, neighbors) - center_xyz.unsqueeze(2)
        relative = relative / relative.norm(dim=-1).mean(dim=(1, 2), keepdim=True).unsqueeze(-1).clamp(min=1e-3)
        grouped = torch.cat([index_points(support, neighbors), relative], dim=-1)
        return self.mlp(grouped).amax(dim=2)


class InvertedResidualMLP(nn.Module):
    """PointNeXt's block: aggregate, expand, project, add."""

    def __init__(self, dim: int, expansion: int) -> None:
        super().__init__()
        self.aggregate = LocalAggregation(dim, dim)
        self.expand = _mlp(dim, dim * expansion)
        self.project = nn.Sequential(nn.Linear(dim * expansion, dim, bias=False), _ChannelNorm(dim))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, xyz: Tensor, x: Tensor, neighbors: Tensor) -> Tensor:
        h = self.aggregate(xyz, xyz, x, neighbors)
        return self.activation(x + self.project(self.expand(h)))


class PointNeXtStage(nn.Module):
    """Set abstraction, then inverted residual blocks over one neighbor index."""

    def __init__(self, in_dim: int, out_dim: int, stride: int, blocks: int, neighbors: int, expansion: int) -> None:
        super().__init__()
        self.stride = stride
        self.neighbors = neighbors
        self.abstraction = LocalAggregation(in_dim, out_dim)
        self.blocks = nn.ModuleList(InvertedResidualMLP(out_dim, expansion) for _ in range(blocks))

    def forward(self, xyz: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        centers = farthest_point_sample(xyz, max(xyz.shape[1] // self.stride, 1))
        center_xyz = index_points(xyz, centers)
        neighbors = nearest_neighbors(center_xyz, xyz, self.neighbors)
        x = self.abstraction(center_xyz, xyz, x, neighbors)
        neighbors = nearest_neighbors(center_xyz, center_xyz, self.neighbors)
        for block in self.blocks:
            x = block(center_xyz, x, neighbors)
        return center_xyz, x


class PointNeXtEncoder(nn.Module):
    """PointNeXt with kNN neighborhoods. Self-contained: it needs PyTorch only.

    The input is rgb + xyz of each point. A token is one point of the last stage: its features
    and its xyz. The token count follows from `num_points` and `strides`.
    """

    def __init__(
        self,
        num_tokens: int = 512,
        num_points: int = 5000,
        width: int = 48,
        strides: tuple[int, ...] = (4, 4, 2),
        blocks: tuple[int, ...] = (3, 5, 3),
        neighbors: int = 24,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_points = num_points
        self.stem = _mlp(6, width)
        self.stages = nn.ModuleList(
            PointNeXtStage(width * 2**i, width * 2 ** (i + 1), stride, count, neighbors, expansion)
            for i, (stride, count) in enumerate(zip(strides, blocks))
        )

    def forward(self, point_cloud: dict) -> tuple[Tensor, Tensor]:
        xyz, color = self._fixed_size_clouds(point_cloud)
        x = self.stem(torch.cat([color, xyz], dim=-1))
        for stage in self.stages:
            xyz, x = stage(xyz, x)
        batch, count, _ = x.shape
        offsets = [count * (i + 1) for i in range(batch)]
        features = torch.cat([x, xyz], dim=-1).flatten(0, 1)
        return _select_voxels(features, offsets, self.num_tokens)

    def _fixed_size_clouds(self, point_cloud: dict) -> tuple[Tensor, Tensor]:
        """Stack to (B, num_points, 3). Voxelization varies the count, so short clouds repeat points.

        Farthest point sampling picks a repeated point only after every distinct one, so the
        tokens stay distinct.
        """
        offset = point_cloud["offset"]
        coords, colors = [], []
        for xyz, color in zip(_split_clouds(point_cloud["coord"], offset), _split_clouds(point_cloud["color"], offset)):
            count = len(xyz)
            kept = torch.randperm(count, device=xyz.device)[: self.num_points]
            repeated = torch.randint(count, (max(self.num_points - count, 0),), device=xyz.device)
            index = torch.cat([kept, repeated])
            coords.append(xyz[index])
            colors.append(color[index])
        return torch.stack(coords), torch.stack(colors)


class PTv3Encoder(nn.Module):
    """Point Transformer V3, from its upstream module, with the encoder stages only.

    The input is the nine channels of the collated cloud (xyz, rgb, zero normals) on a metric
    voxel grid. A token is one voxel of the final stage: its features and its mean xyz, so the
    token count follows from the scene size. `backbone_args` go to the upstream
    `PointTransformerV3` constructor. Attention runs through FlashAttention: upstream's plain PyTorch
    path sizes its attention patches by the smallest cloud in the batch, so a sample would depend on the
    rest of its batch.
    """

    def __init__(self, num_tokens: int = 512, grid_size: float = 0.01, **backbone_args) -> None:
        from PointTransformerV3.model import PointTransformerV3

        super().__init__()
        self.num_tokens = num_tokens
        self.grid_size = grid_size
        self.backbone = PointTransformerV3(cls_mode=True, enable_flash=True, **backbone_args)

    def forward(self, point_cloud: dict) -> tuple[Tensor, Tensor]:
        # Its sparse convolutions allocate on the current CUDA device and take fp32 features.
        with torch.cuda.device(point_cloud["coord"].device), torch.autocast("cuda", enabled=False):
            point = self.backbone(self._voxelize(point_cloud))
        features = torch.cat([point.feat, point.coord], dim=-1)
        return _select_voxels(features, point.offset.tolist(), self.num_tokens)

    def _voxelize(self, point_cloud: dict) -> dict:
        """Keep the first point of each voxel. Sparse convolutions need unique voxels per cloud."""
        offset = point_cloud["offset"]
        coords, features, grids, counts = [], [], [], []
        for xyz, feat in zip(_split_clouds(point_cloud["coord"], offset), _split_clouds(point_cloud["feat"], offset)):
            grid = ((xyz - xyz.amin(0)) / self.grid_size).floor().int()
            voxels, inverse = grid.unique(dim=0, return_inverse=True)
            index = inverse.new_full((len(voxels),), len(grid)).scatter_reduce(
                0, inverse, torch.arange(len(grid), device=grid.device), "amin"
            )
            coords.append(xyz[index])
            features.append(feat[index])
            grids.append(voxels)
            counts.append(len(voxels))
        return {
            "coord": torch.cat(coords),
            "feat": torch.cat(features),
            "grid_coord": torch.cat(grids),
            "offset": torch.tensor(counts, device=offset.device).cumsum(0),
        }
