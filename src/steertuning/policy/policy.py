"""The Steertuning policy: observation encoders, observation transformer, denoiser."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from steertuning.data.normalizer import ActionNormalizer
from steertuning.policy import checkpoint
from steertuning.policy.encoders import (
    ProprioEncoder,
    SigLIPTextEncoder,
    UtoniaEncoder,
    build_mask_encoder,
)
from steertuning.policy.point_encoders import PointNeXtEncoder, PTv3Encoder
from steertuning.policy.transformer import (
    DiffusionTransformer,
    ObservationEncoding,
    ObservationTransformer,
)
from steertuning.policy.unet import UNetDecoder
from steertuning.steering.sampler import build_scheduler, gaussian_noise, steered_ddim


def bf16_autocast(device: torch.device) -> torch.autocast:
    """The policy's precision outside pretraining: bf16 on a GPU, fp32 on a CPU."""
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@dataclass
class PolicyConfig:
    """Architecture values. See configs/policy/."""

    point_encoder: str = "utonia"
    decoder: str = "dit"
    action_dim: int = 156
    chunk_steps: int = 15
    width: int = 1024
    num_heads: int = 8
    observation_blocks: int = 3
    denoiser_blocks: int = 6
    ffn_dim: int = 2048
    dropout: float = 0.1
    token_dropout: float = 0.1
    time_dim: int = 256
    num_point_tokens: int = 512
    point_feature_dim: int = 1227
    mask_feature_dim: int = 512
    text_feature_dim: int = 768
    proprio_hidden_dim: int = 512
    proprio_noise_std: float = 0.04
    train_timesteps: int = 100
    utonia_repo: str = "Pointcept/Utonia"
    utonia_revision: str = field(kw_only=True)
    text_model: str = "google/siglip-base-patch16-224"
    text_revision: str = field(kw_only=True)
    point_encoder_args: dict = field(default_factory=dict)  # ablation encoders
    decoder_args: dict = field(default_factory=dict)  # ablation decoder


def build_point_encoder(config: PolicyConfig) -> nn.Module:
    """Point encoder: maps a collated cloud to features (B, K, point_feature_dim), valid (B, K)."""
    if config.point_encoder == "utonia":
        return UtoniaEncoder(config.num_point_tokens, config.utonia_repo, config.utonia_revision)
    if config.point_encoder == "pointnext":
        return PointNeXtEncoder(config.num_point_tokens, **config.point_encoder_args)
    if config.point_encoder == "ptv3":
        return PTv3Encoder(config.num_point_tokens, **config.point_encoder_args)
    raise ValueError(f"Unknown point encoder: {config.point_encoder}")


def build_decoder(config: PolicyConfig) -> nn.Module:
    """Denoiser: maps noisy actions, timesteps and an ObservationEncoding to predicted noise."""
    if config.decoder == "dit":
        return DiffusionTransformer(
            action_dim=config.action_dim,
            chunk_steps=config.chunk_steps,
            width=config.width,
            num_heads=config.num_heads,
            num_blocks=config.denoiser_blocks,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            time_dim=config.time_dim,
        )
    if config.decoder == "unet":
        return UNetDecoder(
            action_dim=config.action_dim,
            width=config.width,
            time_dim=config.time_dim,
            **config.decoder_args,
        )
    raise ValueError(f"Unknown decoder: {config.decoder}")


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over the valid action elements, averaged per sample. On predicted noise it is the training loss."""
    error = (prediction - target) ** 2
    per_sample = (error * mask).sum((1, 2)) / mask.sum((1, 2)).clamp(min=1)
    return per_sample.mean()


class SteertuningPolicy(nn.Module):
    """Noise predictor for action chunks, conditioned on four observation modalities.

    `point_encoder` and `text_encoder` can be injected, which allows running on CPU without
    the frozen models. See `build_point_encoder` for the point interface. A text encoder maps
    tokens (B, L) and mask (B, L) to features (B, text_feature_dim).
    """

    def __init__(
        self,
        config: PolicyConfig,
        point_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.normalizer = None
        width = config.width
        self.point_encoder = build_point_encoder(config) if point_encoder is None else point_encoder
        if text_encoder is None:
            text_encoder = SigLIPTextEncoder(config.text_model, config.text_revision)
        self.text_encoder = text_encoder
        self.mask_encoder = build_mask_encoder(config.mask_feature_dim)
        self.proprio_encoder = ProprioEncoder(config.proprio_hidden_dim, width, config.proprio_noise_std)
        self.point_proj = nn.Linear(config.point_feature_dim, width)
        self.mask_proj = nn.Linear(config.mask_feature_dim, width)
        self.text_proj = nn.Linear(config.text_feature_dim, width)
        self.token_norm = nn.LayerNorm(width)
        self.token_dropout = nn.Dropout(config.token_dropout)
        self.observation_transformer = ObservationTransformer(
            width=width,
            num_heads=config.num_heads,
            num_blocks=config.observation_blocks,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            num_point_tokens=config.num_point_tokens,
        )
        self.denoiser = build_decoder(config)
        self.scheduler = build_scheduler(config.train_timesteps)

    def encode_observation(self, batch: dict) -> ObservationEncoding:
        """Build Z = (P_e | O_e | S_e | D_e) and the context latents."""
        point_features, point_valid = self.point_encoder(batch["point_cloud"])
        point_tokens = self.point_proj(point_features)
        mask_token = self.mask_proj(self.mask_encoder(batch["object_mask"]))
        proprio_token = self.proprio_encoder(batch["proprio"])
        text_token = self.text_proj(self.text_encoder(batch["text_tokens"], batch["text_mask"]))

        other_tokens = torch.stack([mask_token, proprio_token, text_token], dim=1)
        tokens = torch.cat([point_tokens, other_tokens], dim=1)
        tokens = self.token_dropout(self.token_norm(tokens))
        valid = torch.cat([point_valid, point_valid.new_ones(other_tokens.shape[:2])], dim=1)
        return ObservationEncoding(tokens, valid, self.observation_transformer(tokens, valid))

    def denoise(
        self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, encoding: ObservationEncoding
    ) -> torch.Tensor:
        """Predict the noise in a chunk (B, 15, 156) at timesteps (B,)."""
        return self.denoiser(noisy_actions, timesteps, encoding)

    def noise_loss(
        self,
        encoding: ObservationEncoding,
        actions: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """The training loss: `masked_mse` on the noise predicted at random timesteps."""
        device = actions.device
        noise = torch.randn(actions.shape, device=device, generator=generator)
        timesteps = torch.randint(0, self.config.train_timesteps, (len(actions),), device=device, generator=generator)
        noisy_actions = self.scheduler.add_noise(actions, noise, timesteps)
        return masked_mse(self.denoise(noisy_actions, timesteps, encoding), noise, mask)

    def loss(self, batch: dict) -> torch.Tensor:
        """The training loss of one batch."""
        return self.noise_loss(self.encode_observation(batch), batch["actions"], batch["action_mask"])

    def forward(self, batch: dict) -> torch.Tensor:
        """The training loss, so that distributed wrappers see the call."""
        return self.loss(batch)

    @torch.no_grad()
    def sample(
        self,
        batch: dict,
        steering_target: torch.Tensor | None = None,
        steering_weight: float | torch.Tensor = 0.0,
        num_steps: int = 8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample a normalized action chunk (B, 15, 156). See `steered_ddim` for steering.

        `generator` drives the initial and target noise. The voxel subset uses the global torch RNG,
        so seed that too to repeat a sample.
        """
        encoding = self.encode_observation(batch)
        shape = (encoding.tokens.shape[0], self.config.chunk_steps, self.config.action_dim)
        return steered_ddim(
            lambda noisy, timesteps: self.denoise(noisy, timesteps, encoding),
            gaussian_noise(shape, encoding.tokens.device, generator),
            self.scheduler,
            num_steps,
            steering_target,
            steering_weight,
            generator,
        )

    def save_pretrained(self, path: str | Path) -> None:
        checkpoint.save_pretrained(self, path)

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        device: str | torch.device = "cuda",
        point_encoder: nn.Module | None = None,
        text_encoder: nn.Module | None = None,
    ) -> SteertuningPolicy:
        """Load a pretrained directory in eval mode, with its normalizer in `policy.normalizer`."""
        config = PolicyConfig(**checkpoint.read_config(path))
        policy = cls(config, point_encoder, text_encoder)
        checkpoint.load_weights(policy, path)
        policy.normalizer = ActionNormalizer.from_json(Path(path) / checkpoint.NORMALIZER_FILE)
        return policy.to(device).eval()
