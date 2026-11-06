"""SAM3 latents for the navigation map.

A voxel's latent is the SAM3 image feature of the pixels that fall into it. A target's latent
is the mean image feature inside the mask SAM3 returns for the target's name on the keyframes.
SAM3 matches text to images through its fusion encoder, so its text features cannot be compared
to stored image features directly; the mask turns the name into an image-feature query.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from steertuning.processing.masks import best_query_mask, load_image_model, noun_phrase

LATENT_DIM = 256


@dataclass
class TargetLatent:
    latent: np.ndarray  # (256,) unit length
    score: float  # SAM3 query score of the chosen mask


class Sam3Latents:
    """SAM3 image features and text-prompted masks. SAM3 needs a CUDA device."""

    def __init__(self, device: str, checkpoint: str | None) -> None:
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = torch.device(device)
        self.model = load_image_model(self.device, checkpoint)
        self.processor = Sam3Processor(self.model, device=str(self.device))

    def _encode(self, rgb: np.ndarray) -> dict:
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            return self.processor.set_image(Image.fromarray(rgb))

    @staticmethod
    def _unit_features(state: dict) -> torch.Tensor:
        return F.normalize(state["backbone_out"]["vision_features"][0].float(), dim=0)

    @torch.inference_mode()
    def pixel_latents(self, rgb: np.ndarray) -> torch.Tensor:
        """Unit-length image features of one RGB frame, (256, 72, 72)."""
        return self._unit_features(self._encode(rgb))

    @staticmethod
    @torch.inference_mode()
    def sample(latents: torch.Tensor, pixels: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        """Bilinear latents at `pixels` (N, 2) as (col, row) of an image of `image_size` (H, W)."""
        height, width = image_size
        scale = torch.tensor([2.0 / width, 2.0 / height], device=latents.device)
        grid = (torch.as_tensor(pixels, dtype=torch.float32, device=latents.device) + 0.5) * scale - 1.0
        sampled = F.grid_sample(latents[None], grid[None, None], align_corners=False)
        return F.normalize(sampled[0, :, 0].T, dim=1).cpu().numpy()

    @torch.inference_mode()
    def target_latent(self, keyframes: Sequence[np.ndarray], target: str, min_score: float) -> TargetLatent | None:
        """Mean image feature inside the best mask for `target` over `keyframes`, or None."""
        prompt = noun_phrase(target)
        best = None
        for rgb in keyframes:
            state = self._encode(rgb)
            logits, score = best_query_mask(self.model, self.processor, state, prompt, self.device)
            if score >= min_score and (best is None or score > best.score):
                best = TargetLatent(self._mean_in_mask(self._unit_features(state), logits), score)
        return best

    @staticmethod
    def _mean_in_mask(features: torch.Tensor, logits: torch.Tensor) -> np.ndarray:
        size = features.shape[1:]
        coverage = F.interpolate((logits > 0).float()[None, None], size=size, mode="area")[0, 0]
        # A mask smaller than one feature cell still selects the cells it overlaps most.
        inside = coverage >= min(0.5, float(coverage.max()))
        return F.normalize(features[:, inside].mean(dim=1), dim=0).cpu().numpy()
