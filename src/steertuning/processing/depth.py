"""Metric depth from the rectified stereo pair with S2M2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# S2M2's weights file for each model size.
WEIGHTS_FILES = {"S": "CH128NTR1.pth", "M": "CH192NTR2.pth", "L": "CH256NTR3.pth", "XL": "CH384NTR3.pth"}


class StereoDepth:
    """S2M2 through aria2mano's depth provider."""

    def __init__(self, device: str, model_type: str, num_refine: int, model_dir: str | Path | None = None) -> None:
        from aria2mano.utils.data_keys import S2M2_MODELS_FOLDER
        from aria2mano.utils.depth_provider import DepthProvider

        folder = Path(model_dir) if model_dir else S2M2_MODELS_FOLDER
        if not (folder / WEIGHTS_FILES[model_type]).is_file():
            raise FileNotFoundError(
                f"No S2M2 weights at {folder / WEIGHTS_FILES[model_type]}. Download them as the README says."
            )
        self.provider = DepthProvider(
            model_path=folder,
            model_type=model_type,
            num_refine=num_refine,
            device=device,
        )

    def __call__(
        self, left: np.ndarray, right: np.ndarray, focal_length: float, baseline: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Depth (H, W) in uint16 millimeters and a validity mask (H, W), 255 where valid."""
        return self.provider.compute_depth_from_stereo(left, right, focal_length, baseline)
