"""The robot's observation in the training format.

Rectified stereo gives metric depth with S2M2. Depth pixels become a colored cloud in the RGB camera
frame. The robot is removed with boxes posed by its joint positions, the workspace crop is applied,
and the cloud is resampled to 5,000 points.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from steertuning.data.dataset import TextTokenizer, to_batch
from steertuning.data.representation import (
    MASK_SIZE,
    cloud_transform,
    crop_and_resample,
    in_ellipsoid,
    utonia_input,
)
from steertuning.processing.masks import best_query_mask, load_image_model, noun_phrase, select_device
from steertuning.processing.point_cloud import colored_cloud
from steertuning.remote import TopicReader

DEPTH_STRIDE = 2
META_TIMEOUT = 30.0  # seconds to wait for the stream's first message
FRAME_TIMEOUT = 2.0  # seconds to wait for a new frame


@dataclass
class CameraFrame:
    """cloud: (N, 6) float32, xyz in the RGB camera frame, rgb 0-255. rgb: (H, W, 3) uint8, or None."""

    cloud: np.ndarray
    rgb: np.ndarray | None


class Camera(Protocol):
    """Receiving and decoding are separate, so joints can be read at the moment an image arrives."""

    def receive(self) -> Any:
        """The newest message. Raises TimeoutError if none arrives in time."""

    def frame(self, message: Any) -> CameraFrame:
        """The colored cloud and image of a message."""


class AriaCamera:
    """The Aria stream published by aria2mano, on another computer. Depth runs on this one."""

    def __init__(self, host: str, port: int) -> None:
        """Connects to the stream. Set `depth` to the stereo model before calling `frame`."""
        self.depth = None
        self.images = TopicReader(host, port, "images")
        self.calibration = TopicReader(host, port, "online_calib")
        self.meta = TopicReader(host, port, "meta").read(META_TIMEOUT)
        self.calibration.read(META_TIMEOUT)
        if not (self.meta["has_rgb"] and self.meta["has_stereo"]):
            raise ValueError("The Aria stream must publish RGB and stereo images.")
        self.last_stamp = None

    def receive(self) -> dict:
        deadline = time.monotonic() + FRAME_TIMEOUT
        while (remaining := deadline - time.monotonic()) > 0:
            message = self.images.read(remaining)
            if message["t_pub"] != self.last_stamp:
                self.last_stamp = message["t_pub"]
                return message
            time.sleep(0.002)
        raise TimeoutError("No new frame from the Aria stream.")

    def rgb(self, message: dict) -> np.ndarray:
        """The RGB image: the publisher encodes it with OpenCV as it is, so it decodes to the same channels."""
        return _decode(message["rgb"], cv2.IMREAD_COLOR)

    def frame(self, message: dict) -> CameraFrame:
        calib = self.calibration.read()
        left = _decode(message["stereo_left"], cv2.IMREAD_GRAYSCALE)
        right = _decode(message["stereo_right"], cv2.IMREAD_GRAYSCALE)
        rgb = self.rgb(message)
        depth_mm, valid = self.depth(left, right, float(calib["focal_length"]), float(calib["baseline"]))
        return CameraFrame(_cloud_from_depth(np.where(valid > 0, depth_mm, 0), calib, rgb), rgb)


def _decode(data: bytes, flags: int) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)


def _cloud_from_depth(depth_mm: np.ndarray, calib: dict, rgb: np.ndarray) -> np.ndarray:
    """Colored cloud in the RGB camera frame, from every second depth pixel."""
    K_depth = np.array(calib["K_stereo"], dtype=np.float64)
    K_depth[:2] /= DEPTH_STRIDE
    T_rgb_depth = np.linalg.inv(np.asarray(calib["T_device_rgb"])) @ np.asarray(calib["T_device_rectleft"])
    return colored_cloud(
        depth_mm[::DEPTH_STRIDE, ::DEPTH_STRIDE],
        K_depth,
        T_rgb_depth,
        rgb,
        np.asarray(calib["K_rgb"], dtype=np.float64),
        np.eye(4),
    )


def workspace_scene(cloud: np.ndarray, link_boxes: np.ndarray) -> np.ndarray:
    """The (N, 6) cloud inside the ellipsoid, without points in the robot's link boxes."""
    scene = cloud[in_ellipsoid(cloud[:, :3])]
    on_robot = np.zeros(len(scene), dtype=bool)
    for low, high in link_boxes:
        on_robot |= ((scene[:, :3] >= low) & (scene[:, :3] <= high)).all(axis=1)
    return scene[~on_robot]


class ObjectSegmenter:
    """The manipulated object's binary mask from a text prompt, with SAM3."""

    def __init__(self, prompt: str, device: str, checkpoint: str | None, min_score: float) -> None:
        from sam3.model.sam3_image_processor import Sam3Processor

        self.device = torch.device(device)
        self.model = load_image_model(self.device, checkpoint)
        self.processor = Sam3Processor(self.model, device=str(self.device))
        self.prompt = noun_phrase(prompt)
        self.min_score = min_score

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """Binary (255, 255) uint8 mask of the highest-scoring query, empty if its score is under `min_score`."""
        select_device(self.device)  # the current CUDA device is per thread
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(rgb))
        logits, score = best_query_mask(self.model, self.processor, state, self.prompt, self.device)
        if score < self.min_score:
            return np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
        upsampled = F.interpolate(logits[None, None], size=(MASK_SIZE, MASK_SIZE), mode="bilinear", align_corners=False)
        return (upsampled[0, 0] > 0).cpu().numpy().astype(np.uint8)


class BatchBuilder:
    """Builds the policy's input batch, with the fixed instruction and object prompt of a rollout."""

    def __init__(
        self,
        instruction: str,
        device: str,
        segmenter: Callable[[np.ndarray], np.ndarray] | None,
        text_model: str,
        text_revision: str,
        seed: int,
    ) -> None:
        self.device = device
        self.segmenter = segmenter
        self.tokens, self.text_mask = TextTokenizer(text_model, text_revision)(instruction)
        self.transform = cloud_transform()
        self.rng = np.random.RandomState(seed)

    def __call__(self, scene: np.ndarray, rgb: np.ndarray, proprio: np.ndarray) -> dict:
        """Batch of one from a robot-free cloud, the RGB image, and normalized fingertips (2, 5, 3)."""
        mask = np.zeros((MASK_SIZE, MASK_SIZE), np.float32) if self.segmenter is None else self.segmenter(rgb)
        sample = {
            "point_cloud": self.transform(utonia_input(crop_and_resample(scene, self.rng))),
            "object_mask": torch.as_tensor(mask, dtype=torch.float32)[None],
            "proprio": torch.as_tensor(proprio, dtype=torch.float32),
            "text_tokens": self.tokens,
            "text_mask": self.text_mask,
        }
        return to_batch(sample, self.device)
