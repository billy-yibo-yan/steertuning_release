"""One processing pipeline for pretraining recordings and task demonstrations.

`process_session` runs, in order: SAM3 arm and hand masks on the stereo and RGB views, depth and the
colored cloud with those regions removed, hand-scene distances and snippets, Qwen3-VL instructions,
Gemini keypoints, and SAM3 object masks. The arm and hand masks stay under `annotate/` for the later
stages and the next run. Everything else is written to `preprocessed.partial/`, which becomes
`preprocessed/` when complete. A session that has `preprocessed/` is kept.

The workspace crop and the resampling to 5,000 points happen per sample when the data is read, in the
anchor frame (`steertuning.data.representation.crop_and_resample`). The clouds on disk keep
`cloud_points` points for that. `steertuning.processing.directory` runs this over a directory of recordings.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from steertuning.processing.contact import Snippet, closest_distance, contact_snippets, scene_points
from steertuning.processing.depth import StereoDepth
from steertuning.processing.language import QwenVL, sample_frames
from steertuning.processing.masks import Sam3, read_mask, write_arm_and_hand_masks
from steertuning.processing.object_grounding import GeminiKeypoints, request_keypoints, track_object_masks
from steertuning.processing.point_cloud import clear_masked_pixels, colored_cloud, resample_cloud
from steertuning.processing.recording import Recording, open_recording

log = logging.getLogger(__name__)

ANNOTATE_DIR = "annotate"
OUTPUT_DIR = "preprocessed"
NOTE_FILE = "preprocess_note.json"


@dataclass
class ProcessConfig:
    sam3_device: str = "cuda:0"
    depth_device: str = "cuda:0"
    language_device: str = "cuda:0"
    depth_model: str = "M"
    depth_refine: int = 3
    qwen_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    gemini_model: str = "gemini-2.5-flash"
    contact_threshold_m: float = 0.05
    contact_stride: int = 3
    language_frames: int = 10
    depth_mask_padding_px: int = 4
    sam3_window_frames: int = 128
    cloud_points: int = 40000
    num_paraphrases: int = 8
    seed: int = 0


class Models:
    """The networks of the pipeline, each loaded the first time it is needed."""

    def __init__(self, config: ProcessConfig) -> None:
        self.config = config

    @cached_property
    def sam3(self):
        return Sam3(self.config.sam3_device)

    @cached_property
    def depth(self):
        return StereoDepth(self.config.depth_device, self.config.depth_model, self.config.depth_refine)

    @cached_property
    def language(self):
        return QwenVL(self.config.qwen_model, self.config.language_device)

    @cached_property
    def keypoints(self):
        return GeminiKeypoints(self.config.gemini_model)


def process_session(session_dir: str | Path, config: ProcessConfig, models: Models | None = None) -> dict:
    """Write `preprocessed/` for one recording and return its summary. A complete output is kept."""
    session_dir = Path(session_dir)
    output = session_dir / OUTPUT_DIR
    if output.is_dir():
        log.info("Keeping %s. Delete its %s/ to process it again.", session_dir.name, OUTPUT_DIR)
        return json.loads((output / NOTE_FILE).read_text())
    models = models or Models(config)
    annotate = session_dir / ANNOTATE_DIR
    partial = session_dir / f"{OUTPUT_DIR}.partial"

    recording = open_recording(session_dir)
    keypoints_model = models.keypoints  # a missing API key stops the run here, before the slow stages
    annotate.mkdir(exist_ok=True)
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir()
    _write_masks(recording, annotate, config, models)
    distances = _write_clouds_and_distances(recording, annotate, partial, config, models)
    snippets = contact_snippets(distances, len(recording), config.contact_threshold_m, config.contact_stride)
    instructions = [
        models.language.describe(
            [
                Image.fromarray(recording.rgb(frame))
                for frame in sample_frames(snippet.start, snippet.end, config.language_frames)
            ]
        )
        for snippet in snippets
    ]
    keypoints = request_keypoints(keypoints_model, recording.rgb, snippets, distances)
    object_masks = track_object_masks(
        models.sam3, recording.rgb, snippets, keypoints, annotate / "hand_masks_rgb", len(recording)
    )
    note = {
        "session": session_dir.name,
        "frames_kept": len(recording),
        "snippets": len(snippets),
        "frames_with_object_mask": int((object_masks > 0).any(axis=(1, 2)).sum()),
    }
    log.info("%s: %s", session_dir.name, note)
    _write_session(recording, partial, snippets, instructions, object_masks, note)
    partial.rename(output)
    return note


def _write_masks(recording: Recording, annotate: Path, config: ProcessConfig, models: Models) -> None:
    # Depth lives in the stereo view and object masks in the RGB view, and the two cannot be mapped
    # into each other without the depth under the hand.
    views = (
        ("hand_masks_stereo", recording.stereo_left, ".png"),
        ("hand_masks_rgb", lambda t: recording.rgb(t)[..., ::-1], ".jpg"),
    )
    for name, read_image, suffix in views:
        if not (annotate / name).is_dir():
            write_arm_and_hand_masks(
                models.sam3, read_image, len(recording), annotate / name, config.sam3_window_frames, suffix
            )


def _write_clouds_and_distances(
    recording: Recording, annotate: Path, partial: Path, config: ProcessConfig, models: Models
) -> dict[int, tuple[float | None, float | None]]:
    """Write `point_cloud.npy` and return the hand-scene distances, in meters, per sampled frame."""
    rng = np.random.default_rng(config.seed)
    clouds = np.lib.format.open_memmap(
        partial / "point_cloud.npy", mode="w+", dtype=np.float32, shape=(len(recording), config.cloud_points, 6)
    )
    distances = {}
    for t in range(len(recording)):
        depth_mm = recording.depth(t, models)
        hand_mask = read_mask(annotate / "hand_masks_stereo", t)
        cloud = colored_cloud(
            clear_masked_pixels(depth_mm, hand_mask, config.depth_mask_padding_px),
            recording.K_depth,
            recording.T_world_depth[t],
            recording.rgb(t),
            recording.K_rgb,
            recording.T_world_rgb[t],
        )
        clouds[t] = resample_cloud(cloud, config.cloud_points, rng)
        if t % config.contact_stride == 0:
            scene = scene_points(
                depth_mm,
                hand_mask,
                recording.K_depth,
                recording.T_world_depth[t],
                recording.K_rgb,
                recording.T_world_rgb[t],
                (recording.rgb_width, recording.rgb_height),
            )
            distances[t] = (
                closest_distance(recording.left_landmarks[t], scene),
                closest_distance(recording.right_landmarks[t], scene),
            )
    clouds.flush()
    return distances


def _write_session(
    recording: Recording,
    partial: Path,
    snippets: list[Snippet],
    instructions: list[str],
    object_masks: np.ndarray,
    note: dict,
) -> None:
    """The format of `steertuning.data.session`. `point_cloud.npy` is already there."""
    arrays = {
        "left_landmarks": recording.left_landmarks,
        "right_landmarks": recording.right_landmarks,
        "left_wrist": recording.left_wrist,
        "right_wrist": recording.right_wrist,
        "head_pose": recording.head_pose,
        "hand_valid": recording.hand_valid,
        "object_mask_160x120": object_masks,
        "kept_frame_indices": recording.frames,
    }
    for name, array in arrays.items():
        np.save(partial / f"{name}.npy", array)
    np.savez_compressed(
        partial / "rgb_camera_info.npz",
        K=recording.K_rgb,
        width=np.int32(recording.rgb_width),
        height=np.int32(recording.rgb_height),
        T_device_camera=recording.T_device_rgb,
    )
    segments = [
        {"start": snippet.start, "end": snippet.end, "action": instruction}
        for snippet, instruction in zip(snippets, instructions, strict=True)
    ]
    (partial / "text_prompt_segments.json").write_text(json.dumps(segments, indent=2))
    (partial / NOTE_FILE).write_text(json.dumps(note, indent=2))
