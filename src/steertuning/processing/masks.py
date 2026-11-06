"""SAM3 video segmentation: arm and hand masks from text, object masks from points."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import torch

ARM_AND_HAND_PROMPT = "arm and hand"


def select_device(device: torch.device) -> None:
    # SAM3 places some buffers and its state on the current CUDA device. A device with no index means that one.
    if device.index is not None:
        torch.cuda.set_device(device)


def load_image_model(device: torch.device | str, checkpoint: str | None):
    """SAM3's image model on a CUDA device."""
    from sam3.model_builder import build_sam3_image_model

    select_device(torch.device(device))
    model = build_sam3_image_model(device="cpu", checkpoint_path=checkpoint, load_from_HF=checkpoint is None)
    return model.to(device)


def noun_phrase(name: str) -> str:
    """SAM3 scores a noun phrase ("a pot") well above a bare noun ("pot")."""
    name = name.strip()
    if name.split(" ", 1)[0].lower() in {"a", "an", "the"}:
        return name
    return f"an {name}" if name[0].lower() in "aeiou" else f"a {name}"


def best_query_mask(model, processor, state: dict, prompt: str, device: torch.device) -> tuple[torch.Tensor, float]:
    """Mask logits of the highest-scoring query for `prompt` on an encoded image, and that query's score.

    The score is the per-query one. SAM3's image-level presence score, which its processor
    multiplies in, stays low on many real objects.
    """
    backbone_out = state["backbone_out"]
    with torch.autocast(device.type, dtype=torch.bfloat16):
        backbone_out.update(model.backbone.forward_text([prompt], device=device))
        output = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=processor.find_stage,
            geometric_prompt=model._get_dummy_prompt(),
            find_target=None,
        )
    scores = output["pred_logits"][0, :, 0].sigmoid().float()
    top = int(scores.argmax())
    return output["pred_masks"][0, top].float(), float(scores[top])


class Sam3:
    """A SAM3 video predictor on one GPU."""

    def __init__(self, device: str) -> None:
        from sam3.model_builder import build_sam3_predictor

        self.device = torch.device(device)
        select_device(self.device)
        self.predictor = build_sam3_predictor(version="sam3")
        self.predictor.model.fill_hole_area = 0

    def propagate_text(self, frames_dir: str | Path, prompt: str) -> dict[int, np.ndarray]:
        """Segment a text prompt from the first frame of a folder through the rest.

        Returns one boolean mask per frame, the union of all instances found.
        """
        select_device(self.device)
        handle = self.predictor.handle_request
        response = handle(dict(type="start_session", resource_path=str(frames_dir), offload_video_to_cpu=True))
        session = response["session_id"]
        try:
            handle(dict(type="add_prompt", session_id=session, frame_index=0, text=prompt))
            propagation = self.predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session))
            return {r["frame_index"]: np.asarray(r["outputs"]["out_binary_masks"]).any(axis=0) for r in propagation}
        finally:
            handle(dict(type="close_session", session_id=session))

    def track_points(
        self, frames_dir: str | Path, keyframe: int, points: np.ndarray
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Track one object from positive points on a key frame, forward and backward through a folder.

        `points` is (K, 2) xy in 0-1. Yields (frame, boolean mask).
        """
        select_device(self.device)
        tracker = self.predictor.model.tracker
        tracker_backbone = tracker.backbone
        tracker.backbone = self.predictor.model.detector.backbone  # the two share one image encoder
        try:
            state = tracker.init_state(
                video_path=str(frames_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=True,
            )
            tracker.clear_all_points_in_video(state)
            *_, masks = tracker.add_new_points_or_box(
                inference_state=state,
                frame_idx=keyframe,
                obj_id=1,
                points=torch.as_tensor(points, dtype=torch.float32),
                labels=torch.ones(len(points), dtype=torch.int32),
                clear_old_points=False,
            )
            yield keyframe, _object_mask(masks)
            for reverse, length in ((False, state["num_frames"] - 1 - keyframe), (True, keyframe)):
                for frame, _, _, masks, _ in tracker.propagate_in_video(
                    state,
                    start_frame_idx=keyframe,
                    max_frame_num_to_track=length,
                    reverse=reverse,
                    propagate_preflight=True,
                    tqdm_disable=True,
                ):
                    yield frame, _object_mask(masks)
        finally:
            tracker.backbone = tracker_backbone


def _object_mask(masks) -> np.ndarray:
    return (masks[0].squeeze() > 0.0).cpu().numpy()


def mask_path(directory: Path, frame: int) -> Path:
    return directory / f"{frame:06d}.png"


def read_mask(directory: Path, frame: int) -> np.ndarray:
    """A stored mask as a boolean array."""
    mask = cv2.imread(str(mask_path(directory, frame)), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path(directory, frame))
    return mask > 127


def write_arm_and_hand_masks(
    sam3: Sam3,
    read_image: Callable[[int], np.ndarray],
    num_frames: int,
    directory: Path,
    window_frames: int,
    suffix: str,
) -> None:
    """Write one binary PNG per frame into `directory`, which appears only when complete.

    `read_image` returns a frame ready for `cv2.imwrite`. A new SAM3 session every
    `window_frames` frames bounds memory on long recordings, with the prompt repeated on the first
    frame of each window.
    """
    partial = directory.with_name(directory.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    for first in range(0, num_frames, window_frames):
        frames = range(first, min(first + window_frames, num_frames))
        with tempfile.TemporaryDirectory() as window:
            for local, frame in enumerate(frames):
                cv2.imwrite(f"{window}/{local:06d}{suffix}", read_image(frame))
            masks = sam3.propagate_text(window, ARM_AND_HAND_PROMPT)
        for local, frame in enumerate(frames):
            cv2.imwrite(str(mask_path(partial, frame)), masks[local].astype(np.uint8) * 255)
    partial.rename(directory)
