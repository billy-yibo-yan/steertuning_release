"""Object grounding.

For each snippet with an active hand, Gemini places two keypoints on the contacted object. They
initialize SAM3, which propagates the object mask through the snippet. Arm and hand regions are removed
from the mask. A snippet with no active hand has no contacted object, so its mask stays empty.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from steertuning.processing.contact import Snippet, closest_contact_frame
from steertuning.processing.masks import Sam3, read_mask
from steertuning.processing.point_cloud import clear_masked_pixels
from steertuning.remote import gemini_client

NUM_KEYPOINTS = 2
MASK_HEIGHT, MASK_WIDTH = 120, 160
HAND_PADDING_PX = 3

KEYPOINT_PROMPT = """The image shows the moment the {HANDS} touching an object.
Identify the contacted object and place exactly 2 keypoints on it.

Rules:
- Points must be on the object only, never on hands, arms or people, and not close to the fingers.
- Keep points near the contact region and inside the visible object surface.
- Spread the two points across the object.

Coordinates:
- Use normalized [x, y] in 0..1000.
- (0, 0) is top-left; (1000, 1000) is bottom-right.

Return JSON only:
{
  "object": "object_name",
  "locations": [[x1, y1], [x2, y2]]
}"""

_HAND_WORDS = {"L": "left hand is", "R": "right hand is", "LR": "left and right hands are"}


@dataclass(frozen=True)
class ObjectKeypoints:
    """(2, 2) xy keypoints in 0-1 on the contacted object, on the seed frame of a snippet."""

    keyframe: int
    points: list[list[float]]


def parse_keypoints(text: str) -> list[list[float]]:
    """Two xy points in 0-1 from a model reply. A code fence around the JSON is fine.

    The prompt asks for JSON with coordinates in 0-1000, and anything else raises ValueError, including
    values all within 0-1 or any above 1000: those are another scale, and a guess would seed a wrong mask.
    """
    try:
        points = np.asarray(json.loads(text[text.find("{") : text.rfind("}") + 1])["locations"], dtype=np.float64)
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"expected JSON with {NUM_KEYPOINTS} [x, y] points, got: {text!r}") from error
    if points.shape != (NUM_KEYPOINTS, 2) or not (points >= 0.0).all() or not (points <= 1000.0).all():
        raise ValueError(f"expected {NUM_KEYPOINTS} [x, y] points in 0-1000, got: {text!r}")
    if points.max() <= 1.0:
        raise ValueError(f"expected coordinates in 0-1000, and these are all within 0-1: {text!r}")
    return (points / 1000.0).tolist()


class GeminiKeypoints:
    """Keypoints on the contacted object from Gemini, in one image of the contact."""

    def __init__(self, model: str) -> None:
        self.client = gemini_client()
        self.model = model

    def _generate(self, prompt: str, image: np.ndarray) -> str:
        from google.genai import types

        data = cv2.imencode(".jpg", image[..., ::-1])[1].tobytes()
        part = types.Part.from_bytes(data=data, mime_type="image/jpeg")
        return self.client.models.generate_content(model=self.model, contents=[prompt, part]).text or ""

    def __call__(self, image: np.ndarray, hands: str) -> list[list[float]]:
        """Retries a failed call or an unusable reply once, then raises."""
        prompt = KEYPOINT_PROMPT.replace("{HANDS}", _HAND_WORDS[hands])
        try:
            return parse_keypoints(self._generate(prompt, image))
        except Exception:
            return parse_keypoints(self._generate(prompt, image))


def request_keypoints(
    keypoints_model,
    read_rgb,
    snippets: list[Snippet],
    distances: dict[int, tuple[float | None, float | None]],
) -> list[ObjectKeypoints | None]:
    """One request per snippet with an active hand, None for a snippet with no active hand.

    The request sends the closest contact frame, and that frame seeds SAM3. Any failure propagates, so a
    session never gets a silent empty mask.
    """

    keypoints = []
    for snippet in snippets:
        if snippet.hands is None:
            keypoints.append(None)
            continue
        frame = closest_contact_frame(snippet, distances)
        keypoints.append(ObjectKeypoints(frame, keypoints_model(read_rgb(frame), snippet.hands)))
    return keypoints


def track_object_masks(
    sam3: Sam3,
    read_rgb,
    snippets: list[Snippet],
    keypoints: list[ObjectKeypoints | None],
    hand_mask_dir: Path,
    num_frames: int,
) -> np.ndarray:
    """Object masks of a recording, (T, 120, 160) uint8 in {0, 255}, zero in a snippet with no keypoints.

    Hand removal happens at full resolution with padding.
    """
    masks = np.zeros((num_frames, MASK_HEIGHT, MASK_WIDTH), dtype=np.uint8)
    for snippet, grounded in zip(snippets, keypoints, strict=True):
        if grounded is None:
            continue
        with tempfile.TemporaryDirectory() as frames_dir:
            for frame in range(snippet.start, snippet.end + 1):
                cv2.imwrite(f"{frames_dir}/{frame - snippet.start:06d}.jpg", read_rgb(frame)[..., ::-1])
            tracked = sam3.track_points(frames_dir, grounded.keyframe - snippet.start, np.asarray(grounded.points))
            for local, mask in tracked:
                frame = snippet.start + local
                mask = clear_masked_pixels(mask, read_mask(hand_mask_dir, frame), HAND_PADDING_PX)
                masks[frame] = (
                    cv2.resize(mask.astype(np.uint8), (MASK_WIDTH, MASK_HEIGHT), interpolation=cv2.INTER_NEAREST) * 255
                )
    return masks
