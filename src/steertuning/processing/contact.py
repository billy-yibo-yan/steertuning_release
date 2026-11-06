"""Contact-based segmentation.

The arms and hands are removed from the depth, the rest is unprojected into a scene cloud, and a hand
is active when one of its landmarks comes within the threshold of that cloud. Snippets are intervals
with a consistent active-contact state, see `contact_snippets`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from steertuning.data.representation import transform_points
from steertuning.processing.point_cloud import clear_masked_pixels, project_into_image, unproject_depth

# The palm landmark is an interior point of the hand, not on its surface.
_SURFACE_LANDMARKS = np.arange(20)

# The scene cloud for contact readings is coarser and reaches farther than the policy's cloud.
SCENE_DOWNSAMPLE = 2
SCENE_MASK_PADDING_PX = 2
SCENE_MAX_DEPTH = 15.0  # meters
SCENE_POINT_STRIDE = 10


@dataclass(frozen=True)
class Snippet:
    """Frame interval, inclusive. `hands` is "L", "R", "LR", or None when no hand is active."""

    start: int
    end: int
    hands: str | None


def scene_points(
    depth_mm: np.ndarray,
    arm_and_hand_mask: np.ndarray,
    K_depth: np.ndarray,
    T_world_depth: np.ndarray,
    K_rgb: np.ndarray,
    T_world_rgb: np.ndarray,
    rgb_size: tuple[int, int],
) -> np.ndarray:
    """(N, 3) world-frame scene points in view of the RGB camera, with the embodiment removed."""
    size = depth_mm.shape[1] // SCENE_DOWNSAMPLE, depth_mm.shape[0] // SCENE_DOWNSAMPLE
    depth = cv2.resize(depth_mm, size, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(arm_and_hand_mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    K = K_depth.copy()
    K[:2] /= SCENE_DOWNSAMPLE
    points = unproject_depth(clear_masked_pixels(depth, mask, SCENE_MASK_PADDING_PX), K, SCENE_MAX_DEPTH)
    points = transform_points(points, T_world_depth)
    inside, _ = project_into_image(points, K_rgb, T_world_rgb, *rgb_size)
    return points[inside][::SCENE_POINT_STRIDE]


def closest_distance(landmarks: np.ndarray, points: np.ndarray) -> float | None:
    """Smallest landmark-to-scene distance in meters, None when the hand or the scene is absent."""
    landmarks = landmarks[_SURFACE_LANDMARKS]
    if len(points) == 0 or not np.isfinite(landmarks).all():
        return None
    return float(np.sqrt(((points[:, None] - landmarks[None]) ** 2).sum(axis=-1).min()))


def _contact_label(left: float | None, right: float | None, threshold_m: float) -> str | None:
    label = ("L" if left is not None and left < threshold_m else "") + (
        "R" if right is not None and right < threshold_m else ""
    )
    return label or None


def contact_snippets(
    distances: dict[int, tuple[float | None, float | None]],
    num_frames: int,
    threshold_m: float,
    stride: int,
) -> list[Snippet]:
    """Snippets that cover the recording, from per-frame (left, right) hand-to-scene distances.

    The set of active hands is read every `stride` frames, scanning backward in time, and a reading
    holds until the next one. The rule:

    - A snippet changes when the set of active hands changes.
    - Frames with no active hand join the contact that follows them, so a snippet holds the approach
      before its contact. Frames after the last contact form a snippet with no active hand.
    - A new label met one reading after a change is skipped, and the snippet takes the label of the
      reading before it in time.
    - Adjacent snippets with the same label merge.

    Two contacts of the same hands with only no-contact frames between them are one snippet, however long
    the gap.
    """

    def label(frame: int) -> str | None:
        return _contact_label(*distances.get(frame, (None, None)), threshold_m)

    # The scan runs backward because no-contact frames join what follows them, which it has already read.
    snippets = []
    current = None  # the label of the open snippet
    last = (num_frames - 1) - (num_frames - 1) % stride  # the open snippet's last reading
    first = last  # the reading in hand: the open snippet starts one stride later, until this reading joins it
    for frame in range(last, -1, -stride):
        hands = label(frame)
        if hands == current or hands is None:
            first -= stride
        elif last - first <= stride:  # the open label lasted one reading, so the label before it takes over
            current = label(frame - stride) or hands
            first -= stride
        else:  # the set of active hands changed: the open snippet is done, and this reading opens the next
            snippets.append(Snippet(first + stride, min(last + stride - 1, num_frames - 1), current))
            last = first
            first = last - stride
            current = hands
    snippets.append(Snippet(first + stride, min(last + stride - 1, num_frames - 1), current))
    return _merge_same_label(sorted(snippets, key=lambda s: s.start))


def _merge_same_label(snippets: list[Snippet]) -> list[Snippet]:
    merged = [snippets[0]]
    for snippet in snippets[1:]:
        previous = merged[-1]
        if previous.end + 1 == snippet.start and previous.hands == snippet.hands:
            merged[-1] = Snippet(previous.start, snippet.end, previous.hands)
        else:
            merged.append(snippet)
    return merged


def closest_contact_frame(snippet: Snippet, distances: dict[int, tuple[float | None, float | None]]) -> int | None:
    """The frame of the closest contact of the snippet's active hands, None without a distance.

    With both hands active, a frame scores by its farther hand.
    """
    best_frame, best_score = None, float("inf")
    for frame in range(snippet.start, snippet.end + 1):
        left, right = distances.get(frame, (None, None))
        left = float("inf") if left is None else left
        right = float("inf") if right is None else right
        score = {"L": left, "R": right}.get(snippet.hands, max(left, right))
        if score < best_score:
            best_frame, best_score = frame, score
    return best_frame
