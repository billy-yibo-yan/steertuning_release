"""Choosing the steering trajectory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from steertuning.data.dataset import TextTokenizer, WindowDataset, to_batch
from steertuning.data.representation import WRIST_POSE_POINTS, transform_points
from steertuning.data.session import session_name
from steertuning.policy.policy import bf16_autocast
from steertuning.steering.bank import DemoBank
from steertuning.steering.layout import ObjectLayout, layout_distance, surface_transform
from steertuning.steering.tracking import BlendedTracker, ProgressTracker

VINN_NEIGHBORS = 3


def select_nearest_layout(bank: DemoBank, layout: ObjectLayout) -> int:
    """The demonstration with the closest object layout."""
    return int(np.argmin([layout_distance(demo.layout, layout) for demo in bank]))


def select_by_wrist(bank: DemoBank, layout: ObjectLayout, wrist_points: np.ndarray) -> int:
    """The demonstration with the nearest initial wrist poses, compared in the live camera frame.

    Each pose is its wrist and three axis tips 3 cm away, so wrist position counts most in the mean
    distance and orientation little. `wrist_points` is (2, 4, 3), the points `WRIST_POSE_POINTS` of the
    live action points.
    """
    distances = [
        np.linalg.norm(
            transform_points(demo.trajectory[0, WRIST_POSE_POINTS], surface_transform(demo.layout, layout))
            - wrist_points,
            axis=-1,
        ).mean()
        for demo in bank
    ]
    return int(np.argmin(distances))


def vinn_weights(context: np.ndarray, bank: DemoBank) -> tuple[np.ndarray, np.ndarray]:
    """VINN: nearest neighbors on the mean-pooled context latent of the initial observation.

    L2 distances are normalized over the bank, then a nearer neighbor weighs exponentially more.
    Returns the neighbors' indices and their weights, which sum to one.
    """
    contexts = np.stack([demo.initial_context for demo in bank])
    distances = np.linalg.norm(contexts - context, axis=-1)
    normalized = (distances - distances.mean()) / max(distances.std(), 1e-9)
    neighbors = np.argsort(distances)[:VINN_NEIGHBORS]
    weights = np.exp(-normalized[neighbors])
    return neighbors, weights / weights.sum()


@torch.no_grad()
def initial_context(policy, batch: dict) -> np.ndarray:
    """The mean-pooled context latent of an observation: the last stage's context, (D,)."""
    device = next(policy.parameters()).device
    with bf16_autocast(device):
        return policy.encode_observation(batch).contexts[-1][0].float().cpu().numpy()


def attach_initial_contexts(bank: DemoBank, policy, task_dir: str | Path, instruction: str) -> None:
    """Fill each demonstration's `initial_context` from its frame 0, for the VINN selection.

    The observation carries `instruction`, the text the live context reads, so VINN compares contexts built
    from one text.
    """
    tokenizer = TextTokenizer(policy.config.text_model, policy.config.text_revision)
    windows = WindowDataset(task_dir, policy.normalizer, tokenizer=tokenizer)
    first_segment: dict[str, int] = {}
    for index, entry in enumerate(windows.segments):
        first_segment.setdefault(session_name(entry.directory, windows.root), index)
    device = next(policy.parameters()).device
    for demo in bank:
        if demo.name not in first_segment:
            raise ValueError(f"{demo.name} has no windows under {task_dir}, so VINN cannot read its first frame.")
        sample = windows.sample(first_segment[demo.name], anchor=0, seed=0)
        sample["text_tokens"], sample["text_mask"] = tokenizer(instruction)
        demo.initial_context = initial_context(policy, to_batch(sample, device))


def select_demonstrations(
    bank: DemoBank,
    layout: ObjectLayout,
    selection: str = "layout",
    wrist_points: np.ndarray | None = None,
    context: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the chosen demonstrations and their weights, which sum to one.

    Args:
        bank: the demonstrations to choose from.
        layout: object layout of the current observation.
        selection: "layout", "wrist" or "vinn".
        wrist_points: initial wrist poses (2, 4, 3) in the live camera frame, for "wrist".
        context: mean-pooled context latent of the initial observation, for "vinn".
    """
    if selection == "layout":
        return np.array([select_nearest_layout(bank, layout)]), np.ones(1)
    if selection == "wrist":
        return np.array([select_by_wrist(bank, layout, wrist_points)]), np.ones(1)
    if selection == "vinn":
        return vinn_weights(context, bank)
    raise ValueError(f"Unknown selection: {selection}")


def build_tracker(
    bank: DemoBank, layout: ObjectLayout, indices: np.ndarray, weights: np.ndarray
) -> ProgressTracker | BlendedTracker:
    """Bring the chosen demonstrations into the current camera frame and track them.

    Every selection strategy goes through here, so steering is the same for all of them.
    """
    trackers = []
    for index in indices:
        demo = bank[int(index)]
        trajectory = transform_points(demo.trajectory, surface_transform(demo.layout, layout))
        trackers.append(ProgressTracker(trajectory))
    return trackers[0] if len(trackers) == 1 else BlendedTracker(trackers, weights)
