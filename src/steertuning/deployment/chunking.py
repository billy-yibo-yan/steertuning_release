"""Real-time chunking and its composition with steering.

A new chunk starts at the row the robot is executing, so it overlaps the plan in progress. Rows the
robot will reach before the new chunk arrives are held to that plan, and the following rows hand over
linearly to the new prediction. Both this and steering blend a known action into the sample, so one
target and one per-row weight carry them into `steered_ddim`.
"""

from __future__ import annotations

import numpy as np


def chunking_target(
    plan: np.ndarray, plan_start: int, anchor: int, freeze: int, blend: int
) -> tuple[np.ndarray, np.ndarray]:
    """The plan's rows under a new chunk, and their weights.

    Args:
        plan: rows of the current plan, normalized, (H, D).
        plan_start: global index of the plan's first row.
        anchor: global index of the new chunk's first row.
        freeze: rows at the start of the new chunk to hold to the plan.
        blend: rows over which the weight falls from 1 to zero, after the held rows.

    Returns:
        target (H, D) and weight (H,). Rows the plan does not cover have weight 0.
    """
    horizon = len(plan)
    target = np.zeros_like(plan)
    weight = np.zeros(horizon, dtype=plan.dtype)
    for row in range(horizon):
        index = anchor + row - plan_start
        if not 0 <= index < horizon:
            continue
        target[row] = plan[index]
        if row < freeze:
            weight[row] = 1.0
        elif row < freeze + blend:
            weight[row] = 1.0 - (row - freeze) / blend
    return target, weight


def compose_targets(
    chunking: tuple[np.ndarray, np.ndarray] | None, steering: tuple[np.ndarray, np.ndarray] | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """One target and weight per row, with real-time chunking applied over steering.

    For each row, x <- w_c a_c + (1 - w_c) (w_s a_s + (1 - w_s) x), which is one blend with
    w = w_c + (1 - w_c) w_s and a = (w_c a_c + (1 - w_c) w_s a_s) / w.
    """
    if chunking is None or steering is None:
        return chunking or steering
    (chunk_target, chunk_weight), (steer_target, steer_weight) = chunking, steering
    weight = chunk_weight + (1.0 - chunk_weight) * steer_weight
    numerator = chunk_weight[:, None] * chunk_target + ((1.0 - chunk_weight) * steer_weight)[:, None] * steer_target
    target = np.divide(numerator, weight[:, None], out=np.zeros_like(numerator), where=weight[:, None] > 0.0)
    return target, weight
