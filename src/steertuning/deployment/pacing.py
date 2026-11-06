"""The timing of one chunk row on the arms: its duration, its command times and the speed limit."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterator

import numpy as np


def row_schedule(
    period: float,
    prep: float,
    largest_move: float,
    speed: float,
    points: int,
    control_period: float,
    minimum: float = 0.0,
) -> tuple[float, int]:
    """A row's duration in seconds and its number of commands.

    The time spent preparing the row comes out of its period, and the speed limit can stretch it.
    A row with a `minimum` duration, the first move, is commanded at the control period.
    """
    move_seconds = largest_move / speed
    duration = max(control_period, period - prep, minimum, move_seconds)
    requested = max(1, points, math.ceil(max(minimum, move_seconds) / control_period))
    return duration, min(requested, max(1, math.floor(duration / control_period + 1e-9)))


def row_ticks(duration: float, ticks: int, halt: threading.Event) -> Iterator[float]:
    """Interpolation fractions on the wall clock. A late tick skips ahead and never replays a burst."""
    start = time.monotonic()
    interval = duration / max(1, ticks)
    deadline = start + interval
    while not halt.is_set():
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(remaining, 0.02))
            continue
        elapsed = time.monotonic() - start
        fraction = min(1.0, elapsed / duration)
        yield fraction
        if fraction >= 1.0:
            return
        deadline = start + min(duration, (math.floor((time.monotonic() - start) / interval + 1e-9) + 1) * interval)


def limit_joint_step(current: np.ndarray, waypoint: np.ndarray, speed: float, elapsed: float) -> np.ndarray:
    """Step toward a waypoint, bounded by the speed limit over the elapsed time."""
    limit = speed * max(0.0, elapsed)
    return current + np.clip(waypoint - current, -limit, limit)
