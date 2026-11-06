"""A* on a 2D occupancy grid built from the voxel map."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label

NEIGHBORS = [(dx, dy, float(np.hypot(dx, dy))) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
GRID_MARGIN_METERS = 1.0
WAYPOINT_SPACING_METERS = 0.4
START_SNAP_METERS = 0.75


@dataclass
class OccupancyGrid:
    """`blocked[ix, iy]` is True where the robot center cannot be. Unseen cells are free."""

    blocked: np.ndarray
    origin: np.ndarray  # world xy of the corner of cell (0, 0)
    resolution: float

    def cell(self, xy: np.ndarray) -> tuple[int, int]:
        ix, iy = np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)
        return int(ix), int(iy)

    def world(self, cells: np.ndarray) -> np.ndarray:
        return (np.asarray(cells) + 0.5) * self.resolution + self.origin


@dataclass
class Approach:
    waypoints: np.ndarray  # (N, 2) world xy, ending at the stop position
    yaw: float  # heading at the stop position, facing the target


def build_grid(
    points: np.ndarray,
    include: np.ndarray,
    resolution: float,
    height_band: tuple[float, float],
    robot_radius: float,
) -> OccupancyGrid:
    """Mark cells that hold a voxel in `height_band` and inflate them by `robot_radius`.

    The grid spans `points` and the xy positions in `include` (the robot and the target).
    """
    xy = np.vstack([points[:, :2], include])
    origin = np.floor(xy.min(axis=0) / resolution) * resolution - GRID_MARGIN_METERS
    size = np.ceil((xy.max(axis=0) + GRID_MARGIN_METERS - origin) / resolution).astype(int) + 1
    band = points[(points[:, 2] >= height_band[0]) & (points[:, 2] <= height_band[1])]
    cells = np.floor((band[:, :2] - origin) / resolution).astype(int)
    occupied = np.zeros(size, dtype=bool)
    occupied[cells[:, 0], cells[:, 1]] = True
    # With no cell occupied, scipy's distance transform has no meaning.
    blocked = distance_transform_edt(~occupied) * resolution < robot_radius if occupied.any() else occupied
    return OccupancyGrid(blocked, origin, resolution)


def _line_is_free(blocked: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> bool:
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
    xs = np.rint(np.linspace(start[0], end[0], steps + 1)).astype(int)
    ys = np.rint(np.linspace(start[1], end[1], steps + 1)).astype(int)
    return not blocked[xs, ys].any()


def astar(blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """8-connected A* with a Euclidean heuristic. `goal` must be reachable from `start`."""
    best_cost = np.full(blocked.shape, np.inf)
    parent = np.zeros((*blocked.shape, 2), dtype=np.int32)
    best_cost[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        _, node = heapq.heappop(queue)
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(tuple(int(v) for v in parent[path[-1]]))
            return path[::-1]
        for dx, dy, step in NEIGHBORS:
            next_node = (node[0] + dx, node[1] + dy)
            if not (0 <= next_node[0] < blocked.shape[0] and 0 <= next_node[1] < blocked.shape[1]):
                continue
            if blocked[next_node]:
                continue
            cost = best_cost[node] + step
            if cost < best_cost[next_node]:
                best_cost[next_node] = cost
                parent[next_node] = node
                heapq.heappush(queue, (cost + np.hypot(next_node[0] - goal[0], next_node[1] - goal[1]), next_node))
    raise ValueError(f"No path from cell {start} to cell {goal}.")


def plan_approach(grid: OccupancyGrid, start: np.ndarray, target: np.ndarray, reach_distance: float) -> Approach | None:
    """Path from `start` to the reachable cell closest to `reach_distance` from `target`, facing `target`.

    Obstacles can keep the robot farther away than `reach_distance`, for example the edge of the
    table the target stands on. Of the cells about as close, the goal is the one nearest the start.
    The path starts at the free cell nearest to `start`, since the robot's own cell is blocked when
    it stands close to something it has mapped. Returns None when that cell is over
    START_SNAP_METERS away. Waypoints are at most WAYPOINT_SPACING_METERS apart, with free cells
    on the straight line between them.
    """
    free = np.argwhere(~grid.blocked)  # never empty: the grid has a margin wider than the robot
    to_free = np.linalg.norm(free - grid.cell(start), axis=1)
    if to_free.min() * grid.resolution > START_SNAP_METERS:
        return None
    start_cell = tuple(int(v) for v in free[to_free.argmin()])
    components, _ = label(~grid.blocked, structure=np.ones((3, 3)))
    reachable = np.argwhere(components == components[start_cell])
    gap = np.abs(np.linalg.norm(grid.world(reachable) - target, axis=1) - reach_distance)
    closest = reachable[gap <= gap.min() + grid.resolution]
    goal_cell = tuple(int(v) for v in closest[np.linalg.norm(closest - start_cell, axis=1).argmin()])

    path = astar(grid.blocked, start_cell, goal_cell)
    spacing = WAYPOINT_SPACING_METERS / grid.resolution
    kept = [path[0]]
    for previous, cell in zip(path, path[1:]):
        too_far = np.hypot(cell[0] - kept[-1][0], cell[1] - kept[-1][1]) > spacing
        if too_far or not _line_is_free(grid.blocked, kept[-1], cell):
            kept.append(previous)
    waypoints = grid.world(kept + path[-1:] if len(path) > 1 else kept)
    to_target = target - waypoints[-1]
    return Approach(waypoints, float(np.arctan2(to_target[1], to_target[0])))
