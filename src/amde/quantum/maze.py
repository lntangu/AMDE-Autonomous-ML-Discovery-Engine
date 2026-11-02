"""Small maze-search comparison: BFS versus ideal Grover query model.

The maze adapter constructs a truth table for a reversible-oracle *model*.
This is intentionally not presented as a compiled quantum circuit or a
practical advantage over structure-aware graph search.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any

from .statevector import StatevectorBudget, grover_search_statevector


MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # up, right, down, left


@dataclass(frozen=True)
class MazeSolution:
    distance: int
    path: tuple[int, ...]
    expanded_nodes: int


def validate_grid(grid: list[str]) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]]]:
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("maze must be a non-empty rectangular list of strings")
    starts: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = []
    walls: set[tuple[int, int]] = set()
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == "S":
                starts.append((r, c))
            elif value == "G":
                goals.append((r, c))
            elif value == "#":
                walls.add((r, c))
            elif value != ".":
                raise ValueError(f"unsupported maze cell {value!r}")
    if len(starts) != 1 or len(goals) != 1:
        raise ValueError("maze must contain exactly one S and one G")
    return starts[0], goals[0], walls


def bfs_shortest_path(grid: list[str]) -> MazeSolution:
    start, goal, walls = validate_grid(grid)
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start, tuple())])
    seen = {start}
    expanded = 0
    while queue:
        position, path = queue.popleft()
        expanded += 1
        if position == goal:
            return MazeSolution(len(path), path, expanded)
        for move_id, (dr, dc) in enumerate(MOVES):
            nxt = (position[0] + dr, position[1] + dc)
            if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and nxt not in walls and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + (move_id,)))
    raise ValueError("maze has no path from S to G")


def _path_is_solution(grid: list[str], moves: tuple[int, ...]) -> bool:
    start, goal, walls = validate_grid(grid)
    rows, cols = len(grid), len(grid[0])
    current = start
    for move_id in moves:
        dr, dc = MOVES[move_id]
        current = (current[0] + dr, current[1] + dc)
        if not (0 <= current[0] < rows and 0 <= current[1] < cols) or current in walls:
            return False
    return current == goal


def _decode_path(index: int, depth: int) -> tuple[int, ...]:
    return tuple((index >> (2 * i)) & 0b11 for i in range(depth))


def run_maze_grover_demo(
    grid: list[str] | None = None,
    *,
    device: str = "cpu",
    max_qubits: int = 20,
    max_statevector_mb: int = 256,
) -> dict[str, Any]:
    """Compare BFS with an idealized Grover search over fixed-length move strings."""
    if grid is None:
        grid = [
            "S....",
            ".##..",
            "...#.",
            "..#..",
            "....G",
        ]
    started = time.perf_counter()
    bfs = bfs_shortest_path(grid)
    bfs_seconds = time.perf_counter() - started
    depth = bfs.distance
    n_qubits = 2 * depth
    if n_qubits > max_qubits:
        return {
            "status": "skipped_resource_limit",
            "reason": f"shortest path needs {n_qubits} qubits, configured maximum is {max_qubits}",
            "bfs": {"distance": depth, "expanded_nodes": bfs.expanded_nodes, "seconds": bfs_seconds},
            "maze": grid,
        }
    search_size = 1 << n_qubits
    if search_size > (1 << max_qubits):
        raise AssertionError("qubit and search-space bounds disagree")

    oracle_started = time.perf_counter()
    marked: list[int] = []
    for index in range(search_size):
        if _path_is_solution(grid, _decode_path(index, depth)):
            marked.append(index)
    oracle_seconds = time.perf_counter() - oracle_started
    if not marked:
        raise RuntimeError("BFS found a path but the path oracle marked no encoded routes")
    grover = grover_search_statevector(
        n_qubits,
        marked,
        device=device,
        budget=StatevectorBudget(max_qubits=max_qubits, max_statevector_mb=max_statevector_mb),
    )
    measured_path = _decode_path(int(grover["sampled_measurement"]), depth)
    top_path = _decode_path(int(grover["top_measurement"]), depth)
    return {
        "status": "ok",
        "maze": grid,
        "shortest_path_depth": depth,
        "move_encoding": {"0": "up", "1": "right", "2": "down", "3": "left"},
        "bfs": {
            "algorithm": "breadth_first_search",
            "distance": bfs.distance,
            "path": list(bfs.path),
            "expanded_nodes": bfs.expanded_nodes,
            "wall_seconds": bfs_seconds,
        },
        "grover": {
            **grover,
            "marked_path_count": len(marked),
            "oracle_precompute_seconds": oracle_seconds,
            "measured_path": list(measured_path),
            "measured_path_is_valid": _path_is_solution(grid, measured_path),
            "top_probability_path": list(top_path),
            "top_probability_path_is_valid": _path_is_solution(grid, top_path),
            "theoretical_classical_unstructured_queries": search_size / max(1, len(marked)),
            "theoretical_grover_queries": grover["oracle_queries"],
            "oracle_is_reversible_gate_circuit": False,
        },
        "interpretation": (
            "Query-complexity illustration only. The phase oracle was precomputed by classical enumeration, "
            "and the ideal statevector simulator performs O(2^n) classical work per iteration. "
            "BFS exploits maze graph structure and is the relevant practical classical baseline. "
            "No end-to-end quantum speedup is established."
        ),
    }
