from __future__ import annotations

import math
from typing import Sequence

from src.data.schemas import ExactSearchResult
from src.search.exact_frontier import C_GRID, independent_frontier


def state_to_mask(state: Sequence[int]) -> int:
    if any(value not in (0, 1) for value in state):
        raise ValueError("binary state required")
    return sum(value << index for index, value in enumerate(state))


def mask_to_state(mask: int, width: int) -> tuple[int, ...]:
    if mask < 0 or mask >= 1 << width:
        raise ValueError("mask outside state space")
    return tuple((mask >> index) & 1 for index in range(width))


def _best_submasks(costs: Sequence[float], width: int) -> tuple[list[float], list[int]]:
    best = list(costs)
    arg = list(range(len(costs)))
    for bit in range(width):
        flag = 1 << bit
        for mask in range(1 << width):
            if not mask & flag:
                continue
            submask = mask ^ flag
            candidate = (best[submask], arg[submask])
            current = (best[mask], arg[mask])
            if candidate < current:
                best[mask], arg[mask] = candidate
    return best, arg


def best_binary_nested_chain(
    results: Sequence[ExactSearchResult], levels: Sequence[float] = C_GRID
) -> list[dict[str, object]]:
    if not results:
        return []
    width = len(results[0].state)
    size = 1 << width
    by_mask = {state_to_mask(row.state): row for row in results}
    if len(by_mask) != size:
        raise ValueError(f"require complete binary state space: {len(by_mask)} != {size}")
    if any(len(row.state) != width for row in results):
        raise ValueError("inconsistent state widths")

    feasible_count = 0
    for level in levels:
        if any(row.fidelity >= level for row in results):
            feasible_count += 1
        else:
            break
    active_levels = tuple(levels[:feasible_count])
    if not active_levels:
        return [_infeasible_row(results[0].example_id, level) for level in levels]

    parents: list[list[int | None]] = [[None] * size]
    previous = [
        float(by_mask[mask].tokens) if by_mask[mask].fidelity >= active_levels[0] else math.inf
        for mask in range(size)
    ]
    for level in active_levels[1:]:
        best_cost, best_arg = _best_submasks(previous, width)
        current = [math.inf] * size
        current_parent: list[int | None] = [None] * size
        for mask in range(size):
            row = by_mask[mask]
            if row.fidelity >= level and not math.isinf(best_cost[mask]):
                current[mask] = best_cost[mask] + row.tokens
                current_parent[mask] = best_arg[mask]
        previous = current
        parents.append(current_parent)

    final_mask = min(
        range(size),
        key=lambda mask: (previous[mask], by_mask[mask].tokens, mask),
    )
    if math.isinf(previous[final_mask]):
        raise RuntimeError("no binary nested chain spans the feasible levels")
    chosen = [final_mask]
    for level_index in range(len(active_levels) - 1, 0, -1):
        parent = parents[level_index][chosen[-1]]
        if parent is None:
            raise RuntimeError("broken binary nested-chain backpointer")
        chosen.append(parent)
    chosen.reverse()

    rows: list[dict[str, object]] = []
    cumulative = 0
    for level, mask in zip(active_levels, chosen):
        row = by_mask[mask]
        cumulative += row.tokens
        rows.append(
            {
                "example_id": row.example_id,
                "fidelity_level": level,
                "feasible": True,
                "tokens": row.tokens,
                "state": list(row.state),
                "achieved_fidelity": row.fidelity,
                "answer_f1": row.answer_f1,
                "fact_recall": row.fact_recall,
                "cumulative_tokens": cumulative,
            }
        )
    rows.extend(
        _infeasible_row(results[0].example_id, level)
        for level in levels[feasible_count:]
    )
    return rows


def _infeasible_row(example_id: str, level: float) -> dict[str, object]:
    return {
        "example_id": example_id,
        "fidelity_level": level,
        "feasible": False,
        "tokens": None,
        "state": None,
        "achieved_fidelity": None,
        "answer_f1": None,
        "fact_recall": None,
        "cumulative_tokens": None,
    }


def atomic_frontiers(
    results: Sequence[ExactSearchResult], levels: Sequence[float] = C_GRID
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return independent_frontier(results, levels), best_binary_nested_chain(results, levels)
