from __future__ import annotations

import math
from typing import Sequence

from src.data.schemas import ExactSearchResult
from src.search.atomic_nested_chain import mask_to_state, state_to_mask
from src.search.exact_frontier import C_GRID


def validate_packet_order(order: Sequence[int], width: int) -> tuple[int, ...]:
    value = tuple(int(index) for index in order)
    if len(value) != width or set(value) != set(range(width)):
        raise ValueError("packet order must be a permutation of every packet index")
    return value


def prefix_state(order: Sequence[int], count: int) -> tuple[int, ...]:
    order = validate_packet_order(order, len(order))
    if not 0 <= count <= len(order):
        raise ValueError("prefix count outside packet order")
    state = [0] * len(order)
    for packet_index in order[:count]:
        state[packet_index] = 1
    return tuple(state)


def prefix_states(order: Sequence[int]) -> list[tuple[int, ...]]:
    return [prefix_state(order, count) for count in range(len(order) + 1)]


def order_from_nested_states(states: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not states:
        raise ValueError("nested states are required")
    width = len(states[0])
    if any(len(state) != width for state in states):
        raise ValueError("nested states have inconsistent widths")
    entries = []
    for packet_index in range(width):
        entry = next(
            (level_index for level_index, state in enumerate(states) if state[packet_index]),
            len(states),
        )
        entries.append(entry)
    order = tuple(sorted(range(width), key=lambda index: (entries[index], index)))
    return order, tuple(entries)


def best_prefix_nested_chain(
    results: Sequence[ExactSearchResult],
    order: Sequence[int],
    levels: Sequence[float] = C_GRID,
) -> list[dict[str, object]]:
    """Globally optimal monotone cutoffs for one fixed packet order."""
    if not results:
        return []
    width = len(results[0].state)
    order = validate_packet_order(order, width)
    by_state = {row.state: row for row in results}
    candidates = [by_state[state] for state in prefix_states(order)]

    feasible_count = 0
    for level in levels:
        if any(row.fidelity >= level for row in candidates):
            feasible_count += 1
        else:
            break
    active_levels = tuple(levels[:feasible_count])
    if not active_levels:
        return [_infeasible(results[0].example_id, level) for level in levels]

    size = width + 1
    parents: list[list[int | None]] = [[None] * size]
    previous = [
        float(row.tokens) if row.fidelity >= active_levels[0] else math.inf
        for row in candidates
    ]
    for level in active_levels[1:]:
        prefix_best: list[float] = []
        prefix_arg: list[int] = []
        best = math.inf
        best_index = 0
        for count, value in enumerate(previous):
            if (value, count) < (best, best_index):
                best, best_index = value, count
            prefix_best.append(best)
            prefix_arg.append(best_index)
        current = [math.inf] * size
        current_parent: list[int | None] = [None] * size
        for count, row in enumerate(candidates):
            if row.fidelity >= level and not math.isinf(prefix_best[count]):
                current[count] = prefix_best[count] + row.tokens
                current_parent[count] = prefix_arg[count]
        previous = current
        parents.append(current_parent)

    final_count = min(range(size), key=lambda count: (previous[count], count))
    if math.isinf(previous[final_count]):
        raise RuntimeError("no monotone prefix chain spans its feasible levels")
    counts = [final_count]
    for level_index in range(len(active_levels) - 1, 0, -1):
        parent = parents[level_index][counts[-1]]
        if parent is None:
            raise RuntimeError("broken prefix-chain backpointer")
        counts.append(parent)
    counts.reverse()

    output = []
    cumulative = 0
    for level, count in zip(active_levels, counts):
        row = candidates[count]
        cumulative += row.tokens
        output.append(
            {
                "example_id": row.example_id,
                "fidelity_level": level,
                "feasible": True,
                "cutoff_count": count,
                "tokens": row.tokens,
                "state": list(row.state),
                "achieved_fidelity": row.fidelity,
                "cumulative_tokens": cumulative,
            }
        )
    output.extend(_infeasible(results[0].example_id, level) for level in levels[feasible_count:])
    return output


def best_nested_chain_at_counts(
    results: Sequence[ExactSearchResult],
    counts: Sequence[int],
    levels: Sequence[float] = C_GRID,
) -> list[dict[str, object]]:
    """Oracle ordering with fixed monotone prefix counts.

    Any nested sequence of sets with nondecreasing cardinalities can be extended
    to a total packet order.  This DP therefore isolates cutoff quality without
    committing to an arbitrary tie-breaking order from one oracle chain.  It
    first minimizes contract violations, then cumulative tokens.
    """
    if not results or len(counts) != len(levels):
        raise ValueError("results, counts, and levels must be non-empty and aligned")
    width = len(results[0].state)
    if any(not 0 <= count <= width for count in counts):
        raise ValueError("cutoff count outside packet width")
    if any(a > b for a, b in zip(counts, counts[1:])):
        raise ValueError("cutoff counts must be nondecreasing")
    size = 1 << width
    by_mask = {state_to_mask(row.state): row for row in results}
    if len(by_mask) != size:
        raise ValueError("complete binary state space required")

    parents: list[list[int | None]] = [[None] * size]
    previous: list[tuple[int, float]] = [(width + 1, math.inf)] * size
    for mask in range(size):
        if mask.bit_count() == counts[0]:
            row = by_mask[mask]
            previous[mask] = (int(row.fidelity < levels[0]), float(row.tokens))

    for level_index in range(1, len(levels)):
        best = list(previous)
        arg = list(range(size))
        for bit in range(width):
            flag = 1 << bit
            for mask in range(size):
                if not mask & flag:
                    continue
                submask = mask ^ flag
                if (best[submask], arg[submask]) < (best[mask], arg[mask]):
                    best[mask], arg[mask] = best[submask], arg[submask]
        current = [(width + 1, math.inf)] * size
        current_parent: list[int | None] = [None] * size
        for mask in range(size):
            if mask.bit_count() != counts[level_index] or math.isinf(best[mask][1]):
                continue
            row = by_mask[mask]
            current[mask] = (
                best[mask][0] + int(row.fidelity < levels[level_index]),
                best[mask][1] + row.tokens,
            )
            current_parent[mask] = arg[mask]
        previous = current
        parents.append(current_parent)

    final_mask = min(range(size), key=lambda mask: (previous[mask], mask))
    if math.isinf(previous[final_mask][1]):
        raise RuntimeError("no nested chain realizes the requested cutoff counts")
    masks = [final_mask]
    for level_index in range(len(levels) - 1, 0, -1):
        parent = parents[level_index][masks[-1]]
        if parent is None:
            raise RuntimeError("broken fixed-count chain backpointer")
        masks.append(parent)
    masks.reverse()

    output = []
    cumulative = 0
    for level, count, mask in zip(levels, counts, masks):
        row = by_mask[mask]
        cumulative += row.tokens
        output.append(
            {
                "example_id": row.example_id,
                "fidelity_level": level,
                "feasible": row.fidelity >= level,
                "cutoff_count": count,
                "tokens": row.tokens,
                "state": list(row.state),
                "achieved_fidelity": row.fidelity,
                "cumulative_tokens": cumulative,
            }
        )
    return output


def _infeasible(example_id: str, level: float) -> dict[str, object]:
    return {
        "example_id": example_id,
        "fidelity_level": level,
        "feasible": False,
        "cutoff_count": None,
        "tokens": None,
        "state": None,
        "achieved_fidelity": None,
        "cumulative_tokens": None,
    }


def state_mask_for_order_count(order: Sequence[int], count: int) -> int:
    return state_to_mask(prefix_state(order, count))
