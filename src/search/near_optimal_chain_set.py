from __future__ import annotations

import math
from typing import Sequence

from src.data.schemas import ExactSearchResult
from src.search.atomic_nested_chain import state_to_mask
from src.search.exact_frontier import C_GRID


def _min_subsets(values: Sequence[float], width: int) -> list[float]:
    best = list(values)
    for bit in range(width):
        flag = 1 << bit
        for mask in range(1 << width):
            if mask & flag:
                best[mask] = min(best[mask], best[mask ^ flag])
    return best


def _min_supersets(values: Sequence[float], width: int) -> list[float]:
    best = list(values)
    for bit in range(width):
        flag = 1 << bit
        for mask in range(1 << width):
            if not mask & flag:
                best[mask] = min(best[mask], best[mask | flag])
    return best


def near_optimal_chain_membership(
    results: Sequence[ExactSearchResult],
    levels: Sequence[float] = C_GRID,
    *,
    normalized_slack: float = 0.0,
) -> dict[str, object]:
    """States occurring in any globally near-optimal nested chain.

    Slack is normalized by ``active_levels * full_state_tokens`` and applies to
    the same cumulative-rate objective used by ``best_binary_nested_chain``.
    The returned membership is chain-consistent because each state's minimum
    complete-chain cost is computed with forward and backward lattice DPs.
    """
    if not results:
        raise ValueError("exact results are required")
    if normalized_slack < 0:
        raise ValueError("normalized slack must be non-negative")
    width = len(results[0].state)
    size = 1 << width
    by_mask = {state_to_mask(row.state): row for row in results}
    if len(by_mask) != size:
        raise ValueError("complete binary state space required")

    feasible_count = 0
    for level in levels:
        if any(row.fidelity >= level for row in results):
            feasible_count += 1
        else:
            break
    active = tuple(levels[:feasible_count])
    if not active:
        raise ValueError("no feasible fidelity anchors")

    forward: list[list[float]] = []
    first = [
        float(by_mask[mask].tokens) if by_mask[mask].fidelity >= active[0] else math.inf
        for mask in range(size)
    ]
    forward.append(first)
    for level in active[1:]:
        prior = _min_subsets(forward[-1], width)
        forward.append(
            [
                prior[mask] + by_mask[mask].tokens
                if by_mask[mask].fidelity >= level and not math.isinf(prior[mask])
                else math.inf
                for mask in range(size)
            ]
        )

    backward: list[list[float]] = [[math.inf] * size for _ in active]
    backward[-1] = [
        float(by_mask[mask].tokens) if by_mask[mask].fidelity >= active[-1] else math.inf
        for mask in range(size)
    ]
    for level_index in range(len(active) - 2, -1, -1):
        later = _min_supersets(backward[level_index + 1], width)
        level = active[level_index]
        backward[level_index] = [
            later[mask] + by_mask[mask].tokens
            if by_mask[mask].fidelity >= level and not math.isinf(later[mask])
            else math.inf
            for mask in range(size)
        ]

    optimum = min(forward[-1])
    full_mask = size - 1
    normalizer = len(active) * by_mask[full_mask].tokens
    bound = optimum + normalized_slack * normalizer
    memberships: list[list[int]] = []
    for level_index in range(len(active)):
        members = []
        for mask in range(size):
            complete_cost = (
                forward[level_index][mask]
                + backward[level_index][mask]
                - by_mask[mask].tokens
            )
            if complete_cost <= bound + 1e-9:
                members.append(mask)
        memberships.append(members)
    return {
        "example_id": results[0].example_id,
        "active_levels": list(active),
        "optimal_cumulative_tokens": int(optimum),
        "normalized_slack": normalized_slack,
        "absolute_slack_tokens": normalized_slack * normalizer,
        "state_masks_by_level": memberships,
    }


def summarize_chain_membership(membership: dict[str, object], width: int) -> dict[str, object]:
    rows = []
    all_ambiguous = robust_included = robust_excluded = 0
    for level, masks in zip(
        membership["active_levels"], membership["state_masks_by_level"]
    ):
        counts = [sum(bool(mask & (1 << bit)) for mask in masks) for bit in range(width)]
        ambiguous = sum(0 < count < len(masks) for count in counts)
        included = sum(count == len(masks) for count in counts)
        excluded = sum(count == 0 for count in counts)
        all_ambiguous += ambiguous
        robust_included += included
        robust_excluded += excluded
        rows.append(
            {
                "fidelity_level": level,
                "admissible_state_count": len(masks),
                "ambiguous_packets": ambiguous,
                "robust_included_packets": included,
                "robust_excluded_packets": excluded,
                "packet_inclusion_fraction": [count / len(masks) for count in counts],
            }
        )
    denominator = len(rows) * width
    return {
        "levels": rows,
        "packet_anchor_ambiguity_fraction": all_ambiguous / denominator,
        "robust_included_fraction": robust_included / denominator,
        "robust_excluded_fraction": robust_excluded / denominator,
    }


def identifiable_pair_relations(
    membership: dict[str, object], width: int
) -> dict[tuple[int, int], int]:
    """Return stable precedence labels over a set of admissible chains.

    A value of ``-1`` means the first packet stably precedes the second; ``1``
    means the reverse.  Pairs that can occur in either order, always enter
    together, or are both never revealed are intentionally left unlabeled.
    """
    masks_by_level = membership["state_masks_by_level"]
    relations: dict[tuple[int, int], int] = {}
    for first in range(width):
        for second in range(first + 1, width):
            first_before = any(
                bool(mask & (1 << first)) and not bool(mask & (1 << second))
                for masks in masks_by_level
                for mask in masks
            )
            second_before = any(
                bool(mask & (1 << second)) and not bool(mask & (1 << first))
                for masks in masks_by_level
                for mask in masks
            )
            if first_before != second_before:
                relations[(first, second)] = -1 if first_before else 1
    return relations
