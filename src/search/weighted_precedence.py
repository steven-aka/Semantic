from __future__ import annotations

from math import inf
from typing import Sequence


def exact_weighted_precedence_order(logits: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Maximum-weight total order for an antisymmetric precedence tournament.

    The subset DP appends one packet at a time. Ties are deterministic and no
    confidence threshold, cycle deletion, or secondary learned score is used.
    """
    width = len(logits)
    if width == 0 or any(len(row) != width for row in logits):
        raise ValueError("precedence logits must be a nonempty square matrix")
    size = 1 << width
    best = [-inf] * size
    order: list[tuple[int, ...] | None] = [None] * size
    best[0] = 0.0
    order[0] = ()
    for mask in range(1, size):
        candidate_best = -inf
        candidate_order = None
        for last in range(width):
            flag = 1 << last
            if not mask & flag:
                continue
            prior = mask ^ flag
            increment = sum(float(logits[first][last]) for first in range(width) if prior & (1 << first))
            value = best[prior] + increment
            proposed = order[prior] + (last,)  # type: ignore[operator]
            if value > candidate_best + 1e-12 or (
                abs(value - candidate_best) <= 1e-12
                and (candidate_order is None or proposed < candidate_order)
            ):
                candidate_best = value
                candidate_order = proposed
        best[mask] = candidate_best
        order[mask] = candidate_order
    result = order[-1]
    if result is None:
        raise RuntimeError("weighted precedence decoding failed")
    return result
