from __future__ import annotations

from typing import Sequence


def chain_to_reveal_thresholds(
    states: Sequence[Sequence[int]], levels: Sequence[float]
) -> tuple[list[float], list[int], list[list[int]]]:
    if len(states) != len(levels) or not states:
        raise ValueError("states and levels must be non-empty and aligned")
    width = len(states[0])
    if any(len(state) != width or any(value not in (0, 1) for value in state) for state in states):
        raise ValueError("aligned binary states required")
    if any(not 0 <= level < 1 for level in levels):
        raise ValueError("training fidelity levels must be in [0,1)")
    if any(lower >= higher for lower, higher in zip(levels, levels[1:])):
        raise ValueError("levels must be strictly increasing")
    for lower, higher in zip(states, states[1:]):
        if any(a > b for a, b in zip(lower, higher)):
            raise ValueError("states do not form a nested chain")

    thresholds: list[float] = []
    reveal_bins: list[int] = []
    ordinal_labels: list[list[int]] = []
    for packet_index in range(width):
        labels = [int(state[packet_index]) for state in states]
        first = next((index for index, value in enumerate(labels) if value), len(levels))
        reveal_bins.append(first)
        thresholds.append(float(levels[first]) if first < len(levels) else 1.0)
        ordinal_labels.append(labels)
    return thresholds, reveal_bins, ordinal_labels


def thresholds_to_chain(
    thresholds: Sequence[float], levels: Sequence[float]
) -> list[tuple[int, ...]]:
    if any(not 0 <= threshold <= 1 for threshold in thresholds):
        raise ValueError("thresholds must be in [0,1]")
    return [tuple(int(level >= threshold) for threshold in thresholds) for level in levels]


def soft_reveal_probability(c: float, threshold: float, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    import math

    value = max(-60.0, min(60.0, (c - threshold) / temperature))
    return 1.0 / (1.0 + math.exp(-value))
