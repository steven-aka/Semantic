from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PredictedValue:
    reached: int
    cumulative_tokens: int

    @property
    def key(self) -> tuple[int, int]:
        return (-self.reached, self.cumulative_tokens)


def best_order_from_predicted_attainment(
    attained: Sequence[int], tokens: Sequence[int], level_count: int
) -> tuple[int, ...]:
    size = len(attained)
    if size == 0 or size & (size - 1) or len(tokens) != size:
        raise ValueError("attainment and tokens must describe a complete binary lattice")
    width = size.bit_length() - 1
    if any(not 0 <= value <= level_count for value in attained):
        raise ValueError("predicted attainment outside active anchor range")
    values = [[PredictedValue(0, 0) for _ in range(level_count + 1)] for _ in range(size)]
    choices = [[-1 for _ in range(level_count + 1)] for _ in range(size)]
    full = size - 1
    for reached in range(level_count + 1):
        values[full][reached] = PredictedValue(reached, 0)
    for count in range(width - 1, -1, -1):
        for mask in range(size):
            if mask.bit_count() != count:
                continue
            for reached in range(level_count + 1):
                candidates = []
                for packet in range(width):
                    if mask & (1 << packet):
                        continue
                    next_mask = mask | (1 << packet)
                    next_reached = max(reached, int(attained[next_mask]))
                    suffix = values[next_mask][next_reached]
                    candidate = PredictedValue(
                        suffix.reached,
                        (next_reached - reached) * int(tokens[next_mask]) + suffix.cumulative_tokens,
                    )
                    candidates.append((candidate.key, packet, candidate))
                _, packet, best = min(candidates)
                values[mask][reached] = best
                choices[mask][reached] = packet
    mask = 0
    reached = int(attained[0])
    order = []
    while mask != full:
        packet = choices[mask][reached]
        if packet < 0:
            raise RuntimeError("predicted trajectory reconstruction failed")
        order.append(packet)
        mask |= 1 << packet
        reached = max(reached, int(attained[mask]))
    return tuple(order)
