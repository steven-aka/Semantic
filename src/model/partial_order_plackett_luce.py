from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch


@dataclass(frozen=True)
class _Layer:
    predecessor_positions: torch.Tensor
    added_packets: torch.Tensor
    valid_transitions: torch.Tensor
    remaining_packets: torch.Tensor


def _canonical_preferences(
    preferences: Sequence[tuple[int, int]], width: int
) -> tuple[tuple[int, int], ...]:
    pairs = set()
    for winner, loser in preferences:
        winner, loser = int(winner), int(loser)
        if not 0 <= winner < width or not 0 <= loser < width or winner == loser:
            raise ValueError("invalid partial-order preference")
        pairs.add((winner, loser))
    return tuple(sorted(pairs))


@lru_cache(maxsize=8192)
def _compile_lattice(
    width: int, preferences: tuple[tuple[int, int], ...]
) -> tuple[_Layer, ...]:
    predecessors = [0] * width
    for winner, loser in preferences:
        predecessors[loser] |= 1 << winner

    valid_by_size: list[list[int]] = [[] for _ in range(width + 1)]
    for mask in range(1 << width):
        if all(
            not mask & (1 << packet) or not predecessors[packet] & ~mask
            for packet in range(width)
        ):
            valid_by_size[mask.bit_count()].append(mask)
    if valid_by_size[-1] != [(1 << width) - 1]:
        raise ValueError("partial-order preferences contain a cycle")

    layers = []
    for size in range(1, width + 1):
        previous = valid_by_size[size - 1]
        targets = valid_by_size[size]
        previous_position = {mask: index for index, mask in enumerate(previous)}
        predecessor_rows: list[list[int]] = []
        packet_rows: list[list[int]] = []
        transition_rows: list[list[bool]] = []
        max_predecessors = size
        for target in targets:
            transitions = []
            for packet in range(width):
                if not target & (1 << packet):
                    continue
                prior = target ^ (1 << packet)
                if prior in previous_position:
                    transitions.append((previous_position[prior], packet))
            if not transitions:
                raise RuntimeError("valid ideal has no valid predecessor")
            padding = max_predecessors - len(transitions)
            predecessor_rows.append(
                [position for position, _ in transitions] + [0] * padding
            )
            packet_rows.append([packet for _, packet in transitions] + [0] * padding)
            transition_rows.append([True] * len(transitions) + [False] * padding)
        remaining = [
            [not bool(mask & (1 << packet)) for packet in range(width)]
            for mask in previous
        ]
        layers.append(
            _Layer(
                predecessor_positions=torch.tensor(predecessor_rows, dtype=torch.long),
                added_packets=torch.tensor(packet_rows, dtype=torch.long),
                valid_transitions=torch.tensor(transition_rows, dtype=torch.bool),
                remaining_packets=torch.tensor(remaining, dtype=torch.bool),
            )
        )
    return tuple(layers)


def partial_order_plackett_luce_nll(
    scores: torch.Tensor,
    preferences: Sequence[Sequence[tuple[int, int]]],
) -> torch.Tensor:
    """Exact negative log probability of all valid partial-order extensions.

    A Plackett--Luce distribution is induced by the packet scores.  Dynamic
    programming marginalizes every linear extension consistent with the
    set-valued oracle, so ambiguous packet pairs are never assigned an
    arbitrary order.
    """
    if scores.ndim != 2 or len(preferences) != scores.shape[0]:
        raise ValueError("scores and preference batches must align")
    width = scores.shape[1]
    losses = []
    for batch_index, raw_preferences in enumerate(preferences):
        canonical = _canonical_preferences(raw_preferences, width)
        if not canonical:
            continue
        row = scores[batch_index].float()
        log_probability = row.new_zeros(1)
        for layer in _compile_lattice(width, canonical):
            predecessor_positions = layer.predecessor_positions.to(row.device)
            added_packets = layer.added_packets.to(row.device)
            valid_transitions = layer.valid_transitions.to(row.device)
            remaining_packets = layer.remaining_packets.to(row.device)
            remaining_scores = row.unsqueeze(0).expand(
                remaining_packets.shape[0], -1
            ).masked_fill(~remaining_packets, -torch.inf)
            denominators = torch.logsumexp(remaining_scores, dim=1)
            candidates = (
                log_probability[predecessor_positions]
                + row[added_packets]
                - denominators[predecessor_positions]
            ).masked_fill(~valid_transitions, -torch.inf)
            log_probability = torch.logsumexp(candidates, dim=1)
        losses.append(-log_probability[0])
    if not losses:
        raise ValueError("ranking batch contains no identifiable preference pairs")
    return torch.stack(losses).mean()
