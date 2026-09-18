from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

from src.data.schemas import ExactSearchResult
from src.search.atomic_nested_chain import state_to_mask


@dataclass(frozen=True, order=True)
class TrajectoryValue:
    """Lexicographic value for a suffix of a packet-reveal trajectory.

    ``reached_levels`` is maximized first.  Among suffixes that reach the same
    number of fidelity levels, ``additional_cumulative_tokens`` is minimized.
    """

    reached_levels: int
    additional_cumulative_tokens: int

    @property
    def comparison_key(self) -> tuple[int, int]:
        return (-self.reached_levels, self.additional_cumulative_tokens)


@dataclass(frozen=True)
class ActionAdvantage:
    """Exact severity of an action relative to the best continuation.

    The token component has range at most one. A two-unit penalty for each
    lost anchor therefore preserves the feasibility-first ordering without a
    tuned tradeoff coefficient.
    """

    packet: int
    lost_reachable_anchors: int
    normalized_token_difference: float
    scalar: float


@dataclass(frozen=True)
class HistorySupervision:
    history: tuple[int, ...]
    selected_mask: int
    reached_levels: int
    optimal_actions: tuple[int, ...]
    source: str


class SequentialTrajectoryDP:
    """Exact next-action oracle for the 12-packet binary lattice.

    The history variable ``reached`` is necessary because fidelity is not
    monotone when packets are added.  Two reveal sequences can end at the same
    selected set while having crossed different fidelity anchors earlier.
    """

    def __init__(self, results: Sequence[ExactSearchResult], levels: Sequence[float]):
        if not results or not levels:
            raise ValueError("non-empty exact results and fidelity levels are required")
        self.width = len(results[0].state)
        self.size = 1 << self.width
        self.levels = tuple(float(level) for level in levels)
        if any(first >= second for first, second in zip(self.levels, self.levels[1:])):
            raise ValueError("fidelity levels must be strictly increasing")
        by_mask = {state_to_mask(row.state): row for row in results}
        if len(by_mask) != self.size or any(len(row.state) != self.width for row in results):
            raise ValueError("a complete, fixed-width binary lattice is required")
        self.tokens = tuple(int(by_mask[mask].tokens) for mask in range(self.size))
        self.attained = tuple(
            sum(float(by_mask[mask].fidelity) + 1e-12 >= level for level in self.levels)
            for mask in range(self.size)
        )
        level_count = len(self.levels)
        self._values: list[list[TrajectoryValue]] = [
            [TrajectoryValue(0, 0) for _ in range(level_count + 1)]
            for _ in range(self.size)
        ]
        full = self.size - 1
        for reached in range(level_count + 1):
            self._values[full][reached] = TrajectoryValue(reached, 0)
        for selected_count in range(self.width - 1, -1, -1):
            for mask in range(self.size):
                if mask.bit_count() != selected_count:
                    continue
                for reached in range(level_count + 1):
                    candidates = self.action_values(mask, reached)
                    self._values[mask][reached] = min(
                        (value for _, value in candidates), key=lambda value: value.comparison_key
                    )

    def action_values(self, mask: int, reached: int) -> list[tuple[int, TrajectoryValue]]:
        self._validate_state(mask, reached)
        if mask == self.size - 1:
            return []
        output = []
        for packet in range(self.width):
            flag = 1 << packet
            if mask & flag:
                continue
            next_mask = mask | flag
            next_reached = max(reached, self.attained[next_mask])
            suffix = self._values[next_mask][next_reached]
            immediate = (next_reached - reached) * self.tokens[next_mask]
            output.append(
                (
                    packet,
                    TrajectoryValue(
                        suffix.reached_levels,
                        immediate + suffix.additional_cumulative_tokens,
                    ),
                )
            )
        return output

    def value(self, mask: int, reached: int) -> TrajectoryValue:
        self._validate_state(mask, reached)
        return self._values[mask][reached]

    def optimal_actions(
        self, mask: int, reached: int, *, normalized_slack: float = 0.0
    ) -> tuple[int, ...]:
        if normalized_slack < 0:
            raise ValueError("normalized_slack must be nonnegative")
        candidates = self.action_values(mask, reached)
        if not candidates:
            return ()
        best = min((value for _, value in candidates), key=lambda value: value.comparison_key)
        allowance = normalized_slack * len(self.levels) * self.tokens[-1]
        return tuple(
            packet
            for packet, value in candidates
            if value.reached_levels == best.reached_levels
            and value.additional_cumulative_tokens <= best.additional_cumulative_tokens + allowance + 1e-9
        )

    def action_advantages(self, mask: int, reached: int) -> tuple[ActionAdvantage, ...]:
        """Return exact, nonnegative cost-to-go advantages for every action."""
        candidates = self.action_values(mask, reached)
        if not candidates:
            return ()
        best = min((value for _, value in candidates), key=lambda value: value.comparison_key)
        token_normalizer = len(self.levels) * max(self.tokens)
        if token_normalizer <= 0:
            raise ValueError("positive token costs are required")
        output = []
        for packet, value in candidates:
            lost = best.reached_levels - value.reached_levels
            token_difference = (
                value.additional_cumulative_tokens - best.additional_cumulative_tokens
            ) / token_normalizer
            scalar = 2.0 * lost + token_difference
            if scalar < -1e-12:
                raise RuntimeError("lexicographic action advantage became negative")
            output.append(
                ActionAdvantage(
                    packet=packet,
                    lost_reachable_anchors=lost,
                    normalized_token_difference=token_difference,
                    scalar=max(0.0, scalar),
                )
            )
        return tuple(output)

    def canonical_rollout(self, *, normalized_slack: float = 0.0) -> dict[str, object]:
        mask = 0
        reached = self.attained[0]
        cumulative_tokens = reached * self.tokens[0]
        order: list[int] = []
        action_set_sizes: list[int] = []
        while mask != self.size - 1:
            actions = self.optimal_actions(mask, reached, normalized_slack=normalized_slack)
            if not actions:
                raise RuntimeError("nonterminal state has no optimal action")
            action_set_sizes.append(len(actions))
            # Deterministic artifact generation; the loss still treats every
            # member of the returned action set as correct.
            packet = min(actions)
            order.append(packet)
            mask |= 1 << packet
            next_reached = max(reached, self.attained[mask])
            cumulative_tokens += (next_reached - reached) * self.tokens[mask]
            reached = next_reached
        return {
            "order": order,
            "reached_levels": reached,
            "cumulative_tokens": cumulative_tokens,
            "action_set_sizes": action_set_sizes,
        }

    def reachable_history_levels(self) -> list[set[int]]:
        """Exact history states reachable by some ordering of each mask."""
        reachable = [set() for _ in range(self.size)]
        reachable[0].add(self.attained[0])
        for selected_count in range(self.width):
            for mask in range(self.size):
                if mask.bit_count() != selected_count:
                    continue
                for reached in tuple(reachable[mask]):
                    for packet in range(self.width):
                        flag = 1 << packet
                        if mask & flag:
                            continue
                        next_mask = mask | flag
                        reachable[next_mask].add(max(reached, self.attained[next_mask]))
        return reachable

    def one_run_supervision(
        self,
        *,
        seed: int,
        oracle_rollouts: int = 4,
        random_rollouts: int = 4,
    ) -> list[HistorySupervision]:
        """Build expert and recovery-state labels before model training.

        The pool contains tie-diverse optimal rollouts, one nearest-nonoptimal
        deviation at each reveal depth, and fully random histories.  It is a
        deterministic offline substitute for a train-then-DAgger-then-retrain
        loop.  Every label remains the exact optimal action set at the history
        actually visited.
        """
        if oracle_rollouts < 1 or random_rollouts < 0:
            raise ValueError("at least one oracle rollout and nonnegative random rollouts are required")
        rng = Random(seed)
        rows: dict[tuple[int, ...], HistorySupervision] = {}

        def record(history: list[int], mask: int, reached: int, source: str) -> None:
            key = tuple(history)
            actions = self.optimal_actions(mask, reached)
            if not actions:
                return
            candidate = HistorySupervision(key, mask, reached, actions, source)
            existing = rows.get(key)
            if existing is not None and (
                existing.selected_mask != mask
                or existing.reached_levels != reached
                or existing.optimal_actions != actions
            ):
                raise RuntimeError("one action history produced inconsistent supervision")
            rows.setdefault(key, candidate)

        def run(source: str, deviation_depth: int | None = None, random_policy: bool = False) -> None:
            history: list[int] = []
            mask = 0
            reached = self.attained[0]
            for depth in range(self.width):
                record(history, mask, reached, source)
                remaining = [packet for packet in range(self.width) if not mask & (1 << packet)]
                optimal = self.optimal_actions(mask, reached)
                if random_policy:
                    packet = rng.choice(remaining)
                elif deviation_depth == depth:
                    values = self.action_values(mask, reached)
                    alternatives = [
                        (value.comparison_key, packet)
                        for packet, value in values
                        if packet not in optimal
                    ]
                    packet = min(alternatives)[1] if alternatives else rng.choice(list(optimal))
                else:
                    packet = rng.choice(list(optimal))
                history.append(packet)
                mask |= 1 << packet
                reached = max(reached, self.attained[mask])

        for rollout in range(oracle_rollouts):
            run(f"oracle_{rollout}")
        for depth in range(self.width):
            run(f"single_deviation_{depth}", deviation_depth=depth)
        for rollout in range(random_rollouts):
            run(f"random_{rollout}", random_policy=True)
        return sorted(rows.values(), key=lambda row: (len(row.history), row.history))

    def _validate_state(self, mask: int, reached: int) -> None:
        if not 0 <= mask < self.size:
            raise ValueError("mask outside state space")
        if not 0 <= reached <= len(self.levels):
            raise ValueError("reached level count outside active levels")
