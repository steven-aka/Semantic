from __future__ import annotations

import random
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from src.model.sequential_packet_policy import (
    SequentialEncoderInput,
    build_sequential_encoder_input,
)


class SequentialHistoryDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], tokenizer: Any, max_sequence_length: int):
        self.rows = list(rows)
        self.inputs = [
            build_sequential_encoder_input(
                tokenizer,
                row["question"],
                row["packet_texts"],
                max_sequence_length=max_sequence_length,
            )
            for row in self.rows
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], SequentialEncoderInput]:
        return self.rows[index], self.inputs[index]


def history_class(source: str) -> str:
    if source.startswith("oracle_"):
        return "oracle"
    if source.startswith("single_deviation_"):
        return "single"
    if source.startswith("random_"):
        return "random"
    if source.startswith("model_induced_"):
        return "model"
    raise ValueError(f"unknown history source: {source}")


class SequentialHistoryCollator:
    def __init__(self, seed: int, sample_counts: dict[str, int] | None = None):
        self.rng = random.Random(seed)
        self.sample_counts = sample_counts or {"oracle": 12, "single": 12, "random": 8}

    def _sample(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        groups = {name: [] for name in self.sample_counts}
        for item in row["histories"]:
            groups[history_class(item["source"])].append(item)
        sampled = []
        for name, count in self.sample_counts.items():
            values = groups[name]
            if not values:
                raise ValueError(f"example {row['example_id']} lacks {name} histories")
            if len(values) >= count:
                sampled.extend(self.rng.sample(values, count))
            else:
                sampled.extend(self.rng.choice(values) for _ in range(count))
        self.rng.shuffle(sampled)
        return sampled

    def __call__(
        self, items: Sequence[tuple[dict[str, Any], SequentialEncoderInput]]
    ) -> dict[str, Any]:
        rows = [item[0] for item in items]
        inputs = [item[1] for item in items]
        sampled = [self._sample(row) for row in rows]
        flat = [(example, history) for example, values in enumerate(sampled) for history in values]
        max_history = max(len(history["history"]) for _, history in flat)
        histories = torch.full((len(flat), max_history), -1, dtype=torch.long)
        lengths = torch.zeros(len(flat), dtype=torch.long)
        example_indices = torch.zeros(len(flat), dtype=torch.long)
        reached = torch.zeros(len(flat), dtype=torch.long)
        active_counts = torch.zeros(len(flat), dtype=torch.long)
        optimal: list[list[int]] = []
        advantages: list[list[float | None]] = []
        for index, (example, history) in enumerate(flat):
            values = history["history"]
            histories[index, : len(values)] = torch.tensor(values, dtype=torch.long)
            lengths[index] = len(values)
            example_indices[index] = example
            reached[index] = int(history["reached_levels"])
            active_counts[index] = len(rows[example]["active_levels"])
            optimal.append([int(value) for value in history["optimal_actions"]])
            if "action_advantages" in history:
                advantages.append(
                    [None if value is None else float(value) for value in history["action_advantages"]]
                )
        packet_tokens = torch.tensor([row["packet_tokens"] for row in rows], dtype=torch.float32)
        packet_fractions = packet_tokens / packet_tokens.sum(dim=1, keepdim=True)
        output = {
            "rows": rows,
            "inputs": inputs,
            "example_ids": [row["example_id"] for row in rows],
            "histories": histories,
            "history_lengths": lengths,
            "example_indices": example_indices,
            "reached_levels": reached,
            "active_level_counts": active_counts,
            "optimal_actions": optimal,
            "packet_token_fractions": packet_fractions,
        }
        if advantages:
            if len(advantages) != len(flat):
                raise ValueError("cost supervision must be present for every sampled history")
            output["action_advantages"] = advantages
        return output
