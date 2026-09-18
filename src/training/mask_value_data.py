from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from src.model.sequential_packet_policy import SequentialEncoderInput, build_sequential_encoder_input


class MaskValueDataset(Dataset):
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

    def __getitem__(self, index: int):
        return self.rows[index], self.inputs[index]


class MaskValueCollator:
    def __call__(self, items):
        rows = [item[0] for item in items]
        inputs = [item[1] for item in items]
        masks = []
        attained = []
        sampling_probabilities = []
        example_indices = []
        for example, row in enumerate(rows):
            for item in row["mask_values"]:
                masks.append(int(item["mask"]))
                attained.append(int(item["attained_levels"]))
                sampling_probabilities.append(float(item["sampling_probability"]))
                example_indices.append(example)
        return {
            "rows": rows,
            "inputs": inputs,
            "masks": torch.tensor(masks, dtype=torch.long),
            "attained_levels": torch.tensor(attained, dtype=torch.long),
            "sampling_probabilities": torch.tensor(sampling_probabilities, dtype=torch.float32),
            "example_indices": torch.tensor(example_indices, dtype=torch.long),
        }


class CachedMaskValueDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], cache_path: str):
        self.rows = list(rows)
        cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        expected = [row["example_id"] for row in self.rows]
        if cache["example_ids"] != expected:
            raise ValueError("embedding cache IDs do not exactly match data order")
        self.packets = cache["packets"]
        self.questions = cache["questions"]
        if tuple(self.packets.shape) != (len(self.rows), 12, 512):
            raise ValueError("invalid cached packet embedding shape")
        if tuple(self.questions.shape) != (len(self.rows), 512):
            raise ValueError("invalid cached question embedding shape")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index], self.packets[index], self.questions[index]


class CachedMaskValueCollator(MaskValueCollator):
    def __call__(self, items):
        batch = super().__call__([(item[0], None) for item in items])
        batch.pop("inputs")
        batch["packets"] = torch.stack([item[1] for item in items])
        batch["questions"] = torch.stack([item[2] for item in items])
        return batch
