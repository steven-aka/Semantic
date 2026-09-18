from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class SequentialEncoderInput:
    question_ids: list[int]
    packet_ids: list[list[int]]


def _encode(tokenizer: Any, text: str) -> list[int]:
    ids = list(tokenizer(text, add_special_tokens=False).input_ids)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))
    return ids


def build_sequential_encoder_input(
    tokenizer: Any,
    question: str,
    packet_texts: Sequence[str],
    *,
    max_sequence_length: int = 512,
) -> SequentialEncoderInput:
    if not question.strip() or len(packet_texts) != 12 or any(not value.strip() for value in packet_texts):
        raise ValueError("V8 requires one question and exactly twelve nonempty packets")
    question_ids = _encode(tokenizer, f"Question:\n{question}\n\nQuestion representation.")
    packet_ids = [
        _encode(
            tokenizer,
            f"Question:\n{question}\n\nEvidence packet:\n{text}\n\nEnd of packet.",
        )
        for text in packet_texts
    ]
    lengths = [len(question_ids), *(len(value) for value in packet_ids)]
    if max(lengths) > max_sequence_length:
        raise ValueError(
            f"V8 encoder sequence has {max(lengths)} tokens, exceeds {max_sequence_length}; "
            "scientific inputs are not silently truncated"
        )
    return SequentialEncoderInput(question_ids=question_ids, packet_ids=packet_ids)


class SequentialPacketPolicy(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        *,
        model_dim: int = 512,
        set_layers: int = 2,
        set_heads: int = 8,
        dropout: float = 0.1,
        length_buckets: int = 4,
    ):
        super().__init__()
        if model_dim % set_heads:
            raise ValueError("model dimension must be divisible by attention heads")
        self.lm = lm
        self.model_dim = model_dim
        self.length_buckets = length_buckets
        hidden = int(lm.config.hidden_size)
        self.input_projection = nn.Sequential(
            nn.Linear(hidden, model_dim), nn.GELU(), nn.LayerNorm(model_dim)
        )
        self.source_position = nn.Embedding(12, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=set_heads,
            dim_feedforward=2 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=set_layers)
        self.initial_history = nn.Linear(model_dim, model_dim)
        self.history_gru = nn.GRUCell(model_dim, model_dim)
        self.progress_head = nn.Sequential(
            nn.Linear(3 * model_dim, model_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(model_dim, 5)
        )
        self.progress_projection = nn.Linear(5, 64)
        feature_size = 7 * model_dim + 64 + 2
        self.action_head = nn.Sequential(
            nn.Linear(feature_size, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

    def head_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("lm."):
                yield parameter

    def move_head(self, *, device: torch.device | str, dtype: torch.dtype) -> None:
        for name, module in self.named_children():
            if name != "lm":
                module.to(device=device, dtype=dtype)

    def encode(
        self,
        inputs: Sequence[SequentialEncoderInput],
        *,
        pad_token_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        records: list[tuple[int, int, list[int]]] = []
        for batch_index, item in enumerate(inputs):
            records.append((batch_index, 12, item.question_ids))
            records.extend((batch_index, packet, ids) for packet, ids in enumerate(item.packet_ids))
        ordered = sorted(enumerate(records), key=lambda pair: len(pair[1][2]))
        encoded: dict[int, torch.Tensor] = {}
        for bucket in range(self.length_buckets):
            chunk = ordered[
                bucket * len(ordered) // self.length_buckets :
                (bucket + 1) * len(ordered) // self.length_buckets
            ]
            if not chunk:
                continue
            width = max(len(record[2]) for _, record in chunk)
            ids = torch.full((len(chunk), width), int(pad_token_id), dtype=torch.long, device=device)
            mask = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
            lengths = []
            for row_index, (_, (_, _, values)) in enumerate(chunk):
                length = len(values)
                ids[row_index, :length] = torch.tensor(values, dtype=torch.long, device=device)
                mask[row_index, :length] = 1
                lengths.append(length)
            base = self.lm.get_base_model() if hasattr(self.lm, "get_base_model") else self.lm
            backbone = getattr(base, "model", base)
            output = backbone(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                return_dict=True,
            )
            final = output.last_hidden_state
            for row_index, (flat_index, _) in enumerate(chunk):
                encoded[flat_index] = final[row_index, lengths[row_index] - 1]
        restored = torch.stack([encoded[index] for index in range(len(records))])
        restored = self.input_projection(restored.to(self.input_projection[0].weight.dtype))
        batch = len(inputs)
        restored = restored.reshape(batch, 13, self.model_dim)
        question = restored[:, 0]
        packets = restored[:, 1:]
        positions = self.source_position(torch.arange(12, device=device))[None, :, :]
        packets = self.set_encoder(packets + positions)
        return packets, question

    def history_states(
        self,
        packets: torch.Tensor,
        question: torch.Tensor,
        histories: torch.Tensor,
        history_lengths: torch.Tensor,
        example_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_packets = packets[example_indices]
        state = torch.tanh(self.initial_history(question[example_indices]))
        for depth in range(histories.shape[1]):
            active = history_lengths > depth
            indices = histories[:, depth].clamp_min(0)
            chosen = selected_packets[torch.arange(len(histories), device=packets.device), indices]
            updated = self.history_gru(chosen, state)
            state = torch.where(active[:, None], updated, state)
        selected_mask = torch.zeros(
            (len(histories), 12), dtype=torch.bool, device=packets.device
        )
        for depth in range(histories.shape[1]):
            active = history_lengths > depth
            indices = histories[:, depth].clamp_min(0)
            selected_mask[torch.arange(len(histories), device=packets.device), indices] |= active
        return state, selected_mask

    def score_states(
        self,
        packets: torch.Tensor,
        question: torch.Tensor,
        history_state: torch.Tensor,
        selected_mask: torch.Tensor,
        example_indices: torch.Tensor,
        packet_token_fractions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features, progress_logits = self.action_features(
            packets, question, history_state, selected_mask, example_indices, packet_token_fractions
        )
        logits = self.action_head(features).squeeze(-1)
        logits = logits.masked_fill(selected_mask, torch.finfo(logits.dtype).min)
        return logits, progress_logits

    def action_features(
        self,
        packets: torch.Tensor,
        question: torch.Tensor,
        history_state: torch.Tensor,
        selected_mask: torch.Tensor,
        example_indices: torch.Tensor,
        packet_token_fractions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the frozen V8 state-action representation for residual heads."""
        candidates = packets[example_indices]
        query = question[example_indices]
        selected_float = selected_mask.to(candidates.dtype)
        remaining_float = (~selected_mask).to(candidates.dtype)
        selected_count = selected_float.sum(dim=1, keepdim=True).clamp_min(1)
        remaining_count = remaining_float.sum(dim=1, keepdim=True).clamp_min(1)
        selected_pool = (candidates * selected_float[:, :, None]).sum(dim=1) / selected_count
        remaining_pool = (candidates * remaining_float[:, :, None]).sum(dim=1) / remaining_count
        progress_input = torch.cat((history_state, selected_pool, query), dim=-1)
        progress_logits = self.progress_head(progress_input)
        progress = self.progress_projection(torch.sigmoid(progress_logits))[:, None, :].expand(-1, 12, -1)
        history = history_state[:, None, :].expand(-1, 12, -1)
        selected = selected_pool[:, None, :].expand(-1, 12, -1)
        remaining = remaining_pool[:, None, :].expand(-1, 12, -1)
        query_pair = query[:, None, :].expand(-1, 12, -1)
        token_fraction = packet_token_fractions[example_indices, :, None].to(candidates.dtype)
        cumulative = (
            packet_token_fractions[example_indices] * selected_float
        ).sum(dim=1, keepdim=True)[:, :, None].expand(-1, 12, -1).to(candidates.dtype)
        features = torch.cat(
            (
                candidates,
                history,
                selected,
                remaining,
                query_pair,
                candidates * history,
                candidates * selected,
                progress,
                token_fraction,
                cumulative,
            ),
            dim=-1,
        )
        return features, progress_logits

    def forward_states(
        self,
        inputs: Sequence[SequentialEncoderInput],
        histories: torch.Tensor,
        history_lengths: torch.Tensor,
        example_indices: torch.Tensor,
        packet_token_fractions: torch.Tensor,
        *,
        pad_token_id: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        packets, question = self.encode(inputs, pad_token_id=pad_token_id, device=device)
        state, selected = self.history_states(
            packets, question, histories, history_lengths, example_indices
        )
        logits, progress = self.score_states(
            packets, question, state, selected, example_indices, packet_token_fractions
        )
        return logits, progress, packets, question

    def beam_order(
        self,
        packets: torch.Tensor,
        question: torch.Tensor,
        packet_token_fractions: torch.Tensor,
        *,
        beam_width: int = 8,
    ) -> tuple[int, ...]:
        if packets.shape != (12, self.model_dim) or question.shape != (self.model_dim,):
            raise ValueError("beam decoder expects one twelve-packet example")
        dtype = self.initial_history.weight.dtype
        packets = packets.to(dtype)
        question = question.to(dtype)
        beams: list[tuple[float, tuple[int, ...], torch.Tensor]] = [
            (0.0, (), torch.tanh(self.initial_history(question[None]))[0])
        ]
        for _ in range(12):
            beam_count = len(beams)
            states = torch.stack([value[2] for value in beams])
            selected = torch.zeros((beam_count, 12), dtype=torch.bool, device=packets.device)
            for beam_index, (_, history, _) in enumerate(beams):
                if history:
                    selected[beam_index, list(history)] = True
            logits, _ = self.score_states(
                packets[None].expand(beam_count, -1, -1),
                question[None].expand(beam_count, -1),
                states,
                selected,
                torch.arange(beam_count, dtype=torch.long, device=packets.device),
                packet_token_fractions[None].expand(beam_count, -1),
            )
            log_prob = torch.log_softmax(logits.float(), dim=-1).cpu()
            next_states = self.history_gru(
                packets[None].expand(beam_count, -1, -1).reshape(-1, self.model_dim),
                states[:, None, :].expand(-1, 12, -1).reshape(-1, self.model_dim),
            ).reshape(beam_count, 12, self.model_dim)
            expanded: list[tuple[float, tuple[int, ...], torch.Tensor]] = []
            for beam_index, (score, history, _) in enumerate(beams):
                for packet in range(12):
                    if selected[beam_index, packet]:
                        continue
                    expanded.append(
                        (
                            score + float(log_prob[beam_index, packet].item()),
                            history + (packet,),
                            next_states[beam_index, packet],
                        )
                    )
            expanded.sort(key=lambda value: (-value[0], value[1]))
            beams = expanded[:beam_width]
        return beams[0][1]
