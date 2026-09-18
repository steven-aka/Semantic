from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from src.model.sequential_packet_policy import (
    SequentialPacketPolicy,
    build_sequential_encoder_input,
)
from src.model.sequential_policy_loss import sequential_policy_loss
from src.training.sequential_policy_data import history_class


class TinyTokenizer:
    eos_token_id = 1

    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str, add_special_tokens: bool = False):
        self.texts.append(text)
        return types.SimpleNamespace(input_ids=list(range(2, 2 + len(text.split()))))


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=16)
        self.embedding = nn.Embedding(256, 16)

    def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
        return types.SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class V8SequentialPolicyTest(unittest.TestCase):
    def test_model_induced_history_class_is_available_for_bounded_dagger(self) -> None:
        self.assertEqual(history_class("model_induced_v8_beam8"), "model")

    def test_encoder_text_has_no_packet_index(self) -> None:
        tokenizer = TinyTokenizer()
        item = build_sequential_encoder_input(
            tokenizer, "question", [f"evidence {index}" for index in range(12)]
        )
        self.assertEqual(12, len(item.packet_ids))
        packet_prompts = tokenizer.texts[1:]
        self.assertTrue(all("Evidence packet:" in text for text in packet_prompts))
        self.assertTrue(all("Packet 0" not in text and "packet 0:" not in text for text in packet_prompts))

    def test_history_state_and_action_shapes(self) -> None:
        model = SequentialPacketPolicy(TinyLM(), model_dim=16, set_heads=4, length_buckets=2)
        tokenizer = TinyTokenizer()
        inputs = [
            build_sequential_encoder_input(tokenizer, "q", [f"p {i}" for i in range(12)])
        ]
        histories = torch.tensor([[0, 1], [2, -1]])
        lengths = torch.tensor([2, 1])
        examples = torch.tensor([0, 0])
        fractions = torch.full((1, 12), 1 / 12)
        logits, progress, packets, question = model.forward_states(
            inputs,
            histories,
            lengths,
            examples,
            fractions,
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        self.assertEqual((2, 12), tuple(logits.shape))
        self.assertEqual((2, 5), tuple(progress.shape))
        self.assertEqual((1, 12, 16), tuple(packets.shape))
        self.assertEqual((1, 16), tuple(question.shape))
        self.assertLess(logits[0, 0].item(), -1e20)
        self.assertLess(logits[0, 1].item(), -1e20)

    def test_set_valued_loss_accepts_any_optimal_action(self) -> None:
        logits = torch.zeros((2, 12))
        logits[0, 3] = 4
        logits[1, 5] = 4
        result = sequential_policy_loss(
            logits,
            torch.zeros((2, 5)),
            [[3, 4], [5]],
            torch.tensor([2, 3]),
            torch.tensor([5, 4]),
        )
        self.assertEqual(1.0, result.action_accuracy)
        self.assertTrue(torch.isfinite(result.total))


if __name__ == "__main__":
    unittest.main()
