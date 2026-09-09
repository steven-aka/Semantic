from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from src.target.answer_parser import parse_answer


QA_SYSTEM_PROMPT = (
    "Use only the supplied context. Put only the shortest final answer between "
    "<answer> and </answer>, without explanation. For a yes/no question, the answer "
    "inside the tags must be exactly yes or no."
)


def build_qa_prompt(question: str, context: str) -> str:
    return (
        "Use only the context below to answer the question.\n"
        "Put only the shortest final answer between <answer> and </answer>, without explanation.\n"
        "For a yes/no question, put exactly yes or no inside the tags.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )


class TargetRunner:
    """Frozen Qwen target with a stable single/batch interface."""

    def __init__(
        self,
        model_name: str,
        backend: str = "vllm",
        revision: str = "main",
        max_new_tokens: int = 64,
        gpu_memory_utilization: float = 0.65,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
        engine: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.model = None
        self.tokenizer = tokenizer
        if engine is not None:
            self.model = engine
            return
        if backend == "vllm":
            try:
                from vllm import LLM
            except ImportError as exc:
                raise RuntimeError("vllm is required for the batched V0 experiment") from exc
            executor_backend = "mp" if tensor_parallel_size > 1 else None
            self.model = LLM(
                model=model_name,
                revision=revision,
                trust_remote_code=True,
                dtype="bfloat16",
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend=executor_backend,
            )
            self.tokenizer = self.model.get_tokenizer()
        elif backend == "transformers":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("torch and transformers are required") from exc
            self.tokenizer = self.tokenizer or AutoTokenizer.from_pretrained(
                model_name, revision=revision, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
        else:
            raise ValueError(f"unknown backend: {backend}")

    def _chat_prompts(self, questions: Sequence[str], contexts: Sequence[str]) -> list[str]:
        prompts = []
        for question, context in zip(questions, contexts):
            messages = [
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": build_qa_prompt(question, context)},
            ]
            if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            else:
                prompt = build_qa_prompt(question, context)
            prompts.append(prompt)
        return prompts

    def generate_batch(self, questions: Sequence[str], contexts: Sequence[str]) -> list[str]:
        if len(questions) != len(contexts):
            raise ValueError("questions and contexts must have equal length")
        if not questions:
            return []
        prompts = self._chat_prompts(questions, contexts)
        if self.backend == "vllm":
            from vllm import SamplingParams

            params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens)
            outputs = self.model.generate(prompts, params, use_tqdm=False)
            return [output.outputs[0].text.strip() for output in outputs]
        import torch

        batch = self.tokenizer(prompts, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **batch, do_sample=False, max_new_tokens=self.max_new_tokens
            )
        answers = []
        input_width = batch["input_ids"].shape[1]
        for output in generated:
            answers.append(self.tokenizer.decode(output[input_width:], skip_special_tokens=True).strip())
        return answers

    def answer_batch(self, questions: Sequence[str], contexts: Sequence[str]) -> list[str]:
        return [parse_answer(text) for text in self.generate_batch(questions, contexts)]

    def answer(self, question: str, context: str) -> str:
        return self.answer_batch([question], [context])[0]


class CallableTargetRunner:
    """Small deterministic/injected runner used by tests and dry runs."""

    def __init__(self, function: Callable[[str, str], str]) -> None:
        self.function = function

    def answer(self, question: str, context: str) -> str:
        return self.function(question, context)

    def answer_batch(self, questions: Sequence[str], contexts: Sequence[str]) -> list[str]:
        return [self.function(question, context) for question, context in zip(questions, contexts)]
