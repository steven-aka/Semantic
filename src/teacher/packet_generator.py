from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from src.data.schemas import QAExample, SemanticPacket, read_jsonl, write_jsonl
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import experiment_metadata, write_metadata
from src.teacher.packet_validator import (
    ValidationResult,
    numeric_concepts,
    validate_packet,
)


PACKET_PROMPT = """Compress the semantic unit below into additive packets. Do not use outside facts.

GIST requirements:
- preserve essential facts, entities, important numbers/dates, and negation
- target 20-35% of the source length
- the source is {source_tokens} target-tokenizer tokens
- GIST must be at most {target_gist_tokens} target-tokenizer tokens AND at most {target_gist_words} whitespace-separated words
- count conservatively and make GIST shorter when unsure

RESIDUAL requirements:
- add important source information missing from GIST
- do not repeat GIST and do not replace it
- use only information stated verbatim or unambiguously in the source

IDENTITY requirements:
- every source line beginning with "Document title:" names a source identity
- preserve every complete human-readable title identity somewhere across GIST and RESIDUAL
- MediaWiki brackets, link targets, and emphasis markers are formatting, not identity content

NUMBER requirements:
- preserve numeric meaning exactly and prefer the source's written form
- never calculate, expand, or introduce a number/date absent from the source
- allowed numeric values from this source: {allowed_numbers}

Return JSON only with exactly the keys \"gist\" and \"residual\".

SEMANTIC UNIT:
{source}
"""


def build_packet_prompt(
    source: str,
    source_tokens: int,
    target_gist_tokens: int,
    target_gist_words: int,
    previous_output: str | None = None,
    validation_error: str | None = None,
) -> str:
    prompt = PACKET_PROMPT.format(
        source=source,
        source_tokens=source_tokens,
        target_gist_tokens=target_gist_tokens,
        target_gist_words=target_gist_words,
        allowed_numbers=(", ".join(sorted(numeric_concepts(source))) or "NONE — do not output numbers"),
    )
    if previous_output is not None and validation_error is not None:
        prompt += (
            "\nThe previous output below was rejected. Correct every listed error and return "
            "a new JSON object only.\n"
            f"VALIDATION ERRORS:\n{validation_error}\n"
            f"PREVIOUS OUTPUT:\n{previous_output}\n"
        )
    return prompt


def parse_packet_json(text: str) -> dict[str, str]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    payload = json.loads(candidate)
    if set(payload) != {"gist", "residual"}:
        raise ValueError("teacher JSON must contain exactly gist and residual")
    if not all(isinstance(payload[key], str) for key in payload):
        raise ValueError("gist and residual must be strings")
    return payload


def rebalance_gist(
    gist: str, residual: str, tokenizer: Any, max_gist_tokens: int
) -> tuple[str, str]:
    """Move a word-aligned gist suffix into residual without dropping content."""
    words = gist.split()
    split_at = len(words)
    while split_at > 1 and count_tokens(tokenizer, " ".join(words[:split_at])) > max_gist_tokens:
        split_at -= 1
    kept = " ".join(words[:split_at]).strip()
    moved = " ".join(words[split_at:]).strip()
    combined_residual = " ".join(part for part in (moved, residual.strip()) if part)
    return kept, combined_residual


_INFORMATION_LIGHT_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "in", "is", "it", "of", "on", "or", "system", "that",
    "the", "this", "to", "was", "were", "with",
}


def collapse_redundant_residual(gist: str, residual: str) -> str | None:
    """Return an empty residual only when it adds no content-bearing word.

    Empty residuals have always been valid packets.  This deterministic final
    repair handles atomic one-fact sentences without weakening hallucination,
    number, or gist-length checks.  A residual containing even one new content
    word remains invalid and must not be silently discarded.
    """
    words = lambda text: {
        word.casefold()
        for word in re.findall(r"[A-Za-z][A-Za-z'-]*", text)
        if word.casefold() not in _INFORMATION_LIGHT_WORDS
    }
    return "" if words(residual) <= words(gist) else None


class PacketGenerator:
    def __init__(
        self,
        generate_batch: Callable[[Sequence[str]], Sequence[str]],
        tokenizer: Any,
        max_retries: int = 4,
        max_gist_ratio: float = 0.45,
        target_gist_ratio: float = 0.35,
    ) -> None:
        self._generate_batch = generate_batch
        self.tokenizer = tokenizer
        self.max_retries = max_retries
        self.max_gist_ratio = max_gist_ratio
        self.target_gist_ratio = target_gist_ratio

    def generate_for_units(self, example: QAExample) -> list[SemanticPacket]:
        pending = list(range(len(example.units)))
        packets: dict[int, SemanticPacket] = {}
        errors: dict[int, str] = {}
        last_outputs: dict[int, str] = {}
        for _attempt in range(self.max_retries + 1):
            if not pending:
                break
            prompts = []
            gist_word_budgets: dict[int, int] = {}
            gist_token_budgets: dict[int, int] = {}
            for index in pending:
                source = example.units[index].text
                source_tokens = count_tokens(self.tokenizer, source)
                target_gist_tokens = max(1, int(source_tokens * self.target_gist_ratio))
                gist_token_budgets[index] = target_gist_tokens
                # English words often occupy more than one target-tokenizer token.
                # A conservative word ceiling gives the teacher a bound it can count.
                target_gist_words = max(1, int(target_gist_tokens * 0.75))
                gist_word_budgets[index] = target_gist_words
                prompts.append(
                    build_packet_prompt(
                        source,
                        source_tokens,
                        target_gist_tokens,
                        target_gist_words,
                        last_outputs.get(index),
                        errors.get(index),
                    )
                )
            outputs = list(self._generate_batch(prompts))
            if len(outputs) != len(pending):
                raise RuntimeError("teacher returned a different number of outputs than prompts")
            retry: list[int] = []
            for index, output in zip(pending, outputs):
                last_outputs[index] = output
                source = example.units[index].text
                try:
                    parsed = parse_packet_json(output)
                    packet = SemanticPacket(
                        example_id=example.example_id,
                        unit_id=index,
                        source=source,
                        gist=parsed["gist"].strip(),
                        residual=parsed["residual"].strip(),
                        source_tokens=count_tokens(self.tokenizer, source),
                        gist_tokens=count_tokens(self.tokenizer, parsed["gist"]),
                        residual_tokens=count_tokens(self.tokenizer, parsed["residual"]),
                    )
                    validation = validate_packet(packet, self.max_gist_ratio)
                    if (
                        not validation.valid
                        and _attempt == self.max_retries
                        and any(error.startswith("gist exceeds") for error in validation.errors)
                    ):
                        gist, residual = rebalance_gist(
                            packet.gist,
                            packet.residual,
                            self.tokenizer,
                            gist_token_budgets[index],
                        )
                        packet = SemanticPacket(
                            example_id=packet.example_id,
                            unit_id=packet.unit_id,
                            source=packet.source,
                            gist=gist,
                            residual=residual,
                            source_tokens=packet.source_tokens,
                            gist_tokens=count_tokens(self.tokenizer, gist),
                            residual_tokens=count_tokens(self.tokenizer, residual),
                        )
                        validation = validate_packet(packet, self.max_gist_ratio)
                    if (
                        not validation.valid
                        and _attempt == self.max_retries
                        and validation.errors == ("residual excessively repeats gist",)
                    ):
                        residual = collapse_redundant_residual(packet.gist, packet.residual)
                        if residual is not None:
                            packet = SemanticPacket(
                                example_id=packet.example_id,
                                unit_id=packet.unit_id,
                                source=packet.source,
                                gist=packet.gist,
                                residual=residual,
                                source_tokens=packet.source_tokens,
                                gist_tokens=packet.gist_tokens,
                                residual_tokens=0,
                            )
                            validation = validate_packet(packet, self.max_gist_ratio)
                    if not validation.valid:
                        detail = "; ".join(validation.errors)
                        if packet.gist_tokens > int(packet.source_tokens * self.max_gist_ratio):
                            detail += (
                                f"; GIST has {packet.gist_tokens} target-tokenizer tokens, "
                                f"hard maximum is {int(packet.source_tokens * self.max_gist_ratio)}; "
                                f"rewrite GIST in at most {gist_word_budgets[index]} words"
                            )
                        raise ValueError(detail)
                    packets[index] = packet
                except (ValueError, json.JSONDecodeError) as exc:
                    errors[index] = str(exc)
                    retry.append(index)
            pending = retry
        if pending:
            details = "; ".join(
                f"unit {index}: {errors[index]}; last_output={last_outputs[index][:500]!r}"
                for index in pending
            )
            raise RuntimeError(f"packet generation failed after retries: {details}")
        return [packets[index] for index in range(len(example.units))]


class PacketCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, example_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", example_id)
        return self.directory / f"{safe_id}.jsonl"

    def load(self, example_id: str) -> list[SemanticPacket] | None:
        path = self.path_for(example_id)
        return list(read_jsonl(path, SemanticPacket)) if path.exists() else None

    def save(self, example_id: str, packets: Sequence[SemanticPacket]) -> Path:
        path = self.path_for(example_id)
        write_jsonl(path, packets)
        return path


def cached_packets_valid(
    example: QAExample, packets: Sequence[SemanticPacket] | None, tokenizer: Any | None = None
) -> bool:
    if packets is None or len(packets) != len(example.units):
        return False
    for unit, packet in zip(example.units, packets):
        if (
            packet.example_id != example.example_id
            or packet.unit_id != unit.unit_id
            or packet.source != unit.text
            or not validate_packet(packet).valid
        ):
            return False
        if tokenizer is not None and (
            packet.source_tokens != count_tokens(tokenizer, packet.source)
            or packet.gist_tokens != count_tokens(tokenizer, packet.gist)
            or packet.residual_tokens != count_tokens(tokenizer, packet.residual)
        ):
            return False
    return True


class VLLMTeacher:
    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        max_tokens: int = 512,
        gpu_memory_utilization: float = 0.75,
        max_model_len: int = 2048,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 4096,
        tensor_parallel_size: int = 1,
        cpu_offload_gb: float = 0.0,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm is required for teacher packet generation") from exc
        executor_backend = "mp" if tensor_parallel_size > 1 else None
        self.engine = LLM(
            model=model_name,
            revision=revision,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend=executor_backend,
            cpu_offload_gb=cpu_offload_gb,
        )
        self.tokenizer = self.engine.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        formatted = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                value = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                value = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            formatted.append(value)
        outputs = self.engine.generate(formatted, self.params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]


class TransformersTeacher:
    """Single-process model sharding fallback that does not require NCCL."""

    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        max_tokens: int = 512,
        gpu_memory_gb: str = "12",
        cpu_memory_gb: float = 60.0,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.device_count() < 2:
            raise RuntimeError("transformers teacher fallback requires two visible CUDA GPUs")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        limits = [float(value) for value in gpu_memory_gb.split(",")]
        if len(limits) == 1:
            limits *= torch.cuda.device_count()
        if len(limits) != torch.cuda.device_count():
            raise ValueError(
                "transformers GPU memory limits must contain one value or one per visible GPU"
            )
        max_memory: dict[int | str, str] = {
            device: f"{limits[device]}GiB" for device in range(torch.cuda.device_count())
        }
        max_memory["cpu"] = f"{cpu_memory_gb}GiB"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.max_tokens = max_tokens

    def __call__(self, prompts: Sequence[str]) -> list[str]:
        import torch

        results = []
        # Generate one semantic unit at a time to bound activation/KV memory.
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            encoded = self.tokenizer(formatted, return_tensors="pt")
            input_device = next(self.model.parameters()).device
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = output_ids[0, encoded["input_ids"].shape[1] :]
            results.append(self.tokenizer.decode(generated, skip_special_tokens=True))
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and validate teacher semantic packets")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output-dir", default="data/packets")
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--experiment-id", default="v0_teacher_packets")
    parser.add_argument("--dataset-revision", default="unknown")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--transformers-gpu-memory-gb", default="12")
    parser.add_argument("--transformers-cpu-memory-gb", type=float, default=60.0)
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer)
    examples = list(read_jsonl(args.examples, QAExample))
    cache = PacketCache(args.output_dir)
    pending: list[QAExample] = []
    for index, example in enumerate(examples):
        if args.limit is not None and index >= args.limit:
            break
        cached = None
        if not args.overwrite:
            try:
                cached = cache.load(example.example_id)
            except (OSError, TypeError, ValueError):
                pass
        if not cached_packets_valid(example, cached, tokenizer):
            pending.append(example)
    if not pending:
        print(f"[packets] validated and reused {min(len(examples), args.limit or len(examples))} examples", flush=True)
        return

    if args.backend == "transformers":
        teacher = TransformersTeacher(
            args.model,
            args.revision,
            max_tokens=args.max_new_tokens,
            gpu_memory_gb=args.transformers_gpu_memory_gb,
            cpu_memory_gb=args.transformers_cpu_memory_gb,
        )
    else:
        teacher = VLLMTeacher(
            args.model,
            args.revision,
            max_tokens=args.max_new_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            cpu_offload_gb=args.cpu_offload_gb,
        )
    generator = PacketGenerator(teacher, tokenizer, args.max_retries)
    for index, example in enumerate(pending, start=1):
        print(f"[packets {index}/{len(pending)}] run {example.example_id}", flush=True)
        cache.save(example.example_id, generator.generate_for_units(example))
    write_metadata(
        Path(args.output_dir) / "metadata.json",
        experiment_metadata(
            experiment_id=args.experiment_id,
            seed=args.seed,
            model_name=args.model,
            model_revision=args.revision,
            dataset_revision=args.dataset_revision,
            tokenizer_name=args.tokenizer,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            distributed_executor_backend=("mp" if args.tensor_parallel_size > 1 else "auto"),
            cpu_offload_gb=args.cpu_offload_gb,
            teacher_backend=args.backend,
            transformers_gpu_memory_gb=args.transformers_gpu_memory_gb,
            transformers_cpu_memory_gb=args.transformers_cpu_memory_gb,
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": args.max_new_tokens,
                "max_retries": args.max_retries,
                "final_retry_additive_rebalance": True,
            },
            input_path=args.examples,
        ),
    )


if __name__ == "__main__":
    main()
