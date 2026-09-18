from __future__ import annotations

import argparse
import json
from statistics import mean, median

from transformers import AutoTokenizer

from src.data.schemas import read_jsonl
from src.model.input_builder import build_atomic_model_input
from src.reproducibility import sha256, write_metadata


def packet_input_ids(tokenizer: object, question: str, packet: str) -> list[int]:
    text = (
        f"Question:\n{question}\n\nEvidence packet:\n"
        f"{packet}\n\nEnd of packet."
    )
    ids = list(tokenizer(text, add_special_tokens=False).input_ids)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))
    return ids


def question_input_ids(tokenizer: object, question: str) -> list[int]:
    ids = list(
        tokenizer(
            f"Question:\n{question}\n\nQuestion representation.",
            add_special_tokens=False,
        ).input_ids
    )
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        ids.append(int(eos))
    return ids


def summarize(rows: list[dict], tokenizer: object) -> dict[str, float | int]:
    joint_lengths: list[int] = []
    packet_batch_lengths: list[int] = []
    linear_ratios: list[float] = []
    attention_ratios: list[float] = []
    naive_padded_linear_ratios: list[float] = []
    naive_padded_attention_ratios: list[float] = []
    four_bucket_linear_ratios: list[float] = []
    four_bucket_attention_ratios: list[float] = []
    for row in rows:
        joint = len(
            build_atomic_model_input(tokenizer, row["question"], row["packet_texts"]).input_ids
        )
        separate = [
            len(packet_input_ids(tokenizer, row["question"], packet))
            for packet in row["packet_texts"]
        ]
        separate.append(len(question_input_ids(tokenizer, row["question"])))
        total = sum(separate)
        joint_lengths.append(joint)
        packet_batch_lengths.append(total)
        linear_ratios.append(total / joint)
        attention_ratios.append(sum(length * length for length in separate) / (joint * joint))
        maximum = max(separate)
        naive_padded_linear_ratios.append(len(separate) * maximum / joint)
        naive_padded_attention_ratios.append(
            len(separate) * maximum * maximum / (joint * joint)
        )
        ordered = sorted(separate)
        buckets = [
            ordered[index * len(ordered) // 4 : (index + 1) * len(ordered) // 4]
            for index in range(4)
        ]
        bucket_tokens = sum(len(bucket) * max(bucket) for bucket in buckets if bucket)
        bucket_attention = sum(
            len(bucket) * max(bucket) * max(bucket) for bucket in buckets if bucket
        )
        four_bucket_linear_ratios.append(bucket_tokens / joint)
        four_bucket_attention_ratios.append(bucket_attention / (joint * joint))
    return {
        "examples": len(rows),
        "joint_tokens_mean": mean(joint_lengths),
        "joint_tokens_median": median(joint_lengths),
        "packet_batch_total_tokens_mean": mean(packet_batch_lengths),
        "linear_token_ratio_mean": mean(linear_ratios),
        "attention_sum_squared_length_ratio_mean": mean(attention_ratios),
        "attention_sum_squared_length_ratio_median": median(attention_ratios),
        "naive_single_batch_padded_linear_ratio_mean": mean(naive_padded_linear_ratios),
        "naive_single_batch_padded_attention_ratio_mean": mean(naive_padded_attention_ratios),
        "four_length_bucket_padded_linear_ratio_mean": mean(four_bucket_linear_ratios),
        "four_length_bucket_padded_attention_ratio_mean": mean(four_bucket_attention_ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Token-cost preflight for batched V8 packet encoding")
    parser.add_argument("--split", nargs=2, action="append", metavar=("NAME", "JSONL"), required=True)
    parser.add_argument("--model", default="models/Qwen3-4B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    forbidden = ("calibration300", "final_test300", "test300")
    if any(token in path.lower() for _, path in args.split for token in forbidden):
        raise ValueError("locked calibration/final-test roles must not enter this preflight")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    output = {
        "complete": True,
        "status": "tokenizer-only cost preflight; no Target or model inference",
        "input_format": "one order-invariant (question, packet) sequence per packet plus one question sequence; four length buckets avoid padding waste",
        "model": args.model,
        "splits": {},
        "artifacts": {},
    }
    for name, path in args.split:
        rows = list(read_jsonl(path))
        output["splits"][name] = summarize(rows, tokenizer)
        output["artifacts"][name] = {"path": path, "sha256": sha256(path)}
        print(json.dumps({"completed": name, "examples": len(rows)}), flush=True)
    write_metadata(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
