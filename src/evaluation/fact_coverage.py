from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data.schemas import QAExample, SemanticPacket, read_jsonl, write_jsonl
from src.representation.packet_store import PacketStore
from src.reproducibility import experiment_metadata, write_metadata
from src.target.answer_parser import parse_answer
from src.target.qwen_runner import TargetRunner


FACT_QUESTION = (
    "Does the supplied context explicitly support the complete claim below? "
    "Answer yes only when every part of the claim is stated or directly entailed; "
    "otherwise answer no.\n\nClaim: {claim}"
)


def _normalized_space(text: str) -> str:
    return " ".join(text.split()).casefold()


def raw_supporting_facts(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = row["context"]
    contexts = dict(zip(context["title"], context["sentences"]))
    support = row["supporting_facts"]
    facts = []
    for title, sentence_id in zip(support["title"], support["sent_id"]):
        facts.append(
            {
                "title": str(title),
                "sentence_id": int(sentence_id),
                "text": str(contexts[title][sentence_id]).strip(),
            }
        )
    return facts


def map_facts_to_units(
    example: QAExample,
    facts: Sequence[Mapping[str, Any]],
    *,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    mapped = []
    for fact in facts:
        needle = _normalized_space(str(fact["text"]))
        matches = [
            unit.unit_id
            for unit in example.units
            if (unit.title is None or unit.title == str(fact["title"]))
            and (
                unit.source_sentence_id is None
                or unit.source_sentence_id == int(fact["sentence_id"])
            )
            and needle in _normalized_space(unit.text)
        ]
        if not matches and allow_missing:
            mapped.append({**dict(fact), "unit_id": None})
            continue
        if len(matches) != 1:
            raise ValueError(
                f"{example.example_id}: supporting fact maps to {len(matches)} units: "
                f"{fact['title']}#{fact['sentence_id']}"
            )
        mapped.append({**dict(fact), "unit_id": matches[0]})
    return mapped


def binary_verdict(text: str) -> bool:
    value = parse_answer(text).strip().casefold()
    value = re.sub(r"[.!?]+$", "", value).strip()
    if value not in {"yes", "no"}:
        raise ValueError(f"fact judge did not return yes/no: {text!r}")
    return value == "yes"


def evaluate_fact_coverage(
    examples: Sequence[QAExample],
    raw_rows: Mapping[str, Mapping[str, Any]],
    packets_by_example: Mapping[str, Sequence[SemanticPacket]],
    target: Any,
    *,
    allow_missing: bool = False,
) -> list[dict[str, Any]]:
    requests: list[tuple[str, dict[str, Any], str]] = []
    records: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        if example.example_id not in raw_rows:
            raise ValueError(f"raw HotpotQA row missing for {example.example_id}")
        packets = {packet.unit_id: packet for packet in packets_by_example[example.example_id]}
        facts = map_facts_to_units(
            example,
            raw_supporting_facts(raw_rows[example.example_id]),
            allow_missing=allow_missing,
        )
        records[example.example_id] = facts
        for fact in facts:
            if fact["unit_id"] is None:
                continue
            packet = packets[fact["unit_id"]]
            requests.append((example.example_id, fact, packet.gist.strip()))
            requests.append(
                (
                    example.example_id,
                    fact,
                    "\n".join(part for part in (packet.gist.strip(), packet.residual.strip()) if part),
                )
            )

    questions = [FACT_QUESTION.format(claim=fact["text"]) for _, fact, _ in requests]
    contexts = [context for _, _, context in requests]
    raw_predictions = target.generate_batch(questions, contexts)
    if len(raw_predictions) != len(requests):
        raise RuntimeError("fact judge returned a different number of outputs than requests")

    cursor = 0
    output = []
    for example in examples:
        evaluated = []
        for fact in records[example.example_id]:
            if fact["unit_id"] is None:
                evaluated.append(
                    {
                        **fact,
                        "selection_status": "not_selected",
                        "gist_prediction": None,
                        "full_prediction": None,
                        "gist_supported": False,
                        "full_supported_raw": False,
                        "full_supported": False,
                    }
                )
                continue
            gist_raw = raw_predictions[cursor]
            full_raw = raw_predictions[cursor + 1]
            cursor += 2
            gist_supported = binary_verdict(gist_raw)
            full_supported_raw = binary_verdict(full_raw)
            evaluated.append(
                {
                    **fact,
                    "selection_status": "selected",
                    "gist_prediction": gist_raw,
                    "full_prediction": full_raw,
                    "gist_supported": gist_supported,
                    "full_supported_raw": full_supported_raw,
                    # state 2 contains state 1 verbatim, so semantic coverage cannot decrease.
                    "full_supported": gist_supported or full_supported_raw,
                }
            )
        output.append({"example_id": example.example_id, "facts": evaluated})
    return output


def validate_fact_coverage(
    examples: Sequence[QAExample],
    raw_rows: Mapping[str, Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    *,
    allow_missing: bool = False,
) -> list[str]:
    errors: list[str] = []
    if [row.get("example_id") for row in coverage_rows] != [
        example.example_id for example in examples
    ]:
        errors.append("coverage example ids/order do not match inputs")
        return errors
    for example, row in zip(examples, coverage_rows):
        expected = map_facts_to_units(
            example,
            raw_supporting_facts(raw_rows[example.example_id]),
            allow_missing=allow_missing,
        )
        facts = row.get("facts")
        if not isinstance(facts, list) or len(facts) != len(expected):
            errors.append(f"{example.example_id}: fact count mismatch")
            continue
        for index, (actual, wanted) in enumerate(zip(facts, expected)):
            identity = ("title", "sentence_id", "text", "unit_id")
            if any(actual.get(key) != wanted[key] for key in identity):
                errors.append(f"{example.example_id}: fact {index} identity mismatch")
                continue
            if wanted["unit_id"] is None:
                if actual.get("selection_status") != "not_selected":
                    errors.append(f"{example.example_id}: fact {index} missing selection status")
                if any(
                    actual.get(key) is not False
                    for key in ("gist_supported", "full_supported_raw", "full_supported")
                ):
                    errors.append(f"{example.example_id}: fact {index} missing fact retained")
                continue
            try:
                gist = binary_verdict(str(actual["gist_prediction"]))
                full_raw = binary_verdict(str(actual["full_prediction"]))
            except (KeyError, ValueError) as exc:
                errors.append(f"{example.example_id}: fact {index} verdict error: {exc}")
                continue
            if actual.get("gist_supported") is not gist:
                errors.append(f"{example.example_id}: fact {index} gist verdict mismatch")
            if actual.get("full_supported_raw") is not full_raw:
                errors.append(f"{example.example_id}: fact {index} full verdict mismatch")
            if actual.get("full_supported") is not (gist or full_raw):
                errors.append(f"{example.example_id}: fact {index} monotonic closure mismatch")
    return errors


def _load_selected_raw(path: str | Path, example_ids: set[str]) -> dict[str, Mapping[str, Any]]:
    import pyarrow.parquet as parquet

    selected = {}
    columns = ["id", "context", "supporting_facts"]
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=256, columns=columns):
        for row in batch.to_pylist():
            if row["id"] in example_ids:
                selected[row["id"]] = row
        if len(selected) == len(example_ids):
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit packet coverage of gold HotpotQA facts")
    parser.add_argument("--raw", default="data/raw/hotpot_validation.parquet")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--output", default="results/v0_1/fact_coverage.jsonl")
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--experiment-id", default="v0_1_content_fact_coverage")
    args = parser.parse_args()

    examples = list(read_jsonl(args.examples, QAExample))
    raw_rows = _load_selected_raw(args.raw, {example.example_id for example in examples})
    store = PacketStore(args.packet_dir)
    packets = {example.example_id: store.get(example.example_id) for example in examples}
    target = TargetRunner(
        args.model,
        revision=args.revision,
        max_new_tokens=16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    rows = evaluate_fact_coverage(examples, raw_rows, packets, target)
    write_jsonl(args.output, rows)
    write_metadata(
        Path(args.output).with_suffix(".metadata.json"),
        experiment_metadata(
            experiment_id=args.experiment_id,
            model_name=args.model,
            model_revision=args.revision,
            source_examples=args.examples,
            source_packets=args.packet_dir,
            source_raw=args.raw,
            judgment="frozen_target_binary_explicit_support",
            monotonic_full_closure=True,
            generation_parameters={"temperature": 0.0, "max_tokens": 16},
        ),
    )


if __name__ == "__main__":
    main()
