from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.qampari_metrics import normalize_list_answer, parse_list_prediction, qampari_list_metrics
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.target.qampari_runner import QampariTargetRunner
from src.training.train_v17b1_multi_anchor_viability import read_gzip


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def stratum(f1: float) -> str | None:
    for target, name in ((8/9, "f1_0888889"), (.9, "f1_0900000"), (10/11, "f1_0909091"), (18/19, "f1_0947368")):
        if abs(f1 - target) <= 1e-9:
            return name
    return "far_below_070" if f1 <= .7 else None


def select_masks(config: dict[str, Any], manifest: list[dict[str, Any]], records: dict[tuple[str, tuple[int, ...]], dict[str, Any]], exact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    included: dict[tuple[str, int], dict[str, Any]] = {}
    strata: dict[str, list[tuple[str, str, int, float]]] = defaultdict(list)
    queries = [row["example_id"] for row in manifest]
    for query in queries:
        exact = {state_to_mask(row.state): row for row in read_jsonl(exact_dir / f"{query}.jsonl", ExactSearchResult)}
        for mask, row in exact.items():
            group = stratum(float(row.fidelity))
            if group is not None:
                strata[group].append((sha(f"G0|{query}|{mask}"), query, mask, float(row.fidelity)))
        if query in {row["example_id"] for row in manifest[:config["complete_boundary_groups"]]}:
            selected = next(row for row in manifest if row["example_id"] == query)
            record = records[(query, tuple(selected["history"]))]
            current = int(record["selected_mask"])
            for mask in [current, *(current | (1 << int(action["packet"])) for action in record["actions"])]:
                cached = exact[mask]
                included[(query, mask)] = {"example_id": query, "mask": mask, "cached_prediction": cached.prediction, "cached_f1": float(cached.fidelity), "selection_roles": ["complete_boundary_group"]}
    available = {group: len(rows) for group, rows in strata.items()}
    for group in config["strata"]:
        picked = 0
        per_query: Counter[str] = Counter()
        for _, query, mask, f1 in sorted(strata[group]):
            if per_query[query] >= 2:
                continue
            per_query[query] += 1
            key = (query, mask)
            if key not in included:
                exact_row = next(row for row in read_jsonl(exact_dir / f"{query}.jsonl", ExactSearchResult) if state_to_mask(row.state) == mask)
                included[key] = {"example_id": query, "mask": mask, "cached_prediction": exact_row.prediction, "cached_f1": f1, "selection_roles": []}
            included[key]["selection_roles"].append(group)
            picked += 1
            if picked == config["stratum_size"]:
                break
    return sorted(included.values(), key=lambda row: (row["example_id"], row["mask"])), {"available_strata": available, "selected_roles": dict(Counter(role for row in included.values() for role in row["selection_roles"]))}


def normalized_set(value: str) -> tuple[str, ...]:
    return tuple(sorted(normalize_list_answer(part) for part in value.split(" # ") if part.strip()))


def same_f1(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-9


def aggregate(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    for name in ("parsed_string_agreement", "normalized_set_agreement", "list_f1_agreement", "threshold_label_agreement"):
        metrics[name] = mean(float(value[name]) for row in rows for value in row["reruns"])
    repeated_flips = sum(len({item["success"] for item in row["reruns"]}) > 1 for row in rows)
    cache_flips = sum(any(item["success"] != row["cached_success"] for item in row["reruns"]) for row in rows)
    by_stratum = {}
    for group in config["strata"]:
        subset = [row for row in rows if group in row["selection_roles"]]
        by_stratum[group] = {"masks": len(subset), "cache_to_rerun_threshold_flip_masks": sum(any(item["success"] != row["cached_success"] for item in row["reruns"]) for row in subset), "within_process_threshold_flip_masks": sum(len({item["success"] for item in row["reruns"]}) > 1 for row in subset)}
    return {"masks": len(rows), "queries": len({row["example_id"] for row in rows}), "comparisons": len(rows) * config["repeat_generations"], "metrics": metrics, "cache_to_rerun_threshold_flip_masks": cache_flips, "within_process_threshold_flip_masks": repeated_flips, "by_stratum": by_stratum, "cost": {"target_calls": len(rows) * config["repeat_generations"], "prompt_tokens": sum(item["prompt_tokens"] for row in rows for item in row["reruns"]), "generated_tokens": sum(item["generated_tokens"] for row in rows for item in row["reruns"])}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--training-artifact", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.protocol_config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_RUN":
        raise ValueError("unfrozen protocol")
    manifest = list(read_jsonl(args.selection_manifest))
    records = {(row["example_id"], tuple(row["history"])): row for row in read_gzip(args.training_artifact)}
    masks, selection = select_masks(config, manifest, records, Path(args.exact_dir))
    sources = {row["example_id"]: row for row in read_jsonl(args.train_data)}
    query_ids = {row["example_id"] for row in masks}
    annotations = {row["example_id"]: row["answer_atoms"] for row in read_jsonl(args.annotations) if row["example_id"] in query_ids}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "mask_selection_manifest.jsonl", [{key: value for key, value in row.items() if key != "cached_prediction"} for row in masks])
    target = QampariTargetRunner(config["target_model"], backend="vllm", max_new_tokens=config["max_new_tokens"], gpu_memory_utilization=config["gpu_memory_utilization"], max_model_len=config["max_model_len"])
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, max_tokens=config["max_new_tokens"])
    for row in masks:
        source = sources[row["example_id"]]
        row["question"] = source["question"]
        row["context"] = "\n\n".join(text.strip() for packet, text in enumerate(source["packet_texts"]) if row["mask"] & (1 << packet))
        row["cached_success"] = row["cached_f1"] + 1e-12 >= config["threshold"]
        row["reruns"] = []
    for repeat in range(config["repeat_generations"]):
        for start in range(0, len(masks), config["batch_size"]):
            batch = masks[start:start + config["batch_size"]]
            prompts = target._chat_prompts([row["question"] for row in batch], [row["context"] for row in batch])
            generated = target.model.generate(prompts, params, use_tqdm=False)
            for row, prompt, output in zip(batch, prompts, generated):
                parsed = parse_list_prediction(output.outputs[0].text)
                prediction = " # ".join(parsed)
                value = float(qampari_list_metrics(parsed, annotations[row["example_id"]])["f1"])
                row["reruns"].append({"repeat": repeat, "parsed_string_agreement": prediction == row["cached_prediction"], "normalized_set_agreement": normalized_set(prediction) == normalized_set(row["cached_prediction"]), "list_f1_agreement": same_f1(value, row["cached_f1"]), "threshold_label_agreement": (value + 1e-12 >= config["threshold"]) == row["cached_success"], "f1": value, "success": value + 1e-12 >= config["threshold"], "prompt_tokens": len(target.tokenizer.encode(prompt)), "generated_tokens": len(output.outputs[0].token_ids)})
            print(json.dumps({"repeat": repeat, "done": min(start + config["batch_size"], len(masks)), "total": len(masks)}), flush=True)
    summary = aggregate(masks, config)
    summary.update({"complete": True, "selection": selection, "decision": "GO_V17G0B_DP_LABEL_SENSITIVITY_AUDIT" if summary["cache_to_rerun_threshold_flip_masks"] else "NO_DIRECT_THRESHOLD_FLIPS_IN_G0A_SAMPLE", "scope": config["scope"], "internal_used": False, "development_used": False, "confirmation_used": False, "artifacts": {"protocol_sha256": sha256(args.protocol_config), "exact_cache_metadata_sha256": sha256(Path(args.exact_dir) / "metadata.json")}})
    write_jsonl(output_dir / "per_mask_results.jsonl", [{key: value for key, value in row.items() if key not in {"question", "context", "cached_prediction"}} for row in masks])
    write_metadata(output_dir / "summary.json", summary)
    write_metadata(output_dir / "decision.json", {"decision": summary["decision"], "training_authorized": False, "internal_used": False, "development_used": False, "confirmation_used": False})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
