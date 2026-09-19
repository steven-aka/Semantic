from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.qampari_metrics import parse_list_prediction
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.evaluation.v17e0_frozen_information_bottleneck import evaluate_scores, normalize, paired_bootstrap, train_probe
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.target.qampari_runner import QampariTargetRunner
from src.training.train_v17b1_multi_anchor_viability import read_gzip


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def selected_states(records: list[dict[str, Any]], states: list[dict[str, Any]], limit: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(records) != len(states):
        raise ValueError("boundary artifact alignment mismatch")
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record, state in zip(records, states):
        if (record["example_id"], record["history"]) != (state["example_id"], state["history"]):
            raise ValueError("state order mismatch")
        grouped[record["example_id"]].append((record, state))
    queries = sorted(grouped, key=lambda query: digest("F3-query|" + query))[:limit]
    return [min(grouped[query], key=lambda pair: (pair[0]["provenance"] != "deployed", digest("F3-state|" + query + "|" + repr(pair[0]["history"])))) for query in queries]


def output_signal(output: Any) -> tuple[float, float, float, float]:
    chosen = []
    margins = []
    for token_id, top in zip(output.token_ids, output.logprobs or []):
        if token_id not in top:
            raise ValueError("chosen generated token missing from returned logprobs")
        chosen.append(float(top[token_id].logprob))
        ranked = sorted((float(item.logprob) for item in top.values()), reverse=True)
        if len(ranked) >= 2:
            margins.append(ranked[0] - ranked[1])
    if not chosen:
        raise ValueError("empty target generation or logprobs")
    return mean(chosen), min(chosen), mean(margins) if margins else 0.0, float(len(chosen))


def build_tasks(selected: list[tuple[dict[str, Any], dict[str, Any]]], sources: dict[str, dict[str, Any]], exact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    tasks = []
    grouped = {}
    for record, state in selected:
        query = record["example_id"]
        source = sources[query]
        exact = {state_to_mask(row.state): row for row in read_jsonl(exact_dir / f"{query}.jsonl", ExactSearchResult)}
        mask = int(record["selected_mask"])
        actions = sorted(record["actions"], key=lambda item: item["packet"])
        indices = sorted(state["positive"] + state["negative"])
        if len(actions) != len(indices):
            raise ValueError("action feature alignment mismatch")
        rows = []
        for kind, successor, index in [("current", mask, None), *(("candidate", mask | (1 << int(action["packet"])), index) for action, index in zip(actions, indices))]:
            context = "\n\n".join(text.strip() for packet, text in enumerate(source["packet_texts"]) if successor & (1 << packet))
            rows.append({"example_id": query, "kind": kind, "mask": successor, "action_index": index, "question": source["question"], "context": context, "cached_prediction": exact[successor].prediction})
        grouped[query] = rows
        tasks.extend(rows)
    return tasks, grouped


def collect(target: QampariTargetRunner, tasks: list[dict[str, Any]], batch_size: int, max_new_tokens: int, top_k: int) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=top_k)
    results = []
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start:start + batch_size]
        prompts = target._chat_prompts([row["question"] for row in batch], [row["context"] for row in batch])
        outputs = target.model.generate(prompts, params, use_tqdm=False)
        for row, output, prompt in zip(batch, outputs, prompts):
            generated = output.outputs[0]
            prediction = " # ".join(parse_list_prediction(generated.text))
            results.append({"example_id": row["example_id"], "kind": row["kind"], "mask": row["mask"], "action_index": row["action_index"], "parsed_agreement": prediction == row["cached_prediction"], "signal": output_signal(generated), "prompt_tokens": len(target.tokenizer.encode(prompt)), "generated_tokens": len(generated.token_ids)})
        print(json.dumps({"generated": len(results), "total": len(tasks)}), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--training-artifact", required=True)
    parser.add_argument("--boundary-features", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.protocol_config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_RUN":
        raise ValueError("unfrozen protocol")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.boundary_features, map_location="cpu", weights_only=False)
    states = bundle["states"]
    records = []
    for row in read_gzip(args.training_artifact):
        labels = [action["future_viability"][3] for action in row["actions"] if action["future_viability"][3] is not None]
        if 1 in labels and 0 in labels:
            records.append(row)
    selected = selected_states(records, states, config["query_count"])
    if len(selected) != config["query_count"]:
        raise ValueError("insufficient distinct boundary queries")
    sources = {row["example_id"]: row for row in read_jsonl(args.train_data)}
    tasks, grouped = build_tasks(selected, sources, Path(args.exact_dir))
    manifest = [{"example_id": record["example_id"], "history": record["history"], "provenance": record["provenance"], "selected_mask": record["selected_mask"], "candidate_count": len(rows) - 1} for (record, _), rows in zip(selected, grouped.values())]
    write_jsonl(output_dir / "selection_manifest.jsonl", manifest)
    target = QampariTargetRunner(config["target_model"], backend="vllm", max_new_tokens=config["max_new_tokens"], gpu_memory_utilization=config["gpu_memory_utilization"], max_model_len=config["max_model_len"])
    preflight_query_ids = {record["example_id"] for record, _ in selected[:config["preflight_queries"]]}
    preflight_tasks = [row for row in tasks if row["example_id"] in preflight_query_ids]
    preflight = collect(target, preflight_tasks, config["batch_size"], config["max_new_tokens"], config["logprobs_top_k"])
    agreement = mean(float(row["parsed_agreement"]) for row in preflight)
    if agreement < config["preflight_parsed_agreement_min"]:
        summary = {"complete": True, "stage": "preflight", "parsed_agreement": agreement, "preflight_calls": len(preflight), "decision": "STOP_V17F3_TARGET_CACHE_REPRODUCTION", "internal_used": False, "development_used": False, "confirmation_used": False}
        write_metadata(output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return
    remaining = [row for row in tasks if row["example_id"] not in preflight_query_ids]
    results = preflight + collect(target, remaining, config["batch_size"], config["max_new_tokens"], config["logprobs_top_k"])
    by_query = defaultdict(list)
    for result in results:
        by_query[result["example_id"]].append(result)
    local = bundle["features"]
    action_indices = sorted({index for _, state in selected for index in state["positive"] + state["negative"]})
    position = {index: place for place, index in enumerate(action_indices)}
    logprob = torch.empty((len(action_indices), 5), dtype=torch.float16)
    compact_states = []
    for _, state in selected:
        query = state["example_id"]
        current = next(row for row in by_query[query] if row["kind"] == "current")
        for row in by_query[query]:
            if row["kind"] == "candidate":
                value = [*row["signal"], row["signal"][0] - current["signal"][0]]
                logprob[position[row["action_index"]]] = torch.tensor(value, dtype=torch.float16)
        compact_states.append({**state, "positive": [position[index] for index in state["positive"]], "negative": [position[index] for index in state["negative"]]})
    local = local[action_indices]
    features = {"local": local, "logprob": logprob, "local_plus_logprob": torch.cat((local, logprob), dim=-1)}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assignment = [stable_fold(row["example_id"], config["probe"]["folds"]) for row in compact_states]
    fold_rows = []
    query_results = {}
    for name, raw in features.items():
        queries = {}
        for fold in range(config["probe"]["folds"]):
            train_indices = [i for i, group in enumerate(assignment) if group != fold]
            test_indices = [i for i, group in enumerate(assignment) if group == fold]
            train_actions = sorted({action for i in train_indices for action in compact_states[i]["positive"] + compact_states[i]["negative"]})
            normalized = normalize(raw[train_actions], raw)
            probe = train_probe(normalized, compact_states, train_indices, config, device, config["probe"]["seed"] + fold)
            with torch.no_grad():
                score = probe(normalized.to(device=device, dtype=torch.float32)).cpu()
            metrics, query_scores = evaluate_scores(score, compact_states, test_indices)
            fold_rows.append({"signal": name, "fold": fold, **metrics})
            queries.update(query_scores)
        query_results[name] = queries
    aggregate = {}
    for name in features:
        rows = [row for row in fold_rows if row["signal"] == name]
        aggregate[name] = {"macro_state_accuracy": mean(row["macro_state_accuracy"] for row in rows), "macro_query_accuracy": mean(row["macro_query_accuracy"] for row in rows), "folds": [row["macro_query_accuracy"] for row in rows], "paired_vs_local": paired_bootstrap(query_results[name], query_results["local"], config["teacher_gate"]["bootstrap_replicates"], config["probe"]["seed"])}
    candidate = aggregate["local_plus_logprob"]
    gate = config["teacher_gate"]
    passed = candidate["macro_state_accuracy"] >= gate["macro_state_min"] and candidate["paired_vs_local"]["mean_query_improvement"] >= gate["paired_query_improvement_min"] and candidate["paired_vs_local"]["lower_95"] > gate["paired_query_bootstrap_lower_min"]
    decision = "GO_V17F3_STUDENT_DISTILLATION_PROTOCOL_DESIGN" if passed else "STOP_V17F3_LOGPROB_PILOT_NO_TEACHER_SIGNAL"
    write_jsonl(output_dir / "fold_results.jsonl", fold_rows)
    cost = {"target_calls": len(results), "prompt_tokens": sum(row["prompt_tokens"] for row in results), "generated_tokens": sum(row["generated_tokens"] for row in results), "parsed_agreement": mean(float(row["parsed_agreement"]) for row in results)}
    summary = {"complete": True, "queries": len(selected), "states": len(compact_states), "actions": len(action_indices), "preflight_agreement": agreement, "aggregates": aggregate, "cost": cost, "teacher_gate_passed": passed, "decision": decision, "student_trained": False, "internal_used": False, "development_used": False, "confirmation_used": False, "artifacts": {"protocol_sha256": sha256(args.protocol_config), "training_artifact_sha256": sha256(args.training_artifact), "boundary_features_sha256": sha256(args.boundary_features)}}
    write_metadata(output_dir / "summary.json", summary)
    write_metadata(output_dir / "decision.json", {"decision": decision, "student_training_authorized": False, "internal_used": False, "development_used": False, "confirmation_used": False})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
