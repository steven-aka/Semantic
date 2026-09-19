from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.evaluation.v17d1b_frozen_boundary_separability import stable_fold
from src.evaluation.v17e0_frozen_information_bottleneck import evaluate_scores, normalize, paired_bootstrap, train_probe
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import state_to_mask
from src.training.train_v17b1_multi_anchor_viability import read_gzip


WORD = re.compile(r"[\w]+", re.UNICODE)
HASH_DIM = 64


def words(value: str) -> list[str]:
    return WORD.findall(value.casefold())


def behavior_features(current: str, candidate: str, question: str) -> list[float]:
    current_words, candidate_words = words(current), words(candidate)
    current_set, candidate_set, question_set = set(current_words), set(candidate_words), set(words(question))
    hashed = [0.0] * HASH_DIM
    for token in candidate_words:
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % HASH_DIM
        hashed[index] += 1.0 if digest[4] & 1 else -1.0
    denominator = math.sqrt(max(1, len(candidate_words)))
    hashed = [value / denominator for value in hashed]
    overlap = len(current_set & candidate_set) / max(1, len(current_set | candidate_set))
    question_overlap = len(candidate_set & question_set) / max(1, len(candidate_set))
    return hashed + [
        math.log1p(len(candidate_words)), math.log1p(len(candidate)), float(any(char.isdigit() for char in candidate)),
        question_overlap, float(candidate.strip().casefold() != current.strip().casefold()), overlap,
        math.log1p(len(candidate_words)) - math.log1p(len(current_words)),
    ]


def build_behavior_cache(records: list[dict[str, Any]], states: list[dict[str, Any]], sources: dict[str, dict[str, Any]], exact_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(records) != len(states): raise ValueError("boundary state alignment mismatch")
    feature_count = max(max(row["positive"] + row["negative"]) for row in states) + 1
    features = torch.empty((feature_count, HASH_DIM + 7), dtype=torch.float16)
    grouped = defaultdict(list)
    for record, state in zip(records, states):
        if record["example_id"] != state["example_id"] or record["history"] != state["history"]:
            raise ValueError("record/feature state identity mismatch")
        grouped[record["example_id"]].append((record, state))
    current_calls = all_candidate_calls = current_context_tokens = all_candidate_context_tokens = 0
    unique_masks = set()
    unique_context_tokens = 0
    for number, (example_id, items) in enumerate(grouped.items(), 1):
        exact = {state_to_mask(row.state): row for row in read_jsonl(exact_dir / f"{example_id}.jsonl", ExactSearchResult)}
        question = sources[example_id]["question"]
        local_masks = set()
        for record, state in items:
            mask = int(record["selected_mask"]); current = exact[mask]
            current_calls += 1; current_context_tokens += current.tokens; local_masks.add(mask)
            action_rows = sorted(record["actions"], key=lambda row: row["packet"])
            indices = sorted(state["positive"] + state["negative"])
            if len(action_rows) != len(indices): raise ValueError("action count alignment mismatch")
            for action, index in zip(action_rows, indices):
                successor_mask = mask | (1 << int(action["packet"]))
                candidate = exact[successor_mask]
                features[index] = torch.tensor(behavior_features(current.prediction, candidate.prediction, question), dtype=torch.float16)
                all_candidate_calls += 1; all_candidate_context_tokens += candidate.tokens; local_masks.add(successor_mask)
        unique_masks.update((example_id, mask) for mask in local_masks)
        unique_context_tokens += sum(exact[mask].tokens for mask in local_masks)
        if number % 100 == 0:
            print(json.dumps({"behavior_examples": number, "total": len(grouped)}), flush=True)
    if not torch.isfinite(features).all(): raise ValueError("non-finite behavior feature")
    return features, {
        "examples": len(grouped), "boundary_states": len(states),
        "current_state_calls_without_memoization": current_calls,
        "all_candidate_calls_without_memoization": all_candidate_calls,
        "current_plus_all_candidate_calls_without_memoization": current_calls + all_candidate_calls,
        "unique_mask_calls_with_per_query_memoization": len(unique_masks),
        "current_context_tokens_lower_bound": current_context_tokens,
        "all_candidate_context_tokens_lower_bound": all_candidate_context_tokens,
        "unique_mask_context_tokens_lower_bound": unique_context_tokens,
        "cost_scope": "context text tokens from exact cache only; excludes prompt/question and generated output tokens, latency and KV reuse",
    }


def shuffle_within_state(features: torch.Tensor, states: list[dict[str, Any]], seed: int) -> torch.Tensor:
    result = features.clone()
    for state in states:
        indices = sorted(state["positive"] + state["negative"])
        if len(indices) < 2: continue
        key = f"{seed}|{state['example_id']}|{state['history']}"
        shift = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") % (len(indices) - 1) + 1
        rotated = indices[shift:] + indices[:shift]
        result[indices] = features[rotated]
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="V17-F2 cached target-behavior observability audit")
    p.add_argument("--protocol-config", required=True); p.add_argument("--training-artifact", required=True)
    p.add_argument("--boundary-features", required=True); p.add_argument("--train-data", required=True)
    p.add_argument("--exact-dir", required=True); p.add_argument("--output-dir", required=True)
    a = p.parse_args(); config = json.loads(Path(a.protocol_config).read_text())
    if config["status"] != "FROZEN_APPROVED_TO_RUN": raise ValueError("protocol not approved")
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(a.boundary_features, map_location="cpu", weights_only=False)
    local, states = bundle["features"], bundle["states"]
    records = []
    for row in read_gzip(a.training_artifact):
        values = [action["future_viability"][3] for action in row["actions"] if action["future_viability"][3] is not None]
        if 1 in values and 0 in values: records.append(row)
    sources = {row["example_id"]: row for row in read_jsonl(a.train_data)}
    behavior, cost = build_behavior_cache(records, states, sources, Path(a.exact_dir))
    shuffled = shuffle_within_state(behavior, states, config["probe"]["seed"])
    taps = {
        "local": local,
        "candidate_behavior": behavior,
        "local_plus_candidate_behavior": torch.cat((local, behavior), -1),
        "local_plus_shuffled_candidate_behavior": torch.cat((local, shuffled), -1),
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fold_assignment = [stable_fold(row["example_id"], config["probe"]["folds"]) for row in states]
    fold_rows = []; query_results = {}
    for tap, raw in taps.items():
        by_query = {}
        for fold in range(config["probe"]["folds"]):
            train_idx = [i for i, value in enumerate(fold_assignment) if value != fold]
            test_idx = [i for i, value in enumerate(fold_assignment) if value == fold]
            train_actions = sorted({action for i in train_idx for action in states[i]["positive"] + states[i]["negative"]})
            normalized = normalize(raw[train_actions], raw)
            probe = train_probe(normalized, states, train_idx, config, device, config["probe"]["seed"] + fold)
            with torch.no_grad():
                score = torch.cat([probe(normalized[start:start+4096].to(device=device, dtype=torch.float32)).cpu() for start in range(0, len(normalized), 4096)])
            metrics, queries = evaluate_scores(score, states, test_idx)
            fold_rows.append({"tap": tap, "fold": fold, "dimension": raw.shape[1], **metrics})
            by_query.update(queries)
            print(json.dumps(fold_rows[-1]), flush=True)
        query_results[tap] = by_query
    aggregate = {}
    for tap in taps:
        rows = [row for row in fold_rows if row["tap"] == tap]
        aggregate[tap] = {
            "macro_state_accuracy": mean(row["macro_state_accuracy"] for row in rows),
            "macro_query_accuracy": mean(row["macro_query_accuracy"] for row in rows),
            "fold_macro_state": [row["macro_state_accuracy"] for row in rows],
            "paired_vs_local": paired_bootstrap(query_results[tap], query_results["local"], config["identifiability_gate"]["paired_query_bootstrap_replicates"], config["probe"]["seed"]),
        }
    value = aggregate["local_plus_candidate_behavior"]
    gate = config["identifiability_gate"]
    passed = (value["macro_state_accuracy"] >= gate["macro_state_min"] and value["macro_query_accuracy"] >= gate["macro_query_min"] and min(value["fold_macro_state"]) >= gate["all_folds_macro_state_min"] and value["paired_vs_local"]["lower_95"] > gate["paired_query_bootstrap_lower_vs_local_min"] and value["macro_query_accuracy"] - aggregate["local_plus_shuffled_candidate_behavior"]["macro_query_accuracy"] >= gate["shuffled_control_margin_min"])
    # The all-candidate observability condition is an information ceiling; a controller
    # still needs a separately frozen bounded-call protocol and full cost accounting.
    decision = "GO_V17F3_BOUNDED_BEHAVIOR_PROBE_PROTOCOL_DESIGN" if passed else "STOP_V17F2_CACHED_ANSWER_OBSERVABILITY"
    write_jsonl(out / "fold_results.jsonl", fold_rows)
    summary = {"complete": True, "aggregates": aggregate, "cost": cost, "identifiability_gate_passed": passed, "decision": decision, "scope_limit": "Only cached parsed-answer behavior is tested; log-probabilities, hidden states and bounded rollouts remain untested.", "training_authorized": False, "internal_used": False, "development_used": False, "confirmation_used": False, "artifacts": {"protocol_sha256": sha256(a.protocol_config), "training_artifact_sha256": sha256(a.training_artifact), "boundary_features_sha256": sha256(a.boundary_features)}}
    write_metadata(out / "summary.json", summary)
    write_metadata(out / "decision.json", {"decision": decision, "all_candidate_behavior_is_upper_bound_only": True, "controller_training_authorized": False, "internal_used": False, "development_used": False, "confirmation_used": False})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
