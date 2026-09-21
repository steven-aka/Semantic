"""Grouped train-side audit of costed continuation-value observables."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import normalize_list_answer, parse_cached_list_prediction

ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
OUT = Path("results/v2_rank_then_cut/v17stop_b0_continuation_value_audit")
ARMS = ("depth_count", "answer_selected", "answer_selected_remaining")
CLASSES = ("success_success", "fail_success", "success_fail", "fail_fail")
WORD = re.compile(r"\w+", flags=re.UNICODE)


def unit(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.float()
    return value / value.norm().clamp_min(1e-8)


def hashed_answer(value: str, size: int = 1024) -> np.ndarray:
    words = WORD.findall(value.casefold())
    grams = words + [a + "_" + b for a, b in zip(words, words[1:])]
    result = np.zeros(size, dtype=np.float32)
    for gram in grams:
        digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
        number = int.from_bytes(digest, "little")
        result[number % size] += 1 if number & (1 << 63) else -1
    norm = np.linalg.norm(result)
    if norm:
        result /= norm
    return result


def grounded_fraction(answers: list[str], text: str) -> float:
    if not answers:
        return 0.0
    haystack = normalize_list_answer(text)
    return sum(bool(a and a in haystack) for a in answers) / len(answers)


def features_for(row: dict, data: dict, embeddings: dict, index: int, level: float) -> tuple[np.ndarray, ...]:
    mask = row["mask"]
    packet_texts = data["packet_texts"]
    selected = "\n".join(t for i, t in enumerate(packet_texts) if mask & (1 << i))
    remaining = "\n".join(t for i, t in enumerate(packet_texts) if not mask & (1 << i))
    answers = [normalize_list_answer(x) for x in parse_cached_list_prediction(row["prediction"])]
    count = len(answers)
    depth = row["depth"]
    basic = np.array([*(float(level == x) for x in (.6, .7, .8, .9)),
                      depth / 12, min(count, 20) / 20,
                      row["tokens"] / max(row["full_tokens"], 1),
                      row["generated_tokens"] / 256], dtype=np.float32)
    question = unit(embeddings["questions"][index]).numpy()
    packets = embeddings["packets"][index].float()
    selected_ids = [i for i in range(12) if mask & (1 << i)]
    remaining_ids = [i for i in range(12) if not mask & (1 << i)]
    selected_emb = unit(packets[selected_ids].mean(0)).numpy()
    remaining_emb = unit(packets[remaining_ids].mean(0)).numpy() if remaining_ids else np.zeros_like(question)
    answer_embedding = hashed_answer(row["prediction"])
    selected_extra = np.concatenate((answer_embedding, question, selected_emb,
        question * selected_emb,
        np.array([grounded_fraction(answers, selected)], dtype=np.float32)))
    remaining_extra = np.concatenate((remaining_emb, question * remaining_emb,
        np.array([grounded_fraction(answers, remaining),
                  len(remaining_ids) / 12], dtype=np.float32)))
    return basic, np.concatenate((basic, selected_extra)), np.concatenate((basic, selected_extra, remaining_extra))


def fit_predict(x: torch.Tensor, y: torch.Tensor, folds: list[int],
                query_ids: list[str] | None = None, train_fraction: float = 1.0) -> tuple[np.ndarray, dict]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_probs = np.zeros((len(y), 4), dtype=np.float32)
    train_probabilities = {}
    histories = []
    for heldout in range(4):
        train_idx = torch.tensor([i for i, f in enumerate(folds) if f != heldout and
            (train_fraction == 1.0 or (query_ids is not None and
             int.from_bytes(hashlib.blake2b(query_ids[i].encode(), digest_size=8).digest(), "little")
             / 2**64 < train_fraction))])
        test_idx = torch.tensor([i for i, f in enumerate(folds) if f == heldout])
        train_x, train_y = x[train_idx].to(device), y[train_idx].to(device)
        test_x = x[test_idx].to(device)
        torch.manual_seed(20260920)
        model = torch.nn.Linear(x.shape[1], 4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.05, weight_decay=.01)
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(train_x), train_y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_p = model(train_x).softmax(-1).cpu().numpy()
            test_p = model(test_x).softmax(-1).cpu().numpy()
        all_probs[test_idx.numpy()] = test_p
        train_probabilities[heldout] = (train_idx.numpy(), train_p)
        histories.append({"fold": heldout, "train_cross_entropy": float(loss.item()),
                          "train_rows": len(train_idx), "test_rows": len(test_idx)})
    return all_probs, {"fold_training": histories, "train_probabilities": train_probabilities}


def choose_thresholds(probs: np.ndarray, training: dict, rows: list[dict], fractions: list[float]) -> dict:
    thresholds = {}
    for heldout, (indices, values) in training.items():
        thresholds[heldout] = {}
        for level in (.6, .7, .8, .9):
            risks = sorted(float(values[j, 1]) for j, i in enumerate(indices) if rows[i]["level"] == level)
            if not risks:
                raise AssertionError("missing train anchor")
            thresholds[heldout][str(level)] = {
                str(fraction): (None if fraction == 0 else risks[min(len(risks) - 1,
                    max(0, math.ceil(fraction * len(risks)) - 1))]) for fraction in fractions}
    return thresholds


def evaluate(rows: list[dict], source: dict, chain: dict, probs: np.ndarray, thresholds: dict,
             fractions: list[float]) -> list[dict]:
    out = []
    for fraction in fractions:
        by_query = defaultdict(dict)
        for i, row in enumerate(rows):
            threshold = thresholds[row["fold"]][str(row["level"])][str(fraction)]
            by_query[row["example_id"]][row["level"]] = (threshold is not None and
                float(probs[i, 1]) <= threshold)
        anchor = defaultdict(Counter)
        query_records = []
        for q in sorted(source):
            active = {float(x) for x in source[q]["attainable_levels"]}
            base_context = actual_context = base_compute = actual_compute = 0
            base_complete = actual_complete = True
            extra_calls = 0
            for level, depth in zip((.6, .7, .8, .9, .95), (6, 7, 8, 9, 10)):
                if level not in active:
                    continue
                base = chain[q][10]
                probe = chain[q][depth]
                enabled = fraction > 0 and depth < 10
                stop = enabled and by_query[q][level]
                chosen = probe if stop else base
                b_ok = base["f1"] + 1e-6 >= level
                a_ok = chosen["f1"] + 1e-6 >= level
                base_complete &= b_ok
                actual_complete &= a_ok
                anchor[str(level)]["baseline_success"] += b_ok
                anchor[str(level)]["policy_success"] += a_ok
                anchor[str(level)]["repairs"] += not b_ok and a_ok
                anchor[str(level)]["breaks"] += b_ok and not a_ok
                anchor[str(level)]["early_exits"] += stop
                base_context += base["tokens"]
                actual_context += chosen["tokens"]
                base_compute += base["prompt_tokens"] + base["generated_tokens"]
                if enabled:
                    actual_compute += probe["prompt_tokens"] + probe["generated_tokens"]
                    if not stop:
                        actual_compute += base["prompt_tokens"] + base["generated_tokens"]
                        extra_calls += 1
                else:
                    actual_compute += base["prompt_tokens"] + base["generated_tokens"]
            query_records.append((base_context, actual_context, base_compute, actual_compute,
                                  base_complete, actual_complete, extra_calls))
        out.append({"early_fraction_train_quantile": fraction,
                    "per_anchor": {k: dict(v) for k, v in anchor.items()},
                    "baseline_complete": sum(x[4] for x in query_records),
                    "policy_complete": sum(x[5] for x in query_records),
                    "mean_final_context_tokens": mean(x[1] for x in query_records),
                    "mean_context_tokens_saved": mean(x[0] - x[1] for x in query_records),
                    "mean_target_compute_tokens": mean(x[3] for x in query_records),
                    "mean_target_compute_tokens_saved": mean(x[2] - x[3] for x in query_records),
                    "mean_extra_target_calls": mean(x[6] for x in query_records)})
    return out


def main() -> None:
    torch.set_num_threads(1)
    cfg = json.loads(Path("configs/v17stop_b0_continuation_value_audit.json").read_text())
    if cfg["status"] != "FROZEN_TRAIN_SIDE_DIAGNOSTIC":
        raise AssertionError("unfrozen protocol")
    source = {r["example_id"]: r for r in read_jsonl(ROOT / "sel_c0_train_top10_candidates.jsonl")
              if fold(r["example_id"]) != 4}
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in source}
    embeddings = torch.load(ROOT / "v8_train_embeddings.pt", map_location="cpu", weights_only=True)
    index = {q: i for i, q in enumerate(embeddings["example_ids"]) if q in source}
    chain = defaultdict(dict)
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            if row["example_id"] in source:
                chain[row["example_id"]][row["depth"]] = row
    if any(len(x) != 1421 for x in (source, data, index, chain)) or any(len(v) != 12 for v in chain.values()):
        raise AssertionError("train-side join mismatch")
    rows, feature_parts = [], {arm: [] for arm in ARMS}
    transition = Counter()
    for q in sorted(source):
        for level, depth in zip(cfg["levels"], cfg["probe_depths"]):
            probe = chain[q][depth]
            base = chain[q][10]
            a = probe["f1"] + 1e-6 >= level
            b = base["f1"] + 1e-6 >= level
            cls = (0 if a and b else 1 if not a and b else 2 if a and not b else 3)
            transition[f"{level}:{CLASSES[cls]}"] += 1
            rows.append({"example_id": q, "fold": fold(q), "level": level,
                         "depth": depth, "transition": CLASSES[cls], "class": cls})
            values = features_for(probe, data[q], embeddings, index[q], level)
            for arm, value in zip(ARMS, values):
                feature_parts[arm].append(value)
    if len(rows) != 1421 * 4 or set(r["fold"] for r in rows) != {0, 1, 2, 3}:
        raise AssertionError("query-grouped population mismatch")
    folds = [r["fold"] for r in rows]
    y = torch.tensor([r["class"] for r in rows], dtype=torch.long)
    report = {"protocol": cfg["protocol"], "queries": 1421, "decision_rows": len(rows),
              "transition_counts": dict(transition), "feature_arms": {},
              "limitations": "Linear cheap lexical/embedding probe only. It does not exhaust semantic observers. Existing V8 embeddings have a separate frozen-encoder cost; critic inference and latency are not included in replay.",
              "new_target_calls": 0, "sealed_outcome_sets_read": False}
    OUT.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        x = torch.from_numpy(np.stack(feature_parts[arm]))
        probs, training = fit_predict(x, y, folds)
        oof_nll = float(-np.log(np.clip(probs[np.arange(len(rows)), y.numpy()], 1e-12, 1)).mean())
        oof_accuracy = float((probs.argmax(1) == y.numpy()).mean())
        thresholds = choose_thresholds(probs, training["train_probabilities"], rows,
                                       cfg["risk_curve_fractions"])
        curve = evaluate(rows, source, chain, probs, thresholds, cfg["risk_curve_fractions"])
        report["feature_arms"][arm] = {"input_dim": x.shape[1],
            "oof_multiclass_nll": oof_nll, "oof_multiclass_accuracy_diagnostic": oof_accuracy,
            "fold_training": training["fold_training"],
            "train_fold_thresholds": thresholds, "risk_curve": curve}
        if arm != "depth_count":
            # Post-primary descriptive diagnostic, not a model-selection gate.
            learning_curve = []
            query_ids = [row["example_id"] for row in rows]
            for fraction in (.25, .5, .75):
                subset_probs, subset_training = fit_predict(x, y, folds, query_ids, fraction)
                learning_curve.append({"train_query_fraction": fraction,
                    "mean_train_rows": mean(x["train_rows"] for x in subset_training["fold_training"]),
                    "oof_multiclass_nll": float(-np.log(np.clip(
                        subset_probs[np.arange(len(rows)), y.numpy()], 1e-12, 1)).mean())})
            learning_curve.append({"train_query_fraction": 1.0,
                "mean_train_rows": mean(x["train_rows"] for x in training["fold_training"]),
                "oof_multiclass_nll": oof_nll})
            report["feature_arms"][arm]["post_result_descriptive_learning_curve"] = learning_curve
        with (OUT / f"oof_{arm}.jsonl").open("w") as handle:
            for row, p in zip(rows, probs):
                handle.write(json.dumps({**row, "probabilities": p.tolist()}) + "\n")
        print(json.dumps({"arm": arm, "done": True}), flush=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"transition_counts": dict(transition), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
