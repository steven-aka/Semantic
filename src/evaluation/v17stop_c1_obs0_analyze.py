"""OOF native-confidence ablations and direct two-stage policy replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.data.schemas import read_jsonl


CFG = Path("configs/v17stop_c1_obs0_native_confidence.json")
OUT = Path("results/v2_rank_then_cut/v17stop_c1_obs0_native_confidence")
LEVELS = (0.6, 0.7, 0.8, 0.9, 0.95)


def fold(qid: str, n: int) -> int:
    return int(hashlib.sha256(("C1-OBS0-FOLD|" + qid).encode()).hexdigest(), 16) % n


def aset(prediction: str) -> set[str]:
    return {x.strip().lower() for x in prediction.split("#") if x.strip()}


def fit_predict(train_x, train_y, test_x):
    torch.set_num_threads(1)
    train_x = np.asarray(train_x, dtype=np.float32)
    test_x = np.asarray(test_x, dtype=np.float32)
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = torch.tensor((train_x - mean) / std)
    y = torch.tensor(np.asarray(train_y, dtype=np.float32))
    xt = torch.tensor((test_x - mean) / std)
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=0.05)
    for _ in range(500):
        logits = x @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + 1e-4 * weight.square().sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(xt @ weight + bias).numpy()


def main() -> None:
    cfg = json.loads(CFG.read_text())
    traces = {(r["example_id"], int(r["depth"])): r for r in read_jsonl(OUT / "traces.jsonl")}
    qids = sorted({q for q, _ in traces})
    assert len(qids) == cfg["sample_size"] and len(traces) == len(qids) * len(cfg["depths"])

    examples = []
    for qid in qids:
        for tau in LEVELS[:-1]:
            depth = int(cfg["early_depths"][str(tau)])
            cur = traces[(qid, depth)]
            previous_depth = {6: None, 7: 6, 9: 7}[depth]
            prev = traces[(qid, previous_depth)] if previous_depth else None
            cur_set, prev_set = aset(cur["prediction"]), aset(prev["prediction"]) if prev else set()
            union = cur_set | prev_set
            jaccard = len(cur_set & prev_set) / len(union) if union else 1.0
            added = len(cur_set - prev_set)
            dropped = len(prev_set - cur_set)
            output_features = [tau, depth / 10, cur["answer_count"], cur["generated_tokens"],
                               len(cur_set), jaccard, added, dropped]
            native_features = [tau, depth / 10, cur["mean_chosen_logprob"], cur["p10_chosen_logprob"],
                               cur["min_chosen_logprob"], cur["chosen_token_logprobs"][-1],
                               cur["mean_top2_margin"], cur["mean_top2_partial_entropy"]]
            examples.append({"example_id": qid, "tau": tau, "depth": depth,
                             "label": float(cur["f1"]) + 1e-12 >= tau,
                             "output": output_features, "native": native_features})

    feature_sets = {
        "B0_output_only": lambda x: x["output"],
        "B1_native_only": lambda x: x["native"],
        "B2_combined": lambda x: x["output"] + x["native"][2:]
    }
    nfold = cfg["oof_folds"]
    scores = {name: {} for name in feature_sets}
    fold_stats = []
    for held in range(nfold):
        train = [x for x in examples if fold(x["example_id"], nfold) != held]
        test = [x for x in examples if fold(x["example_id"], nfold) == held]
        fold_stats.append({"fold": held, "train_queries": len({x['example_id'] for x in train}),
                           "test_queries": len({x['example_id'] for x in test}),
                           "train_examples": len(train), "test_examples": len(test),
                           "train_positive": sum(x["label"] for x in train),
                           "test_positive": sum(x["label"] for x in test)})
        for name, fn in feature_sets.items():
            probs = fit_predict([fn(x) for x in train], [x["label"] for x in train], [fn(x) for x in test])
            for x, p in zip(test, probs):
                scores[name][(x["example_id"], x["tau"])] = float(p)

    def policy_metrics(name: str, threshold: float):
        per_level = {}
        complete = 0
        total_context = total_compute = total_calls = 0
        for qid in qids:
            q_success = []
            for tau in LEVELS:
                early_depth = int(cfg["early_depths"][str(tau)])
                early = traces[(qid, early_depth)]
                late = traces[(qid, 10)]
                if tau == 0.95:
                    stop = True
                else:
                    stop = scores[name][(qid, tau)] >= threshold
                selected = early if stop else late
                ok = float(selected["f1"]) + 1e-12 >= tau
                q_success.append(ok)
                d = per_level.setdefault(str(tau), {"success": 0, "stop_early": 0, "eligible": 0})
                d["eligible"] += 1
                d["success"] += ok
                d["stop_early"] += stop
                total_context += selected["context_tokens"]
                total_compute += early["prompt_tokens"] + early["generated_tokens"]
                total_calls += 1
                if not stop:
                    total_compute += late["prompt_tokens"] + late["generated_tokens"]
                    total_calls += 1
            complete += all(q_success)
        return {"per_level": per_level, "complete": complete,
                "mean_final_context_tokens_per_query": total_context / len(qids),
                "mean_target_compute_tokens_per_query": total_compute / len(qids),
                "mean_calls_per_request": total_calls / (len(qids) * len(LEVELS))}

    def fixed(depths):
        per_level, complete = {}, 0
        tc = comp = 0
        for qid in qids:
            qs = []
            for tau, depth in zip(LEVELS, depths):
                r = traces[(qid, depth)]
                ok = float(r["f1"]) + 1e-12 >= tau
                per_level[str(tau)] = per_level.get(str(tau), 0) + ok
                qs.append(ok); tc += r["context_tokens"]; comp += r["prompt_tokens"] + r["generated_tokens"]
            complete += all(qs)
        return {"per_level_success": per_level, "complete": complete,
                "mean_final_context_tokens_per_query": tc / len(qids),
                "mean_target_compute_tokens_per_query": comp / len(qids)}

    aggressive = fixed([6, 7, 7, 9, 10])
    depth10 = fixed([10, 10, 10, 10, 10])
    # Exact traced-cohort oracle under the same two-stage contract.
    oracle_context = oracle_compute = oracle_calls = 0
    oracle_level = {str(x): 0 for x in LEVELS}; oracle_complete = 0
    for qid in qids:
        qs = []
        for tau in LEVELS:
            ed = int(cfg["early_depths"][str(tau)]); early = traces[(qid, ed)]; late = traces[(qid, 10)]
            early_ok = float(early["f1"]) + 1e-12 >= tau
            selected = early if early_ok else late
            ok = float(selected["f1"]) + 1e-12 >= tau
            qs.append(ok); oracle_level[str(tau)] += ok; oracle_context += selected["context_tokens"]
            oracle_compute += early["prompt_tokens"] + early["generated_tokens"]; oracle_calls += 1
            if not early_ok and ed != 10:
                oracle_compute += late["prompt_tokens"] + late["generated_tokens"]; oracle_calls += 1
        oracle_complete += all(qs)
    oracle = {"per_level_success": oracle_level, "complete": oracle_complete,
              "mean_final_context_tokens_per_query": oracle_context / len(qids),
              "mean_target_compute_tokens_per_query": oracle_compute / len(qids),
              "mean_calls_per_request": oracle_calls / (len(qids) * len(LEVELS))}

    curves = {}
    quality_tol = cfg["quality_noninferiority_pp"] / 100 * len(qids)
    oracle_saving = depth10["mean_final_context_tokens_per_query"] - oracle["mean_final_context_tokens_per_query"]
    for name in feature_sets:
        curves[name] = []
        for threshold in cfg["fixed_stop_probability_thresholds"]:
            m = policy_metrics(name, threshold)
            quality_pass = all(m["per_level"][str(t)]["success"] >= depth10["per_level_success"][str(t)] - quality_tol for t in LEVELS)
            quality_pass &= m["complete"] >= depth10["complete"] - quality_tol
            captured = (depth10["mean_final_context_tokens_per_query"] - m["mean_final_context_tokens_per_query"]) / oracle_saving
            compute_gain = depth10["mean_target_compute_tokens_per_query"] - m["mean_target_compute_tokens_per_query"]
            m.update({"threshold": threshold, "quality_gate_pass": bool(quality_pass),
                      "oracle_context_saving_capture": captured,
                      "target_compute_tokens_saved_vs_depth10": compute_gain,
                      "go": bool(quality_pass and captured >= cfg["minimum_oracle_context_saving_capture"] and compute_gain > 0)})
            curves[name].append(m)

    b0_go = any(x["go"] for x in curves["B0_output_only"])
    b1_go = any(x["go"] for x in curves["B1_native_only"])
    b2_go = any(x["go"] for x in curves["B2_combined"])
    b1_quality_context = [x for x in curves["B1_native_only"] if x["quality_gate_pass"] and x["oracle_context_saving_capture"] >= cfg["minimum_oracle_context_saving_capture"]]
    native_increment = b2_go and (not b0_go or max(x["oracle_context_saving_capture"] for x in curves["B2_combined"] if x["quality_gate_pass"] and x["target_compute_tokens_saved_vs_depth10"] > 0) > max([x["oracle_context_saving_capture"] for x in curves["B0_output_only"] if x["quality_gate_pass"] and x["target_compute_tokens_saved_vs_depth10"] > 0] or [-1]))
    decision = "GO_C1_NATIVE_CONFIDENCE_DESIGN" if b2_go and native_increment else "STOP_C1_NATIVE_CONFIDENCE"
    summary = {"protocol": cfg["protocol"], "queries": len(qids), "decision_examples": len(examples),
               "folds": fold_stats, "baselines": {"aggressive": aggressive, "depth10": depth10, "two_stage_oracle": oracle},
               "fixed_threshold_curves": curves,
               "gates": {"B0_output_go": b0_go, "B1_native_go": b1_go, "B2_combined_go": b2_go,
                         "native_increment_over_B0": native_increment},
               "decision": decision,
               "diagnosis": {
                   "native_confidence_has_quality_context_signal": bool(b1_quality_context),
                   "best_native_quality_context_point": max(b1_quality_context, key=lambda x: x["oracle_context_saving_capture"]) if b1_quality_context else None,
                   "cost_is_binding_when_native_signal_qualifies": bool(b1_quality_context and all(x["target_compute_tokens_saved_vs_depth10"] <= 0 for x in b1_quality_context)),
                   "next_action": "GO_C2_COST_PREFIX_STATE_REUSE_PREFLIGHT" if b1_quality_context and all(x["target_compute_tokens_saved_vs_depth10"] <= 0 for x in b1_quality_context) else "NO_COST_CONTRACT_CHANGE"
               },
               "limitations": "Train-side query-grouped OOF on a traced design cohort. Threshold grid was frozen before analysis. No sealed split was read.",
               "sealed_sets_read": False}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (OUT / "oof_scores.jsonl").open("w") as handle:
        for x in examples:
            row = {k: x[k] for k in ("example_id", "tau", "depth", "label")}
            row["fold"] = fold(x["example_id"], nfold)
            row.update({name: scores[name][(x["example_id"], x["tau"])] for name in feature_sets})
            handle.write(json.dumps(row) + "\n")
    best = summary["diagnosis"]["best_native_quality_context_point"]
    report = f"""# C1-OBS0 native-confidence two-stage stopping

## Cohort and contract

We collected 1,024 same-call traced states (`d6/d7/d9/d10`) for 256 naturally selected train-side queries, excluding all 32 prior trace-preflight IDs. Answers, F1 labels, token log-probabilities and real prompt/generation costs came from the same traced execution path. These queries are design-exposed; no sealed split was read.

## Baselines and oracle

| Policy | 0.60/0.70/0.80/0.90/0.95 | Complete | Final context/query | Target compute/query |
|---|---|---:|---:|---:|
| Aggressive fixed | {'/'.join(str(aggressive['per_level_success'][str(t)]) for t in LEVELS)} | {aggressive['complete']} | {aggressive['mean_final_context_tokens_per_query']:.2f} | {aggressive['mean_target_compute_tokens_per_query']:.2f} |
| Depth10 fixed | {'/'.join(str(depth10['per_level_success'][str(t)]) for t in LEVELS)} | {depth10['complete']} | {depth10['mean_final_context_tokens_per_query']:.2f} | {depth10['mean_target_compute_tokens_per_query']:.2f} |
| Two-stage oracle | {'/'.join(str(oracle['per_level_success'][str(t)]) for t in LEVELS)} | {oracle['complete']} | {oracle['mean_final_context_tokens_per_query']:.2f} | {oracle['mean_target_compute_tokens_per_query']:.2f} |

## OOF result

No output-only, native-only, or combined point passes all quality, context-capture, and real Target-compute gates. The formal decision is `{decision}`.

The informative diagnostic is native-only threshold `{best['threshold'] if best else 'N/A'}`: it yields five-anchor successes `{('/'.join(str(best['per_level'][str(t)]['success']) for t in LEVELS)) if best else 'N/A'}`, Complete `{best['complete'] if best else 'N/A'}`, captures `{best['oracle_context_saving_capture']:.1%}` of oracle context saving, and uses `{best['mean_final_context_tokens_per_query']:.2f}` final-context tokens/query. It nevertheless uses `{best['mean_target_compute_tokens_per_query']:.2f}` Target tokens/query, `{(-best['target_compute_tokens_saved_vs_depth10']):.2f}` more than direct depth10.

Thus token-level confidence contains some quality/context signal that B0 lacks, but repeated full inference prevents deployment Pareto improvement. B2 did not pass, so this is not a successful stopper claim. It authorizes only `C2-COST_PREFIX_STATE_REUSE_PREFLIGHT`: verify whether the actual prompt/KV execution can reuse prefix state without changing text, answers, or costs before implementing any serving optimization.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
