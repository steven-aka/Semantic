"""Read-only evaluation/decision alignment audit on current-contract train data.

No model fitting, Target inference, gold-as-feature, or sealed-set access.
Alternative thresholds are sensitivity diagnostics, not new benchmark gates.
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluation.qampari_metrics import normalize_list_answer, qampari_list_metrics


ROOT = Path("results/v2_rank_then_cut")
OUT = ROOT / "v17eval_a0_alignment_audit"
EPS = 1e-6
SENSITIVITY_THRESHOLDS = (0.85, 0.90)


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def cached_answers(prediction):
    """Read the stored, already-parsed ` # ` list without parsing raw LM text again."""
    seen, answers = set(), []
    for part in re.split(r"\s*#\s*", prediction):
        value = part.strip()
        key = normalize_list_answer(value)
        if key and key not in seen:
            seen.add(key)
            answers.append(value)
    return answers


def matched(prediction, atoms):
    alias = {}
    for i, atom in enumerate(atoms):
        for value in (atom["answer_text"], *atom.get("aliases", [])):
            alias.setdefault(normalize_list_answer(str(value)), i)
    return {alias[key] for answer in cached_answers(prediction)
            if (key := normalize_list_answer(answer)) in alias}


def hit(score, level):
    return score + EPS >= level


def main():
    records = list(read(ROOT / "v17packet_r1_label_scale512/per_query.jsonl"))
    ids = {r["example_id"] for r in records}
    assert len(ids) == len(records) == 512
    atoms = {r["example_id"]: r["answer_atoms"] for r in read("data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    data = {r["example_id"]: r for r in read(ROOT / "v17sel_b2b_lineage_clean_holdout/data/v10_v12_train_train_clean.jsonl") if r["example_id"] in ids}
    orders = {r["example_id"]: r["decoded_order"] for r in read(ROOT / "v17sel_b2b_lineage_clean_holdout/sel_c0_train_v8_rollouts.jsonl") if r["example_id"] in ids}
    baseline = {}
    for path in glob.glob(str(ROOT / "v17canon_p0_fresh_v8_prefix_chain/shard_*_of_3/per_prefix.jsonl")):
        for row in read(path):
            if row["example_id"] in ids and row["depth"] == 9:
                assert row["example_id"] not in baseline
                baseline[row["example_id"]] = row
    assert len(atoms) == len(data) == len(orders) == len(baseline) == 512
    for record in records:
        q = record["example_id"]
        assert hit(baseline[q]["f1"], .90) == record["baseline"]["hits"][3]
        actual = qampari_list_metrics(cached_answers(baseline[q]["prediction"]), atoms[q])["f1"]
        assert abs(actual - baseline[q]["f1"]) < 1e-8

    tally = Counter()
    by_query = defaultdict(Counter)
    eligible_by_query = defaultdict(list)
    candidates = list(read(ROOT / "v17packet_r1_label_scale512/per_sentence.jsonl"))
    for row in candidates:
        q = row["example_id"]
        actual = qampari_list_metrics(cached_answers(row["prediction"]), atoms[q])["f1"]
        assert abs(actual - row["f1"]) < 1e-8
        tally["all_actions"] += 1
        if row["tokens"] > baseline[q]["tokens"]:
            tally["excluded_added_context"] += 1
            continue
        b, a = baseline[q], row
        old_answers, new_answers = matched(b["prediction"], atoms[q]), matched(a["prediction"], atoms[q])
        title = data[q]["packet_texts"][orders[q][9]].split("\n", 1)[0].removeprefix("Document title: ")
        matches = [i for i, atom in enumerate(atoms[q]) if atom["answer_text"].casefold() == title.casefold()]
        assert len(matches) <= 1
        title_atom = matches[0] if matches else None
        added, lost = new_answers - old_answers, old_answers - new_answers
        delta = a["f1"] - b["f1"]
        category = "f1_gain" if delta > 1e-8 else "f1_loss" if delta < -1e-8 else "f1_equal"
        tally[category + "_actions"] += 1
        by_query[q][category] += 1
        tally[category + "_with_gold_loss"] += bool(lost)
        tally[category + "_with_gold_gain"] += bool(added)
        tally[category + "_with_title_atom_gain"] += title_atom in added
        tally[category + "_with_answer_identity_change"] += bool(added or lost)
        if category == "f1_gain" and hit(a["f1"], .90) == hit(b["f1"], .90):
            tally["f1_gain_same_090_label"] += 1
            by_query[q]["f1_gain_same_090_label"] += 1
        if category == "f1_loss" and hit(a["f1"], .90) == hit(b["f1"], .90):
            tally["f1_loss_same_090_label"] += 1
            by_query[q]["f1_loss_same_090_label"] += 1
        for level in SENSITIVITY_THRESHOLDS:
            key = f"{level:.2f}"
            if not hit(b["f1"], level) and hit(a["f1"], level):
                tally[f"repair_{key}_actions"] += 1
                by_query[q][f"repair_{key}"] += 1
            if hit(b["f1"], level) and not hit(a["f1"], level):
                tally[f"break_{key}_actions"] += 1
                by_query[q][f"break_{key}"] += 1
        tally["eligible_actions"] += 1
        by_query[q]["eligible"] += 1
        eligible_by_query[q].append(a)

    baseline_dist = Counter(round(row["f1"], 6) for row in baseline.values())
    chain = defaultdict(dict)
    for path in glob.glob(str(ROOT / "v17canon_p0_fresh_v8_prefix_chain/shard_*_of_3/per_prefix.jsonl")):
        for row in read(path):
            chain[row["example_id"]][row["depth"]] = row["f1"]
    assert len(chain) == 1421 and all(len(depths) == 12 for depths in chain.values())
    chain_sensitivity = {}
    for level in (0.85, 0.90, 0.95):
        c = Counter()
        for depths in chain.values():
            bits = [hit(depths[d], level) for d in range(1, 13)]
            runs, current = [], 0
            for active in [*bits, False]:
                if active:
                    current += 1
                elif current:
                    runs.append(current)
                    current = 0
            c["ever_success"] += bool(runs)
            c["depth9_success"] += bits[8]
            c["depth10_success"] += bits[9]
            c["successful_run_width_only_one"] += bool(runs) and max(runs) == 1
            c["multiple_disjoint_success_runs"] += len(runs) > 1
            c["success_to_failure_transitions"] += sum(a and not b for a, b in zip(bits, bits[1:]))
        chain_sensitivity[f"{level:.2f}"] = dict(c)
    by_query_rows = []
    continuous_oracle = Counter()
    for record in records:
        q = record["example_id"]
        counts = by_query[q]
        base = baseline[q]
        # This is hindsight over observed Target outcomes, not a deployable
        # policy. STAY is always in the set and the candidate costs no extra
        # depth9 context tokens.
        options = [(base["f1"], base["tokens"], None)] + [
            (item["f1"], item["tokens"], item["sentence_index"])
            for item in eligible_by_query[q]]
        chosen_f1, chosen_tokens, chosen_index = min(options, key=lambda x: (-x[0], x[1], x[2] if x[2] is not None else -1))
        continuous_oracle["sum_baseline_f1"] += base["f1"]
        continuous_oracle["sum_chosen_f1"] += chosen_f1
        continuous_oracle["baseline_090_success"] += hit(base["f1"], .90)
        continuous_oracle["chosen_090_success"] += hit(chosen_f1, .90)
        continuous_oracle["baseline_complete"] += record["baseline"]["complete"]
        continuous_oracle["chosen_complete"] += all(record["baseline"]["hits"][i] is not False for i in (0, 1, 2, 4)) and hit(chosen_f1, .90)
        continuous_oracle["changed_queries"] += chosen_index is not None
        continuous_oracle["continuous_gain_queries"] += chosen_f1 > base["f1"] + 1e-8
        continuous_oracle["sum_cumulative_token_delta"] += chosen_tokens - base["tokens"]
        by_query_rows.append({"example_id": q, "baseline_f1": baseline[q]["f1"],
                              "baseline_090": hit(baseline[q]["f1"], .90),
                              "eligible_actions": counts["eligible"],
                              "f1_gain_actions": counts["f1_gain"],
                              "f1_loss_actions": counts["f1_loss"],
                              "repair_090_actions": counts["repair_0.90"],
                              "break_090_actions": counts["break_0.90"],
                              "f1_gain_same_090_label": counts["f1_gain_same_090_label"]})
    oof = [x for x in read(ROOT / "v17packet_r2g0_relative_bag_probe/oof.jsonl")
           if x["best_action"] is not None]
    assert len(oof) == 422
    gain_ids = {x["example_id"] for x in by_query_rows if x["f1_gain_actions"] > 0}
    positives = [x["best_score"] for x in oof if x["example_id"] in gain_ids]
    negatives = [x["best_score"] for x in oof if x["example_id"] not in gain_ids]
    auc = sum((a > b) + .5 * (a == b) for a in positives for b in negatives) / (len(positives) * len(negatives))
    globally_ranked = sorted(oof, key=lambda x: -x["best_score"])
    old_score_sensitivity = {"score_source": "already-frozen R2G0 grouped-OOF best-action score",
                             "eligible_queries": len(oof), "positive_continuous_gain_queries": len(positives),
                             "global_score_auc_for_any_continuous_gain": auc,
                             "global_budget": {str(n): {"gain_queries_found": sum(x["example_id"] in gain_ids for x in globally_ranked[:n]),
                                                        "random_expectation": n * len(positives) / len(oof)}
                                               for n in (26, 52)}}
    summary = {
        "scope": "R1 train512, design-exposed, current Qwen3-8B contract, no-extra-depth9-context actions",
        "target_calls": 0, "model_training": False,
        "threshold_085_role": "retrospective sensitivity only; original five-anchor contract unchanged",
        "queries": len(records), "action_counts": dict(tally),
        "fresh_v8_chain_sensitivity_1421": chain_sensitivity,
        "continuous_f1_hindsight_oracle": {
            "baseline_mean_f1_090_stop": continuous_oracle["sum_baseline_f1"] / len(records),
            "chosen_mean_f1_090_stop": continuous_oracle["sum_chosen_f1"] / len(records),
            "baseline_090_success": continuous_oracle["baseline_090_success"],
            "chosen_090_success": continuous_oracle["chosen_090_success"],
            "baseline_complete": continuous_oracle["baseline_complete"],
            "chosen_complete": continuous_oracle["chosen_complete"],
            "changed_queries": continuous_oracle["changed_queries"],
            "continuous_gain_queries": continuous_oracle["continuous_gain_queries"],
            "mean_cumulative_context_token_delta": continuous_oracle["sum_cumulative_token_delta"] / len(records),
        },
        "old_oof_score_diagnostic": old_score_sensitivity,
        "query_counts": {
            "with_eligible_action": sum(x["eligible_actions"] > 0 for x in by_query_rows),
            "with_any_continuous_gain": sum(x["f1_gain_actions"] > 0 for x in by_query_rows),
            "with_any_continuous_loss": sum(x["f1_loss_actions"] > 0 for x in by_query_rows),
            "with_any_090_repair": sum(x["repair_090_actions"] > 0 for x in by_query_rows),
            "with_any_090_break": sum(x["break_090_actions"] > 0 for x in by_query_rows),
            "with_gain_hidden_by_090_label": sum(x["f1_gain_same_090_label"] > 0 for x in by_query_rows),
        },
        "baseline_depth9_f1_point_mass": {f"{value:.6f}": baseline_dist[value]
                                          for value in (0.888889, 0.900000, 0.947368, 1.000000)},
        "limitations": [
            "Action outcomes from one query are correlated; neither action counts nor this cohort estimate unseen-query policy performance.",
            "The 0.85 threshold is a sensitivity analysis, not a replacement target or a new passing gate.",
            "An answer identity change at fixed F1 does not itself establish a semantic failure under the current set-F1 contract.",
            "Counterfactual action outcomes and gold matching are retrospective, forbidden as deployment inputs.",
            "All candidate actions substitute a rank10 sentence for rank9 at depth9; this audit does not test protected insertion or a learned policy.",
        ],
    }
    assert tally["all_actions"] == 1485 and tally["eligible_actions"] == 1441
    assert summary["query_counts"]["with_any_090_repair"] == 39
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in by_query_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
