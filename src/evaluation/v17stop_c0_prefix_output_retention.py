"""Gold-free answer-set retention rules on fresh depth-9/10 Target outputs."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from src.data.build_v17sel_b2b_lineage_holdout import fold
from src.data.schemas import read_jsonl
from src.evaluation.qampari_metrics import normalize_list_answer, qampari_list_metrics

LEVELS = (.60, .70, .80, .90, .95)
ROOT = Path("results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout")
P0 = Path("results/v2_rank_then_cut/v17canon_p0_fresh_v8_prefix_chain")
OUT = Path("results/v2_rank_then_cut/v17stop_c0_prefix_output_retention")


def normalized_map(prediction: str) -> dict[str, str]:
    # CANON-P0 stores already-parsed answers joined by this exact delimiter.
    # Running the raw-generation parser again would truncate answers such as
    # "No. 17 Squadron" at the period and corrupt the cached F1 contract.
    return {normalize_list_answer(x): x for x in prediction.split(" # ") if x.strip()}


def rules(a9: dict[str, str], a10: dict[str, str], old_text: str, new_text: str) -> dict[str, list[str]]:
    old = normalize_list_answer(old_text)
    new = normalize_list_answer(new_text)
    return {
        "depth9_only": list(a9.values()),
        "depth10_only": list(a10.values()),
        "union_latest_first": list(a10.values()) + [v for k, v in a9.items() if k not in a10],
        "intersection": [v for k, v in a10.items() if k in a9],
        "depth9_plus_new_packet_supported_depth10": list(a9.values()) +
            [v for k, v in a10.items() if k not in a9 and k in new],
        "depth10_plus_old_context_supported_depth9": list(a10.values()) +
            [v for k, v in a9.items() if k not in a10 and k in old],
    }


def paired_bootstrap_ci(deltas: list[int], seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(deltas, dtype=np.int16)
    indices = rng.integers(0, len(values), size=(5000, len(values)))
    return [float(x) for x in np.quantile(values[indices].sum(axis=1), [.025, .975])]


def main() -> None:
    config = json.loads(Path("configs/v17stop_c0_prefix_output_retention.json").read_text())
    if config["status"] != "FROZEN_READ_ONLY_TRAIN_SIDE":
        raise AssertionError("unfrozen protocol")
    source = {r["example_id"]: r for r in read_jsonl(ROOT / "sel_c0_train_top10_candidates.jsonl")
              if fold(r["example_id"]) != 4}
    ids = set(source)
    data = {r["example_id"]: r for r in read_jsonl(ROOT / "data/v10_v12_train_train_clean.jsonl")
            if r["example_id"] in ids}
    atoms = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        "data/units/qampari_rank_v2_candidates5000_annotations.jsonl") if r["example_id"] in ids}
    chain = defaultdict(dict)
    for path in sorted(P0.glob("shard_*_of_3/per_prefix.jsonl")):
        for row in read_jsonl(path):
            if row["example_id"] in ids and row["depth"] in (9, 10):
                if row["depth"] in chain[row["example_id"]]:
                    raise AssertionError("duplicate fresh prefix")
                chain[row["example_id"]][row["depth"]] = row
    if any(len(x) != 1421 for x in (source, data, atoms, chain)) or any(len(x) != 2 for x in chain.values()):
        raise AssertionError("incomplete train-side join")
    records = []
    for query in sorted(ids):
        ninth, tenth = chain[query][9], chain[query][10]
        if ninth["mask"] & ~tenth["mask"] or (tenth["mask"] ^ ninth["mask"]).bit_count() != 1:
            raise AssertionError("not a one-packet add-only transition")
        packet = (tenth["mask"] ^ ninth["mask"]).bit_length() - 1
        packets = data[query]["packet_texts"]
        old_text = "\n".join(text for i, text in enumerate(packets) if ninth["mask"] & (1 << i))
        generated = rules(normalized_map(ninth["prediction"]), normalized_map(tenth["prediction"]),
                          old_text, packets[packet])
        if list(generated) != config["global_gold_free_rules"]:
            raise AssertionError("operator list drift")
        values = {}
        active = {float(x) for x in source[query]["attainable_levels"]}
        for name, answer_list in generated.items():
            f1 = float(qampari_list_metrics(answer_list, atoms[query])["f1"])
            values[name] = {"f1": f1, "answer_count": len(answer_list),
                            "success": {str(level): f1 + 1e-6 >= level for level in LEVELS if level in active},
                            "complete": all(f1 + 1e-6 >= level for level in LEVELS if level in active)}
        for depth, name in ((9, "depth9_only"), (10, "depth10_only")):
            if abs(values[name]["f1"] - chain[query][depth]["f1"]) > 1e-6:
                raise AssertionError(f"scorer or parser mismatch at {query}, depth {depth}")
        records.append({"example_id": query, "added_packet": packet,
                        "depth9_context_tokens": ninth["tokens"],
                        "depth10_context_tokens": tenth["tokens"],
                        "depth9_target_tokens": ninth["prompt_tokens"] + ninth["generated_tokens"],
                        "depth10_target_tokens": tenth["prompt_tokens"] + tenth["generated_tokens"],
                        "outputs": values})
    baseline = "depth10_only"
    report = {"protocol": config["protocol"], "population": len(records), "rules": {},
              "new_target_calls": 0, "sealed_outcome_sets_read": False,
              "limits": config["interpretation"]}
    for name in config["global_gold_free_rules"]:
        rows = [r["outputs"][name] for r in records]
        based = [r["outputs"][baseline] for r in records]
        pairs = list(zip(rows, based))
        two_calls = name not in ("depth9_only", "depth10_only")
        report["rules"][name] = {
            "success": {str(level): sum(row["success"].get(str(level), False) for row in rows)
                        for level in LEVELS},
            "complete": sum(row["complete"] for row in rows),
            "mean_f1": mean(row["f1"] for row in rows),
            "mean_answer_count": mean(row["answer_count"] for row in rows),
            "090_repairs_vs_depth10": sum(not old["success"]["0.9"] and new["success"]["0.9"]
                                         for new, old in pairs),
            "090_breaks_vs_depth10": sum(old["success"]["0.9"] and not new["success"]["0.9"]
                                        for new, old in pairs),
            "complete_repairs_vs_depth10": sum(not old["complete"] and new["complete"]
                                               for new, old in pairs),
            "complete_breaks_vs_depth10": sum(old["complete"] and not new["complete"]
                                              for new, old in pairs),
            "090_net_paired_bootstrap_95ci": paired_bootstrap_ci([
                int(new["success"]["0.9"]) - int(old["success"]["0.9"])
                for new, old in pairs], 20260920),
            "complete_net_paired_bootstrap_95ci": paired_bootstrap_ci([
                int(new["complete"]) - int(old["complete"])
                for new, old in pairs], 20260921),
            "rescued_090_success9_fail10": sum(
                r["outputs"]["depth9_only"]["success"]["0.9"] and
                not r["outputs"]["depth10_only"]["success"]["0.9"] and
                r["outputs"][name]["success"]["0.9"] for r in records),
            "mean_target_tokens_per_090_request": mean(
                r["depth9_target_tokens"] + r["depth10_target_tokens"] if two_calls else
                r["depth9_target_tokens"] if name == "depth9_only" else r["depth10_target_tokens"]
                for r in records),
            "target_calls_per_090_request": 2 if two_calls else 1,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (OUT / "per_query.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps(report["rules"], indent=2), flush=True)


if __name__ == "__main__":
    main()
