from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

from src.data.schemas import ExactSearchResult, read_jsonl, write_jsonl
from src.reproducibility import sha256, write_metadata
from src.search.atomic_nested_chain import best_binary_nested_chain
from src.search.rank_then_cut import best_prefix_nested_chain, prefix_state
from src.search.weighted_precedence import exact_weighted_precedence_order


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _pair_text(row: dict[str, Any], winner: int, loser: int) -> str:
    return " ".join(
        (row["question"], row["packet_texts"][winner], row["packet_texts"][loser])
    )


def _closest_repair(
    source: dict[str, Any],
    exact: Sequence[ExactSearchResult],
    order: Sequence[int],
    target: float,
) -> dict[str, Any]:
    width = len(order)
    by_state = {row.state: row for row in exact}
    positions = {packet: position for position, packet in enumerate(order)}
    all_packets = set(range(width))
    prefix_rows = [(count, by_state[prefix_state(order, count)]) for count in range(width + 1)]
    best_count, best_row = max(
        prefix_rows, key=lambda item: (item[1].fidelity, -item[1].tokens, -item[0])
    )
    candidates = []
    for row in exact:
        if row.fidelity < target:
            continue
        selected = {index for index, value in enumerate(row.state) if value}
        excluded = all_packets - selected
        current = set(order[: len(selected)])
        missing = selected - current
        intruders = current - selected
        crossings = sum(
            positions[outside] < positions[inside]
            for inside in selected
            for outside in excluded
        )
        candidates.append(
            (
                (crossings, len(missing), row.tokens, -row.fidelity, row.state),
                row,
                missing,
                intruders,
            )
        )
    if not candidates:
        raise ValueError(f"{source['example_id']}: target {target} has no feasible exact state")
    key, repaired, missing, intruders = min(candidates, key=lambda item: item[0])
    return {
        "best_prefix_count": best_count,
        "best_prefix_fidelity": best_row.fidelity,
        "fidelity_shortfall": target - best_row.fidelity,
        "minimum_adjacent_crossings": key[0],
        "minimum_boundary_replacements": key[1],
        "closest_feasible_state": list(repaired.state),
        "closest_feasible_tokens": repaired.tokens,
        "closest_feasible_fidelity": repaired.fidelity,
        "missing_packets": sorted(missing),
        "intruder_packets": sorted(intruders),
    }


def _evaluate_logits(
    source_rows: Sequence[dict[str, Any]],
    exact_dir: Path,
    logits_by_id: dict[str, list[list[float]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs = []
    successes = anchors = trajectories = 0
    regrets = []
    per_level: dict[float, list[tuple[bool, float | None]]] = defaultdict(list)
    for source in source_rows:
        example_id = source["example_id"]
        logits = logits_by_id[example_id]
        order = exact_weighted_precedence_order(logits)
        exact = list(read_jsonl(exact_dir / f"{example_id}.jsonl", ExactSearchResult))
        by_state = {row.state: row for row in exact}
        levels = [float(value) for value in source["active_levels"]]
        full_tokens = by_state[(1,) * len(order)].tokens
        oracle = best_binary_nested_chain(exact, levels)
        learned = best_prefix_nested_chain(exact, order, levels)
        all_success = True
        learned_tokens = oracle_tokens = 0
        anchor_rows = []
        for level, predicted, baseline in zip(levels, learned, oracle):
            success = bool(predicted["feasible"])
            all_success &= success
            successes += int(success)
            anchors += 1
            tokens = int(predicted["tokens"]) if predicted["tokens"] is not None else None
            learned_tokens += tokens or 0
            oracle_tokens += int(baseline["tokens"])
            regret = (
                (tokens - int(baseline["tokens"])) / full_tokens
                if success and tokens is not None
                else None
            )
            per_level[level].append((success, regret))
            anchor_rows.append(
                {
                    "fidelity_level": level,
                    "contract_success": success,
                    "cutoff_count": predicted["cutoff_count"],
                    "tokens": tokens,
                    "oracle_nested_tokens": baseline["tokens"],
                    "rate_regret_normalized_if_success": regret,
                    "state": predicted["state"],
                }
            )
        trajectories += int(all_success)
        trajectory_regret = (
            (learned_tokens - oracle_tokens) / (len(levels) * full_tokens)
            if all_success
            else None
        )
        if trajectory_regret is not None:
            regrets.append(trajectory_regret)
        outputs.append(
            {
                "example_id": example_id,
                "precedence_logits": logits,
                "decoded_order": list(order),
                "all_active_contracts_success": all_success,
                "oracle_cutoff_ranking_regret_normalized": trajectory_regret,
                "anchors": anchor_rows,
            }
        )
    summary = {
        "examples": len(source_rows),
        "oracle_cutoff_active_contract_success_fraction": successes / anchors,
        "oracle_cutoff_all_active_contracts_success_fraction": trajectories / len(source_rows),
        "oracle_cutoff_feasible_trajectory_examples": len(regrets),
        "mean_oracle_cutoff_ranking_regret_normalized_feasible": mean(regrets),
        "per_level": {
            str(level): {
                "examples": len(values),
                "contract_successes": sum(success for success, _ in values),
                "contract_success_fraction": mean(success for success, _ in values),
                "mean_rate_regret_normalized_successful": mean(
                    regret for _, regret in values if regret is not None
                )
                if any(regret is not None for _, regret in values)
                else None,
            }
            for level, values in sorted(per_level.items())
        },
    }
    return outputs, summary


def _regret_decomposition(
    source: dict[str, Any], prediction: dict[str, Any], exact: Sequence[ExactSearchResult]
) -> dict[str, Any]:
    levels = [float(value) for value in source["active_levels"]]
    oracle = best_binary_nested_chain(exact, levels)
    packet_tokens = [int(value) for value in source["packet_tokens"]]
    full_tokens = next(row.tokens for row in exact if all(row.state))
    anchors = []
    total_extra = total_missing = total_excess = 0
    for learned, baseline in zip(prediction["anchors"], oracle):
        if not learned["contract_success"]:
            continue
        learned_state = tuple(int(value) for value in learned["state"])
        oracle_state = tuple(int(value) for value in baseline["state"])
        extra = [i for i, (left, right) in enumerate(zip(learned_state, oracle_state)) if left and not right]
        missing = [i for i, (left, right) in enumerate(zip(learned_state, oracle_state)) if right and not left]
        extra_tokens = sum(packet_tokens[index] for index in extra)
        missing_tokens = sum(packet_tokens[index] for index in missing)
        excess = int(learned["tokens"]) - int(baseline["tokens"])
        total_extra += extra_tokens
        total_missing += missing_tokens
        total_excess += excess
        anchors.append(
            {
                "fidelity_level": learned["fidelity_level"],
                "learned_cutoff_count": learned["cutoff_count"],
                "learned_tokens": learned["tokens"],
                "oracle_tokens": baseline["tokens"],
                "token_excess": excess,
                "extra_packets": extra,
                "extra_packet_tokens": extra_tokens,
                "missing_oracle_packets": missing,
                "missing_oracle_packet_tokens": missing_tokens,
            }
        )
    return {
        "example_id": source["example_id"],
        "regret": prediction["oracle_cutoff_ranking_regret_normalized"],
        "full_state_tokens": full_tokens,
        "total_token_excess": total_excess,
        "total_extra_packet_tokens": total_extra,
        "total_missing_oracle_packet_tokens": total_missing,
        "anchors": anchors,
    }


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": median(ordered),
        "mean": mean(ordered),
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V4 three-seed tail-risk and rate diagnostic")
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--train-oracle", required=True)
    parser.add_argument("--exact-dir", required=True)
    parser.add_argument("--selected-label", required=True)
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "DETAILS"), required=True)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source_rows = list(read_jsonl(args.oracle))
    source_by_id = {row["example_id"]: row for row in source_rows}
    train_rows = list(read_jsonl(args.train_oracle))
    exact_dir = Path(args.exact_dir)
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    artifacts: dict[str, Any] = {
        "oracle": args.oracle,
        "oracle_sha256": sha256(args.oracle),
        "train_oracle": args.train_oracle,
        "train_oracle_sha256": sha256(args.train_oracle),
        "runs": [],
    }
    for label, path in args.run:
        rows = list(read_jsonl(path))
        by_id = {row["example_id"]: row for row in rows}
        if set(by_id) != set(source_by_id):
            raise ValueError(f"{label}: details and oracle IDs differ")
        runs[label] = by_id
        artifacts["runs"].append({"label": label, "path": path, "sha256": sha256(path)})
    if args.selected_label not in runs:
        raise ValueError("selected label is absent from --run inputs")

    labels = list(runs)
    ensemble_logits = {}
    for example_id in source_by_id:
        matrices = [runs[label][example_id]["precedence_logits"] for label in labels]
        width = len(matrices[0])
        ensemble_logits[example_id] = [
            [mean(matrix[i][j] for matrix in matrices) for j in range(width)]
            for i in range(width)
        ]
    ensemble_details, ensemble_summary = _evaluate_logits(source_rows, exact_dir, ensemble_logits)

    selected = runs[args.selected_label]
    repair_rows = []
    repair_relations = []
    supervision_counts: Counter[str] = Counter()
    for source in source_rows:
        example_id = source["example_id"]
        prediction = selected[example_id]
        anchor = next(
            (row for row in prediction["anchors"] if math.isclose(float(row["fidelity_level"]), args.target)),
            None,
        )
        if anchor is None or anchor["contract_success"]:
            continue
        exact = list(read_jsonl(exact_dir / f"{example_id}.jsonl", ExactSearchResult))
        repair = _closest_repair(source, exact, prediction["decoded_order"], args.target)
        preferences = {tuple(int(value) for value in pair) for pair in source["pairwise_preferences"]}
        relation_rows = []
        for winner in repair["missing_packets"]:
            for loser in repair["intruder_packets"]:
                if (winner, loser) in preferences:
                    supervision = "supports_repair"
                elif (loser, winner) in preferences:
                    supervision = "opposes_repair"
                else:
                    supervision = "unlabeled"
                supervision_counts[supervision] += 1
                logits = {
                    label: float(runs[label][example_id]["precedence_logits"][winner][loser])
                    for label in labels
                }
                signs = {label: value > 0 for label, value in logits.items()}
                row = {
                    "example_id": example_id,
                    "winner": winner,
                    "loser": loser,
                    "supervision": supervision,
                    "seed_logits": logits,
                    "seed_supports_repair": signs,
                    "mean_logit": mean(logits.values()),
                    "selected_logit": logits[args.selected_label],
                    "selected_absolute_margin": abs(logits[args.selected_label]),
                    "supporting_seed_count": sum(signs.values()),
                }
                relation_rows.append(row)
                repair_relations.append((source, row))
        repair_rows.append({"example_id": example_id, **repair, "relations": relation_rows})

    # A lightweight, auditable support-density proxy. It measures lexical
    # neighborhood coverage and is not treated as a representation-space proof.
    train_pairs = []
    inverted: dict[str, set[int]] = defaultdict(set)
    for train in train_rows:
        for winner, loser in train["pairwise_preferences"]:
            token_set = _tokens(_pair_text(train, int(winner), int(loser)))
            index = len(train_pairs)
            train_pairs.append((train["example_id"], int(winner), int(loser), token_set))
            for token in token_set:
                inverted[token].add(index)
    support_rows = []
    for source, relation in repair_relations:
        target_tokens = _tokens(_pair_text(source, relation["winner"], relation["loser"]))
        candidates: set[int] = set()
        for token in target_tokens:
            candidates.update(inverted.get(token, ()))
        scored = sorted(
            (
                (_jaccard(target_tokens, train_pairs[index][3]), index)
                for index in candidates
            ),
            reverse=True,
        )
        neighbors = []
        for similarity, index in scored[:5]:
            example_id, winner, loser, _ = train_pairs[index]
            neighbors.append(
                {
                    "similarity": similarity,
                    "example_id": example_id,
                    "winner": winner,
                    "loser": loser,
                }
            )
        support_rows.append(
            {
                "example_id": source["example_id"],
                "winner": relation["winner"],
                "loser": relation["loser"],
                "supervision": relation["supervision"],
                "neighbors_at_jaccard_0.25": sum(value >= 0.25 for value, _ in scored),
                "neighbors_at_jaccard_0.50": sum(value >= 0.50 for value, _ in scored),
                "top_neighbors": neighbors,
            }
        )

    feasible = [
        row for row in selected.values()
        if row["oracle_cutoff_ranking_regret_normalized"] is not None
    ]
    top_regret = sorted(
        feasible,
        key=lambda row: float(row["oracle_cutoff_ranking_regret_normalized"]),
        reverse=True,
    )[:10]
    regret_rows = []
    for prediction in top_regret:
        source = source_by_id[prediction["example_id"]]
        exact = list(read_jsonl(exact_dir / f"{prediction['example_id']}.jsonl", ExactSearchResult))
        regret_rows.append(_regret_decomposition(source, prediction, exact))

    selected_margins = [
        abs(float(row["selected_logit"])) for _, row in repair_relations
    ]
    supporting_seed_counts = Counter(
        int(row["supporting_seed_count"]) for _, row in repair_relations
    )
    summary = {
        "complete": True,
        "status": "consumed-development diagnostic; not a confirmatory result",
        "target_anchor": args.target,
        "selected_label": args.selected_label,
        "artifacts": artifacts,
        "repair_failures": len(repair_rows),
        "repair_relations": len(repair_relations),
        "repair_supervision": dict(supervision_counts),
        "minimum_boundary_replacements": dict(
            Counter(row["minimum_boundary_replacements"] for row in repair_rows)
        ),
        "selected_wrong_edge_absolute_margin": _quantiles(selected_margins),
        "supporting_seed_count_distribution": dict(sorted(supporting_seed_counts.items())),
        "unanimous_wrong_relations": supporting_seed_counts[0],
        "at_least_one_seed_correct_relations": sum(
            count for supporting, count in supporting_seed_counts.items() if supporting > 0
        ),
        "lexical_support_proxy": {
            "description": "Jaccard overlap over query, winner packet, and loser packet; diagnostic only",
            "relations_with_neighbor_at_0.25": sum(
                row["neighbors_at_jaccard_0.25"] > 0 for row in support_rows
            ),
            "relations_with_neighbor_at_0.50": sum(
                row["neighbors_at_jaccard_0.50"] > 0 for row in support_rows
            ),
            "median_top_similarity": median(
                row["top_neighbors"][0]["similarity"] for row in support_rows if row["top_neighbors"]
            )
            if support_rows
            else None,
        },
        "ensemble": ensemble_summary,
        "top_regret": {
            "examples": len(regret_rows),
            "sum_regret": sum(float(row["regret"]) for row in regret_rows),
            "sum_total_token_excess": sum(row["total_token_excess"] for row in regret_rows),
            "sum_extra_packet_tokens": sum(row["total_extra_packet_tokens"] for row in regret_rows),
            "sum_missing_oracle_packet_tokens": sum(
                row["total_missing_oracle_packet_tokens"] for row in regret_rows
            ),
        },
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "repair_edges.jsonl", repair_rows)
    write_jsonl(output / "support_density.jsonl", support_rows)
    write_jsonl(output / "top_regret_decomposition.jsonl", regret_rows)
    write_jsonl(output / "ensemble_details.jsonl", ensemble_details)
    write_metadata(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
