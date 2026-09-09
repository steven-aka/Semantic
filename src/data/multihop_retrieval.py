from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from src.data.label_free_pilot import bm25_scores, sentence_candidates
from src.data.schemas import QAExample, SemanticUnit, read_jsonl, write_jsonl
from src.evaluation.fact_coverage import raw_supporting_facts
from src.representation.token_counter import count_tokens, load_tokenizer
from src.reproducibility import sha256, write_metadata


Policy = Callable[[Mapping[str, Any], float, float], list[int]]


def _rendered(row: Mapping[str, Any]) -> tuple[list[tuple[int, str, int, str]], list[str]]:
    candidates = sentence_candidates(row)
    return candidates, [f"Title: {title}\n{sentence}" for _, title, _, sentence in candidates]


def _fill(
    selected: list[int], ranked: Sequence[int], count: int = 6
) -> list[int]:
    seen = set(selected)
    for index in ranked:
        if index not in seen:
            selected.append(index)
            seen.add(index)
        if len(selected) == count:
            break
    return sorted(selected[:count])


def global_bm25(row: Mapping[str, Any], k1: float, b: float) -> list[int]:
    _, rendered = _rendered(row)
    scores = bm25_scores(str(row["question"]), rendered, k1=k1, b=b)
    ranked = sorted(range(len(rendered)), key=lambda index: (-scores[index], index))
    return sorted(ranked[:6])


def _title_groups(
    candidates: Sequence[tuple[int, str, int, str]]
) -> tuple[list[str], dict[str, list[int]]]:
    titles: list[str] = []
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (_, title, _, _) in enumerate(candidates):
        if title not in groups:
            titles.append(title)
        groups[title].append(index)
    return titles, groups


def hierarchical(row: Mapping[str, Any], k1: float, b: float) -> list[int]:
    candidates, rendered = _rendered(row)
    titles, groups = _title_groups(candidates)
    paragraph_documents = [
        f"Title: {title}\n" + " ".join(candidates[index][3] for index in groups[title])
        for title in titles
    ]
    paragraph_scores = bm25_scores(
        str(row["question"]), paragraph_documents, k1=k1, b=b
    )
    title_rank = sorted(
        range(len(titles)), key=lambda index: (-paragraph_scores[index], index)
    )
    sentence_scores = bm25_scores(str(row["question"]), rendered, k1=k1, b=b)
    selected: list[int] = []
    for title_index in title_rank[:3]:
        title = titles[title_index]
        ranked = sorted(groups[title], key=lambda index: (-sentence_scores[index], index))
        selected.extend(ranked[:2])
    global_rank = sorted(
        range(len(rendered)), key=lambda index: (-sentence_scores[index], index)
    )
    return _fill(selected, global_rank)


def _normalized_title(text: str) -> str:
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text.casefold())
    return " ".join(re.findall(r"\w+", text, re.UNICODE))


def linked_title_pair(row: Mapping[str, Any], k1: float, b: float) -> list[int]:
    candidates, rendered = _rendered(row)
    titles, groups = _title_groups(candidates)
    paragraph_documents = [
        f"Title: {title}\n" + " ".join(candidates[index][3] for index in groups[title])
        for title in titles
    ]
    paragraph_scores = bm25_scores(
        str(row["question"]), paragraph_documents, k1=k1, b=b
    )
    maximum = max(paragraph_scores) if paragraph_scores else 0.0
    normalized_scores = [score / maximum if maximum else 0.0 for score in paragraph_scores]
    normalized_titles = [_normalized_title(title) for title in titles]
    paragraph_text = [" ".join(document.casefold().split()) for document in paragraph_documents]

    def linked(left: int, right: int) -> bool:
        return (
            len(normalized_titles[right]) >= 3
            and normalized_titles[right] in _normalized_title(paragraph_text[left])
        ) or (
            len(normalized_titles[left]) >= 3
            and normalized_titles[left] in _normalized_title(paragraph_text[right])
        )

    pairs: list[tuple[float, int, int]] = []
    for left in range(len(titles)):
        for right in range(left + 1, len(titles)):
            score = normalized_scores[left] + normalized_scores[right]
            if linked(left, right):
                score += 1.5
            pairs.append((score, left, right))
    if not pairs:
        return global_bm25(row, k1, b)
    _, left, right = max(pairs, key=lambda item: (item[0], -item[1], -item[2]))
    expanded_query = " ".join(
        (str(row["question"]), titles[left], titles[right])
    )
    expanded_scores = bm25_scores(expanded_query, rendered, k1=k1, b=b)
    selected: list[int] = []
    for title_index in (left, right):
        ranked = sorted(
            groups[titles[title_index]],
            key=lambda index: (-expanded_scores[index], index),
        )
        selected.extend(ranked[:3])
    global_rank = sorted(
        range(len(rendered)), key=lambda index: (-expanded_scores[index], index)
    )
    return _fill(selected, global_rank)


POLICIES: dict[str, Policy] = {
    "global_bm25": global_bm25,
    "hierarchical_3_titles_x_2_sentences": hierarchical,
    "linked_title_pair_3_sentences_each": linked_title_pair,
}


def build_example(
    row: Mapping[str, Any], selected: Sequence[int]
) -> QAExample:
    candidates, rendered = _rendered(row)
    units = [
        SemanticUnit(
            unit_id=unit_id,
            text=rendered[index],
            supporting=False,
            title=candidates[index][1],
            source_sentence_id=candidates[index][2],
        )
        for unit_id, index in enumerate(sorted(selected))
    ]
    return QAExample(
        example_id=str(row["id"]),
        dataset="hotpotqa",
        question=str(row["question"]).strip(),
        answer=str(row["answer"]).strip(),
        context="\n\n".join(unit.text for unit in units),
        units=units,
    )


def evaluate_policy(
    rows: Sequence[Mapping[str, Any]], policy: Policy, k1: float, b: float
) -> dict[str, Any]:
    selected_facts = total_facts = selected_titles = total_titles = 0
    unique_titles: list[int] = []
    per_example: list[dict[str, Any]] = []
    for row in rows:
        candidates = sentence_candidates(row)
        selected = policy(row, k1, b)
        selected_coordinates = {(candidates[i][1], candidates[i][2]) for i in selected}
        selected_title_set = {candidates[i][1] for i in selected}
        facts = raw_supporting_facts(row)
        fact_hits = sum(
            (fact["title"], fact["sentence_id"]) in selected_coordinates
            for fact in facts
        )
        gold_titles = {fact["title"] for fact in facts}
        title_hits = len(gold_titles & selected_title_set)
        selected_facts += fact_hits
        total_facts += len(facts)
        selected_titles += title_hits
        total_titles += len(gold_titles)
        unique_titles.append(len(selected_title_set))
        per_example.append(
            {
                "example_id": row["id"],
                "selected_facts": fact_hits,
                "gold_facts": len(facts),
                "selected_support_titles": title_hits,
                "gold_support_titles": len(gold_titles),
            }
        )
    return {
        "examples": len(rows),
        "selected_facts": selected_facts,
        "gold_facts": total_facts,
        "fact_coverage": selected_facts / total_facts if total_facts else 0.0,
        "selected_support_titles": selected_titles,
        "gold_support_titles": total_titles,
        "support_title_coverage": selected_titles / total_titles if total_titles else 0.0,
        "mean_unique_titles": mean(unique_titles) if unique_titles else 0.0,
        "per_example": per_example,
    }


def _rank(example_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{example_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen label-free multi-hop retrieval screen")
    parser.add_argument("--config", default="configs/retrieval_r0.json")
    parser.add_argument("--raw", default="data/raw/hotpot_validation.parquet")
    parser.add_argument("--output-dir", default="results/retrieval_r0")
    parser.add_argument("--candidate-output", default="data/units/hotpot_v0_4_candidates.jsonl")
    parser.add_argument("--tokenizer", default="models/Qwen3-8B")
    args = parser.parse_args()

    import pyarrow.parquet as parquet

    with Path(args.config).open(encoding="utf-8") as handle:
        config = json.load(handle)
    excluded = {
        example.example_id
        for path in config["excluded_id_sources"]
        for example in read_jsonl(path, QAExample)
    }
    columns = ["id", "question", "answer", "context", "supporting_facts"]
    ranked_rows: list[tuple[str, Mapping[str, Any]]] = []
    for batch in parquet.ParquetFile(args.raw).iter_batches(batch_size=256, columns=columns):
        for row in batch.to_pylist():
            if row["id"] not in excluded and len(sentence_candidates(row)) >= config["n_units"]:
                ranked_rows.append((_rank(row["id"], config["split"]["salt"]), row))
    ranked_rows.sort(key=lambda item: (item[0], item[1]["id"]))
    tokenizer = load_tokenizer(args.tokenizer)
    valid: list[Mapping[str, Any]] = []
    required = config["split"]["development_examples"] + config["split"]["locked_test_examples"]
    for _, row in ranked_rows:
        _, rendered = _rendered(row)
        if all(count_tokens(tokenizer, text) <= config["max_tokens"] for text in rendered):
            valid.append(row)
            if len(valid) == required:
                break
    if len(valid) != required:
        raise RuntimeError(f"only {len(valid)} valid rows; need {required}")
    development = valid[: config["split"]["development_examples"]]
    locked_test = valid[config["split"]["development_examples"] :]
    k1, b = config["bm25"]["k1"], config["bm25"]["b"]

    development_metrics = {
        name: evaluate_policy(development, POLICIES[name], k1, b)
        for name in config["candidate_policies"]
    }
    chosen = max(
        config["candidate_policies"],
        key=lambda name: (
            development_metrics[name]["fact_coverage"],
            -config["candidate_policies"].index(name),
        ),
    )
    test_metrics = evaluate_policy(locked_test, POLICIES[chosen], k1, b)
    threshold = config["locked_test_gate"]["minimum_top6_gold_fact_coverage"]
    passed = test_metrics["fact_coverage"] >= threshold

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(output_dir / "development_metrics.json", development_metrics)
    write_metadata(
        output_dir / "locked_test_metrics.json",
        {"chosen_policy": chosen, "passed": passed, "threshold": threshold, **test_metrics},
    )
    if passed:
        exact_rows = locked_test[: config["split"]["v0_4_exact_examples"]]
        examples = [
            build_example(row, POLICIES[chosen](row, k1, b)) for row in exact_rows
        ]
        # The selector itself must remain invariant to answer/support-label edits.
        for row, example in zip(exact_rows, examples):
            perturbed = {**row, "answer": "LABEL_ERASED", "supporting_facts": {"title": [], "sent_id": []}}
            rebuilt = build_example(perturbed, POLICIES[chosen](perturbed, k1, b))
            left = [(u.title, u.source_sentence_id, u.text) for u in example.units]
            right = [(u.title, u.source_sentence_id, u.text) for u in rebuilt.units]
            if left != right:
                raise AssertionError(f"label leakage for {example.example_id}")
        write_jsonl(args.candidate_output, examples)
    manifest = {
        "complete": True,
        "passed": passed,
        "decision": "BUILD_V0_4" if passed else "STOP_RETHINK_N6",
        "chosen_policy": chosen,
        "config": args.config,
        "config_sha256": sha256(args.config),
        "raw_sha256": sha256(args.raw),
        "excluded_ids": len(excluded),
        "development_ids": [row["id"] for row in development],
        "locked_test_ids": [row["id"] for row in locked_test],
        "candidate_output": args.candidate_output if passed else None,
        "candidate_output_sha256": sha256(args.candidate_output) if passed else None,
        "label_permutation_invariance": passed,
    }
    write_metadata(output_dir / "manifest.json", manifest)
    print(json.dumps({
        "development": {name: metrics["fact_coverage"] for name, metrics in development_metrics.items()},
        "chosen_policy": chosen,
        "locked_test_fact_coverage": test_metrics["fact_coverage"],
        "passed": passed,
        "decision": manifest["decision"],
    }, indent=2))


if __name__ == "__main__":
    main()
