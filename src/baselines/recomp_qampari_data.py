"""Build train-only sentence triplets for a QAMPARI-adapted RECOMP baseline."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from src.data.schemas import read_jsonl


SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
WORD = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    return " ".join(WORD.findall(text.lower()))


def sentence_rows(record: dict, atoms: list[dict]) -> list[dict]:
    aliases = [[norm(x) for x in a["aliases"] if norm(x)] for a in atoms]
    rows = []
    for packet_id, packet in enumerate(record["packet_texts"]):
        for sentence_id, text in enumerate(x.strip() for x in SPLIT.split(packet) if x.strip()):
            normalized = f" {norm(text)} "
            covered = [i for i, group in enumerate(aliases) if any(f" {a} " in normalized for a in group)]
            rows.append({"packet_id": packet_id, "sentence_id": sentence_id, "text": text,
                         "answer_indices": covered})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("configs/baseline_recomp_qampari_extractive.json"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/v2_rank_then_cut/baseline_recomp_qampari_extractive"))
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    annotations = {r["example_id"]: r["answer_atoms"] for r in read_jsonl(
        Path("data/units/qampari_rank_v2_candidates5000_annotations.jsonl"))}
    excluded = set(json.loads(Path(cfg["exclude_query_ids"].split(":")[0]).read_text())["screen_query_ids"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"protocol": cfg["protocol"], "excluded_screen_queries": len(excluded), "splits": {}}
    for split, source_key in (("train", "train_source"), ("validation", "validation_source")):
        output = args.output_dir / f"{split}.jsonl"; queries = positives = negatives = 0
        with output.open("w", encoding="utf-8") as f:
            for record in read_jsonl(Path(cfg[source_key])):
                if record["example_id"] in excluded:
                    continue
                units = sentence_rows(record, annotations[record["example_id"]])
                pos = [u for u in units if u["answer_indices"]]; neg = [u for u in units if not u["answer_indices"]]
                if not pos or not neg:
                    continue
                queries += 1; positives += len(pos); negatives += len(neg)
                rec = {"example_id": record["example_id"], "question": record["question"],
                       "positive_sentences": pos, "negative_sentences": neg}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        summary["splits"][split] = {"queries": queries, "positive_sentences": positives,
                                     "negative_sentences": negatives, "sha_source": cfg[source_key]}
    summary.update({"target_loaded": False, "sealed_sets_read": False})
    (args.output_dir / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
