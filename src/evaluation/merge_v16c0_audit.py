from __future__ import annotations

import argparse
import json

from src.data.schemas import read_jsonl, write_jsonl
from src.evaluation.v16c0_first_irreversible_divergence import summarize
from src.reproducibility import sha256, write_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge disjoint V16-C0 audit shards")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--role", required=True)
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in read_jsonl(path)]
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("V16-C0 shards overlap")
    rows.sort(key=lambda row: row["example_id"])
    result = summarize(rows)
    result.update({
        "role": args.role,
        "merged_shards": len(args.inputs),
        "input_sha256": {path: sha256(path) for path in args.inputs},
        "definition": {
            "exact_optimal": "prefix has selected only exact lexicographic DP-optimal actions",
            "contract_viable": "some suffix can still attain every active fidelity anchor",
            "regret_qualified": "contract viable and minimum final per-trajectory normalized regret <= 0.03",
        },
    })
    write_jsonl(args.output, rows)
    write_metadata(args.summary, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
