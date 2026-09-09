from __future__ import annotations

import argparse

from src.data.normalize_hotpot import select_controlled_units
from src.data.schemas import QAExample, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Select fixed-size controlled exact-pilot units")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--units", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    selected = []
    for example in read_jsonl(args.input, QAExample):
        if len(example.units) < args.units:
            continue
        try:
            selected.append(select_controlled_units(example, args.units))
        except ValueError:
            continue
        if args.limit is not None and len(selected) >= args.limit:
            break
    if args.limit is not None and len(selected) < args.limit:
        raise RuntimeError(f"only {len(selected)} eligible examples found; requested {args.limit}")
    write_jsonl(args.output, selected)


if __name__ == "__main__":
    main()

