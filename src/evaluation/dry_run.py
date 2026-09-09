"""Dependency-free plumbing check. Outputs are explicitly non-scientific."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.schemas import QAExample, SemanticPacket, SemanticUnit, write_jsonl
from src.representation.packet_store import PacketStore
from src.representation.token_counter import WhitespaceTokenizer, count_tokens
from src.search.best_nested_chain import best_nested_chain, structural_gap_rows
from src.search.exact_frontier import independent_frontier, write_csv
from src.search.exact_search import run_exact_search
from src.target.qwen_runner import CallableTargetRunner


def build_fixture(tokenizer: WhitespaceTokenizer) -> tuple[QAExample, list[SemanticPacket]]:
    texts = [
        "Aria was born in Northport.",
        "Northport is in the country Selene.",
        "The harbor opened in 1902.",
        "Mira painted the eastern station.",
        "The western station closed in 1980.",
    ]
    units = [SemanticUnit(i, text, i in (0, 1)) for i, text in enumerate(texts)]
    example = QAExample(
        "dry_run_0001",
        "synthetic_non_scientific",
        "In which country was Aria born?",
        "Selene",
        "\n\n".join(texts),
        units,
    )
    gist_residual = [
        ("Aria: born in Northport.", "No additional relevant detail."),
        ("Northport is in Selene.", "Selene is a country."),
        ("Harbor opened in 1902.", "The source concerns a harbor."),
        ("Mira painted a station.", "It was the eastern station."),
        ("Western station closed.", "Closure was in 1980."),
    ]
    packets = [
        SemanticPacket(
            example.example_id,
            index,
            source,
            gist,
            residual,
            count_tokens(tokenizer, source),
            count_tokens(tokenizer, gist),
            count_tokens(tokenizer, residual),
        )
        for index, (source, (gist, residual)) in enumerate(zip(texts, gist_residual))
    ]
    return example, packets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/dry_run")
    args = parser.parse_args()
    output = Path(args.output_dir)
    exact_dir = output / "exact_search"
    packet_dir = output / "packets"
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = WhitespaceTokenizer()
    example, packets = build_fixture(tokenizer)
    write_jsonl(output / "examples.jsonl", [example])
    PacketStore(packet_dir).put(example.example_id, packets)

    def answer(_question: str, context: str) -> str:
        if "Aria" in context and "Selene is a country." in context:
            return "Selene"
        if "Aria" in context and "Selene" in context:
            return "Selene uncertain"
        return "unknown"

    results = run_exact_search(
        example, packets, CallableTargetRunner(answer), tokenizer, batch_size=243
    )
    write_jsonl(exact_dir / f"{example.example_id}.jsonl", results)
    independent = independent_frontier(results)
    nested = best_nested_chain(results)
    write_csv(output / "exact_frontier.csv", independent)
    write_csv(output / "nested_frontier.csv", nested)
    write_csv(output / "structural_gap.csv", structural_gap_rows(independent, nested))
    print(f"NON-SCIENTIFIC dry run complete: {len(results)} states -> {output}")


if __name__ == "__main__":
    main()
