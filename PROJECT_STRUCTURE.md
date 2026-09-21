# Project code boundaries

The repository separates the proposed method, shared infrastructure, external
baselines, and immutable experiment artifacts.

## Proposed method

- `src/model/`: learned components of the proposed method.
- `src/training/`: training entrypoints for those components.
- `src/representation/`: packet/state representations used by the method.
- `src/search/`: progressive reveal and exact-search machinery.
- `src/evaluation/v*.py`: versioned method evaluations and mechanism audits.
- `src/data/`: method datasets and supervision builders.

An experiment in these directories may change the proposed method. It must not
be described as an external baseline.

## External and training-free baselines

- `src/baselines/`: baseline protocols, official checkpoint adapters, and
  deterministic controls.
- `third_party/`: upstream source checkouts; never imported as proposed-method
  code and never modified to improve the proposed method silently.
- `results/v2_rank_then_cut/frontier_*`: immutable baseline outputs.

Baseline code may call shared readers, metrics, and the frozen Target runner.
It may not import the project's trainable policy/model/search implementation.
Compositions with a frozen V8 context must be explicitly labeled `v8_*` and
reported alongside an original-context arm.

## Shared infrastructure

- `src/target/`: frozen Target execution and answer parsing.
- `src/evaluation/qampari_metrics.py`: common task metric.
- `src/data/schemas.py`: common JSONL/schema utilities.
- `configs/`: frozen experiment contracts.

Shared infrastructure must not encode method-specific ranking or stopping
decisions. Both method and baseline code may depend on it.

## Results and paper reporting

Existing result directories are not moved during code reorganization because
reports and SHA-256 manifests refer to their current paths. Baseline results
are indexed in `results/v2_rank_then_cut/BASELINES.md`; the chronological method
record remains in `results/v2_rank_then_cut/RESULTS.md`.
