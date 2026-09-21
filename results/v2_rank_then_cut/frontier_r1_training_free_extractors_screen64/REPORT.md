# FRONTIER-R1 Training-Free Extractive Baselines (Screen-64)

## Protocol

- Cohort: the same deterministic 64-query design screen used by the other
  frozen-Target baselines.
- Source: the complete original 12-packet context.
- Target: frozen `Qwen3-8B`, greedy decoding, fresh paired outputs.
- Methods: sentence-level BM25 ranking and a deterministic random ranking.
- Budgets: requested keep rates 0.50, 0.25, and 0.125, enforced with the
  Qwen3-8B tokenizer. Selected sentences are rendered in original source order.
- Training: none. No development, confirmation, or other sealed set was read.

The random arm is seeded from the example ID and is reproducible. BM25 uses
only the query and candidate sentence text. These are simple frozen-Target
controls rather than claimed reproductions of a particular paper.

## Results

The fresh uncompressed control has mean F1 **0.9495** and **53/64** successes
at the 0.90 threshold.

| Method | Requested keep | Actual keep | Mean tokens | Mean F1 | 0.90 success | Repair / break |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.50 | 0.4984 | 401.95 | 0.5541 | 7/64 | 1 / 47 |
| BM25 | 0.25 | 0.2486 | 200.48 | 0.3465 | 1/64 | 0 / 52 |
| BM25 | 0.125 | 0.1237 | 99.83 | 0.2208 | 0/64 | 0 / 53 |
| Random | 0.50 | 0.4971 | 400.86 | 0.6008 | 3/64 | 0 / 50 |
| Random | 0.25 | 0.2478 | 200.08 | 0.3984 | 0/64 | 0 / 53 |
| Random | 0.125 | 0.1234 | 99.72 | 0.2474 | 0/64 | 0 / 53 |

At all three matched budgets, deterministic random extraction has higher mean
F1 than BM25, although neither method preserves the high-fidelity contract.
BM25's single 0.90 repair at 0.50 keep is outweighed by 47 breaks.

## Decision and interpretation

Decision: **`SAVE_AS_PAPER_BASELINES_NO_ESCALATION`**.

Both baselines are retained as paper controls, with no larger evaluation or
training triggered. The result supports a narrow design observation: lexical
query relevance is not an adequate proxy for the distributed evidence coverage
needed by this QAMPARI Target. It does not show that every sparse retrieval or
domain-trained extractor must fail.

## Reproducibility artifacts

- `compressed.jsonl`: every selected context and its exact token count.
- `target_outputs.jsonl`: every fresh Target output and per-query F1.
- `summary.json`: aggregate metrics and the frozen decision.
- `run.log`: execution log.
- `artifact_manifest.json`: SHA-256 hashes and sizes of core artifacts.

Run command:

```bash
source scripts/cuda_env.sh
CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 .venv/bin/python -u \
  -m src.baselines.frontier_r1_training_free_extractors --mode all
```
