# FRONTIER-R1: RECOMP NQ Extractive Screen-64

## Qualification

This is a zero-shot, cross-domain baseline using the official public
`fangyuan/nq_extractive_compressor` checkpoint. The compressor was trained for
Natural Questions, not QAMPARI. It is therefore a valid frozen off-the-shelf
baseline, but its failure cannot establish that a QAMPARI-trained RECOMP model
would fail. The downstream Qwen3-8B Target remains frozen and no training was
performed.

Following the official implementation, the query and candidate sentences are
encoded together, token embeddings are mean pooled, and query--sentence dot
products provide extraction scores. Sentences are selected greedily under
actual Qwen3-8B token budgets and restored in original source order.

## Results

| Requested keep | Actual keep | Mean context tokens | Mean F1 | Paired delta F1 | 0.90 success | Repair / break |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 control | 1.0 | — | 0.9495 | — | 53/64 | — |
| 0.50 | 0.4984 | 402.0 | 0.5374 | -0.4121 | 1/64 | 0 / 52 |
| 0.25 | 0.2484 | 200.4 | 0.3315 | -0.6180 | 0/64 | 0 / 53 |
| 0.125 | 0.1240 | 100.1 | 0.2347 | -0.7148 | 0/64 | 0 / 53 |

## Decision

`STOP_RECOMP_EXTRACTIVE_AFTER_SCREEN64`

The off-the-shelf NQ checkpoint fails the frozen Screen-64 quality gate and is
not promoted to Design-256. The result does not authorize domain training.

## Reusable observation, without retraining

Sentence-level extraction is less destructive than the tested
LongLLMLingua-Qwen3-1.7B port at approximately 2x (0.5374 versus 0.4293 mean
F1), but it still loses almost the entire 0.90-success set. QAMPARI requires a
set of distributed answer-bearing sentences; a single query-sentence relevance
score does not protect set coverage. Any later extractive component would need
set-level coverage supervision rather than a direct transplant of this scorer.

## Preserved artifacts

- `compressed.jsonl`: all contexts, selected source units, scores, and costs.
- `target_outputs.jsonl`: all 192 fresh frozen-Target outputs.
- `summary.json`: aggregate threshold table and decision.
- `compress.log` and `target.log`: execution logs.

The run adds 192 Target calls and reuses the existing 64-query fresh original-
context control. Sealed sets were not read.
