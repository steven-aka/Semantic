# FRONTIER-R1: LLMLingua-2 Screen-64

## Purpose and status

This is a saved paper-baseline screen for the official
`microsoft/llmlingua-2-xlm-roberta-large-meetingbank` compressor with the
canonical frozen Qwen3-8B Target. It uses 64 deterministic, design-exposed
CANON-P0 training queries. No lineage-clean 581, internal, development, or
confirmation set was read.

The result is a valid negative baseline. It is not promoted to Design-256 and
does not authorize retraining or integration into V8.

## Arms

- `original_native`: global LLMLingua-2 compression of all 12 original packets.
- `native`: global compression of the canonical-order context selected by V8
  at depth 10. The legacy name is retained in JSON for reproducibility.
- `v8_wrapped`: each of the ten V8-selected packets is compressed separately,
  then rendered in canonical source order.

All compressed arms use requested keep rates 0.50, 0.25, and 0.125. Actual
Qwen3-8B tokenizer ratios are reported, rather than assuming requested and
realized ratios are equal.

## Results

| Arm | Requested keep | Actual keep | Mean context tokens | Mean F1 | Paired delta F1 | 0.90 success | Repair / break |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original full context | 1.0 | 1.0 | — | 0.9495 | — | 53/64 | — |
| original_native | 0.50 | 0.5148 | 417.8 | 0.5847 | -0.3505 | 13/64 | 1 / 41 |
| original_native | 0.25 | 0.2785 | 225.9 | 0.3742 | -0.5611 | 1/64 | 0 / 52 |
| original_native | 0.125 | 0.1487 | 120.5 | 0.2893 | -0.6459 | 1/64 | 0 / 52 |
| Fresh V8 depth-10 | 1.0 | 1.0 | — | 0.9353 | — | 56/64 | — |
| native | 0.50 | 0.5136 | 351.3 | 0.6579 | -0.2774 | 22/64 | 0 / 34 |
| native | 0.25 | 0.2797 | 191.2 | 0.4104 | -0.5248 | 5/64 | 0 / 51 |
| native | 0.125 | 0.1500 | 101.9 | 0.3245 | -0.6108 | 1/64 | 0 / 55 |
| v8_wrapped | 0.50 | 0.5176 | 353.6 | 0.5811 | -0.3541 | 22/64 | 2 / 36 |
| v8_wrapped | 0.25 | 0.2933 | 199.2 | 0.4285 | -0.5068 | 6/64 | 0 / 50 |
| v8_wrapped | 0.125 | 0.1550 | 104.2 | 0.1845 | -0.7507 | 3/64 | 0 / 53 |

Fresh V8 depth-10 F1 (0.93525) closely reproduces its historical cached F1
(0.93509). The large compressed-quality losses therefore cannot be explained
by cache/runtime drift. The complete threshold table is in `summary.json`.

## Decision

`STOP_LLMLINGUA2_AFTER_SCREEN64`

No arm passes the frozen screen gate at an actual keep rate at or below 0.25.
Losses are already substantial near 2x compression, so a larger Design-256 run
would spend Target calls without a plausible path to the quality gate.

## Reusable observations, without retraining

These are mechanism observations to retain for later design work, not evidence
for immediately changing or retraining V8:

1. Global compression of the V8-selected context is consistently better than
   packetwise compression at roughly 2x, suggesting cross-packet context helps
   token selection.
2. Compressing the full original context is worse than first applying V8 and
   then globally compressing at roughly 2x. V8 therefore supplies useful
   coarse evidence filtering before token pruning.
3. Requested keep rates systematically understate realized Qwen3 token ratios;
   future baselines must match on realized Target-token cost.
4. Generic token deletion destroys dispersed answer evidence on QAMPARI. If a
   later method adds within-packet compression, it should preserve answer-set
   coverage explicitly and be evaluated as a separate frozen protocol.

## Preserved artifacts

- `compressed.jsonl`: all 576 compressed contexts and measured token counts.
- `target_outputs.jsonl`: all 576 per-query compressed-context Target outputs.
- `fresh_baseline_outputs.jsonl`: 64 fresh V8 depth-10 controls.
- `fresh_original_baseline_outputs.jsonl`: 64 fresh full-context controls.
- `summary.json`: aggregate metrics, thresholds, gate, and decision.
- `fresh_baseline_run.log`, `original_native_compress.log`, and
  `original_native_target.log`: execution logs.

Total fresh frozen-Target calls represented in the final result: 704.
