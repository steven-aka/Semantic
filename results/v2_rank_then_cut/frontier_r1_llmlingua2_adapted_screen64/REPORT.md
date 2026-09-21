# FRONTIER-R1 LLMLingua-2 Frozen-Target Adaptation (Screen-64)

## Goal and protocol

This experiment strengthens the LLMLingua-2 baseline without training or
changing the frozen Qwen3-8B Target. It evaluates the official MeetingBank
checkpoint at mild requested keep rates 0.60, 0.70, 0.80, 0.90, and 0.95.

Two global-compression arms separate the effects of coarse selection and token
pruning:

- `v8_global`: V8 depth-10 packet selection followed by one global
  LLMLingua-2 pass;
- `original_global`: one global LLMLingua-2 pass over all twelve packets.

The question and Target output instruction are never compressed. Results use
actual Qwen3-8B token counts. The 64 queries are the same design-exposed screen
as the earlier baseline; no sealed set was read.

## Results

| Arm | Requested keep | Actual keep | Tokens | Mean F1 | 0.90 success | Repair / break |
|---|---:|---:|---:|---:|---:|---:|
| V8 control | 1.00 | 1.00 | 677.36 | 0.9353 | 56/64 | — |
| v8_global | 0.60 | 0.6062 | 413.22 | 0.7098 | 30/64 | 2 / 28 |
| v8_global | 0.70 | 0.6985 | 475.45 | 0.8620 | 40/64 | 2 / 18 |
| v8_global | 0.80 | 0.7995 | 544.28 | 0.8969 | 50/64 | 4 / 10 |
| v8_global | 0.90 | 0.8965 | 609.58 | 0.9022 | 51/64 | 1 / 6 |
| **v8_global** | **0.95** | **0.9333** | **634.11** | **0.9271** | **53/64** | **2 / 5** |
| Original control | 1.00 | 1.00 | 806.20 | 0.9495 | 53/64 | — |
| original_global | 0.60 | 0.6071 | 491.50 | 0.7704 | 31/64 | 2 / 24 |
| original_global | 0.70 | 0.7007 | 566.39 | 0.8718 | 44/64 | 3 / 12 |
| original_global | 0.80 | 0.8012 | 648.09 | 0.8765 | 42/64 | 3 / 14 |
| original_global | 0.90 | 0.8966 | 725.06 | 0.9123 | 52/64 | 3 / 4 |
| original_global | 0.95 | 0.9335 | 754.47 | 0.8987 | 52/64 | 4 / 5 |

The strongest adapted point is V8 plus global LLMLingua-2 at requested 0.95.
It removes 43.25 Qwen3 context tokens on average (6.39% relative to V8), while
mean F1 decreases by 0.0081 and 0.90 success decreases by three queries. Its
success counts at thresholds 0.60/0.70/0.80/0.90/0.95 are
61/60/59/53/46, versus V8's 61/60/60/56/47.

## Interpretation and decision

Decision: **`RETAIN_V8_GLOBAL_RATE095_AS_STRONG_TRAINING_FREE_BASELINE`**.

The adapted configuration is substantially stronger than the original hard
compression screen and should replace the 0.50 point when presenting the best
LLMLingua-2 operating point. It is not lossless and does not dominate V8: the
6.39% extra token reduction costs 0.0081 mean F1 and three 0.90 successes.

Quality is not monotonic in requested keep rate for the original-context arm,
which shows that deletion identity and Target response matter in addition to
token count. The MeetingBank encoder also warns that some inputs exceed its
nominal 512-token sequence limit. Thus this remains a cross-domain frozen
checkpoint baseline, not a native QAMPARI-trained reproduction.

## Artifacts

`compressed.jsonl`, `target_outputs.jsonl`, `summary.json`, `run.log`, and
`artifact_manifest.json` preserve all 640 contexts, fresh Target outputs,
aggregate metrics, logs, and hashes.
