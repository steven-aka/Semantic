# FRONTIER-R1: LongLLMLingua Qwen3-1.7B Port Screen-64

## Qualification

This is a mechanism port, not an exact paper-checkpoint reproduction. It uses
the official `llmlingua==0.2.2` LongLLMLingua algorithm with a frozen local
Qwen3-1.7B compressor because the default Llama-2 compressor checkpoint is not
available in the project. The downstream Target is the canonical frozen
Qwen3-8B. No model was trained.

The compressor receives the query and all twelve original packets. It uses
LongLLMLingua query-conditioned ranking, canonical source ordering, context and
token filtering, and requested keep rates 0.50, 0.25, and 0.125. Costs are
measured with the Qwen3-8B tokenizer.

## Compatibility adaptation

LLMLingua 0.2.2 represents KV cache as a legacy list, while the installed
Transformers Qwen3 implementation requires `DynamicCache`. The first run
stopped after 15 rows at this interface mismatch. The saved adapter converts
the container to `DynamicCache` immediately before Qwen3 forward and converts
it back afterward. It does not alter logits, perplexity, selection, ordering,
or budgets. The partial run and failure log are preserved separately.

## Results

| Requested keep | Actual keep | Mean context tokens | Mean F1 | Paired delta F1 | 0.90 success | Repair / break |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 control | 1.0 | — | 0.9495 | — | 53/64 | — |
| 0.50 | 0.5057 | 403.2 | 0.4293 | -0.5201 | 1/64 | 0 / 52 |
| 0.25 | 0.2990 | 237.4 | 0.3091 | -0.6404 | 0/64 | 0 / 53 |
| 0.125 | 0.1400 | 114.5 | 0.1332 | -0.8163 | 0/64 | 0 / 53 |

## Decision

`STOP_LONGLMLINGUA_PORT_AFTER_SCREEN64`

The port fails the frozen quality gate by a wide margin and is not promoted to
Design-256. This result is evidence about this explicit Qwen3-1.7B port under
QAMPARI, not a claim about the original paper's default compressor.

## Reusable observation, without retraining

Query-conditioned perplexity alone does not preserve the dispersed answer set
needed by this task. At a comparable requested 0.50 rate, this port is also
worse than LLMLingua-2 on the same original contexts (0.4293 versus 0.5847 mean
F1). This supports retaining V8's learned packet-level coarse selection as a
distinct component if token-level compression is reconsidered later. It does
not authorize retraining or a new selector.

## Preserved artifacts

- `compressed.jsonl`: all 192 compressed contexts and measured costs.
- `target_outputs.jsonl`: every fresh Qwen3-8B output.
- `summary.json`: threshold table and frozen decision.
- `compress.log` and `target.log`: successful run logs.
- `compress_failed_legacy_cache.log` and
  `compressed_pre_adapter_partial.jsonl`: failed pre-adapter attempt.

The run adds 192 Target calls and reuses the already fresh 64-query full-context
control. Sealed sets were not read.
