# Frozen-Target baseline implementations

This package contains comparison methods and controls that are **not part of
the proposed progressive compression method**.

## Boundary

Baseline runners may use only:

- frozen input cohorts and cached V8 contexts as fixed comparison inputs;
- shared dataset readers and QAMPARI metrics;
- the frozen Target runner;
- official external checkpoints/packages or deterministic heuristics.

They must not import trainable policy heads from `src.model`, training code from
`src.training`, oracle search code from `src.search`, or experimental V17/V18
decision logic. A baseline may consume a frozen V8 output only when the arm is
explicitly named as a composition such as `v8_global`; it must retain a direct
original-context arm for attribution.

## Entrypoints

| Module | Role |
|---|---|
| `frontier_compression_r0` | frozen cohort and method-eligibility manifest |
| `frontier_target_interface_audit` | read-only frozen-Target interface check |
| `frontier_r1_hard_screen` | official LLMLingua-2 hard-compression screen |
| `frontier_r1_llmlingua2_adapted` | training-free mild-rate LLMLingua-2 adaptation |
| `frontier_r1_longllmlingua_port` | LongLLMLingua mechanism port |
| `frontier_r1_recomp_extractive` | official RECOMP NQ extractive zero-shot |
| `frontier_r1_recomp_abstractive` | official RECOMP NQ abstractive zero-shot |
| `frontier_r1_training_free_extractors` | BM25 and deterministic-random controls |

Run modules as `python -m src.baselines.<module>`. Immutable outputs remain
under their existing `results/v2_rank_then_cut/frontier_*` directories so that
paper links, manifests, and artifact hashes stay valid.
