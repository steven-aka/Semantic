# Frozen-Target baseline index

This file indexes comparison methods only. Proposed-method experiments remain
in `RESULTS.md` and the versioned V8/V17/V18 result directories.

| Baseline | Type | Result | Status |
|---|---|---|---|
| LLMLingua-2 hard screen | official checkpoint, cross-domain | `frontier_r1_hard_screen64` | saved negative baseline |
| LLMLingua-2 mild adaptation | official checkpoint, no training | `frontier_r1_llmlingua2_adapted_screen64` | strongest point: V8-global rate 0.95 |
| LongLLMLingua | official algorithm, Qwen3-1.7B mechanism port | `frontier_r1_longllmlingua_qwen17b_screen64` | qualified negative port |
| RECOMP extractive | official NQ checkpoint, zero-shot | `frontier_r1_recomp_nq_extractive_screen64` | saved negative baseline |
| RECOMP extractive, QAMPARI-adapted | trained compressor; Target frozen | `baseline_recomp_qampari_extractive` | retained learned baseline |
| RECOMP abstractive | official NQ checkpoint, zero-shot | `frontier_r1_recomp_nq_abstractive_screen64` | saved negative baseline |
| RECOMP abstractive, QAMPARI-adapted | Qwen3-14B teacher-distilled T5 compressor; Target frozen | `baseline_recomp_qampari_abstractive` | training launched after 1906/2032 summaries passed frozen filter |
| BM25 sentence extraction | deterministic, no training | `frontier_r1_training_free_extractors_screen64` | saved control |
| Random sentence extraction | deterministic, no training | `frontier_r1_training_free_extractors_screen64` | saved control |
| xRAG | interface eligibility audit | `frontier_r2_xrag_eligibility_audit` | retained interface audit |
| xRAG-style Qwen3 projector | frozen Qwen3-8B and frozen QAMPARI retriever; projector only | `baseline_xrag_qwen3_projector` | gradient/memory preflight passed; training launched |
| 500xCompressor | asset/interface eligibility audit | `frontier_r2_500x_eligibility_audit` | public checkpoint unavailable; Qwen port requires training |

The `v8_global` LLMLingua-2 arm is explicitly a composition baseline. V8 is
frozen before compression; LLMLingua-2 is not trained on project data. Direct
original-context arms are retained to separate V8 coarse selection from token
compression.
