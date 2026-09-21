# SEM-A0: existing semantic packet audit

This zero-call audit asks whether the repository's existing Qwen3-14B semantic packets can directly instantiate the proposed semantic action-space positive control. The answer is **no**; this is a protocol mismatch, not evidence that semantic representation itself fails.

The two versioned teacher-generated assets contain 60 unique old development examples (360 packet rows) and have **zero overlap with the canonical train1421 cohort**. `PacketGenerator.generate_for_units` receives a `QAExample`, but its generation prompt is built only from each unit's source text, source token budget, source document identities, and source-derived number allowlist. It does not pass the question, gold answers, or Qwen3-8B outcomes. Thus the existing generator is source-conditioned and gold-clean, but **not query-conditioned**.

The assets do preserve the full source beside every gist/residual, and current validation hard-checks gist length, unexpected numbers, and preservation of document titles. That is useful provenance. It is not a full grounding certificate: possible unseen entities are warnings rather than errors, no source-span offsets are stored, and no entailment check is recorded. One historical generated asset has 151/180 rows valid under the current validator; the later closure asset has 180/180. The generated gists average roughly 35% of their source length under the Qwen3-8B tokenizer. Existing metadata records model/configuration, but not per-packet formatted prompt tokens, generated tokens, or latency, so online compressor cost cannot be reconstructed at the required granularity.

Decision: **`STOP_EXISTING_PACKETS_AS_DIRECT_SEM_A1_INPUT`**. They may serve as a historical query-agnostic baseline, but using them as the proposed query-conditioned SEM-A1 arm would test a different hypothesis and mix old development data with the canonical train lineage. Before Target calls, a new positive-control contract must freeze query+source-only inputs, prohibit gold and Target outcomes, store exact source provenance and execution costs, validate grounding, and pre-register a quantitative improvement over M2 rather than the vague criterion “noticeably denser.”

[Audit script](../../../src/evaluation/v17sem_a0_existing_packet_audit.py) and [summary](summary.json).

The replacement contract is frozen separately in
[SEM-A0B](../../../configs/v17sem_a0b_query_conditioned_positive_control.json).
It is not yet an executed experiment: it requires exact quote provenance,
hard entity/number checks, a blinded grounding audit, per-call cost logging,
and a same-query raw-versus-semantic comparison before SEM-A2 is considered.
