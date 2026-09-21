# SEM-B0 single-atom eligibility ceiling

The frozen zero-Target audit package contains the next 64 SHA-ordered
M0B-eligible canonical train queries after SEM-A0B, excluding recent action
cohorts. Each item exposes only question, V8 S6, and the frozen rank10 atom.
Gold, Target outputs, fidelity, action outcomes, and SEM-A0B generations were
not loaded. Its immutable content hash is
`04509a5b457f240ac02cd451dff9afbfd3c9b12eefa60927e70444badfce9218`.

Atoms range from 11 to 48 Qwen3-8B tokens (median 32.5, mean 31.22); only
10/64 raw atoms are <=16 tokens. Length does not determine eligibility.

Status: **`AWAITING_TWO_INDEPENDENT_HUMAN_ANNOTATIONS`**. The same AI cannot
scientifically impersonate two blinded reviewers. Until both forms pass the
frozen agreement and coverage gates, there is no eligibility estimate and no
authorization for teacher or Target calls.

[Protocol](../../../configs/v17sem_b0_single_atom_eligibility.json),
[builder/scorer](../../../src/evaluation/v17sem_b0_single_atom_eligibility.py),
[items](audit_items.jsonl), [instructions](ANNOTATION_PROTOCOL.md), and
[manifest](manifest.json).
