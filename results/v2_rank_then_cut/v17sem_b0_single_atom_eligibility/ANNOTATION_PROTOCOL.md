# SEM-B0R AI-assisted single-atom feasibility protocol

The originally planned two-human reliability study is unavailable for this
run (`HUMAN_IAA_NOT_AVAILABLE`).  `annotator_a.jsonl` is therefore an
AI-assisted primary review followed by an adversarial error audit by the same
reviewer.  It is a mechanism/feasibility annotation, not independent human
agreement evidence.  `annotator_b.jsonl` remains blank, Cohen's kappa is N/A,
and the two-human formal gate is not claimed.

Annotate `audit_items.jsonl` independently in `annotator_a.jsonl` or
`annotator_b.jsonl`. Do not read the other form, SEM-A0B generations, gold
answers, Qwen3-8B outputs, fidelity scores, or prior action outcomes.

Mark `ELIGIBLE` only when an exact contiguous quote supports every part of an
affirmative fact of at most 16 Qwen3-8B tokens; the fact must be query-relevant
and novel relative to S6. Supply relevance and novelty reasons. Otherwise use
`ABSTAIN` with one frozen reason: `NO_RELEVANT_FACT`, `ALREADY_IN_S6`,
`RELATION_NOT_CLOSED`, `CANNOT_FIT_16_TOKENS`, or `AMBIGUOUS_GROUNDING`.

Validate and summarize the single review with:

```bash
.venv/bin/python -m src.evaluation.v17sem_b0_single_atom_eligibility --single-review
```

If two genuinely independent human annotations later become available, run:

```bash
.venv/bin/python -m src.evaluation.v17sem_b0_single_atom_eligibility --score
```

The reliability gate is raw agreement >=0.85 and Cohen kappa >=0.70. Final GO
also requires >=24/64 adjudicated ELIGIBLE and >=20 direct-agreement ELIGIBLE;
<16 is STOP and 16–23 is a gray-zone mechanism audit. AI-generated duplicate
labels are not a substitute for two independent human annotations.
