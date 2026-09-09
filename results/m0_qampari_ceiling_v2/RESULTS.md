# M0 QAMPARI ceiling-v2: evidence-closed conditional benchmark

Status: **GO for V1 readiness review; V1 training has not started.**

Date: 2026-09-09

## Outcome

The original lossless M0 result remains a valid NO-GO on its original
population: only 25/30 examples reached any-state alias-aware list F1 >= 0.9.
A counterfactual audit showed that this mixed target enumeration failures with
evidence/label closure defects, so it was not a clean ceiling for the semantic
compression experiment.

Ceiling-v2 fixes the benchmark contract rather than the compression objective
or the frozen gate. Each example has exactly ten source-certified answer
atoms. Certification requires MediaWiki display/title alias closure,
same-article proof provenance, question-relation support, explicit numeric
bound checks when applicable, and retention of the complete official proof.
The representation remains six units with additive states `{0,1,2}`, exact
`3^6=729` search, Qwen3-8B hard list F1, the unchanged five fidelity anchors,
and the same conjunctive thresholds.

The actual all-state-2 representation was evaluated before exact search. Of
98 certified candidates, 61 (62.24%) satisfied the frozen target-answerability
precondition: full F1 >= 0.90, empty F1 <= 0.20, context gain >= 0.70, and no
length stop. This screening yield is part of the result. The GO claim applies
to this evidence-certified, target-answerable population; it is not a claim
that 90% of arbitrary QAMPARI questions are answerable.

The first 30 eligible examples formed development. Eligible positions 30--59
formed a disjoint heldout exact test. Its config and all input hashes were
frozen after development GO and before any heldout exact output. Official
QAMPARI test target outputs remain untouched.

## Frozen results

| Check | Development | Disjoint heldout | Threshold |
|---|---:|---:|---:|
| Complete examples | 30 | 30 | >=30 |
| Exact states | 21,870 | 21,870 | 729/example |
| Evidence certificates | 30/30 | 30/30 | all |
| Ceiling precondition | 30/30 | 30/30 | all |
| Strict rate increases | 109/110 = 99.09% | 111/111 = 100% | >=20% |
| Non-nested independent switches | 16 | 9 | >=1 |
| Nested reuse, any optimal tie | 85.32% | 93.69% | >=80% |
| Mean atom-recall gain, 0.60 to 0.90 | 0.3967 | 0.4000 | >=0.15 |
| Atom-recall decline fraction | 0 | 0 | <=0.10 |
| Full-state F1 >= 0.8 | 27/30 = 90% | 30/30 = 100% | >=90% |
| Any-state F1 >= 0.9 | 30/30 = 100% | 30/30 = 100% | >=90% |
| Mean normalized structural gap | 0.00460 | 0.00328 | <=0.10 |
| Normalized structural-gap p90 | 0.00914 | 0 | <=0.20 |

Both independent verifications report `complete=true`, `errors=[]`, and every
conjunctive check `true`. Development and heldout example-ID overlap is zero.
The heldout evaluator inherits a historical generic result label saying
`READY_TO_FREEZE_FRESH_LOCKED_M0_NOT_V1`; the already-frozen heldout config's
decision rule is authoritative: passing permits V1 readiness review but does
not start training.

## Interpretation and boundary

The former 83.3% failure is resolved for the intended structural experiment
by separating target-answerability from compression quality and making it an
explicit, hashed population precondition. The heldout result shows that the
rate/fidelity and structural-gap behavior transfers to a disjoint subset of
that conditional population. No threshold was weakened and no post-heldout
tuning was performed.

This benchmark construction reads official answer proofs and is therefore an
oracle diagnostic benchmark, not an unbiased retrieval benchmark. Its purpose
is to provide trustworthy exact-search supervision for the next compression
stage. Generalization to label-free natural-document selection, official
QAMPARI test, other target models, and learned compression remains future
work.

## Authoritative artifacts

- Development config: `configs/m0_qampari_ceiling_v2_dev_gate.json`
- Heldout config: `configs/m0_qampari_ceiling_v2_heldout_gate.json`
- Development result: `results/m0_qampari_ceiling_v2/dev_gate_result.json`
- Heldout result: `results/m0_qampari_ceiling_v2/heldout_gate_result.json`
- Conditional population baseline:
  `results/m0_qampari_ceiling_v2/high_state_candidates98.jsonl`
- Reverification: `bash scripts/20_verify_m0_qampari_ceiling_v2.sh`

The chain is intentionally stopped at the V1 boundary. No QLoRA job,
checkpoint, or V1 result was created.
