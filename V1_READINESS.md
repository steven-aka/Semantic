# V1/V2 readiness handoff

## Latest authoritative status (2026-09-10)

The atomic M0 scientific prerequisite remains passed. The first learned V1
absolute-threshold policy was subsequently trained and evaluated, and is a
frozen **NO-GO**: on its one-shot fresh30 evaluation it obtained 86.39% active
contract success, 50% all-active-trajectory success, and 0.35390 feasible
normalized policy gap. Its calibration was invalid. These results do not undo
M0; they show that the learned threshold policy did not recover M0's nested
oracle. See `results/v1_atomic/RESULTS.md`.

The consumed fresh30 has now been used once more for diagnosis only, with no
new training or heldout claim. The four-way ordering/cutoff decomposition shows
that learned-order + oracle-cutoff reaches 95.24% anchor contract success and
0.03859 regret, whereas oracle-order + learned-count reaches only 53.74% and
0.18435. Cutoff is therefore the dominant failure; ordering is secondary but
not yet sufficient at the highest fidelity. See
`results/v1_atomic/RANK_CUT_DIAGNOSTIC.md`.

A new protocol, `configs/v2_rank_then_cut_hypothesis.json`, was frozen after
that consumed-data diagnosis and before inference on the new candidate pool.
It preserves lossless atomic packets, Qwen3-1.7B as the learned backbone, the
frozen Qwen3-8B Target, and nested representations, but replaces absolute
threshold regression with set-valued near-optimal partial-order ranking. The
official QAMPARI train pool was partitioned target-blind into disjoint training,
ranking-validation, calibration, and final-test roles.

V2 is currently constructing exact 4096-state oracle lattices for the frozen
5000-candidate pool. **No V2 ranker has been trained, the cutoff head has not
been implemented, and calibration/final test remain untouched.** Ranking is
trained first on the frozen 500/1000/2000 learning curve only after exact
construction and split eligibility checks finish. A conjunctive rank-only gate
on train2000, evaluated as learned order + global oracle cutoff, must pass before
cutoff work is permitted. FULL remains a separate lossless fallback and is not
the endpoint of the learned control trajectory.

Current decision:

```text
M0_ATOMIC_GATE=PASS
V1_ABSOLUTE_THRESHOLD_POLICY=NO_GO_FROZEN
V2_RANK_THEN_CUT=ORACLE_CONSTRUCTION_IN_PROGRESS
V2_TRAINING=NOT_STARTED
V2_CUTOFF_HEAD=NOT_IMPLEMENTED_PENDING_RANKING_GATE
```

## Current gate

The newest authoritative handoff is
`results/m0_qampari_ceiling_v2/final_readiness.json`. The evidence-closed
ceiling-v2 development and disjoint heldout exact runs each contain 30 examples
and 21,870 states, report `complete=true`, `errors=[]`, and pass every frozen
conjunctive check. Both obtain 30/30 any-state F1 >= 0.9. Heldout also obtains
111/111 strict rate increases, nine non-nested switches, 93.69% nested reuse,
0.40 mean answer-atom gain, and mean normalized structural gap 0.00328.

The earlier `results/m0_qampari/lossless_dev_gate_result.json` remains an
unaltered historical NO-GO at 25/30. Ceiling-v2 did not weaken that gate. It
removed a benchmark confound by requiring source-certified answer atoms and a
hashed all-state-2 target-answerability precondition before structural study.
Of 98 certified candidates, 61 qualified (62.24%); the GO therefore applies to
that conditional population, not arbitrary QAMPARI questions.

The paragraphs below preserve the pre-V1 handoff as historical provenance.
Statements there that V1 had not started were true at that handoff but are
superseded by the latest authoritative status above.

V0.2 is complete and end-to-end verified. `results/v0_2/verification.json`
records `complete=true`, `errors=[]`, the expected artifact counts, primary and
attainable-grid diagnostics, and SHA-256 hashes for the key handoff files.

No V1/QLoRA job, checkpoint, result file, or project-owned model process is
active. The repository is intentionally stopped at the implementation plan's
human review gate.

Retrieval R1 has passed on a fresh 200-example locked test (89.40% fact
coverage, 1.5% fallback), and V0.4 completed all 30 examples, 180 packets, and
21,870 exact states. Its independent verifier is clean, but the scientific
gate remains NO-GO: strict primary-grid rate transitions are 4.88%, nested
reuse is 50%, and mean 0.60-to-0.90 fact-recall gain is 0.0583. The current
authoritative historical handoff for that run is `results/v0_4/gate_result.json`.

V0.5 has now executed the final plan-aligned repair by selecting 30/30
full-context-answerable examples from an unused frozen R1 pool. It raises
nested reuse to 100% and coverage to 93.06%, but still fails rate transitions
(6.10% versus 20%) and fact gain (0.0833 versus 0.15). The current authoritative
handoff is `results/v0_5/gate_result.json`: `complete=true`, `errors=[]`,
`scientific_gate_passed=false`, `decision=NO_GO_STOP_BEFORE_V1`.

V0.3 has now completed the approved sentence-atomic + label-free BM25 test on
30 previously unused examples.  Its artifact gate is clean, but its
preregistered scientific gate failed (`results/v0_3/gate_result.json`): fact
coverage 70.83% < 80%, strict primary-grid rate transitions 13.89% < 20%, and
mean fact-recall gain 0.111 < 0.15.  The current decision is therefore
`NO_GO_STOP_BEFORE_V1`; no human approval can treat this V0.3 result as a pass
without explicitly defining a new experiment.

## Reproduce or resume V0.2

Run the full chain from raw-data reconstruction through final verification:

```bash
CUDA_VISIBLE_DEVICES=<physical_gpu> bash scripts/09_run_v0_2.sh all
```

The `baseline`, `packets`, and `target` stages validate their caches before
loading a model. A complete rerun therefore uses CPU-only reconstruction and
analysis; an interrupted or stale artifact causes only the required GPU stage
or example to run again. JSONL and metadata writes use same-directory temporary
files followed by atomic replacement.

Individual stages are available as:

```bash
bash scripts/09_run_v0_2.sh prepare
bash scripts/09_run_v0_2.sh baseline
bash scripts/09_run_v0_2.sh packets
bash scripts/09_run_v0_2.sh target
bash scripts/09_run_v0_2.sh analyze
bash scripts/09_run_v0_2.sh verify
```

The final verifier independently checks raw gold mapping, baseline metrics and
eligibility, packet/source/token consistency, fact judgments, all 12,393 raw
and rescored states, both exact/nested frontiers, structural-gap arithmetic,
diagnostics, metadata paths, tests, compilation, and shell syntax.

## Frozen V0.2 invariants

- Teacher: local frozen Qwen3-14B BF16, greedy decoding.
- Target/judge: local frozen Qwen3-8B BF16, greedy decoding.
- Representation states: exactly `{0: omit, 1: gist, 2: gist+residual}`.
- Fidelity: `min(answer_f1, content-aware gold supporting-fact recall)`.
- Controlled state space: six units, exactly `3^6=729` states per example.
- Primary fidelity grid: `{0.60,0.70,0.80,0.90,0.95}`; attainable breakpoints
  remain a separately labelled sensitivity analysis.
- V0.2 fact-atomic segmentation is a gold-aware oracle stress test and must not
  be reported as an unbiased end-to-end benchmark.

## Decisions required before V1

1. The M0 scientific prerequisite is now satisfied on the explicitly scoped
   evidence-certified, target-answerable population. Review and freeze V1
   training/evaluation criteria before starting; do not reinterpret this as a
   general QAMPARI or retrieval result.
2. Keep V0.2 as oracle supervision/diagnosis only. Keep V0.3 frozen as the
   single label-free BM25 result; do not retune it after observing metrics.
3. Freeze the exact-search artifacts and their hashes as the oracle target for
   selector/compressor supervision. Do not overwrite them during V1.
4. Stop numbered V0/M0 repair attempts. Preserve the old NO-GO and both
   ceiling-v2 exact trees as immutable provenance/supervision. Do not
   retroactively replace any earlier primary result.
5. Define the V1 acceptance criteria and ablations before starting QLoRA.

Any proposed multi-hop, label-free selector remains a separate experiment with
frozen splits and acceptance criteria. The conjunctive M0 prerequisite is now
GO; V1 remains stopped until its own protocol is frozen and explicitly
started.

That bounded CPU retrieval screen has now been run once.  Its best development
policy, linked-title-pair retrieval, achieved only 75.65% on the 200-example
locked test versus the frozen 80% gate.  The R0 decision is
`STOP_RETHINK_N6`; its locked test must not be reused for further tuning.
That R0 recommendation has now been executed as Retrieval R1. Its frozen
semantic selector passed and `data/units/hotpot_v0_4_candidates.jsonl` exists.
V0.4 shows that further retrieval tuning is not the next action; the remaining
blocker is the sparsity/platform behavior of the fidelity signal itself.
