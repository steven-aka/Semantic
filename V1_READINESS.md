# V1 readiness handoff — intentionally stopped before training

## Current gate

The newest authoritative handoff is the independently verified QAMPARI M0
lossless development gate at
`results/m0_qampari/lossless_dev_gate_result.json`. It records
`complete=true`, `errors=[]`, and 30 examples × 729 = 21,870 exact states.
The structural bottleneck is resolved: 104/104 strict rate increases, 16
non-nested independent switches, 87.5% nested reuse, and 0.40 mean answer-atom
recall gain from 0.60 to 0.90. The only failed frozen check is
target–benchmark answerability measured by any-state
F1≥0.9 coverage: 25/30=83.33% versus the required 90%. Therefore
`scientific_gate_passed=false` and `decision=NO_GO_STOP_BEFORE_V1`.
A read-only failure audit found both eight-of-ten target enumeration errors and
evidence/label closure defects, so this must not be reported as compressor
failure or as a clean estimate of target-model capacity.

No QAMPARI test target inference, V1/QLoRA job, checkpoint, or project-owned
model process was started. Earlier V0.x results below are historical evidence,
not the current decision point.

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

1. Do not approve V1 from the M0 development result: the final conjunctive
   gate is NO-GO even though the representation and structural-signal checks
   now pass. Do not revive confidence/NLL proxies, weaken the frozen 90%
   capacity threshold, or inspect the locked test to repair this result.
2. Keep V0.2 as oracle supervision/diagnosis only. Keep V0.3 frozen as the
   single label-free BM25 result; do not retune it after observing metrics.
3. Freeze the exact-search artifacts and their hashes as the oracle target for
   selector/compressor supervision. Do not overwrite them during V1.
4. Stop numbered V0 repair attempts. The user-authorized next method study is
   the bounded M0 continuous-fidelity redesign specified in
   `FIDELITY_SIGNAL_BOTTLENECK_PLAN.md`. It must preregister and hash its fresh
   locked protocol before inference. Do not retroactively replace the V0.5
   primary gate with its attainable-breakpoint sensitivity result.
5. Define the V1 acceptance criteria and ablations before starting QLoRA.

Any proposed multi-hop, label-free selector must be registered as a new bounded
experiment with frozen splits and acceptance criteria. M0 method development
is now authorized, but V1 remains stopped; only a conjunctive M0 GO followed by
an explicit V1 authorization can change that status.

That bounded CPU retrieval screen has now been run once.  Its best development
policy, linked-title-pair retrieval, achieved only 75.65% on the 200-example
locked test versus the frozen 80% gate.  The R0 decision is
`STOP_RETHINK_N6`; its locked test must not be reused for further tuning.
That R0 recommendation has now been executed as Retrieval R1. Its frozen
semantic selector passed and `data/units/hotpot_v0_4_candidates.jsonl` exists.
V0.4 shows that further retrieval tuning is not the next action; the remaining
blocker is the sparsity/platform behavior of the fidelity signal itself.
