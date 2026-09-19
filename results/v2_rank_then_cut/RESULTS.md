# V2 Rank-then-Cut status

Status: **V8 one-run sequential policy completed and stopped at the consumed-development gate**

Date: 2026-09-17

## Why this protocol exists

The failed V1 absolute-threshold policy did not establish that the nested
representation was inadequate. On the already consumed fresh30, learned order
with a global oracle cutoff recovered 140/147 active contracts (95.24%), while
oracle order constrained to the learned counts recovered only 79/147 (53.74%).
The learned count/cutoff scale was the dominant failure. A 0.5% near-optimal
chain slack also made 26.57% of packet-anchor memberships ambiguous, showing
that a single tie-broken oracle chain supplies unjustified exact-order labels.

V2 therefore freezes the scientific object as query-conditioned packet
priority under a fidelity-constrained nested-prefix objective. It learns a
partial order from all identifiable relations in a near-optimal chain set,
then—only if ranking passes its own gate—learns a separate monotone cutoff.

## Frozen boundaries

- Representation: twelve source-preserving lossless binary packets.
- Learned backbone: Qwen3-1.7B NF4 QLoRA; no larger model and no RL.
- Target: frozen Qwen3-8B, greedy, thinking disabled.
- Exact lattice: all 4096 packet subsets per example.
- Primary anchors: 0.60, 0.70, 0.80, 0.90, 0.95.
- Primary near-optimal slack: 0.005, normalized by active anchors times
  full-state tokens; ambiguous pairs are not labeled.
- Eligibility: maximum fidelity over any exact state is at least 0.90. Full
  context is deliberately not an eligibility filter because added evidence can
  be harmful.
- Training curve: nested 500/1000/2000 sets, three frozen seeds.
- FULL: independent lossless reconstruction fallback; control 1.0 never means
  reveal all packets.

The complete preregistration and implementation/data hashes are in
`configs/v2_rank_then_cut_hypothesis.json`. A pre-training mechanical amendment,
`configs/v2_rank_then_cut_pretraining_amendment1.json`, freezes the nested
learning-curve builder and the independent three-seed gate checker. It changes
no hypothesis, threshold, data role, Target inference, model, or metric; it
prevents manual set construction and ensures the already-preregistered
highest-active-anchor check cannot be omitted.

Pre-training amendment 2,
`configs/v2_rank_then_cut_pretraining_amendment2.json`, records the only
zero-identifiable-pair row found in ranking-validation300. That row remains in
the frozen 300-example contract-evaluation population but contributes no
undefined RankNet loss. No example is replaced, dropped from contract
evaluation, or assigned an arbitrary pair label. Train500/1000/2000 contain no
zero-pair rows. This clarification was frozen before any V2 ranking training.

## Current data and execution

The target-blind candidate pool contains 5000 certified QAMPARI-train examples
and 60,000 lossless packets. Its frozen hashes are:

```text
examples = cce862a91fc2562ac6cf8c456ce663c4bcf0e4c24b66fb67e39f8ad43345be04
annotations = 71c50309631b488b252024cc114ff72c1369457166ec85ab2b1cd1c93c0f17b3
packet tree = d0df345cdf13191a64ea4affc697db6df5d6c73d9bc96a88bc90d6dd31b33364
protocol config = 2376fd274372ac6c595e1cfeb30a13c36dc23d72c302fcec308f89563e7aeb79
pre-training amendment 1 = 00e875df1306d049074251da77ab263c7d231e3736be97f52c8a51da927f7a21
pre-training amendment 2 = ed84f603809f82a1da68b57d3c7153914a26564ef230ee10828114b1dfc2ef7c
```

Exact Target inference is defined by six deterministic scientific shards in
`results/v2_rank_then_cut/candidates5000_exact`. At the 2026-09-13 09:07 CST
snapshot, shards 0, 2, 3, and 5 were complete, shard 1 was at 802/834, and the
aggregate was 4135/5000 (82.70%). Every worker uses the same Qwen3-8B Target
and a 0.45 vLLM memory cap. Physical GPU4 is explicitly excluded by operator
instruction.

The shard-4 waiting scheduler did not launch between September 10 and 13
because its `nvidia-smi` CSV reader failed to split the comma-delimited GPU and
free-memory fields. This was an operational scheduling defect, not an
inference or data error. The parser was corrected before shard 4 produced any
example output. To remove the resulting tail without changing its membership,
the original set `index mod 6 = 4` was partitioned into the mutually disjoint
sets `index mod 18 = 4, 10, 16` (278, 278, and 277 examples). Their union is
exactly the original 833-example shard. These workers were started on physical
GPUs 1, 3, and 5 respectively; GPU4 remains unused.

The first attempt
at shard 0 used a 0.60 cap on GPU4 and failed during sampler warm-up after an
external process grew; it produced no example output and was safely restarted
at 0.45 on GPU0. A shard 4 launch on GPU4 was also rejected before model loading
because only 18.11GB was free. These are operational retries with identical
frozen model, decoding, data, and shard membership—not scientific reruns, and
neither produced an example file. Progress is the count of completed
per-example JSONL files; atomic replacement makes a file visible only after all
4096 states for that example finish.

No ranker training may start until all 5000 lattices are complete, each lattice
passes structural validation, and the frozen target-blind role partitions meet
their required post-exact eligibility counts without cross-role reallocation.

Exact inference completed successfully on 2026-09-14 at 09:24 CST: all 5000
lattices contain 4096 unique binary states and no temporary output remains. The
frozen partition-then-attainability filter produced the following counts:

| Role | Attainable in frozen slice | Required/selected |
|---|---:|---:|
| Train pool | 3163 | 2000 |
| Ranking validation | 510 | 300 |
| Calibration | 485 | 300 |
| Final test | 487 | 300 |

The 2900 selected IDs are pairwise disjoint. Set-valued oracle construction at
normalized slack 0.005 produced 89,609 identifiable training pairs over all
2000 training examples. The nested learning-curve subsets contain 22,227,
44,864, and 89,609 pairs at sizes 500, 1000, and 2000 respectively. The fixed
ranking-validation300 contains 13,404 identifiable pairs across 299 supervised
examples; all 300 remain available for oracle-cutoff contract evaluation.

## Rank-only decision gate

At train size 2000, the mean across the three frozen seeds must satisfy every
condition below under learned ordering plus a global oracle monotone cutoff:

| Metric | Required |
|---|---:|
| Active contract success | at least 0.97 |
| Complete active trajectories | at least 0.90 |
| Highest active-anchor success | at least 0.90 |
| Feasible normalized ranking regret | at most 0.03 |
| Identifiable pairwise accuracy | at least 0.85 |

Only a conjunctive pass permits implementation or training of the cutoff head.
If it fails, V2 stops at ranking diagnosis rather than adding mechanisms.

## Observed rank-only result

All nine frozen RankNet runs completed without interruption: train sizes
500/1000/2000, each with seeds 20260910/20260911/20260912. Checkpoint selection
restored the minimum-validation-loss epoch, so the consistent loss increase
after epoch 1--2 did not replace the selected weights. Mean results were:

| Train size | Active success | Full active trajectory | Ranking regret | Pair accuracy |
|---:|---:|---:|---:|---:|
| 500 | 0.96676 | 0.88556 | 0.03206 | 0.92967 |
| 1000 | 0.96885 | 0.89556 | 0.02705 | 0.94422 |
| 2000 | 0.97327 | 0.90667 | 0.02695 | 0.94706 |

The train2000 mean passed all five frozen checks. The authoritative decision
is `results/v2_rank_then_cut/rank_only_gate_result.json` with
`decision=GO_IMPLEMENT_CUTOFF`. This is a ranking-stage result under oracle
cutoffs, not an end-to-end learned-cutoff or final-test claim.

## Post-gate validation diagnosis

The existing final risk target requires a Bonferroni-corrected one-sided
Clopper--Pearson lower bound of at least 0.90 at every anchor. At fidelity 0.90,
the three learned-order oracle-cutoff runs obtained 276/300, 278/300, and
275/300; 282/300 is required for that lower bound. A learned cutoff cannot
exceed its oracle-cutoff ceiling, so the already-consumed validation set was
used for diagnosis before cutoff implementation.

Across 84 failed seed-trajectories, 78 require only one boundary replacement,
75 have a one-packet-drop rescue, and 67/90 closest-repair intruders are robust
never-reveal packets. Of 102 missing--intruder repair relations, 73 were already
identifiable preferences that the model violated, 16 opposed the selected
closest repair, and 13 were unlabeled. Twenty-three examples failed under all
three seeds. Exact details are in
`results/v2_rank_then_cut/rank_failure_audit_summary.json`.

Two oracle-label counterfactuals isolate the cause. Stable projection of each
learned order onto its existing partial-order labels reached 298/300 at the
0.90 anchor for every seed, 99.33% complete-trajectory success, and
0.00947--0.01101 feasible normalized regret. Merely moving robust never-reveal
packets to the tail reached 296--297/300 and 98.67--99.00% trajectory success
without material regret inflation. These are non-deployable diagnostics on
consumed validation, not new heldout results. They show that the representation
and oracle are sufficient; mean pairwise RankNet underweights joint constraint
satisfaction.

`configs/v2_1_listwise_ranking_amendment.json` therefore froze one controlled
post-gate revision: replace mean RankNet with the exact Plackett--Luce marginal
likelihood of all linear extensions satisfying the same set-valued partial
order. All three train2000 seeds completed. Their 0.90-anchor results were
275/300, 278/300, and 275/300, so none reached the frozen 282/300 risk-headroom
check. The authoritative decision is
`results/v2_rank_then_cut/v2_1_listwise_decision.json` with
`decision=STOP_V2_1_RANKING`. Best checkpoints came from epochs 2, 1, and 1;
later validation-NLL deterioration did not replace them.

## V3 critical-boundary hypothesis

The stopped V2.1 result rules out merely replacing an average pair surrogate
with a whole-partial-order surrogate. Before any further training, an
exhaustive consumed-validation diagnostic tested whether failure is localized
to critical prefix boundaries. For every admissible state at every anchor, it
labels a local positive as critical when deleting it violates the anchor and a
local excluded packet as harmful when adding it violates the anchor. A
direction is retained only if it is also identifiable across the complete set
of near-optimal nested chains.

This set-valued filter is necessary: naive local counterfactual directions
conflicted on 68/300 development examples. After filtering, stable-boundary
projection reached 298/300 at fidelity 0.90 for all six existing RankNet and
listwise runs. It rescued 23--28 failed trajectories per run without breaking
any previously successful trajectory. This is an oracle-label diagnostic on
consumed data, not a deployable result.

`configs/v3_critical_boundary_ranking.json` freezes V3 before training. It
keeps Qwen3-1.7B, the scalar rank head, lossless packets, and all Target
lattices fixed. Its sole primary objective is the example-balanced softplus of
the worst stable critical--harmful score violation; no RankNet, listwise, or
never-reveal auxiliary loss is added. Train2000 supplies 38,045 stable boundary
pairs across 1,772 supervised examples.

The unused tail of the originally frozen ranking-validation role contains 210
eligible examples and is frozen as rank-confirm210. It overlaps none of the
previous train, validation, calibration, or final-test selections. Three V3
seeds are selected using only the already consumed development300; exactly one
selected checkpoint will then be evaluated once on rank-confirm210. Cutoff
implementation is permitted only if every per-anchor Bonferroni
Clopper--Pearson lower bound reaches 0.90 and the frozen contract/regret checks
also pass. Calibration300 and final-test300 remain untouched.

## Authoritative inputs

V3 completed all three seeds without training errors. The seed selected using
consumed development loss was 20260911. Its one-shot rank-confirm210
oracle-cutoff evaluation gave active contract success 0.97126, feasible ranking
regret 0.02351, and all-active trajectory success 187/210 (0.89048). At the
0.90 anchor it reached 197/210 versus 199/210 required; at 0.95 it reached
156/169 versus 162/169 required. The frozen decision is `STOP_V3_RANKING`,
not permission to train a cutoff or use locked roles.

Boundary-pair accuracy on the consumed confirmation was 3576/3860 (0.9264),
but complete stable-boundary separation was only 128/182 (0.7033). An
oracle-label projection rescued 22/23 failed trajectories; this is a
non-deployable counterfactual, not evidence that the ranker can infer those
corrections. V3 already optimizes the worst retained boundary inversion, so
another margin or auxiliary loss is not justified by these results alone.

The next post-hoc audit on consumed roles measures within-anchor marginal
effect and contract-crossing flips, raw local versus globally stable
precedence cycles, near-optimal slack sensitivity, and train/development/fresh
joint-boundary generalization. Neither sign flips nor raw local cycles alone
prove that no static total order can reach a feasible prefix. Any new model
formulation requires a separately frozen protocol and new confirmation
population. Calibration300 and final-test300 remain locked.

The completed consumed-role audit is
`results/v2_rank_then_cut/V3_MODEL_ASSUMPTION_AUDIT.md`. It finds strict
within-anchor marginal sign flips in about 23--25% of harmful packet-anchor
groups, raw local cycles in about 23--25% of examples, but zero cycles in
globally stable relations. Selected V3 complete boundary separation is 77.09%
on train2000, 66.92% on development300, and 70.33% on confirm210. This is not
evidence that static total orders are mathematically impossible, nor a pure
fresh-only generalization collapse. The audit keeps V3 stopped and does not
authorize V4 architecture selection from these rates alone.

After that audit, `configs/v4_relational_precedence_ranking.json` froze one
controlled representation test before any new-candidate Target output. V4
replaces unary scalar differences with antisymmetric pairwise precedence
logits, then uses an exact subset DP to produce the maximum-weight total order.
It retains one Qwen3-1.7B forward, the same stable-boundary worst-edge objective,
lossless packets, nested prefixes, and frozen Qwen3-8B Target. It adds no
margin, auxiliary loss, confidence threshold, cycle deletion, RL, larger
backbone, or cutoff head.

A new target-blind candidate tail of 500 QAMPARI-train examples was frozen at
canonical source-order positions [5000,5500), with zero overlap against the
entire parent candidate5000. All 500 complete exact lattices finished and the
first 300 eligible examples were frozen as V4 rank-confirm300. This role was
never evaluated because V4 failed its preceding consumed-development gate.

All three V4 seeds selected epoch 2. The chosen seed 20260912 achieved
279/300 at fidelity 0.90, 275/300 complete trajectories, and 0.03397 feasible
normalized ranking regret. The frozen gate required 282/300 and regret at most
0.03, so the authoritative decision is `STOP_V4_WITHOUT_FRESH_CONFIRM`.
Calibration300 and final-test300 remain untouched.

The consumed-development tail audit localizes the remaining problem. Of 32
closest-repair relations for the 21 fidelity-0.90 failures, 24 are supported
by the frozen critical-edge labels, but 25/32 are predicted in the wrong
direction by all three seeds. Averaging the three seed logits gives only
275/300 at fidelity 0.90, although its regret falls to 0.02812. The error is
therefore mostly systematic rather than seed variance. Separately, the ten
largest-regret examples accumulate 10,192 early extra-packet tokens and 8,366
net excess tokens. The reproducible diagnostic is in
`results/v2_rank_then_cut/v4_tail_cost_diagnostic/`.

`configs/v5a_tail_risk_cost_aware_ranking.json` freezes the resulting V5-A
test. The model, 1.7B backbone, packet representation, and exact decoder are
unchanged. Its only controlled change is the training objective: top-quartile
CVaR over cost-weighted stable safety edges plus a 0.25-weighted rate-dominance
loss. Train2000 contains 38,045 safety and 51,564 non-overlapping rate edges.
Three seeds are running concurrently on GPUs 3, 0, and 2. Seed selection and
the gate use only the already consumed development300. V4 rank-confirm300 is
not reused; a new target-blind confirmation population may be frozen only if
V5-A first reaches 282/300 at fidelity 0.90, at least 90% complete trajectory
success, and regret at most 0.03. Monitor with
`bash scripts/55_monitor_v5a_pipeline.sh`.

V5-A completed and selected seed 20260910 at epoch 2. Rate alignment improved:
feasible normalized regret fell to 0.02979 and passed its 0.03 threshold.
However, fidelity 0.90 reached only 274/300 and complete-trajectory success
was 267/300 (0.89). The frozen decision is
`STOP_V5A_WITHOUT_FRESH_CONFIRM`; no fresh role was opened.

The post-stop decoder audit rules out the proposed criticality-aware decoder
as the next isolated change. Across all 5,697 development safety edges, the
decoded total order agreed exactly with every raw pair-logit sign: zero edges
were predicted correctly and then sacrificed by the maximum-weight decoder.
For the 26 failed 0.90 examples, all 32 closest-repair logits pointed in the
wrong direction, with median absolute margin 2.30; 30/32 were wrong for all
three seeds. An oracle-label safety projection reaches 298/300, but that is a
non-deployable upper bound because it supplies the missing directions.

Failure is concentrated in harder evidence-composition groups. V5-A reaches
81/98 at fidelity 0.90 on `wikitables_composition`, versus 183/190 on
`wikidata_simple`; 17/26 failures are table-composition examples. The train
and development group proportions are similar, so simple resampling does not
explain the gap. These results motivate a controlled backbone-capacity test
while keeping the V5-A objective and decoder fixed.

V6 completed that controlled capacity test with Qwen3-4B. The selected
seed 20260912 checkpoint came from epoch 1; its validation objective then
worsened from 0.31964 to 0.35136 and 0.48348 while training loss continued to
fall. On consumed development300 it reached 277/300 at fidelity 0.90,
273/300 complete trajectories, and 0.02622 feasible normalized regret. The
trajectory and regret checks passed, but the frozen fidelity gate required
282/300. The authoritative decision is `STOP_V6_WITHOUT_FRESH_CONFIRM`.
No fresh, calibration, or final-test role was opened.

The completed V6 post-stop audit again rules out variance and decoding as the
next bottlenecks. All 29 closest-repair relations for the 23 failed 0.90
examples were predicted in the wrong direction by all three 4B seeds; their
selected-seed median absolute margin was 2.45 and mean absolute margin was
3.91. The three-seed logit ensemble remained at 277/300. The decoder
sacrificed zero raw-correct safety edges. Fifteen of the 23 failures are
`wikitables_composition`, and 20 failures persist from V4 to V6. The
diagnostics are reproducible under
`results/v2_rank_then_cut/v6_three_seed_tail_diagnostic/` and
`results/v2_rank_then_cut/v6_decoder_alignment_diagnostic/`.

Backbone scaling alone is therefore stopped. The remaining target-blind
training-role reserve offers one clean coverage experiment: the originally
frozen train candidate slice contains 3,163 attainable examples, of which
only the first 2,000 were used. A V7 data-scaling hypothesis may use all
3,163 in the same deterministic training role while holding the V6 backbone,
objective, and decoder fixed. If that isolated coverage test does not reach
the consumed-development gate, further seed ensembles, larger backbones, or
decoder weighting are not supported by these audits; the next model family
should condition each reveal decision on the already selected packet set so
that compositional and redundancy effects are represented directly.

V7 is frozen in `configs/v7_train3163_data_scale.json`. It deterministically
uses all 3,163 attainable examples in the original training candidate slice;
the original train2000 is verified as its exact prefix. This expands safety
supervision from 38,045 to 59,562 edges and rate supervision from 51,564 to
82,160 edges. To isolate coverage from additional optimization exposure, V7
uses the same 750 optimizer updates and the same three checkpoint-selection
times as V6. Its batch size is four with gradient accumulation two, preserving
the effective batch size of eight while using available GPU memory.

V7 completed and selected seed 20260912 at optimizer step 500. It reached
277/300 at fidelity 0.90, 273/300 complete trajectories, and 0.02578 feasible
normalized regret. The latter two checks passed, but the frozen fidelity gate
again required 282/300. The authoritative decision is
`STOP_STATIC_ONE_SHOT_RANKING_AFTER_V7`; no fresh role was opened. Relative to
V6, 22/23 fidelity-0.90 failures persisted, one table-composition example was
rescued, and one different table-composition example broke. Failure counts
remain 15 table-composition, six simple, and two intersection examples.
Expanding training coverage by 58% therefore improved rate slightly but did
not improve the primary fidelity result; stable-boundary edge accuracy also
fell from 0.93628 to 0.93260. This closes the tested one-shot relational scorer
family. The next supported method test is an autoregressive ordering policy
whose next-packet score conditions on the already selected set, while retaining
one backbone encoding and the same nested total-order output space.

The V7 decoder audit confirms that this change must happen before decoding.
The exact total-order decoder sacrificed zero raw-correct safety edges. All 29
closest-repair relations for the 23 failed 0.90 examples already had the wrong
raw sign, with mean absolute margin 2.69. Of those repair relations, 18 are
supported by current safety labels, seven are unlabeled, and four are labeled
in the opposite direction. Supplying oracle safety directions to the unchanged
decoder raises fidelity-0.90 success from 277/300 to 298/300 and complete
trajectories from 273/300 to 298/300. The failure is therefore in inference of
the context-dependent relation and partly in the static pair-label target, not
in the maximum-weight decoder.

The history-aware V8 preflight is recorded in
`results/v2_rank_then_cut/v8_sequential_preflight_development300.json`. Its
exact dynamic program uses state `(selected mask, highest anchor reached in
the reveal history)`, maximizes anchors reached, and then minimizes cumulative
tokens at first anchor crossings. It matches the existing globally optimal
nested-chain cost on 300/300 consumed-development examples, both at the root
and after reconstructing a canonical total order. Exact optimal action sets
contain 2.12 packets on average; the frozen 0.005 near-optimal slack raises
this to 2.97, supporting a set-valued next-action loss.

The history component cannot be dropped from the oracle definition. Among
105,602 selected masks reachable with more than one prior highest-anchor
state, 43,930 (41.60%) have different exact optimal next-action sets across
those histories; 292/300 examples contain at least one such mask. A model that
receives only an unordered selected-set mean would therefore be trained on
conflicting labels during off-policy rollouts. The proposed V8 policy should
combine selected-set pooling with a small recurrent history state. The exact
anchor history is used only to compute offline labels and must never be an
inference input.

The subsequent pretraining review is recorded in
`results/v2_rank_then_cut/V8_ONE_RUN_DESIGN_REVIEW.md`. It removes the proposed
train--DAgger--retrain sequence: four exact-oracle rollouts, twelve fixed
single-deviation recovery rollouts, and four random recovery rollouts are all
labeled before training. On development300 this deterministic construction
produces 60,504 unique ordered histories (201.68 per example). Formal labels
use zero per-step slack to prevent tolerance from accumulating across twelve
actions. The reviewed encoder also removes packet IDs from packet text and adds
a question-only vector. To avoid a frozen-head failure followed by a second
formal run, the recommended single run jointly trains a fresh Qwen3-4B LoRA
and the new sequential head with one seed.

All 23 V7 fidelity-0.90 failures leave the exact optimal action set before
their failed order is complete; the median first divergence is reveal step 3.
Only 10/29 closest final-order repair relations are direct next-action
corrections at the prefix before the intruder. This confirms that another loss
over those pair repairs would remain locally misaligned, while the sequential
teacher directly labels the earlier causal decision in every failed example.

The reviewed V8 implementation and data are now frozen in
`configs/v8_one_run_sequential.json`. Train3163 contains 638,403 deduplicated
ordered-history labels; development300 contains 60,504. All 108 unit tests and
all frozen implementation, data, and Qwen3-4B hashes passed. Longest-input
memory tests established batch eight with gradient checkpointing as the
largest stable effective-batch-eight configuration. Disabling checkpointing
exhausted a 47.4 GiB A6000 even at batch four, so the formal run keeps
checkpointing rather than accepting an OOM-prone configuration. The single
formal seed 20260912 started on physical GPU2 at 2026-09-17 08:05 CST, with
checkpoints fixed at optimizer steps 250, 500, and 750 and beam width eight.

V8 completed all 750 optimizer steps without an OOM or training error in
8418.74 seconds. The fixed checkpoint results were:

| Step | Fidelity-0.90 success | Complete trajectories | Feasible normalized regret | Validation action accuracy |
|---:|---:|---:|---:|---:|
| 250 | 277/300 | 273/300 (0.9100) | 0.02793 | 0.82802 |
| 500 | 274/300 | 273/300 (0.9100) | 0.02360 | 0.86575 |
| 750 | 273/300 | 272/300 (0.90667) | 0.02287 | 0.86873 |

The preregistered selector chose step 250. It passed the trajectory threshold
of 0.90 and regret threshold of 0.03, but missed the required fidelity-0.90
count of 282/300 by five examples. The authoritative decision is
`STOP_V8_ONE_RUN`; no fresh confirmation, calibration, or final-test role was
opened. Twenty-two of V7's 23 fidelity-0.90 failures persisted at the selected
V8 checkpoint; one was repaired and one different table-composition example
failed. Validation action loss improved from 0.45993 at step 250 to 0.37197 at
step 750, while fidelity-0.90 success fell from 277 to 273. This establishes a
surrogate-to-contract mismatch rather than a failure of optimizer convergence.
The authoritative artifacts are under
`results/v2_rank_then_cut/v8_one_run_seed20260912/`.

## V8 post-stop audits and bounded V9 probes

The V8 first-divergence audit found that all 23 fidelity-0.90 failures had an
exact optimal action in the local top eight at their first nonoptimal action;
the mean best-optimal rank was 2.43 and mean optimal probability mass was
0.286. Nevertheless, every exact-oracle-consistent path was eventually pruned
from beam eight, at median reveal depth five. Only 9/23 first-divergence
histories and 19.93% of all decoded-prefix histories were present in the fixed
V8 supervision pool. Greedy decoding was not a remedy: it reached 276/300 at
fidelity 0.90 versus beam eight's 277/300, with 272 complete trajectories and
0.02886 regret.

The exact DP was extended to retain each action's downstream severity: lost
reachable anchors first, then normalized future cumulative-token excess. This
produced cost labels for all 638,403 train histories and 60,504 consumed-
development histories. A frozen-backbone V9-A probe retrained the complete
sequential head for 250 steps with a cost-weighted ranking loss. Beam one and
beam eight both reached only 274/300 at fidelity 0.90 and 270/300 complete
trajectories; regret was 0.03528 and 0.03360. It repaired three V8 failures but
broke six previous successes. The preregistered probe decision therefore stops
the fixed-pool cost-head hypothesis.

A separate train-role coverage audit rolled the selected V8 beam-eight policy
over all 3,163 training examples. Only 8,531/37,956 (22.48%) deployed prefix
states were present in the fixed pool. Optimal-action accuracy was 93.73% on
covered states and 78.37% on uncovered states; lost-anchor actions occurred in
0.387% and 1.057% respectively. A V9-B probe therefore isolated one bounded
aggregation round while restoring the original V8 set-valued loss. It reached
275/300 at fidelity 0.90, 272/300 complete trajectories, and 0.02392 regret,
so the coverage-only hypothesis also stops.

The two probes do not justify a post-hoc combination. Their failure sets share
22 examples; even an oracle that chooses per example between V9-A and V9-B
would reach only 278/300, and an oracle over V8, V9-A, and V9-B would reach only
280/300. Twenty fidelity-0.90 failures persist across all three. No fresh,
calibration, or final-test role was opened. Authoritative probe artifacts are
under `results/v2_rank_then_cut/v8_first_divergence_audit/`,
`results/v2_rank_then_cut/v9a_cost_probe_step250/`, and
`results/v2_rank_then_cut/v9b_coverage_probe_step250/`.

## V10 mask-value probe: ready for pretraining review

The next bounded diagnostic is frozen in `configs/v10_mask_value_probe.json`
and documented in `results/v2_rank_then_cut/V10_PRETRAINING_REVIEW.md`. It
reuses the selected V8 step250 encoder without updating the backbone, LoRA, or
existing sequential head. A new mask-conditioned ordinal head predicts the
active fidelity anchors for any of the 4096 packet subsets, after which an
exact subset DP constructs the nested total order. Train3163 contributes
253,973 deterministic attained-class-stratified states; consumed
development300 retains its complete 1,228,800-state lattice for one final
evaluation only. Calibration and final-test roles remain untouched.

The frozen endpoint is batch 16 for 400 optimizer steps (about 2.02 epochs),
with no intermediate checkpoint selection. The same 282/300 fidelity-0.90,
0.90 complete-trajectory, and 0.03 regret gates apply. All 123 unit tests and
the data-integrity audit pass. The protocol status is
`READY_FOR_REVIEW_NOT_STARTED`; the launch script is deliberately locked and
no V10 training process, optimizer step, checkpoint, or result exists.

The subsequent scientific review in `V10_SCIENTIFIC_REVIEW.md` **does not
approve this frozen version for training**. The decoder currently receives
the true per-example active-anchor count from the Target lattice; a synthetic
counterexample proves that this information can alter both order and success.
Unweighted stratified supervision also changes the 0.90 positive rate from
0.7159% to 20.3234%, without defining a corresponding inference correction.
The CPU-only oracle-value positive control is encouraging: the new planner
recovers all anchors and matches the exact oracle cost on 300/300 consumed
development examples. This validates the planner with true values, not a
learned value model. See `v10_pretraining_audit.json` for evidence. The review
leaves training and configuration unchanged and requires label-free decoding,
an explicit sampling/loss interpretation, strict gate validation, and a
fresh-confirmation protocol before scientific progression.


- `configs/v2_rank_then_cut_hypothesis.json`
- `results/v1_atomic/RANK_CUT_DIAGNOSTIC.md`
- `results/v1_atomic/fresh30_rank_cut_decomposition_summary.json`
- `data/units/qampari_rank_v2_candidates5000.jsonl`
- `data/units/qampari_rank_v2_candidates5000_annotations.jsonl`
- `data/packets_qampari_rank_v2_candidates5000/`

## V10 result and fidelity-0.90 root cause

The corrected V10 frozen-encoder mask-value probe completed its single frozen
400-step endpoint. It reached 269/300 at fidelity 0.90, 246/300 complete
trajectories (0.82), and 0.02453 feasible normalized regret. The regret gate
passed, while both contract gates failed; the authoritative decision is
`STOP_MASK_VALUE_HYPOTHESIS`. Its direct predicted-cutoff trajectory success
was only 0.3467, so this result does not support cutoff or calibration work.
The preregistered confirmation300 role remains unopened.

A post-stop audit compared ten endpoints spanning V4, V5-A, V6, V7, all three
V8 checkpoints, V9-A, V9-B, and V10. Sixteen development examples fail at
fidelity 0.90 under every endpoint; nine are table-composition examples. Their
0.90-feasible subset is exceptionally sparse: 3.75 of 4096 masks on average
and two at the median, versus 33.17 and 37 among 260 examples that never fail.
All sixteen have full-context fidelity below 0.90. A uniformly random packet
order visits a 0.90-feasible prefix with median probability 5.68%, versus 100%
for the never-failed group. The issue is therefore not ordinary rate ranking:
the compressor must assemble a narrow evidence combination before revealing a
harmful packet.

The training population already contains this tail: 227/3163 examples have at
most four fidelity-0.90-positive masks, close to the development prevalence.
This explains why data scaling alone did not help. It also exposes the V10
estimand mismatch. Its inverse-probability loss correctly estimates uniform
full-lattice classification risk, under which only 0.724% of development
masks are 0.90-positive. That objective can improve aggregate mask accuracy
without learning to retrieve the one or two masks that determine query-level
success. V8 provides a second independent negative control: from step250 to
step500 and step750 it rescued zero 0.90 failures and broke three, then one
more, even while action loss and accuracy improved.

A union of the terminal beam-eight candidates from V8, V9-A, and V9-B contains
19.77 unique orders per query on average. Even a nondeployable Target oracle
selecting among them reaches only 280/300 at fidelity 0.90, below the 282 gate.
A learned V10 reranker reaches 277/300. Candidate reranking or a modest beam
increase therefore lacks sufficient support.

A different bounded construction has real headroom. Starting from the frozen
V8 order, place the packets in a candidate 0.90 mask before the remaining
packets while preserving V8's relative order within both groups. An oracle over
all true 0.90 masks reaches every active anchor on every development example,
300/300 complete trajectories, and 0.01894 regret. More relevantly, V10 already
retrieves a true 0.90 mask in its top 16 for 289/300 queries. The fixed candidate
set consisting of V8 plus those sixteen mask projections has an oracle result
of 294/300 at fidelity 0.90, 294/300 complete trajectories, 234/234 at 0.95,
and 0.02056 regret. This supports one V11 selector test: freeze V8 and V10,
train only a query-balanced candidate head to select among those seventeen
orders using a strict contracts-first, cumulative-token-second target.
Reproducible diagnostics are in `v10_high_fidelity_root_cause.json`,
`v11_candidate_rerank_audit/`, and `v11_mask_retrieval_audit/`.

## V11 selector and V12 direct-retrieval results

The frozen V11 candidate selector did not convert the mask-projection oracle
headroom into a deployable improvement. It reached 277/300 at fidelity 0.90,
273/300 complete trajectories (0.91), and 0.02800 feasible normalized regret.
Among 126 development queries where the frozen candidate set had a strictly
better choice than the baseline, it selected a true optimum only six times.
The learned selector almost always changed the order, but repaired one
contract and broke one. This stops the frozen-representation candidate-selector
hypothesis; no confirmation role was opened.

V12 then tested the more direct estimand: a query-balanced multiple-positive
softmax loss over masks attaining fidelity 0.90, initialized from V10 and with
the V8 encoder and projection rule frozen. The single 400-step endpoint placed
a true 0.90 mask in its top 16 for 291/300 queries, but its deployable top-one
projection reached only 277/300 at fidelity 0.90, 274/300 complete trajectories
(0.91333), and 0.02776 regret. It therefore passes the complete-trajectory and
regret gates but fails the preregistered 282/300 fidelity-0.90 gate. The formal
decision is `STOP_RETRIEVAL_HYPOTHESIS`, recorded in
`v12_high_fidelity_retriever_seed20260918/decision.json`; no confirmation role
was opened.

The V12 top-four candidate oracle reaches exactly 282/300 at fidelity 0.90,
282/300 complete trajectories (0.94), 234/234 at fidelity 0.95, and 0.02583
regret. The top-16 oracle reaches 294/300 and 0.98 complete trajectories. These
are nondeployable upper bounds. Together with V11, they locate the remaining
problem more narrowly: candidate generation has adequate headroom, while the
current frozen query/mask representation and contracts-first supervision do
not reliably identify which retrieved mask should control the order.

## V13 joint representation-retrieval result

V13 jointly adapted the selected V8 LoRA encoder and the V12 high-fidelity
retrieval head. Its endpoint was selected without development300: among steps
100, 200, and 300, the fixed train3163 internal-validation retrieval losses
were 0.08131, 0.08477, and 0.08046, selecting step300. A batch-32 systems
attempt stopped after step10 from a length-dependent OOM, before any checkpoint
or development access; the formal run restarted from the original initialization
at batch16 and completed all 300 steps.

The single frozen development evaluation does not validate the direct top-one
hypothesis. Its projected top-one order reached 276/300 at fidelity 0.90, 272/300
complete trajectories (0.90667), and 0.02692 feasible normalized regret, versus
V12's 277/300, 274/300, and 0.02776. The formal representation decision is
`STOP_V13_TOP1_RETRIEVAL`.

The candidate-space result supports the separately defined downstream
feasibility gate. Oracle selection plus oracle cutoff over the frozen top four
reached 286/300 at fidelity 0.90, 286/300 complete trajectories (0.95333),
234/234 at fidelity 0.95, and 0.02576 regret. This improves the V12 top-four
ceiling from 282 to 286 and gives real margin above the 282 development target.
The top-16 ceiling was 293/300 with 0.97667 complete trajectories and 0.01917
regret. Accordingly the decision is `GO_DOWNSTREAM_DEVELOPMENT`, while fresh
confirmation remains closed. These oracle results justify training a label-free
selector/cutoff for the frozen candidate architecture; they are not deployable
performance claims.

## V14 conservative evidence-sufficiency selector

V14 froze the V13 step300 top-four candidate generator and the V8 fallback,
then trained only a five-anchor evidence-sufficiency selector. Candidate labels
came from the train3163 exact lattice; checkpoint and switch-threshold selection
used only the fixed internal validation300. That internal role had six
fidelity-0.90 oracle-rescuable failures (279 fallback versus 285 top-four
oracle). Step400 was selected: it reached 280/300 at fidelity 0.90 versus the
279 fallback, improved complete trajectories from 274 to 277, and produced no
aggregate per-anchor regression.

The frozen selector did not transfer enough of the candidate ceiling on
consumed development300. It switched on 148/300 queries and reached 278/300 at
fidelity 0.90, 274/300 complete trajectories (0.91333), and 0.02831 regret. It
rescued one V8 fidelity-0.90 failure and broke no prior success, for only one
net improvement over the 277 fallback and four fewer successes than the frozen
282 selector-feasibility gate. The authoritative decision is
`STOP_CONSERVATIVE_SELECTOR`; the multi-anchor cutoff is not trained from this
selector and fresh confirmation remains closed. The top-four oracle ceiling of
286 remains a candidate-space diagnostic, but V11 and V14 now independently
show that available learned selectors cannot realize enough of it.

## V15 counterfactual repair verifier

V15 was the preregistered final test of a selector on the frozen V13 top-four
candidate space. Unlike V14, it represented each candidate relative to the V8
prefix with the same packet count: candidate evidence, matched-prefix evidence,
added evidence, removed evidence, and their interactions. The verifier learned
four explicit outcomes (`repair`, `break`, `both_success`, and `both_fail`) plus
the change in maximum trajectory fidelity. Five-fold out-of-fold predictions
on train2863 alone selected a frozen conservative rule with break risk multiplier
2.0 and threshold 0.2; neither internal validation nor development was used to
fit the rule.

The out-of-fold estimate improved fidelity 0.90 by 12 examples without aggregate
regression, but this did not generalize. On the post-freeze internal validation,
none of six available repairs was selected. On consumed development300, V15
made 24 switches: two repairs, zero breaks at fidelity 0.90, twenty
`both_success` selections, and two `both_fail` selections. It improved the
frozen fallback from 277 to 279 at fidelity 0.90, while decreasing successes by
2 at fidelity 0.70, by 1 at 0.80, and by 2 at 0.95. It retained 273/300 complete
trajectories (0.91) and 0.02870 feasible normalized regret.

The formal decision is `STOP_FIXED_REPRESENTATION_SELECTOR_VERIFIER_FAMILY`.
V15 is three examples below the 282 target and also violates the other-anchor
no-regression condition. No threshold is retuned on development, no learned
cutoff is trained from this endpoint, and fresh confirmation remains closed.
The top-four oracle ceiling remains real, but V11, V14, and V15 now show that
the fixed V13 representation does not expose enough information to identify
the rare repair decisions reliably. Further work must change upstream
representation or candidate construction, or test a different compression
mechanism, rather than add another selector loss or head to this candidate set.

## V16-A root-cause representation audit

V16-A separated three possible causes of V15's failure without opening fresh
confirmation. A structural audit found that 109/115 train repair pairs and all
six internal-validation and nine consumed-development repair pairs were
one-packet swaps. Every validation and development repair was trajectory-safe.
The top-four masks were highly redundant, with mean pairwise Hamming distance
1.62, but they still contained enough safe repairs to cross the 282 gate.
Consequently, unsafe repair candidates and packet granularity are not the
immediate bottleneck.

Three representation controls used the same train2863 labels and fixed internal
validation. The V15 pooled representation recovered 2/6 repair pairs above all
break pairs by repair probability, although its frozen decision rule selected
none. A three-layer packet-interaction transformer over the unpooled frozen V13
packet embeddings recovered only 1/6 and selected none. Its failure shows that
a simple pooling replacement is insufficient. A raw-text Qwen3-4B LoRA
cross-encoder was then trained as a positive control. Its first run was declared
invalid because the prompt placed swap evidence after retained context and
truncated 5/6 repair inputs. The corrected swap-first endpoint recovered only
1/6 repairs above all breaks and also failed its preregistered 3/6 diagnostic
gate.

A frozen-V13 nearest-neighbor audit provides evidence of local label conflict:
five of six validation repairs had no repair among their ten closest train
pairs, whose labels were overwhelmingly `both_success`. However, because the
corrected raw-text model did not improve this result, V16-A cannot attribute the
failure to V13 information loss alone. The formal decision is
`DO_NOT_ATTRIBUTE_FAILURE_TO_V13_REPRESENTATION_ALONE`.

The common bottleneck is more consistent with only 115 repair pairs among
11,452 train pairs, neutral-class dominance, and discontinuous frozen-Target
behavior around one-packet swaps. The next experiment should first construct a
near-policy decision-boundary dataset from the exact lattice, labeling add,
drop, and swap actions by five-anchor state-action advantage and future minimum
token cost. Its positive support and train/validation coverage must be audited
before training a behavior-aware progressive model. Another selector head,
more top-K candidates, or immediate repacketization is not supported by V16-A.

## V16-B0 boundary-causal data feasibility audit

V16-B0 tested whether the proposed progressive-policy direction contains new,
usable supervision beyond V9. It scanned every V8 deployed prefix in train2863
and fixed internal validation against the complete exact lattice. Swaps were
used only as counterfactual diagnostics and were backtracked to the state before
the deferred packet first entered; every evaluated deployment order remained
strictly add-only.

The broad audit found safe fidelity-crossing actions near 15,734/34,356 train
prefix states (45.80%), 31,050 unique causal preference labels, and 98.71%
coarse validation-pattern coverage. A nondeployable oracle choosing among
single-swap-corrected add-only orders reached 299/300 at fidelity 0.90. However,
only 19.25% of preferred additions were exact-DP optimal at their predecessor.
Training all immediate crossings would therefore conflict with future trajectory
quality and is rejected.

The corrected supervision retains only preferences where the proposed earlier
packet is exact-DP optimal and the deferred packet is not. This leaves 3,520
train labels over 3,304 predecessor states and 357 internal-validation labels;
strict validation-pattern coverage remains 92.16%. The corresponding consumed-
development oracle, restricted to legal add-only orders constructed from these
DP-consistent single swaps, reaches 291/300 at fidelity 0.90, 291/300 complete
trajectories, 234/234 at 0.95, and 0.01287 feasible normalized regret.

Every causal predecessor was already present in V9's on-policy supervision.
Thus V16-B does not discover a new cost-to-go label: its controlled novelty is
to concentrate training on the small DP-consistent subset whose later
counterfactual swap crosses a fidelity boundary. The formal decision is
`GO_V16B_BOUNDARY_FOCUSED_MODEL_DESIGN`. One preregistered model experiment may
use these 3,304 states with matched non-boundary replay and set-valued DP action
targets. Unfiltered causal labels, online swap/drop, learned cutoff, and fresh
confirmation remain closed.

## V16-B1 DP-consistent boundary-focused replay

V16-B1 isolated the sampling-distribution hypothesis against V9-B. It reused
V8 step250 initialization, froze the backbone and LoRA, trained only the
unchanged sequential head for the same 250 steps with the same optimizer and
set-valued loss, and retained beam-eight add-only decoding. Each optimizer step
kept the V9-B effective size of eight queries and 44 histories per query: half
came from query-balanced strict boundary states and half from all-train-query
current-policy-correct retention states matched by depth and reached level.
Only the fixed step250 endpoint was evaluated.

The intervention failed. It reached 273/300 at fidelity 0.90, 267/300 complete
trajectories (0.89), and 0.02800 feasible normalized regret. Relative to V8 it
repaired one 0.90 failure and broke five successes, for a net loss of four.
It also fell below the frozen floors at 0.70 (297), 0.80 (293), and 0.95
(227/234). Relative to V9-B it repaired one and broke three, for a net loss of
two. Internal-validation top-one set accuracy was 30.98% on strict boundary
states and 89.27% on retention states.

The post-stop first-divergence audit explains the failure. Mean first-divergence
depth remained 2.96 versus V8's 3.00, and median oracle-compatible beam
extinction stayed at depth five. None of the 27 V16-B1 fidelity-0.90 failure
queries had its actual first-divergence history in the strict boundary pool.
Thus V16-B0's 92.16% coarse structural-pattern coverage did not imply coverage
of the deployment states that caused failure. The formal decision is
`STOP_BOUNDARY_REPLAY_RATIO_TUNING`; no alternative replay ratio, learned
cutoff, or fresh confirmation is opened. Any further policy work must redefine
boundary mining around train-role deployed first divergences and demonstrate
direct heldout first-divergence coverage before training.

## V16-C0 deployed first-irreversible-divergence audit

V16-C0 froze the selected V8 step250 policy and reproduced its beam-eight
add-only deployment on train2863 and the fixed internal-validation300. It
corrected the earlier extinction definition by separating paths that remain
strictly exact-DP optimal, paths that can still attain fidelity 0.90, paths
that can complete all active anchors, and paths that can additionally finish
within per-trajectory normalized regret 0.03. Development300 and all locked
roles remained untouched.

The primary fidelity-0.90 failure mechanism is consistent across the two
roles. On train2863, 197/218 failures (90.37%) first lost every 0.90-viable
beam path, while 21 retained a viable path but failed terminal top-one
selection. On internal-validation300, the corresponding counts were 19/20
(95%) and 1. FID depth was also aligned: train mean 7.77 and median 9 versus
validation mean 7.89 and median 9. At the extinction step, the expansion still
contained a median of 10 viable children on train and 16 on validation, but
the best such child was below the beam-eight pruning threshold in every case.
Median score deficits were 0.410 and 0.261 log-probability units respectively.

This audit also invalidates a broader interpretation of the old
oracle-compatible extinction statistic. Exact-DP paths became extinct for
1947/2863 train examples and 211/300 validation examples, far more often than
the primary contract failed. The original report also classified 615 train
and 57 validation examples as lacking a complete trajectory. V17-A later
showed that this was an artifact of reading V10/V13's fixed five-class
`active_levels` grid instead of its role-specific `attainable_levels` field;
the corrected complete-trajectory counts are recorded below.

The decision is `GO_V16C1_PROTOCOL_DESIGN`, not authorization to train an
unfrozen objective. There are 197 train-role causal beam-extinction examples,
and validation independently exhibits the same mechanism. V16-C1 may therefore
be designed around primary-0.90 path survival with same-rollout pre-FID
retention and an auxiliary global-DP constraint. Its replay ratio, endpoint,
and bounded aggregation rule must be frozen before training; development300,
learned cutoff, and fresh confirmation remain closed.

## V16-C1 failure-triggered beam-survival policy

V16-C1 tested the mechanism isolated by C0 rather than another local
next-action loss. Starting from V8 step250, it froze the backbone and LoRA and
trained the complete sequential head for one fixed 150-step endpoint. Each
update used train2863 FID beam parents, a cumulative-score margin requiring at
least one fidelity-0.90-viable child to clear the beam-eight pruning threshold,
same-rollout pre-FID V8 distribution retention, and a low-weight exact-DP
set-valued auxiliary loss. No development example entered training or endpoint
selection.

Under the then-preregistered interpretation, the internal-validation gate
passed. Fidelity-0.90 success rose
from 280/300 to 284/300, comprising five repairs and one break. The other
anchors changed from 296/295/294/238 to 298/296/295/238 at fidelity
0.60/0.70/0.80/0.95. Complete trajectories were reported as 238/300 and mean
regret as 0.02405. V17-A later invalidated these complete/regret diagnostics
because 0.95-unattainable examples were evaluated as five-anchor examples. This
authorized one evaluation of the fixed endpoint on consumed development300.

The improvement did not transfer. Development fidelity-0.90 remained 277/300:
one V8 failure was repaired and one V8 success broke. Fidelity 0.60 improved
to 299, but 0.70 fell from 298 to 296 and 0.80 from 295 to 294; 0.95 remained
230/234. Complete trajectories increased by one to 274/300 and mean feasible
regret remained acceptable at 0.02903, but the joint gate failed. A post-stop
mechanism audit found only a one-example reduction in primary beam-extinction
failures, from 22 to 21, consistent with the absence of deployable gain.

The formal decision is `STOP_STATIC_FID_CORRECTION`. C0 correctly identified
the deployment failure mechanism, and C1 showed that direct static correction
can improve the fixed internal role, but supervision from only 197 rare,
query-specific train failures did not generalize to the consumed development
role. No margin, learning-rate, step-count, or replay retuning is allowed; the
second aggregation round is not opened. Learned cutoff and fresh confirmation
remain closed.

## V17-A counterfactual decision-critical branch audit

V17-A first uncovered a data-semantics error that affects the interpretation
of V16-B/C train and internal diagnostics. The V10/V13 mask-value files use
`active_levels` for the fixed five-class value-head grid and preserve the
actual role-specific contracts in `attainable_levels`. Later DP audits treated
the fixed grid as the contract set. This incorrectly added unattainable 0.95
to 672/3163 examples. The original V8 and consumed-development files retained
the correct four- versus five-anchor distinction. C0/C1's primary 0.90 counts
are unchanged, and C1's fixed development result remains empirical, but its
auxiliary DP labels and earlier complete/regret causal interpretation are not
valid. The shared level-selection code now prefers `attainable_levels`.

After correcting the contract set, V17-A froze an exact decision-critical
rule and scanned every V8 deployed prefix in train2863 plus one- and two-hop
add-only counterfactual branches. The deployed train prefixes contain 6,200
feasibility-critical states, 1,834 primary-0.90-critical states, and 201,413
critical state-action examples. One-hop expansion raises these counts to
41,118, 12,253, and 1,382,109. Two-hop expansion is much larger at 315,204
feasibility-critical states and over ten million critical action examples.

The rule covers the actual failure mechanism across roles. All 19 internal
validation FID queries and all 22 consumed-development FID queries contain at
least one decision-critical beam parent. Using consequence signatures that
exclude query type, depth, and packet identity, deployed plus one-hop train
states exactly cover 129/129 critical internal FID-parent feasibility
signatures and 147/151 development signatures. Two hops increase development
coverage only to 149/151. Fine-grained token-cost signatures remain much less
covered: even through two hops they cover only 67/129 internal and 75/151
development critical parents.

The decision is `GO_V17B_MULTI_ANCHOR_VIABILITY_DESIGN`, not immediate model
training and not approval for continuous cost-Q regression. A V17-B protocol
may use train2863 deployed plus one-hop critical states, per-anchor future
viability targets, and beam-survival loss with add-only beam-eight decoding.
It must use `attainable_levels`, keep the V8 backbone and LoRA frozen, use one
fixed endpoint, and gate only on the fixed internal role. Hop-two expansion
is not selected from the consumed-development result, continuous future-token
cost prediction remains closed, and fresh confirmation is unopened.

## V17-B0 frozen multi-anchor viability protocol

V17-B0 converts the V17-A mechanism result into an audited training contract
without running the formal V17-B1 optimization. The canonical train artifact
contains one row per `(example_id, ordered_history)`, with legal actions nested
inside the row. It reproduces exactly 6,200 deployed and 41,118 one-hop
feasibility-critical states. The train role contains 2,248 five-anchor and 615
four-anchor examples; every 0.95 entry for the four-anchor group is undefined
and masked. Anchors already reached by the state are also masked from the main
future-viability loss rather than becoming trivial positive targets.

The frozen model change is a five-anchor action-viability head over V8's
state-action features and a zero-initialized residual scale. The V8 backbone,
LoRA, and sequential policy head remain frozen. The primary target is exact
per-anchor future reachability, macro-averaged across active anchors; the
auxiliary target marginalizes the complete lexicographic exact-DP optimal
action set. Sampling is query-balanced with an exact 50/50 split between
deployed and one-hop states. Hop two, continuous cost regression, learned
cutoff, selector expansion, and fresh confirmation remain disabled.

The engineering-only smoke test used no development examples. Its loss fell
from 0.77796 to 0.03455, every active anchor output received a nonzero gradient,
the set-valued loss stayed finite, and the zero residual reproduced the V8
score exactly. The corrected attainable-level recomputation of V16-C1 internal
results is 284/300 at 0.90, 279/300 complete trajectories, and mean complete
regret 0.02749. Relative to corrected V8 it has five 0.90 repairs and one
break; lower anchors have no breaks.

The B0 decision is `READY_FOR_V17B1_SINGLE_FORMAL_RUN`. This authorizes one
fixed-endpoint train2863 run followed by the preregistered internal opening
gate. It does not authorize development-driven tuning or confirmation access.

## V17-B1 multi-anchor viability residual

V17-B1 ran the single frozen step750 endpoint on the 47,318-state B0 artifact.
It used cached V8 embeddings and updated only the five-anchor viability head
and its zero-initialized residual scale. The viability objective fell from
0.663 at step one to 0.082 at the endpoint; the mean of the final five logged
viability losses was 0.099 versus 0.219 for the first five. The local survival
term also decreased from 0.060 to 0.030 across those windows. The exact-DP
auxiliary did not improve (0.639 to 0.722), and the learned residual scale
remained modest at 0.0633.

The preregistered internal gate failed. Relative to corrected V8, fidelity
0.60/0.70/0.80/0.95 stayed exactly at 296/295/294/238, while 0.90 fell from
280 to 279 through zero repairs and one break. Complete trajectories fell from
275 to 274, although mean complete regret improved from 0.02510 to 0.02370.
The endpoint changed 105/300 decoded orders and changed ten of the twenty V8
0.90-failure orders, yet repaired none. The sole new failure was
`56847__wikitables_composition__train`, whose first order difference occurred
at reveal depth five.

This rules out the narrow explanation that the residual was simply too small
to alter decoding: it changed one third of the orders and half of the existing
0.90 failures. The evidence instead points to a mismatch between state-local
future-viability supervision and cumulative beam-level trajectory competition.
The unchanged or worse exact-DP auxiliary supports the same interpretation.
The formal decision is `STOP_V17B1_INTERNAL_GATE`; development300, learned
cutoff, and fresh confirmation remain closed. No learning-rate, loss-weight,
step-count, or residual-scale retuning is authorized from this run.

## V17-C0 beam trajectory failure audit

V17-C0 replayed residual-off and V17-B1 beam8 under one deterministic harness
before attributing any order change to the learned residual. This exposed an
important evaluation-path confound: residual-off reproduced only 202/300 full
historical V8 orders. The historical traces used online bfloat16 encoding,
whereas B1 used cached embeddings and a float32 frozen head. Under the shared
C0 harness, residual-off and B1 both reached 279/300 at fidelity 0.90 and had
the same 21 failures. Thus the apparent `56847` clean break was not caused by
the residual; residual-off also fails that example in the paired harness.

Of the 21 shared failures, 19 lose their last 0.90-viable beam branch and two
retain a viable terminal branch but select a non-viable top-1. At the 19 first
irreversible prune events, nine have a positive normalized residual margin and
ten have a nonpositive residual margin. The mean base margin is -0.7220, the
mean normalized residual margin is -0.00136, and the median minimum rescue
margin is 0.26134 cumulative log-probability. The most common extinction depth
is nine, accounting for 9/19 events.

`oracle-keep-one-viable` leaves at least one viable terminal trajectory in all
21 cases, but it repairs zero top-1 outputs by itself. Adding oracle terminal
selection repairs all 21. This validates the exact-DP 0.90 labels and decoder
state semantics, while showing that branch survival and terminal selection are
separate bottlenecks.

The frozen decision rule does not authorize a V17-C1 trajectory objective:
only 9/19 prune events have residual help in the correct direction, while
10/19 show the local viability residual worsening the critical comparison.
The result instead requires returning to viability calibration and explicitly
auditing terminal selection. Development, hop two, continuous cost prediction,
learned cutoff, and fresh confirmation remain closed.

## V17-D0 canonical harness and two-bottleneck audit

V17-D0 freezes the cached-bfloat16-embedding/float32-head replay as the causal
V17 harness. Its residual-off 0.90 baseline is 279/300; the historical online
bfloat16 V8 result of 280/300 remains a deployment reference and is not used
for repair/break attribution. Beam width, candidate enumeration, cumulative
normalized-log-probability scoring, lexicographic tie-breaking, and terminal
top-1 selection are recorded in a regression-tested manifest.

The ten wrong-direction first-prune events do not support immediate critical
pairwise training. Nine have strict 0.90 target separation between the viable
action and boundary competitor, but the learned 0.90 logit favors the viable
action in only four events. Three of those four become wrong after multi-anchor
aggregation. Five events have no training-state structural analogue for at
least one side of the comparison, and five have zero raw residual at the
critical step because the frozen predicted-progress deployment mask suppresses
all relevant anchor contributions. Neither exact internal query state appears
in train2863, as required by the role split.

The terminal audit confirms that a standalone terminal reranker has a ceiling
of only 2/21 current failures. Those two standard-beam failures naturally retain
two and four viable terminal candidates respectively. The other 19 failures
contain no viable candidate in the standard terminal beam and therefore require
a survival repair before any terminal reranker can help. Terminal candidate
records use first-attainment depth/tokens and cumulative trajectory cost rather
than the uninformative full-mask terminal state.

The D0 decision is to design a critical calibration/coverage correction first.
Critical pairwise training is not yet authorized because the dominant evidence
is wrong per-level prediction, missing structural coverage, and progress-mask
suppression. A standalone terminal reranker is also not authorized from only
two directly repairable examples. Development, hop two, continuous cost
prediction, learned cutoff, and fresh confirmation remain closed.

## V17-D1A scoring counterfactual and boundary-coverage audit

D1A replayed six frozen scoring modes over the canonical internal300 without
training or threshold selection. Forcing the 0.90 residual active increased
the original ten fixed-event correct residual directions from 1/10 to 4/10.
Using only the 0.90 head while retaining the predicted progress mask remained
at 1/10; forcing 0.90 active and using target-local scoring also reached 4/10.
Thus the gain comes from repairing progress-mask suppression rather than from
discarding multi-anchor aggregation.

Full beam8 replay gives the same conclusion. Both force-0.90-active modes and
the exact-unresolved diagnostic ceiling repair one canonical failure with zero
breaks: 0.90 improves from 279/300 to 280/300 and complete trajectories from
274 to 275. Fidelity 0.60/0.70/0.80/0.95 remains exactly
296/295/294/238. Target-local scoring without the mask repair changes no
contract outcome. The repaired example is `56847__wikitables_composition__train`.
Mean complete regret remains below the frozen 0.03 limit, although it rises
from 0.02370 to 0.02469 under the selected multi-anchor force-active mode.

The repair is real but limited. The remaining first-prune median rescue margin
does not improve, and even the exact-unresolved target-local ceiling stops at
280/300. Therefore the mask accounts for one failure but cannot explain the
remaining learned 0.90 ranking errors. The allowed train2863 deployed plus
one-hop pool contains 14,087 strict 0.90 boundary states, comprising 1,834
deployed and 12,253 one-hop states, 117,153 positive-negative action pairs,
and 428 strict structural signatures.

The D1A decision is `STOP_TRAINING_APPLY_SCORING_PROTOCOL_FIX_FIRST`. The
deployable scoring fix keeps the existing multi-anchor aggregation but treats
attainable 0.90 as conservatively active instead of allowing frozen predicted
progress to suppress it. Target-local scoring is not adopted. Pairwise or
representation training remains unauthorized until the existing boundary
corpus is tested for direction learnability under frozen representations.
Development, hop two, trajectory loss, terminal reranking, continuous cost,
learned cutoff, and confirmation remain closed.

## V17-D1B frozen boundary separability audit

D1B extracted the 3,650-dimensional frozen input to the viability head for
106,986 actions in all 14,087 train-only strict 0.90 boundary states. Diagnostic
shared scorers used state-equal sampling and antisymmetric pair differences;
probe weights were neither saved nor used as model checkpoints. All model
components and development remained sealed.

The linear probe is locally effective but fails the preregistered cross-query
gate. Macro-state accuracy is 0.826 under state-grouped folds and 0.820 when
428 structural signatures are held out, but falls to 0.766 under the stricter
example-grouped split, below the frozen 0.80 threshold. Query-balanced sampling
does not repair this result (0.765). A matched-capacity nonlinear diagnostic
improves local and signature-held-out accuracy to 0.883 and 0.859, but worsens
example-grouped accuracy to 0.732, which is consistent with query-specific
overfitting rather than insufficient head capacity.

The one-time ten-event internal confirmation also fails decisively. The linear
probe gets 4/10 correct and preserves only three of the four events already
correct under D1A. Query-balanced linear training remains at 4/10, while the
matched-capacity probe reaches only 3/10. Structural support does not explain
the failures: only 1/5 supported events is corrected, compared with 3/5 events
without support.

Frozen-feature geometry is not degenerate within individual states. Nearest
action label consistency is 0.758, no opposite-label pair has cosine similarity
at least 0.999, median maximum same-label cosine is 0.843 versus 0.721 for the
nearest opposite label, and the median gap is 0.0814. Thus the representation
contains useful local information, but it does not yield a boundary direction
that transfers reliably across queries or to the confirmatory events.

The preregistered pairwise-calibration gate fails, and signature-held-out
performance does not support the boundary-coverage branch. The formal decision
is `GO_V17D2_REPRESENTATION_ADAPTER_DESIGN`: design, but do not yet train, a
minimal adapter aimed specifically at cross-query boundary invariance. Pairwise
head training is not authorized because it would fit a representation that
already fails example-held-out and internal transfer. Development, terminal
reranking, hop two, learned cutoff, confirmation, and all model training remain
closed.

## V17-D2A 0.90-branch representation-adapter protocol

D2A freezes a shared-trajectory adapter rather than introducing a new
target-routed decoder. A rank-8, bias-free linear residual correction follows a
parameter-free LayerNorm on the frozen 3,650-dimensional action representation.
Only the corrected 0.90 logit replaces its original value; the 0.60, 0.70,
0.80, and 0.95 logits are copied from the original frozen-head evaluation.
D1A's force-attainable-0.90-active rule and existing multi-anchor mean
aggregation remain unchanged. The zero-initialized up projection makes step
zero an exact no-op.

The architecture audit confirms exact isolation and initialization. All four
untargeted logits, the 0.90 logit, and the aggregate residual have maximum
absolute step-zero difference 0.0. Exactly 58,400 parameters are trainable,
all in the adapter. Canonical internal300 replay produces 300/300 identical
beam orders and exactly reproduces the D1A baseline: 296/295/294/280/238 by
anchor and 275 complete trajectories.

The possible D2B training protocol is frozen before any optimization: train2863
deployed plus hop-at-most-one strict 0.90 boundary states, state-normalized
pairwise logistic ranking, a dimensionless relative-L2 drift penalty over both
boundary and train-only reference states, three-fold query-grouped
out-of-fold evaluation, rank fixed at eight, fixed optimizer and final-epoch
selection, and no hyperparameter sweep. The train-only opening gate requires
both primary and query-balanced macro-state accuracy of at least 0.80 and no
fold below its corresponding D1B baseline before internal data may be read.

The D2A decision is `GO_V17D2B_ADAPTER_TRAINING`; this authorizes the single
frozen training protocol but does not start it. Internal300 remains
design-exposed research validation, and the ten historical critical events are
descriptive only. Development, fresh confirmation, hop two, terminal
reranking, and learned cutoff remain closed.

## V17-D2B 0.90-branch adapter train-only gate

D2B executed the single frozen rank-8 adapter protocol. The drift reference
corpus was made deterministic before optimization: all 6,200 deployed states
and all 28,865 one-hop non-boundary states, with the smallest legal action used
per state and deployed/one-hop query-balanced sampling during training. No
intermediate epoch or alternative rank was selected.

The adapter fails the train-only held-out-query gate in every fold. Macro-state
accuracy is 0.749, 0.721, and 0.763 versus the corresponding D1B linear-probe
baselines of 0.763, 0.759, and 0.775. The three-fold mean is 0.744, below the
frozen 0.80 threshold and below D1B's 0.766 mean. Macro-query accuracy is only
0.733, also below 0.80. Thus neither the absolute gate nor the no-regression
gate passes.

The formal decision is `STOP_V17D2B_HELD_OUT_QUERY_GATE`. By protocol, no
all-data checkpoint was trained and canonical internal300 was not read. The
result strengthens the D1B diagnosis: a rank-8 input correction passed through
the frozen nonlinear viability head does not create transferable 0.90 boundary
geometry. It does not justify opening a larger adapter automatically; the next
stage must first distinguish missing query-action interaction information from
an objective/head-coordinate mismatch. Development, confirmation, hop two,
terminal reranking, and learned cutoff remain closed.

## V17-E0 frozen information-bottleneck localization

E0 compared eight preregistered frozen action-dependent taps under the same
three query-grouped folds, per-dimension training-fold standardization, linear
probe, optimizer, and final-epoch rule. It included native candidate and
interaction tensors, the frozen GRU next-state and transition delta, explicit
candidate-query and transition-query diagnostic interactions, and the complete
3,650-dimensional action representation. No model component was updated.

No clean held-out-query tap passes the 0.80 gate. Candidate is best by
macro-state at 0.777, with macro-query 0.757 and fold macro-state values
0.775/0.763/0.795. The complete action representation reaches 0.770/0.751.
Transition-next reaches 0.769/0.757; transition delta reaches 0.767/0.754; and
transition-query interaction falls to 0.761/0.740. Paired query bootstrap
intervals show no reliable improvement over the complete representation for
candidate, transition-next, transition-delta, or either query interaction.

The existing frozen B1 0.90 head scores 0.808 macro-state and 0.799 macro-query,
but this is an in-sample descriptive reference: that head was trained on all
train2863 queries, including every diagnostic fold. It is therefore excluded
from the sidecar gate and cannot establish cross-query generalization. Its
relative strength is evidence that the labels can be fitted within the observed
queries, while the clean probes show that the learned direction does not
transfer reliably.

The decision is
`STOP_FROZEN_LINEAR_TAP_BRANCH_AUDIT_TARGET_AND_FEATURE_SUFFICIENCY`. There is
no evidence for an upstream scalar sidecar, and simple query/action or
query/transition products do not justify opening a new representation module.
The next admissible work is a train-only audit of label consistency, query-type
conditional structure, and whether the frozen inputs contain the information
needed to distinguish exact-DP future viability. Internal300, development,
confirmation, hop two, terminal reranking, and learned cutoff remain closed.
