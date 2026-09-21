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

## V17-E1A/B frontier information and local rank profile

E1A/B constructed deployment-available successor frontiers for every strict
0.90 boundary action. Each candidate was executed through the frozen history
GRU, the resulting selected/remaining state was rebuilt, and all legal next
actions were rescored. The audit compared local features, ten scalar successor
score statistics, mean/max pooled remaining candidate embeddings, and both
frontier variants concatenated with the local representation under the same
query-grouped out-of-fold probe protocol.

Frontier information does not improve held-out-query viability. The local
baseline reaches 0.770 macro-state and 0.751 macro-query. Frontier scalar alone
falls to 0.697/0.705; pooled frontier reaches 0.769/0.751. Local plus scalar is
0.771/0.751, with paired-query improvement 0.0010 and a 95% bootstrap interval
of [-0.0002, 0.0022]. Local plus pooled falls to 0.764/0.750. No frontier tap
passes the preregistered 0.80 gate or shows a reliable paired improvement.

The local OOF scorer has best-viable recall of 0.934/0.998/1.000/1.000 at
top 1/2/4/8. Those large top-k values are not evidence for protected beam
allocation: strict boundary states contain 79.0% viable actions on average, so
the random expected recall is already 0.790/0.974/0.99998/1.000. Recall@4 lift
is only 0.000018, far below the frozen 0.02 robust-search opening gate. A
state-local top-k policy would therefore add almost no selectivity at beam4 or
beam8, and does not justify choosing a protected/Pareto rule.

The decision is `GO_V17E2_EVIDENCE_PROGRESS_TARGET_AUDIT`. Frontier-aware
critic design and robust-beam rule selection remain closed. The next experiment
must test whether a more local, decomposable evidence-progress target is
consistent and learnable across queries, while separately auditing exact-DP
viability label ambiguity. Internal300, development, confirmation, hop two,
terminal reranking, and learned cutoff remain closed.

## V17-E2 evidence-progress target audit

E2 tested three preregistered local targets without training a deployable
model: raw next-state fidelity gain, fidelity gain clipped at the 0.90 target,
and immediate newly attained anchor count. Each used the same frozen local
representation, query-grouped folds, standardized linear pairwise probe, and
fixed optimization protocol.

Raw and clipped fidelity progress are non-tied in 13,515/14,087 states (95.9%)
but reach only 0.734/0.729 and 0.735/0.734 macro-state/macro-query accuracy.
New-anchor gain is non-tied in only 6,629 states (47.1%) and reaches
0.729/0.729. Every fold remains below 0.80 for every target.

The local targets also have a consequential tie problem. At least one
maximum-progress action preserves exact 0.90 viability in 97.8--98.0% of
non-tied states, but all tied maximum-progress actions are safe in only 80.4%
for fidelity progress and 84.7% for anchor gain. A controller would therefore
need another unavailable tie-breaking signal and could not safely treat local
progress as a replacement objective.

The decision is
`STOP_EVIDENCE_PROGRESS_AUDIT_LABEL_AND_INPUT_SUFFICIENCY`. No progress
controller is authorized. Together E0--E2 show that local viability, explicit
successor-frontier summaries, and immediate evidence progress all fail clean
cross-query gates. The next admissible stage is a label/input sufficiency audit:
measure observational conflicts, query-family dependence, and whether distinct
exact-DP outcomes are distinguishable from deployment-available inputs before
designing another learned controller. Internal300, development, confirmation,
hop two, terminal reranking, and learned cutoff remain closed.

## V17-F0 state sufficiency and target identifiability

F0 tested the hypothesis that existential binary viability hides a widespread
thin-versus-robust distinction. Oracle tree information was used only to define
diagnostic targets, never as a model input. Within binary-viable actions, R1 and
R2 are the fractions of ordered one- and two-step descendants that retain exact
0.90 reachability; successful-family count is the number of viable immediate
successor packets.

Shallow robustness heterogeneity is limited at one step but becomes material at
two steps. 2,084/14,087 states (14.8%) distinguish viable actions by R1 or
successful-family count, while 2,917 (20.7%) distinguish them by R2. The median
within-state range remains zero for all three targets. Thus existential labels
usually agree at immediate depth, but just over the preregistered 20% threshold
expose a thin-versus-robust distinction after two further actions.

The heterogeneous minority is also poorly identifiable from deployment inputs.
For R1, the best macro-state is 0.755 and the best macro-query is 0.726. For R2,
the best macro-state is 0.736 and the best macro-query is 0.719. Concatenating
local and frontier features does not reliably improve either target.
Successful-family results equal R1 because the number of legal next actions is
fixed within each state.

The decision is `GO_V17F1_ROBUST_SEARCH_WITH_WEAK_HEURISTIC_DESIGN`. This does
not authorize a robustness critic: no target/feature pair passes the prediction
gate. It authorizes only a protocol design for explicit search that does not
assume shallow robustness can be compressed into another accurate local
scalar. Binary viability, local progress, frontier summaries, and shallow
robustness all remain inadequate as learned standalone controllers. Internal300,
development, confirmation, hop two, terminal reranking, and learned cutoff
remain closed.

## V17-F1 robust-search replay

F1 replayed four preregistered beam8 pruning rules on all 546 train-only
queries contributing a strict 0.90 boundary state, with unchanged D1A scoring
and terminal selection. The canonical global rule succeeds at 0.90 on 328
queries and completes 312 trajectories, with 195 first irreversible prunes.

Top-six plus two uncovered-parent slots produces exactly zero repairs and zero
breaks, while increasing first prunes to 198. The fixed random-parent control
has the same contract counts and 198 first prunes. Full parent balancing makes
one 0.90 repair but also one 0.90 break, loses one complete trajectory, and
increases first prunes to 213. It therefore fails both the zero-break gate and
the requirement to outperform the random structural control.

The decision is `STOP_V17F1_ROBUST_SEARCH_BRANCH`. Parent coverage by itself is
not the missing search invariant: mild protection has no effect, while strong
protection displaces useful high-score branches and causes harm. No internal or
development replay is authorized. Together F0 and F1 show that shallow branch
robustness exists in a minority of states, cannot be reliably predicted, and
is not repaired by generic parent-diversity allocation. The next stage must
reassess the controller/input contract rather than add another scalar target or
beam-allocation heuristic.

## V17-F2 cached target-behavior observability audit

F2 tested the distinct information-source hypothesis on the same 14,087
train-only strict-0.90 boundary states from 546 queries. It used cached parsed
frozen-target predictions at each current state and one-action successor, never
gold answers or cached F1/fidelity as inputs. Four linear pairwise probes used
the frozen three-fold example/query split and fixed training protocol. This is
an offline all-candidate information ceiling, not a deployable controller.

The local-only probe reached macro-state 0.7704 and macro-query 0.7505.
Candidate-answer behavior alone reached 0.5956 and 0.6108. Adding it to local
features reached 0.7659 and 0.7487; the paired query improvement was -0.0016
with 95% bootstrap interval [-0.0097, 0.0066]. Shuffling candidate behavior
within each state reached 0.7677 and 0.7461. The added answer signal fails the
predeclared 0.80 state/query/fold thresholds, the positive paired-improvement
criterion, and the 0.02 margin over the shuffled control.

The probe cost is substantial even before model design: 14,087 current-state
and 106,986 successor calls without memoization (121,073 total), or 71,960
unique masks with per-query memoization. Cached context text alone accounts for
at least 35.9 million tokens without memoization, or 21.8 million with
memoization; these are lower bounds excluding question/prompt and generated
output tokens, latency, and any KV reuse. The cache does not make calls free at
deployment.

The decision is `STOP_V17F2_CACHED_ANSWER_OBSERVABILITY`. Parsed answer text did
not improve cross-query boundary identification. This does not rule out
log-probability, hidden-state, or bounded-rollout behavior, which the current
cache does not contain; nor does it prove that every closed-loop method fails.
No controller training or internal300/development/confirmation access occurred.
The D1A 280/300 and complete 275/300 remain the canonical internal baseline.
Protocol, folds, cost ledger, and decision are in
`configs/v17f2_target_behavior_observability.json` and
`results/v2_rank_then_cut/v17f2_target_behavior_observability/`.

## V17-F3 bounded logprob observability pilot

F3 tested a richer frozen-Target signal without changing the deployable policy.
The protocol selected 180 distinct train-only strict-0.90 boundary queries by
hash and one deployed-preferred boundary state per query. It reran Qwen3-8B
with the atomic QAMPARI prompt, deterministic generation, and top-two generated
token logprobs. The first 12-query preflight reached 65/70 (92.9%) exact parsed
answer agreement with the old cache, above the frozen 90% opening criterion.
The full 1,191 calls reached 88.9% agreement, however, so this pilot also
exposes a nontrivial old-cache/new-run reproducibility limit.

On the same 180 one-state-per-query grouped folds, the local-only linear probe
reached macro-query 0.7101. The five-number logprob signal alone reached 0.5200.
Local plus logprob reached 0.7028, a paired query difference of -0.0060 with
95% bootstrap interval [-0.0202, 0.0033]. It fails the predeclared 0.80
absolute gate, +0.03 paired improvement, and positive lower confidence bound.
This local baseline is lower than F2's 0.751 because F3 uses only one selected
state per query; only within-F3 comparisons are causal.

The signal consisted of mean/minimum chosen-token logprob, mean top-two token
margin, generated length, and current-to-candidate mean-logprob change. These
whole-generation values include fixed answer tags and are not calibrated
probabilities of a set-valued answer or full-vocabulary entropy. The 1,191
Target calls consumed 601,972 prompt tokens and 40,254 generated tokens. This
cost was offline only; no Target feature enters deployment.

The decision is `STOP_V17F3_LOGPROB_PILOT_NO_TEACHER_SIGNAL`: the tested
low-dimensional logprob summary does not justify student distillation or an
internal replay. The decision is scoped to this one-state-per-query pilot and
these features. The 88.9% cache agreement and small grouped folds preclude a
general claim that richer Target state or all uncertainty measures fail. No
student/controller was trained, and internal300, development, and confirmation
remained sealed. The D1A 280/300 canonical internal baseline is unchanged.

## V17-G0A threshold-label reproducibility audit

G0A held all model training and validation roles sealed. From F3's 180
train-only queries, it selected 154 masks across 70 queries: all current and
legal successor masks for 12 strict-boundary states, plus hash-selected masks
at actual list-F1 values 0.888889, 0.900000, 0.909091, 0.947368, and at most
0.70. Each prompt was regenerated three times in one Qwen3-8B/vLLM process
using the original atomic QAMPARI prompt and greedy 256-token contract.

Against the original exact cache, 462 comparisons agree on exact parsed answer
string 82.9%, order-insensitive normalized answer set 86.1%, list-F1 86.1%,
and the 0.90 success label 96.75%. Seven of 154 masks cross the success
threshold in at least one rerun; four masks cross it among the three new runs
themselves. This demonstrates real threshold-label instability in the sampled
environment, beyond harmless answer-string changes. It does not estimate a
population flip rate because the sample deliberately enriches boundary values.
The changes are not solely tiny 0.899-to-0.901 perturbations: one cached
0.947368 mask reran at 0.181818, and another at 0.736842.

The fidelity support is discrete: with ten answer atoms, list-F1 is
`2 * correct / (10 + predicted)`. In the 180-query exact lattice there are only
four 0.909091 masks, compared with 379 at exactly 0.900000 and 11,096 at
0.888889. Therefore an arbitrary continuous `[0.89, 0.91]` gray zone is not
appropriate without first checking which discrete answer-count configurations
it actually removes. G0A used 462 Target calls, 305,697 prompt tokens, and
20,965 generated tokens. It did not use internal300 or development.

## V17-G0B localized DP-label sensitivity

G0B held all unobserved exact-lattice masks fixed and substituted only the
seven masks whose G0A reruns crossed 0.90, one rerun index at a time. It then
recomputed exact 0.90 existential reachability and compared the train2863
critical state-action targets. The baseline recomputation reproduced every
checked stored viability label (692/692 on the three affected queries).

Three of the seven queries cause any critical-label change. Taking each
affected query's most sensitive rerun separately, 33 critical action labels
change in total, including
one deployed-state action label; the remainder are one-hop states. Repeated
scenarios count 58 action-label changes, but those are correlated repeats of
the same query/mask perturbations and must not be treated as 58 independent
examples. The other four query flips are absorbed by alternative successful
continuations and change no reachability label. This shows both that a real
mask-level flip *can* propagate into strict boundary supervision and that DP
redundancy often protects it.

This is a localized sensitivity test, not a second exact lattice generation or
an estimate of stochastic viability prevalence. The formal decision is
`AUDIT_TARGET_REPRODUCIBILITY_BEFORE_SUPERVISION_REDESIGN`: first distinguish
same-prompt batch/numeric variation from environment or cache-contract drift,
then decide whether probabilistic labels are warranted. The 0.90 final
evaluation contract and D1A baseline are unchanged. No training or sealed-role
replay was authorized. Protocols and per-mask/per-query audit rows are under
`configs/v17g0*.json` and `results/v2_rank_then_cut/v17g0*/`.

## V17-H0 frozen-candidate multi-anchor cutoff pilot

To test cutoff development separately from upstream ranking, H0 froze the V13
top-four/V8-fallback candidate trajectories, V13 embedding cache, and V14
candidate choices. It trained one five-anchor cutoff head for 400 fixed steps
on train2863 prefix states, with a hash-grouped 606-query train-only holdout.
Labels came from the exact prefix cache; unattainable anchors were masked. The
one training prefix already known from G0A to have an unstable 0.90 label was
masked for that loss. The loader verified every candidate's per-anchor oracle
success against the exact cache before training. No Target calls, development,
or confirmation were used. The internal300 replay is design-exposed and is
reported once for diagnosis, not for choosing a threshold or checkpoint.

**Evaluation correction.** The original H0 summary converted exact-cache
fidelity to float32 and then compared Python scalars using `+1e-12`. Values
equal to a threshold, notably 0.900000, could then be misclassified as below
it. The checkpoint was not retrained. A read-only replay uses `1e-6` tolerance,
consistent with the exact-cache and tensor labels; the original summary is
preserved as `original_summary_before_float32_correction.json`. All figures
below are corrected. The earlier 223/300 and 57-gap statement is superseded.

The cutoff head still fails the contract. On the train-only holdout, the V8
fallback reached 0.90 on 510/606 versus 567/606 with an oracle cutoff on the
**same fixed orders**; complete was 360/606 versus 552/606. On internal300,
the frozen V14 candidate selector plus H0 cutoff reached 0.90 on 243/300
versus 280/300 with same-order oracle cutoff; complete was 165/300 versus
277/300. The V8 fallback was 245/300 versus 279/300 at 0.90. Even the
nondeployable oracle candidate choice plus H0 cutoff reached only 252/300
versus its same-order oracle ceiling of 285/300. Mean regret among complete
V14/H0 trajectories was 0.13147, above the 0.03 target.

This separates two bottlenecks. The V14-selected fixed-order oracle ceiling of
280/300 is below the 282/300 0.90 gate, so **a cutoff-only replacement cannot
clear that gate with this selector**, even if its cutoff were perfect. H0 also
falls far below its own fixed-order oracle: 37 additional 0.90 failures and
112 additional incomplete cases on the V14-selected internal trajectories.
For 0.90, 209/300 chosen cutoffs were at packet 10 and 60/300 were at packets
11–12; late stopping can still fail because observed fidelity is not monotone
along add-only trajectories. These are diagnostic counts, not proof that timing
alone caused every failure. The fixed per-prefix BCE and 0.90 probability
crossing rule is therefore not validated as a deployable cutoff protocol.

Decision: **STOP_V17H0_CUTOFF_PILOT; no confirmation or deployment claim.** Do
not tune the probability threshold on internal300. The fastest scientifically
useful follow-up is a train-only error decomposition of H0 probabilities and
oracle-first-success positions, followed by a *new frozen protocol* that
optimizes ordered stopping under nonmonotone fidelity. That protocol must pass
query-held-out cutoff success and regret before one internal replay. Separately,
the fixed candidate selector must gain at least two 0.90 successes to reach
282/300 even under oracle cutoff. G0's Target/cache reproducibility concern
also remains unresolved for any eventual fresh confirmation. The H0 protocol,
checkpoint, history, internal per-example outcomes, and machine-readable
summary are in `configs/v17h0_multianchor_cutoff_pilot.json` and
`results/v2_rank_then_cut/v17h0_multianchor_cutoff_pilot/`.

## V17-CUT-A0 fixed-trajectory stop-error audit

CUT-A0 used only H0's 606-query train2863 holdout and the unchanged H0
checkpoint. With the corrected threshold comparison, it reproduced 510/606
0.90 successes and 360/606 complete, versus fixed-order oracle counts of
567/606 and 552/606. Its 57 recoverable 0.90 failures split into 35 early
stops, 17 late stops where fidelity had fallen back below 0.90, and five cases
where an earlier-anchor cutoff blocked every later 0.90-successful prefix.
Another 39 trajectories had no 0.90-successful prefix at all. The median
predicted 0.90 sufficiency at the first available successful prefix among the
52 failures with such a prefix was 0.779, below the frozen decision threshold
of 0.90.

The prespecified same-holdout threshold scan reached at most 516/606 0.90
successes at threshold 0.95, still well short of the 567/606 fixed-order
oracle. Complete improved to 391/606 there, while mean regret among complete
trajectories increased from 0.1344 to 0.1817. These are optimistic,
non-independent diagnostics on the same holdout, **not** a selected new
threshold or a validation result. The scan does not support a simple
probability-threshold fix; observed errors include both early and
nonmonotone-late failures. No internal300, development, confirmation, Target
calls, or training were used in CUT-A0. Results and per-query classifications
are under `results/v2_rank_then_cut/v17cut_a0_fixed_trajectory_stop_audit/`.

## V17-SEL-A0 candidate ceiling audit

SEL-A0 separated the frozen V14 selector from the five-candidate pool at 0.90.
In the design-exposed internal300, the selected order is reachable for 280
queries; another five have a successful order in the frozen candidate pool but
the selector chose a failing one. For the remaining 15, none of the five
candidate orders has a successful prefix, yet **all 15 have at least one
successful state somewhere in the complete 4096-mask exact lattice**. Thus the
current 285/300 candidate-pool ceiling is a generator/order coverage limit,
not an impossibility of the 12-packet add-only action space. The 20 selected
failures are exactly five selector misses plus 15 candidate-pool misses. This
audit cannot attribute those 15 to a specific beam prune without replaying the
generator's beam trace.

Train2863 candidate-space counts are 2643/2863 for the V8 fallback order and
2739/2863 for the five-candidate oracle, but they are descriptive only: V14's
selector was trained on train2863 and its checkpoint was selected using the
internal role. Neither count validates a new selector. The full-lattice check
was restricted to the 15 internal pool misses. No model was trained and no
development or confirmation data were used. The per-query categories are in
`results/v2_rank_then_cut/v17sel_a0_candidate_ceiling_audit/`.

Together CUT-A0 and SEL-A0 support the next research order: first improve
candidate coverage/selection under oracle cutoff using a train-only grouped
protocol, then freeze that candidate policy and train a selector-matched
stopping model. H0's cutoff cannot be assumed to transfer to changed orders.
The existing internal300 is already design-exposed; any future claim of
generalization requires a properly held-out role and the unresolved G0
Target/cache reproducibility check.

## V17-SEL-B0 projected-trajectory coverage localization

SEL-B0 computed, for each frozen V14 proposal prefix, whether any successful
exact-lattice mask is a superset. A proposal that had already reached a
successful prefix was counted as covered thereafter, so later fidelity
rollback could not erase earlier coverage. This is a structural reachability
diagnostic, not a learned selector evaluation or a beam causal trace.

All 2863 train queries have some successful exact 0.90 mask, but 124 have no
successful prefix in the five current projected orders. Their first depth at
which *all* proposals lost successful completions is distributed from 1 to 11:
8, 9, 4, 7, 13, 8, 14, 12, 19, 28, and 2 queries, respectively. At that
first loss, 123/124 had only one distinct viable parent prefix remaining;
this describes narrow proposal coverage at the bottleneck but does not prove
that a diversity rule would repair it. The prior design-exposed internal300
shows 15 pool misses, with first-loss depths 2:2, 4:1, 6:3, 7:1, 8:2, 9:5,
and 10:1. All 15 also have globally successful exact masks, reproducing
SEL-A0. Because V14 candidates are V13-mask projections of one V8 order,
first loss cannot by itself distinguish original generator shortlist recall
from beam pruning.

## V17-SEL-B1 frozen V13 proposal-pool expansion ceiling

SEL-B1 replayed the **unchanged** V13 step300 mask retrieval head with V14's
original batch32/bfloat16 inference path. It asserted exact agreement with
the saved top-four mask rankings on all 606 H0 hash-fold queries before
evaluating any additional proposals. Under the same V8-order projection,
oracle 0.90 coverage was 567/606 for V8 alone, 582/606 for top-four,
599/606 for top-ten, 602/606 for top-sixteen, and 605/606 for top-twenty;
the full exact lattice reached 606/606. Among the 24 top-four misses, 17
first acquire a successful projection at ranks 5–10, six at ranks 12–20,
and one remains uncovered at top-twenty. This is a large candidate-space
headroom gain without retraining the generator.

The numbers are **oracle-pool ceilings**, not deployable results. For the
single 0.90 existential metric, oracle selecting five from top-twenty has
the same ceiling as oracle selecting one: both merely ask whether any
successful proposal exists. A learned, label-free five-slot selector may be
harder with twenty candidates than with four. The hash fold was held out from
H0 cutoff training but **was not held out from V13 retriever training**;
therefore SEL-B1 is a mechanism/capacity audit, not an end-to-end
generalization estimate. It did not use internal300 to choose K, did not
train, and did not access development, confirmation, or the Target. The
frozen top-K list and per-query outcomes are under
`results/v2_rank_then_cut/v17sel_b1_frozen_topk_coverage/`.

The next useful experiment is a train-only, query-grouped protocol for
selecting a small proposal set from the expanded frozen pool, with explicit
checks that the retriever's prior exposure to these queries does not get
misrepresented as independent validation. Oracle coverage, deployable
selection, multi-anchor complete trajectory, and later learned-cutoff error
must be reported separately. No top-K expansion is yet authorized as a
deployed improvement.

## V17-SEL-B2A frozen V14 selector on expanded proposals

B2A asked whether the existing selector could directly exploit the B1
candidate-pool headroom. V14's rank embedding has indices 0–4 only, so ranks
5–20 cannot literally use the unchanged input contract. This diagnostic kept
all V13/V14 weights and the selected V14 utility threshold fixed, mapped
every new rank to the last trained rank bucket (4), and used oracle cutoff
only to isolate selection. V13's original batch32/bfloat16 path reproduced
all saved top-four masks on the 606 H0 hash-fold queries before expansion.
The fold was seen during V13 and V14 training, and rank clamping is outside
V14's training support; results are strictly mechanistic.

For top-four, the frozen selector selected a 0.90-successful order on 567/606
queries and a complete order on 553/606, versus pool oracle coverage 582/606.
With top-ten, pool coverage rose to 599/606 and the selector chose a new
candidate on 29 queries, yet 0.90 remained 567/606 and complete remained
553/606: zero repairs and zero breaks. With top-twenty, pool coverage was
605/606 and it chose a new candidate on 63 queries, but 0.90 fell to
566/606 and complete to 552/606: one repair and two breaks. Oracle-pool
utilization fell from 97.4% at top-four to 94.7% at top-ten and 93.6% at
top-twenty. The mask-retrieval head still scored all 4096 masks per query;
selector inference grew from 5 to 11 or 21 scored candidates. No additional
Target calls were made.

This does **not** prove a new selector cannot exploit the larger pool: old
V14 supervision, rank vocabulary, and threshold were all designed for top-four.
It does show that simply widening the pool under this frozen selector and
rank-clamp protocol fails to convert additional oracle coverage into useful
decisions, even on training-exposed queries. Because neither B1 nor B2A gives
clean cross-query evidence for V13 top-K coverage, the next generalization
test requires a query-held-out **upstream refit**, not just a V13-head refit:
V13 initialized from V8 and V12 artifacts that were themselves trained on
train3163, including the H0 hash fold. Reusing those initial weights in a
nominal cross-fit would still leak the held-out queries. A full upstream
cross-fit is expensive and must have its own costed, frozen protocol before
launch. B2A's per-query repairs, breaks, and choices are under
`results/v2_rank_then_cut/v17sel_b2a_frozen_selector_replay/`.

## V17-SEL-B2B lineage-clean top-K pool gate

The frozen B2B split reserved 581 train-slice queries as a lineage-clean
holdout and 550 separate queries for upstream checkpoint selection. V8 was
trained on the remaining 2,032 train-slice examples, followed by new V10,
V12, and V13 runs initialized only from clean predecessors. V10's historical
single endpoint was used as initialization without opening its old development
evaluation. V8 step500 was selected on inner validation: 0.90 success was
508/550, below its rate-normalized gate of 517/550, so this is a best
available checkpoint rather than a passed V8 end-to-end gate. V13 step100
had the minimum inner-validation retrieval loss and was selected before the
holdout was accessed. An initial V13 attempt had a GPU-contention OOM before
its first checkpoint; the run was restarted from the same frozen inputs and
hyperparameters on an otherwise free GPU.

On the 581 lineage-clean queries, oracle 0.90 pool coverage was **552/581
(95.0%)** at top-4, **565/581 (97.25%)** at top-10, and **572/581 (98.45%)**
at top-20. V8 fallback alone covered 539/581. Top-10 improved on top-4 by
13/581 = 2.24 percentage points, passing the preregistered 1.5-point gain
and 97% absolute-ceiling gates. The resulting decision is
`GO_SEL_C0_TOPK_SET_SELECTOR_PROTOCOL`: an authorization to design a
selector-training protocol, **not** evidence that a deployable selector or
cutoff works. The absolute gate was passed by only 0.25 percentage points;
this single design-exposed holdout should not be presented as a precise
generalization guarantee. No development, confirmation, or Target calls were
used in B2B.

The run status and stage logs are under
`results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/`. Monitor with
`watch -n 10 bash scripts/93_monitor_v17sel_b2b.sh`. This run does not train a
selector or cutoff, and it does not access development or confirmation data.

## V17-SEL-C0/C1 train-only quality–token test

C0 fixed one selected trajectory at deployment, frozen B2B V8/V13, top-4
versus top-10 arms of the same candidate-set scorer, a query-grouped selector
train/validation split within the 2,032 clean-train queries, fixed optimization
and checkpoints, and a set-valued successful-prefix loss with a small
within-success token-cost term. Identical projected orders were deduplicated
before set attention and top-1 choice. The 581-query B2B holdout remained
closed throughout this stage. No cutoff, development, confirmation, or Target
calls were used.

The train-only oracle cost audit found 1,941/2,032 top-4 versus 1,996/2,032
top-10 reachable 0.90 trajectories. The 55 newly covered queries required a
mean 80.4% of their full-context tokens at their earliest 0.90-successful
prefix, with mean global-lattice token regret 0.0147 of full-context tokens.
Thus much of the new coverage is genuinely expensive in context tokens, but
the projected order is close to the exact-cache minimum for those queries.
On the 611-query selector-validation subset, the corresponding oracle counts
were 587 and 601; top-10 reduced the failure-penalized oracle token fraction
from 0.7231 to 0.7086.

The learned selector did not turn this opportunity into a quality–token
improvement. On the same 611 queries, top-4 selected one 0.90-successful
trajectory on 570 and top-10 on 572, with 12 repairs and 10 breaks. Complete
was 538 versus 543. However, the mean failure-penalized earliest-0.90 token
fraction rose from **0.7519** to **0.7667**. The fixed V13 rank-1 candidate
alone had 567 successes, 559 complete trajectories, and fraction 0.7405;
the V8 fallback had 565, 551, and 0.7409. Among the 14 new top-10 oracle
opportunities in validation, the top-10 selector converted only five. It
selected ranks 5–10 on 466/611 queries, and among the 560 queries where both
arms succeeded, its choice used more tokens on 240 and fewer on 112, with a
mean +15.6-token difference. This is a failure of the frozen C1 selection
protocol, not a refutation of the top-10 oracle pool ceiling.

The preregistered quality–token gate therefore returned
`STOP_SEL_C1_TRAIN_ONLY_GATE`; the 581-query holdout was **not read again**.
Do not tune the selector against those 581 outcomes or treat this checkpoint
as deployable. The train-only cost audit, both selector checkpoints, summary,
choices, and read-only error decomposition are under
`results/v2_rank_then_cut/v17sel_b2b_lineage_clean_holdout/`.

## V17-SEL-D0 train-only tail opportunity audit

D0 read only the same 2,032 clean-train candidate records. The 581-query
lineage-clean gate, internal300, development, confirmation, cutoff, and Target
were not accessed. The frozen top-4 **pool** is preserved; the following are
oracle pool ceilings, not deployable top-1 outcomes. The artifact and its
per-query audit are in `v17sel_b2b_lineage_clean_holdout/sel_d0_train_only/`.

Top-4 covers 1,941/2,032 at 0.90. Full top-10 covers 1,996/2,032, adding
55 success opportunities and lowering the oracle failure-penalized token
fraction from 0.72692 to 0.71058. It also finds an earlier successful
prefix on 352 queries that top-4 already covers. Thus tail opportunity is
not limited to the 55 new successes, although no learned policy has yet
converted this oracle token saving into a deployed benefit.

An adaptive *per-query oracle* with one tail slot equals full top-10 coverage
by definition: whenever any tail candidate succeeds, it chooses that one.
It says nothing about whether a controller can select the tail without
labels. Among the 55 new opportunities, 30 have exactly one successful tail
rank; their first successful ranks are distributed across rank 5 through 10
as 15, 9, 13, 6, 5, and 7. A precommitted single rank covers at most 21/55
new opportunities (rank 7). Selecting two fixed ranks (7 and 10) solely on
the 1,421-query training fold recovers 24/41 there, but only 8/14 on the
611-query selector-validation fold. The corresponding validation oracle
counts are 587 top-4, 595 with those two ranks, and 601 full top-10.

Decision: `DO_NOT_INFER_ONE_ESCAPE_SLOT_DEPLOYABILITY_FROM_ORACLE_COVERAGE`.
The next candidate-policy experiment must jointly test **whether to expand**
and **which tail candidate to use**, under a fixed final-output budget and a
specified final chooser. A gate that predicts only “need expansion” is
insufficient. Its train-only criterion must compare actual repairs, breaks,
Complete, failure-penalized context tokens, and added inference cost against
the V13 rank-1, V8 fallback, and learned top-4 baselines. No D0 oracle number
is evidence that the C1 top-4 selector is a Pareto-dominant deployed anchor.

## V17-SEL-D0B baseline-relative value and SEL-D1 conservative switch

D0B fixed the clean-lineage V8 rank-0 trajectory as STAY. On the 1,421-query
selector-training fold it had 1,315 oracle 0.90 successes and 1,294 complete
trajectories, versus 1,314 and 1,290 for V13 rank-1. The other 611 queries
had already been used in C1 and are explicitly design-exposed, so neither this
anchor choice nor D1's later result gets a fresh-gate interpretation. D0B
re-read only the 2,032 train-side exact lattices, verified every candidate's
cached success/complete/cumulative-token labels, and flagged known G0A 0.90
flips on candidate prefixes. It made no Target calls.

Relative to STAY, 151/2,032 queries have a unique rank-5--10 action with a
non-breaking oracle repair at some attainable anchor or Complete; 113 can
repair 0.90 and 140 can repair Complete (these categories overlap). Another
502 queries have a unique tail action with the same success vector and
Complete but lower *legal cumulative multi-anchor* tokens. These are
candidate-wise oracle opportunities, not learned-policy results. A rank-1--4
action offers a repair on 94 queries and safe token saving on 203, so a V8
anchor cannot scientifically restrict its action set to tail ranks alone.
Among the 295 tail repair actions that reach 0.90, 194 have only one
consecutive successful prefix beginning at first attainment. This is a
descriptive stopping-robustness warning, not a training feature or a repeated
Target stability estimate. The per-query D0B table and summary are under
`v17sel_b2b_lineage_clean_holdout/sel_d0b_train_only/`.

D1 then executed **one** frozen 300-step run of the unchanged C1 set scorer,
now trained with baseline-relative, set-valued preferred actions and STAY in
the action set. A read-only endpoint replay found STAY on **1,421/1,421**
training queries as well as **611/611** design-exposed queries. Thus this is
an objective/prevalence collapse, not only a held-out generalization miss. Its
oracle-cutoff 0.90/Complete counts were therefore exactly the V8 reference
565/611 and 551/611, with zero repairs, zero breaks, and zero paired token
change. V13 rank-1 alone had 567/611 and 559/611 on those same queries.
This is `STOP_SEL_D1_CONSERVATIVE_SWITCH_PROTOCOL`: the particular frozen
supervision/architecture produced no realized improvement. It does not prove
tail evidence has no value or that every conservative selector must fail. No
threshold, class weight, checkpoint, or rank sweep follows from this run.
The 581-query B2B role was not read; neither was internal300, development, or
confirmation. D1's frozen checkpoint and choices remain local; its protocol,
code, and summary are in the repository.

## V17-CUT-B0 fixed-V8 structured stopping pilot

CUT-B0 independently tested the proposed structured stopping objective on the
same clean-lineage fixed V8 trajectory. Its training label is the *set* of
minimum-cumulative-token, fidelity-valid, nondecreasing five-anchor stopping
vectors. The model partitions over all nondecreasing vectors; inference uses
only its scores and the known attainable-level mask, never exact fidelity.
It trained one fixed step-400 endpoint on the 1,294 complete trajectories in
the 1,421-query training fold. Incomplete orders were excluded from gradient
but kept in evaluation. All 2,032 rows were checked against the frozen exact
candidate labels before training.

On the design-exposed 611, its selected stops attained 0.90 on **501/611**
versus **565/611** for a same-order oracle cutoff; Complete was **179/611**
versus **551/611**. Per-anchor successes were 531/611, 431/611, 265/611,
501/611, and 394/481 at 0.60 through 0.95 respectively. Most predicted
stops clustered at depths 6--8 for the first three anchors and depth 10 for
0.90. The endpoint is `STOP_CUT_B0_STRUCTURED_MIN_COST_IMITATION`: this
specific objective did not solve nonmonotone stopping and cannot be deployed.
The H0 numbers came from a different candidate lineage/split and are not a
paired quantitative comparator. No second objective, hyperparameter sweep,
581 read, internal replay, development, or confirmation was performed. The
protocol, code, summary, and local checkpoint are under
`v17sel_b2b_lineage_clean_holdout/cut_b0_train_only/`.

These independent negative results leave the final Pareto claim unachieved.
The next design should first test whether a deployment-visible statistic can
discriminate rare beneficial candidate switches and whether the optimum stop
vector is learnable rather than merely minimizing cached oracle cost. More
training of these two frozen protocols is not justified by their present
train-only evidence.

**CUT-B0 inference-contract correction.** Its frozen protocol called
`attainable_levels` a known mask, but that field was derived from the same
exact-lattice supervision and is not automatically available to a deployed
controller. A read-only checkpoint replay decoded all five requested anchors
without this mask, then scored only the benchmark's attainable anchors. On
the design-exposed 611, including 130 four-anchor examples, all active stop
vectors were identical to the original run: 501/611 at 0.90 and 179/611
Complete in both. Thus this particular result is numerically unaffected,
but future deployment-identifiability audits must not treat exact-derived
attainability as an observable input. The unchanged frozen protocol and the
separate replay summary are preserved under `cut_b0_mask_replay/`.

## V17-DEP-A0 train-only deployment-decision identifiability audit

DEP-A0 used only the 1,421 clean-lineage training queries (four query-grouped
out-of-fold rounds). Fold 4's 611, B2B's 581, internal300, development, and
confirmation were not read. It made no Target calls and produced diagnostic
linear probes, not deployable checkpoints. The frozen protocol and compact
summaries are under `v17sel_b2b_lineage_clean_holdout/dep_a0_train_only/`.

For candidate switching, 7,831 unique rank-5--10 tail actions had a 12.34%
beneficial-action prevalence under D0B's baseline-relative oracle labels.
Frozen retrieval/rank/size scalars reached action AP 0.185; the full frozen C1
feature difference reached AP 0.157. At the prespecified 5% query-switch
budget, the scalar probe's *oracle-stopped* replay yielded 5 repairs and 1
break at 0.90, also 5 Complete gains and 1 loss, among 69 switches. For the
50 switched queries Complete under both actions, the mean legal cumulative
token change was **+79** (more tokens). The full-feature probe yielded zero
0.90 repairs and two breaks at the same budget. These results show weak
ranking signal but no demonstrated quality--token Pareto gain; oracle stopping
also makes them an upper-bound candidate diagnostic, not deployed behavior.
The fixed-rank-5 control occasionally matched or beat the learned tail choice,
so the probe has not established reliable *which-tail* selection.

For stopping, DEP-A0 used the fixed V8 order and the easier unconditional 0.90
safe-prefix label. There were 1,315 reachable orders among 1,421 queries and
86 orders with exactly one safe prefix. A depth-only prior, fitted in the
other three training folds, hit 1,195/1,315 reachable safe windows, including
33/86 one-prefix windows. A linear probe using query, selected-prefix mean,
depth, and token fraction hit 1,184/1,315 and 30/86; adding the last packet
hit 1,180/1,315 and 31/86. Its safe-prefix AP improved slightly (0.826 versus
depth prior 0.815), but the final stop decision worsened. The probe chose the
highest-scoring prefix among all 13 positions, so it already assumes the
whole candidate order can be inspected before stopping. A truly online
cutoff has no stronger observability under this setup. These are 0.90-only
diagnostics and cannot replace the five-anchor Complete evaluation.

Decision: `STOP_CURRENT_FEATURE_SEL_CUT_VARIANTS`. D1's STAY collapse and
CUT-B0's depth collapse are consistent with weak conditional decision signal,
but DEP-A0 does not establish an information-theoretic impossibility. It only
tests the frozen feature sets and linear diagnostics above; finite-sample
uncertainty also matters for the five rare selector repairs. The next
experiment must change the *deployment-visible evidence or action/stopping
contract* and compare end-to-end quality and context-token cost, rather than
train another local scorer or tune a threshold on the same information.

## V17-DEP-B0 frozen raw-text lexical order control

DEP-B0 tested a different deployment-visible evidence path without training:
standard BM25 word overlap between the question and each of the 12 original
packet texts, with fixed constants and V8 order only for ties. It produced a
single add-only order per query. Evaluation read the exact lattice only after
the orders were fixed. As in DEP-A0, only the 1,421 clean-lineage training
queries were read; 611, 581, internal300, development, and confirmation remain
sealed, and no new Target calls were made. All values below are *oracle-cutoff*
trajectory properties, not deployed stopping results.

| Train-only order | 0.90 reachable | Five-anchor Complete | Failure-penalized earliest-0.90 token fraction |
| --- | ---: | ---: | ---: |
| Clean V8 fallback | 1,315/1,421 | 1,294/1,421 | 0.7420 |
| Frozen V13 rank-1 | 1,314/1,421 | 1,290/1,421 | 0.7423 |
| Raw-text BM25 | 1,260/1,421 | 1,142/1,421 | 0.8466 |

Relative to V8, BM25 repaired 19 and broke 74 queries at 0.90; it gained 25
and broke 177 Complete trajectories. Among the 1,241 queries where both orders
could reach 0.90, BM25 first success required 83.9 more context tokens on
average. Among 1,117 paired Complete queries, its ordered five-anchor oracle
cost was 470.5 tokens higher. Decision:
`STOP_DEP_B0_SIMPLE_LEXICAL_ORDER`. Raw question--packet overlap alone is not
an improvement to candidate ordering or semantic compression, and this
negative result forbids adapting BM25 constants on the untouched fold. It does
not rule out a different packet representation or a richer causal evidence
signal, but such a proposal now needs a new train-only test that can preserve
V8's already successful trajectories.

## V17-CUT-C0 quality-first fixed-depth schedule control

CUT-C0 exhaustively enumerated nondecreasing five-anchor stop-depth vectors
using only 1,421 clean-lineage train-side queries. The fixed selection rule
maximized train-side Complete, then 0.90 successes, then minimized context
cost. It selected `(10, 10, 10, 10, 10)`. No exact fidelity, oracle label, or
attainability mask is needed by this inference rule. Only after the vector was
fixed were metrics computed for the already design-exposed 611-query fold;
B2B 581, internal300, development, and confirmation remained sealed. This is
a fixed-order/fixed-stop deployed policy control, not a learned cutoff result.

On the 611 queries, fixed depth 10 yielded 514/611 at 0.90 and 466/611
Complete, compared with CUT-B0's 501/611 and 179/611 on the same V8 order.
Its paired changes were 16 repairs and 3 breaks at 0.90, and 287 Complete
gains with no Complete break. Yet average legal five-anchor context fraction
over all queries rose from **0.5596** for CUT-B0 to **0.8175** for fixed depth
10. Among the 179 queries Complete for both, fixed depth 10 added **0.2267**
normalized context fraction on average; its mean normalized regret against
the same-order oracle on its own Complete cases was **0.2675**, far above the
project's 0.03 regret target. Thus fixed depth 10 is a necessary quality-first
control and exposes CUT-B0's over-early stopping, but it is **not** a verified
quality--token Pareto improvement. Its 0.90 and Complete counts also remain
below same-order oracle ceilings of 565 and 551 on this role.

Decision: `KEEP_FIXED_DEPTH_AS_CONTROL_ONLY`. The empirical tradeoff makes the
next cutoff question more specific: recover the robust depth-10 successes
while spending much less context on the easier queries. Any such method must
demonstrate per-query stopping information beyond depth and report both
Complete and all-query context cost against the fixed-depth and CUT-B0 controls.

## V17-CUT-C1 frozen V13 mask-value stopping signal audit

CUT-C1 evaluated the existing clean-lineage V13 ordinal mask-value head on
each prefix of the unchanged V8 trajectory. The head sees only frozen query and
packet embeddings plus the current mask; exact fidelity is used solely to
score the resulting decisions. A four-fold query-grouped linear calibration
using the five head logits, depth, token fraction, and 0.90 logit change was
fit only on the 1,421 train-side queries, with out-of-fold predictions for
every query. V13 itself had ancestry exposure to these queries, so a positive
result would have required lineage-clean verification. No Target calls were
made and 611, 581, internal300, development, and confirmation stayed sealed.

| Train-side 0.90 policy | Success / 1,421 | Mean context fraction | One-safe-prefix hits / 86 |
| --- | ---: | ---: | ---: |
| Fixed depth 10 | 1,195 | 0.820 | 33 |
| Direct V13 logit, first nonnegative prefix | 961 | 0.681 | 13 |
| Grouped-OOF calibrated logit, first nonnegative prefix | 1,060 | 0.726 | 18 |

The direct and fitted safe-prefix average precisions were 0.794 and 0.803,
below DEP-A0's depth-only prior AP 0.815. Even an optimistic *offline* argmax
over all 13 prefix scores hit only 1,159 and 1,139 safe 0.90 prefixes,
respectively, below the fixed depth-10 hit count. The online policies save
context by stopping earlier but lose many more successes; neither dominates
the fixed control. This was a 0.90 signal audit, not a five-anchor Complete or
regret evaluation. The preregistered train-only opening gate failed, so the
611-query fold was not opened for CUT-C1.

Decision: `STOP_CUT_C1_V13_MASK_SIGNAL`. The existing retriever head's score
does not supply the missing safe-window observability. A threshold sweep on
the same role would turn a diagnostic into post-hoc tuning. Further stopping
work must introduce a distinct deployment-visible signal or a redesigned
stopping contract, and still beat fixed-depth quality at materially lower
context cost before it can support the final Pareto claim.

## V17-CUT-D0 cached Target-answer stability stopping gate

CUT-D0 tested a distinct, online-observable signal on the same 1,421
train-side V8 orders, using only cached frozen-Target outputs. Its frozen rule
compares the order-insensitive normalized answer sets at depths 7 and 8. If
they match and contain at least nine distinct answers, the 0.60--0.90 anchors
stop at depth 8; otherwise they stop at depth 10. A requested 0.95 anchor
always stops at depth 10. Nine predicted answers are necessary, but not
sufficient, for 0.90 list-F1 with ten reference answer atoms. The rule never
sees reference answers or exact fidelity at inference. No new Target calls
were made in this cache replay, and 611, 581, internal300, development, and
confirmation remained sealed.

Only **1/1,421** query met the early-stop condition. The rule left 0.90
success at 1,195 and Complete at 1,045, exactly the fixed-depth-10 baseline.
Mean legal final-context fraction fell negligibly from 0.82009 to 0.81993.
Actual deployment would require an average of 2.999 Target calls rather than
one; the cached context-text-only prompt-token lower bound rose from 667.5 to
1,527.5 per query, excluding common prompt/question and output tokens. The
extra calls overwhelm the tiny final-context saving even before considering
G0's single-run Target-output instability.

Decision: `STOP_CUT_D0_ANSWER_STABILITY_GATE`. Cached Target behavior is an
available information source, but this strict, cost-accounted stability
signal does not justify opening the exposed 611 or fitting a new cutoff. No
claim is made about all possible bounded-feedback policies; any successor
must specify its Target-call budget and demonstrate a material train-only
quality--cost advantage before using another validation role.

## V17-CUT-E0 conservative lower-anchor early stopping

The train-side V8 0.90 safe-window audit located first success at depths 8,
9, 10, 11, and 12 for **1, 1,016, 234, 44, and 20** of the 1,315
reachable queries. This explains why CUT-D0's depth-8 early stop had almost
no opportunity; it is an exact-cache descriptive result, not an inference
feature. E0 therefore kept 0.90 and 0.95 at the fixed depth-10 baseline and
tested only whether the 0.60--0.80 anchors could safely stop at depth 9.

The frozen rule evaluated the clean V13 mask-value head at the V8 depth-9
mask, then stopped all three lower anchors at depth 9 when its 0.80 ordinal
logit was at least 4.0. This requires no Target call or exact label at
inference. On the 1,421 train-side queries it switched 1,261 and reduced mean
legal five-anchor final-context fraction from **0.8201 to 0.7458**.
0.90 success stayed at **1,195**, as its stop depth was unchanged; Complete
fell from **1,045 to 1,011**, with **34 breaks and zero gains**. Lower-anchor
success counts fell from 1,345/1,332/1,293 to 1,330/1,309/1,272 at
0.60/0.70/0.80. The head's apparently high 0.80 logit was not a guarantee
of safe lower-anchor stopping; ordinal-score miscalibration and nonmonotone
Target fidelity remain material.

Decision: `STOP_CUT_E0_HIGH_LOGIT_EARLY_STOP`. The preregistered train-only
quality gate failed, so fold4=611, B2B 581, internal300, development, and
confirmation were not read for E0. The cost reduction is a useful
quality--token tradeoff point but **does not dominate** fixed depth 10.
No threshold sweep on these same outcomes is authorized by this protocol.

## V17-CUT-F0 fixed V8 single-depth frontier and F1 conservative early-exit risk

F0 enumerated a restricted fixed-depth family: all five requested anchors
stop at the same V8 prefix depth 1--12. It used the exact cache only for
read-only scoring, with no new Target calls. On both the 1,421 clean-train
queries and the already design-exposed 611, depth 10 was the unique maximum
for both 0.90 success and Complete in this family. Depth 11 and full-context
depth 12 actually lost success because frozen-Target fidelity is nonmonotone.
Selected points are:

| Role / fixed depth | 0.90 success | Complete | Mean final-context fraction |
| --- | ---: | ---: | ---: |
| Train 1,421 / depth 9 | 1,017 | 87 | 0.6873 |
| Train 1,421 / depth 10 | 1,195 | 1,045 | 0.8201 |
| Train 1,421 / depth 11 | 1,138 | 975 | 0.9068 |
| Exposed 611 / depth 9 | 436 | 42 | 0.6862 |
| Exposed 611 / depth 10 | 514 | 466 | 0.8175 |
| Exposed 611 / depth 11 | 482 | 398 | 0.9058 |

The machine-readable F0 summary reports every depth, all five anchor counts,
Complete-only cost and regret, and three-objective nondominated depths. This
is a one-dimensional reference family, not the full fixed five-depth-vector
frontier. B2B 581, internal300, development and confirmation remained sealed.

F1 then held the V8 order and depth-10 fallback fixed. Its only early-exit
action was to stop the 0.60--0.80 anchors at depth 9 when the *existing*
frozen V13 0.80 mask-value score was high; 0.90/0.95 always stayed at depth
10. The train-only risk--coverage curve ranked 1%, 2%, 5%, 10% and 20% of
the 1,421 queries by that score. The protocol named the 10% point in advance
as the sole candidate for an exposed-role replay. Results:

| Nominal early-exit coverage | Switches | Complete breaks | Mean final-context fraction |
| --- | ---: | ---: | ---: |
| 0% fixed depth 10 | 0 | 0 | 0.8201 |
| 1% | 14 | 0 | 0.8191 |
| 2% | 28 | 0 | 0.8182 |
| 5% | 71 | 2 | 0.8155 |
| 10% precommitted candidate | 142 | 5 | 0.8110 |
| 20% | 284 | 6 | 0.8022 |

The 10% candidate also broke 2, 2 and 6 lower-anchor successes at 0.60,
0.70 and 0.80 respectively, despite a few lower-anchor gains; its five
Complete breaks had no Complete gains. 0.90 remained 1,195 by construction.
The 1--2% zero-break region is a descriptive result selected from this same
train role, with at most 28 switched queries and less than 0.002 mean context
saving. It is not independent evidence of safety and cannot be promoted
post hoc into a new threshold. The preregistered 10% opening gate failed,
so F1 did **not** read fold4=611 or B2B581.

Decision: `STOP_CURRENT_FEATURE_CONSERVATIVE_EARLY_EXIT`. This is stronger
than CUT-B0's negative result: even when the model only has permission to
make a sparse, quality-first early exit, the tested existing score does not
provide a useful risk--coverage region. The next stopping experiment must
measure a genuinely new deployment-visible sequential signal with full
Target-call accounting, or audit selective packet refinement; it must not
relabel the F1 curve as a successful cutoff by changing its threshold after
inspection.

## V17-CUT-F2 prefix-change signal and F3 packet-granularity audit

F2 tested one new deployment-visible sequential feature class while keeping
the V8 trajectory and depth-10 fallback unchanged: frozen V13 0.80 logits at
depths 8 and 9, their difference, and the known packet-token increment. A
four-fold query-grouped linear risk probe predicted whether stopping the
0.60--0.80 anchors at depth 9 would break any baseline anchor success. It
used only the 1,421 train-side queries and out-of-fold scores; 611, B2B581,
internal300, development, and confirmation remained sealed. Unsafe early
exits had 6.12% prevalence. The grouped probe's unsafe-event AP was 0.134,
above prevalence but insufficient to make sparse early exits reliably safe.

| Train-only top score coverage | Static V13 Complete breaks | Sequential probe Complete breaks | Sequential mean final-context fraction |
| --- | ---: | ---: | ---: |
| 1% | 0 | 0 | 0.8195 |
| 2% | 0 | 0 | 0.8188 |
| 5% | 2 | 3 | 0.8165 |
| 10% precommitted candidate | 5 | 5 | 0.8130 |
| 20% | 6 | 9 | 0.8065 |

At 10%, the sequential probe also broke 6, 7, and 7 successes at
0.60/0.70/0.80 respectively. Its preregistered zero-break gate failed, so no
checkpoint or exposed-role replay followed. Decision:
`STOP_CUT_F2_FROZEN_PREFIX_DELTA_SIGNAL`. This does not rule out every
sequential observation, but the tested free prefix-change statistic adds no
safe early-exit region over the static score.

F3 next audited all **17,052** V8 add-only packet transitions on the same
train-side queries. There were **344** transitions from a 0.90-successful
prefix to a failing successor, all at depths 10--12 (56, 133, and 155).
The three largest within-query token increments had an unadjusted rollback
rate of 3.47%, versus 1.53% for the other nine, a ratio of **2.27**. This
crossed F3's initially frozen *unadjusted* association gate and would appear
to support a bounded split-packet pilot. It does **not** support that causal
decision because the gate omitted two requirements: rollback is only possible
after the previous prefix reaches 0.90, and large packets are not uniformly
distributed by depth.

F3B corrected the comparison using only F3's saved transitions. Within the
already-0.90-successful risk set, the large-to-other rollback risk ratios
were **0.53 at depth 10, 0.93 at depth 11, and 0.97 at depth 12**. Holding
the original risk-set depth distribution fixed gives a standardized ratio
of **0.85**, with query-cluster bootstrap 95% interval **[0.67, 1.09]**.
This is a concrete confounding reversal, not evidence that large packets
protect against rollback. The unadjusted F3 gate is preserved as an audit
trail, but `NO_SIZE_TARGETED_SPLIT_PILOT` is the correct current decision.
No new Target calls were made, and the 611/581/internal/development/
confirmation roles stayed closed. Packet granularity may still matter via a
different mechanism; any future half-packet test must target a predeclared
causal question rather than this overturned size association.
## V17-TRAJ-A0 robust-chain oracle audit (train-only)

The frozen read-only protocol is `configs/v17traj_a0_robust_chain_oracle_audit.json`;
the exact 4096-mask audit is implemented in
`src/evaluation/v17traj_a0_robust_chain_oracle_audit.py`. It uses only the
lineage-clean train folds 0–3 (1,421 queries), existing V8 orders and V13
top-10 projected orders, and cached Target results. No new Target calls,
training, or sealed-set reads occurred. Per-anchor undefined levels are masked.

| Anchor | Defined | V8 success | Pool oracle success | Lattice success ceiling | V8 any width ≥3 | Pool oracle any width ≥3 | Lattice width ≥3 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.60 | 1421 | 1411 | 1419 | 1421 | 1287 | 1367 | 1421 |
| 0.70 | 1421 | 1402 | 1416 | 1421 | 1261 | 1366 | 1421 |
| 0.80 | 1421 | 1388 | 1412 | 1421 | 1210 | 1352 | 1420 |
| 0.90 | 1421 | 1315 | 1395 | 1421 | 1031 | 1178 | 1276 |
| 0.95 | 1131 | 1106 | 1129 | 1131 | 521 | 578 | 578 |

An exact subset-DAG DP computes the maximum number of consecutive successful
prefixes attainable after any mask and the minimum token cost at which a run
of length 1, 2, or 3 can start. At 0.90, a three-prefix stable run is possible
in 1,276/1,421 lattices, versus 1,031 on the actual V8 chain and 1,178 in the
top-10 pool oracle. The lattice's mean *minimum starting context fraction* for
such a run is 0.6951 among feasible queries. The pool oracle improves 0.90
reachability from 1,315 to 1,395; its mean earliest-success context fraction
among its successes is 0.7060 versus V8's 0.7212 among V8 successes. These
means have different success populations and are not a paired Pareto claim.

The ordered-cutoff Complete oracle is 1,294/1,421 for the V8 chain and
1,393/1,421 in the candidate-pool envelope. This is a per-query oracle
selection with oracle stopping, not a deployable model. The 0.95 stable-run
ceiling is already met by the pool (578/1,131); the extra full-lattice
stability headroom is mainly at 0.90. Moreover, the full-lattice results above
are **separate per-anchor ceilings**: they need not arise on one trajectory
that satisfies all five ordered cutoffs. A width-3 run is three adjacent
prefix states, not a guarantee across repeated stochastic Target calls.

**Decision:** the current packet set contains real 0.90 stability headroom,
so it would be premature to redesign packets or buy new Target verifier calls.
This audit does **not** yet authorize a learned ordering claim: it has not
shown that a single joint five-anchor, low-token stable trajectory can be
selected from deployment-visible information across held-out queries. The
next bounded step is a train-only joint-chain witness and learnability audit,
with ordered cutoffs and paired token cost; only then freeze a stability-aware
trajectory-generation protocol. Selector and current-feature cutoff training
stay closed; fold4=611, B2B581, internal300, development and confirmation
remain sealed.

## V17-TRAJ-A1 single-chain multi-anchor oracle audit (train-only)

The guidance to test a *single* order for all anchors was useful, but its
suggestion that A0's joint coverage might largely disappear overlooks the
contract: every anchor thresholds the same scalar fidelity. On one chain, a
three-prefix run above the highest attainable threshold is simultaneously a
run for every lower threshold; their first qualifying stop positions are
automatically ordered. This gives an **exact full-lattice joint stable-coverage
ceiling**, although it does not solve joint token-cost optimization or
cross-query learnability. The frozen protocol and exact minimum-inversion
witness DP are in `configs/v17traj_a1_single_chain_oracle.json` and
`src/evaluation/v17traj_a1_single_chain_oracle.py`.

| Query type | N | V8 single chain stable Complete | Top-10 pool, one chain/query | Full lattice, one chain/query |
|---|---:|---:|---:|---:|
| Four attainable anchors, highest 0.90 | 290 | 73 | 115 | 168 |
| Five attainable anchors, highest 0.95 | 1131 | 521 | 578 | 578 |
| Total | 1421 | **594** | **693** | **746** |

The full-lattice joint-stability opportunity beyond V8 is 152 queries, of
which 99 already have a witness in the frozen candidate pool and 53 require a
different order. For those 152 queries, exact subset DP found the
minimum-Kendall-distance valid order: 94 need at most two inverted packet
pairs, and 112 first diverge from V8 at depth 8 or later. These are structural
oracle witnesses, not predictions from deployment-visible features. None
breaks a V8 ordinary or stable anchor success because attaining the highest
stable anchor necessarily attains all lower anchors. Only **25/152** convert a
V8 ordinary highest-anchor failure to success; 127 merely widen an already
reachable success window.

The nearest witness reaches the first three-prefix stable run at a mean of
0.774 of full context, but that run can only be *confirmed by observing two
further prefixes*, at a mean of 0.951 of full context. For five-anchor queries
the confirmation mean is exactly 1.0. Treating its retrospectively known
start as an online stopping point would leak future information. Among queries
where V8 already has stable Complete, per-query selection of the best pool
order saves a mean 70.2 cumulative tokens in the four-anchor cohort and 15.3
in the five-anchor cohort; these are modest oracle opportunities, not learned
Pareto gains. A full-lattice multi-anchor token Pareto frontier was not
computed, so no claim is made that the nearest-inversion witness is token
optimal.

**Decision:** A1 supports the existence of some local single-chain repairs,
but it does not support launching TRAJ-B0 on width-three labels alone. Stable
window width is a retrospective surrogate; most new witnesses do not repair
ordinary quality, and 0.95 width-three confirmation costs full context. The
next experiment must test whether deployment-visible features can identify
the **25 ordinary-quality repairs or paired token savings**, rather than just
predict which late swap widens an oracle window. Keep the 611/581/internal/
development/confirmation sets sealed and do not train a full permutation
generator or a new cutoff on this evidence.

## V17-TRAJ-A2 fixed-stop local-edit oracle (train-only)

The proposed bridge from retrospective A1 windows to the actual compression
objective is scientifically useful, with one correction: freezing the stop
schedule removes **stopping** hindsight, but per-query selection of the best
edit still reads exact Target outcomes. A2 is an opportunity ceiling, not a
deployable result. The protocol froze three monotone schedules before A2 ran:
the previous C0 quality control `[10,10,10,10,10]`, balanced
`[9,9,9,10,10]`, and compression-oriented `[8,8,9,9,10]`. Each query gets
one V8 order or one of its 76 distinct alternatives at Kendall distance at
most two. The primary oracle may select only an edit that breaks no previously
successful anchor and does not increase that query's cumulative context
tokens. It then maximizes repaired anchors and minimizes tokens. Undefined
0.95 labels on four-anchor queries remain masked.

| Fixed schedule | Method | 0.90 | 0.95 / eligible | Complete | Mean normalized cumulative context | Edited queries |
|---|---|---:|---:|---:|---:|---:|
| `[10,10,10,10,10]` | V8 | 1195/1421 | 914/1131 | 1045/1421 | 0.8201 | 0 |
| | bounded local oracle | **1274/1421** | 914/1131 | **1083/1421** | **0.8018** | 366 |
| `[9,9,9,10,10]` | V8 | 1195/1421 | 914/1131 | 1004/1421 | 0.7363 | 0 |
| | bounded local oracle | **1260/1421** | 914/1131 | **1038/1421** | **0.7241** | 506 |
| `[8,8,9,9,10]` | V8 | 1017/1421 | 914/1131 | 863/1421 | 0.6628 | 0 |
| | bounded local oracle | **1058/1421** | 914/1131 | **890/1421** | **0.6573** | 569 |

All five anchor counts, paired repairs/breaks, per-anchor context fractions,
both-Complete token changes, and width-three diagnostics are in the A2
summary. The quality-depth-10 bounded oracle repairs 79 instances at 0.90
and 38 Complete examples with **zero anchor or Complete breaks**, while
reducing mean cumulative context by 71.2 tokens per query. It edits 366
queries, but only 108 obtain any quality repair; the rest are chiefly oracle
token savings. All three fixed operating points have the same qualitative
oracle direction. The V8 depth-10 counts reproduce the earlier F0 control
exactly. The optional quality-first oracle, which permits token increases on
individual queries, reaches 1302/1421 at 0.90 and 1100/1421 Complete, but
its extra freedom is not needed to establish the bounded opportunity.

Width-three does not track these gains monotonically: under the most
compressed schedule, the bounded oracle improves fixed-stop 0.90 success
from 1017 to 1058 while its retrospective 0.90 width-three count falls from
1031 to 1018. This supports using *actual fixed-stop quality and token cost*
as the primary target, not A1's window label. The 0.95 count remains unchanged
under the bounded oracle at all three schedules; early claims should focus on
preserving this anchor while improving other quality/cost coordinates.

**Decision:** A2 establishes a nontrivial local-edit opportunity under causal
fixed stopping, so do not close the trajectory branch or buy verifier calls
yet. It does **not** establish a new deployable Pareto point: the oracle has
looked up the result of 77 edits per query. The next bounded experiment is a
train-only, query-grouped V8 residual editor using *deployment-visible*
features and set-valued beneficial edits from the frozen depth-10 objective.
Other two schedules remain fixed sensitivity checks. Evaluate actual generated
orders end to end against same-query V8 for all five anchors, Complete,
paired repairs/breaks, and tokens; stop if no held-out-query Pareto gain.
Keep fold4=611, B2B581, internal300, development, and confirmation sealed.

## V17-TRAJ-A3-0 local-edit label-structure audit (train-only)

Before training an editor, A3-0 enumerated all 76 Kendall-distance-≤2 V8
edits on the same 1,421 training queries and all three frozen A2 schedules.
An edit is strictly beneficial only if it breaks no previously successful
attainable anchor, costs no more cumulative context than V8, and either
repairs an anchor or saves tokens. Outcome and edit signatures use only the
existing exact cache; no Target calls or sealed-set reads occurred.

| Schedule | Queries with ≥1 beneficial edit | Median positive edit IDs among positive queries | Mean distinct stop-mask outcomes among all 77 actions | Mean beneficial stop-mask outcomes among positive queries |
|---|---:|---:|---:|---:|
| Quality `[10,10,10,10,10]` | 366 | 12 | **4** | **2.12** |
| Balanced `[9,9,9,10,10]` | 506 | 11 | **7** | **2.48** |
| Compression `[8,8,9,9,10]` | 569 | 10 | **11** | **2.37** |

The primary quality schedule's 77 nominal actions collapse to exactly four
distinct depth-10 packet subsets for **every** query: STAY, replace V8 rank 9
with rank 10, replace rank 9 with rank 11, or replace rank 8 with rank 10
(zero-based ranks). Sixty-four different orders reproduce STAY's depth-10
subset; their extra within-prefix permutations are invisible to this primary
evaluation. Thus training a 77-way classifier would create redundant action
labels and squander data. Among the 366 queries with a beneficial primary
edit, 79 have some local edit that repairs 0.90, 38 repair Complete, and 301
have a token-only edit. Different beneficial edit IDs frequently share the
same stop-mask outcome; the 12-ID median is misleading without this quotient.

Across schedules, positive-query overlap is 353/519 between quality and
balanced, with mean edit-set Jaccard 0.821 where both are positive. Quality
versus compression is less stable: 293/642 positive-query overlap, mean
Jaccard 0.420 where both are positive. Therefore a policy trained on the
quality schedule should not be assumed to transfer unchanged to aggressive
compression; those schedules are sensitivity checks rather than selectable
post-hoc winners.

**Decision:** the guidance's set-valued principle is correct, but the proposed
77-action residual ranker is too redundant under fixed depth 10. The next
bounded train-only test should score the **four distinct stop-mask choices**
with a canonical minimum-edit order per mask, explicit STAY, and frozen V8
query/packet embeddings plus packet-token features. Labels are set-valued
strict Pareto improvements on the primary quality schedule; quality-breaking
edits must be negative for a quality-constrained primary objective. Query-
grouped out-of-fold *rollout quality and token cost*, not edit-ID accuracy,
must decide whether this representation can capture any A2 oracle headroom.
The existing V8 lineage uses frozen Qwen3-4B; the supplied guidance's
“Qwen3-8B” description does not match the current checkpoint.

## V17-TRAJ-A3-1 four-outcome boundary-editor CV (train-only)

A3-0's quotient suggested a small, controlled train-only test rather than
the supplied guidance's 77-action ranker. A3-1 kept V8 and its frozen
Qwen3-4B embeddings fixed. For quality-depth-10, it retained STAY plus the
three distinct edited packet subsets, with a canonical minimum-inversion
order per subset. A single shared linear scorer used only frozen V8 query and
packet embeddings, their local difference/interaction, packet token lengths,
and changed V8 ranks. STAY had fixed score zero. Four query-grouped folds
trained for the frozen 300-step budget, final checkpoint only, and each held-
out query received exactly one editor decision. Target outcomes were used
only for training labels and cached rollout evaluation. No Target calls or
611/581/internal/development/confirmation reads occurred.

| Fixed schedule | Method | 0.90 | 0.95 / eligible | Complete | Mean normalized cumulative context |
|---|---|---:|---:|---:|---:|
| Quality `[10,10,10,10,10]` | V8 | 1195/1421 | 914/1131 | 1045/1421 | 0.8201 |
| | OOF learned editor | **1146/1421** | **760/1131** | **878/1421** | 0.8038 |
| Balanced `[9,9,9,10,10]` | V8 | 1195/1421 | 914/1131 | 1004/1421 | 0.7363 |
| | Same OOF editor | 1146/1421 | 760/1131 | 849/1421 | 0.7314 |
| Compression `[8,8,9,9,10]` | V8 | 1017/1421 | 914/1131 | 863/1421 | 0.6628 |
| | Same OOF editor | 1022/1421 | 760/1131 | 747/1421 | 0.6612 |

The model edited 306/1,421 queries. On the primary schedule it made 18
0.90 repairs but 67 0.90 breaks, and only 6 Complete repairs against 173
Complete breaks. At 0.95 it made **zero repairs and 154 breaks**. Among those
306 selected edits, 205 broke at least one originally successful anchor;
only 29 were safe quality repairs and 63 were token-only savings. This is a
large quality loss in exchange for modest token savings, not a Pareto point.
The predeclared decision is `STOP_A3_1_NO_OOF_PARETO`. Even restricting this
editor to four-anchor queries would not solve it: in that cohort 0.90 falls
from 131 to 118 with 6 repairs and 19 breaks.

This failure does not prove *no* local editor is learnable. It does refute the
specific frozen-representation linear residual scorer with empirical pairwise
labels as a useful next deployment step. The train-only folds are not
lineage-clean for upstream V8, which saw these train queries; that caveat
would weaken a positive claim, but cannot explain away this negative one.
Do not spend 611/581/internal/development/confirmation on this checkpoint,
or tune its threshold/rank/steps against these outcomes. A bounded follow-up
may audit whether any high-confidence subset has acceptable risk; absent
that, close the current-feature local-editor branch and only then design an
explicit Target sufficiency feasibility test with full call/token accounting.

## V17-TRAJ-A3-2 OOF margin risk audit (train-only)

The A3-1 model and four query-grouped fits were rerun unchanged solely to
record each held-out query's maximum edit score; all A3-1 decisions and
aggregate metrics reproduced. A read-only audit then applied predeclared
1%, 2%, 5%, 10%, and 20% fold-relative highest-margin coverage points.
These are **batch-relative diagnostics**, not deployable online thresholds.

| Nominal coverage | Edited | 0.90 repairs / breaks | Complete repairs / breaks | 0.95 breaks | Mean normalized cumulative context |
|---|---:|---:|---:|---:|---:|
| V8 / 0% | 0 | 0 / 0 | 0 / 0 | 0 | 0.8201 |
| 1% | 12 | **0 / 2** | **0 / 5** | **5** | 0.8192 |
| 2% | 27 | 1 / 4 | 1 / 14 | 14 | 0.8184 |
| 5% | 68 | 5 / 14 | 2 / 33 | 28 | 0.8159 |
| 10% | 140 | 9 / 28 | 3 / 68 | 59 | 0.8117 |
| 20% | 264 | 17 / 55 | 6 / 142 | 127 | 0.8059 |

Even the highest-confidence 12 edits contain no quality repair and five
Complete breaks. There is no observed low-risk region to justify a separately
frozen online threshold. This is not a proof that all local editing or all
representations fail; it is a decisive **STOP for the current frozen V8
embedding + linear four-outcome scorer**. Do not sweep rank, loss, epochs,
margin, or thresholds on these same OOF outcomes, and do not read 611/581/
internal/development/confirmation. The next independent mechanism to assess
is explicit Target sufficiency feedback, with input/output tokens and call
count included in the deployment cost comparison. A new verifier protocol
must be frozen before any such calls.

## V17-ACT-C0 depth-10 omission structure audit (train-only)

The proposed omission view is meaningful, but the existing A3-1 scorer already
used frozen query/packet embeddings, packet differences, and query interactions.
Merely renaming the same features as an omission classifier would not test a
new mechanism. ACT-C0 therefore first audited the actual 10-of-12 endpoint
masks, without training or new Target calls. The population was the same 1,421
train-side queries; 611/581/internal/development/confirmation remained sealed.

| Fixed depth-10 endpoint | 0.90 | 0.95 / eligible | Complete | Mean normalized context |
|---|---:|---:|---:|---:|
| V8 | 1195/1421 | 914/1131 | 1045/1421 | 0.8201 |
| Cost-bounded local-four oracle | 1274/1421 | 914/1131 | 1083/1421 | 0.8018 |
| Cost-bounded all-66 omitted-pair oracle | 1324/1421 | 915/1131 | 1125/1421 | 0.8000 |

The oracle requires no anchor break and no higher context-token cost than V8
for each query. The four local masks cover 108 of 165 queries with an
any-anchor repair opportunity under all 66 masks; 57 have opportunities only
outside the local four. At 0.90 the local four capture 79 of 129 possible
repairs; for Complete they capture 38 of 80. The all-66 result is an endpoint
ceiling, not a deployable policy or a guarantee of a good progressive ordering.

Packet omissions interact. Among the 1,071 queries whose full 12-packet
context succeeds at 0.90, both individual omissions remain successful but
their joint omission fails in 27,206 of 70,686 packet pairs. Conversely, 304
pairs succeed jointly although both single omissions fail. Thus independent
packet-deletion labels are unsafe as a general rule. This is a thresholded
full-12 counterfactual; because fidelity can roll back, it does not directly
label necessity relative to the V8 depth-10 context. For 0.95, only 914 V8
queries succeed at depth 10 and the local oracle produces zero repairs, so
any later learned selector must protect this anchor explicitly.

The next defensible cheap-feature test, if run, is one predeclared linear
four-outcome comparison that adds an explicit symmetric omitted-pair and
query-pair interaction to A3-1 while keeping its splits, labels, optimizer,
and STAY decision unchanged. It must be judged by query-held-out rollout
quality and tokens, not pair accuracy. A negative result would close this
specific cheap-feature branch; the all-66 oracle does not justify a 66-way
classifier. The read-only artifacts and audit test are under
`v17act_c0_depth10_omission_audit`.

## V17-ACT-C0 learning-chain spot audit

Before ACT-C1, a hash-stable 64-query train-side spot check sampled 32 A3
OOF-edited and 32 STAY queries. It independently read the raw exact-cache
masks, reconstructed the fixed-depth-10 five-anchor outcomes and cumulative
tokens, and assigned benefit/harm labels without calling the A3 label builder.
All 192 non-STAY labels matched that builder. The four action masks were
distinct for every sampled query, and no two distinct masks rendered the
same packet context text. This checks the sampled cache-to-label and
mask-to-rendering paths, not all 1,421 queries or Target reproducibility.

The same 64 queries were used only for a training-set memorization diagnostic
with the original A3 frozen features and linear head. At 300 full-batch steps,
189/190 labeled directions were correct (loss 0.0745); 21 selected edits were
positive, zero negative, and one ignored/incomparable. At 3,000 steps all
190/190 directions were correct (loss 0.00210), with the same selected-label
counts. This argues against a gross gradient, sign, or STAY-plumbing failure.
It does not establish held-out query generalization; deliberately fitting
the training sample is not a model-selection result.

The A3 training loss compares each edit with the fixed-zero STAY score. When
those binary directions are correct, the argmax cannot choose a labeled
negative edit, as this memorization check confirms. It does not explicitly
rank multiple positive or ignored edits against one another, so the chosen
edit can still be incomparable with the desired quality-token trade-off.
The earlier G0A Target rerun already found threshold-label instability on a
boundary-enriched sample; any further Target reruns should target rare
beneficial-edit versus STAY *decision flips*, use the Qwen3-8B Target
(not the Qwen3-4B V8 encoder), and report call/token costs. They are not
silently replaced by this cache arithmetic check.

### Paired Target decision rerun

To test the cache-to-Target leg, 12 hash-selected positive edit/STAY pairs
from the audited 64 queries were rerun twice with the frozen Qwen3-8B,
the original atomic QAMPARI prompt and greedy generation. All 24 paired
masks were regenerated in each repeat, for 48 Target calls, 40,488 prompt
tokens and 2,972 generated tokens. This is deliberately enriched for rare
positive edit decisions and cannot estimate their population flip rate.

Eight of the 12 cached-positive edit decisions remained positive; four
became harmful in both new repeats. The two new repeats agreed on all 12
pair decisions. In all four changed pairs the STAY result improved relative
to the original exact cache while the edit did not, so a quality loss appeared
under the new paired evaluation. This shows decision-label drift across the
old exact cache and the current Target run, not evidence that the same-process
Target is randomly flipping these labels. The original exact-search code
defaults to a 512-state batch, while this pilot used a 24-mask batch; the
cache metadata does not establish the original effective batch size. vLLM batching, runtime
environment and model artifact identity need to be isolated before deciding
whether the old cache can be used as an unqualified ACT-C1 training contract.
For all 12 rerun queries, the current train-side question matched the original
exact-search example and all 12 packet texts matched the original atomic
packet files. A source-text mismatch does not explain these four label flips.

Therefore the deterministic cache-to-label spot check passes, but the
cache-to-current-Target decision check does not. Freeze ACT-C1 training
pending a small exact-contract replication on these four pairs, including
prompt/token hashes, model artifact hashes, runtime versions and batch
conditions. Do not treat the 4/12 as a general error rate or tune a new
label threshold against it.

## V17-EXEC-A0 Target provenance and execution contract

The old 4,096-mask exact cache was generated for **Qwen3-8B**, not Qwen3-4B:
its metadata names `models/Qwen3-8B`, the exact-search CLI defaults to that
model, and a located shard launch script explicitly passes it. Qwen3-4B was
the frozen V8 representation model. Thus the paired label disagreements are
not explained by accidentally applying 4B oracle labels to an 8B Target.

The QAMPARI prompt/generation wrapper and parser/metric source files are
byte-identical to the initial atomic exact-search implementation in Git. The
current 8B weight shards, tokenizer, config, runtime versions, GPU type and
prompt fingerprint for all 24 rerun masks are recorded in
`configs/v17exec_a0_target_provenance.json` and the paired rerun artifacts.
Historical exact-cache metadata has `git_commit=unversioned` and did not
record historical weight/tokenizer hashes, resolved prompt token IDs,
effective per-example batch size, or runtime versions. The current hashes
therefore identify today's Target but cannot prove byte-for-byte equivalence
to the old run.

A controlled current-runtime submission-batch probe compared batch size 1
with the earlier 24-mask batch for all four changed pairs and four stable
controls. All eight pairs reproduced the same F1 and decision label at both
batch sizes. Then one changed query, `57509__wikitables_composition__train`,
was replayed in the exact-search `product` order with the full 512-state
submission batch containing both masks. The current F1 remained
`STAY=1.0, edit=0.947368`, versus old-cache `0.947368, 0.947368`. The old
STAY prediction omitted `Dragons Forever`; the current STAY prediction added
it, while the edit prediction stayed the same. This directly locates this
one disagreement in Target generation, upstream of parsing, scoring and the
threshold. It does not establish that batching was irrelevant for every
query or recover the unrecorded historical execution environment. Costs:
16 calls/14,734 prompt tokens/1,109 generated tokens for batch-1, plus
512 calls/296,192 prompt tokens/19,360 generated tokens for the original-
geometry batch probe.

Since the old cache cannot currently be reproduced on a disputed pair even
with its original submission-batch geometry, a new train-only four-action
depth-10 cache under a fully recorded *current* 8B contract is the smallest
decisive next step. It contains 1,421 × 4 = 5,684 contexts, not 1,421 ×
4,096 lattice states. It must first re-establish the A2 V8-versus-local-oracle
quality/token headroom; no selector/editor training or sealed-set reads are
authorized by this audit.

### V17-EXEC-A1 fresh four-action depth-10 cache

The current Qwen3-8B contract was used to regenerate exactly four canonical
depth-10 contexts for each of the 1,421 train-side queries: 5,684 Target
calls, 4,304,288 prompt tokens and 302,291 generated tokens. No model was
trained, no other schedule was regenerated, and 611/581/internal/development/
confirmation remained sealed. The 12 earlier rerun pairs agree in F1 with
their corresponding entries in this fresh cache despite the new submission
batch size of 128.

| Current-contract depth-10 | 0.90 | 0.95 / eligible | Complete | Mean context fraction |
|---|---:|---:|---:|---:|
| Frozen V8 | 1223/1421 | 973/1131 | 1123/1421 | 0.82009 |
| Cost-bounded four-action oracle | 1293/1421 | 973/1131 | 1155/1421 | 0.80432 |

The local four-action headroom therefore survives under the current Target:
70 additional 0.90 successes and 32 additional Complete queries, with no
anchor breaks by construction and lower context cost. It is smaller than
historical A2's 79 and 38 repairs, and remains an outcome-aware oracle, not
a deployable learned editor. The 0.95 anchor has no repair opportunity in
these four depth-10 masks and must be protected.

The historical versus fresh paired-mask audit found changed F1 on 788/5,684
contexts. At 0.90, 81 contexts changed success→failure and 260 changed
failure→success. The old four-action oracle had 108 queries with any-anchor
repairs; the fresh one has 91. These cross-run differences show that old
cache labels are not a reliable unqualified supervision contract for the
current Target execution. They do not measure same-run aleatoric noise or
independent query generalization. The fresh four-mask cache can support a
new train-only, predeclared four-action experiment, but that experiment must
use fresh labels and preserve the same frozen Target manifest. The frozen
V8 order itself was learned on the historical training lineage, so the
positive oracle result is still design-side evidence only.

## V17-ACT-C1 fresh-target factorized omission CV (train-only)

After EXEC-A1 re-established four-action headroom under the current 8B
contract, one controlled model test was frozen before fitting. ACT-C1 used
the fresh 5,684 mask labels, the same 1,421 train-side queries and four
query-grouped OOF folds as A3, and the same 300-step linear edit-versus-STAY
optimizer. The only feature addition was a symmetric representation of the
two omitted packets, their interaction and query interaction. STAY remained
at score zero; no threshold, capacity or loss sweep occurred. No Target calls
or sealed-set reads were used for this training experiment.

| Fixed depth-10 current Target | 0.90 | 0.95 / eligible | Complete | Mean context fraction |
|---|---:|---:|---:|---:|
| Frozen V8 | 1223/1421 | 973/1131 | 1123/1421 | 0.82009 |
| Four-action oracle | 1293/1421 | 973/1131 | 1155/1421 | 0.80432 |
| ACT-C1 OOF editor | **1153/1421** | **700/1131** | **823/1421** | 0.79830 |

The learned editor chose an edit for 466 queries; 362 chosen edits have a
negative fresh-cache supervision label, 101 positive and 3 ignored. It made
20 repairs versus 90 breaks at 0.90; at 0.95, one repair versus 274 breaks;
Complete had 12 repairs versus 312 breaks. The lower context fraction is
therefore bought with a major quality loss, not a Pareto improvement. The
predeclared decision is `STOP_ACT_C1_NO_TRAIN_ONLY_PARETO`.

A read-only fold-relative score diagnostic found no observed low-risk tail:
among the top 1% highest-scoring edits in each fold (16 total), 13 were
negative, with zero 0.90 or Complete repair and eight Complete breaks. Top
2% and 5% also had more breaks than repairs. These batch-relative fractions
are diagnostics, not deployable thresholds. This result closes the specific
cheap frozen-V8-feature four-action editor branch. It does not prove that
the packet omission task is unlearnable with every possible information
source. The next distinct hypothesis is explicit frozen-Target sufficiency
feedback, with extra calls, prompt/output tokens and latency measured as part
of the compression cost before designing or training a verifier/controller.

### VERIFY-A0 static-depth10 Target-call cost floor

Before spending more Target calls on that hypothesis, a read-only calculation
compared the fresh four-action oracle's actual token saving with a measured
extra 8B answer call. The oracle saves only **13.95 context tokens per query**
over all 1,421 queries, or **63.94** among its 310 edited queries. One
observed partial-context Target call averages **757.26 prompt + 53.18
generated = 810.45 tokens**. An extra call on every query is about 58 times
the oracle's average saving; even a hypothetical perfect cheap gate that
probed only the 310 oracle-edited queries faces about 12.7 times the saving
per edited query. A yes/no verifier might generate fewer tokens, but it still
needs the context in its prompt and cannot make this static depth-10
omission selector token-efficient through an extra full Target call.

The decision is `STOP_EXTRA_TARGET_PROBE_FOR_STATIC_DEPTH10_OMISSION_TOKEN_SAVINGS`.
This cost floor does not rule out a qualitatively different closed-loop
progressive method that *reuses a Target call already required for the final
answer*, nor quality improvements at higher total cost. Such a method must
specify when the feedback is observed, whether the call is reusable, and all
Target/compressor costs before any new training or claim of Pareto gain.

### V17-CANON-P0 fresh V8 prefix-chain protocol

The static depth-10 four-action branch is closed after ACT-C1 and VERIFY-A0.
The next measurement returns to progressive V8 prefixes under the *current*
Qwen3-8B execution contract. Its frozen protocol is
`configs/v17canon_p0_fresh_v8_prefix_chain.json`; its resumable generator and
auditor are `src/evaluation/v17canon_p0_fresh_v8_prefix_chain.py` and
`src/evaluation/v17canon_p0_analyze.py`.

The population is train-side folds 0–3 (1,421 queries). For each query the
generator evaluates V8 depths 1–12. The fresh EXEC-A1 STAY result at depth 10
is reused by exact query/mask, leaving 15,631 new Target calls rather than
17,052. The resulting per-depth success, Complete and context cost are
*single fixed-depth diagnostics*. A level-wise deployment schedule has a
different cost and Complete definition and must be frozen and evaluated
separately. This stage does not train a model or read the 611, 581, internal,
development or confirmation sets.

The next progressive neighborhood is not authorized by the historical 4096
cache or the depth-10 oracle alone. It requires a fresh-chain Pareto audit,
then a predeclared local action set, causal stopping schedule and Target-call
budget. Any later Qwen3-4B semantic utility experiment must show how its
state-conditioned representation differs from the frozen Qwen3-4B features
already used by A3/ACT-C1, and must pass query-grouped rollout rather than
classifier accuracy alone.

CANON-P0 completed on all 1,421 queries: 17,052 distinct query/depth
records, including 1,421 reused EXEC-A1 depth-10 records and 15,631 new
Qwen3-8B calls (7,491,571 prompt tokens and 556,934 generated tokens).
The three historical A2 fixed schedules were evaluated on the fresh chain:

| Fixed schedule | 0.90 | 0.95 / eligible | Complete | Mean normalized cumulative context |
|---|---:|---:|---:|---:|
| `[10,10,10,10,10]` | 1223/1421 | 973/1131 | 1123/1421 | 0.8201 |
| `[9,9,9,10,10]` | 1223/1421 | 973/1131 | 1084/1421 | 0.7363 |
| `[8,8,9,9,10]` | 1050/1421 | 973/1131 | 942/1421 | 0.6628 |

The anchor transition depths are strongly uneven: at depth 6, success is
`1265/1421` for 0.60 but just `2/1421` for 0.80; at depth 7, 0.80 reaches
`998/1421`; at depth 9, 0.90 reaches `1050/1421`; at depth 10, 0.95 reaches
`973/1131`. Thus the proposed generic "depth 5–9" focus is too broad.
A bounded local counterfactual should target the observed anchor boundaries
at depths 6, 7 and 9, while preserving the depth-10 packet set so that the
0.95 endpoint cannot be accidentally perturbed. The fresh chain does not
itself establish that those swaps can be learned or improve a deployed
quality–token Pareto point.

### V17-PROG-A0 fresh adjacent-swap opportunity audit

We froze one V8-adjacent swap at reveal positions 6/7, 7/8 or 9/10, at most
one swap per query. Each changes only one intermediate prefix; the depth-10
packet set is unchanged. Fresh Qwen3-8B outcomes were generated for 4,263
counterfactual contexts (2,345,386 prompt and 180,460 output tokens). The
primary schedule `[6,7,7,9,10]` was selected from the CANON-P0 train-side
anchor transitions before these calls; `[7,8,8,9,10]` was a frozen
sensitivity schedule.

| Fixed schedule | Method | 0.90 | 0.95 / eligible | Complete | Mean cumulative context tokens |
|---|---|---:|---:|---:|---:|
| `[6,7,7,9,10]` | V8 | 1050/1421 | 973/1131 | 775/1421 | 2183.62 |
| | no-break, no-extra-token oracle | **1067/1421** | 973/1131 | **797/1421** | **2180.59** |
| `[7,8,8,9,10]` | V8 | 1050/1421 | 973/1131 | 909/1421 | 2410.62 |
| | no-break, no-extra-token oracle | **1067/1421** | 973/1131 | **920/1421** | **2408.41** |

The primary oracle selects 456 swaps but only 53 queries gain any anchor;
the net token saving is only 3.03 tokens/query. All 17 new 0.90 successes
come from swapping positions 9 and 10. The frozen minimum opportunity gate
(20 safely repaired queries) passes, but this is a small cost-bounded ceiling
for an expensive new semantic model. It does **not** establish that a model
can identify the 53 queries without breaking others.

A secondary, explicitly *exploratory* quality-first oracle removes the
per-query no-extra-token restriction while still forbidding all anchor
breaks. On the primary schedule it reaches 0.90 `1166/1421` and Complete
`937/1421`, with mean cumulative context **+2.61 tokens/query**; on the
sensitivity schedule it reaches `1184/1421` and `1003/1421`, with
**+0.99 tokens/query**. This is a substantially larger quality–token
opportunity, but it is outcome-aware and was calculated after the primary
protocol. It may motivate one bounded *train-only* state-conditioned
learnability test; it is not a deployable result or a revised primary gate.

### V17-PROG-A1 frozen V8 state-conditioned OOF test

One bounded learning test used the fresh A0 outcomes and V8's actual frozen
history-conditioned action features (alternate packet minus V8 packet). It
trained a single linear set-valued utility head with four fixed query folds;
STAY remained an explicit zero-logit action. It did not update Qwen3-4B,
LoRA, V8 heads, Target, cutoff or decoding. All evaluation below is
train-lineage query-grouped out-of-fold for this *new head* only; V8's older
representation training did see these train-side queries.

| Primary `[6,7,7,9,10]` | 0.90 | 0.95 / eligible | Complete | Mean cumulative context tokens |
|---|---:|---:|---:|---:|
| V8 | 1050/1421 | 973/1131 | 775/1421 | 2183.62 |
| A1 OOF | **1055/1421** | 973/1131 | **781/1421** | **2193.17** |

A1 selected 637 swaps. At 0.90 it made 26 repairs and 21 breaks; for
Complete, 24 repairs and 18 breaks. The sensitivity schedule similarly moves
0.90 `1050→1055` and Complete `909→911`, while adding 6.57 context tokens per
query. The predeclared strict-dominance gate fails:
`STOP_PROG_A1_NO_OOF_PARETO`. This is a real but very small net quality gain
paid for with tokens and substantial paired breaks. It captures little of
the exploratory A0 oracle's 116 repairs / 162 Complete gains.

A post-result read-only enumeration found that no monotone V8 static schedule
with depths 1–12 simultaneously matches all six A1 primary quality counts
at or below its 2193.17 mean cumulative tokens. This means A1's OOF point
may be a **new trade-off point on design-side data**, even though it does not
strictly dominate its frozen primary baseline. This secondary observation
does not override the predeclared STOP decision or authorize opening the
611/581/internal/development sets. In particular, 21 newly broken 0.90
queries make a small aggregate gain too fragile to treat as a solved
compression method.

The decision decomposition clarifies the learning gap: 354/1,421 queries
have at least one no-break anchor-repairing local swap, yet A1 chooses such
an action on only **69** of them and misses **285**. Of its 637 selected
swaps, 511 change no anchor outcome, 69 safely repair at least one anchor,
and 57 break at least one. Thus the OOF failure is mainly poor targeting of
the rare beneficial state/action pairs, not an optimizer that never switches.

### V17-STOP-V0 fresh hindsight stopping value bound

Using CANON-P0 alone, a dynamic program chose nondecreasing stop depths along
each frozen V8 chain while preserving *every* fixed-schedule attainable-anchor
success/failure. Baseline-failed anchors stayed at their fixed depth, avoiding
artificial savings from abandoning already-failed levels. This oracle reads
future Target outcomes, so it is a context-token **upper bound**, not an
online policy. No new Target calls or sealed sets were used.

| Fixed schedule | Mean baseline cumulative tokens | Mean hindsight minimum | Upper-bound saving / query |
|---|---:|---:|---:|
| `[10,10,10,10,10]` | 3191.06 | 2222.86 | **968.19** |
| `[9,9,9,10,10]` | 2864.61 | 2197.66 | **666.95** |
| `[8,8,9,9,10]` | 2577.84 | 2153.00 | **424.84** |
| `[6,7,7,9,10]` | 2183.62 | 2084.92 | **98.69** |

These are *cumulative* context tokens over the attainable fidelity requests,
not per-request savings. The quality-heavy schedule leaves substantial
stopping headroom, chiefly at lower anchors, while the already-early schedule
leaves much less. Extra Target feedback calls, prompt/output tokens, latency
and compressor inference are excluded. A perfect hindsight stopper cannot
be called a deployable "perfect online stopper" without specifying an
observable success signal and charging its acquisition cost.

Before a wider PROG-B0 run, a read-only action-mask count found that an
arbitrary remaining-packet move changes *multiple* downstream fixed-stop
contexts. With both `[6,7,7,9,10]` and `[7,8,8,9,10]` evaluated, extending
the existing P0/A0 caches needs **5,684** new query/mask Target evaluations
for move-at-depth-9 only, **17,052** for depths 7+9, or **24,157** for
depths 6+7+9. These are deduplicated state counts, not new actions. The
cost-controlled next candidate is depth-9 only, because it directly tests
0.90 and its movement can also alter the depth-10 0.95 endpoint. The
all-depth design should not be launched as if each move cost one Target call.

### V17-STOP-C0 target-feedback gate reassessment

The proposed Target-feedback stopping protocol was mapped onto the existing
fresh canonical train1,421 V8 prefix cache before any new calls or fitting.
Against its required `[6,7,7,9,10]` baseline, the free clairvoyant bound saves
only **98.69 / 2,183.62 = 4.52%** cumulative context, below the preregistered
5% gate. This is an optimistic bound with no observation cost, so adding
STOP/CONTINUE probes cannot repair the failed final-context ceiling.

The opportunity gate also fails. Earlier success occurs for 1,134/1,421
queries at 0.60 and 1,051/1,421 at 0.70, but only 3/1,421 at 0.80,
1/1,421 at 0.90, and 1/1,131 attainable queries at 0.95. Thus only **2/5**
anchors exceed 10%, versus the required 3/5. The higher anchors coincide with
the sharp natural V8 quality transitions and provide essentially no earlier
stopping opportunity under this schedule.

Decision: `STOP_CURRENT_CONTRACT_ADAPTIVE_STOPPING_AT_C0`. The positive
costed oracle relative to `[10]*5` remains a distinct quality-heavy operating
point; it does not make the aggressive schedule pass. Do not repeat output
dynamics or train another output-only stopper. If the contract is explicitly
relaxed, the smallest scientifically distinct next interface is read-only
native Target confidence (token log-probability/margin), with labels and
traces collected in the same fresh calls. The reproducible artifacts are
[summary](v17stop_c0_feedback_gate_reassessment/summary.json) and
[report](v17stop_c0_feedback_gate_reassessment/REPORT.md).

### V17-STOP-V1A/B costed probe and simple observable diagnostic

The fresh P0 chain permits a sharper stopping cost bound without new Target
calls. A predeclared one-probe policy at depths `[6,7,8,9,10]` was given an
**omniscient** success decision; when a probe failed it called depth 10 as
fallback. Counting prompt **and** generated tokens for both calls, the
independent-request simulation saved 790.18 final-context tokens and 410.28
Target compute tokens per query relative to `[10]*5`, while improving the
sum of attainable-anchor successes from 6,191 to 6,354 and Complete from
1,123 to 1,143. This is a positive *costed oracle ceiling*, not a feasible
online policy. A monotone sequential-request accounting also remained
positive (594.30 Target compute tokens saved/query), but assumes calls at
each requested anchor rather than reusing identical responses. These two
accounting conventions must not be combined into one deployment claim.

For an actual observable, a four-fold train-side OOF diagnostic used only
the distinct parsed answer count at each probe depth. Per-anchor integer
thresholds 1–20 were selected on the other three folds only if they yielded
at least 20 early exits and **zero** baseline-success breaks. No threshold
qualified in any fold. The resulting no-probe fallback exactly reproduced
the depth-10 baseline: zero early exits, zero extra Target calls, no quality
or token gain. This rejects the answer-count signal under the frozen strict
risk condition; it does not reject richer low-cost stopping information.
The lineage 611, 581, internal, development and confirmation sets remained
sealed. The next bounded root-cause test is the already costed depth-9
remaining-packet move oracle, rather than another cutoff loss or threshold
sweep.

### V17-PROG-B0 depth-9 bounded remaining-packet move oracle

We tested **only** move-to-next at depth 9, with fixed causal stop schedules,
the same Qwen3-8B Target execution contract, and no training. P0 and A0
states were reused; exactly 5,684 previously unseen query/mask states were
generated (768 in the initial run plus 4 disjoint GPU shards of 1,229).
The summary and per-query action outcomes are in
`v17prog_b0_depth9_move_oracle/`.

| Primary `[6,7,7,9,10]` | 0.90 success | 0.95 success | Complete | Mean cumulative context tokens |
|---|---:|---:|---:|---:|
| Frozen V8 STAY | 1050 | 973 | 775 | 2183.62 |
| Oracle STAY / adjacent depth-9 swap | 1189 | 973 | 840 | 2184.63 |
| Oracle any remaining-packet move, no per-query extra context | 1067 | 973 | 784 | 2171.13 |
| Oracle any remaining-packet move, quality first, no anchor break | **1190** | **974** | **841** | **2176.65** |

The sensitivity `[7,8,8,9,10]` schedule gives the same 0.90 counts;
Complete is 909 for STAY, 993 for adjacent, 920 for strict no-extra-context,
and 994 for unrestricted quality-first. The quality-first B0 oracle chose
STAY for 1,000 queries, adjacent rank-10 for 249, rank-11 for 110, and
rank-12 for 62. The strict no-extra-context oracle chose STAY for 1,075,
rank-10 for 127, rank-11 for 138, and rank-12 for 81.

This resolves an important ambiguity in A0: its preregistered +17 0.90
result applied a **per-query no-extra-context constraint**. Removing that
constraint while preserving all successful anchors lets even the *existing*
adjacent depth-9 move repair 139 additional 0.90 queries, for just +1.01
mean cumulative context tokens. Allowing later packets adds only **one**
further 0.90 repair and one 0.95/Complete repair, while reducing mean
context by about eight tokens relative to the adjacent oracle. Hence the
one-step action restriction is not the primary source of missed quality
headroom at this depth. The unresolved problem is identifying beneficial
query-specific actions without oracle Target truth or introducing breaks;
A1 already failed that deployment-side test. B0 is a train-side
outcome-aware upper bound and does **not** meet final five-anchor gates.

Next-method decision: close further depth-9 action-space expansion and
another A1-like head. Preserve the positive costed stopping ceiling from
STOP-V1A, but do not infer that answer-count can realize it: STOP-V1B found
no zero-break OOF trigger. The next useful experiment should isolate a
deployment-visible, low-cost sufficiency signal and compare its *full*
quality–context–Target-compute Pareto against the fixed-depth frontier.
If it cannot clear a preregistered low-break gate on query-held-out train
folds, revisit the Target feedback contract or packet/state definition;
do not train another cutoff merely because the hindsight bound is large.

### V17-STOP-B0 continuation-value identifiability audit

This train-side diagnostic used the fresh CANON-P0 Qwen3-8B prefix chain and
four query-grouped OOF folds over 1,421 queries. It trained no deployment
checkpoint and made no new Target calls. The four outcome labels for each
early probe were `(success at probe, success at depth 10)`: success/success,
fail/success, success/fail (rollback), fail/fail. At 0.90 with a depth-9
probe the counts are **1,000 / 223 / 50 / 148**, respectively. The 50
rollback cases are a necessary class missing from a three-way “must continue /
safe stop / cannot rescue” taxonomy.

Three fixed linear-probe input arms were compared. The weak arm uses requested
level, depth, answer count and token fraction. The second adds signed-hash
word unigrams/bigrams of the *full parsed answer*, plus frozen V8 query and
selected-packet embeddings and an exact answer-in-evidence overlap. The third
also adds the frozen remaining-packet mean and overlap. This is a cheap
**lexical-plus-frozen-embedding** test, not a complete semantic observer or
proof about Bayes-optimal identifiability. The V8 representation was trained
on these design queries; only the critic was query-held-out. Upstream lineage
exposure limits any independent generalization claim.

| Arm | OOF four-class NLL ↓ | OOF accuracy, diagnostic only |
|---|---:|---:|
| Depth + count | **0.642** | **0.808** |
| Full answer + query/selected context | 0.829 | 0.765 |
| Above + remaining packets | 0.825 | 0.767 |

For each fold and anchor, train-fold predicted `fail at probe / success at
depth10` risk set seven frozen early-exit fractions. The held-out replay
charged every enabled probe's prompt and generation, plus a depth-10 call on
CONTINUE. Relative to the no-probe `[10]*5` baseline (0.90=1223, Complete=1123),
at the 40% train-quantile fraction the weak arm reached 1198/1049 and **spent
990 more** Target tokens/query; the full-answer arm reached 1177/1051 and
spent **1,278 more**. At 80%, the weak arm saved 274 Target tokens/query but
fell to 1139/979; full-answer saved only 21 while falling to 1102/947;
adding remaining packets saved 36 while falling to 1098/941. No tested arm
provided a quality-preserving Target-compute improvement. Context-only saving
would conceal the expensive fallback calls.

After seeing the primary OOF result, a **descriptive, non-gating** hash-grouped
training-size curve was run with the same fixed model. Full-answer OOF NLL at
25/50/75/100% of available training queries was 0.901/0.888/0.851/0.829;
with remaining packets it was 0.912/0.886/0.846/0.825. Its downward slope
means additional *independent* queries could help this overfit high-dimensional
probe; it cannot establish how many would be required, or that the arm would
eventually surpass the weak 0.642 NLL baseline or the end-to-end Pareto gate.
Repeated states from the same query do not provide independent query diversity.

Decision: `STOP_STOPB0_CHEAP_LINEAR_OBSERVER_NO_OOF_PARETO`. Do not open the
sealed 611/581/internal/development/confirmation outcomes or scale this
particular head by default. The remaining hypothesis is whether a *different,
costed semantic observer* can represent answer–evidence sufficiency, rather
than merely memorize sparse answer words. Any such experiment needs a
query-grouped learning curve and a full Target/critic cost ledger before a
new-data collection decision. The B0 negative result does not prove that the
deployment-visible full context lacks information.

### V17-STOP-C0 fresh prefix answer-retention audit

This read-only train1421 audit tested fixed, gold-free ways to combine the
already cached Qwen3-8B depth-9 and depth-10 answer lists. No new Target
calls were made offline, but every two-output rule requires **both calls at
deployment**. The current prediction cache stores *already parsed* answers
joined by ` # `. A first implementation that passed this serialization back
through the raw-generation parser corrupted names such as `No. 17 Squadron`;
the corrected audit splits the stored delimiter directly and asserts exact
reproduction of every cached depth-9/10 F1 before scoring any operator.

| Fixed output rule | 0.90 | 0.95 | Complete | 0.90 repairs / breaks vs depth10 | Target prompt+generation tokens per 0.90 request |
|---|---:|---:|---:|---:|---:|
| Depth10 only (one call) | 1223 | 973 | 1123 | 0 / 0 | 840.12 |
| Depth9 only (one call) | 1050 | 1 | 95 | 50 / 223 | 724.33 |
| Union, latest answers first | 1243 | 1006 | 1177 | 45 / 25 | 1564.45 |
| Intersection | 939 | 1 | 43 | 7 / 291 | 1564.45 |
| Depth9 plus new-packet-supported depth10 answers | 1209 | 934 | 1091 | 52 / 66 | 1564.45 |
| Depth10 plus old-context-supported depth9-only answers | **1248** | **1009** | **1183** | **45 / 20** | **1564.45** |

The last rule is deterministic and uses only the visible depth-9 context to
filter depth-9-only answer strings. It rescues 35/50 cases where depth9
succeeds at 0.90 but depth10 fails, plus 10 cases where *both* original
outputs fail. It nevertheless breaks 18 baseline successes that were
fail-at-9/success-at-10 and two successes at both depths. Paired query
bootstrap 95% intervals for its net change are [+10,+41] at 0.90 and
[+40,+80] Complete, conditional on this train-side population; selecting
the best among multiple predeclared operators on the same train data makes
these intervals **descriptive, not independent validation**.

The main result is a quality/computation tradeoff, not a free rollback fix:
the best rule gains 25 net 0.90 successes and 60 Complete, while an always-on
two-call deployment costs **724.33 additional Target tokens per 0.90 request**
before any critic overhead. Its final depth-10 context is unchanged, but
total context processed by Target rises. Also, answer-set postprocessing
changes the final-output contract; it must be represented as a system-level
answer mechanism, not as if Qwen3-8B itself produced the merged list from a
single compressed context. An omniscient STOP at depth9 followed by this
retention rule otherwise reaches only 1265/1421 at 0.90 with 954.42 Target
tokens/request, below the simpler omniscient STOP-V1A success ceiling of
1273/1421 at approximately the same cost. Thus retention helps a fallible
always-continue policy, but does not itself solve continuation decisions.

Decision: `GO_STOP_C1_COST_PREFLIGHT_ONLY`. Do not train a semantic critic
yet. First choose and benchmark a genuinely small pretrained interaction
encoder against the actual probe/fallback service path, including encoder
latency and inference cost. The repository currently contains Qwen3 models
of 1.7B parameters and larger, but no locally cached 100M--300M pretrained
cross-encoder. Training a large model on the 1,421 independent queries or
calling 8B again merely to verify answers is not justified by C0. The
611/581/internal/development/confirmation outcomes remain sealed.

### V17-STOP-C1.0 deployed STOP/retained-CONTINUE contract

We recomputed the proposed two-action contract on the fresh train1421 cache:
`STOP` returns the depth-9 answer after one Target call; `CONTINUE` makes a
depth-10 call and applies the single fixed C0 retention rule. Labels now
compare depth9 with **retained depth10**, rather than bare depth10. This is a
read-only hindsight ceiling, not a learned decision rule. The
[summary](v17stop_c1_0_deployment_contract/summary.json) and
[per-request table](v17stop_c1_0_deployment_contract/per_request.jsonl) are
reproducible with `python -m src.evaluation.v17stop_c1_0_deployment_contract`.

| Attainable target | STOP/CONTINUE SS, SF, FS, FF | Single depth10 success | Hindsight max success | Depth10 Target tokens/request | Hindsight minimum tokens at max success |
|---|---:|---:|---:|---:|---:|
| 0.60 | 1328 / 3 / 47 / 43 | 1347 | 1378 | 840.12 | 753.35 |
| 0.70 | 1304 / 8 / 50 / 59 | 1337 | 1362 | 840.12 | 755.62 |
| 0.80 | 1259 / 15 / 60 / 87 | 1311 | 1334 | 840.12 | 764.28 |
| 0.90 | 1033 / 17 / 215 / 156 | 1223 | 1265 | 840.12 | 856.95 |
| 0.95 (1131 attainable) | 1 / 0 / 1008 / 122 | 973 | 1009 | 828.27 | 1450.91 |

For 0.90, an omniscient selector can match 1223 single-depth10 successes
at a minimum 819.82 Target tokens/request, or reach 1265 successes at 856.95.
These are **optimistic lower-cost bounds**: they use post-hoc correctness to
avoid unproductive calls and omit critic cost. At 0.95, even the cheapest
hindsight policy matching 973 single-depth10 successes costs 1405.92 versus
828.27 tokens/request. Thus the earlier 410-token/query STOP-V1A bound over
five anchors does not finance this specific depth9-to-10 critic across all
anchors. Complete under independently chosen attainable-anchor actions has
an oracle ceiling of 1189 versus 1123 at depth10 and 1183 always-continue;
it is not a single shared stopping decision.

Decision: `GO_STOP_C1_COST_PREFLIGHT_090_ONLY`. Before training, benchmark a
genuinely small pretrained semantic encoder on deployment input `(query,
target, depth9 answer, new packet)`, including serialized length, batch-1
latency, GPU time, memory, and comparison with direct depth10. Keep critic
cost and Target tokens on separate axes. The 0.95 case requires a different
earlier-probe schedule or should go directly to depth10. The merged output
is a two-call system answer, not a single-call compressed-context Target
result. No sealed outcome set or new Target call was used here.

### V17-STOP-C1.1 small semantic encoder cost preflight

On the same 1421 train-side depth9 states, we serialized the exact proposed
critic input `(query, 0.90, parsed A9, actual depth10 packet)` and measured
it with a frozen, untrained English DistilBERT base encoder (66.36M
parameters). Full sequence length was median 164, P90 222, P99 304, maximum
468 tokens; **none exceeded the 512-token limit**, so the added packet was
not silently truncated. On an RTX A6000 at fp16 and batch size 1, 100
deterministically sampled inputs gave median forward 8.41 ms, P90 15.47 ms;
tokenization plus transfer median 1.89 ms, P90 2.56 ms. Peak PyTorch
allocated memory was 145 MiB. The model used no classifier head and was not
trained. [Summary](v17stop_c1_1_cost_preflight/summary.json) and
[decision](v17stop_c1_1_cost_preflight/decision.json) record the preflight.

This confirms a local semantic encoder can process the proposed input at
modest latency and memory, but does **not** prove predictive power or final
quality–compute Pareto improvement. GPU milliseconds and Target tokens remain
separate axes, and this audit did not benchmark a matched Qwen3-8B batch-1
fallback wall time. Decision:
`GO_STOP_C1_SEMANTIC_CRITIC_PROTOCOL_DESIGN_090_ONLY`. The next experiment
must freeze the query-grouped training split, policy loss/threshold rule and
full STOP/retained-CONTINUE replay before fitting anything. The 0.95 case is
excluded under this depth9 probe contract. No Target calls or sealed outcome
sets were used.

### V17-STOP-C1.2 frozen semantic continuation critic

We trained exactly one 66.36M-parameter pretrained DistilBERT sequence
classifier on the frozen 0.90 `SS/SF/FS/FF` contract. Four query-hash folds
gave one OOF prediction per train1421 query. Every fold used two epochs,
batch size 16, learning rate 2e-5, no class weighting or OOF-tuned
hyperparameters. The true class population was SS=1033, SF=17, FS=215,
FF=156. Training loss fell on all four folds; OOF multiclass NLL was 0.734.
The [summary](v17stop_c1_2_semantic_critic/summary.json) and
[OOF predictions](v17stop_c1_2_semantic_critic/oof.jsonl) provide the exact
replay. No deployable checkpoint was selected.

| Policy | Continue count | 0.90 success | Target tokens/request | Final context tokens/request |
|---|---:|---:|---:|---:|
| Direct depth10, one Target call | — | 1223 | 840.12 | 667.50 |
| Critic threshold ≤0.05 (always continue) | 1421 | 1248 | 1564.45 | 667.50 |
| Critic threshold 0.10 | 832 | 1173 | 1233.47 | 625.61 |
| Critic threshold 0.20 | 57 | 1063 | 758.39 | 563.22 |
| Critic threshold ≥0.50 (always stop) | 0 | 1050 | 724.33 | 558.68 |

At threshold 0.10, direct depth10 is better on success and Target compute.
At 0.20, the critic reduces both Target compute and final context, but loses
160 successes relative to direct depth10; it finds only 14 of the 215
`FS` opportunities among 57 continuations. Lower thresholds recover C0's
two-call answer-retention gain without making any learned decisions. These
are train-side OOF results, not independent confirmation. The observed
failure applies to this one fixed input, model and objective; it does not
prove that continuation value is unlearnable. Decision:
`STOP_CURRENT_DEPTH9_SEMANTIC_CRITIC_BRANCH`. Do not sweep model size,
epochs or thresholds on the exposed OOF outcomes. The fixed-depth and
earlier five-anchor schedules remain the defensible deployable baselines
while the next method question is reconsidered. No new Target calls or
sealed outcome sets were used.

### V17-STOP-C2A native trace contract preflight

The proposed Target-native feature cannot be extracted from the existing
CANON-P0 cache: it saved parsed answers and token counts, but no generation
logprobs. We therefore ran exactly 32 train-side depth9 queries (eight per
frozen 0.90 `SS/SF/FS/FF` class) twice in one Qwen3-8B/vLLM engine:
ordinary greedy generation versus the same request with top-two logprobs.
This cost 64 additional Target calls, 22,139 prompt tokens **per pass**, and
1,593 generated tokens in each pass. Chosen-token logprobs and the final EOS
token were available in all 32 scored outputs. Batch elapsed time was 8.60 s
plain and 8.46 s with logprobs; one small batch does not establish an ongoing
latency advantage.

Only 27/32 pairs matched raw token IDs, 28/32 matched parsed answers and
30/32 matched the 0.90 success label. Plain reruns matched the old cache's
parsed answer in 29/32 and 0.90 label in 31/32. The logprob rerun flipped
the 0.90 label in **two of the eight historical SF cases**, from 0.90 to
0.842 and 0.700. The [summary](v17stop_c2a_native_trace_preflight/summary.json),
[paired rows](v17stop_c2a_native_trace_preflight/paired.jsonl), and
[decision](v17stop_c2a_native_trace_preflight/decision.json) preserve this
result. A sequential pair cannot attribute the discrepancy specifically to
logprobs rather than run-order/numerical effects; it does show that joining
new traces to old C1 labels would corrupt some scarce critical examples.
Decision: `STOP_FULL_NATIVE_TRACE_COLLECTION_UNDER_CURRENT_CACHE_CONTRACT`.
We did not regenerate all 1421 traces or fit a trace critic.

### V17-STOP-D0 forced multi-probe value of information

With no new Target calls, we replayed fresh cached prefixes for three
predeclared 0.90 schedules. The oracle stops at the first successful probe;
if it reaches depth10 it uses the same fixed C0 retention rule. Every call's
prompt and generated tokens are charged.

| Oracle schedule | 0.90 success | Target tokens/request | Final context tokens/request | Calls/request |
|---|---:|---:|---:|---:|
| Direct depth10 | 1223 | 840.12 | 667.50 | 1.00 |
| 9→10 | 1265 | 954.42 | 588.32 | 1.26 |
| 7→9→10 | 1265 | 1501.93 | 588.32 | 2.26 |
| 6→9→10 | 1265 | 1427.25 | 588.32 | 2.26 |

On these 1421 queries, depth6 and depth7 have **zero** 0.90-success
outputs. Thus the extra forced early probe cannot improve this oracle's
quality or final context and costs an additional 472.83–547.50 Target
tokens/request over 9→10. The [summary](v17stop_d0_multi_probe_oracle/summary.json)
and [decision](v17stop_d0_multi_probe_oracle/decision.json) close these
fixed 0.90 multi-probe schedules. This does not rule out every other depth
or an adaptive information-gathering action, but it sharply limits the
economics of the proposed 6/7→9→10 path. No sealed set was read.

### V17-SAMECALL-A0 binary self-report and matched prompt control

We froze a minimally modified Qwen3-8B prompt that requests the ordinary
QAMPARI `<answer>` list followed by `<sufficient_090>YES/NO</sufficient_090>`
in **the same call**. The pilot deliberately included all 17 historical SF
queries and 37 each from SS/FS/FF (128 distinct train-side queries), with
two runs at each of depth9 and depth10. This enrichment tests rare errors;
it is not a representative population sample. The 512 new-prompt calls had
100% valid answer/status tags. Across the two repeats, depth9 status agreed
128/128 and the 0.90 answer label agreed 128/128; the parsed answer agreed
127/128. Depth10 answers and 0.90 labels agreed 128/128.

Despite that stability, the self-report said YES for **121/128** depth9
queries, including **49/53 actual 0.90 failures**. A policy stopping on YES
achieved only 76/128 versus 98/128 for direct depth10 under the *same new
prompt*. It is a highly overconfident insufficiency signal, not a working
stopper. The [summary](v17samecall_a0_sufficiency_pilot/summary.json),
[per-call outputs](v17samecall_a0_sufficiency_pilot/per_call.jsonl), and
[replay](v17samecall_a0_sufficiency_pilot/replay.jsonl) record both repeats.

An unexpected separate result concerns the **answer prompt itself**. We
reran the canonical old prompt on the same 128 queries at both depths (256
additional Target calls), using the same current Qwen3-8B runner. At depth9,
old-prompt rerun had 54/128 0.90 successes and the new prompt had 75/128:
23 paired repairs and two breaks. At depth10, it was 81→98, with 22 repairs
and five breaks. The historical cache was 54 and 77 respectively, so rerun
drift explains part of the old-vs-cache depth10 difference, **not** all the
matched prompt difference. The new prompt cost 888.67 versus 742.01 Target
tokens/call at depth9, and 1005.09 versus 864.23 at depth10. Of this,
roughly 19 extra tokens came from generated output at depth9; the remainder
mostly comes from the longer prompt. See the
[matched-control summary](v17samecall_a0_prompt_control/summary.json).

Decision: `STOP_BINARY_SELF_REPORT_GO_PROMPT_QUALITY_VALIDATION`. The
self-report cannot guide early stopping. The apparent answer-quality gain is
a new hypothesis requiring a **fresh unstratified train-side sample**, all
attainable anchors and full Target-cost accounting before any claim about the
final quality–token Pareto. The 611/581/internal/development/confirmation
sets remain sealed.

### V17-SAMECALL-B0/B4 prompt gain correction and replication

The A0 self-report itself failed, but its answer prompt appeared to improve
0.90 success in an intentionally enriched 128-query set. We tested that
answer-generation hypothesis on 128 previously unused train-side queries,
with both old and new prompts at depths 9 and 10 and all five attainable
anchors. **B0's initial comparison was invalid as a gate**: it interleaved
old/new templates within each batch. At depth10 it reported 104→119 0.90
successes, but 10/15 net gains came from old-prompt responses without a
valid `<answer>` tag and with zero parsed F1. A targeted rerun of all 23
old-untagged query/depth cases in an old-only batch reached 0.90 in 21/23;
many raw outputs were parseable `#`-separated lists. Thus B0's old arm had
a substantial batch/run-condition failure, not 15 confirmed semantic gains.
The B0 raw result and its superseding [decision](v17samecall_b0_unstratified_prompt_gate/decision.json)
are both retained for audit.

A format-only ablation on B0's already exposed IDs achieved 111/128 depth10
0.90 successes versus 104 in B0's old mixed arm and 119 with the longer
self-report prompt. That ablation was diagnostic only: B0's old arm was
unstable and cannot be used to estimate format-only deployment benefit.
The [format-only summary](v17samecall_b1_format_only_ablation/summary.json)
and [untagged raw-output audit](v17samecall_b2_untagged_output_audit/summary.json)
preserve the mechanism checks.

We then reran old and new prompts in **separate homogeneous batches** on
the same B0 IDs. At depth10, old→new was 114→119/128 at 0.90, 95→95/102
at 0.95 and 110→114 Complete, costing 845.52→989.95 Target tokens/query.
Because those IDs were already design-exposed, B3 was descriptive only.
The [B3 summary](v17samecall_b3_homogeneous_batch_control/summary.json)
records this corrected comparison.

Finally B4 used a fresh 128 train-side queries excluding **all** A0 and B0
IDs. It kept the prompt and homogeneous batching fixed. At depth10, 0.90
was 115→119/128 (**+4**, six repairs/two breaks), 0.95 was 96→94/105,
Complete was 107→110, and Target cost was 818.77→961.45 tokens/query.
Depth9 0.90 was 103→104 with eight breaks and nine repairs. The frozen B4
research gate required at least +5/128 at depth10, no Complete regression,
no anchor worse than -3, and <=25% extra Target tokens. It **failed the
primary +5 threshold**, so the prompt-only branch stops; we do not adjust
the gate to +4 after observing B4. The
[B4 summary](v17samecall_b4_fresh_prompt_replication/summary.json) and
[decision](v17samecall_b4_fresh_prompt_replication/decision.json) are the
final train-side evidence. No sealed outcome set was read.

### V17-STATE-A0 early state-dependent packet move audit

We tested one move-to-next action at V8 depth 5 or 6, changing the depth-6
or depth-7 prefix. Only V8 ranks up to 10 could move, so every action kept
the same depth-10 packet set. The Target prompt and fixed five-anchor stop
schedule `[6,7,7,9,10]` were unchanged. P0/A0 states were reused by exact
query/mask; the initial 64 queries required 320 fresh Target states and the
next 192 queries required 960. Both cohorts are train-side. Action choice
below is outcome-aware, never a deployable policy.

| Cohort/policy | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Complete | Mean cumulative context tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| First 64, V8 STAY | 58 | 58 | 45 | 48 | 43 | 41 | 2149.81 |
| First 64, quality-first oracle | 61 | 62 | 57 | 52 | 43 | 47 | 2153.41 |
| First 64, no-extra-context oracle | 59 | 58 | 45 | 48 | 43 | 41 | 2148.83 |
| New 192, V8 STAY | 168 | 165 | 131 | 142 | 132 | 106 | 2141.96 |
| New 192, quality-first oracle | 186 | 185 | 166 | 157 | 132 | 137 | 2162.69 |
| New 192, no-extra-context oracle | 170 | 170 | 134 | 143 | 132 | 109 | 2139.45 |

After the first 64, we froze the next 192 SHA256-selected query IDs and a
learning-design gate before reading their outcomes: at least 9 additional
0.80 repairs **or** 6 additional 0.90 repairs, no baseline anchor breaks,
and at most 10 extra *mean* cumulative context tokens/query. The extension
confirmed quality headroom (+35 at 0.80, +15 at 0.90, +31 Complete), but the
quality-first oracle cost **+20.73 tokens/query**, so the frozen gate fails.
The no-extra-context oracle recovered only +3 at 0.80 and +1 at 0.90.
A read-only cost-cap sensitivity on all 256 found that a **per-query**
20-token added-context cap yields 0.90 192 versus 190 STAY; a 40-token cap
yields 197. These per-query caps are distinct from the frozen mean-cost gate
and are descriptive only. The unchanged 0.95 count follows from preserving
the depth-10 packet set; it does not imply general action safety.

Decision: `STOP_CURRENT_EARLY_MOVE_TRAINING_UNDER_FROZEN_GATE`. Early actions
have quality headroom, especially at 0.80, but their cheap region is narrow
and the oracle uses Target truth. Inspect whether packet boundaries force
beneficial evidence to arrive with unnecessary tokens before another
controller is trained. [Pilot summary](v17state_a0_early_move_pilot/summary.json),
[256-query summary](v17state_a0_early_move_expand256/summary.json), and
[decision](v17state_a0_early_move_expand256/decision.json) record the audit.
No sealed set was read.

### V17-PACKET-R0 targeted sentence atomicity mechanism audit

STATE-A0 did not establish that extra context was useless. We therefore
separated adding the moved packet from displacing the previous packet at the
single affected stop. The train-side sample used all 15 new-192 queries with
a quality-first 0.90 repair, 15 hash-selected 0.80-only repairs and 15
hash-selected controls. Each original packet is one document title and one
source-evidence block; only natural sentence boundaries within that block
were considered. The title accompanies each tested sentence. We did **not**
invent a semantic chunker or train a model. Existing exact query/mask Target
rows were reused; 225 fresh calls cost 142,745 prompt and 10,242 generated
tokens. The tested variants were V8 baseline, whole-packet move, baseline
plus the full moved packet, baseline plus one sentence, and the moved context
with the whole packet replaced by one sentence.

At the affected stop, the 15 historical 0.90 whole-packet repairs included
12 for which baseline-plus-full also succeeded, so removal of the displaced
packet is **not universally necessary**. A sentence variant recovered 13/15;
9/15 recovered 0.90 without exceeding that query's V8 context tokens at
depth 9. For 0.80-only repairs the corresponding counts were 12/15
baseline-plus-full, 13/15 any sentence and 10/15 no-extra-context. These
are best-of-several *outcome-aware* choices, not a learned selector.

We reran the 9 no-extra-context 0.90 cases in one paired Qwen3-8B batch:
baseline depth9, whole move depth9, chosen sentence depth9, and chosen
sentence at depth7 (36 new calls, 22,141 prompt and 1,658 generated tokens).
Depth9 success was **0/9 baseline, 9/9 whole move, 9/9 sentence**; every
sentence was still no longer than its paired V8 depth9 context. However,
revealing that sentence already at depth7 caused **two baseline 0.80 breaks**.
If instead the sentence is revealed only at depth9, the previously cached
lower-anchor outputs remain unchanged by definition, the 9 depth9 repairs
survive, 5/9 reach Complete, and cumulative fixed-schedule context drops by
56 tokens per selected query on average. This is a *hindsight counterfactual*
over nine design-exposed cases, not a realizable ten-step policy yet: a
refined add-only schedule must specify how the displaced packet and residual
evidence are restored at the depth10 macro-step.

Selection risk is substantial. In the 15 matched controls, 13 were baseline
0.90 successes; at least one tested replacement sentence broke 0.90 in 8
of those 13. The study therefore locates a genuine granularity-and-timing
opportunity, while leaving cross-query identification unresolved. Decision:
`GO_PACKET_R1_LATE_FRAGMENT_CONTRACT_DESIGN_ONLY`. First freeze an exact
lossless/add-only restoration and cost contract, then run a broader train-side
five-anchor oracle under that contract. Do not train a sentence selector or
claim deployment Pareto gain from these selected examples. See
[R0 summary](v17packet_r0_targeted_atomicity_pilot/summary.json),
[paired replay](v17packet_r0_paired_replay/summary.json), and
[decision](v17packet_r0_targeted_atomicity_pilot/decision.json).

### V17-PACKET-R1 rank-10 late-sentence five-anchor oracle

We froze one refined action before Target calls: at macro-depth 9, substitute
one natural sentence (with its original document title) from V8 rank 10 for
the scheduled rank-9 packet. At macro-depth 10, restore the displaced rank-9
packet and the rank-10 residual. The renderer is source-order canonical:
the evidence set grows add-only, although the prompt string is re-rendered
rather than extended literally. Code asserts exact final full-text recovery,
one title/one occurrence per original sentence, and the original depth-10
packet set. Thus 0.60/0.70/0.80/0.95 use the exact cached V8 inputs; only
the depth-9 0.90 context differs. Position is a macro-step coordinate; the
comparison uses actual context tokens, not a claim that one sentence costs
as much as one packet.

The next 128 SHA256-selected train queries, positions 256:384, were not used
by STATE-A0 or R0. Of these, 99 had splittable rank-10 proof blocks, yielding
329 candidate sentences. The current Qwen3-8B Target was called exactly 329
times (219,493 prompt and 16,110 generated tokens). The other 29 queries
retained STAY. The five-anchor fixed schedule remained `[6,7,7,9,10]`.

| Policy on 128 train queries | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Complete | Mean cumulative context tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8 STAY | 111 | 113 | 91 | 93 | 96 | 71 | 2158.30 |
| Rank-10 whole-packet move | 111 | 113 | 91 | 91 | 96 | 73 | 2178.86 |
| Outcome-aware sentence/STAY, no per-query added context or anchor break | 111 | 113 | 91 | **104** | 96 | **81** | **2124.01** |

The frozen **oracle-opportunity** gate passes: +11 at 0.90 and +10 Complete
with -34.29 mean cumulative context tokens/query and zero baseline-success
anchor breaks. The oracle changed 86/128 decisions, often to save context
on queries without a new success. Eleven 0.90 repairs include ten cases
also repaired by the full-packet move and one additional case. This is a
large enough *upper bound* to justify collecting train-side labels, not a
deployable controller result.

A read-only, explicitly post-hoc risk diagnostic shows why learning remains
the bottleneck. Always taking the first available sentence yields 0.90
56/128 (six repairs, 43 breaks); shortest yields 52/128 (six repairs, 47
breaks); last yields 82/128 (ten repairs, 21 breaks). The original whole
packet move gives 15 repairs but 17 breaks. These simple policies are
descriptive, not contenders selected on this exposed set. They show that
sentence resolution alone does not tell the system when to STAY or which
snippet to reveal.

Decision: `GO_TRAIN_SIDE_LABEL_SCALE_AND_PRE_REGISTER_R2_LEARNABILITY_TEST`.
Keep the 128 outcomes design-exposed and sealed sets untouched. Before any
deployment claim, collect more train-side labels under the exact action
contract and evaluate a single frozen policy by query-grouped rollout against
V8, including compressor compute and Target cost. [Protocol](../../configs/v17packet_r1_rank10_late_fragment_oracle.json),
[summary](v17packet_r1_rank10_late_fragment_oracle/summary.json),
[per-query actions](v17packet_r1_rank10_late_fragment_oracle/per_query.jsonl),
and [decision](v17packet_r1_rank10_late_fragment_oracle/decision.json) record
the result.

The 11 outcome-selected new 0.90 repairs were subsequently rerun as
baseline/sentence pairs in one Target batch (22 calls). Baseline remained
0/11; the sentence remained successful in **10/11**, with no added depth9
context in all 11. This shows a small but real single-run-label instability;
the initial +11 oracle number should not be presented as 11 independently
confirmed repairs. [Paired replay](v17packet_r1_repair_replay/summary.json).

With the same frozen contract, a further 512 previously unqueried train-side
queries (SHA256 positions 384:896) yielded 424 splittable rank-10 proof
blocks and 1,485 sentence candidates. All candidates were evaluated once
under the current Qwen3-8B prompt (1,030,034 prompt and 71,796 generated
tokens). The preregistered label-scale opportunity gate asked for at least
24 no-extra-context, no-anchor-break 0.90 repairs.

| Policy on new 512 | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Complete | Mean cumulative context tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8 STAY | 449 | 453 | 354 | 381 | 338 | 273 | 2227.73 |
| Whole-packet rank-10 move | 449 | 453 | 354 | 346 | 338 | 261 | 2248.18 |
| Outcome-aware strict sentence/STAY oracle | 449 | 453 | 354 | **420** | 338 | **288** | **2185.73** |

The opportunity gate passes with +39 0.90, +15 Complete and -42.00 mean
cumulative context tokens/query. The oracle changed 373/512 queries and
never broke a baseline-success anchor by construction. But blindly taking
the first, shortest or last sentence yielded respectively 0.90 **211, 220,
310**, versus V8 **381**; they incurred 186, 176 and 93 breaks. The whole
packet move had 33 repairs and 68 breaks. Thus the refined action space is
valuable **only if the policy can identify safe deviations**. The 512 are
train-side, design-exposed by this audit, not final validation.

We also froze and ran a single low-cost query-grouped lexical diagnostic on
these 512 queries: hashed words/bigrams from query, chosen sentence and
displaced rank-9 packet; one linear head, fixed optimization, four OOF folds.
Its maximum OOF predicted success probability was only 0.704, so every
predeclared threshold 0.8/0.9/0.95/0.98 chose STAY. A descriptive top-5%
score ranking gave two repairs but eight breaks. This **does not prove**
semantic observability impossible: it only closes this cheap lexical probe.
See [512-label summary](v17packet_r1_label_scale512/summary.json) and
[lexical probe](v17packet_r2_lexical_probe/summary.json). No sealed set was
used, and no deployable checkpoint has been selected.

### V17-PACKET-R2 one frozen semantic action-success probe

Following the positive R1 oracle, we ran exactly one preregistered
DistilBERT cross-encoder diagnostic on the same 512 train-side queries.
All actions for a query stayed in one of four folds. Input contained only
the query, first-eight packet titles, the candidate rank-10 sentence and
the displaced rank-9 text. The binary training label was the actual 0.90
success of the refined depth-9 context. It used two epochs, 256 input
tokens, no checkpoint selection and no new Target calls. Fold training loss
fell from about 0.65–0.67 to 0.54–0.59. The frozen *primary* policy took
the highest-scored no-extra-context sentence on only the top 5% of OOF
queries; all others remained STAY.

| OOF policy | Switches | 0.90 | Complete | Mean cumulative context | Repairs | Breaks |
|---|---:|---:|---:|---:|---:|---:|
| V8 STAY | 0 | 381 | 273 | 2227.73 | 0 | 0 |
| Semantic top 1% (descriptive) | 6 | 381 | 273 | 2227.25 | 0 | 0 |
| Semantic top 5% (frozen primary) | 26 | **378** | 271 | 2225.26 | 1 | 4 |
| Semantic top 10% (descriptive) | 52 | 376 | 270 | 2223.17 | 3 | 8 |

The primary train-only gate fails. Input-length audit found 254/1,485
actions beyond the 256-token limit, but only 6/39 oracle-repair actions
were truncated, and the displaced-packet marker was not completely lost in
any of those 39. This limits interpretation without explaining away the
main failure. A separate **exploratory** rule using the already-paid depth7
answer count and rank-10 title absence, with a fixed last-sentence choice,
also had more breaks than repairs (at count threshold 8: 1 repair/10
breaks). It is not a validated deployment rule.

Decision: `STOP_R2_SEMANTIC_OOF_GATE`. This closes the *specific* binary
action-success head and input contract. It does not prove that the refined
evidence is intrinsically unlearnable: the actual decision is a paired,
asymmetric **STAY versus action** comparison, while this head learned only
absolute action success. Do not sweep this model's thresholds, epochs or
capacity on the exposed OOF outcomes. Further work would need a separately
frozen paired-decision hypothesis with a query-held-out test and full
encoder-compute accounting, or a change in deployable observations.
[Summary](v17packet_r2_semantic_probe/summary.json) and
[decision](v17packet_r2_semantic_probe/decision.json) retain the result.

### V17-PACKET-R2A paired-label and intervention audit

We reclassified the existing 512 train-side R1 outcomes relative to V8 and
replayed the saved R2 grouped-OOF top action. There were **no new Target calls
or training**. Among 1,441 eligible (no-extra-context) sentence actions,
56 repaired 0.90, 511 broke an existing 0.90 success, 871 preserved quality
and saved tokens, and three had no value. The 56 repair actions occurred on
only **39 independent queries**; 26 of these had exactly one repair sentence.
There were 369 queries with at least one token-only saving opportunity. Under
this frozen contract, only 0.90 can change, so a multi-anchor repair/break
tradeoff cannot occur. Complete is derived from that change and is not an
independent label.

| Diagnostic policy on 512 | 0.90 | Complete | Repair | Break | Mean cumulative context |
|---|---:|---:|---:|---:|---:|
| V8 STAY | 381 | 273 | 0 | 0 | 2227.73 |
| R2 top-5% gate + R2 choice | 378 | 271 | 1 | 4 | 2225.26 |
| R2 top-5% gate + oracle choice | 382 | 273 | 1 | 0 | 2224.80 |
| Oracle *quality-repair* gate + saved R2 choice | 406 | 283 | 25 | 0 | 2224.49 |
| Oracle quality-repair gate + oracle choice | 420 | 288 | 39 | 0 | 2223.70 |

The R2 top-5% gate selected 26 queries, of which only one had a 0.90 repair
opportunity. Thus the dominant diagnosed failure is **when to leave V8**;
sentence choice also misses 14/39 repairs even with a perfect repair gate.
An oracle choice under R2's gate removes four breaks but cannot find repairs
outside the selected queries. These are hindsight diagnostic ceilings, not
deployable gains. R2 saved only each query's best-action score, so full
within-query ranking cannot be reconstructed. The 512 outcomes are now
design-exposed; neither paired supervision nor a two-stage model has yet
shown held-out benefit.

Decision: `GO_R2B_PROTOCOL_FREEZE_ONLY`. Any next model must distinguish
quality repair from the far more prevalent token-only saving, retain an
explicit STAY action, freeze risk and compute accounting before training,
and use a fresh train-side query gate where feasible. Do not open sealed
sets or claim that an oracle gate is implementable. [Summary](v17packet_r2a_paired_label_audit/summary.json),
[per-query audit](v17packet_r2a_paired_label_audit/per_query.jsonl), and
[decision](v17packet_r2a_paired_label_audit/decision.json) record the audit.

### V17-PACKET-R2A frozen-score intervention gate audit

We ranked the same 512 design-exposed train queries by R2's **saved best-action
absolute-success score**, without model inference or Target calls. Of 512
queries, 381 already succeed at 0.90, 39 are repairable by an eligible R1
sentence, and 92 fail with no eligible repair. R2 had scores for 422 eligible
queries, including all 39 repairable failures. Random expectations below
therefore sample from those **same 422 score-eligible queries**.

| Budget among 512 | Repair opportunities found | Random expectation | Actual repair/break |
|---|---:|---:|---:|
| 1% (6) | 0/39 | 0.55 | 0 / 0 |
| 5% (26) | 1/39 | 2.40 | 1 / 4 |
| 10% (52) | 3/39 | 4.81 | 3 / 8 |
| 15% (77) | 8/39 | 7.12 | 7 / 9 |

There is no useful low-budget enrichment from **this saved score**. The 15%
point is close to random opportunity capture and still loses net 0.90
success. This does not disprove a new paired, candidate-aware representation:
the R2 OOF artifact contains only the best candidate score, no full score
distribution or saved encoder checkpoint. Nor does this audit justify
immediate expansion of costly Target labels purely to improve the old gate.
The next inexpensive diagnostic is a separately frozen candidate-set,
relative-to-V8 signal test on existing training-side outcomes. Quality
repair and token-only saving must remain separate labels. No sealed set was
opened. [Summary](v17packet_r2a_gate_signal_audit/summary.json) and
[decision](v17packet_r2a_gate_signal_audit/decision.json) retain the result.

### V17-PACKET-R2G0 relative-to-V8 candidate-bag probe

We froze one DistilBERT-sized three-class probe on the existing 512
design-exposed train queries. Each candidate input explicitly contains the
query, proposed sentence, displaced V8 rank-9 packet and current packet
titles, with fixed per-field token budgets. The labels are relative to V8:
quality break (511 eligible actions), unchanged 0.90 quality (874), or 0.90
repair (56). Four query-grouped OOF folds use the R2 split, two epochs, fixed
class weighting and no checkpoint selection. The bag score is the maximum
candidate `P(repair)-P(break)` among no-extra-context actions. No Target call
or sealed data was used.

| OOF budget among 512 | Opportunities found | Random expectation in 422 eligible queries | Repair / break | 0.90 | Complete | Mean cumulative context |
|---|---:|---:|---:|---:|---:|---:|
| V8 STAY | — | — | 0 / 0 | 381 | 273 | 2227.73 |
| 5% (26) | 2/39 | 2.40 | 1 / 6 | 376 | 270 | 2225.23 |
| 10% (52) | 4/39 | 4.81 | 3 / 14 | 370 | 268 | 2222.63 |
| 15% (77), descriptive | 7/39 | 7.12 | 6 / 18 | 369 | 266 | 2220.01 |

The frozen primary gate fails: no low-budget opportunity enrichment, and
context savings come with substantially worse quality. The 5% global score
selection is also uneven across folds (17/7/1/1 selected queries), which
limits interpretation of a globally compared OOF score. The fixed input
budget retained only part of the displaced V8 packet in 598/1,441 eligible
actions; all title-list fields exceeded their 28-token allocation. These are
material limitations of this *specific* representation, not evidence that
all observable text lacks the needed signal. Batched OOF inference averaged
about 13.1 GPU-ms/query on a shared device; this is diagnostic timing, not
end-to-end deployment latency. No deployable checkpoint was selected.

Decision: `STOP_R2G0_FROZEN_PROBE_NO_LABEL_SCALE`. Do not expand expensive
Target labels or open sealed sets on the basis of this run. Next, use the
existing 39 repairable and 92 unrepairable baseline failures to audit what
deployment-visible evidence distinguishes them and whether input truncation
hits the rare repairs, before freezing another learning hypothesis.
[Protocol](../../configs/v17packet_r2g0_relative_bag_probe.json),
[summary](v17packet_r2g0_relative_bag_probe/summary.json),
[OOF decisions](v17packet_r2g0_relative_bag_probe/oof.jsonl), and
[decision](v17packet_r2g0_relative_bag_probe/decision.json) record this run.

### V17-PACKET-R2G1 existing-output attribution audit

We used only R1 train labels, R2G0 OOF scores and the frozen tokenizer to
separate three possible causes of the failed gate: input truncation, fold
score scale, and failure-type discrimination. No model was trained and no
Target or sealed data were read.

| Query group | Count | V8 action truncated | Scored query count | Best candidate/V8 pair complete |
|---|---:|---:|---:|---:|
| Repairable V8 failure | 39 | 12 | 39 | 27 |
| Unrepairable V8 failure | 92 | 25 | 66 | 41 |
| V8 already successful | 381 | 109 | 317 | 210 |

Thus V8-action truncation affects some repairs, but is **not disproportionately
concentrated** in the 39 repairable queries. All 27 queries with a complete
repair pair remain an important test subset. Using R2G0's saved query score,
the AUC for repairable versus unrepairable failures is **0.481** pooled, or
**0.547** when both the best candidate and V8 text were fully visible (27
versus 41 scored queries). Repairable versus already-successful queries gives
0.553 pooled and 0.571 on complete best pairs. These weak AUCs apply only to
this one saved score; they do not prove the text lacks predictive information.

The global OOF ranking did suffer fold-scale distortion. Diagnostic
within-fold top-5% selection retrieves 5 opportunities against 2.59 expected
from random selection within each fold, but produces **3 repair / 3 break**.
Within-fold top-10% retrieves 9 against 4.88 random expectation, but yields
**6 repair / 12 break**. Fold-local ranking is not a deployable per-query
calibration method, and the run remains design-exposed. Input truncation or
fold calibration alone does not account for the observed deployment failure.

Decision: `STOP_TRUNCATION_OR_FOLD_SCALE_AS_SOLE_FIX`. Do not expand labels or
train a larger encoder by default. Next use existing outcomes to inspect the
mechanism distinguishing repairable and unrepairable V8 failures, including
Target-answer and evidence-combination changes; a new learning experiment
must state what additional deployment-visible information it tests.
[Audit summary](v17packet_r2g1_attribution_audit/summary.json),
[per-query measurements](v17packet_r2g1_attribution_audit/per_query.jsonl), and
[decision](v17packet_r2g1_attribution_audit/decision.json) record the result.

### V17-PACKET-MECH-A0/A1 cached answers and factorial replay

We stopped changing classifiers and decomposed the 39 R1 repairs on the
same train-side 512. The cached answer audit used no new Target calls. In
38/39 repairs, the AD answer adds a previously missing correct atom and the
selected rank-10 sentence lexically mentions a newly correct gold entity;
13/39 also have fewer false answers, and one is driven by fewer false answers
without a newly correct atom. The cached D-only arm is the original depth-8
V8 context; it succeeds in **0/39**. Yet 62/92 failures unrepairable by the
R1 substitution also have a candidate sentence mentioning a gold entity not
seen in the V8 depth-9 text. Gold mentions are retrospective diagnostics,
not proof of relation support or permitted deployment features.

We then froze 39 selected repairs and 24 unrepairable-failure controls for a
small B/A/D/AD replay. A keeps the V8 rank-9 packet and adds the selected
rank-10 sentence; AD replaces rank-9 with that sentence. A changes token
cost and is **not** a deployable budget-neutral compression action. We made
141 new Qwen3-8B calls, consuming 105,457 prompt and 7,674 generated tokens.
In paired reruns, 35/39 repairs still had B fail and AD succeed; among these
stable cases A succeeded in **26**, while the cached D alone succeeded in
none. A failed but AD succeeded for the other nine stable cases. A uses
about **46 more** depth-9 context tokens than B among stable cases, whereas
AD saves about **55**. Seven of 24 previously unrepairable controls succeeded
under A despite failing under the AD substitution. Four of the 39 original
repairs did not reproduce their original B-fail/AD-success pairing, so they
were excluded from the mechanism count.

This supports new evidence as the dominant *selected-sample* mechanism and
shows why removing the entire V8 rank-9 packet can also be harmful. It does
not identify a deployable gate or establish a population-level gain.
[Cached audit](v17packet_mech_a0_cached_audit/summary.json),
[factorial protocol](../../configs/v17packet_mech_a1_factorial_replay.json),
[factorial result](v17packet_mech_a1_factorial_replay/summary.json), and
[per-case replay](v17packet_mech_a1_factorial_replay/per_case.jsonl) retain
the evidence.

### V17-PACKET-MECH-A2 budget-neutral partial replacement pilot

To preserve more V8 evidence, we enumerated every original sentence in its
rank-9 packet that could be deferred while adding the already frozen rank-10
sentence, requiring exact Qwen3-8B depth-9 context tokens no greater than
V8. At depth10 the deferred sentence and rank-10 remainder restore the
original lossless context. On the 63 A1 mechanism cases this yields 80
feasible actions across 41 queries. We evaluated all 80 with the current
Target (71,404 prompt and 4,012 generated tokens). The rank-10 sentence was
selected using the earlier A1/R1 outcome; the rank-9 omission was not
selected using this pilot's Target output before evaluation.

Among **18** repeat-stable R1 repair cases with a feasible action, **13**
have at least one 0.90-successful partial replacement. The best successful
action saves an average **11.85 depth-9 context tokens** versus V8 across
these 13 cases. Four of 21 feasible, previously unrepairable controls also
have a successful action. This is a useful **selected-case oracle ceiling**:
the 63 cases were chosen from earlier outcome audits, and the winning rank-9
sentence omission is chosen retrospectively. We have not measured how often
the same action generator breaks queries where V8 already succeeds.

Decision: `GO_PRE_REGISTERED_SAFETY_AND_GENERALIZATION_AUDIT_NO_TRAINING`.
Next freeze a train-side V8-success sample and measure break prevalence and
token costs for this exact action generator before training a controller or
opening sealed data. [Protocol](../../configs/v17packet_mech_a2_budget_neutral_swap.json),
[result](v17packet_mech_a2_budget_neutral_swap/summary.json),
[per-action Target outcomes](v17packet_mech_a2_budget_neutral_swap/per_action.jsonl),
and [decision](v17packet_mech_a2_budget_neutral_swap/decision.json) record
the pilot.

### V17-PACKET-MECH-A3 V8-success safety audit

Before learning or deploying partial replacement, we froze the SHA256-first
64 queries from the R1 train-side sample on which V8 already succeeds at
0.90. This sample was not selected using A3 outcomes; earlier R1 outcomes on
the same train queries are design-exposed. We enumerated every original
rank-10 sentence paired with every one-sentence rank-9 omission that keeps
actual depth-9 context tokens no greater than V8. Thirty-two queries admit
153 actions. An unconditional, deployment-visible rule was frozen in advance:
choose the feasible action with fewest context tokens, then lowest sentence
indices; otherwise STAY.

| Safety diagnostic | Result |
|---|---:|
| Feasible actions that break V8's 0.90 success | 33/153 |
| Eligible queries with at least one oracle-safe action | 26/32 |
| Eligible queries where every action breaks | 6/32 |
| Frozen minimum-token rule breaks | 11/64 queries |
| Frozen rule mean depth-9 context-token change | -12.09/query |

The 11/64 break rate is 17.2% (descriptive 95% Wilson interval roughly
9.9%–28.2%). We made 153 Qwen3-8B calls, consuming 117,652 prompt and 7,480
generated tokens. The token saving under this fixed rule comes with a large
quality loss; it is not a Pareto improvement. A2's 13/18 selected-case
oracle repairs remain genuine action-space headroom, but A3 shows the
countervailing safety constraint. Even a perfect selector must STAY on at
least 6/32 eligible already-successful queries.

Decision: `STOP_UNCONDITIONAL_BUDGET_NEUTRAL_SWAP`. Before another selector
training run, use these existing action outcomes to test whether harm is
associated with deferring uniquely answer-bearing rank-9 text versus other
evidence interactions. Any gold-based analysis is retrospective only. A
future deployment rule must default STAY and demonstrate paired repair-over-
break plus token improvement on query-held-out data; no sealed set was read.
[Protocol](../../configs/v17packet_mech_a3_success_safety.json),
[summary](v17packet_mech_a3_success_safety/summary.json),
[per-action outcomes](v17packet_mech_a3_success_safety/per_action.jsonl), and
[decision](v17packet_mech_a3_success_safety/decision.json) preserve this audit.

### V17-PACKET-EVICT-A0 gold-alias protection diagnostic

We tested the proposed "protect unique answer support" explanation against
all 153 A3 actions without a new Target call. This first pass uses gold-answer
alias *mentions* as a transparent retrospective proxy, not a support-relation
classifier or deployment feature. None of the omitted rank-9 sentences
uniquely mentions a gold answer relative to the other evidence retained at
depth9. All 33 break actions defer a sentence mentioning a baseline correct
answer, but so do **95/120 safe actions**. A rule that protects every such
sentence would leave 25 safe actions from only **one query**. Thus simple
answer-name coverage cannot yield a useful safety gate on this sample.

Among the breaks, 17/33 lose a correct output answer whose gold name appears
in the deferred sentence, versus 7/120 safe actions. This suggests a real
link between some answer loss and delayed evidence, but it is known only
*after* observing the Target response and cannot be used by a controller.
Lexical duplication is not proof of equivalent relational support, and the
153 actions are clustered in 32 queries. A richer relation-level graph
remains a hypothesis, not an established remedy.

Decision: `STOP_GOLD_ALIAS_UNIQUE_SUPPORT_AS_SAFETY_GATE`. Do not train a
protected-evidence classifier from this proxy. The next efficient causal
test is a frozen +8/+16/+32 token-slack oracle on both repair and V8-success
samples: determine whether less aggressive deferral reduces break risk while
retaining a useful quality–token frontier. This remains train-side mechanism
work, not deployment validation. [Summary](v17packet_evict_a0_gold_diagnostic/summary.json),
[per-action audit](v17packet_evict_a0_gold_diagnostic/per_action.jsonl), and
[decision](v17packet_evict_a0_gold_diagnostic/decision.json) record it.

### V17-PACKET-SLOT-A0 cached functional provenance

We audited parsed answer transitions for all 153 budget-neutral A3 actions
on 32 V8-success train queries, plus one retrospectively selected successful
action for each of 13 repeat-stable A2 repair queries. This required **zero**
new Target calls and did not read sealed sets. Correct-answer losses at depth9
were classified relative to the cached depth8 and V8 depth9 outputs:

| Depth9 action outcome | Actions | Late-acquired loss only | Depth8-present loss only | Both | No correct-answer loss |
|---|---:|---:|---:|---:|---:|
| Break | 33 | 11 | 6 | 15 | 1 |
| Safe | 120 | 7 | 0 | 0 | 113 |
| Selected repair | 13 | 2 | 0 | 0 | 11 |

All 13 selected repairs add at least one correct answer compared with V8
depth9. However, these are outcome-selected oracle examples, not an estimate
of a deployment policy's repair rate. The break categories are observations,
not causal diagnoses: each action both defers default rank9 text and adds
candidate text. In particular, losing an answer already present at depth8
does not by itself establish candidate interference. A separate omission-only
arm is needed to identify this distinction. The 153 safety actions are
clustered within 32 queries, so action counts are not independent samples.

The depth10 context is canonically re-rendered from the full packet set and
does not carry forward depth9 Target state. Differences between identical
depth10 contexts cannot be called persistent reveal-order effects; they
would require an execution-reproducibility investigation. These findings
support a small targeted factorial replay before any dual-sided slot-value
training. The previously proposed +8/+16/+32 slack audit remains a separate
Pareto question; the present answer audit does not supersede it.

[Reproducible audit](../../src/evaluation/v17packet_slot_a0_functional_provenance.py),
[summary](v17packet_slot_a0_functional_provenance/summary.json), and
[per-action transitions](v17packet_slot_a0_functional_provenance/per_action.jsonl)
preserve the result.

### V17-PACKET-SLOT-A1 paired depth-9 factorial replay

We froze 13 train-side actions before new Target calls: seven mechanism-
stratified historical breaks, four outcome-selected stable repairs, and two
same-query safe controls. Each action was evaluated in one Qwen3-8B run under
four canonically rendered conditions: rank9 remainder alone (`neither`),
remainder plus original sentence (`baseline`), remainder plus candidate
(`swap`), and all three (`both`). This used **52 fresh Target calls**, 43,442
prompt tokens and 2,522 generated tokens; no model was trained and no sealed
set was read. The four contexts differ in evidence *and* length, and `both`
is a mechanism probe rather than a budget-neutral deployment action.

The cached baseline success bit reproduced in 12/13 cases; the cached swap
success bit reproduced in 13/13. Among the six historical break cases with
both bits reproduced, `both` restored 0.90 success in **5/6**; `neither`
was successful in **2/6**. The remaining stable break still failed under
`both`. All four selected repairs stayed successful under `both`, but all
four paid extra depth9 context tokens (13–40 versus baseline). One of two
matched safe swaps became a failure under `both`, demonstrating that merely
adding evidence is not uniformly safe. These small, deliberately enriched
samples cannot estimate a population break or repair rate.

The experiment supports a substantial default-evidence delay cost in the
selected failures, alongside non-monotone Target response to additional
evidence. It does **not** justify a dual-head selector yet: the prospective
deployment-visible signal for deciding *which* query should change remains
unproven. The next economical test is a preregistered bounded-slack
micro-insertion frontier on both repair and V8-success train queries, with
actual context tokens and five-level paired quality. The current four-arm
probe itself is not a Pareto result. [Frozen protocol](../../configs/v17packet_slot_a1_factorial.json),
[runner](../../src/evaluation/v17packet_slot_a1_factorial.py),
[summary](v17packet_slot_a1_factorial/summary.json),
[per-case contrasts](v17packet_slot_a1_factorial/per_case.jsonl), and
[per-call outputs](v17packet_slot_a1_factorial/per_call.jsonl) retain the
evidence.

### V17-PACKET-SLOT-B0 bounded-slack insertion pilot

Before querying the Target, we selected 12 V8-0.90-success and 12 V8-0.90-
failure examples from the design-exposed R1 train512 frame by SHA256 order,
requiring at least one exact original rank10 sentence with at most 48 actual
Qwen3-8B depth9 context tokens of slack. This eligibility depends on text and
token length, not insertion outcomes. We enumerated all such sentences:
**84 fresh Qwen3-8B calls** (76,861 prompt and 4,089 generated tokens).
Caps of +16/+32/+48 admit 20/44/84 action calls, respectively. V8's full
depth10 prefix is the next discrete V8 context point; it cannot be
interpolated to an arbitrary insertion cost.

| Rule on 24 selected queries | 0.90 success | Complete | Mean added depth9 context tokens/query |
|---|---:|---:|---:|
| V8 depth9 | 12 | 10 | 0 |
| Frozen shortest-sentence insertion | 14 (4 repair, 2 break) | 9 | 22.25 |
| +16 oracle with STAY | 14 (2 repair, 0 break) | 10 | 0.96 |
| +32 oracle with STAY | 15 (3 repair, 0 break) | 10 | 2.21 |
| +48 oracle with STAY | 19 (7 repair, 0 break) | 10 | 8.58 |

The oracle chooses STAY on already successful queries and picks the cheapest
successful candidate retrospectively on failures. Its low mean cost is thus
an **upper bound**, not a deployable policy. The fixed shortest rule is fully
deployment-visible but loses one Complete case despite net +2 at 0.90; it
does not establish the required five-level quality–token Pareto improvement.
The other four anchor outputs are structurally unchanged under this fixed
schedule: 0.60/0.70/0.80 were read before depth9 and 0.95 at the canonically
identical depth10 context. Across these 24 queries, V8 depth10 itself reaches
17/24 at 0.90 but requires a mean **+120.58** depth9 context tokens relative
to V8 depth9; none of the selected insertion points reaches depth10's token
cost. This comparison is between attainable discrete contexts, not an
exact-token matched V8 control.

The sample is intentionally balanced on V8 success and already exposed to
earlier R1 design work, so none of these rates estimates the train or unseen-
query population. The result supports a bounded-slack **action-space**
hypothesis while again exposing the unresolved opportunity/safety decision.
Do not open sealed sets or train an insertion controller from 24 queries.
The next study must freeze a larger query frame and a single deployment-
visible STAY/INSERT rule before measuring paired five-anchor outcomes; if
that rule cannot protect Complete, the oracle headroom cannot be claimed as
a method result. [Protocol](../../configs/v17packet_slot_b0_pilot.json),
[runner](../../src/evaluation/v17packet_slot_b0_pilot.py),
[manifest](v17packet_slot_b0_pilot/manifest.json),
[summary](v17packet_slot_b0_pilot/summary.json),
[per-query frontier](v17packet_slot_b0_pilot/per_query.jsonl), and
[Target outputs](v17packet_slot_b0_pilot/per_call.jsonl) preserve the pilot.

### V17-PACKET-INSERT-D0 cached gate versus candidate-choice decomposition

We decomposed the 24 design-exposed SLOT-B0 train examples using their 84
existing Target outputs, with **zero** new Target calls and no training. The
two proposed formulations "oracle gate + fixed shortest candidate" and
"fixed shortest candidate + its oracle gate" are mathematically identical;
they are reported once. A distinct forced-insertion counterfactual isolates
the cost of leaving the gate permanently ON while selecting the candidate
with hindsight.

| Decision rule | 0.90 success | Repair/break | Complete | Interventions | Mean added depth9 context tokens/query |
|---|---:|---:|---:|---:|---:|
| V8 STAY | 12/24 | 0/0 | 10/24 | 0 | 0 |
| Always choose shortest sentence | 14/24 | 4/2 | 9/24 | 24 | 22.25 |
| Shortest sentence + oracle gate | 16/24 | 4/0 | 10/24 | 4 | 3.63 |
| Always insert + oracle candidate | 19/24 | 7/0 | 10/24 | 24 | 31.38 |
| Full oracle gate + candidate | 19/24 | 7/0 | 10/24 | 7 | 8.58 |

Audit correction: the original SLOT-B0/INSERT-D0 summaries counted an
unattainable 0.95 anchor (`None`) as a Complete failure for four-anchor
examples. The corrected figures mask that anchor; original summaries are
preserved alongside the corrected JSON. Relative changes and decisions are
unchanged. No Target calls were made for this correction.

Thus a perfect gate with the frozen shortest candidate captures only **4/7**
repair opportunities: choosing a different sentence matters for the other
three. Conversely, hindsight candidate choice can avoid breaks even with
insertion always ON in these selected 24 examples, but pays 31.38 rather
than 8.58 mean extra tokens. This does not show that a deployable candidate
selector can do so; it proves both decisions matter to the desired frontier.
All three oracle rules use forbidden Target outcome information, the 24
examples were baseline-stratified and design-exposed, and no Complete gain
appears even at the full oracle ceiling on this sample.

Decision: `GO_FROZEN_DEPLOYMENT_FEATURE_PROTOCOL_DESIGN`; do not spend a new
training-query cohort on a threshold or "relation novelty" rule until the
exact pre-depth9 inputs, extractor/scorer, candidate ranking, STAY condition,
compute cost and stopping criteria are specified. A subsequent single frozen
policy needs an unbalanced train-side query sample and paired V8 depth9/depth10
evaluation. Measuring oracle-opportunity recall there would additionally
require a preregistered random subset with all eligible candidate outcomes;
one policy-chosen Target call per query cannot reveal missed opportunities.
[Reproducible decomposition](../../src/evaluation/v17packet_insert_d0_decomposition.py),
[summary](v17packet_insert_d0_decomposition/summary.json), and
[per-query categories](v17packet_insert_d0_decomposition/per_query.jsonl)
preserve the audit.

### V17-SEM-B0 single-atom eligibility audit

SEM-B0 is frozen on the next 64 untouched SHA-ordered M0B-eligible train
examples. It exposes only question, V8 S6, and the frozen rank10 atom; no
gold, Target output, fidelity, outcome, or SEM-A0B generation is loaded.
Atom lengths range 11–48 Qwen3-8B tokens (median 32.5). The package is
**`AWAITING_TWO_INDEPENDENT_HUMAN_ANNOTATIONS`** under frozen agreement,
kappa, coverage, and direct-agreement gates. No teacher/Target call is
authorized until this non-automatable requirement is met. [SEM-B0 report](v17sem_b0_single_atom_eligibility/REPORT.md)
and [protocol](v17sem_b0_single_atom_eligibility/ANNOTATION_PROTOCOL.md)

SEM-B0R replaced the unavailable two-human study with one transparently
AI-assisted primary annotation plus same-reviewer adversarial audit; human IAA
and kappa remain N/A.  Only 15/64 single atoms were eligible, so that contract
stopped.  A fixed adjacent-atom closure audit then reached 40/64, showing that
the atomizer often split required relations.  Fresh SEM-B1 V8/Raw/Semantic
Target runs produced Complete 38/40/40; Semantic matched Raw's eight safe
repairs while reducing any-anchor-break queries 10→8 and cumulative overhead
+222.22→+47.88 tokens/query.  A ≤fact-length extractive control was statistically
inconclusive against Semantic (active-query Complete 24 vs 25; paired bootstrap
95% interval [-6,8]) and neither dominated the other.  See the
[B0R report](v17sem_b0_single_atom_eligibility/REPORT_B0R.md),
[C0 report](v17sem_c0_two_atom_closure/REPORT.md), and
[B1 report](v17sem_b1_ideal_representation_positive_control/REPORT.md).

### V17-SEM-D0 automatic grounded extractive audit

On 96 new train queries, 29 had a valid one/two-span, exact-source compact
representation within 16 Qwen3-8B tokens.  The label-blind lexical policy
recovered 23 of those 29 opportunities (79.3% recall), but emitted on 79
queries and achieved only 29.1% semantic precision.  Exact provenance therefore
does not solve relation closure: many emitted spans mention the requested
entity while omitting the predicate that makes it an answer.  The frozen 95%
precision gate fails, so SEM-D1 Target inference remains closed.  See the
[SEM-D0 report](v17sem_d0_automatic_grounded_extractive/REPORT.md) and
[summary](v17sem_d0_automatic_grounded_extractive/summary.json).

### V17-SEM-D0.5 relation-closure verifier

SEM-D0.5 decomposed all 56 invalid SEM-D0 emissions and tested a frozen,
precision-first relation verifier on 96 previously unused canonical training
queries. The fresh cohort contained 77 query/source pairs for which AI-assisted
review found a legal closed extract under the 16-token contract. The verifier
emitted 17 candidates: 16 were valid (94.1%; Wilson 95% lower bound 73.0%), for
only 20.8% recall over eligible source universes. It failed the preregistered
95% precision, 80% Wilson-lower-bound, 50% recall, and 20-emission gates. No
Teacher/Target calls were made and SEM-D1 remains closed. The failure localizes
the next problem to relation-directed span construction, rather than generic
exact-span provenance or a looser verifier threshold. See the
[SEM-D0.5 report](v17sem_d05_relation_closure_verifier/REPORT.md).

### V17-SEM-E0 slot-first design audit

SEM-E0 tested the proposed `relation first -> construct -> minimize` ordering on
the exposed D0.5 cohort before spending a fresh cohort. A question-only parser
covered all 96 queries, but this was syntactic template coverage rather than a
semantic success metric. The slot-first constructor emitted 40 candidates;
adding answer-type/domain cues reduced this to 29. A stricter second-pass
AI-assisted design review found only 19/29 (65.5%) closed all relations and
domain constraints. Errors included missing domain constraints,
predicate mismatches, and a proposition-linkage failure in which joining a
document title to a source fragment changed the fragment's subject. E0C was
therefore not run. The next admissible implementation must preserve explicit
subject-predicate-object/constraint linkage and store inspectable offsets for
every claimed witness. No Teacher/Target calls or sealed-set reads occurred.
See the [SEM-E0 report](v17sem_e0_slot_first_source_backed/REPORT.md).

### V17-SEM-E0P proposition-backed preflight

E0P upgraded the design object from matching tokens to a source proposition
with offsets and an explicit linkage type. For the 19 surviving E0 designs,
using the entire enclosing source sentence as a conservative provenance witness
required a median 54 Qwen3-8B tokens (range 22–83); 0/19 fit 16 tokens, 1/19 fit
24, and 4/19 fit 32. These are upper bounds on minimal closure length, not proof
that compact clauses cannot exist. The result exposes the missing component:
reliable clause/proposition extraction. The current environment has no frozen
dependency/SRL runtime, and offsets alone do not prove entailment. Fresh E0C2
was therefore not opened. No Teacher/Target calls or sealed-set reads occurred.

### V17-SEM-E0Q constrained proposition linker

The frozen Qwen3-14B positive control received question schemas and numbered
source sentences and could only select exact quotes plus a finite link type or
abstain. Across the 96 exposed design queries, only 60 outputs passed JSON and
exact-quote checks; 46 were hard-valid links. Strict AI-assisted semantic review
accepted 25/46 (54.3%). It failed known regressions including subject swap,
Higher-Broughton argument mismatch, unresolved coreference, wrong predicate,
missing constraint, and answer-role inversion. This used 96 Teacher calls but
zero Target calls and no sealed sets. The result stops relation-extraction
engineering before fresh E0C2 rather than initiating prompt tuning on the
exposed cohort. See the
[SEM-E0Q report](v17sem_e0q_constrained_linker/REPORT.md).
provide the reviewable next step.

### V17-PACKET-FRAG-A0/A1 title–sentence component and boundary audit

We froze 21 decisive SLOT-B0 actions (16 repairs, five breaks) and six
SHA-first neutral/continuous-change controls, then ran a single fresh
Qwen3-8B four-arm comparison at the same V8 depth-9 slot: baseline,
title-only, sentence-only, and original title+sentence. All 108 calls used
the same Target prompt; no model was trained or sealed set read. Fresh
baseline reproduced all 21 decisive cached success bits, while the original
combined action reproduced 20/21.

| Cached outcome | Cases | Title-only 0.90 success | Sentence-only | Combined fresh |
|---|---:|---:|---:|---:|
| Repair (baseline fail) | 16 | 8 | 8 | 15 |
| Break (baseline success) | 5 | 5 | 3 | 0 |

Four repairs occur only with title+sentence combined. Three breaks occur
only with the combination, even though title and sentence individually
preserve the baseline's success. Thus neither component has a uniform
positive or negative effect. The sampled title-only repairs are inexpensive
(mean +7.2 context tokens among the 16 originally repairable actions), but
the title is usually the gold-answer string in this constructed QAMPARI
pool. This is a mechanism diagnostic on nine outcome-selected decisive
queries, **not** evidence for a deployable title-only policy or answer
support model.

A separate zero-call abbreviation-boundary audit shows that merging only
single-letter initial splits changes 1,485 R1 candidate fragments into
1,311 lossless units and reduces ≤3-word fragments from 227 to 135. Only
three of the 21 decisive actions would change, all from one repair query.
So fragment construction needs repair, but this particular error cannot be
the dominant cause of the selected repair/break pattern. Bare headings and
other discourse boundaries were not automatically filtered.

Decision: `STOP_SUPPORT_OR_BOUNDARY_AS_SINGLE_ROOT_CAUSE`. Preserve the
matched component result and the proposed lossless unit reconstruction,
but do not train or scale labels from either yet. A next policy study must
use a deployment-visible candidate rule on a natural train-side cohort,
account for title-as-answer shortcuts, and test paired five-anchor quality,
Complete, context and Target cost against V8. [Full report](v17packet_frag_a0_component_ablation/REPORT.md),
[A0 summary](v17packet_frag_a0_component_ablation/summary.json),
[A0 per-case comparison](v17packet_frag_a0_component_ablation/per_case.jsonl),
and [A1 boundary summary](v17packet_frag_a1_boundary_audit/summary.json)
preserve the findings.

### V17-PACKET-FRAG-B0 natural-query title-only pilot

Because FRAG-A0 was selected by cached repair/break outcome, we froze a
natural 128-query train-side cohort outside R1 by SHA256 order and paired
fresh V8 depth9 calls against V8 plus the rank-10 document title only. This
used 256 Qwen3-8B calls and no training or sealed-set access.

| Paired depth9 result | V8 | Title-only |
|---|---:|---:|
| 0.90 success | 100/128 | 101/128 |
| Repairs / breaks | — | 6 / 5 |
| Added context tokens/query | 0 | +8.45 |
| Added Target tokens/query | 0 | +10.78 |

Mean F1 changed by +0.00905, but the paired-query bootstrap 95% interval
includes zero (−0.00080 to +0.01896); the net-success-count interval is
−5 to +8. Other anchors and Complete were not freshly evaluated, so this
is **not** a five-anchor Pareto result. The uniform title-only action fails
to reproduce the apparent safe benefit of A0's outcome-selected cases.
The source title is often the gold-answer string in this constructed pool;
even a stronger title-only result would need an answer-cue validity check.
Decision: `STOP_UNIFORM_TITLE_ONLY_REVEAL`; do not fit a new gate on this
exposed cohort. [Report](v17packet_frag_b0_title_only_natural_pilot/REPORT.md),
[summary](v17packet_frag_b0_title_only_natural_pilot/summary.json), and
[paired outcomes](v17packet_frag_b0_title_only_natural_pilot/per_query.jsonl)
record the test.

### V17-PACKET-OBS-A0 protected-insertion data gate

A read-only audit checked whether the existing protected-insertion labels can
support a strong paired-effect observability probe. SLOT-B0 has **84 actions
but only 24 independent, outcome-balanced queries**: 16 repairs on seven
queries, five breaks on two queries, and 63 unchanged 0.90 outcomes. FRAG-A0
replayed 27 outcome-selected actions on 15 queries: the baseline 0.90 bit
agreed in 27/27 and the combined title-plus-sentence bit in 26/27 (15/16
selected repairs, 5/5 selected breaks). The repeat subset cannot estimate
population label noise. R1/R2G0 has more queries but uses replacement rather
than protected insertion, so its labels cannot be pooled here.

Decision: `STOP_STRONG_OBSERVABILITY_PROBE_ON_CURRENT_INSERTION_LABELS`.
First collect an outcome-blind train-side cohort under one frozen insertion
rule with matched baseline/action Target calls and a small repeat subset.
This preserves the distinction between oracle action headroom and query-heldout
learnability while avoiding a misleading strong-model result on 24 queries.
[Report](v17packet_obs_a0_data_gate/REPORT.md) and
[summary](v17packet_obs_a0_data_gate/summary.json) record the gate.

### V17-PACKET-OBS-A1 natural protected-insertion labels and repeat

To replace the 24-query outcome-balanced pilot with a natural train-side
frame, we froze the shortest eligible rank10 title-plus-sentence insertion
at V8 depth9 (default context preserved, slack <=48 tokens). SHA-first
selection yielded 256 independent queries from 500 eligible among 781
P0 queries not used by R1 or the title-only pilot. The canonical Qwen3-8B
ran 512 fresh, paired baseline/action calls. No sealed set was accessed.

| Depth9 policy | 0.90 | Repairs / breaks | Mean extra context tokens |
|---|---:|---:|---:|
| V8 baseline | 180/256 | — | 0 |
| Fixed protected insertion | 186/256 | 27 / 21 | +26.64 |

Mean F1 increased by 0.01394, but query-bootstrap 95% intervals include
zero for net 0.90 success (-7 to +20) and F1 (-0.00362 to +0.03113).
Target tokens rose by 28.93/query. A selected matched repeat of 10 original
repairs, 10 breaks and 10 neutral actions reproduced all 30 categories;
this is conditional repeatability, not a population noise estimate.
Historical P0 depth10 was 216/256 at 92.70 more context tokens/query than
the insertion point, but it was not freshly paired. Other anchors and
Complete were not freshly tested here, so this is **not** an end-to-end
Pareto gain. The uniform action is too risky for deployment; the 256-query
cache can support a limited diagnostic of pre-Target effect prediction,
with only 27 repair and 21 break queries. [Report](v17packet_obs_a1_natural_insertion/REPORT.md),
[summary](v17packet_obs_a1_natural_insertion/summary.json), and
[conditional repeat](v17packet_obs_a1_natural_insertion/repeat_summary.json)
record the result.

### V17-PACKET-OBS-A2 paired-text lexical control

We ran one frozen four-fold, query-grouped TF–IDF/ridge control on the
OBS-A1 protected-insertion labels, using full baseline/action texts with
explicit field markers and continuous ΔF1 supervision. OOF Spearman for
effect prediction was **0.1045** and repair-versus-break AUC on 48 decisive
queries was **0.5291**. The fixed 5% intervention budget yielded **0 repair /
1 break**; 10% yielded **2 repair / 2 breaks**, no better than random
opportunity capture. No new Target calls or sealed data were used.

Decision: `STOP_CHEAP_PAIRED_TEXT_GATE`. This tests one lexical
representation, **not** a semantic or information-theoretic observability
ceiling. With only 27 repair and 21 break queries, a stronger-model negative
result would remain uncertain; uniform insertion itself is also costly and
break-prone. Do not tune thresholds or claim deployable Pareto improvement.
[Report](v17packet_obs_a1_natural_insertion/PAIRED_TEXT_CONTROL.md) and
[OOF summary](v17packet_obs_a1_natural_insertion/paired_text_control_summary.json)
preserve the diagnostic.

### V17-PACKET-OBS-A3 five-anchor value bound

We checked the final-objective opportunity before scaling labels or training
a stronger semantic controller. Under the frozen `[6,7,7,9,10]` schedule,
OBS-A1's fresh depth9 0.90 outcomes and P0's unchanged other prefixes
project V8 versus uniform protected insertion as:

| Policy (256 natural train queries) | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Complete |
|---|---:|---:|---:|---:|---:|---:|
| V8 | 223 | 222 | 172 | 180 | 173 | 129 |
| Uniform shortest insertion | 223 | 222 | 172 | 186 | 173 | 127 |

Complete changes by nine repairs and eleven breaks. Only **9/27** 0.90
repairs can become Complete repairs; among the other 18, fifteen still fail
the earlier 0.80 anchor (possibly also other anchors). Even a hindsight
perfect Complete gate for this single action is only **129→138/256** and
requires identifying nine rare queries. The projection combines fresh
OBS-A1 depth9 calls with historical P0 calls at other depths; it is not a
freshly paired end-to-end validation.

Decision: `STOP_DEPTH9_INSERTION_LABEL_SCALE_AND_STRONG_PROBE_FOR_COMPLETE`.
This action cannot change the earlier 0.80 or later 0.95 context; training a
larger gate on it would optimize a narrow 0.90 surrogate instead of the
five-anchor quality–token goal. First establish a multi-anchor action-space
value ceiling before new controller training. [Report](v17packet_obs_a1_natural_insertion/COMPLETE_BOUND.md)
and [summary](v17packet_obs_a1_natural_insertion/complete_bound.json) retain
the analysis.

### V17-TRAJ-M0A single-trajectory failure-locus map

A zero-Target-call audit mapped the 1,421 canonical Qwen3-8B V8 chains under
the frozen `[6,7,7,9,10]` schedule. Complete reproduced as 775/1,421; 646
queries fail at least one attainable anchor. Failures at 0.80 and 0.90
co-occur on **221** queries. Isolated single-anchor failures number 127 at
0.80, 106 at 0.90, and 53 at 0.95. The largest paired failure locus is
thus early 0.80 plus later 0.90, not isolated late 0.95.

Decision: `GO_TRAJ_M0B_CONTRACT_AND_COST_PREFLIGHT`. A rank-10 atom promoted
to depth7 is the first bounded single-trajectory hypothesis able to touch
both the 0.80 and 0.90 reads. Its early insertion cost is paid repeatedly
over the 0.70, 0.80 and 0.90 requests; residual restoration at depth10 only
avoids duplicate final text and does not erase earlier cumulative cost.
Existing R0 replay already found two 0.80 breaks among nine selected early
revelations, so safety remains unresolved. The decline in 0.95 from depth10
to depth12 does not identify the rank10 packet as the cause and does not
justify a rank10 partial-reveal Target run yet. First freeze the exact
add-only/recovery and cost contract, then run a small paired multi-anchor
pilot only if the preflight is valid. [Report](v17traj_m0a_failure_locus/REPORT.md)
and [summary](v17traj_m0a_failure_locus/summary.json) preserve the audit.

### V17-TRAJ-M0B/M0C rank-10 atom early borrow

M0B froze the shortest exact rank-10 title-plus-sentence fragment with at
most 48 actual Qwen3-8B depth7 context tokens of slack. It is eligible on
951/1,421 train trajectories; all 1,421 depth10 contexts reconstruct exactly.
Its cost is **78.38 extra cumulative context tokens per eligible query**
under `[6,7,7,9,10]`, since depth7 is read twice and the fragment remains
visible at depth9. Recovering residual text at depth10 does not refund these
earlier prompt tokens.

M0C then selected SHA-first 128 eligible queries from a training frame
excluding R1, OBS-A1 and FRAG-B0, and ran **768 fresh matched Qwen3-8B calls**
at depths 6, 7, 9 and 10. The same action trajectory serves all five levels.

| Uniform policy | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Complete | Mean cumulative context |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8 | 116 | 108 | 84 | 93 | 89 | 68 | 2291.14 |
| Early rank-10 atom | 116 | 112 | 100 | 101 | 89 | 79 | 2367.52 |

Complete gains 19 repairs and suffers eight breaks, net +11/128 with paired
bootstrap interval +1 to +21. The uniform action uses +76.38 cumulative
context and +71.88 actual Target tokens/query. This is the first fresh
paired single-trajectory action in the current branch to improve both 0.80
and 0.90, but it does not strictly dominate V8 because cost rises.

The preregistered safe-Complete oracle gate required >=8 repairs at <=10
mean extra cumulative context tokens/query. Its 19 repairs cost **+11.37**
even under hindsight selection, so the formal decision is
`STOP_M0C_COSTED_ORACLE_GATE`. The near miss is not a license to retune the
threshold or train a controller. Earlier P0 fixed-schedule comparisons and
post-result shorter-slack slices are descriptive only; they suggest a
possible new trade-off point but do not establish a deployment Pareto gain.
Existing safety and cross-query identification problems persist, while
depth10 0.95 cannot change under exact final-context recovery.
[M0B report](v17traj_m0b_borrow_preflight/REPORT.md),
[M0C report](v17traj_m0c_borrow_pilot/REPORT.md), and
[M0C paired summary](v17traj_m0c_borrow_pilot/summary.json) document the
contract, costs and decision.

### V17-TRAJ-M1 low cumulative cost borrow

The M1 protocol kept M0B's exact shortest rank-10 atom and the same
depth-7/depth-9 borrow trajectory, but required the **measured five-level
cumulative context slack <=72 tokens**; otherwise the policy stayed on V8.
After excluding prior action cohorts, only 116 minimally exposed eligible
train queries remained; 47 passed the cost cap. We made 558 fresh paired
Qwen3-8B calls and kept the lineage-clean 581, internal, development and
confirmation sets closed. The new action changed five-anchor success from
`[108,104,86,79,77]` to `[108,103,91,81,77]`, and Complete from
`62/116` to `66/116` (six repairs, two breaks), at **+18.08 cumulative
context tokens/query** and +29.10 Target prompt/output tokens/query.
The paired Complete bootstrap interval for the +4 net change spans roughly
−1 to +10 queries. Six hindsight no-anchor-break Complete repairs cost 39.5
extra cumulative context tokens per repaired query. The frozen gate required
at least eight such repairs and <=50 tokens per repair, so its decision is
**`STOP_M1_EXTRACTIVE_BORROW_GATE`**. The cheaper action retains some
multi-anchor benefit, but this narrow oracle opportunity does not authorize
selector training. It does not prove that generated semantic evidence is
necessary or rule out other action designs. [M1 report](v17traj_m1_low_cost_borrow/REPORT.md)
and [paired summary](v17traj_m1_low_cost_borrow/summary.json) preserve the
protocol, data and caveats.

### V17-TRAJ-M2 budget-neutral promote–delay exchange

M2A froze a narrow extractive exchange: promote the same M0B rank10 atom
at depth7 while delaying one rank7 proof sentence, then restore exact V8
evidence at depth10. Costs were checked on **rendered token sequences**,
not inferred from raw fragment lengths. Among 1,421 train trajectories,
508 had at least one legal exchange, totaling 941 actions. The real
`[6,7,7,9,10]` schedule means **0.70 can change**, contrary to the original
proposal. Exact depth10 recovery means 0.95 is identical by construction.

M2B prospectively tested 223 such actions on 128 SHA-selected train queries
with 958 fresh paired Qwen3-8B calls. A fixed cost-nearest-zero action saved
**36.35 cumulative context tokens/query** but changed Complete **52→44**:
11 repairs and 19 breaks. It changed anchor successes from
`[107,114,71,84,73]` to `[107,107,73,77,73]`, with marked 0.70 and 0.90
regressions. The hindsight no-anchor-break oracle found **15 Complete
repairs across five failure patterns**, passing the frozen action-space gate
(`GO_M2_LEARNABILITY_DESIGN_ONLY`). This only establishes that some safe
budget-neutral opportunities exist; it does not solve cross-query action
selection or establish a deployable Pareto gain. [M2A report](v17traj_m2a_promote_delay_preflight/REPORT.md),
[M2B report](v17traj_m2b_promote_delay_pilot/REPORT.md), and
[M2B summary](v17traj_m2b_promote_delay_pilot/summary.json) contain the
contract and paired outcomes.

Separately, existing CANON-P0 prefix outputs locate 0.95 rollback among
1,131 attainable train queries: 179 succeeded at depth10 but failed at
depth12; 73 first failed at 10→11 and 106 at 11→12. This is a location
audit, not a causal attribution to either late packet.

M2C then tested whether the 15 hindsight-positive M2B queries can be found
from `(query, S6, promoted atom, delayed sentence)` without new Target calls.
A frozen three-fold query-grouped DistilBERT probe obtained safe-action AP
**0.101** at 0.085 prevalence and harm AP **0.329** at 0.314 prevalence.
At the primary 10% intervention budget it found one safe repair versus 1.52
expected opportunity queries under random query selection, caused three
Complete breaks, and changed Complete 52→50. At 20%, it found two repairs
and caused five breaks. The formal decision is
**`STOP_M2C_RELATIVE_EXCHANGE_PROBE`**. Lower token use with lower quality
is not a Pareto gain. Do not expand labels, tune a threshold, or train a
larger selector for this exact exchange representation. [M2C report](v17traj_m2c_relative_exchange_probe/REPORT.md)
and [summary](v17traj_m2c_relative_exchange_probe/summary.json) preserve the
OOF result.

### V17-SEM-A0 existing semantic packet audit

A zero-call provenance/coverage audit found that the repository's versioned
Qwen3-14B semantic packet assets cover 60 old development examples and have
**zero overlap with canonical train1421**. More importantly, the existing
`PacketGenerator` prompt uses source text and source-derived constraints but
does not receive the query. It is gold-clean and source-conditioned, not the
query-conditioned representation proposed for SEM-A1. Packet rows store
their source and hard-check numbers/titles, but unseen entities are only
warnings; no source-span or entailment certificate exists. Per-packet prompt
tokens, generated tokens, and latency were not recorded. The decision is
**`STOP_EXISTING_PACKETS_AS_DIRECT_SEM_A1_INPUT`**. This does not reject
semantic representation; it prevents an invalid test using the wrong
generator contract. [SEM-A0 report](v17sem_a0_existing_packet_audit/REPORT.md)
and [summary](v17sem_a0_existing_packet_audit/summary.json) document the gap.

SEM-A0B then generated two disjoint SHA-selected 32-query grounding-audit
cohorts with Qwen3-14B, loading only the question, one frozen rank10 atom,
and the token budget. No gold, Target output, fidelity, or outcome label was
loaded. V1 obtained exact quotes in 31/32 but only 18/32 facts within the
16-token limit and **8/32** automatic-valid items. A revised prompt was not
rescored on V1: it used the next 32 queries, forbade absence statements, and
added a ten-word limit. V2 reached 30/32 token compliance but only 26/32 exact
quotes and **18/32** automatic-valid items. Empty outputs, non-exact quotes,
and query-suggested relations unsupported by the source remained. Both are
far below the frozen >=31/32 predicate-grounding gate, so no Target calls or
SEM-A1 occurred. Decision:
**`STOP_SEM_A0B_SINGLE_ATOM_GENERATOR_CONTRACT`**. This rejects the current
`q + single atom -> short fact` generator contract, not semantic compression
in general. [SEM-A0B report](v17sem_a0b_query_conditioned_positive_control/REPORT.md)
records generation costs and failure modes.

### V17-PACKET-INSERT-C0 pre-Target cheap-signal audit

We audited the **84 existing protected-insertion actions on 24 design-exposed
train queries**, with no Target calls, training, gold features or sealed-set
access. Continuous F1 and the 0.90 threshold were kept separate: 16 action
repairs from 7 queries, 5 breaks from 2 queries, 18 same-threshold F1 gains,
5 same-threshold F1 losses and 40 unchanged outcomes. These are action counts,
not independent-query frequencies. Because only depth9/0.90 changes under
this fixed schedule, a threshold repair on a query already failing another
anchor cannot improve Complete.

We deliberately tested only inexpensive, fully specified pre-Target textual
proxies: question-token overlap with candidate proof; query terms in the
candidate absent from V8 depth9 proof; maximum candidate/V8-packet Jaccard
redundancy; their simple overlap-times-nonredundancy product; and actual
token slack. These are **not** semantic relation entailment or an
interference detector. Literal novel query-term overlap is nonzero for just
**1/84 actions**, so it cannot operationalize the proposed relational novelty
feature here.

| Pre-Target candidate feature (max over candidate set) | Query AUC for any repair | Query-bootstrap descriptive 95% interval |
|---|---:|---:|
| Query-token overlap | 0.412 | [0.178, 0.657] |
| Literal novel query-term overlap | 0.471 | [0.400, 0.500] |
| Maximum packet redundancy | 0.546 | [0.289, 0.773] |
| Overlap × nonredundancy | 0.424 | [0.200, 0.667] |
| Token slack | 0.626 | [0.385, 0.833] |

Within-query continuous-F1 ordering gives overlap × nonredundancy a
descriptive macro concordance of 0.727 across 16 queries with varied action
F1, but the strict repair/non-repair candidate-ranking comparison is
informative in only **three** queries. Thus a possible candidate-ranking hint
does not solve query-level opportunity detection. With harmful actions in
only **two** queries, no interference-risk separation can be estimated
credibly. All intervals are unstable because the 24-query frame was
baseline-balanced and already design-exposed.

Decision: `STOP_INSERT_C1_CURRENT_CHEAP_PROXIES`. Do not choose a Q/N/I
threshold from this sample and spend a new 256-query cohort on it. This
stops these *literal proxies*, not every possible low-cost deployment-visible
semantic signal. Any next signal study must first name an actual frozen
relation-support/interference extractor, verify that it never reads Target
outputs or gold, measure its per-query compute cost, and define a single
pre-Target policy. Only then can a fresh natural train cohort test that
policy without tuning. [Reproducible audit](../../src/evaluation/v17packet_insert_c0_visible_signal.py),
[summary](v17packet_insert_c0_visible_signal/summary.json), and
[per-action features and outcomes](v17packet_insert_c0_visible_signal/per_action.jsonl)
preserve the result.

### V17-PACKET-REP-A0 certified-proof provenance audit

Before training a query-conditioned answer-support encoder, we ran a
zero-Target-call, retrospective provenance audit on the design-exposed R1
train512 and SLOT-B0 train24. The constructed QAMPARI pool stores ten
certified answer atoms per query. Its `Document title` is the answer string,
and the rank10 packet can be matched back to one complete atom proof.

| Train-side frame | Gold-answer rank10 title | Exact full-proof match |
|---|---:|---:|
| R1 train512 | 511/512 | 511/512 |
| SLOT-B0 actions | 84/84 | 84/84 |

Crucially, **all 39 R1 repairable failures and all 92 unrepairable failures**
have a gold-answer rank10 title. In SLOT-B0, every sentence option within a
query inherits the same gold title; gold-atom novelty varies in **0/24**
queries and distinguishes **0/12** within-query repair versus nonrepair
pairs. Thus even perfect *answer-identity* coverage cannot be the missing
decision variable for this particular rank10 action pool. It does **not**
rule out sentence-level relation support, partial-proof sufficiency,
interference, or ordering effects.

The construction certificate is block-level provenance, not sentence-level
logical entailment. In SLOT-B0, only 6/16 repair sentences and 1/5 break
sentences literally contain a term from their proof's relation-overlap
certificate; this lexical diagnostic cannot safely label the remaining
sentences. `Document title` also reveals the gold answer by construction,
creating a serious shortcut risk for a support encoder. Gold/provenance must
remain retrospective or train-only, never a deployment input.

Decision: `STOP_REP_A0_AUTO_SENTENCE_LABELING`. Do not train on every proof
sentence as a positive, and do not treat atom novelty as evidence of
deployable repairability. First validate sentence-level support labels on a
small, query-grouped train-side set; then test their incremental value over
the title/provenance shortcut on existing counterfactuals. The balanced,
design-exposed 24-query pilot cannot justify a population-level claim.
[Script](../../src/evaluation/v17packet_rep_a0_provenance_audit.py),
[summary](v17packet_rep_a0_provenance_audit/summary.json), and
[per-action audit](v17packet_rep_a0_provenance_audit/per_action.jsonl)
make these counts reproducible.

### V17-PACKET-REP-A1 cached answer-transition audit

We next compared the cached depth9 answer list against all 84 protected
insertion outputs from SLOT-B0, using exactly the same QAMPARI parser,
aliases, and F1 scorer as the Target evaluation. Recomputed F1 was asserted
equal to every stored score. No Target calls or sealed data were used.

| Action outcome | Actions / queries | Candidate's gold answer newly output | Other gold answers newly output | Previously output gold answers lost |
|---|---:|---:|---:|---:|
| 0.90 repair | 16 / 7 | 9 actions / 6 queries | 11 actions / 4 queries | 0 |
| 0.90 break | 5 / 2 | 0 | 0 | 5 actions / 2 queries |
| Same-threshold F1 gain | 18 / 12 | 16 actions / 11 queries | 7 actions / 3 queries | 3 actions / 1 query |
| Same-threshold F1 loss | 5 / 3 | 0 | 0 | 4 actions / 2 queries |

The 16 repair actions are **not all direct recovery of the inserted packet's
gold answer**: seven repairs do not output that answer, but do add at least
one different gold answer. Some actions add both. Every observed 0.90 break
loses existing gold answers; none adds the inserted packet's gold answer.
Thus the Target's answer-list response is a material part of the action
effect. Sentence-level support may help identify direct gains, but factual
coverage alone cannot explain all repairs or the observed breaks.

These are descriptive transitions, not a causal attribution to attention,
order, or decoding. Only seven repair queries and two break queries occur in
this selected, baseline-balanced frame; actions within each query are
correlated. The next proposed evidence-support model must demonstrate
query-held-out incremental value **beyond** the gold-title/provenance
shortcut and retain explicit STAY protection before any new Target cohort is
spent. If sentence labels cannot be audited reliably, stop this branch rather
than turning block-level certificates into noisy sentence targets.
[Script](../../src/evaluation/v17packet_rep_a1_answer_transition_audit.py),
[summary](v17packet_rep_a1_answer_transition_audit/summary.json), and
[per-action transitions](v17packet_rep_a1_answer_transition_audit/per_action.jsonl)
preserve the decomposition.

### V17-EVAL-A0 quality-label and decision-alignment audit

We ran a read-only train-side audit to test whether the 0.90 binary boundary
itself explains the repeated learned-decision failures. The original five
anchors and all gates remain unchanged. The R1 train512 cache has 1,485
sentence actions; 1,441 do not increase actual depth9 context tokens.
Every cached baseline/action F1 was independently recomputed from the saved
parsed answer list and ten gold atoms. Stored lists must **not** be passed
through the raw-output parser again: a saved list beginning `No. 17 Squadron`
would then be mistaken for a yes/no response. This is an audit parsing pitfall,
not evidence of an error in the original Target calls.

| No-extra-context action outcome | Actions | Independent queries with at least one |
|---|---:|---:|
| Continuous F1 gain | 143 | 67 |
| Continuous F1 loss | 670 | 296 |
| Equal F1 | 628 | — |
| 0.90 repair | 56 | 39 |
| 0.90 break | 511 | 238 |

Of the 143 F1-improving actions, **87 do not change the 0.90 success bit**;
38 queries have at least one such hidden gain. Conversely 159 F1-losing
actions leave that bit unchanged. Among 628 equal-F1 actions, 575 exchange
some matched gold-answer identities. This last observation is not a failure
under the existing set-F1 contract; it illustrates how much trajectory
behavior a scalar quality label can conceal.

A hindsight policy that maximizes continuous depth9 F1 over eligible actions
plus STAY, then minimizes tokens on ties, raises mean F1 from 0.87720 to
0.90074. It chooses a real F1 gain on 67/512 queries, preserves the original
0.90 oracle ceiling **381→420**, Complete **273→288**, and changes mean
cumulative context by **-37.10 tokens/query**. This is an upper bound that
uses forbidden Target outcomes; it shows that continuous supervision contains
additional partial-credit opportunities, not that a deployable policy can
find them. The existing frozen R2G0 OOF best-action score has AUC **0.532**
for detecting any such F1-gain query among 422 eligible queries. Its top-26
and top-52 globally scored queries find 5 and 9 gain opportunities versus
random expectations 4.13 and 8.26. Thus simply relabeling the same saved
score does not expose a useful low-budget intervention region.

Changing the threshold is not a free solution. On these same no-extra-context
R1 actions, a retrospective 0.85 comparison still yields 54 repairs and 205
breaks, versus 56 and 511 at 0.90; this is **not** a new benchmark or a
matched-rate policy comparison. On the overlapping fresh V8 train1421
prefix chains, depth9/depth10 success counts are 1175/1270 at 0.85 and
1050/1223 at 0.90. Success-to-failure transitions remain common along the
12-prefix chains (transition edges, not independent queries): 204 at 0.85,
186 at 0.90, and 271 at 0.95. A lower
threshold changes prevalence but does not make the Target response monotone.

Decision: `KEEP_FIVE_ANCHOR_EVALUATION; ADD_CONTINUOUS_F1_DIAGNOSTICS`.
The 0.90 label discards meaningful partial credit, but the current frozen
OOF score does not identify the extra opportunities and action safety remains
the main constraint. Do not claim that a new threshold passes the original
gate, or train the same controller again solely with a smoother label.
Before any new Target cohort, a new signal must demonstrate query-held-out
incremental value over V8 and the old OOF score, protect existing successes,
and count its inference cost. These are design-exposed diagnostics, not an
independent estimate of deployment performance.
[Audit script](../../src/evaluation/v17eval_a0_alignment_audit.py),
[summary](v17eval_a0_alignment_audit/summary.json), and
[per-query counts](v17eval_a0_alignment_audit/per_query.jsonl) preserve the
read-only result.

### V17-PACKET-REP-A2 relation-support necessary-condition audit

Before training a `(query, answer, sentence)` support encoder, we hand-reviewed
the exact inserted title and sentence for all **21** decisive SLOT-B0 actions
(16 repair, five break), plus four clear-support controls. A `full` judgment
requires the inserted text itself to explicitly assert **every** relation in
the query for the titled answer; `partial` asserts only some conjuncts. This
is a small, post-hoc falsification exercise, not independent support-label
validation or a learned score.

| Cached action outcome | Full relation support | Partial | No explicit full support |
|---|---:|---:|---:|
| 0.90 repair (16) | 2 | 2 | 12 |
| 0.90 break (5) | 0 | 0 | 5 |
| Four clear-support controls | 4 | 0 | 0 |

Three controls improved continuous F1 without crossing the 0.90 threshold;
the fourth lost F1 while staying in the same threshold class. For example, a
sentence explicitly naming the cast member of the titled film can still
reduce Target F1, whereas a one-word section heading or a sentence unrelated
to the requested relation can coincide with a repair. The latter may act
through title cues or interaction with prior context; these observations do
not identify a causal mechanism. They *do* refute the strong claim that
explicit new sentence-level relation support is necessary for most observed
repairs or sufficient for safety in this action pool.

The existing sentence splitter is another concrete action-quality issue:
**227/1,485** R1 candidate fragments have at most three word tokens,
including **3/56** threshold-repair actions and **103/524** threshold-break
actions when all candidates are counted (the earlier 511-break figure applies
only to no-extra-context eligible actions). Abbreviation splitting and bare
headings are visible in the reviewed texts. Fragment length alone is not a
validated safety score, and filtering these actions would require a new
frozen action contract and paired Target comparison.

Decision: `STOP_AUTOMATIC_PROOF_TO_SENTENCE_SUPPORT_LABELING`. The block-level
certificate and gold-answer title cannot supply trusted sentence-level
entailment labels, and these post-hoc 25 judgments cannot train or validate
an encoder. A blinded, independently reviewed support annotation could still
be useful as an auxiliary feature study, but it must first show incremental
query-held-out Target-utility signal beyond title and V8 STAY. Do not switch
all queries to support-per-token scheduling on this evidence. No new Target
calls, training, or sealed-set access occurred. [Script](../../src/evaluation/v17packet_rep_a2_relation_support_falsification.py),
[summary](v17packet_rep_a2_relation_support_falsification/summary.json), and
[inspectable judgments](v17packet_rep_a2_relation_support_falsification/per_action.jsonl)
preserve the audit.

### V17-SEM-F0 query-instantiated entailment final rescue

SEM-F0 performed the terminal query-instantiated entailment rescue on the same
96 design-exposed queries. A deterministic question schema plus the exact
future document title formed the complete hypothesis; Qwen3-14B only judged
that hypothesis against the future packet and S6. Of 192 paired judgments,
181 passed structural/exact-quote checks. The hard-valid policy emitted 61
future candidates and zero S6 candidates, but the frozen regression suite
passed only 12/17 cases. Clear false positives remained for award-subject
attribution, formation-location inference, and explorer-class membership
(IDs 11, 22, and 31); two frozen positive controls were also rejected. A
conservative, non-independent review gives at most 58/61 = 95.1% precision
(Wilson 95% interval 86.5%–98.3%). Under the pre-registered terminal rule this
is `STOP_AUTOMATIC_SEMANTIC_EVIDENCE_CONSTRUCTION`: no prompt revision, larger
Teacher, fresh F0B, or Qwen3-8B Target test is authorized. Query instantiation
fixed many E0Q failures, but did not make the required semantic boundary
reliable enough to protect V8.

### V17-PACKET-H0A source-structure preflight

PACKET-H0A tested whether the V8 action unit itself had split recoverable source
structure. Across all 2,032 lineage-clean training queries, all 24,384 V8
packets contained exactly one title-bound source block; the 12 packets exactly
reconstructed both the stored context and the pre-packetization source blocks
for every query. The upstream representation merely groups two distinct source
blocks per unit and preserves no sentence/table/list metadata beyond the block
text. The short sentence atoms implicated by `15/64 -> 40/64` were created
post hoc by later SEM/R1 diagnostic scripts and are not V8 boundaries. Thus a
macro merge would join distinct documents rather than restore a split
proposition, while sentence splitting would introduce the very fragmentation
being investigated. Decision:
`STOP_PACKET_H0_NO_DISTINCT_STRUCTURAL_REPACKETIZATION`; H0B Target calls are
not authorized. This is a zero-call implementation audit, not evidence that no
conceivable text packetization can help.

### V17-LATENT-L0A legality and accounting audit

LATENT-L0A checked the proposed soft-memory branch against the actual method
and deployment contract before model construction. The current project is
explicitly deterministic, lossless, source-preserving, text-rendered, and
add-only; the canonical Qwen3-8B runner accepts rendered chat text/token IDs
and defines no deployment `inputs_embeds` interface. A single ordered latent
sequence could be prefix-additive and could backpropagate through a frozen
Target in principle, but it would be a learned lossy representation through a
new white-box API. Latent positions also cannot be reported as text tokens
without separate compressor compute, latency, VRAM, and Target-position
accounting. Therefore the conjunctive legality gate gives
`STOP_LATENT_UNDER_CURRENT_LOSSLESS_TEXT_API_CONTRACT`; L0B training and
Target calls are not authorized. Latent memory remains a possible separately
named research project only if the lossless/text-API scope is explicitly
changed.

### V17-STOP-C1 native-confidence two-stage result

The corrected stopping question is valuable: stop at the aggressive checkpoint
when it is already sufficient, otherwise rescue at depth10. On all 1,421 fresh
canonical chains, its zero-call oracle saves 815.51 final-context tokens/query
(25.56%) and 378.60 fully charged Target tokens/query (9.42%) relative to
depth10, with 1.164 calls per attainable request. It raises anchor-success sum
from 6,191 to 6,345 and Complete from 1,123 to 1,142 by preserving early
answers that later roll back. This authorized a native-confidence design
cohort, not a deployable policy.

We generated 1,024 same-call traced states at d6/d7/d9/d10 for 256 naturally
selected train queries, excluding all 32 earlier trace-preflight queries.
Answers, F1, token log-probabilities and costs came from the same traced path.
Four-fold query-grouped OOF compared output-only (B0), native-only (B1), and
combined (B2) linear scores over thresholds frozen before analysis. No system
passed quality, context-capture and real Target-compute gates; formal decision:
`STOP_C1_NATIVE_CONFIDENCE`.

Native-only at threshold 0.95 is diagnostically useful. It gives
0.60--0.95 counts `245/245/243/219/183` and Complete 182, versus depth10
`245/244/242/220/183` and Complete 183, while capturing 33.1% of the oracle
final-context saving. But repeated calls raise Target compute from 4,185.21 to
5,725.41 tokens/query. B0 has no quality-qualified point and B2 does not pass,
so this is partial observability rather than a successful method.

An exact prompt-prefix reuse preflight closes the cheap systems rescue under
the current rendering. Because context is re-rendered in original packet-index
order, early and depth10 prompts share only 20.3% (d6), 24.6% (d7), or 45.9%
(d9) of the depth10 token prefix. Even optimistically reusing every common
prefix token saves only 726.75 tokens/query at the B1 point, leaving it 813.45
Target tokens/query more expensive than direct depth10. Decision:
`STOP_C2_COST_UNDER_CURRENT_PROMPT_RENDERING`. Reveal-order prompt rendering
would be a behavior-changing contract, not a free serving optimization.

### V17-STOP-C2 internal-state preflight

A paired preflight tested a read-only forward hook on the final Qwen3-8B
transformer layer. Sixteen C1 design queries at d6/d7/d9/d10 produced 64
states, each generated once without and once with the hook in the same
Transformers execution path. Token IDs, parsed answers and F1 matched for all
64/64 states; all H1 prompt-terminal, H2 answer-pooled and H3 terminal vectors
were finite. Hook latency was 126.75 versus 119.41 seconds (1.061x), while
peak allocated VRAM increased by only 2,048 bytes. Decision:
`GO_C2_DESIGN_COLLECTION`. This path is not identical to the earlier vLLM C1
path, so all C2 baselines and costs must be regenerated within C2; old labels
cannot be joined to new activations.

The full C2 design collection then generated 1,024 same-path states for 256
queries and stored fixed final-layer H1/H2/H3 vectors. Query-grouped linear
OOF tested output-only, each hidden pooling, and output plus H2 over the frozen
threshold grid. No point from any probe satisfied the one-percentage-point
quality gate. Near-quality high-threshold points also remained more expensive
than direct depth10; compute-oracle capture was negative, far below the binding
85% requirement. Decision: `STOP_INTERNAL_OBSERVABILITY`. Layer sweeps, MLPs,
larger critics and native-logit recollection are not authorized. The next
contract question is C3 prefixable reveal-order prompt serialization.

### V17-STOP-C3-A0 ideal prefix-reuse ceiling

Before changing serialization or calling Target, the frozen C1 native-only
0.95 policy was recomputed under ideal 100% evidence-prefix KV reuse. Effective
compute is 4,034.50 versus 4,185.21 tokens/query for direct depth10, a saving
of 150.71 or **3.60%**. Final context remains 3,055.59 tokens/query. This
passes the 2% gate and yields `GO_C3_PREFIX_TOKEN_PREFLIGHT`, but assumes zero
cache-management overhead and leaves only a narrow deployment margin.

C3-P1 then constructed one component-tokenized serialization with an exact
fixed header, unchanged packet text in V8 reveal order, fixed delimiters and a
separate canonical generation/question suffix. All 256/256 design queries
satisfied token-ID `P6 < P7 < P9 < P10` prefix checks and source preservation.
The corrected C3-P2 paired depth10 test held the chat template, question suffix,
packet set, backend and decoding fixed; only block order changed. On 128 design
queries, canonical versus reveal-order success counts were respectively
`123/122/121/108/88` and `121/120/118/109/83`, while Complete was 99 versus
101. The losses at 0.60, 0.70, 0.80 and especially 0.95 exceed the frozen
one-percentage-point tolerance. Decision: `STOP_C3_ORDER_PERTURBATION`.
Do not run the real-cache microbenchmark, adaptive C3, or relearn an ordering
for cacheability. This is direct evidence of a quality--cacheability conflict.

### V17-STOP-C4-A0 required-stopper feasibility

C4-A0 performed the final zero-call feasibility audit before authorizing an
independent auxiliary judge. Ordinary sufficiency TPR/FPR is not
deployment-aligned: stopping on an early failure that also fails at depth 10
is quality-neutral and cheaper, while stopping on an early success that later
rolls back preserves quality. The corrected decision is to CONTINUE only on
FS states (early failure, depth-10 success). The audit swept rescue recall
`P(CONTINUE|FS)` and unnecessary fallback `P(CONTINUE|SS/SF/FF)` using exact
expected five-level quality, Complete, context and Target compute.

The depth-10 baseline is `245/244/242/220/183`, Complete `183`, at 4,185.21
Target compute tokens/query. Even with zero unnecessary fallback, the first
quality-qualified point needs approximately **96.6% rescue recall**. At a 1%
unnecessary-fallback rate, 90% rescue recall yields expected Complete 175.79
and 95% yields 179.30, both outside the one-percentage-point tolerance. Compute
is not binding; reliable identification of almost every rescue state is. The
formal decision is `STOP_STRONGER_STOPPER_BEFORE_TRAINING`. No auxiliary judge
or further C-series stopping feature is authorized. This closes adaptive
stopping under the current contract and leaves V8 plus a fixed causal schedule
as the supported method pending independent confirmation or an explicitly
revised research contract. [Script](../../src/evaluation/v17stop_c4_a0_required_stopper_feasibility.py),
[summary](v17stop_c4_a0_required_stopper_feasibility/summary.json), and
[report](v17stop_c4_a0_required_stopper_feasibility/REPORT.md) preserve the
audit. No Target call or sealed-set access occurred.

### V18-L0A progressive-latent contract and runtime preflight

After freezing V17, L0A explicitly opened a separately named lossy white-box
contract while retaining frozen Qwen3-8B, query conditioning, nested add-only
prefixes, five fidelity levels and the quality--cost objective. The single
candidate is a once-per-query frozen Qwen3-1.7B encoder plus one trainable
resampler/projection at budgets `[4,8,16,24,32]`.

The zero-call screen projects 943.71 latent Target positions plus 165.84
8B-equivalent compressor parameter-positions, versus 4,185.21 for the five-call
depth10 baseline (screening ratio 0.265). A synthetic Qwen3-8B bf16 runtime
test then found exact equality between token IDs and embedding inputs, exact
equality and 100% top-1 agreement between cached token-ID and embedding
suffixes, finite latent gradients, and no Target parameter gradients. Decision:
`GO_V18_L0B_TINY_MEMORIZATION_PREFLIGHT`. This authorizes only a tiny
plumbing/overfit test. Future monotonicity evaluation requires at least 50%
lower SF at matched early-prefix success to prevent a trivial all-fail result.
[Static summary](v18_l0a_latent_contract_preflight/summary.json),
[runtime result](v18_l0a_latent_contract_preflight/runtime.json), and
[report](v18_l0a_latent_contract_preflight/REPORT.md) preserve the decision.

### V18-L0B-1 32-slot tiny memorization

The single frozen V18 architecture was tested on 32 design-exposed queries for
320 updates. The frozen Qwen3-1.7B encoded query plus original context once;
only the cross-attention resampler/projection trained, and Qwen3-8B had zero
trainable parameters. Teacher answers were supervision and never compressor
inputs. NLL decreased from 4.1591 to 2.1727, a 47.76% reduction versus the 70%
gate. Cached free-generation teacher answer-set F1 was only 0.0313 versus the
0.90 gate. Slots were finite and full-rank (32/32), so the implementation
learned a weak loss-reducing channel but not a behavior-preserving one.
Decision: `STOP_LATENT_CHANNEL`. Nested L0B-2, more steps, more slots,
architecture changes and query-held-out L0C are not authorized. This stops the
narrow V18 rescue, not every conceivable latent-compression architecture.
[Report](v18_l0b1_tiny_memorization/REPORT.md) and
[summary](v18_l0b1_tiny_memorization/summary.json) preserve the result.
