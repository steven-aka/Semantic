# V7 failure diagnosis and executable V8 direction

Date: 2026-09-17

## Decision

Stop changing the V4--V7 one-shot relational scorer. The next experiment
should replace its representation and supervision with a frozen-backbone,
history-aware sequential packet policy. This is a change in scorer
factorization and inductive bias; the deployed output remains one total packet
order with nested prefix cutoffs.

This conclusion does **not** claim that no static total order exists. Exact
nested-chain search proves that good orders exist. It says that the tested
one-shot pair-logit model does not infer the few decisive relations reliably.

## What failed

V7 selected seed 20260912 at optimizer step 500. On the already consumed
development300 it reached 277/300 at fidelity 0.90, 273/300 complete
trajectories, and 0.02578 feasible normalized regret. The rate target passed;
the 282/300 fidelity gate failed. Twenty-two of V6's 23 failures persisted
after training coverage grew from 2,000 to all 3,163 attainable examples.

The alternatives already tested do not explain the remaining gap:

- Decoder mismatch: zero of 5,313 raw-correct safety edges were sacrificed by
  the exact maximum-weight total-order decoder.
- Seed variance: V6's 29 repair relations were wrong under all three seeds,
  and its three-seed ensemble stayed at 277/300.
- Backbone capacity: Qwen3-4B did not improve the 0.90 result over the tested
  1.7B revisions.
- Training coverage: adding 1,163 examples, 21,517 safety edges, and 30,596
  rate edges left the 0.90 result at 277/300.
- Rate objective: V6 and V7 pass regret while still failing fidelity.

The V7 decoder diagnostic gives the direct localization. All 29 nearest
repair relations for the 23 failed examples have the wrong raw sign before
decoding, with mean absolute margin 2.69. An oracle-label projection through
the unchanged decoder reaches 298/300 at fidelity 0.90, so the available
order/decoder space has ample headroom.

Current supervision is also incomplete for the required local repair: 18/29
relations support it, 7/29 are unlabeled, and 4/29 explicitly oppose it. A new
loss over the same static pair labels cannot directly correct 11/29 of the
observed repair directions.

## Why these examples fail

The failures are concentrated in longer, constraint-dense composition cases.

| Diagnostic | V7 0.90 failures | V7 0.90 successes |
|---|---:|---:|
| Examples | 23 | 277 |
| Mean stable safety edges | 33.30 | 17.80 |
| Mean wrong stable edges | 7.52 | 0.76 |
| Complete stable-edge separation | 4.35% | 80.87% |
| Mean packet tokens | 983.35 | 825.96 |

`wikitables_composition` fails on 15/98 examples (15.31%), compared with
6/190 (3.16%) for `wikidata_simple`, a 4.85x failure-rate ratio. Twenty-one of
the 23 failed examples need only one boundary replacement. This combination
means that many relations must be simultaneously correct, while one misplaced
packet can sharply change Target behavior.

The current representation makes this generalization harder. It concatenates
all packets into one causal Qwen sequence and mean-pools the hidden states in
each packet span. Earlier packet representations cannot attend to later
packets, and early tokens within a packet cannot attend to the packet's own
suffix. A single final-token global vector is shared by every pair. The model
therefore has no explicit representation of the selected prefix and must
compress all redundancy, complementarity, and distraction effects into fixed
pair logits.

This conflicts with the exact lattice. The prior audit found strict marginal
sign flips in 22--25% of harmful packet-anchor groups. The new exact V8
preflight finds an additional history effect: of 105,602 masks that can be
reached with different prior highest-anchor states, 43,930 (41.60%) require a
different exact next-action set depending on that history. This occurs in
292/300 development examples.

## Correct sequential oracle

The simple proposal `V*(S)` is insufficient because fidelity is non-monotone.
Two action sequences can end at the same selected set while one crossed a
fidelity anchor earlier and the other did not. Rate cost also depends on when
an anchor was first crossed.

Use the offline oracle state

```text
(S, h)
S = selected packet mask
h = highest fidelity anchor reached anywhere in the exact reveal history
```

For action `i`, set `S' = S union {i}` and
`h' = max(h, anchors_satisfied(F(S')))`. The dynamic program compares suffixes
lexicographically:

1. maximize the final number of reached anchors;
2. minimize the sum of token counts at first anchor crossings.

If one action crosses several new anchors, the token count of `S'` is charged
once for each newly crossed anchor. The implemented DP matches the existing
globally optimal nested-chain objective on 300/300 development examples.

For training, retain every action with the best reachable-anchor count and
cost within the frozen normalized slack. The loss is

```text
-log sum(P(i | action history, evidence) for i in A*(S,h))
```

The exact `h` is used only to create labels from existing lattices. It must not
be supplied to the deployed model because that would require Target inference.

## Reviewed one-run V8 architecture

Use a packet-order-invariant encoder rather than reusing the current causal
span means:

1. Encode 12 order-invariant `(question, packet)` sequences plus one
   question-only sequence in one batch with Qwen3-4B. Packet IDs are excluded
   from encoder text, and each sequence's final hidden state is retained.
2. Project the 12 packet vectors, add a learned source-coordinate embedding,
   and run a small two-layer set-attention block. The coordinate travels with
   its packet under batching; it is not induced by causal input placement.
3. Maintain both a selected-set pool and a small GRU state updated by the
   sequence of chosen packet vectors.
4. Score every remaining packet from its vector, selected-set pool, remaining
   pool, question vector, recurrent history, packet token count, and cumulative
   token count. An auxiliary ordinal head predicts the highest anchor reached
   from the same observable history representation.
5. Decode with a fixed beam of eight action histories and return the order with
   the highest cumulative policy log-probability. Only the small head branches;
   Qwen is not rerun during the 12 decisions.

The recurrent state is needed to distinguish different paths to the same set;
the set pool is needed to represent redundancy and complementarity. The true
history anchor is an auxiliary target only: it is never an inference input.
Packet encoding remains deployable and Target-blind.

The earlier two-stage suggestion (frozen encoder first, LoRA only after a
failure) is rejected because it creates the exact repeat-training risk this
review is intended to avoid. The formal run should train a fresh low-rank
adapter and the new head jointly from the start, with a lower learning rate for
LoRA than for the randomly initialized head. There is one seed and one formal
training run; checkpointing within that run is not a second experiment.

A tokenizer preflight over all 3,163 training examples measured 899.75 tokens
on average for the current joint input and 1,154.01 tokens across the 13
logical inputs, a 1.306x unpadded token ratio. A naive padded batch would raise
this to 2.209x. Four fixed length buckets reduce the padded linear ratio to
1.525x and the squared-length attention proxy to 0.205x. On development300 the
four-bucket ratios are 1.516x and 0.204x. This requires four short backbone
microcalls in one logical encode stage; actual latency still needs a GPU
benchmark, but it remains far from twelve full-context forwards.

## Single-training experiment sequence

1. Freeze the V8 data builder, exact DP, architecture, optimizer budget, and
   unchanged development gate before training.
2. Before model training, deterministically generate four tie-diverse exact
   oracle rollouts, one nearest-nonoptimal deviation rollout at each of the 12
   reveal depths, and four random recovery rollouts per example. Deduplicate by
   ordered history and label every visited state with the exact DP action set.
   This replaces train--DAgger--retrain with one fixed offline pool.
3. Use zero action slack. Exact ties remain set-valued, but a per-step 0.005
   tolerance is removed because repeated tolerant actions can accumulate more
   than the intended global rate slack.
4. Jointly train fresh Qwen3-4B LoRA, the set encoder, GRU, action head, and
   auxiliary history-progress head in one run. Balance expert, deviation, and
   random-history losses per example so recovery states do not swamp the
   on-policy objective.
5. Save checkpoints at fixed optimizer steps. Evaluate the preregistered
   beam-8 order and select one checkpoint by the
   preregistered consumed-development ordering: first satisfy trajectory and
   regret gates, then maximize fidelity-0.90 successes, then minimize regret
   and action loss. Do not select only by surrogate validation loss, which was
   poorly aligned in V4--V7.
6. Evaluate the selected checkpoint on consumed development300 with the unchanged
   requirements: at least 282/300 at fidelity 0.90, at least 90% complete
   trajectories, and regret at most 0.03.
7. Open a new target-blind confirmation role only after a conjunctive pass.
   Calibration300 and final-test300 remain locked.

The offline-pool preflight produces 60,504 unique ordered histories for
development300, or 201.68 per example, with 96.94 distinct `(mask, history
anchor)` states per example. Exact next-action sets average 1.73 actions and
95.90% of these states can still reach every active anchor. Scaling the same
fixed construction to train3163 should yield roughly 0.64 million supervised
histories without any new Target call.

Do not add a cutoff head, more seeds, a larger compressor, WikiTables
oversampling, or another pairwise loss before this test. Those changes either
repeat a falsified explanation or mix the conditional-selection hypothesis
with unrelated changes.

## Reproducible artifacts

- `results/v2_rank_then_cut/v7_decoder_alignment_diagnostic/summary.json`
- `results/v2_rank_then_cut/v7_decoder_alignment_diagnostic/repair_edges.jsonl`
- `results/v2_rank_then_cut/v8_sequential_preflight_development300.json`
- `results/v2_rank_then_cut/v8_packet_encoding_cost_preflight.json`
- `results/v2_rank_then_cut/v8_one_run_supervision_preflight_development300.json`
- `results/v2_rank_then_cut/v8_failed_repair_action_coverage.json`
- `results/v2_rank_then_cut/v8_v7_failure_first_action_deviation.json`
- `src/search/sequential_trajectory_dp.py`
- `src/evaluation/v8_sequential_preflight.py`
- `src/evaluation/v8_packet_encoding_cost_preflight.py`
- `src/evaluation/v8_one_run_supervision_preflight.py`
- `tests/test_sequential_trajectory_dp.py`
