# V8 one-run pretraining design review

Date: 2026-09-17

## Verdict

The original V8 direction is justified, but its first draft was not safe to
train as written. A mask-only oracle, per-step near-optimal slack, a frozen-head
trial followed by optional LoRA, and post-training DAgger could each force a
second formal training run or create train/inference mismatch.

The reviewed protocol below removes those failure modes. It is the recommended
single formal V8 attempt. No performance result is guaranteed, but it directly
supervises the earliest decisions that fail under V7 and has much more oracle
state coverage than V4--V7's static edge labels.

## Findings that change the design

1. **History is required by the oracle.** Among 105,602 masks with multiple
   reachable prior-anchor states, 43,930 (41.60%) have different optimal
   actions across those histories. The exact teacher state is `(S,h)`.
2. **The true `h` cannot be an inference feature.** It comes from Target
   fidelity. It is retained as an auxiliary prediction target, while the model
   receives only the question, packets, selected mask, ordered action history,
   and token features.
3. **Per-step slack is unsafe.** A 0.005 allowance at each of 12 decisions can
   compound beyond the intended global rate tolerance. Formal action labels
   use zero slack and include every exact tie.
4. **The first encoder draft leaked source position.** `Packet 0`, `Packet 1`,
   and so on are removed from independent encoder inputs. Set attention has no
   positional embedding. Ordered history enters only through the GRU.
5. **Two-stage frozen-head then LoRA training risks a rerun.** The one formal
   run jointly trains a fresh Qwen3-4B LoRA and the new head, with separate
   learning rates.
6. **Post-training DAgger is avoidable.** Expert, nearest-deviation recovery,
   and random recovery histories can all be generated before training from the
   existing exact lattices.
7. **Surrogate validation loss is an unreliable selector.** V4--V7 validation
   loss and the actual trajectory gate did not align consistently. Fixed
   checkpoints must be selected by the consumed-development trajectory metrics
   under a preregistered lexicographic rule.

## Evidence that the new target addresses V7's failures

Every one of the 23 V7 fidelity-0.90 failures takes an action outside the exact
optimal action set. The first divergence occurs at mean reveal depth 2.78 and
median depth 3; three failures diverge at the first action and three more at
the second. Five first divergences immediately reduce the maximum number of
anchors that any continuation can reach.

Only 10/29 nearest pair-repair relations are also direct next-action
corrections at the prefix before the intruder. This is expected: a closest
final-order swap is not necessarily the optimal local decision. It also shows
why converting those 29 repairs into another pairwise loss would remain
misaligned. The sequential oracle labels the earlier causal divergence in
23/23 failures.

## Frozen data construction for one training run

For each train3163 example, construct all labels before model training:

- four rollouts that sample among exact optimal action ties;
- twelve rollouts, each forcing the nearest nonoptimal action at one reveal
  depth and then following the exact recovery policy;
- four uniformly random rollouts with exact recovery labels at every visited
  state;
- deduplication by complete ordered packet history;
- exact set-valued action target and reached-anchor auxiliary target at every
  retained history.

On development300 this construction yields 60,504 unique histories, mean
201.68 and median 203 per example. It covers 96.94 distinct `(mask,h)` states
per example on average. Exact action sets contain 1.73 actions on average, and
95.90% of states retain a continuation that reaches every active anchor. The
same construction is expected to produce roughly 0.64 million train histories.

Losses must be balanced first by example and then by source class (expert,
single-deviation, random), so the larger recovery pool cannot overwhelm the
root/on-policy decisions. Random states receive the lowest weight. No model
rollout or new Target inference is needed to build this pool.

## Reviewed model

The backbone receives one logical encode stage containing a question-only
sequence and 12 `(question, packet)` sequences per example. Packet text
contains no packet index. To avoid padding every short sequence to the longest
packet, the 13 sequences are sorted into four fixed length buckets, requiring
four short backbone microcalls. Their final hidden states are restored to
packet identity and projected to 512 dimensions. A learned source-coordinate
embedding is then added explicitly, preserving useful source order without
making packet content embeddings depend on preceding packets.

The head contains:

- two permutation-equivariant self-attention layers over 12 packet vectors;
- a selected-set pool and remaining-set pool;
- a GRU initialized from the question vector and updated by selected packets;
- normalized candidate-token and cumulative-token features;
- a set-valued action softmax over remaining packets;
- a masked ordinal auxiliary head for the highest active anchor reached.

The auxiliary head is trained from exact history labels but its prediction,
never the true label, is available to the action scorer. This prevents Target
leakage while making the partially observed progress variable explicit.

Primary decoding uses a beam of eight histories and returns the complete order
with maximum cumulative action log-probability. The beam branches only the
small head over cached packet vectors, so its cost is negligible relative to
encoding. Beam width eight is frozen before training; greedy versus beam is not
a post-result model-selection choice.

Use a fresh Qwen3-4B NF4 LoRA rather than the V7 adapter. The V7 adapter learned
the causal joint-input geometry and the systematic relation errors that V8 is
intended to replace. A lower LoRA learning rate limits overfitting; the new
head uses a higher learning rate. Fix one seed (`20260912`) and one optimizer
budget before the run.

The reviewed fixed defaults are LoRA rank 16, alpha 32, dropout 0.05 over the
same attention and MLP projection modules used by V7; model dimension 512; two
set-attention layers with eight heads and dropout 0.10; and a 512-dimensional
GRU. Use AdamW with LoRA learning rate `2e-5`, head learning rate `2e-4`, weight
decay `0.01`, gradient clipping `1.0`, 50 warmup steps, and cosine decay. The
progress auxiliary weight is `0.2`. These values must be frozen rather than
tuned after observing the gate.

## Training and selection rule

Use one 750-step formal run with fixed checkpoints at steps 250, 500, and 750.
Each optimizer batch encodes an example once and samples 32 of its precomputed
histories: 12 expert, 12 single-deviation, and eight random histories. Sampling
with replacement is allowed within a source class when needed. The planned
micro-batch is two examples with four-step gradient accumulation, preserving
the effective batch of eight used by V7. A predeclared memory-only fallback is
micro-batch one with accumulation eight; it changes no example exposure or
optimizer step. Hundreds of history labels reuse one backbone representation.

Checkpoint selection on consumed development300 is preregistered as:

1. if any checkpoint passes all three development gates, restrict selection to
   those checkpoints;
2. otherwise maximize the number of development gates passed;
3. maximize fidelity-0.90 successes;
4. maximize complete trajectories;
5. minimize regret;
6. minimize set-valued action loss.

After selecting one checkpoint, apply the unchanged gate: fidelity-0.90 at
least 282/300, complete trajectories at least 0.90, and regret at most 0.03.
Only a conjunctive pass permits a newly frozen confirmation role.

## Efficiency

The reviewed 13-sequence input has a 1.306x unpadded token ratio to the current
joint input on train3163. Naive padding would increase actual linear work to
2.209x. Four length buckets reduce padded linear work to 1.525x and the padded
squared-length attention proxy to 0.205x. Development ratios are 1.516x and
0.204x. This is one logical encoding stage with four short microcalls, followed
by twelve cheap head decisions. A one-step GPU memory/throughput smoke remains
required before freezing the batch size; it is a systems check, not a
scientific training run.

## Remaining pretraining checks

- Materialize and hash the train3163 history pool.
- Implement the no-ID packet encoder, progress head, source-balanced loss,
  beam-8 evaluator, and gate-aligned checkpoint selector.
- Unit-test that true `h` never enters model inputs.
- Confirm permutation equivariance of packet encodings and set attention.
- Run one optimizer-step memory smoke on a longest-input batch.
- Freeze implementation, data, model, optimizer, seed, checkpoints, and gate
  hashes before starting the only formal run.

## Artifacts

- `results/v2_rank_then_cut/v8_sequential_preflight_development300.json`
- `results/v2_rank_then_cut/v8_one_run_supervision_preflight_development300.json`
- `results/v2_rank_then_cut/v8_failed_repair_action_coverage.json`
- `results/v2_rank_then_cut/v8_v7_failure_first_action_deviation.json`
- `results/v2_rank_then_cut/v8_packet_encoding_cost_preflight.json`
- `src/search/sequential_trajectory_dp.py`
- `src/evaluation/v8_one_run_supervision_preflight.py`
