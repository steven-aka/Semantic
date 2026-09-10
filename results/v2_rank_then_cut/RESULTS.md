# V2 Rank-then-Cut status

Status: **protocol frozen; exact-oracle construction in progress; no training**

Date: 2026-09-10

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

## Current data and execution

The target-blind candidate pool contains 5000 certified QAMPARI-train examples
and 60,000 lossless packets. Its frozen hashes are:

```text
examples = cce862a91fc2562ac6cf8c456ce663c4bcf0e4c24b66fb67e39f8ad43345be04
annotations = 71c50309631b488b252024cc114ff72c1369457166ec85ab2b1cd1c93c0f17b3
packet tree = d0df345cdf13191a64ea4affc697db6df5d6c73d9bc96a88bc90d6dd31b33364
protocol config = 2376fd274372ac6c595e1cfeb30a13c36dc23d72c302fcec308f89563e7aeb79
pre-training amendment 1 = 00e875df1306d049074251da77ab263c7d231e3736be97f52c8a51da927f7a21
```

Exact Target inference is split into six deterministic shards in
`results/v2_rank_then_cut/candidates5000_exact`. Five shards (0, 1, 2, 3, 5)
are assigned to separate A6000 GPUs with a 0.45 vLLM memory cap. Shard 4 is
queued under `scripts/24_wait_launch_v2_shard4.sh`: it launches only after a
project-unused GPU remains above 32GB free for two checks 30 seconds apart.
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

## Authoritative inputs

- `configs/v2_rank_then_cut_hypothesis.json`
- `results/v1_atomic/RANK_CUT_DIAGNOSTIC.md`
- `results/v1_atomic/fresh30_rank_cut_decomposition_summary.json`
- `data/units/qampari_rank_v2_candidates5000.jsonl`
- `data/units/qampari_rank_v2_candidates5000_annotations.jsonl`
- `data/packets_qampari_rank_v2_candidates5000/`
