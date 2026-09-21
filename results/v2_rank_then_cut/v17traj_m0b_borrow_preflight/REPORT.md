# TRAJ-M0B: rank-10 atom borrow contract and cost

The frozen, outcome-blind rule chooses the shortest exact rank-10 title-plus-sentence fragment whose Qwen3-8B depth-7 context slack is 1–48 tokens. It preserves all V8 packets, exposes the fragment at depths 7–9, and restores the exact full rank-10 packet at depth 10. No Target calls or outcome labels entered selection.

Across all 1,421 canonical train trajectories, **951** have an eligible fragment. Every reconstructed depth-10 context is byte-identical to V8's. Among eligible queries the fragment adds 26.13 tokens at depth 7 and 26.13 at depth 9. Because depth 7 is read twice under `[6,7,7,9,10]`, this is **78.38 extra cumulative context tokens per action query**; uniform action on all eligible queries would add 52.46 per population query. Later residual restoration prevents duplicate final text but does not refund the earlier 0.70/0.80/0.90 prompt cost. Target output cost cannot be known without calls.

The evidence-set trajectory is add-only, but the source-order renderer reconstructs each prompt and does not guarantee a literal string-prefix trajectory. This distinction must be retained in any paper or deployment claim. A matched Target pilot is justified by M0A's 221 joint 0.80/0.90 failures and the nontrivial eligible pool, but its value must be judged against this measured cumulative cost, not against a single 26-token insertion.

[Protocol](../../../configs/v17traj_m0b_borrow_preflight.json), [code](../../../src/evaluation/v17traj_m0b_borrow_preflight.py), [summary](summary.json), [per-query cost](per_query.jsonl).
