# TRAJ-M0A: frozen single-trajectory failure-locus map

This is a zero-Target-call audit of the 1,421 train-side V8 Qwen3-8B prefix chains at the already frozen five-level schedule `[6,7,7,9,10]`. Each query follows **one** trajectory. Unattainable 0.95 is masked, never treated as failure. The recomputed Complete count is 775/1,421, matching the earlier CANON-P0 result.

Of 646 non-Complete queries, 0.80 fails on 423 and 0.90 on 371; both fail on **221**. Single-anchor failures are 127 at 0.80, 106 at 0.90, 53 at 0.95, 19 at 0.60 and zero at 0.70. These categories do not cover every failure: many queries fail several anchors. The dominant paired bottleneck is therefore 0.80+0.90, not isolated 0.95.

The proposed `TRAJ-M0` direction is worthwhile as an **action-space audit before learning**, but its three primitives should not be launched together. To affect both the 0.80 depth-7 read and the 0.90 depth-9 read while preserving the existing V8 packets, a rank-10 atom promoted to depth 7 is the most direct single-action hypothesis. Moving an atom from rank 8 or 9 may improve depth 7 but, after its original packet is fully reached, need not change depth 9. This is an inference from the frozen schedule and source-order rendering, not evidence of Target benefit.

Three contract and cost qualifications matter:

1. **Cumulative cost is not refunded.** A rank-10 atom visible at depth 7 remains visible at the depth-7 0.70/0.80 reads and depth-9 0.90 read. Restoring only the residual at depth 10 avoids duplicated *final-context* text, but does not recover tokens already paid at earlier fidelity calls. Real prompt and generated tokens must also be reported.
2. **Set-add-only differs from literal prompt-prefix add-only.** Existing source-order `render` reconstructs the whole prompt at every step; restoring residual can change text order even while evidence sets grow. A future protocol must state which monotonicity contract it claims and verify exact canonical depth-10 recovery.
3. **The late-interference attribution in the proposed guidance is too strong.** The 0.95 drop from depth 10 to 12 is caused somewhere in the additional rank-11/12 material or its interaction with context; it does not establish that partially revealing rank 10 at depth 10 helps. Earlier R0 paired replay also found two 0.80 breaks among nine outcome-selected repairs when a rank-10 sentence was revealed already at depth 7. Those are small, selected data but a concrete safety warning.

**Decision:** `GO_TRAJ_M0B_CONTRACT_AND_COST_PREFLIGHT`, not immediate Target oracle or controller training. First freeze one rank-10-to-depth-7 atomic action with complete residual recovery, count its real cumulative context and Target cost on an outcome-blind train-side frame, and verify unchanged depth-10 full context. Then preregister a small matched Qwen3-8B pilot measuring the *same trajectory* at depths 7, 9 and 10 against V8, all anchor repairs/breaks, Complete and cost. Only if its single-action oracle has material, safe multi-anchor headroom should a second complementary primitive or learned policy be considered. The 0.95 partial-reveal idea remains a separate hypothesis, with no causal evidence from the depth10→12 rollback alone.

[Script](../../../src/evaluation/v17traj_m0a_failure_locus.py), [summary](summary.json), and [per-query vectors](per_query.jsonl) make this audit reproducible.
