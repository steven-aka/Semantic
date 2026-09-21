# TRAJ-M0C: paired rank-10-to-depth-7 borrow pilot

We froze one M0B action per query and selected SHA-first 128 eligible train queries from 244 remaining after exclusion of R1, OBS-A1 and FRAG-B0 query sets. No action outcome entered cohort or candidate selection. The Qwen3-8B run made 768 fresh calls: per query, shared V8 depths 6 and 10, plus paired V8/action depths 7 and 9. All five attainable anchors follow **one evidence-set-add-only trajectory**; the depth-10 full context is exactly restored. No model was trained and no sealed data was read.

| Uniform policy | 0.60 | 0.70 | 0.80 | 0.90 | 0.95 | Complete | Mean cumulative context tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| V8 | 116 | 108 | 84 | 93 | 89 | 68 | 2291.14 |
| Borrow one rank-10 atom at depth 7 | 116 | 112 | 100 | 101 | 89 | 79 | 2367.52 |

Complete changes by **19 repairs and eight breaks**, net +11/128; a paired query bootstrap 95% interval for the net count is **+1 to +21**. Per-anchor repairs/breaks are 0/0 at 0.60, 9/5 at 0.70, 21/5 at 0.80, 17/9 at 0.90 and 0/0 at 0.95. The uniform action costs +76.38 cumulative context tokens/query and +71.88 measured Target prompt-plus-output tokens/query. It improves quality but does not strictly dominate V8 on cost.

The preregistered hindsight opportunity gate required at least eight no-break Complete repairs **and** at most +10 mean cumulative context tokens/query if only those repairs were selected. There are 19 such repairs, but their hindsight-selected cost is **+11.37**, so the formal decision is **`STOP_M0C_COSTED_ORACLE_GATE`**. This threshold is operational, not a mathematical boundary; missing it by 1.37 tokens does not negate the observed multi-anchor headroom, but it does not authorize controller training under the frozen protocol.

Exploratory comparison against earlier P0 results on these same 128 queries found no monotone fixed V8 schedule with equal or lower cumulative context cost that also matched all five action success counts and Complete. Earlier P0 outputs differ slightly from the fresh baseline (for example Complete 66 versus fresh 68), so this is a research hint, not a causal Pareto claim. Post-result slack slices likewise suggest that some repairs occur with shorter atoms, but selecting a new cap from this cohort would be tuning on exposed outcomes. A different cap would require its own frozen protocol and untouched cohort.

For transparency, descriptive caps on the **same already-tested action** give 7, 12, 14 and 19 hindsight-safe Complete repairs at depth-7 slack caps 16, 24, 32 and 48, respectively. Their hindsight mean cumulative costs are +2.13, +4.71, +6.05 and +11.37 tokens/query. Uniform deployment at those caps still causes 2, 5, 7 and 8 Complete breaks. These are post-result slices and do not replace the frozen 48-token gate.

The outcome is more informative than either “ordering is hopeless” or “train the selector now.” The single trajectory can improve 0.80 and 0.90 together, addressing the failure locus found in M0A. Yet 8/128 Complete breaks, repeated early-token cost, and the earlier failures of cheap deployment signals remain unsolved. The 0.95 endpoint is unchanged by this exact-recovery action; it cannot alone meet a five-anchor target whose 0.95 ceiling is insufficient. Continue only with a separately frozen cost-reduction/action-safety hypothesis; do not relax M0C's gate or open sealed sets.

[Frozen protocol](../../../configs/v17traj_m0c_borrow_pilot.json), [runner](../../../src/evaluation/v17traj_m0c_borrow_pilot.py), [summary](summary.json), [post-run diagnostics](analysis.json), and [paired outcomes](per_query.jsonl) preserve the result.
