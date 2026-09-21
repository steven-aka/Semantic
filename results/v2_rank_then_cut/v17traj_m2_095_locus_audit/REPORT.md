# 0.95 rollback location audit

This read-only audit uses the existing CANON-P0 fresh V8 Qwen3-8B prefix cache. Among 1,131 train queries with attainable 0.95, 179 succeeded at depth 10 and failed at depth 12. For 73, the first observed failure was depth 10→11; for 106, depth 11 remained successful and failure occurred at 11→12. The split uses the canonical F1 threshold with `1e-6` tolerance.

This locates the failure transition but does not prove that the new packet alone caused the rollback, nor that either late packet is safe to remove. No new Target calls or sealed-set access occurred. [Script](../../../src/evaluation/v17traj_m2_095_locus_audit.py), [summary](summary.json), [query-level transitions](rollback_queries.jsonl).
