# TRAJ-M2B: fresh paired five-anchor promote–delay audit

This prospective **train-side research audit** tested the M2A budget-neutral action under frozen Qwen3-8B. The 128 query IDs were chosen by SHA order from 198 M2A-legal train queries after excluding the recent M0C, M1, OBS-A1, and FRAG-B0 cohorts. Earlier R1 action outcomes may have been studied on some queries, but these M2 action outcomes were not used for selection. Per query, the first at most three distinct legal actions were frozen by smallest absolute cumulative cost change, then delayed-sentence index. `action0` is the deployment-visible fixed rule; the best of up to three is a forbidden-outcome oracle. The 958 Target calls were freshly paired with V8; no 581-query, internal, development, or confirmation results were opened.

| Anchor | Fresh V8 success | Fixed action success | Fixed repairs / breaks |
| --- | ---: | ---: | ---: |
| 0.60 | 107/128 | 107/128 | 0 / 0 |
| 0.70 | 114/128 | 107/128 | 4 / 11 |
| 0.80 | 71/128 | 73/128 | 21 / 19 |
| 0.90 | 84/128 | 77/128 | 17 / 24 |
| 0.95 | 73/128 | 73/128 | 0 / 0 |
| Complete | 52/128 | 44/128 | 11 / 19 |

The fixed rule saves **36.35 cumulative context tokens/query**, but Complete falls by eight. Its measured Target prompt-plus-output token count falls by 18.48/query; output length variation means this is not identical to context savings. There are 39 queries with a break at any originally successful anchor. A paired 10,000-draw query bootstrap (seed 20260921) gives an approximate 95% interval of **−19 to +2 queries** for the fixed Complete difference.

The **hindsight no-anchor-break oracle** finds 15 Complete repairs across five baseline failure masks: only 0.80 (5), only 0.90 (4), both 0.80/0.90 (3), 0.70/0.80 (2), and 0.70/0.80/0.90 (1). Eleven of these 15 are already repaired by `action0`; additional action choices add four opportunities. Choosing a safe action would save an average 44 cumulative context tokens on those 15 repaired queries, but this selection uses forbidden Target outcomes. The prespecified opportunity gate (at least 12 safe Complete repairs across two failure masks) therefore yields **`GO_M2_LEARNABILITY_DESIGN_ONLY`**. It does **not** establish a deployed quality–token Pareto gain or authorize immediate training on this small, design-exposed cohort.

Interpretation: the zero-cost action space has meaningful multi-anchor oracle headroom, but the fixed rule's 19 Complete breaks exceed its 11 repairs. The unresolved problem is identifying safe exchanges across queries, particularly protecting 0.70 and 0.90. A future model would need query-grouped, independent validation against V8 with STAY available; ordinary action accuracy or this hindsight oracle cannot stand in for that result. M2 also cannot address 0.95 because the final context is exactly V8's.

The pilot includes only 15 positive oracle queries, and its sentence atoms
inherit the known splitter defects. Those constraints make immediate
high-capacity selector training especially prone to optimistic fitting.

[Protocol](../../../configs/v17traj_m2b_promote_delay_pilot.json), [code](../../../src/evaluation/v17traj_m2b_promote_delay_pilot.py), [manifest](manifest.json), [summary](summary.json), [per-query outcomes](per_query.jsonl), [raw Target outputs](per_call.jsonl).
