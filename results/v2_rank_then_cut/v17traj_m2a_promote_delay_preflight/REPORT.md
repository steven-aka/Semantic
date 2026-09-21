# TRAJ-M2A: promote–delay cost and legality preflight

The proposed M2 action is a **rank-10 exact title-plus-sentence fragment promoted at depth 7**, while one sentence of the rank-7 packet is withheld until the exact originals are restored at depth 10. It uses the M0B outcome-blind promoted fragment and changes no Target model parameters. Legal actions must keep at least one sentence in the rank-7 packet, use distinct sentence text, recover exact S6 and depth-10 contexts, and have nonpositive **actual rendered-token deltas at both depth 7 and depth 9**. This tests costs directly; a comparison of raw sentence lengths would not account for titles, separators, or tokenizer effects.

On 1,421 canonical train trajectories, 951 had an M0B fragment. Of these, 676 had a nonduplicate rank-7 sentence-removal candidate. **508 queries had at least one budget-neutral exchange**, totaling **941 legal actions**; 78 queries had at least three. Actual depth-7 and depth-9 token deltas range from −86 to 0, and five-anchor cumulative deltas from −258 to 0. No Target calls or sealed-set access occurred.

Two corrections to the proposed interpretation matter. The frozen read schedule is `[6,7,7,9,10]`, so the depth-7 edit also changes **0.70**, not just 0.80/0.90. Exact depth-10 recovery leaves **0.95 unchanged by construction**; any 0.95 repair needs a different action. These trajectories are add-only at the evidence-set level, but source-order rendering does not guarantee literal prompt-prefix appending. This audit proves action/cost feasibility only, not quality or learnability.

The inherited sentence splitter has known abbreviation and short-heading
artifacts. M2A preserves it to isolate this action contract; a future
deployment design must separately validate atom quality.

[Protocol](../../../configs/v17traj_m2a_promote_delay_preflight.json), [code](../../../src/evaluation/v17traj_m2a_promote_delay_preflight.py), [summary](summary.json), [legal actions](legal_actions.jsonl).
