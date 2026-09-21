# SEM-B1 ideal representation positive control

Fresh Qwen3-8B runs compared V8, the raw paired atoms, and the manually audited
short semantic fact on the same 64 train-side queries.  Only the 40 SEM-C0
ELIGIBLE queries received an intervention; all other queries stayed at V8.
All arms used the fixed `[6,7,7,9,10]` schedule and additive evidence from
depth 7 onward.

V8 / Raw / Semantic Complete counts were 38 / 40 / 40.  Raw and Semantic each
produced eight safe Complete repairs and six Complete breaks.  Semantic reduced
queries with any anchor break from Raw's 10 to 8 and reduced mean cumulative
context overhead over all 64 queries from +222.22 to +47.88 tokens.  It passed
the preregistered ideal-representation gate, while the six Complete breaks show
that the fixed policy is not deployable.

The follow-up extractive control used a manually audited exact source span no
longer than its paired semantic fact.  On the 40 active queries, Extractive vs
Semantic Complete was 24 vs 25; the paired bootstrap 95% interval for the
difference was [-6, 8].  Semantic improved 0.80 by four cases but lost one case
at both 0.70 and 0.95, used 8.9 more cumulative tokens per active query, and
had one more Complete break.  Neither representation Pareto-dominated the
other.  The defensible conclusion is that compact query-relevant evidence has
value, but these data do not isolate a semantic-rewriting advantage over
shortening.
