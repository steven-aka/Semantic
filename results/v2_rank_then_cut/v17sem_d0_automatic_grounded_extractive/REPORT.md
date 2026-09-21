# SEM-D0 automatic grounded extractive audit

SEM-D0 used 96 previously untouched SHA-ordered train queries and reserved the
remaining 11 under the prior M0B eligibility contract.  Before feasibility
labels were read, it froze an exact-source policy that joined a novel document
title with a query-overlapping evidence window, emitted at most two spans, and
required the final serialization to fit 16 Qwen3-8B tokens.

AI-assisted primary annotation plus an adversarial source/length review found
29/96 queries with at least one valid compact extractive representation.  This
is a useful action-space ceiling, but not independent-human IAA evidence.

The frozen automatic policy emitted on 79/96 queries.  It recovered 23/29
eligible cases (79.3% recall), but only 23/79 emissions were semantically valid
(29.1% precision).  Exact quotation guaranteed provenance, yet did not
guarantee that the emitted text closed every relation in the question.  Common
false positives contained the requested person/place but omitted the cast,
birth, location, league, or other linking predicate.

The preregistered 95% emission-precision gate fails decisively, so no SEM-D1
Target calls are authorized.  The next hypothesis must add an explicit
relation-closure/ABSTAIN mechanism and evaluate it on a new cohort.  Threshold
tuning on these 96 exposed items would not be a valid confirmation.
