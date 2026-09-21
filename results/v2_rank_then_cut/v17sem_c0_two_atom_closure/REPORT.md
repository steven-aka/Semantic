# SEM-C0 paired adjacent-atom closure audit

The fixed rule paired each SEM-B0R rank-10 atom with its next atom, or with the
previous atom when it was last.  The rule was fixed without Target outcomes.
The same AI-assisted primary plus adversarial review found 40/64 ELIGIBLE,
versus 15/64 for one atom.  Abstentions comprised 13 incomplete relations,
nine irrelevant pairs, one ambiguous grounding case, and one relation that
could not be expressed unambiguously within 16 Qwen3-8B tokens.

This +25 paired gain localizes a major upstream problem: the prior atomic
split often separated the entity/title from the predicate or the two halves of
a required relation.  It passes the route threshold for an ideal semantic
representation positive control.  It is not human-IAA evidence.
