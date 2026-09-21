# NATIVE-CODE-A0 consumption positive control

This 32-query design-exposed positive control tested only whether frozen
Qwen3-8B reliably consumes short oracle answer-item codes. Five nested budgets
were evaluated. Every correct code was paired with an item-count-matched code
from the next query, for 320 fresh Target calls. No compressor was trained.

The pre-registered gates were mean code-recovery F1 at least 0.95, nested
retention at least 0.95, and a correct-minus-shuffled gold-F1 gap at least
0.50.

## Result

- code-recovery F1: **0.8867** (fail)
- nested prior-code retention: **0.8427** (fail)
- correct-minus-shuffled gold-F1: **0.7367** (pass)
- direct parsing of correct code, gold F1: **0.8375**
- Qwen3-8B output from correct code, gold F1: **0.7367**
- Target value over direct code: **-0.1008**

The zero shuffled-code score confirms that Qwen3-8B is responding to the code
rather than answering from query priors. However, it omits or alters enough
provided items that recovery and progressive retention fail substantially.
Retention worsens with larger prefixes, reaching 0.795 at 0.95. Passing the
already-answer-like representation through Qwen3-8B loses about 10.1 F1 points
relative to direct parsing.

Decision: `STOP_NATIVE_CODE_TARGET_COMPATIBILITY`.

Do not run A1 automatic-code memorization. Even perfect oracle code prediction
would feed a consumer that is less accurate than the code itself, so training
the compressor would mainly create another QA solver and add Target-induced
errors. This closes the proposed discrete answer-code route under the frozen
Target. It does not claim that all textual summaries are impossible; it shows
that the only zero-call construction with large compression headroom is both
answer-equivalent and harmed by the Target.

No sealed set was read.

