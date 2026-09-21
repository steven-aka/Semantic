# FRONTIER-R1: RECOMP NQ Abstractive Screen-64

This zero-shot cross-domain baseline uses the official public
`fangyuan/nq_abstractive_compressor` (T5-large) with frozen Qwen3-8B. The input
uses the official `Question / Document / Summary` format; generation uses beam
4, no-repeat 3, source limit 1024 and output limit 512. No training occurred.

The fresh full-context control has mean F1 0.9495 and 53/64 successes at 0.90.
RECOMP produces 25.70 Qwen3 tokens on average (3.46% keep rate), but mean F1 is
0.1888, 0.90 success is 0/64, and 10/64 summaries are empty. There are 0
repairs and 53 breaks.

Decision: `STOP_RECOMP_ABSTRACTIVE_AFTER_SCREEN64`. This proves that the NQ
checkpoint's extreme compactness does not transfer faithfully to QAMPARI; it
does not prove a QAMPARI-trained compressor is impossible or authorize it.

The retained non-training observation is that abstractive text can reach the
desired compression regime and is consumable by the frozen Target, but
task/domain faithfulness is binding. Every output and log is preserved. The
run adds 64 Target calls, reuses the fresh control, and reads no sealed set.
