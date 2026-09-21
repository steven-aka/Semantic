# QAMPARI-domain abstractive RECOMP baseline

This is a RECOMP-style in-domain reproduction, not an official QAMPARI checkpoint.
Qwen3-14B produces query-focused evidence summaries from training contexts; a frozen
filter retains summaries with at least 0.60 answer-atom recall and at most 256 T5
tokens. T5-large is then fine-tuned as the compressor. Qwen3-8B is absent from
compressor training and remains frozen during the final baseline evaluation.

The 32-query pilot retained 30/32 summaries. The frozen full-data filter retained
1906/2032 training summaries and 523/550 validation summaries. No sealed evaluation
split was read.
