from __future__ import annotations

from typing import Sequence

from src.target.qwen_runner import TargetRunner


QAMPARI_SYSTEM_PROMPT = (
    "Use only the supplied context. Return every distinct answer to the list question "
    "that is explicitly supported by the context. Do not add unsupported answers. "
    "Put the answers between <answer> and </answer>, separated only by #."
)


def build_qampari_prompt(question: str, context: str) -> str:
    return (
        "Use only the context below to answer the list question.\n"
        "Return every distinct supported answer, separated only by #, inside "
        "<answer> and </answer>. Do not explain.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )


class QampariTargetRunner(TargetRunner):
    def _chat_prompts(self, questions: Sequence[str], contexts: Sequence[str]) -> list[str]:
        prompts = []
        for question, context in zip(questions, contexts):
            messages = [
                {"role": "system", "content": QAMPARI_SYSTEM_PROMPT},
                {"role": "user", "content": build_qampari_prompt(question, context)},
            ]
            if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
            else:
                prompt = build_qampari_prompt(question, context)
            prompts.append(prompt)
        return prompts
