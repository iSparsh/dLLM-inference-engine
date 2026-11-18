"""The one GSM8K prompt every run uses, for both models (docs/specs/accuracy.md)."""

import torch
from transformers import PreTrainedTokenizerBase

INSTRUCTION = "Think step by step, then give the final numeric answer after '####'."


def gsm8k_messages(question: str) -> list[dict]:
    """Chat messages for one GSM8K problem: the question, a newline, the instruction."""
    return [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}]


def gsm8k_prompt_ids(
    tokenizer: PreTrainedTokenizerBase, question: str, device: torch.device | str
) -> torch.Tensor:
    """1-D prompt token ids, wrapped in the model's own chat template."""
    return tokenizer.apply_chat_template(
        gsm8k_messages(question), add_generation_prompt=True, return_tensors="pt"
    ).to(device)[0]
