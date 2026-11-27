"""Experimental commit rules for P4 (docs/specs/accuracy.md, Problem 5).

A rule runs after the standard commit of a denoise pass and may commit more of the
block's masked slots in place, so it changes the generated text.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from .config import MASK

CommitRule = Callable[[dict, Tensor, Tensor], None]


def slot_predictions(logits: Tensor) -> tuple[Tensor, Tensor]:
    """Confidence and token for every slot; logits row i - 1 predicts slot i."""
    confidence, token = logits.softmax(-1).max(-1)
    predicting_row = (torch.arange(len(logits), device=logits.device) - 1).clamp_min(0)
    return confidence[predicting_row], token[predicting_row]


def commit_masked(tokens: Tensor, chosen: Tensor, predicted_token: Tensor) -> None:
    tokens.copy_(torch.where(chosen & (tokens == MASK), predicted_token, tokens))


def look_ahead(threshold: float) -> CommitRule:
    """Also commit slots after the current window whose confidence exceeds the threshold."""

    def rule(denoise_pass: dict, logits: Tensor, tokens: Tensor) -> None:
        confidence, predicted_token = slot_predictions(logits)
        slots = torch.arange(len(tokens), device=tokens.device)
        after_window = slots >= denoise_pass["window"][1]
        commit_masked(tokens, after_window & (confidence > threshold), predicted_token)

    return rule


class StableGuess:
    """Also commit window slots whose guess repeats the previous pass's, above a confidence."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self.previous_pass = None
        self.previous_token = None

    def __call__(self, denoise_pass: dict, logits: Tensor, tokens: Tensor) -> None:
        confidence, predicted_token = slot_predictions(logits)
        block, pass_index = denoise_pass["block"], denoise_pass["pass"]
        if self.previous_pass == (block, pass_index - 1):
            slots = torch.arange(len(tokens), device=tokens.device)
            window_start, window_end = denoise_pass["window"]
            in_window = (slots >= window_start) & (slots < window_end)
            repeated = predicted_token == self.previous_token
            commit_masked(
                tokens,
                in_window & repeated & (confidence > self.threshold),
                predicted_token,
            )
        self.previous_pass = block, pass_index
        self.previous_token = predicted_token
