import math

import torch
from torch import Tensor

from dllm.commit_rules import StableGuess, look_ahead
from dllm.config import MASK

VOCABULARY = 4
WINDOW = [0, 2]


def logits_predicting(predictions_from_slot_1: list[tuple[int, float]]) -> Tensor:
    """Logits whose row i - 1 predicts slot i as the given (token, confidence)."""
    logits = torch.zeros(len(predictions_from_slot_1) + 1, VOCABULARY)
    for row, (token, confidence) in enumerate(predictions_from_slot_1):
        odds = confidence * (VOCABULARY - 1) / (1 - confidence)
        logits[row, token] = math.log(odds)
    return logits


def denoise_pass(block: int, pass_index: int) -> dict:
    return {"block": block, "pass": pass_index, "window": WINDOW}


def test_look_ahead_commits_only_confident_masked_slots_after_the_window() -> None:
    tokens = torch.tensor([7, MASK, MASK, MASK, 9])
    logits = logits_predicting([(1, 0.99), (2, 0.99), (3, 0.9), (1, 0.99)])

    look_ahead(0.98)(denoise_pass(0, 0), logits, tokens)

    assert tokens.tolist() == [7, MASK, 2, MASK, 9]


def test_stable_guess_commits_a_window_slot_whose_guess_repeats() -> None:
    rule = StableGuess(0.5)
    tokens = torch.tensor([7, MASK, MASK, MASK])
    first = logits_predicting([(1, 0.6), (2, 0.6), (3, 0.6)])
    second = logits_predicting([(1, 0.6), (3, 0.6), (3, 0.6)])

    rule(denoise_pass(0, 0), first, tokens)
    assert tokens.tolist() == [7, MASK, MASK, MASK]

    rule(denoise_pass(0, 1), second, tokens)
    assert tokens.tolist() == [7, 1, MASK, MASK]


def test_stable_guess_needs_confidence_above_its_threshold() -> None:
    rule = StableGuess(0.5)
    tokens = torch.tensor([7, MASK])
    logits = logits_predicting([(1, 0.4)])

    rule(denoise_pass(0, 0), logits, tokens)
    rule(denoise_pass(0, 1), logits, tokens)

    assert tokens.tolist() == [7, MASK]


def test_stable_guess_does_not_compare_across_blocks() -> None:
    rule = StableGuess(0.5)
    tokens = torch.tensor([7, MASK])
    logits = logits_predicting([(1, 0.6)])

    rule(denoise_pass(0, 4), logits, tokens)
    rule(denoise_pass(1, 5), logits, tokens)

    assert tokens.tolist() == [7, MASK]
