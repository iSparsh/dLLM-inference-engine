"""M1: compare CUDA K1 against the reference on real decoder logits.

Requires the locally downloaded dLLM checkpoint. Run with:
    uv run pytest tests/test_select_live.py -v -m slow
"""
import json
from pathlib import Path

import pytest
import torch

from dllm import ops
from dllm.config import MASK
from dllm.models import prompt_ids
from dllm.ops.reference import select_commit as reference_select
from dllm.reference_decode import decode


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tau", [0.7, 0.9, 1.0])
def test_real_generation_selection(checkpoint, tau):
    model, tokenizer = checkpoint
    dataset = [json.loads(line) for line in Path("eval/gsm8k_100.jsonl").read_text().splitlines()]
    compared = 0

    @torch.inference_mode()
    def observe(row, tokens, logits, cache):
        nonlocal compared
        if logits is None or row["window"] is None:
            return
        bounds = torch.tensor(row["window"], dtype=torch.int32, device=tokens.device)
        expected_tokens = tokens.clone()
        actual_tokens = tokens.clone()
        expected = reference_select(logits, expected_tokens, bounds, tau, MASK)
        ops.set_backend("select_commit", "ours")
        try:
            actual = ops.select_commit(logits, actual_tokens, bounds, tau, MASK)
        finally:
            ops.set_backend("select_commit", "torch")
        assert torch.equal(actual_tokens, expected_tokens), f"token mismatch at {row}"
        assert torch.equal(actual, expected), f"stats mismatch at {row}"
        compared += 1

    for sample in dataset[:5]:
        ids = prompt_ids(tokenizer, sample["question"], model.device)
        decode(model, ids, max_new_tokens=32, tau=tau, observe=observe)
    assert compared > 0, "No denoise steps were observed"
