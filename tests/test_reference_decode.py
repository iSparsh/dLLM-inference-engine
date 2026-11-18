import pytest
import torch

from dllm.models import prompt_ids
from dllm.reference_decode import ar_decode, decode


@pytest.mark.slow
@pytest.mark.parametrize(
    "question",
    [
        "Four boxes hold six pencils each. How many pencils?",
        "A car travels 50 miles twice. How far?",
    ],
)
def test_generate(checkpoint: tuple, question: str) -> None:
    model, tok = checkpoint
    ids = prompt_ids(tok, question, model.device)
    expected = model.generate(ids[None], threshold=0.9, max_new_tokens=64)
    assert torch.equal(decode(model, ids, 64, 0.9), expected[0])


@pytest.mark.slow
@pytest.mark.parametrize("checkpoint", ["ar"], indirect=True)
def test_ar(checkpoint: tuple) -> None:
    model, tok = checkpoint
    ids = prompt_ids(tok, "What is two plus three?", model.device)
    expected = model.generate(
        ids[None],
        attention_mask=torch.ones_like(ids[None]),
        max_new_tokens=8,
        do_sample=False,
        repetition_penalty=1.0,
    )
    assert torch.equal(ar_decode(model, ids, 8), expected[0])
