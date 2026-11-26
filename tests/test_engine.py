"""Driver parity: the static-cache, graph-captured engine against the reference decoders.

Run after switching ops to your CUDA backend, e.g.
    DLLM_OURS=silu_mul,add_rmsnorm uv run pytest -q -rs tests/test_engine.py
DLLM_OURS takes a comma-separated op list, or "all".
"""

import json
import os
from pathlib import Path

import pytest
import torch

from dllm import ops
from dllm.engine import Engine
from dllm.prompts import gsm8k_prompt_ids
from dllm.reference_decode import decode

QUESTIONS = [
    json.loads(line)["question"]
    for line in Path("eval/gsm8k_100.jsonl").read_text().splitlines()[:2]
]
# Cover every prefill branch of INTERFACES.md §3.6: no prefill (N < 32), the exactly-32
# quirk, a prefill whose last row seeds the first block (N = 64) and two real prompts.
PROMPTS = ["gsm0", "gsm1", "short20", "exact32", "aligned64"]


@pytest.fixture(scope="module", autouse=True)
def backends():
    chosen = os.environ.get("DLLM_OURS", "")
    names = list(ops.available()) if chosen == "all" else chosen.split(",")
    saved = ops.available()
    for name in filter(None, names):
        ops.set_backend(name, "ours")
    yield
    for name, backend in saved.items():
        ops.set_backend(name, backend)


def prompt(tok, name: str, device) -> torch.Tensor:
    ids = [gsm8k_prompt_ids(tok, q, device) for q in QUESTIONS]
    sizes = {"short20": 20, "exact32": 32, "aligned64": 64}
    return ids[int(name[-1])] if name.startswith("gsm") else ids[0][: sizes[name]]


@pytest.fixture(scope="module")
def dllm_engine(checkpoint):
    model, tok = checkpoint
    return model, tok, Engine(model, "dllm", capacity=1024)


@pytest.mark.slow
@pytest.mark.parametrize("tau", [1.0, 0.9, 0.7])
@pytest.mark.parametrize("name", PROMPTS)
def test_dllm_parity(dllm_engine, name: str, tau: float) -> None:
    model, tok, engine = dllm_engine
    ids = prompt(tok, name, model.device)
    passes = 0

    def count(*_):
        nonlocal passes
        passes += 1

    expected = decode(model, ids, 128, tau, observe=count)
    got = engine.generate(ids, 128, tau)
    assert torch.equal(got, expected), f"{name} tau={tau}: tokens differ"
    assert engine.stats.passes == passes
    assert engine.stats.gen_tokens == len(expected) - len(ids)


@pytest.mark.slow
@pytest.mark.parametrize("checkpoint", ["ar"], indirect=True)
def test_ar_parity(checkpoint) -> None:
    """Greedy tokens equal HF generate, up to a one-ulp tie in HF's own logits.

    HF's cached M=1 step runs unmasked SDPA over the exact length; the stock
    block_attention op runs masked SDPA, a different kernel. Where HF's top two logits
    are one bf16 step apart, the two can pick different tokens (HF's own full
    recompute and cached decode disagree there too). Any other difference fails.
    """
    model, tok = checkpoint
    engine = Engine(model, "ar", capacity=1024)
    for question in QUESTIONS + ["What is two plus three?"]:
        ids = gsm8k_prompt_ids(tok, question, model.device)
        expected = model.generate(
            ids[None],
            attention_mask=torch.ones_like(ids[None]),
            max_new_tokens=128,
            do_sample=False,
            repetition_penalty=1.0,
        )[0]
        got = engine.ar_generate(ids, 128)
        if torch.equal(got, expected):
            assert engine.stats.gen_tokens == len(expected) - len(ids)
            continue
        n = min(len(got), len(expected))
        diff = (got[:n] != expected[:n]).nonzero()
        assert len(diff), "one output is a prefix of the other"
        at = int(diff[0, 0])
        with torch.inference_mode():
            logits = model(expected[:at][None]).logits[0, -1]
        top = logits.max()
        ulp = 2.0 ** (torch.floor(torch.log2(top.float().abs())) - 7)
        for token in (got[at], expected[at]):
            assert top - logits[token] <= ulp, f"not a near-tie at {at}: {question!r}"
