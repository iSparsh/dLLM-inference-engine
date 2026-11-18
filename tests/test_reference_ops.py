"""Compare composed reference contracts with the real HF layer, cache and head."""

import pytest
import torch

from dllm import ops
from dllm.config import MASK
from dllm.engine import SinglePass
from dllm.models import prompt_ids


@pytest.mark.parametrize("checkpoint", ["dllm"], indirect=True)
@torch.inference_mode()
def test_full_layer(checkpoint: tuple) -> None:
    model, tok = checkpoint
    ids = prompt_ids(tok, "A shop sells fifteen pears per day. " * 20, model.device)[
        :96
    ]
    prefix = model(
        ids[None], use_cache=True, update_past_key_values=True
    ).past_key_values
    block = ids[:32].clone()
    block[3:] = MASK
    engine = SinglePass(model, 160, 32)
    engine.prefix(prefix)
    actual = engine.forward(block)
    ref = model(
        block[None],
        past_key_values=prefix,
        use_cache=True,
        update_past_key_values=True,
    )
    torch.testing.assert_close(actual, ref.logits[0], rtol=0, atol=0)
    for i, (k, v) in enumerate(ref.past_key_values):
        torch.testing.assert_close(
            engine.k[i, :, : len(ids) + len(block)], k[0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            engine.v[i, :, : len(ids) + len(block)], v[0], rtol=0, atol=0
        )
    replay = engine.capture(block)
    assert torch.equal(replay(block), actual)
    window = torch.tensor([0, 8], device=ids.device, dtype=torch.int32)
    tokens = block.clone()
    guesses, probs = model.sample_with_top_p(
        torch.cat((actual[:1], actual[:-1]))[:8], temperature=0
    )
    scores = (
        probs.gather(-1, guesses[:, None])
        .flatten()
        .masked_fill(tokens[:8] != MASK, -torch.inf)
    )
    keep = scores > 0.9
    keep[scores.argmax()] = True
    expected = block.clone()
    expected[:8][keep] = guesses[keep]
    stats = ops.select_commit(actual, tokens, window, 0.9, MASK)
    assert torch.equal(tokens, expected)
    assert stats.tolist() == [
        int(keep.sum()),
        int(MASK not in expected[:8]),
        int(MASK not in expected),
    ]


@pytest.mark.parametrize("checkpoint", ["ar"], indirect=True)
@torch.inference_mode()
def test_ar_attention_against_hf(checkpoint: tuple) -> None:
    """Use HF's actual first-layer Q and updated K/V; HF dispatches unmasked AR to Flash."""
    model, tok = checkpoint
    ids = prompt_ids(tok, "A store sells six boxes each day. " * 20, model.device)[:96]
    prefix = model(ids[None], use_cache=True).past_key_values
    a = model.model.layers[0].self_attn
    capture = {}

    def save_q(module: object, inputs: tuple, out: torch.Tensor) -> None:
        capture["q"] = out[0].view(1, 12, 128)

    def save_att(module: object, inputs: tuple) -> None:
        capture["out"] = inputs[0][0].view(1, 12, 128)

    hooks = [
        a.q_proj.register_forward_hook(save_q),
        a.o_proj.register_forward_pre_hook(save_att),
    ]
    try:
        model(ids[-1:].view(1, 1), past_key_values=prefix, use_cache=True)
    finally:
        for hook in hooks:
            hook.remove()
    pos = torch.tensor([len(ids)], device=ids.device, dtype=torch.int32)
    cos, sin = model.model.rotary_emb(capture["q"], pos.long()[None])
    x = capture["q"]
    left, right = x.chunk(2, -1)
    q = x * cos[0] + torch.cat((-right, left), -1) * sin[0]
    k, v = prefix[0]
    output = ops.block_attention(q, k[0], v[0], pos, 1)
    torch.testing.assert_close(output, capture["out"], atol=0.02, rtol=0)
