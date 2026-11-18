"""Transparent batch-one drivers for HF forward. Token IDs and return value are 1-D."""

from collections.abc import Callable

import torch
from torch import Tensor
from transformers import PreTrainedModel

from .config import BLOCK, EOS, MASK, SUB_BLOCK
from .ops.reference import select_commit
from .trace import TraceHook, predictions

PassHook = Callable[[dict, Tensor, Tensor | None, object], None]


@torch.inference_mode()
def decode(
    model: PreTrainedModel,
    ids: Tensor,
    max_new_tokens: int = 64,
    tau: float = 0.9,
    trace: TraceHook | None = None,
    observe: PassHook | None = None,
) -> Tensor:
    """Match Fast-dLLM generate, including its exactly-32 and budget edge cases."""
    if ids.ndim != 1 or not ids.numel():
        raise ValueError("Pass one nonempty prompt as a 1-D token tensor.")
    n = len(ids)
    cache = None
    x = ids.clone()
    step = 0

    def forward(block_ids: Tensor, commit: bool) -> Tensor:
        nonlocal cache
        out = model(
            block_ids[None],
            past_key_values=cache,
            use_cache=True,
            update_past_key_values=commit,
        )
        if commit:
            cache = out.past_key_values
        return out.logits[0]

    if n > BLOCK:
        logits = forward(x[: n // BLOCK * BLOCK], True)
        if n % BLOCK == 0:
            x = torch.cat((x, logits[-1].argmax().view(1)))
    for block in range(max_new_tokens // BLOCK):
        if EOS in x[n:]:
            break
        start = len(x)
        x = torch.cat((x, x.new_full((BLOCK - start % BLOCK,), MASK)))
        tokens = x[-BLOCK:]
        q_start = cache.get_seq_length() if cache is not None else 0

        def stopped() -> bool:
            stops = (x[start:] == EOS).nonzero()
            return bool(len(stops)) and MASK not in x[start : start + int(stops[0, 0])]

        t0_guess = None
        for lo in range(0, BLOCK, SUB_BLOCK):
            hi = lo + SUB_BLOCK
            window = torch.tensor([lo, hi], dtype=torch.int32, device=x.device)
            while MASK in tokens[lo:hi] and not stopped():
                row = {
                    "block": block,
                    "pass": step,
                    "q_start": q_start,
                    "kind": "denoise",
                    "window": [lo, hi],
                }
                logits = forward(tokens, False)
                if observe:
                    observe(row, tokens, logits, cache)
                before = tokens.clone() if trace else None
                select_commit(logits, tokens, window, tau, MASK)
                if trace:
                    committed, guesses = predictions(logits, before, tokens, MASK)
                    trace({**row, "committed": committed, "guesses": guesses})
                    t0_guess = int(logits[-1].argmax())
                step += 1
        if stopped():
            break
        row = {
            "block": block,
            "pass": step,
            "q_start": q_start,
            "kind": "commit",
            "window": None,
        }
        # Observers need the prefix *before* HF mutates the DynamicCache.
        if observe:
            observe(row, tokens, None, cache)
        seed = forward(tokens, True)[-1].argmax().view(1)
        x = torch.cat((x, seed))
        if trace:
            trace(
                {
                    **row,
                    "committed": [],
                    "guesses": [],
                    "t0_guess": t0_guess,
                    "t0_actual": int(seed),
                }
            )
        step += 1
    stops = (x[n:] == EOS).nonzero()
    return x[: n + int(stops[0, 0]) + 1] if len(stops) else x


@torch.inference_mode()
def ar_decode(
    model: PreTrainedModel,
    ids: Tensor,
    max_new_tokens: int = 64,
    trace: TraceHook | None = None,
) -> Tensor:
    """Greedy HF forward with DynamicCache. Include the first generated EOS."""
    x, query, cache = ids.clone(), ids, None
    eos = model.generation_config.eos_token_id
    stops = {eos} if isinstance(eos, int) else set(eos or [])
    for step in range(max_new_tokens):
        out = model(
            query[None], past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        cache = out.past_key_values
        logits = out.logits[0, -1]
        query = logits.argmax().view(1)
        if trace:
            trace(
                {
                    "block": step,
                    "pass": step,
                    "kind": "ar",
                    "window": None,
                    "q_start": len(x) - (len(ids) if step == 0 else 1),
                    "committed": [
                        [0, int(query), float(logits.float().softmax(-1)[query])]
                    ],
                    "guesses": [],
                }
            )
        x = torch.cat((x, query))
        if int(query) in stops:
            break
    return x
