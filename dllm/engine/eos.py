"""Graph-safe end-of-text check for the driver (docs/INTERFACES.md §3.6, "Stop")."""

import torch
from torch import Tensor


def eos_ready(tokens: Tensor, gen_start: Tensor, mask_id: int, eos_ids: Tensor) -> Tensor:
    """True when a generated stop token is committed with no MASK before it.

    tokens: int64[N], the whole sequence buffer (prompt + generated so far).
    gen_start: int32/int64[1] on the device, index of the first generated token.
    eos_ids: int64[k] stop ids on the device. Returns a bool[1] device tensor; no host sync.
    """
    pos = torch.arange(tokens.numel(), device=tokens.device)
    generated = pos >= gen_start
    is_stop = generated & torch.isin(tokens, eos_ids)
    big = torch.full_like(pos, tokens.numel())
    first_stop = torch.where(is_stop, pos, big).min()
    first_mask = torch.where(generated & (tokens == mask_id), pos, big).min()
    return ((first_stop < big[0]) & (first_stop < first_mask)).view(1)
