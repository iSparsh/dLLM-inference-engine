"""PyTorch op contracts from docs/INTERFACES.md §3. All indices are block-local except q_start."""

import torch
import torch.nn.functional as F
from torch import Tensor


def add_rmsnorm(
    x: Tensor | None, residual: Tensor, weight: Tensor, eps: float
) -> Tensor:
    """Update the bf16 residual in place and return the normalized activation."""
    if x is not None:
        residual.add_(x)
    f = residual.float()
    return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)).to(
        residual.dtype
    ) * weight


def rope_kv_write(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    q_start: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> None:
    """Rotate Q in place and write rotated K and unchanged V at absolute positions."""
    pos = q_start + torch.arange(len(q), device=q.device)
    c, s = cos[pos, None], sin[pos, None]

    def rotate(t: Tensor) -> Tensor:
        a, b = t.chunk(2, -1)
        return t * c + torch.cat((-b, a), -1) * s

    q.copy_(rotate(q))
    k_cache.index_copy_(1, pos, rotate(k).transpose(0, 1))
    v_cache.index_copy_(1, pos, v.transpose(0, 1))


def block_attention(
    q: Tensor, k_cache: Tensor, v_cache: Tensor, q_start: Tensor, block_size: int
) -> Tensor:
    """Use stock bf16 SDPA with an absolute-position mask and fixed cache capacity."""
    pos = q_start + torch.arange(len(q), device=q.device)
    keys = torch.arange(k_cache.shape[1], device=q.device)
    mask = (keys < q_start + len(q)) & (
        keys // block_size <= pos[:, None] // block_size
    )
    k, v = (t.repeat_interleave(6, 0)[None].contiguous() for t in (k_cache, v_cache))
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None].contiguous(), k, v, attn_mask=mask
    )
    return out[0].transpose(0, 1).contiguous()


def silu_mul(gate_up: Tensor) -> Tensor:
    """SiLU rounds to bf16 before multiplication."""
    gate, up = gate_up.chunk(2, -1)
    return F.silu(gate) * up


def select_commit(
    logits: Tensor, tokens: Tensor, window: Tensor, tau: float, mask_id: int
) -> Tensor:
    """Mutate eligible tokens and return three device int32 counters/flags."""
    i = torch.arange(len(tokens), device=tokens.device)
    active = (i >= window[0]) & (i < window[1])
    eligible = active & (tokens == mask_id)
    confidence, guess = logits.softmax(-1).max(-1)
    rows = (i - 1).clamp_min(0)
    confidence = confidence[rows].masked_fill(~eligible, -torch.inf)
    selected = eligible & ((confidence > tau) | (i == confidence.argmax()))
    tokens.copy_(torch.where(selected, guess[rows], tokens))
    masked = tokens == mask_id
    return torch.stack((selected.sum(), ~(masked & active).any(), ~masked.any())).int()


def example_axpy(x: Tensor, y: Tensor, alpha: float) -> Tensor:
    """Compute alpha*x + y for contiguous float32 vectors."""
    return torch.add(y, x, alpha=alpha)
