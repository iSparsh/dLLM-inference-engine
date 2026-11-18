"""Synthetic op inputs shared by contract tests and microbenchmarks."""

import torch

from .config import MASK, VOCAB


def random_case(
    name: str,
    m: int,
    q_start: int,
    capacity: int,
    block_size: int,
    device: str = "cuda",
) -> dict:
    if q_start + m > capacity:
        raise ValueError("Increase capacity to at least q_start + M.")

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=torch.bfloat16)

    pos = torch.tensor([q_start], device=device, dtype=torch.int32)
    if name == "add_rmsnorm":
        return dict(
            x=rand(m, 1536), residual=rand(m, 1536), weight=rand(1536), eps=1e-6
        )
    if name == "silu_mul":
        return dict(gate_up=rand(m, 17920))
    if name == "select_commit":
        if m != 32:
            raise ValueError("select_commit requires M=32.")
        return dict(
            logits=rand(32, VOCAB),
            tokens=torch.full((32,), MASK, device=device),
            window=pos.new_tensor([8, 16]),
            tau=0.9,
            mask_id=MASK,
        )
    if name == "example_axpy":
        return dict(x=rand(m * 1536).float(), y=rand(m * 1536).float(), alpha=0.5)
    common = dict(
        q=rand(m, 12, 128),
        k_cache=rand(2, capacity, 128),
        v_cache=rand(2, capacity, 128),
        q_start=pos,
    )
    if name == "block_attention":
        return dict(**common, block_size=block_size)
    if name == "rope_kv_write":
        angles = rand(capacity, 128).float()
        return dict(
            **common,
            k=rand(m, 2, 128),
            v=rand(m, 2, 128),
            cos=angles.cos().bfloat16(),
            sin=angles.sin().bfloat16(),
        )
    raise ValueError(f"Unknown op {name}")
