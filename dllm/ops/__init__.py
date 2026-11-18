"""Per-op backend selection. available() returns a snapshot of configured backends."""

from collections.abc import Callable

import torch
from torch import Tensor

from . import cuda, reference

_backends = dict.fromkeys(
    (
        "add_rmsnorm",
        "rope_kv_write",
        "block_attention",
        "silu_mul",
        "select_commit",
        "example_axpy",
    ),
    "torch",
)


def available() -> dict[str, str]:
    return _backends.copy()


def set_backend(name: str, backend: str) -> None:
    if name not in _backends or backend not in ("torch", "ours"):
        raise ValueError(f"Use one of {list(_backends)} and backend 'torch' or 'ours'.")
    _backends[name] = backend


def set_all(backend: str) -> None:
    for name in _backends:
        set_backend(name, backend)


def _get(name: str) -> Callable:
    if _backends[name] == "torch":
        return getattr(reference, name)
    cuda.load()
    return getattr(torch.ops.dllm, name)


def add_rmsnorm(
    x: Tensor | None, residual: Tensor, weight: Tensor, eps: float
) -> Tensor:
    return _get("add_rmsnorm")(x, residual, weight, eps)


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
    return _get("rope_kv_write")(q, k, v, k_cache, v_cache, q_start, cos, sin)


def block_attention(
    q: Tensor, k_cache: Tensor, v_cache: Tensor, q_start: Tensor, block_size: int
) -> Tensor:
    return _get("block_attention")(q, k_cache, v_cache, q_start, block_size)


def silu_mul(gate_up: Tensor) -> Tensor:
    return _get("silu_mul")(gate_up)


def select_commit(
    logits: Tensor, tokens: Tensor, window: Tensor, tau: float, mask_id: int
) -> Tensor:
    return _get("select_commit")(logits, tokens, window, tau, mask_id)


def example_axpy(x: Tensor, y: Tensor, alpha: float) -> Tensor:
    return _get("example_axpy")(x, y, alpha)
