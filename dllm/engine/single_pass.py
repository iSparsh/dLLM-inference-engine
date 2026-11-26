"""Static-cache, batch-one example. The caller owns prefix validity and advancement."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

from .. import ops


class SinglePass:
    @torch.inference_mode()
    def __init__(
        self, model: PreTrainedModel, capacity: int = 4096, block_size: int = 32
    ) -> None:
        self.model, self.block_size = model, block_size
        base = model.model
        weight = base.embed_tokens.weight
        self.k = weight.new_zeros(len(base.layers), 2, capacity, 128)
        self.v = torch.zeros_like(self.k)
        self.start = torch.zeros(1, device=weight.device, dtype=torch.int32)
        positions = torch.arange(capacity, device=weight.device)[None]
        c, s = base.rotary_emb(weight, positions)
        self.cos, self.sin = c[0], s[0]
        self.gate_up = [
            torch.cat((l.mlp.gate_proj.weight, l.mlp.up_proj.weight))
            for l in base.layers
        ]

    @torch.inference_mode()
    def prefix(self, cache: object | None) -> None:
        """Copy an HF prefix into static storage. Call outside capture."""
        length = cache.get_seq_length() if cache is not None else 0
        if length > self.k.shape[2]:
            raise ValueError(
                "Increase capacity to hold the prefix and the query block."
            )
        if length:
            for i, (k, v) in enumerate(cache):
                self.k[i, :, :length].copy_(k[0])
                self.v[i, :, :length].copy_(v[0])
        self.start.fill_(length)

    @torch.inference_mode()
    def forward(
        self,
        ids: Tensor,
        length: int | None = None,
        last_only: bool = False,
        packed: bool = True,
    ) -> Tensor:
        """Return [M,V] logits, or [1,V] with last_only. Ensure start+M <= length.

        length limits attention to the first cache slots (default: all of them), so a
        caller can bucket the stock attention op's cost by context length. Keys past
        start+M are masked either way, so the bucket does not change the result.

        packed=False runs HF's separate gate and up GEMMs. cuBLAS can choose a different
        algorithm for the packed [M,1536]x[1536,17920] shape than for two N=8960 GEMMs,
        which changes bf16 rounding: on the RTX 4060 the results are bitwise equal for
        M <= 32 but not for M = 64 or 96, so prefill uses the unpacked form for parity.
        """
        base = self.model.model
        k_cache, v_cache = self.k[:, :, :length], self.v[:, :, :length]
        residual = base.embed_tokens(ids)
        x = None
        for i, layer in enumerate(base.layers):
            norm = ops.add_rmsnorm(
                x=x,
                residual=residual,
                weight=layer.input_layernorm.weight,
                eps=self.model.config.rms_norm_eps,
            )
            a = layer.self_attn
            q, k, v = [
                proj(norm).reshape(len(ids), -1, 128)
                for proj in (a.q_proj, a.k_proj, a.v_proj)
            ]
            ops.rope_kv_write(
                q=q,
                k=k,
                v=v,
                k_cache=k_cache[i],
                v_cache=v_cache[i],
                q_start=self.start,
                cos=self.cos,
                sin=self.sin,
            )
            att = ops.block_attention(
                q=q,
                k_cache=k_cache[i],
                v_cache=v_cache[i],
                q_start=self.start,
                block_size=self.block_size,
            )
            norm = ops.add_rmsnorm(
                x=a.o_proj(att.flatten(1)),
                residual=residual,
                weight=layer.post_attention_layernorm.weight,
                eps=self.model.config.rms_norm_eps,
            )
            if packed:
                gate_up = F.linear(norm, self.gate_up[i])
            else:
                gate_up = torch.cat((layer.mlp.gate_proj(norm), layer.mlp.up_proj(norm)), -1)
            hidden = ops.silu_mul(gate_up=gate_up)
            x = layer.mlp.down_proj(hidden)
        norm = ops.add_rmsnorm(
            x, residual, base.norm.weight, self.model.config.rms_norm_eps
        )
        return self.model.lm_head(norm[-1:] if last_only else norm)

    @torch.inference_mode()
    def capture(self, ids: Tensor) -> Callable[[Tensor], Tensor]:
        """Capture one shape; returned logits alias storage overwritten on replay."""
        static = ids.clone()
        for _ in range(3):
            self.forward(static)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.forward(static)

        def replay(tokens: Tensor) -> Tensor:
            if tokens.shape != static.shape:
                raise ValueError("Recapture for the new input shape.")
            static.copy_(tokens)
            graph.replay()
            return out

        return replay
