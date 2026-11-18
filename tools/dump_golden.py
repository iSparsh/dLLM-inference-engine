"""Save fixed dLLM denoise/commit and AR step fixtures from HF projection hooks."""

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from transformers import PreTrainedModel

from dllm.config import MASK
from dllm.golden import save
from dllm.models import load, prompt_ids
from dllm.ops.reference import rope_kv_write
from dllm.reference_decode import decode
from dllm.results import git_sha


def layer_cases(
    model: PreTrainedModel, tokens: Tensor, cache: object, root: Path, meta: dict
) -> None:
    """Run HF once. Copy the old prefix before AR's DynamicCache appends the new row."""
    values = {}
    handles = []

    def collect(key: tuple, take_input: bool = False):
        def hook(module: object, inputs: tuple, output: Tensor) -> None:
            values[key] = (inputs[0] if take_input else output)[0]

        return hook

    base = model.model
    layers = (0, 13)
    start = meta["q_start"]
    capacity = start + len(tokens) + 32
    positions = torch.arange(capacity, device=tokens.device)
    cos, sin = (
        t[0] for t in base.rotary_emb(base.embed_tokens.weight, positions[None])
    )
    prefix = {}
    try:
        for i in layers:
            layer = base.layers[i]
            prefix[i] = [cos.new_zeros(2, capacity, 128) for _ in range(2)]
            if start:
                for dest, src in zip(prefix[i], cache[i]):
                    dest[:, :start].copy_(src[0])
            for name, module in (
                ("residual", layer.input_layernorm),
                ("x", layer.self_attn.o_proj),
                ("q", layer.self_attn.q_proj),
                ("k", layer.self_attn.k_proj),
                ("v", layer.self_attn.v_proj),
                ("gate", layer.mlp.gate_proj),
                ("up", layer.mlp.up_proj),
            ):
                handles.append(
                    module.register_forward_hook(collect((i, name), name == "residual"))
                )
        model(tokens[None], past_key_values=cache, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    for i in layers:

        def case(op: str, **args: object) -> None:
            folder = root / f"{meta['model']}-{meta['kind']}-{i}-{op}"
            save(folder, {**meta, "layer": i, "op": op}, args)

        layer = base.layers[i]
        case(
            "add_rmsnorm",
            x=values[i, "x"],
            residual=values[i, "residual"],
            weight=layer.post_attention_layernorm.weight,
            eps=model.config.rms_norm_eps,
        )
        case("silu_mul", gate_up=torch.cat((values[i, "gate"], values[i, "up"]), -1))
        q, k, v = [values[i, n].reshape(len(tokens), -1, 128) for n in ("q", "k", "v")]
        kc, vc = prefix[i]
        pos = tokens.new_tensor([start], dtype=torch.int32)
        args = dict(
            q=q, k=k, v=v, k_cache=kc, v_cache=vc, q_start=pos, cos=cos, sin=sin
        )
        case("rope_kv_write", **args)
        rope_kv_write(**args)
        case(
            "block_attention",
            q=q,
            k_cache=kc,
            v_cache=vc,
            q_start=pos,
            block_size=meta["block_size"],
        )


@torch.inference_mode()
def dump(root: Path, model_name: str, item: dict) -> None:
    model, tok = load(model_name)
    ids = prompt_ids(tok, item["question"], model.device)
    seen = set()
    windows = set()
    sha = git_sha()

    def record(row: dict, tokens: Tensor, logits: Tensor | None, cache: object) -> None:
        kind = row["kind"]
        meta = {
            "prompt_id": item["id"],
            "model": model_name,
            "git_sha": sha,
            "M": len(tokens),
            "block_size": 32 if model_name == "dllm" else 1,
            "q_start": row["q_start"],
            "pass": row["pass"],
            "kind": kind,
        }
        if kind not in seen:
            layer_cases(model, tokens, cache, root, meta)
            seen.add(kind)
        if kind == "denoise" and len(windows) < 3:
            lo = row["window"][0]
            if lo not in windows:
                tau = (1.0, 0.9, 0.7)[len(windows)]
                windows.add(lo)
                save(
                    root / f"dllm-select-{tau}",
                    {**meta, "op": "select_commit", "layer": None},
                    dict(
                        logits=logits,
                        tokens=tokens,
                        window=tokens.new_tensor(row["window"], dtype=torch.int32),
                        tau=tau,
                        mask_id=MASK,
                    ),
                )

    if model_name == "dllm":
        decode(model, ids, 64, 0.9, observe=record)
        if "commit" not in seen:
            raise RuntimeError("Choose a longer-answer prompt for the commit fixture.")
    else:
        out = model(ids[None], use_cache=True, logits_to_keep=1)
        token = out.logits[:, -1].argmax(-1)
        record(
            {"kind": "ar", "pass": 1, "q_start": len(ids)},
            token,
            None,
            out.past_key_values,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/golden"))
    root = p.parse_args().output
    item = json.loads(Path("eval/gsm8k_100.jsonl").read_text().splitlines()[0])
    for model_name in ("dllm", "ar"):
        dump(root, model_name, item)
    print(root)
