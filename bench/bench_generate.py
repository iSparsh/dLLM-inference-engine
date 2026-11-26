"""Time end-to-end generation on GSM8K-100 prompts and write one results row per prompt.

The first waterfall bars (docs/specs/engine.md Problem 2e): NVIDIA's reference
generate() and the stage-1 engine (static cache + CUDA Graphs, stock ops), on the same
prompts and settings. Times are wall clock from the prompt ids on the GPU to the output
ids on the GPU, synchronized at both ends; ms_per_pass divides by every forward pass,
prefill included. Token counts run through the first stop token.

  uv run python bench/bench_generate.py --engine reference --model dllm --tau 0.9
  uv run python bench/bench_generate.py --engine ours --model dllm --tau 0.9
  uv run python bench/bench_generate.py --engine ours --ours silu_mul,select_commit
"""

import argparse
import functools
import json
import time
from pathlib import Path

import torch

from dllm import ops
from dllm.config import BLOCK, EOS
from dllm.engine import Engine
from dllm.models import load
from dllm.prompts import gsm8k_prompt_ids
from dllm.results import ResultWriter


def through_stop(out: torch.Tensor, n: int, stops: set[int]) -> int:
    """Generated tokens up to and including the first stop token."""
    for i, token in enumerate(out[n:].tolist()):
        if token in stops:
            return i + 1
    return len(out) - n


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["dllm", "ar"], default="dllm")
    p.add_argument("--engine", choices=["reference", "ours"], default="ours")
    p.add_argument("--tau", type=float, default=0.9)
    p.add_argument("--prompts", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--capacity", type=int, default=1024)
    p.add_argument("--bucket", type=int, default=256)
    p.add_argument("--ours", default="", help="ops to run on the CUDA backend: comma list or 'all'")
    p.add_argument("--out-dir", default="engine", help="experiment folder under results/")
    a = p.parse_args()
    names = list(ops.available()) if a.ours == "all" else filter(None, a.ours.split(","))
    for name in names:
        ops.set_backend(name, "ours")

    model, tok = load(a.model, local_only=True)
    items = [json.loads(l) for l in Path("eval/gsm8k_100.jsonl").read_text().splitlines()[: a.prompts]]
    prompts = [gsm8k_prompt_ids(tok, item["question"], model.device) for item in items]
    if a.model == "dllm":
        stops = {EOS}
    else:
        eos = model.generation_config.eos_token_id
        stops = {eos} if isinstance(eos, int) else set(eos)

    calls = 0
    if a.engine == "reference":
        kw = dict(max_new_tokens=a.max_new_tokens)
        if a.model == "dllm":
            kw.update(threshold=a.tau, block_size=BLOCK, small_block_size=8, use_block_cache=False)
        else:
            kw.update(do_sample=False, repetition_penalty=1.0)
        original = model.forward

        @functools.wraps(original)
        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        model.forward = counted

        def run(ids: torch.Tensor) -> tuple[torch.Tensor, int]:
            nonlocal calls
            calls = 0
            extra = {"attention_mask": torch.ones_like(ids[None])} if a.model == "ar" else {}
            return model.generate(ids[None], **kw, **extra)[0], calls
    else:
        engine = Engine(model, a.model, capacity=a.capacity, bucket=a.bucket)
        engine.prepare(max(len(i) for i in prompts) + a.max_new_tokens + BLOCK, a.tau)

        def run(ids: torch.Tensor) -> tuple[torch.Tensor, int]:
            if a.model == "dllm":
                out = engine.generate(ids, a.max_new_tokens, a.tau)
            else:
                out = engine.ar_generate(ids, a.max_new_tokens)
            prefill = int(a.model == "ar" or len(ids) > BLOCK)
            return out, engine.stats.passes + prefill

    run(prompts[0])  # warm-up: clocks, cuBLAS handles, any lazy capture
    writer = ResultWriter(
        a.out_dir,
        {
            "engine": "hf" if a.engine == "reference" else "ours",
            "model": a.model,
            "tau": a.tau if a.model == "dllm" else None,
            "block_size": BLOCK if a.model == "dllm" else 1,
            "sub_block": 8 if a.model == "dllm" else None,
            "max_new_tokens": a.max_new_tokens,
            "capacity": a.capacity if a.engine == "ours" else None,
            "bucket": a.bucket if a.engine == "ours" else None,
            "graphs": a.engine == "ours",
        },
    )
    total_ms = total_tokens = 0.0
    for item, ids in zip(items, prompts):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        out, passes = run(ids)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - tic) * 1000
        gen = through_stop(out, len(ids), stops)
        row = {
            "prompt_id": item["id"],
            "prompt_tokens": len(ids),
            "gen_tokens": gen,
            "passes": passes,
            "ms_total": round(ms, 2),
            "ms_per_pass": round(ms / passes, 3),
            "tok_per_s": round(gen * 1000 / ms, 2),
        }
        writer.write(row)
        total_ms += ms
        total_tokens += gen
        print(row, flush=True)
    print(f"{a.engine} {a.model}: {total_tokens * 1000 / total_ms:.2f} tok/s over {len(items)} prompts -> {writer.path}")


if __name__ == "__main__":
    main()
