"""Profile NVIDIA's reference generate() at batch size 1 (docs/research/reference-profile.md).

For each of the first N GSM8K-100 prompts: generated tokens, forward calls, wall ms per
call, CUDA-event ms per forward, time outside the forward, tok/s. With --profile, also runs
torch.profiler over 4 post-prefill forwards of the first prompt and prints GPU-busy time,
kernels per pass and the top kernels grouped by category.

Example: uv run python tools/profile_reference.py --model dllm --prompts 3 --tau 0.9 --profile
"""

import argparse
import collections
import functools
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from dllm.models import load
from dllm.prompts import gsm8k_prompt_ids

CATEGORIES = [
    ("attention", ["attention", "fmha", "flash", "sdpa"]),
    ("GEMM", ["gemm", "gemv", "cutlass", "cublas", "s168", "mma"]),
    ("copies/cat", ["cat", "copy"]),
    ("softmax/selection", ["softmax", "argmax", "max_kernel"]),
    ("norms/elementwise", ["elementwise", "reduce", "norm"]),
]


def category(kernel: str) -> str:
    key = kernel.lower()
    for name, needles in CATEGORIES:
        if any(n in key for n in needles):
            return name
    return "other"


def union_ms(spans: list[tuple[float, float]]) -> float:
    end = total = 0.0
    for a, b in sorted(spans):
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total / 1000


def analyze(trace_file: str) -> None:
    events = json.load(open(trace_file))["traceEvents"]
    marks = sorted(
        (e for e in events if e["name"] == "REF_FORWARD" and e.get("cat") == "user_annotation"),
        key=lambda e: e["ts"],
    )
    t0, t1, n = marks[0]["ts"], marks[-1]["ts"], len(marks) - 1
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and t0 <= e["ts"] < t1]
    kernels = [e for e in gpu if e["cat"] == "kernel"]
    print(f"profiled passes: {n}")
    print(f"  wall ms/pass:      {(t1 - t0) / 1000 / n:.3f}")
    print(f"  GPU-busy ms/pass:  {union_ms([(e['ts'], min(t1, e['ts'] + e['dur'])) for e in gpu]) / n:.3f}")
    print(f"  kernels per pass:  {len(kernels) / n:.0f}")
    by_cat, by_name = collections.Counter(), collections.Counter()
    for e in kernels:
        by_cat[category(e["name"])] += e["dur"]
        by_name[(category(e["name"]), e["name"][:90])] += e["dur"]
    print("  ms/pass by category:", {k: round(v / 1000 / n, 3) for k, v in by_cat.most_common()})
    for (cat, name), dur in by_name.most_common(15):
        print(f"    {dur / 1000 / n:7.3f}  {cat:18s} {name}")
    print("  (profiler overhead inflates wall time; use the unprofiled rows for throughput)")


def gpu_clocks() -> str:
    q = "pstate,clocks.sm,clocks.mem,power.draw"
    r = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip()


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["dllm", "ar"], default="dllm")
    p.add_argument("--prompts", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--profile", action="store_true", help="also run torch.profiler on the first prompt")
    a = p.parse_args()
    model, tok = load(a.model)
    items = [json.loads(l) for l in Path("eval/gsm8k_100.jsonl").read_text().splitlines()[: a.prompts]]
    kw = dict(max_new_tokens=a.max_new_tokens)
    if a.model == "dllm":
        kw.update(threshold=a.tau, block_size=32, small_block_size=8, use_block_cache=False)
    else:
        kw.update(do_sample=False, repetition_penalty=1.0)
    original = model.forward
    print("model prompt_id prompt_tokens new_tokens calls wall_ms ms/call forward_ms/call outside_ms/call tok/s")
    for j, item in enumerate(items):
        ids = gsm8k_prompt_ids(tok, item["question"], model.device)[None]
        if a.model == "ar":
            kw["attention_mask"] = torch.ones_like(ids)
        if j == 0:
            model.generate(ids, **kw)  # warm-up
        for profiled in [False, True] if (j == 0 and a.profile) else [False]:
            pairs: list = []
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if profiled else None

            def wrapped(*args, **kwargs):
                i = len(pairs)
                if prof and i == 1:
                    prof.start()
                if prof and i == 5:
                    prof.stop()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                if prof and 1 <= i < 5:
                    with record_function("REF_FORWARD"):
                        out = original(*args, **kwargs)
                else:
                    out = original(*args, **kwargs)
                end.record()
                pairs.append((start, end))
                return out

            model.forward = functools.wraps(original)(wrapped)
            torch.cuda.synchronize()
            tic = time.perf_counter()
            out = model.generate(ids, **kw)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - tic) * 1000
            model.forward = original
            if prof:
                with tempfile.TemporaryDirectory() as d:
                    trace = os.path.join(d, "trace.json")
                    prof.export_chrome_trace(trace)
                    analyze(trace)
                continue
            n, gen = len(pairs), out.shape[1] - ids.shape[1]
            fwd = sum(s.elapsed_time(e) for s, e in pairs) / n
            print(
                f"{a.model} {item['id']} {ids.shape[1]} {gen} {n} {wall:.1f} {wall / n:.3f} "
                f"{fwd:.3f} {wall / n - fwd:.3f} {gen * 1000 / wall:.2f}"
            )
            print("  clocks:", gpu_clocks())


if __name__ == "__main__":
    main()
