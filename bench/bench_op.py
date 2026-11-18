"""Time one operator, reporting logical minimum bytes per §8 (no temporary traffic)."""

import argparse
from pathlib import Path
from collections.abc import Callable
from itertools import product

import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import repeat_kv

from dllm import ops
from dllm.golden import MUTATES
from dllm.op_cases import random_case
from dllm.results import ResultWriter
from dllm.timing import benchmark


def attention_baseline(name: str, args: dict, start: int) -> Callable[[], torch.Tensor]:
    """Prepare real-length inputs; time HF-style assembly or just the unmasked SDPA call."""
    q = args["q"].transpose(0, 1)[None]
    m = q.size(2)
    length = start + m
    block_size = args["block_size"]
    k, v = (args[key][None, :, :length] for key in ("k_cache", "v_cache"))
    if name == "sdpa_nomask":
        q = q.contiguous()
        keys, values = (repeat_kv(t, 6).contiguous() for t in (k, v))
        return lambda: F.scaled_dot_product_attention(q, keys, values)

    if name != "hf_cat":
        raise ValueError(f"Unknown attention baseline: {name}")
    prefix_k, prefix_v = k[:, :, :start].contiguous(), v[:, :, :start].contiguous()
    # New HF projections are row-major [1,M,2,128] viewed head-first.
    new_k, new_v = (
        t[:, :, start:].transpose(1, 2).contiguous().transpose(1, 2) for t in (k, v)
    )

    def hf_cat() -> torch.Tensor:
        keys = repeat_kv(torch.cat((prefix_k, new_k), dim=2), 6)
        values = repeat_kv(torch.cat((prefix_v, new_v), dim=2), 6)
        pos = args["q_start"] + torch.arange(m, device=q.device)
        key_pos = torch.arange(length, device=q.device)
        mask = key_pos // block_size <= pos[:, None] // block_size
        out = F.scaled_dot_product_attention(
            q.contiguous(), keys, values, attn_mask=mask
        )
        return out.transpose(1, 2).contiguous()[0]

    return hf_cat


def logical_bytes(op: str, m: int, start: int, args: dict) -> int:
    if op == "example_axpy":
        return 3 * args["x"].nbytes
    if op == "silu_mul":
        return 3 * m * 8960 * 2
    if op == "add_rmsnorm":
        return (4 * m + 1) * 1536 * 2
    if op == "block_attention":
        return 4 + (2 * m * 12 + 4 * (start + m)) * 128 * 2
    if op == "rope_kv_write":
        return 4 + m * (24 + 4 + 4 + 2) * 128 * 2
    # Eight distinct predictor rows for [8,16); read block for block_done.
    # Random logits and tau=.9 force a single token store.
    return 8 * args["logits"].shape[-1] * 2 + 32 * 8 + 8 + 8 + 12


def capture_graph(fn, prepare) -> object:
    """Warm up fn on a side stream, capture it once, and return the graph's replay."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            prepare()
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    prepare()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", choices=ops.available(), default="example_axpy")
    p.add_argument("--backend", choices=["torch", "ours"], default="torch")
    p.add_argument("--m", "--M", type=int, choices=[1, 32, 64])
    p.add_argument("--q-start", type=int, default=96)
    p.add_argument("--capacity", type=int, default=4096)
    p.add_argument(
        "--L", type=int, help="Logical KV length including the current query"
    )
    p.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Use 1..19 for profiler launches only; 20+ for timing",
    )
    p.add_argument(
        "--baseline",
        choices=["hf_cat", "sdpa_nomask"],
        help="Library comparison for block_attention; default torch is the full-capacity reference.",
    )
    p.add_argument(
        "--sweep-L",
        action="store_true",
        help="Time all 15 attention shape/length combinations.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/bench_op"),
        help="results folder, e.g. results/attention",
    )
    p.add_argument(
        "--graph",
        action="store_true",
        help="time a CUDA Graph replay of the op (GPU time without per-launch CPU overhead, as in the engine)",
    )
    a = p.parse_args()
    if (a.baseline or a.sweep_L) and a.op != "block_attention":
        p.error("--baseline and --sweep-L require --op block_attention")
    if a.baseline and a.backend == "ours":
        p.error("Library baselines use torch; omit --backend ours")
    if a.sweep_L and (
        a.m is not None or a.L is not None or a.capacity < 4096 or a.iters < 20
    ):
        p.error(
            "--sweep-L requires all shapes, capacity >= 4096 and at least 20 timed iterations"
        )
    if a.iters < 1:
        p.error("--iters must be positive")
    ops.set_backend(a.op, a.backend)
    shapes = [a.m] if a.m else ([32] if a.op == "select_commit" else [1, 32, 64])
    lengths = [128, 512, 1024, 2048, 4096] if a.sweep_L else [a.L]
    implementation = a.baseline or (
        "torch_full_capacity_reference"
        if a.op == "block_attention" and a.backend == "torch"
        else a.backend
    )
    if a.graph:
        implementation += "+graph"
    for m, length in product(shapes, lengths):
        b = 1 if m == 1 else 32
        start = a.q_start if length is None else length - m
        if start < 0 or start + m > a.capacity:
            p.error("Require 0 <= q_start <= capacity-M")
        args = random_case(a.op, m, start, a.capacity, b)
        baseline = {k: args[k].clone() for k in MUTATES.get(a.op, ())}

        def prepare() -> None:
            for k, v in baseline.items():
                args[k].copy_(v)

        def run() -> None:
            getattr(ops, a.op)(**args)

        timed = attention_baseline(a.baseline, args, start) if a.baseline else run
        if a.graph:
            timed = capture_graph(timed, prepare)
        if a.iters < 20:
            for _ in range(a.iters):
                prepare()
                timed()
            torch.cuda.synchronize()
            print(f"Profiler mode: {implementation}, {a.iters} launches at M={m}")
            continue
        ms = benchmark(timed, repeats=a.iters, prepare=prepare)
        writer = ResultWriter(
            a.out_dir.name,
            {
                "engine": "ours" if a.backend == "ours" else "reference",
                "model": "ar" if m == 1 else "dllm",
                "tau": 0.9,
                "block_size": b,
                "sub_block": 8,
            },
            root=a.out_dir.parent,
        )
        size = logical_bytes(a.op, m, start, args)
        if a.baseline == "sdpa_nomask":
            size = (
                (2 * m + 2 * (start + m)) * 12 * 128 * 2
            )  # Q, output and two expanded BF16 inputs.
        row = dict(
            op=a.op,
            implementation=implementation,
            L=start + m,
            backend=a.backend,
            M=m,
            q_start=start,
            capacity=a.capacity,
            block_size=b,
            ms=ms,
            bytes=size,
            GB_per_s=size / (1e6 * ms),
        )
        writer.write(row)
        print(row)
        print(writer.path)


if __name__ == "__main__":
    main()
