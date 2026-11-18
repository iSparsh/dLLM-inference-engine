"""Check installation, run the teaching kernel and time one short reference generation."""

import platform
import time

import torch
import transformers

from dllm import ops
from dllm.models import load, prompt_ids
from dllm.reference_decode import decode


@torch.inference_mode()
def main() -> None:
    print(
        f"Python {platform.python_version()}, torch {torch.__version__}, transformers {transformers.__version__}, CUDA available={torch.cuda.is_available()}"
    )
    if not torch.cuda.is_available():
        y = ops.silu_mul(torch.ones(1, 17920, dtype=torch.bfloat16))
        assert y.shape == (1, 8960)
        print(
            "CPU-only subset passed. CUDA extension and checkpoint generation require an NVIDIA GPU for this smoke test."
        )
        return
    ops.set_backend("example_axpy", "ours")
    x = torch.arange(256, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(ops.example_axpy(x, x, 2.0), 3 * x)
    model, tok = load()
    ids = prompt_ids(
        tok,
        "Mia buys three bags of four pears. How many pears does she buy?",
        model.device,
    )
    count = 0

    def count_forward(*_: object) -> None:
        nonlocal count
        count += 1

    hook = model.register_forward_hook(count_forward)
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        out = decode(model, ids, 64, 0.9)
    finally:
        hook.remove()
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - start)
    print(
        f"example_axpy passed; generated tokens={len(out) - len(ids)}, passes={count} including prefill, ms/pass={ms / count:.3f}"
    )


if __name__ == "__main__":
    main()
