"""Check outputs and mutations, skipping only an explicit unfinished kernel."""

import json
from pathlib import Path

import pytest
import torch

from dllm import golden, ops
from dllm.op_cases import random_case
from dllm.ops import reference


def run_ours(name: str, args: dict) -> torch.Tensor | None:
    ops.set_backend(name, "ours")
    try:
        return getattr(ops, name)(**args)
    except RuntimeError as exc:
        if f"not implemented yet: {name}" in str(exc):
            pytest.skip(f"{name} CUDA stub")
        raise
    finally:
        ops.set_backend(name, "torch")


def compare_ulp(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dtype != torch.bfloat16:
        torch.testing.assert_close(a, b)
        return
    below = torch.nextafter(b, torch.full_like(b, -torch.inf))
    above = torch.nextafter(b, torch.full_like(b, torch.inf))
    assert ((a >= below) & (a <= above)).float().mean().item() >= 0.999


def compare(name: str, args: dict, expected: dict) -> None:
    actual = run_ours(name, args)
    if "out" in expected and expected["out"] is not None:
        assert actual.dtype == expected["out"].dtype
        assert actual.shape == expected["out"].shape
    if name == "block_attention":
        oracle = reference.block_attention(
            args["q"].float(),
            args["k_cache"].float(),
            args["v_cache"].float(),
            args["q_start"],
            args["block_size"],
        )
        error = (actual.float() - oracle).abs().max()
        assert error <= 0.02 and error <= (expected["out"].float() - oracle).abs().max()
    elif actual is not None:
        if name == "select_commit":
            assert torch.equal(actual, expected["out"])
        else:
            compare_ulp(actual, expected["out"])
    for key in golden.MUTATES.get(name, ()):
        if key == "q":
            compare_ulp(args[key], expected[key])
        elif key == "k_cache":
            start = int(args["q_start"])
            end = start + len(args["q"])
            assert torch.equal(args[key][:, :start], expected[key][:, :start])
            assert torch.equal(args[key][:, end:], expected[key][:, end:])
            compare_ulp(args[key][:, start:end], expected[key][:, start:end])
        else:
            assert torch.equal(args[key], expected[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("name", "m"),
    [
        (name, m)
        for name in ops.available()
        for m in ([32] if name == "select_commit" else [1, 32, 64])
    ],
)
def test_random(name: str, m: int) -> None:
    args = random_case(name, m, 96, 192, 1 if m == 1 else 32)
    # Non-dense outer strides are valid. Run each reference before trying the stub.
    for k, t in list(args.items()):
        if isinstance(t, torch.Tensor) and t.ndim >= 2:
            view = t.new_empty(t.shape[:-1] + (t.shape[-1] + 3,))[..., : t.shape[-1]]
            args[k] = view.copy_(t)
    work = dict(args)
    for key in golden.MUTATES.get(name, ()):
        work[key] = args[key].clone()
    out = getattr(reference, name)(**work)
    compare(name, args, {"out": out, **work})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "case",
    sorted(Path("artifacts/golden").glob("*/meta.json")),
    ids=lambda p: f"{json.loads(p.read_text())['op']}-{p.parent.name}",
)
def test_golden(case: Path) -> None:
    meta, tensors = golden.load(case.parent)
    args = {
        **meta["scalars"],
        **{k: t.cuda() for k, t in tensors.items() if not k.startswith("expected_")},
    }
    expected = {
        k.removeprefix("expected_"): t.cuda()
        for k, t in tensors.items()
        if k.startswith("expected_")
    }
    compare(meta["op"], args, expected)


def test_select_commit_tie_cpu() -> None:
    logits = torch.full((32, 16), -100, dtype=torch.bfloat16)
    logits[:, :2] = torch.tensor([0.0, 0.001])
    tokens = torch.full((32,), 151665)
    stats = reference.select_commit(
        logits, tokens, torch.tensor([8, 16], dtype=torch.int32), 1.0, 151665
    )
    assert tokens[8] == 0 and stats.tolist() == [1, 0, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_example_graph() -> None:
    args = random_case("example_axpy", 1, 0, 32, 1)
    run_ours("example_axpy", args)
    graph = torch.cuda.CUDAGraph()
    ops.set_backend("example_axpy", "ours")
    try:
        with torch.cuda.graph(graph):
            out = ops.example_axpy(**args)
        args["x"].fill_(3)
        graph.replay()
        torch.testing.assert_close(out, reference.example_axpy(**args))
    finally:
        ops.set_backend("example_axpy", "torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m", [1, 32, 64])
@pytest.mark.parametrize("b", [1, 32])
@pytest.mark.parametrize("start", [0, 47, 128])
def test_block_attention_boundaries(m: int, b: int, start: int) -> None:
    args = random_case("block_attention", m, start, 192, b)
    compare("block_attention", args, {"out": reference.block_attention(**args)})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_add_rmsnorm_without_x() -> None:
    args = random_case("add_rmsnorm", 32, 0, 32, 32)
    args["x"] = None
    residual = args["residual"].clone()
    compare(
        "add_rmsnorm",
        args,
        {"out": reference.add_rmsnorm(**args), "residual": residual},
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("empty", [False, True])
def test_select_commit_rounded_tie(empty: bool) -> None:
    args = random_case("select_commit", 32, 0, 32, 32)
    args["logits"].fill_(-100)
    args["logits"][:, 0] = 0
    args["logits"][:, 1] = 0.001
    if empty:
        args["tokens"][8:16] = 42
    tokens = args["tokens"].clone()
    expected = reference.select_commit(**{**args, "tokens": tokens})
    compare("select_commit", args, {"out": expected, "tokens": tokens})
