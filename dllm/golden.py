"""Store signature-named inputs and expected outputs for one operator invocation."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .ops import reference

MUTATES = {
    "add_rmsnorm": ("residual",),
    "rope_kv_write": ("q", "k_cache", "v_cache"),
    "select_commit": ("tokens",),
}


def save(case_dir: Path, meta: dict, args: dict) -> None:
    """Compute expectations with the torch reference without changing live activations."""
    name = meta["op"]
    tensors = {
        k: v.detach().to("cpu", copy=True).contiguous()
        for k, v in args.items()
        if isinstance(v, torch.Tensor)
    }
    scalars = {k: v for k, v in args.items() if k not in tensors}
    work = {**args}
    for k in MUTATES.get(name, ()):
        work[k] = args[k].clone()
    out = getattr(reference, name)(**work)
    if out is not None:
        tensors["expected_out"] = out.detach().cpu().contiguous()
    for k in MUTATES.get(name, ()):
        tensors["expected_" + k] = work[k].detach().cpu().contiguous()
    case_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(case_dir / "tensors.safetensors"))
    (case_dir / "meta.json").write_text(
        json.dumps({**meta, "scalars": scalars}, indent=2) + "\n"
    )


def load(case_dir: Path | str) -> tuple[dict, dict[str, torch.Tensor]]:
    case_dir = Path(case_dir)
    return json.loads((case_dir / "meta.json").read_text()), load_file(
        str(case_dir / "tensors.safetensors")
    )
