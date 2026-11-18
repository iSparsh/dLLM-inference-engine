"""Build the dispatcher library on demand using PyTorch's versioned build cache."""

import os
import sysconfig
from functools import cache
from pathlib import Path

import torch


@cache
def load() -> None:
    from torch.utils.cpp_extension import CUDA_HOME
    from torch.utils.cpp_extension import load as jit_load

    if not torch.cuda.is_available() or CUDA_HOME is None:
        raise RuntimeError(
            "The ours backend needs an NVIDIA GPU and CUDA toolkit. Select the torch backend or load CUDA_HOME."
        )
    src = Path(__file__).resolve().parents[2] / "csrc"
    if not (src / "ops.cpp").exists():
        src = Path(sysconfig.get_path("data")) / "share/dllm/csrc"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6;8.9")
    os.environ.setdefault("MAX_JOBS", "2")  # Bound simultaneous nvcc host compilations.
    sources = [src / "ops.cpp", *sorted(src.glob("*.cu"))]
    jit_load(
        name="dllm_course_ops", sources=list(map(str, sources)), is_python_module=False
    )
