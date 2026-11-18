import pytest
import torch
from huggingface_hub.errors import LocalEntryNotFoundError

from dllm.models import load


@pytest.fixture(scope="module")
def checkpoint(request: pytest.FixtureRequest) -> tuple:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for model parity")
    try:
        return load(getattr(request, "param", "dllm"), local_only=True)
    except LocalEntryNotFoundError:
        pytest.skip("Download the model checkpoint before running model parity tests")
