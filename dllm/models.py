"""Load public checkpoints using their model-specific chat templates."""

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import MODEL_IDS


def load(
    model: str = "dllm", device: str | None = None, *, local_only: bool = False
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    if model not in MODEL_IDS:
        raise ValueError("Choose model='dllm' or 'ar'.")
    # Use local snapshot path also for remote-code class resolution in offline jobs.
    path = snapshot_download(
        MODEL_IDS[model],
        local_files_only=local_only,
        allow_patterns=["*.json", "*.py", "*.txt", "*.model", "*.safetensors"],
    )
    kwargs = {"trust_remote_code": model == "dllm", "local_files_only": True}
    network = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, **kwargs
    )
    if model == "ar":
        network.generation_config.do_sample = False
        network.generation_config.repetition_penalty = 1.0
    return network.to(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    ).eval(), AutoTokenizer.from_pretrained(path, **kwargs)


def prompt_ids(
    tokenizer: PreTrainedTokenizerBase, question: str, device: torch.device
) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)[0]
