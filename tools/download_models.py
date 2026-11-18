"""Download both checkpoints (about 3.1 GB each) into the Hugging Face cache.

The cache is $HF_HOME if set, otherwise ~/.cache/huggingface. Safe to rerun: files
already present are skipped. Usage: uv run python tools/download_models.py [dllm|ar ...]
"""

import sys

from huggingface_hub import snapshot_download

from dllm.config import MODEL_IDS

PATTERNS = ["*.json", "*.py", "*.txt", "*.model", "*.safetensors"]


def main() -> None:
    names = sys.argv[1:] or list(MODEL_IDS)
    for name in names:
        if name not in MODEL_IDS:
            sys.exit(f"Unknown model '{name}'. Choose from: {', '.join(MODEL_IDS)}")
        path = snapshot_download(MODEL_IDS[name], allow_patterns=PATTERNS)
        print(f"{name}: {MODEL_IDS[name]} -> {path}")


if __name__ == "__main__":
    main()
