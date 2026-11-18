"""Write one metadata sidecar and append JSONL measurements per run."""

import json
import os
import platform
import subprocess
import uuid
from pathlib import Path

import torch

from . import ops
from .timing import gpu_state


def git_sha() -> str:
    p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else "unknown"


class ResultWriter:
    def __init__(
        self, experiment: str, metadata: dict, root: Path = Path("results")
    ) -> None:
        self.run_id = uuid.uuid4().hex
        folder = root / experiment
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{self.run_id}.jsonl"
        meta = {
            **metadata,
            **gpu_state(),
            "git_sha": git_sha(),
            "host": platform.node(),
            "torch": str(torch.__version__),
            "ops": ops.available(),
            "run_id": self.run_id,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        self.path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )

    def write(self, row: dict) -> None:
        with self.path.open("a") as stream:
            stream.write(json.dumps({**row, "run_id": self.run_id}) + "\n")
