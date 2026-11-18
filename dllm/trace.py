"""Host-side trace construction, opt-in because recording synchronizes the device."""

import json
from collections.abc import Callable
from pathlib import Path

from torch import Tensor

TraceHook = Callable[[dict], None]


def predictions(
    logits: Tensor, before: Tensor, after: Tensor, mask_id: int
) -> tuple[list, list]:
    conf, guess = logits.softmax(-1).max(-1)
    scores, ids = conf.tolist(), guess.tolist()
    committed, guesses = [], []
    for i, (old, new) in enumerate(zip(before.tolist(), after.tolist())):
        if old == mask_id:
            j = max(0, i - 1)
            record = [i, ids[j], scores[j]]
            (guesses if new == mask_id else committed).append(record)
    return committed, guesses


class TraceWriter:
    def __init__(self, path: Path, **metadata: object) -> None:
        self.path, self.metadata = path, metadata
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, row: dict) -> None:
        with self.path.open("a") as out:
            out.write(json.dumps({**self.metadata, **row}) + "\n")
