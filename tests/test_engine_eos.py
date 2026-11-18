import torch

from dllm.config import EOS, MASK
from dllm.engine.eos import eos_ready

STOPS = torch.tensor([EOS, 151643])


def ready(seq: list[int], gen_start: int) -> bool:
    return bool(eos_ready(torch.tensor(seq), torch.tensor([gen_start]), MASK, STOPS))


def test_no_stop_yet() -> None:
    assert not ready([1, 2, 3, MASK, MASK], 2)


def test_stop_after_filled_slots() -> None:
    assert ready([1, 2, 3, EOS, MASK], 2)


def test_stop_with_hole_before_it_waits() -> None:
    assert not ready([1, 2, MASK, EOS, MASK], 2)


def test_prompt_stop_is_ignored() -> None:
    assert not ready([EOS, 2, 3, MASK], 1)


def test_either_stop_id_counts() -> None:
    assert ready([1, 2, 151643, MASK], 2)
