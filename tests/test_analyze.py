import pytest

from eval.analyze import (
    PairedCounts,
    Run,
    mcnemar_p_value,
    paired_counts,
    wilson_interval,
)


def run_with(correct_by_problem: dict[str, bool]) -> Run:
    rows = {
        prompt_id: {"correct": correct}
        for prompt_id, correct in correct_by_problem.items()
    }
    return Run(model="dllm", tau=0.9, commit_rule=None, gpu="any", rows=rows)


def test_wilson_interval_matches_the_textbook_value() -> None:
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4)
    assert high == pytest.approx(0.5962, abs=1e-4)


def test_wilson_interval_stays_inside_zero_and_one() -> None:
    low, high = wilson_interval(0, 100)
    assert low == pytest.approx(0)
    assert 0 < high < 0.05


@pytest.mark.parametrize(
    ("only_first_right", "only_second_right", "expected"),
    [
        (1, 9, 22 / 1024),
        (9, 1, 22 / 1024),
        (5, 5, 1.0),
        (0, 0, 1.0),
        (0, 6, 2 / 64),
    ],
)
def test_mcnemar_p_value(
    only_first_right: int, only_second_right: int, expected: float
) -> None:
    assert mcnemar_p_value(only_first_right, only_second_right) == pytest.approx(
        expected
    )


def test_paired_counts() -> None:
    setting = run_with({"a": True, "b": True, "c": False, "d": False, "e": True})
    baseline = run_with({"a": True, "b": False, "c": True, "d": False, "e": False})

    assert paired_counts(setting, baseline) == PairedCounts(
        both_right=1, only_setting_right=2, only_baseline_right=1, both_wrong=1
    )
