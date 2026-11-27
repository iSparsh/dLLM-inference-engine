import pytest

from eval.gsm8k import extract_answer, is_correct


@pytest.mark.parametrize(
    ("generated_text", "expected"),
    [
        ("4 * 16 = 64\n#### 64", "64"),
        ("#### 57,500", "57500"),
        ("#### $1,234.50", "1234.50"),
        ("#### -7", "-7"),
        ("#### -$7", "-7"),
        ("#### 0.25", "0.25"),
        ("#### 18.", "18"),
        ("#### 18 dollars, not 20", "18"),
        ("#### The answer is 12.", "12"),
        ("3 + 4 = 7, so she has 7 apples.", "7"),
        ("He pays 20-8=12 dollars", "12"),
        ("Final answer: 15####", None),
        ("#### no number here", None),
        ("I cannot solve this.", None),
    ],
)
def test_extract_answer(generated_text: str, expected: str | None) -> None:
    assert extract_answer(generated_text) == expected


def test_subtraction_is_not_a_negative_number() -> None:
    assert extract_answer("That leaves 20-8") == "8"


@pytest.mark.parametrize(
    ("predicted", "gold", "expected"),
    [
        ("64", "64", True),
        ("64.00", "64", True),
        ("-7", "-7", True),
        ("65", "64", False),
        (None, "64", False),
    ],
)
def test_is_correct(predicted: str | None, gold: str, expected: bool) -> None:
    assert is_correct(predicted, gold) is expected
