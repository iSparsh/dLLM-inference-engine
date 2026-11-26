"""Run GSM8K-100 on one model setting and grade it (docs/specs/accuracy.md, Problem 2).

Writes one results row per problem (docs/INTERFACES.md §6) plus the generated text.

Example: uv run python eval/gsm8k.py --model dllm --tau 0.9 --limit 2
The P4 commit rules are experimental: --stable-guess 0.5 or --look-ahead 0.98.
"""

import argparse
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor
from transformers import PreTrainedModel

from dllm.commit_rules import CommitRule, StableGuess, look_ahead
from dllm.config import BLOCK, SUB_BLOCK
from dllm.models import load
from dllm.prompts import gsm8k_prompt_ids
from dllm.reference_decode import ar_decode, decode
from dllm.results import ResultWriter

PROBLEMS = Path("eval/gsm8k_100.jsonl")
ANSWER_MARKER = "####"
NUMBER = re.compile(r"(?<!\d)-?\$?\d[\d,]*(?:\.\d+)?")

Decoder = Callable[[Tensor], Tensor]


def extract_answer(generated_text: str) -> str | None:
    """The first number after the marker, or the last number when there is no marker."""
    _, marker, after_marker = generated_text.partition(ANSWER_MARKER)
    numbers = NUMBER.findall(after_marker if marker else generated_text)
    if not numbers:
        return None
    answer = numbers[0] if marker else numbers[-1]
    return answer.replace(",", "").replace("$", "")


def is_correct(predicted: str | None, gold: str) -> bool:
    return predicted is not None and float(predicted) == float(gold)


def prefill_passes(model_name: str, prompt_tokens: int) -> int:
    return 1 if model_name == "ar" or prompt_tokens > BLOCK else 0


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def commit_rule(args: argparse.Namespace) -> CommitRule | None:
    if args.stable_guess is not None:
        return StableGuess(args.stable_guess)
    if args.look_ahead is not None:
        return look_ahead(args.look_ahead)
    return None


def generate(
    model: PreTrainedModel, model_name: str, run_decoder: Decoder, prompt_ids: Tensor
) -> tuple[Tensor, int, float]:
    """Generated token ids, passes after prefill, and wall-clock milliseconds."""
    forward_calls = 0

    def count_forward_call(*_: object) -> None:
        nonlocal forward_calls
        forward_calls += 1

    hook = model.register_forward_hook(count_forward_call)
    synchronize()
    start = time.perf_counter()
    try:
        output_ids = run_decoder(prompt_ids)
    finally:
        hook.remove()
    synchronize()
    ms_total = 1000 * (time.perf_counter() - start)
    passes = forward_calls - prefill_passes(model_name, len(prompt_ids))
    return output_ids[len(prompt_ids) :], passes, ms_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=["dllm", "ar"], default="dllm")
    parser.add_argument("--engine", choices=["reference"], default="reference")
    parser.add_argument("--tau", type=float, default=0.9, help="dLLM threshold")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, help="run only the first N problems")
    extra_commits = parser.add_mutually_exclusive_group()
    extra_commits.add_argument(
        "--stable-guess",
        type=float,
        metavar="CONFIDENCE",
        help="P4: also commit window slots whose guess repeats, above this confidence",
    )
    extra_commits.add_argument(
        "--look-ahead",
        type=float,
        metavar="CONFIDENCE",
        help="P4: also commit slots after the window above this confidence",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/gsm8k"), help="results folder"
    )
    args = parser.parse_args()

    lines = PROBLEMS.read_text().splitlines()[: args.limit]
    problems = [json.loads(line) for line in lines]
    model, tokenizer = load(args.model)
    if args.model == "dllm":
        rule = commit_rule(args)
        decoding = {
            "tau": args.tau,
            "block_size": BLOCK,
            "sub_block": SUB_BLOCK,
            "stable_guess": args.stable_guess,
            "look_ahead": args.look_ahead,
        }

        def run_decoder(prompt_ids: Tensor) -> Tensor:
            return decode(
                model, prompt_ids, args.max_new_tokens, args.tau, extra_commits=rule
            )
    else:
        decoding = {"tau": None}

        def run_decoder(prompt_ids: Tensor) -> Tensor:
            return ar_decode(model, prompt_ids, args.max_new_tokens)

    writer = ResultWriter(
        args.out.name,
        {
            "engine": args.engine,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            **decoding,
        },
        root=args.out.parent,
    )

    def prompt_ids_for(problem: dict) -> Tensor:
        return gsm8k_prompt_ids(tokenizer, problem["question"], model.device)

    # Untimed warm-up, so the first timed problem does not pay start-up costs.
    generate(model, args.model, run_decoder, prompt_ids_for(problems[0]))

    correct_count = 0
    for problem in problems:
        prompt_ids = prompt_ids_for(problem)
        generated_ids, passes, ms_total = generate(
            model, args.model, run_decoder, prompt_ids
        )
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted = extract_answer(text)
        correct = is_correct(predicted, problem["answer"])
        correct_count += correct
        writer.write(
            {
                "prompt_id": problem["id"],
                "prompt_tokens": len(prompt_ids),
                "gen_tokens": len(generated_ids),
                "passes": passes,
                "ms_total": ms_total,
                "ms_per_pass": ms_total / passes,
                "tok_per_s": 1000 * len(generated_ids) / ms_total,
                "pred": predicted,
                "gold": problem["answer"],
                "correct": correct,
                "text": text,
                "token_ids": generated_ids.tolist(),
            }
        )
        print(
            f"{problem['id']} correct={correct} pred={predicted} "
            f"gold={problem['answer']} passes={passes} ms={ms_total:.0f}",
            flush=True,
        )
    print(f"accuracy {correct_count}/{len(problems)}")
    print(writer.path)


if __name__ == "__main__":
    main()
