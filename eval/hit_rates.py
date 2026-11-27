"""Measure how often the P2 and P3 guesses would be right, from per-pass traces.

Trace format: docs/INTERFACES.md §5. Definitions: docs/specs/accuracy.md, Problem 4.
A P3 rule sees a block's denoise passes up to pass t-1 and predicts {position: token}
for the slots pass t will commit.

Example: uv run python eval/hit_rates.py traces/n30.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from itertools import groupby
from pathlib import Path
from statistics import mean
from typing import NamedTuple

from dllm.results import ResultWriter

PILOT_DOUBLED_PASS_EXTRA_COST = 0.022
LOWERED_THRESHOLDS = (0.8, 0.7, 0.6, 0.5, 0.4, 0.3)

Commits = dict[int, int]
Rule = Callable[[list[dict]], Commits]


class Guess(NamedTuple):
    position: int
    token: int
    confidence: float


def masked_guesses(denoise_pass: dict) -> list[Guess]:
    return [Guess(*guess) for guess in denoise_pass["guesses"]]


def window_candidates(denoise_pass: dict) -> list[Guess]:
    start, end = denoise_pass["window"]
    return [g for g in masked_guesses(denoise_pass) if start <= g.position < end]


def most_confident(guesses: list[Guess]) -> Guess:
    """Leftmost on ties, like the one commit the decoder forces on every pass."""
    return max(guesses, key=lambda guess: guess.confidence)


def as_commits(guesses: list[Guess]) -> Commits:
    return {guess.position: guess.token for guess in guesses}


def actual_commits(denoise_pass: dict) -> Commits:
    return {position: token for position, token, _ in denoise_pass["committed"]}


def most_confident_only(block_history: list[dict]) -> Commits:
    return as_commits([most_confident(window_candidates(block_history[-1]))])


def confidence_above(threshold: float) -> Rule:
    def predict(block_history: list[dict]) -> Commits:
        candidates = window_candidates(block_history[-1])
        confident = [g for g in candidates if g.confidence > threshold]
        return as_commits(confident or [most_confident(candidates)])

    return predict


def repeated_and_rising(block_history: list[dict]) -> Commits:
    candidates = window_candidates(block_history[-1])
    if len(block_history) < 2:
        return as_commits([most_confident(candidates)])
    earlier = {g.position: g for g in masked_guesses(block_history[-2])}
    rising = [
        g
        for g in candidates
        if g.token == earlier[g.position].token
        and g.confidence > earlier[g.position].confidence
    ]
    return as_commits(rising or [most_confident(candidates)])


def top_k_by_previous_commit_count(block_history: list[dict]) -> Commits:
    previous = block_history[-1]
    by_confidence = sorted(
        window_candidates(previous), key=lambda g: g.confidence, reverse=True
    )
    return as_commits(by_confidence[: len(previous["committed"])])


def p3_rules(tau: float) -> dict[str, Rule]:
    lowered = {
        f"confidence above {threshold}": confidence_above(threshold)
        for threshold in LOWERED_THRESHOLDS
        if threshold < tau
    }
    return {
        "most confident slot only": most_confident_only,
        **lowered,
        "same guess twice, confidence rising": repeated_and_rising,
        "top k, k = previous commit count": top_k_by_previous_commit_count,
    }


def load_generations(trace_path: Path) -> dict[float, list[list[dict]]]:
    """Each generation's passes in recorded order, grouped by tau."""
    passes_by_generation = defaultdict(list)
    for line in trace_path.read_text().splitlines():
        record = json.loads(line)
        key = record["run_id"], record["prompt_id"], record["tau"]
        passes_by_generation[key].append(record)
    generations_by_tau = defaultdict(list)
    for (_, _, tau), passes in passes_by_generation.items():
        generations_by_tau[tau].append(passes)
    return generations_by_tau


def p2_outcomes(generations: list[list[dict]]) -> list[bool]:
    """One outcome per commit pass that another block follows."""
    return [
        record["t0_guess"] == record["t0_actual"]
        for passes in generations
        for record in passes[:-1]
        if record["kind"] == "commit"
    ]


def p3_opportunities(
    generations: list[list[dict]],
) -> Iterator[tuple[list[dict], dict]]:
    """Each pass that is not the first of its window, after its block's earlier passes."""
    for passes in generations:
        denoise_passes = [record for record in passes if record["kind"] == "denoise"]
        for _, block in groupby(denoise_passes, key=lambda record: record["block"]):
            block_passes = list(block)
            for index in range(1, len(block_passes)):
                current = block_passes[index]
                if block_passes[index - 1]["window"] == current["window"]:
                    yield block_passes[:index], current


def summarize(
    tau: float, generations: list[list[dict]], doubled_pass_extra_cost: float
) -> dict:
    all_passes = [record for passes in generations for record in passes]
    commit_counts = Counter(
        len(record["committed"]) for record in all_passes if record["kind"] == "denoise"
    )
    p2 = p2_outcomes(generations)
    opportunities = list(p3_opportunities(generations))
    applicable_fraction = len(opportunities) / len(all_passes)

    rules = {}
    for name, rule in p3_rules(tau).items():
        hit_rate = mean(
            rule(block_history) == actual_commits(current)
            for block_history, current in opportunities
        )
        rules[name] = {
            "hit_rate": hit_rate,
            "expected_speedup": (1 + hit_rate * applicable_fraction)
            / (1 + doubled_pass_extra_cost),
        }

    return {
        "tau": tau,
        "generations": len(generations),
        "passes": len(all_passes),
        "p2_hit_rate": mean(p2),
        "p2_commit_passes": len(p2),
        "p3_applicable_fraction": applicable_fraction,
        "p3_single_commit_fraction": mean(
            len(current["committed"]) == 1 for _, current in opportunities
        ),
        "p3_rules": rules,
        "denoise_passes_by_commit_count": dict(sorted(commit_counts.items())),
    }


def print_summary(summary: dict) -> None:
    print(
        f"\ntau = {summary['tau']}: {summary['generations']} generations, "
        f"{summary['passes']} passes"
    )
    print(
        f"P2 hit rate: {summary['p2_hit_rate']:.1%} "
        f"of {summary['p2_commit_passes']} commit passes followed by another block"
    )
    print(
        f"P3 can apply to {summary['p3_applicable_fraction']:.1%} of passes; "
        f"{summary['p3_single_commit_fraction']:.1%} of those commit a single slot"
    )
    print(f"{'P3 rule':<40}{'hit rate':>10}{'speedup':>10}")
    for name, rule in summary["p3_rules"].items():
        print(f"{name:<40}{rule['hit_rate']:>10.1%}{rule['expected_speedup']:>9.3f}x")
    print("Slots committed per denoise pass:")
    denoise_passes = sum(summary["denoise_passes_by_commit_count"].values())
    for commit_count, passes in summary["denoise_passes_by_commit_count"].items():
        print(f"{commit_count:>4}: {passes / denoise_passes:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "traces", type=Path, help="JSONL file from tools/dump_traces.py"
    )
    parser.add_argument(
        "--doubled-pass-extra-cost",
        type=float,
        default=PILOT_DOUBLED_PASS_EXTRA_COST,
        help="extra cost of a 64-row pass over a 32-row one, as a fraction",
    )
    args = parser.parse_args()

    writer = ResultWriter(
        "hit_rates",
        {
            "traces": str(args.traces),
            "doubled_pass_extra_cost": args.doubled_pass_extra_cost,
        },
    )
    for tau, generations in sorted(load_generations(args.traces).items()):
        summary = summarize(tau, generations, args.doubled_pass_extra_cost)
        writer.write(summary)
        print_summary(summary)
    print(f"\n{writer.path}")


if __name__ == "__main__":
    main()
