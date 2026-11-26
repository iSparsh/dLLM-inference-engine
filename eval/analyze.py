"""Summarize the GSM8K runs: accuracy, paired comparisons and the speed-vs-accuracy chart.

Reads every run eval/gsm8k.py wrote to one results folder and prints Markdown tables
(docs/specs/accuracy.md, Problems 3 and 5).

Example: uv run python eval/analyze.py results/gsm8k
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from math import comb, sqrt
from pathlib import Path
from statistics import mean
from typing import NamedTuple

import matplotlib.pyplot as plt

Z_FOR_95_PERCENT = 1.959964
ONE_TOKEN_PER_PASS_TAU = 1.0

AR_COLOR, DLLM_COLOR, COMMIT_RULE_COLOR = "#f47230", "#257de0", "#00b894"
TEXT_COLOR, GRID_COLOR, LEADER_COLOR = "#191919", "#e3e3e3", "#9c9c9c"
RULE_LABEL_PLACEMENT = {
    "stable guess > 0.5": ((10, -3), "left"),
    "stable guess > 0.8": ((10, 30), "left"),
    "look-ahead > 0.98": ((22, 14), "left"),
    "look-ahead > 0.9": ((-12, 24), "right"),
}


@dataclass(frozen=True)
class Run:
    model: str
    tau: float | None
    commit_rule: str | None
    gpu: str
    rows: dict[str, dict]

    @property
    def label(self) -> str:
        if self.model == "ar":
            return "AR"
        threshold = f"τ = {self.tau}"
        return f"{threshold}, {self.commit_rule}" if self.commit_rule else threshold

    @property
    def correct(self) -> int:
        return sum(row["correct"] for row in self.rows.values())

    @property
    def accuracy(self) -> float:
        return self.correct / len(self.rows)

    @property
    def tokens_per_pass(self) -> float:
        rows = self.rows.values()
        return sum(row["gen_tokens"] for row in rows) / sum(
            row["passes"] for row in rows
        )

    @property
    def passes_per_problem(self) -> float:
        return mean(row["passes"] for row in self.rows.values())

    @property
    def seconds_per_problem(self) -> float:
        return mean(row["ms_total"] for row in self.rows.values()) / 1000


class PairedCounts(NamedTuple):
    both_right: int
    only_setting_right: int
    only_baseline_right: int
    both_wrong: int


def commit_rule_label(meta: dict) -> str | None:
    if meta.get("stable_guess") is not None:
        return f"stable guess > {meta['stable_guess']}"
    if meta.get("look_ahead") is not None:
        return f"look-ahead > {meta['look_ahead']}"
    return None


def load_run(meta_path: Path) -> Run:
    meta = json.loads(meta_path.read_text())
    rows_path = meta_path.with_name(f"{meta['run_id']}.jsonl")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    return Run(
        model=meta["model"],
        tau=meta["tau"],
        commit_rule=commit_rule_label(meta),
        gpu=meta["gpu"],
        rows={row["prompt_id"]: row for row in rows},
    )


def load_runs(results_folder: Path) -> list[Run]:
    """One run per setting: AR, then the dLLM by falling tau, then the commit rules."""
    runs = [load_run(path) for path in sorted(results_folder.glob("*.meta.json"))]
    repeated = [
        label
        for label, count in Counter(run.label for run in runs).items()
        if count > 1
    ]
    if repeated:
        raise ValueError(f"More than one run for {repeated}; keep one per setting.")
    if len({frozenset(run.rows) for run in runs}) > 1:
        raise ValueError("The runs cover different problems, so they cannot be paired.")
    if len({run.gpu for run in runs}) > 1:
        gpus = {run.label: run.gpu for run in runs}
        raise ValueError(f"Timings need one GPU model for every run; found {gpus}.")
    return sorted(
        runs,
        key=lambda run: (run.model != "ar", run.commit_rule or "", -(run.tau or 0)),
    )


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    z = Z_FOR_95_PERCENT
    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    half_width = (
        z
        * sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return center - half_width, center + half_width


def mcnemar_p_value(only_first_right: int, only_second_right: int) -> float:
    """Exact two-sided binomial test on the problems where the two settings disagree."""
    disagreements = only_first_right + only_second_right
    smaller_side = min(only_first_right, only_second_right)
    tail = sum(comb(disagreements, k) for k in range(smaller_side + 1))
    return min(1.0, 2 * tail / 2**disagreements)


def paired_counts(setting: Run, baseline: Run) -> PairedCounts:
    outcomes = Counter(
        (row["correct"], baseline.rows[prompt_id]["correct"])
        for prompt_id, row in setting.rows.items()
    )
    return PairedCounts(
        outcomes[True, True],
        outcomes[True, False],
        outcomes[False, True],
        outcomes[False, False],
    )


def comparisons(runs: list[Run]) -> list[tuple[Run, Run]]:
    """Each dLLM setting against AR and tau = 1.0; each commit rule against its plain tau."""
    autoregressive = next(run for run in runs if run.model == "ar")
    plain_by_tau = {
        run.tau: run for run in runs if run.model == "dllm" and not run.commit_rule
    }
    pairs = []
    for run in runs:
        if run.commit_rule:
            pairs.append((run, plain_by_tau[run.tau]))
        elif run.model == "dllm":
            pairs.append((run, autoregressive))
            if run.tau != ONE_TOKEN_PER_PASS_TAU:
                pairs.append((run, plain_by_tau[ONE_TOKEN_PER_PASS_TAU]))
    return pairs


def format_p_value(p_value: float) -> str:
    return f"{p_value:.3f}" if p_value >= 0.001 else "< 0.001"


def print_settings_table(runs: list[Run]) -> None:
    print(
        "| Setting | Correct | Accuracy (95% interval) | Tokens per pass "
        "| Passes per problem | Seconds per problem |"
    )
    print("|---|---|---|---|---|---|")
    for run in runs:
        low, high = wilson_interval(run.correct, len(run.rows))
        print(
            f"| {run.label} | {run.correct} of {len(run.rows)} "
            f"| {run.accuracy:.0%} ({low:.0%}–{high:.0%}) | {run.tokens_per_pass:.2f} "
            f"| {run.passes_per_problem:.0f} | {run.seconds_per_problem:.2f} |"
        )


def print_paired_table(runs: list[Run]) -> None:
    print(
        "\n| Setting | Baseline | Both right | Only setting right "
        "| Only baseline right | Both wrong | McNemar p |"
    )
    print("|---|---|---|---|---|---|---|")
    for setting, baseline in comparisons(runs):
        counts = paired_counts(setting, baseline)
        p_value = mcnemar_p_value(counts.only_setting_right, counts.only_baseline_right)
        print(
            f"| {setting.label} | {baseline.label} | {counts.both_right} "
            f"| {counts.only_setting_right} | {counts.only_baseline_right} "
            f"| {counts.both_wrong} | {format_p_value(p_value)} |"
        )


def plot_points(
    axes: plt.Axes, runs: list[Run], color: str, marker: str, name: str
) -> None:
    axes.scatter(
        [run.seconds_per_problem for run in runs],
        [100 * run.accuracy for run in runs],
        s=64,
        color=color,
        marker=marker,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
        label=name,
    )


def label_point(
    axes: plt.Axes, run: Run, text: str, offset: tuple[int, int], align: str
) -> None:
    axes.annotate(
        text,
        (run.seconds_per_problem, 100 * run.accuracy),
        xytext=offset,
        textcoords="offset points",
        ha=align,
        color=TEXT_COLOR,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": LEADER_COLOR, "linewidth": 0.8},
    )


def draw_chart(runs: list[Run], chart_path: Path) -> None:
    autoregressive = [run for run in runs if run.model == "ar"]
    plain = [run for run in runs if run.model == "dllm" and not run.commit_rule]
    with_rule = [run for run in runs if run.commit_rule]

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(
        [run.seconds_per_problem for run in plain],
        [100 * run.accuracy for run in plain],
        color=DLLM_COLOR,
        linewidth=2,
        zorder=2,
    )
    plot_points(axes, plain, DLLM_COLOR, "o", "dLLM, by threshold τ")
    plot_points(axes, autoregressive, AR_COLOR, "s", "AR (Qwen2.5-1.5B-Instruct)")
    if with_rule:
        rule_series = f"dLLM, τ = {with_rule[0].tau} + commit rule"
        plot_points(axes, with_rule, COMMIT_RULE_COLOR, "^", rule_series)
    for run in plain + autoregressive:
        label_point(axes, run, run.label, offset=(0, -16), align="center")
    for run in with_rule:
        offset, align = RULE_LABEL_PLACEMENT.get(run.commit_rule, ((0, 10), "center"))
        label_point(axes, run, run.commit_rule, offset, align)

    axes.set_xlabel("Mean time per problem (s)")
    axes.set_ylabel("Accuracy on GSM8K-100 (%)")
    axes.set_title(
        f"Speed vs accuracy, reference decoder, {runs[0].gpu}\n"
        "100 problems per point; 95% intervals are about ±10 points",
        loc="left",
        fontsize=10,
        color=TEXT_COLOR,
    )
    axes.margins(x=0.08, y=0.12)
    axes.grid(color=GRID_COLOR, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(frameon=False, loc="lower right")
    figure.tight_layout()
    figure.savefig(chart_path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, help="folder written by eval/gsm8k.py")
    parser.add_argument(
        "--chart", type=Path, default=Path("docs/report/speed-vs-accuracy.png")
    )
    args = parser.parse_args()

    runs = load_runs(args.results)
    print_settings_table(runs)
    print_paired_table(runs)
    draw_chart(runs, args.chart)
    print(f"\n{args.chart}")


if __name__ == "__main__":
    main()
