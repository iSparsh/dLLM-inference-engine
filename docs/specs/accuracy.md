# Slice: Accuracy and Decoding Analysis

**ECE 759 Final Project · Work slice spec**
Milestones: M0 Thu 11/19 · M1 Sun 11/22 · hit rates Wed 11/25 · M2 Sun 11/29 · M3 Wed 12/2 · slides 12/3 · report section 12/6

**What you hand in** (on branch `slice/accuracy`, merged to `main` by pull request):
- `eval/gsm8k.py` (the harness), `eval/analyze.py` (statistics and charts), `eval/hit_rates.py` (trace analysis).
- `scripts/euler/run_gsm8k.sh` and the sweep results in `results/gsm8k/` (small JSONL files; commit them).
- A short hit-rate note to the Engine slice by Wed 11/25 (`docs/report/hit-rates.md`).
- Your report section draft in `docs/report/accuracy.md`, the speed-vs-accuracy chart, and 2–3 slides.

**Hardware:** none of your own. Everything that needs a GPU runs on Euler; analysis runs on any laptop, macOS included.

---

## Background

The [GUIDE](../GUIDE.md) sections 4, 7 and 8 explain the ideas this slice measures. In short:

**Speed is only half the question.** The diffusion model (dLLM) commits several tokens per pass when it is confident. The threshold τ sets how confident it must be: a lower τ lets more slots through on each pass, which usually means fewer passes and more wrong guesses, but how much it changes speed and accuracy over a whole answer is exactly what you measure. Our engine makes each pass cheaper. This slice measures what that speed costs in accuracy, and whether the dLLM's accuracy holds up against the normal (AR) model, Qwen2.5-1.5B-Instruct.

**GSM8K** is a set of grade-school math word problems whose answer is one number. The model writes its reasoning and a final number; grading checks only the number. A single wrong guessed token in the arithmetic usually ruins the answer, which is exactly what aggressive parallel decoding risks. We use a fixed set of 100 problems, `eval/gsm8k_100.jsonl` (chosen with a fixed random seed, so it is the same everywhere).

**Statistics with 100 problems.** Near 50% accuracy, one run's 95% confidence interval is roughly ±10 points, too wide to tell two settings apart if they differ by 5 points. But every setting runs on the *same* 100 problems, so you can compare them **pairwise**: count the problems AR got right and the dLLM got wrong, and vice versa. McNemar's test on those two counts is far more sensitive than comparing two percentages. How sensitive it is depends on how many problems the two settings disagree on. That paired analysis is the core of this slice's report section.

**Why the accuracy numbers must come from our own runs.** NVIDIA's released evaluation script runs GSM8K at batch size 32 with τ = 1 (one token per pass); the paper's speed figures use τ = 0.9. A user reports that accuracy in the reference code depends on batch size (GitHub issue #39: 63.0% at batch 64 vs 57.3% at batch 4), and we run at batch size 1. Our October pilot on 20 problems got 55% (AR) vs 40–45% (dLLM), with a different prompt and settings. Your runs settle it for our setting.

**The decoding-strategy analysis (P2, P3, P4).** On our laptop GPU, a pass over 64 positions costs about the same as a pass over 32. The Engine slice can use that spare capacity, but only if certain guesses are usually right:
- **P2** merges the commit pass with the next block's first pass. It needs a guess of the next block's first token, taken from the previous pass. Worth building if the guess is right about ≥ 70% of the time.
- **P3** runs "this pass" and "a guessed next pass" together. It needs to guess, before pass `t` finishes, which tokens pass `t` will commit. Worth building if right about ≥ 40% of the time.
- **P4** changes the commit rule itself (approximate; it changes the output).

You'll measure the P2 and P3 hit rates from recorded traces, which needs no GPU, and evaluate P4 rules on GSM8K on Euler.

---

## Problem 1 (M0). Set up

a) Set up your environment ([ENVIRONMENT.md](../setup/ENVIRONMENT.md)): locally (macOS works) for analysis, and on Euler for runs:
```
uv run pytest -q                                  # locally: GPU tests skip, the rest pass
sbatch scripts/euler/setup_env.sh                 # on Euler, once
```
Check the Euler job output: the smoke test should load the dLLM and print ms per pass.

b) Read `dllm/reference_decode.py` (the decode loop you'll run), `dllm/results.py` (how results are written) and INTERFACES.md §3.6, §5, §6, §7.

c) Create your branch `slice/accuracy` and push it.

## Problem 2 (M1). The GSM8K harness

a) Write `eval/gsm8k.py`. For each problem in `eval/gsm8k_100.jsonl` (or a `--limit` subset), it builds the prompt, generates, extracts the answer, grades it, and writes one results row per problem with `dllm.results.ResultWriter`. Options: `--model {dllm, ar}`, `--engine {reference, ours}`, `--tau`, `--max-new-tokens` (default 512), `--limit`, `--out`.
   - **Prompt:** use `dllm.prompts.gsm8k_prompt_ids` (already in the starter code). It builds the same user message for both models (the question, a new line, and `Think step by step, then give the final numeric answer after '####'.`) and wraps it in each model's own chat template.
   - **Generation:** the dLLM through `dllm.reference_decode.decode` (block 32, sub-block 8, the given τ); the AR model through `dllm.reference_decode.ar_decode` (plain greedy). Count generated tokens up to and including the first stop token. Count passes the same way for every setting (excluding prefill), and don't turn on tracing while timing: it synchronizes the GPU.
   - **Extraction:** grade only the *generated* text (the prompt itself contains `####`). Take the first number after `####`; if there is no `####`, the last number in the generated text. Strip `,`, `$` and a trailing `.`; compare numerically with the gold answer.
   - **Row fields:** everything in INTERFACES.md §6 (the Slurm job id is recorded automatically), plus the generated `text` so a human can audit failures. If the texts make a results file bigger than 1 MB, write them to a separate, uncommitted file keyed by `prompt_id`.

b) Test the extraction on tricky strings (unit tests in `tests/test_gsm8k.py`): commas, dollars, negative numbers, decimals, `####` missing, several numbers after `####`.

c) Write `scripts/euler/run_gsm8k.sh` from `scripts/euler/job_template.sh`: one job runs one setting and must finish within 45 minutes. First time a 10-problem job for AR and for τ = 1.0 (the slowest settings) to see whether 100 fit in one job; if not, add `--shard i/n` (split by problem id) and run several jobs per setting. Run AR and dLLM τ = 0.9 first. Sanity-check against the pilot (AR ≈ 55%, dLLM ≈ 40–45% on 20 problems) and read 10 wrong answers by hand to make sure grading is fair.

## Problem 3 (M2). The threshold sweep

a) Run the full sweep on Euler: AR, and the dLLM at τ ∈ {1.0, 0.95, 0.9, 0.8, 0.7, 0.5}, all on the same 100 problems. Submit them as separate jobs (or a Slurm array); each fits in one job.

b) Write `eval/analyze.py`:
   - accuracy per setting with a 95% Wilson interval;
   - the paired table for each dLLM setting against AR and against τ = 1.0 (right/right, right/wrong, wrong/right, wrong/wrong) with McNemar's exact p-value (a two-sided binomial test on the discordant counts; implement it directly, it's a few lines);
   - tokens per pass and passes per problem per setting;
   - **the speed-vs-accuracy chart**: time per problem (x) against accuracy (y), one point per setting, the dLLM points connected in τ order, AR as its own point. Use the times from the same Euler runs, and label the chart with the GPU.

c) Commit the results files and the chart.

## Problem 4 (by Wed 11/25). P2 and P3 hit rates from traces

a) Generate traces on Euler for 30 GSM8K-100 problems at τ ∈ {0.9, 0.7} (same prompt and 512-token budget as the harness):
```
python tools/dump_traces.py --n 30 --tau 0.9 0.7 --out traces/n30.jsonl
```
Each line is one pass (INTERFACES.md §5): what was committed, and the guess and confidence for every slot still masked after it.

b) Write `eval/hit_rates.py` and compute:
   - **P2 hit rate:** over the commit passes that are followed by another block (the last block of an answer has no use for a guess), the fraction where `t0_guess == t0_actual`.
   - **P3 hit rate `h`:** P3 must decide, *before* pass `t` runs, which tokens pass `t` will commit, using only what pass `t−1` produced. The obvious rule, "apply τ to pass `t−1`'s confidences for the slots still masked", is useless: those slots are still masked precisely because they failed τ, so it only ever predicts the single forced commit. You need rules that anticipate confidence rising. Design and compare at least two, for example:
     - a lower threshold τ′ on pass `t−1`'s confidences (sweep τ′),
     - "the guess token was the same at `t−2` and `t−1` and its confidence rose",
     - the top-k most confident slots, with k from the recent commit counts.
     For each rule report `h` (the predicted set *and* tokens exactly equal pass `t`'s actual commits), and separately how often pass `t` committed only the forced slot vs more than one.
   - the distribution of tokens committed per pass.
   - the expected speedup from P3, `(1 + h·f) / (1 + c)`, where `f` is the fraction of passes where P3 could apply at all (not the first pass of a window) and `c` is the extra cost of the doubled pass. For `c`, use the cuBLAS estimate for now (a 64-row pass costs ~2.2% more than a 32-row one on the 4060) and say it is a lower bound; the real two-branch pass also does extra attention and selection work.

c) Write the go/no-go note `docs/report/hit-rates.md` (half a page: the numbers, and "build P2 / build P3: yes or no") and tell the Engine slice. This is a result in the report whether or not P2 and P3 get built.

## Problem 5 (M2). P4: a smarter commit rule

a) Add one or two approximate commit rules to `dllm/reference_decode.py` behind an explicit experimental option that is off by default (coordinate with the Engine slice, which owns that file). Candidates:
   - **stability commit:** also commit a masked slot whose guess has been the same for 2 consecutive passes with confidence above a lower threshold (e.g. 0.5);
   - **look-ahead commit:** also commit slots beyond the current 8-slot window if their confidence exceeds a stricter τ₂ (e.g. 0.98).

b) Run each rule on GSM8K-100 on Euler at τ = 0.9 and report passes per problem and accuracy against plain τ = 0.9, with the paired test. Add the points to the speed-vs-accuracy chart.

## Problem 6 (M3). Our engine's accuracy

When the Engine slice's driver runs with our kernels, rerun GSM8K-100 with `--engine ours` at τ = 0.9 and AR on Euler. Report per-problem agreement with the reference runs by comparing the generated token IDs, including the stop token (identical text or grades can hide differences). The claim "our engine is faster *at the same accuracy*" rests on this check.

## Problem 7 (M3). Write-up

a) Report section (`docs/report/accuracy.md`, about 1.5 pages): setup, the sweep table with intervals, the paired comparisons, the speed-vs-accuracy chart, the hit-rate results and what they implied for P2/P3, the P4 results, and the engine accuracy check.

b) 2–3 slides for the video.

---

## Seams

| You need | From | When |
|---|---|---|
| `reference_decode.py`, results writer, GSM8K-100 list, trace tool | starter code | now |
| `c` (64-row vs 32-row pass cost) | Selection slice (cross-GPU baselines) | M3; use the pilot's 4060 value until then |
| Engine with `ours` backend | Engine slice | M2 |

| You provide | To | When |
|---|---|---|
| P2/P3 hit rates and go/no-go | Engine slice | Wed 11/25 |
| Accuracy for every setting | Engine slice (final grid), report | M2 |
| Engine accuracy check | report | M3 |

## Hints

- Start with `--limit 2` locally on CPU (slow, but it works on macOS once `scripts/check_macos.sh` passes) or one short Euler job before running 100.
- Keep every generated text; the first thing anyone asks about an accuracy drop is "show me the failures".
- Store the Slurm job id in the results metadata so any number can be traced to the job that produced it.
- Don't tune the prompt per model. One prompt for everything keeps the comparison fair.
