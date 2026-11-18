# Work Slices

The project is split into four slices. Each slice is written like a course assignment: background, numbered problems, exact commands, and what to hand in by each milestone. Pick the slice that fits your hardware and interests; the lead takes the Engine slice because it is the critical path and touches everything else.

Read [GUIDE.md](../GUIDE.md) first, then [INTERFACES.md](../INTERFACES.md) (the contracts every slice codes against), then your slice. Set up your machine with [setup/ENVIRONMENT.md](../setup/ENVIRONMENT.md).

## The slices

| Slice | What you build | Hardware you need | LLM knowledge | Difficulty |
|---|---|---|---|---|
| [**Engine**](engine.md) (lead) | The driver: decode loop, static KV cache, CUDA Graph, AR mode; integrating every kernel; P1–P3; the final benchmark grid | Local NVIDIA GPU | Deep | Hard, wide |
| [**Attention path**](attention.md) | K2 block-causal attention and the RoPE + KV-cache-write kernel | Any NVIDIA GPU, or Euler | Some (explained in the spec) | Hard, deep |
| [**Selection and fused ops**](selection.md) | K1 parallel selection, the add + RMSNorm and SiLU × mul kernels, and the cross-GPU baselines | Any NVIDIA GPU, or Euler; an NVIDIA GPU of your own for the baselines | Little | Medium, several pieces |
| [**Accuracy and decoding analysis**](accuracy.md) | The GSM8K harness and threshold sweep on Euler, the P2/P3 hit-rate analysis, the P4 commit rules | Euler only (macOS or any laptop is fine) | Some | Medium, analysis-heavy |

Each slice is sized to be about one person's share of a course project worth three programming assignments.

## Milestones (same for every slice)

| Milestone | Date | Meaning |
|---|---|---|
| **M0 Setup** | Thu 11/19 | Your environment works (`smoke_test.py` and `pytest` pass locally or on Euler). You have read your spec and opened your branch. |
| **M1 Correct** | Sun 11/22 | Your first piece is correct against the reference and pushed. Its tests must actually run and pass: a green `pytest` that only *skips* your op doesn't count. |
| **M2 Fast and integrated** | Sun 11/29 | Optimized, profiled, merged into `main`, and running inside the engine (or on its results). |
| **M3 Final numbers** | Wed 12/2 | Your final measurements are in `results/`. Results are frozen. |
| **Slides** | Thu 12/3 | Your 2–3 slides are in the shared deck. Recording around 12/4. |
| **Report** | Sun 12/6 | Your report section is drafted. Final report due 12/7 23:59. |

## Rules for everyone

- **Branch and pull request.** Work on `slice/<name>`. Merge into `main` by pull request with at least one teammate's review. `main` must always pass `pytest`.
- **Commit and push small, often.** The course grades the commit history. Several commits a week from each person, each with a clear message.
- **Code against INTERFACES.md.** If a contract is wrong or awkward, propose the change in a pull request to that file and tell the team; don't silently diverge.
- **Results go through `dllm.results`.** Every number in the report comes from a results file that a script produced, never from a hand-copied terminal line.
- **Euler etiquette.** One GPU per job, 45 minutes maximum, nothing run on the login node.

## Where the slices meet

| Seam | Producer | Consumer | Ready by |
|---|---|---|---|
| Op contracts and reference ops (`dllm/ops/reference.py`) | starter code | Attention, Selection, Engine | now |
| Golden tensors (`tools/dump_golden.py`) | Engine publishes a copy; anyone with a GPU or Euler can regenerate them | Attention, Selection | M0 |
| Traces (`tools/dump_traces.py`) | Engine (tool in starter code) | Accuracy, Selection | M0 |
| GSM8K-100 list (`eval/gsm8k_100.jsonl`) | starter code | Accuracy, Engine | now |
| CUDA ops `torch.ops.dllm.*` | Attention, Selection | Engine | M1 correct, M2 fast |
| Hit rates for P2 and P3 (they decide whether P2/P3 get built at all) | Accuracy | Engine | 11/25 |
| Engine with `ours` backend | Engine | Accuracy (re-check accuracy), Selection (baselines grid) | M2 |
| Cross-GPU baselines | Selection | Engine (final grid), report | M3 |

If you are blocked on a seam for more than a day, say so in the team chat. Every slice has a way to keep moving: random tensors, the reference code, or the PyTorch version of an op.
