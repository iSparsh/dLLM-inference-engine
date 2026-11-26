# P2 and P3 hit rates: go/no-go

**Recommendation: build P2. Build P3 only for τ = 0.9, with the simplest guessing rule; it does not clear the bar at τ = 0.7.**

Measured from per-pass traces of the reference decoder on the first 30 GSM8K-100 problems, 512-token budget, on Euler (job 130451). Numbers come from `results/hit_rates/190d58b6e1544ec9adc78cf5906e00e6.jsonl`, produced by `eval/hit_rates.py traces/n30.jsonl`.

| | τ = 0.9 | τ = 0.7 | Bar to build |
|---|---|---|---|
| Passes (30 problems) | 2,934 | 2,282 | |
| **P2 hit rate** | **92.6%** (175 of 189) | **91.7%** (176 of 192) | ≥ 70% |
| Passes P2 would save | 6.0% | 7.7% | |
| **P3 hit rate, best rule** | **47.3%** | **35.5%** | ≥ 40% |
| Passes where P3 can apply (`f`) | 67.0% | 56.1% | |
| P3 expected speedup, at best | 1.29× | 1.17× | |

## P2: yes

The guess of the next block's first token is right more than 9 times in 10 at both thresholds, well above the 70% bar. The gain is bounded by how many passes are commit passes: 6–8% of all passes.

## P3: yes at τ = 0.9, no at τ = 0.7

A hit means the predicted slots and tokens equal exactly what the next pass commits.

| Rule for predicting pass `t` from pass `t−1` | τ = 0.9 | τ = 0.7 |
|---|---|---|
| The most confident still-masked slot of the window, alone | 47.3% | 34.7% |
| Every slot above a lower threshold τ′ (best τ′) | 47.1% (τ′ = 0.8) | 35.5% (τ′ = 0.5) |
| Top k slots, k = commits of pass `t−1` | 40.3% | 33.6% |
| Same guess at `t−2` and `t−1` with rising confidence | 32.5% | 28.6% |

- **The simplest rule is as good as any.** None of the rules that try to anticipate rising confidence beats "predict the single most confident slot" by more than one point. That rule can only be right when the next pass commits one slot, which is 71.5% of applicable passes at τ = 0.9 and 60.0% at τ = 0.7.
- **The speedup figures are upper bounds.** They use `(1 + h·f) / (1 + c)` with `c` = 2.2%, the pilot's cuBLAS estimate for a 64-row pass on the 4060. The real two-branch pass also does extra attention and selection work.
- **`f` is counted over all passes,** commit passes included; a pass is applicable when it is not the first pass of its window.

## Limits

- 30 problems per threshold. The traces come from the reference decoder at commit `4ebaf9d` plus the chat-template download fix.
- The hit rates are properties of the model's decisions, not of the GPU, but they were measured on one GPU type; bf16 rounding can differ slightly across devices.
