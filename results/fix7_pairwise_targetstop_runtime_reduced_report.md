# Fix7 Pairwise-Only Target-Stop Runtime-Reduction Results

This experiment keeps pairwise-only QUBOs and reduces long-chain runtime using target-aware early stopping.
Rows for 20merA through 50mer are carried forward from the previous fresh pairwise-only validation.

## Complete Benchmark Table

| Benchmark | Target | Best | Runtime s | Hit rate | Notes |
|---|---:|---:|---:|---:|---|
| 20merA | 9 | 9 | 1.336 | 0.5000 | fresh pairwise QUBO; no warm start |
| 20merB | 10 | 10 | 1.356 | 0.6250 | fresh pairwise QUBO; no warm start |
| 24mer | 9 | 9 | 1.466 | 0.3750 | fresh pairwise QUBO; no warm start |
| 25mer | 8 | 8 | 1.710 | 0.4167 | fresh pairwise QUBO; no warm start |
| 36mer | 14 | 14 | 3.376 | 0.1562 | fresh pairwise QUBO; no warm start |
| 48mer | 23 | 23 | 37.103 | 0.1562 | fresh pairwise QUBO; no warm start |
| 50mer | 21 | 21 | 24.862 | 0.1562 | fresh pairwise QUBO; no warm start |
| 60mer | 36 | 36 | 746.805 | 0.0036 | target-stop staged run; 3 in-directory stages; warm seed contacts=35 |
| 64mer | 42 | 42 | 282.110 | 0.0833 | validated fixed-budget setting with target-aware stop; no warm start |

## New 60mer/64mer Iterations

| Benchmark | Iter | Label | Stop contacts | Warm contacts | Runtime s | Best | Actual trials | Hit rate | Target hit | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|:---:|---|
| 60mer | 1 | `stage1_nowarm_seed35_128tr_2M_targetstop` | 35 |  | 85.232 | 35 | 9 | 0.0000 | no | in-directory 35-contact seed discovery; no warm start |
| 60mer | 2 | `stage2_selfwarm_256tr_2M_targetstop` | 36 | 35 | 425.990 | 35 | 256 | 0.0000 | no | self-warm refinement from in-directory seed; self-warm from in-directory best (35 contacts) |
| 60mer | 3 | `stage3_selfwarm_256tr_2M_targetstop_retry` | 36 | 35 | 235.583 | 36 | 9 | 0.1111 | yes | self-warm retry if stage 2 misses target; self-warm from in-directory best (35 contacts) |
| 64mer | 1 | `pilot_nowarm_128tr_1p5M_targetstop` | 42 |  | 193.763 | 39 | 128 | 0.0000 | no | pilot smaller trial count; no warm start |
| 64mer | 2 | `pilot_nowarm_256tr_1M_targetstop` | 42 |  | 255.894 | 40 | 256 | 0.0000 | no | pilot smaller step count; no warm start |
| 64mer | 3 | `pilot_nowarm_192tr_1p2M_targetstop` | 42 |  | 243.909 | 39 | 192 | 0.0000 | no | pilot balanced trial/step budget; no warm start |
| 64mer | 4 | `fallback_nowarm_256tr_1p5M_targetstop` | 42 |  | 282.110 | 42 | 12 | 0.0833 | yes | validated fixed-budget setting with target-aware stop; no warm start |

## Interpretation

- `Runtime s` in the complete table is the selected runtime for the benchmark result.
- For `60mer`, runtime is cumulative because the target run depends on the in-directory seed-discovery stage.
- For `64mer`, runtime is the fastest successful no-warm pilot/fallback configuration, because failed pilots are tuning overhead rather than required warm-start stages.
- `Hit rate` is computed from completed trial traces. In target-stop runs, some workers may be terminated before writing a worker-best file, so the denominator is the number of completed trials, not the requested full budget.




