# Table 8. RB-Full Pairwise Target-Stop Validation And Runtime Results

This table combines the previous runtime-reduced Table 8, all-benchmark validation Table 10, and the Istrail-long target-refinement runs into one primary RB-full result table. It reports target recovery, best HP energy, runtime, hit rate, and run notes for the full benchmark set from `20merA` through `100merB`.

Sources: short/medium benchmark rows are carried forward from `runs\fix7_pairwise_fresh_qubo_summary.csv`; 60mer/64mer target-stop rows come from `runs\fix7_pairwise_targetstop_runtime_reduced_summary.csv`; 85mer comes from `runs\istrail_long_fix7_16w_big_budget_summary.csv`; 100merA/B come from `runs\istrail_100mer_L13_target_refine_summary.csv`.

Pairwise-only QUBOs contain no 3-body or 4-body terms. For target-stop rows, hit rate is computed over completed trial traces before termination, not over the originally requested full budget.

| Benchmark | N | Target contacts | Target HP energy | RB-full best contacts | RB-full best HP energy | Target hit | Runtime s | Hit rate | Notes |
|---|---:|---:|---:|---:|---:|:---:|---:|---:|---|
| 20merA | 20 | 9 | -9 | 9 | -9 | yes | 1.336 | 0.5000 | fresh pairwise QUBO; no warm start |
| 20merB | 20 | 10 | -10 | 10 | -10 | yes | 1.356 | 0.6250 | fresh pairwise QUBO; no warm start |
| 24mer | 24 | 9 | -9 | 9 | -9 | yes | 1.466 | 0.3750 | fresh pairwise QUBO; no warm start |
| 25mer | 25 | 8 | -8 | 8 | -8 | yes | 1.710 | 0.4167 | fresh pairwise QUBO; no warm start |
| 36mer | 36 | 14 | -14 | 14 | -14 | yes | 3.376 | 0.1562 | fresh pairwise QUBO; no warm start |
| 48mer | 48 | 23 | -23 | 23 | -23 | yes | 37.103 | 0.1562 | fresh pairwise QUBO; no warm start |
| 50mer | 50 | 21 | -21 | 21 | -21 | yes | 24.862 | 0.1562 | fresh pairwise QUBO; no warm start |
| 60mer | 60 | 36 | -36 | 36 | -36 | yes | 746.805 | 0.0036 | target-stop staged run; 3 in-directory stages; warm seed contacts=35 |
| 64mer | 64 | 42 | -42 | 42 | -42 | yes | 282.110 | 0.0833 | validated fixed-budget setting with target-aware stop; no warm start |
| 85mer | 85 | 53 | -53 | 53 | -53 | yes | 1124.237 | 0.0159 | 16-worker RB-full warm-budget run; L=12; RB-full-produced warm seed contacts=52 |
| 100merA | 100 | 48 | -48 | 48 | -48 | yes | 7285.698 | 0.0526 | L=13 target refinement; L=12 RB-full warm start remapped by coordinate; hp_reward=-4.0 |
| 100merB | 100 | 50 | -50 | 50 | -50 | yes | 6698.349 | 0.0667 | L=13 target refinement; L=12 RB-full warm start remapped by coordinate; hp_reward=-4.0 |

This merged table should be used as the main RB-full validation/runtime table in the manuscript.




