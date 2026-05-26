# Fix7 Pairwise-Only Fresh-QUBO Self-Warm Experiment

Every QUBO in this experiment was freshly exported with 3-body and 4-body terms disabled.
Warm starts are only allowed from `best_overall.json` files generated inside the same benchmark's `fix7_pairwise_fresh_qubo` directory.

## Final Benchmark Table

| Benchmark | Target | Best | Runtime s | Hit rate | Notes |
|---|---:|---:|---:|---:|---|
| 20merA | 9 | 9 | 1.336 | 0.5000 | fresh pairwise QUBO; no warm start |
| 20merB | 10 | 10 | 1.356 | 0.6250 | fresh pairwise QUBO; no warm start |
| 24mer | 9 | 9 | 1.466 | 0.3750 | fresh pairwise QUBO; no warm start |
| 25mer | 8 | 8 | 1.710 | 0.4167 | fresh pairwise QUBO; no warm start |
| 36mer | 14 | 14 | 3.376 | 0.1562 | fresh pairwise QUBO; no warm start |
| 48mer | 23 | 23 | 37.103 | 0.1562 | fresh pairwise QUBO; no warm start |
| 50mer | 21 | 21 | 24.862 | 0.1562 | fresh pairwise QUBO; no warm start |
| 60mer | 36 | 36 | 855.086 | 0.0020 | fresh pairwise QUBO; target reached after 2 in-directory iterations; warm seed contacts=35 |
| 64mer | 42 | 42 | 381.052 | 0.0039 | fresh pairwise QUBO; no warm start |

## All Iterations

| Benchmark | Iter | Label | Warm contacts | Runtime s | Contacts | Hit rate | Target hit |
|---|---:|---|---:|---:|---:|---:|:---:|
| 20merA | 1 | `nowarm_4tr_20k` |  | 1.336 | 9 | 0.5000 | yes |
| 20merB | 1 | `nowarm_16tr_50k` |  | 1.356 | 10 | 0.6250 | yes |
| 24mer | 1 | `nowarm_24tr_60k` |  | 1.466 | 9 | 0.3750 | yes |
| 25mer | 1 | `nowarm_24tr_60k` |  | 1.710 | 8 | 0.4167 | yes |
| 36mer | 1 | `nowarm_32tr_100k` |  | 3.376 | 14 | 0.1562 | yes |
| 48mer | 1 | `nowarm_64tr_500k` |  | 37.103 | 23 | 0.1562 | yes |
| 50mer | 1 | `nowarm_64tr_300k` |  | 24.862 | 21 | 0.1562 | yes |
| 60mer | 1 | `nowarm_256tr_2M` |  | 422.335 | 35 | 0.0000 | no |
| 60mer | 2 | `selfwarm_256tr_2M` | 35 | 432.750 | 36 | 0.0039 | yes |
| 64mer | 1 | `nowarm_256tr_1p5M` |  | 381.052 | 42 | 0.0039 | yes |

## Files

- Combined summary CSV: `runs/fix7_pairwise_fresh_qubo_summary.csv`
- Per-benchmark folders: `runs/<benchmark>/local/Final_run/fix7_pairwise_fresh_qubo/`
- Pairwise-only QUBOs: `runs/<benchmark>/local/Final_run/fix7_pairwise_fresh_qubo/qubos/`




