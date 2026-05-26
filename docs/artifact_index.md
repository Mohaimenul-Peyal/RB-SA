# Artifact Index

This file summarizes the public repository artifacts needed to inspect or
recreate the RB-full experiments.

## Core Code

Some file names retain the historical internal label `fix7`. These files are
the final RB-full implementation used for the manuscript experiments.

| File | Purpose |
|---|---|
| `src/HP_export_manybody_ising.py` | Exports the pairwise HP QUBO/Ising JSON files used in the final benchmark runs. |
| `src/HP_ssa_manybody_ising_cpu_numba_saw_pivot_fix7_contactguided.py` | Final RB-full annealer. |
| `src/HP_ssa_manybody_parallel_runner_cpu_fast_passthru.py` | Multi-worker launcher with optional target-aware stopping. |
| `src/plot_best_fold_best_overall.py` | Creates fold plots from decoded `best_overall.json` outputs. |

## Experiment Wrappers

| File | Purpose |
|---|---|
| `experiments/run_fix7_pairwise_fresh_qubo_selfwarm.py` | Fresh pairwise QUBO export and RB-full runs for 20merA through 64mer. |
| `experiments/run_fix7_pairwise_targetstop_runtime_reduction.py` | Target-stop runtime-reduced 60mer and 64mer validation. |
| `experiments/run_fix7_istrail_long_scalability.py` | Initial long-chain RB-full scalability stage for 85mer and 100mer warm-state discovery. |
| `experiments/run_fix7_istrail_long_warm_budget.py` | Long-chain warm-budget refinement stage. |
| `experiments/run_fix7_istrail_long_16w_big_budget.py` | 16-worker long-chain refinement stage; final 85mer target-hit source. |
| `experiments/run_fix7_istrail_100mer_L13_target_refine.py` | L=13 target refinement for 100merA and 100merB. |

## Final Results

| File | Purpose |
|---|---|
| `benchmarks/istrail_complete_suite.csv` | Complete benchmark index from 20merA through 100merB with paths to final artifacts. |
| `results/final_benchmark_results.csv` | Primary final validation table from 20merA through 100merB. |
| `results/qubos/final_qubo_manifest.csv` | Final QUBO paths, lattice sizes, objective weights, penalties, and expected source sizes. |
| `results/qubos/README.md` | Explains why full generated QUBO JSON payloads are regenerated or attached through release/LFS rather than committed to the standard checkout. |
| `results/ground_folds/index.csv` | Inventory of included final fold JSONs and plots. |
| `results/commands/final_run_commands.csv` | Final per-benchmark command snapshots. |
| `results/tables/` | Manuscript-facing CSV/Markdown tables copied for reproducibility. |
| `results/figure_tables/` | Figure-source tables copied for traceability. |

## Ground Fold Folders

Each folder under `results/ground_folds/<benchmark>/` contains:

| File | Meaning |
|---|---|
| `best_overall.json` | Final decoded fold used for the reported contact count. |
| `warm_start_best.json` | Included only when the final row was warm-started. |
| `fold.png` | Raster fold plot for quick inspection. |
| `fold.pdf` | Vector fold plot for manuscript or presentation use. |
| `metadata.json` | QUBO metadata, run configuration, and command array. |
| `summary.csv` | One-row summary for the selected run. |
| `command.txt` | Repository-relative command snapshot. |
| `command_relative.txt` | Same command retained under an explicit name. |

## Ground Fold Rows

| Benchmark | Target | Included contacts | Fold folder |
|---|---:|---:|---|
| 20merA | 9 | 9 | `results/ground_folds/20merA/` |
| 20merB | 10 | 10 | `results/ground_folds/20merB/` |
| 24mer | 9 | 9 | `results/ground_folds/24mer/` |
| 25mer | 8 | 8 | `results/ground_folds/25mer/` |
| 36mer | 14 | 14 | `results/ground_folds/36mer/` |
| 48mer | 23 | 23 | `results/ground_folds/48mer/` |
| 50mer | 21 | 21 | `results/ground_folds/50mer/` |
| 60mer | 36 | 36 | `results/ground_folds/60mer/` |
| 64mer | 42 | 42 | `results/ground_folds/64mer/` |
| 85mer | 53 | 53 | `results/ground_folds/85mer/` |
| 100merA | 48 | 48 | `results/ground_folds/100merA/` |
| 100merB | 50 | 50 | `results/ground_folds/100merB/` |

## QUBO Payload Policy

The final QUBO definitions are tied to every reported command and fold through
`results/qubos/final_qubo_manifest.csv`. The normal Git checkout stores the
manifest and the exporter, not the generated JSON payloads, because the final
QUBO set is about 2.5 GB. The experiment wrappers regenerate the files under
the listed `runs/.../qubos/` paths. A public archival release can additionally
attach the generated JSON payloads as GitHub Release assets or store them with
Git LFS.

## Excluded Generated Files

Full generated QUBO JSONs, worker logs, and full run folders are not stored in
the repository because they are large and can be regenerated. The `.gitignore`
keeps those outputs out of version control while explicitly allowing the curated
ground-fold JSON and plot artifacts.




