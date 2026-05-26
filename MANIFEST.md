# Manifest

This manifest lists the main files in the RB-full HP folding reproducibility
package.

## Source

| File | Purpose |
|---|---|
| `src/HP_export_manybody_ising.py` | QUBO/k-local Ising exporter. |
| `src/HP_ssa_manybody_ising_cpu_numba_saw_pivot_fix7_contactguided.py` | Final RB-full solver. |
| `src/HP_ssa_manybody_parallel_runner_cpu_fast_passthru.py` | Parallel CPU runner with target-aware stopping support. |
| `src/plot_best_fold_best_overall.py` | Fold plotting utility. |

## Experiment Wrappers

| File | Purpose |
|---|---|
| `experiments/run_fix7_pairwise_fresh_qubo_selfwarm.py` | Fresh pairwise QUBO export and RB-full runs for 20merA through 64mer. |
| `experiments/run_fix7_pairwise_targetstop_runtime_reduction.py` | Target-stop runtime-reduction schedule for 60mer and 64mer. |
| `experiments/run_fix7_istrail_long_scalability.py` | Initial long-chain scalability/warm-state stage. |
| `experiments/run_fix7_istrail_long_warm_budget.py` | Long-chain warm-budget refinement stage. |
| `experiments/run_fix7_istrail_long_16w_big_budget.py` | 16-worker long-chain refinement stage. |
| `experiments/run_fix7_istrail_100mer_L13_target_refine.py` | L=13 target refinement for 100merA and 100merB. |

## Configuration

| File | Purpose |
|---|---|
| `configs/benchmarks.csv` | Istrail benchmark sequences, targets, and QUBO output locations from 20merA through 100merB. |

## Documentation

| File | Purpose |
|---|---|
| `README.md` | Main reproduction guide. |
| `docs/reproduce_final_results.md` | Step-by-step final benchmark reproduction commands. |
| `docs/artifact_index.md` | Inventory of included artifacts and fold folders. |

## Results

| File | Purpose |
|---|---|
| `results/final_benchmark_results.csv` | Primary final RB-full validation table from 20merA through 100merB. |
| `results/final_benchmark_results.md` | Markdown version of the primary final validation table. |
| `results/ground_folds/index.csv` | Inventory of curated final fold artifacts. |
| `results/ground_folds/<benchmark>/best_overall.json` | Final decoded fold for each target-hit benchmark. |
| `results/ground_folds/<benchmark>/fold.png` | PNG fold plot for each target-hit benchmark. |
| `results/ground_folds/<benchmark>/fold.pdf` | PDF fold plot for each target-hit benchmark. |
| `results/ground_folds/<benchmark>/command_relative.txt` | Repository-relative command snapshot for the final row. |
| `results/commands/final_run_commands.csv` | Aggregate final command table. |
| `results/tables/` | Manuscript-facing tables copied into the public artifact. |
| `results/figure_tables/` | Figure-source tables copied into the public artifact. |

## Examples

| File | Purpose |
|---|---|
| `examples/run_20merA_smoke.ps1` | Windows PowerShell smoke test. |
| `examples/run_20merA_smoke.sh` | Bash smoke test. |

## GitHub Metadata

| File | Purpose |
|---|---|
| `.gitignore` | Excludes generated runs, large QUBOs, logs, and local environments while allowing curated fold artifacts. |
| `.github/workflows/syntax-check.yml` | Optional GitHub Actions workflow for Python syntax checks. |
| `LICENSE_NOT_SELECTED.md` | Placeholder reminding the author to choose a public license before publishing. |




