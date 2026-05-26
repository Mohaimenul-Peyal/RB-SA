# RB-Full HP Folding Reproducibility Package

This repository contains the code and curated artifacts for the final
Residue-Block QUBO-Guided Simulated Annealing (RB-full) experiments on the
Istrail 2D HP folding benchmark suite.

The package is prepared as a public GitHub-ready artifact. It includes the
active solver, QUBO exporter, parallel runner, final experiment wrappers,
benchmark definitions, QUBO manifests, result tables, exact command records,
representative output folders, and decoded ground-fold plots for the validated
benchmark rows.

## What Is Included

| Path | Purpose |
|---|---|
| `src/HP_export_manybody_ising.py` | Pairwise HP QUBO and Ising exporter used for the final benchmark runs. |
| `src/HP_ssa_manybody_ising_cpu_numba_saw_pivot_fix7_contactguided.py` | Final RB-full solver used for the reported experiments. |
| `src/HP_ssa_manybody_parallel_runner_cpu_fast_passthru.py` | CPU multi-worker runner with target-aware stopping support. |
| `src/plot_best_fold_best_overall.py` | Plot utility for decoded `best_overall.json` folds. |
| `experiments/` | Reproduction wrappers for fresh pairwise QUBOs, target-stop runs, and long-chain refinement. |
| `benchmarks/istrail_complete_suite.csv` | Human-readable index of all benchmark chains from 20merA through 100merB. |
| `configs/benchmarks.csv` | Istrail benchmark sequences, targets, and expected QUBO locations. |
| `results/qubos/final_qubo_manifest.csv` | Final QUBO path, lattice size, objective weights, and expected file-size manifest. |
| `results/final_benchmark_results.csv` | Primary final RB-full validation table from 20merA through 100merB. |
| `results/ground_folds/` | Curated target-hit fold JSONs, PNG/PDF plots, metadata, and commands. |
| `results/commands/final_run_commands.csv` | Per-benchmark command snapshots using repository-relative paths. |
| `docs/` | Concise reproduction and artifact-index notes. |
| `examples/` | Small smoke-test scripts. |
| `CITATION.cff` | Citation metadata placeholder; update the repository URL before public release. |

Some source and wrapper filenames retain the historical internal label `fix7`.
In the manuscript and public documentation, that final configuration is referred
to as RB-full.

The normal Git checkout does not bundle the full generated QUBO JSON payloads,
worker logs, or complete run folders. The final QUBO definitions are captured in
`results/qubos/final_qubo_manifest.csv` and are regenerated under `runs/` by the
experiment wrappers. The generated QUBO JSON set is about 2.5 GB; if a release
needs the payloads themselves, attach them as GitHub Release assets or store
them with Git LFS using the same relative paths listed in the manifest. The
curated `results/ground_folds/` directory keeps the files needed to inspect and
verify the final decoded folds.

## Environment

Before publishing, use `rb-full-hp-folding-repro/` as the repository root. The
publication checklist is in `docs/github_publication_checklist.md`.

Tested on Windows with Python from Anaconda and an AMD Ryzen 7 9700X CPU.
Python 3.10 or newer is recommended.

```powershell
pip install -r requirements.txt
```

Required packages:

- `numpy`
- `numba`
- `matplotlib`

The final RB-full path is CPU/Numba based. Some command-line arguments are
retained for interface compatibility, such as `--device cpu`; GPU execution is
not required for the reported runs.

## Quick Smoke Test

Run a small 20merA check to verify the exporter, runner, solver, and plotting
paths:

```powershell
.\examples\run_20merA_smoke.ps1
```

The smoke test is not a paper-quality timing run. It is only a local
installation check.

## Reproducing The Final Benchmark Workflow

The standard final workflow uses pairwise-only QUBOs, because the Istrail HP
benchmark target is the nonconsecutive H-H contact count.

Run from the repository root.

The complete benchmark set is grouped in:

```text
benchmarks/istrail_complete_suite.csv
```

The exact final QUBO paths, lattice sizes, hydrophobic rewards, penalties, and
expected source sizes are listed in:

```text
results/qubos/final_qubo_manifest.csv
```

### 1. Fresh pairwise QUBOs and RB-full runs through 64mer

```powershell
python experiments\run_fix7_pairwise_fresh_qubo_selfwarm.py --benchmarks 20merA 20merB 24mer 25mer 36mer 48mer 50mer 60mer 64mer --resume
```

### 2. Target-stop runtime-reduced 60mer and 64mer validation

```powershell
python experiments\run_fix7_pairwise_targetstop_runtime_reduction.py --benchmarks 60mer 64mer --resume
```

### 3. Long Istrail rows: 85mer, 100merA, and 100merB

These scripts use only RB-full-produced in-directory warm starts.

```powershell
python experiments\run_fix7_istrail_long_scalability.py --benchmarks 85mer 100merA 100merB --resume
python experiments\run_fix7_istrail_long_warm_budget.py --benchmarks 85mer 100merA 100merB --resume
python experiments\run_fix7_istrail_long_16w_big_budget.py --benchmarks 85mer 100merA 100merB --resume
python experiments\run_fix7_istrail_100mer_L13_target_refine.py --benchmarks 100merA 100merB --resume
```

For a clean rebuild of QUBO files, add `--force-export` to the scripts that
support it. The long-chain scripts are computationally expensive; inspect the
reported budgets in `results/final_benchmark_results.csv` before running them
on shared machines.

## Inspecting Final Ground Folds

The final decoded folds are included here:

```text
results/ground_folds/<benchmark>/
```

Each benchmark folder contains:

- `best_overall.json`: decoded target-hit fold used for the paper result;
- `fold.png` and `fold.pdf`: ground-fold visualization;
- `metadata.json`: run configuration and QUBO metadata;
- `summary.csv`: one-row summary for that run;
- `command.txt`: repository-relative command snapshot;
- `command_relative.txt`: same command retained under an explicit name;
- `warm_start_best.json`, when the final run used a self-warm seed.

The complete index is `results/ground_folds/index.csv`.

## Final Result Tables

| File | Meaning |
|---|---|
| `results/final_benchmark_results.csv` | Primary final RB-full target recovery table. |
| `results/qubos/final_qubo_manifest.csv` | Final QUBO export manifest and size policy. |
| `results/tables/table2_ground_state_validation_standard.csv` | Manuscript ground-state validation context with VA comparison rows. |
| `results/tables/table5_runtime_scaling_summary.csv` | Runtime-scaling context table. |
| `results/tables/table31_istrail_literature_energy_comparison.csv` | Literature energy comparison table. |
| `results/commands/final_run_commands.csv` | Repository-relative command snapshots for all final fold rows. |

## License

This reproducibility package is released under the MIT License. See `LICENSE`.

## Method Summary

RB-full combines a residue-site HP QUBO/Ising objective with feasible-chain
annealing moves:

- sequence-aware self-avoiding-walk initialization;
- residue-block local proposals;
- pivot, pull-like, reptation, endpoint regrowth, and internal fragment regrowth
  moves;
- diversity-aware archive and self-warm refinement;
- contact-guided exploration and contact-band paving;
- final pure-QUBO polish;
- target-aware early stopping in the parallel runner.

The method contribution is the integration of these HP-specific feasible-chain
mechanisms inside a reproducible residue-block QUBO-guided annealing workflow.
