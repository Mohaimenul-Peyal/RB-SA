# GitHub Publication Checklist

Use this checklist before publishing the artifact repository.

## Repository Root

This folder should be published as its own repository root:

```text
rb-full-hp-folding-repro/
```

Do not publish the parent research workspace. If this folder is currently inside
a larger Git repository, copy it to a clean location or initialize a separate Git
repository from inside `rb-full-hp-folding-repro/`.

## Required Public Files

Before the first public release, confirm that the repository contains:

- `src/`: final solver, QUBO exporter, parallel runner, and plotting utility;
- `experiments/`: wrappers that regenerate the reported QUBOs and runs;
- `benchmarks/istrail_complete_suite.csv`: full benchmark index from 20merA through 100merB;
- `configs/benchmarks.csv`: Istrail sequences and targets;
- `results/qubos/final_qubo_manifest.csv`: final QUBO settings and paths;
- `results/ground_folds/`: decoded final folds, plots, metadata, and commands;
- `results/commands/final_run_commands.csv`: command snapshots;
- `results/final_benchmark_results.csv`: final validation table;
- `docs/reproduce_final_results.md`: command-level reproduction guide.

## Large Artifacts

The normal Git checkout should not include the full generated QUBO JSON set or
full worker-output directories. The final QUBO payloads are about 2.5 GB in
total. Keep the manifest in Git and provide the payloads only if needed through
one of these routes:

- regenerate them with the experiment wrappers;
- attach them to a GitHub Release;
- store them with Git LFS using the manifest paths.

## License And Citation

The repository uses the MIT License in `LICENSE`. Before public release, update
`CITATION.cff` with the final GitHub URL and, when available, the manuscript DOI
or preprint URL.

## Pre-Release Checks

Run these checks from the repository root:

```powershell
pip install -r requirements.txt
.\examples\run_20merA_smoke.ps1
python -m py_compile src\HP_export_manybody_ising.py src\HP_ssa_manybody_ising_cpu_numba_saw_pivot_fix7_contactguided.py src\HP_ssa_manybody_parallel_runner_cpu_fast_passthru.py src\plot_best_fold_best_overall.py
```

Then inspect:

```text
results/final_benchmark_results.csv
results/ground_folds/index.csv
results/qubos/final_qubo_manifest.csv
```
