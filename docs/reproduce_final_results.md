# Reproducing the Final RB-Full Experiments

This document gives the command-level workflow used to recreate the final
RB-full benchmark evidence. Run all commands from the repository root.

## Setup

```powershell
pip install -r requirements.txt
```

Optional smoke test:

```powershell
.\examples\run_20merA_smoke.ps1
```

## Benchmark Definitions

All final benchmark sequences and target contact counts are in:

```text
configs/benchmarks.csv
```

The target HP energy is `-target_contacts`, because the final validation uses
the standard pairwise nonconsecutive H-H contact objective.

The exact QUBO export settings used for the final rows are listed in:

```text
results/qubos/final_qubo_manifest.csv
```

The manifest records the intended local QUBO JSON path, lattice size `L`,
variable count, hydrophobic reward, constraint penalties, and expected source
file size. The JSON payloads are generated under `runs/` by the wrappers below.
They are not part of the normal Git checkout because the final set is about
2.5 GB.

## Final Workflow

### 1. Fresh pairwise QUBO runs through 64mer

```powershell
python experiments\run_fix7_pairwise_fresh_qubo_selfwarm.py --benchmarks 20merA 20merB 24mer 25mer 36mer 48mer 50mer 60mer 64mer --resume
```

This exports pairwise-only QUBOs under:

```text
runs/<benchmark>/local/Final_run/fix7_pairwise_fresh_qubo/qubos/
```

and writes the run summary to:

```text
runs/fix7_pairwise_fresh_qubo_summary.csv
```

### 2. Runtime-reduced target-stop validation for 60mer and 64mer

```powershell
python experiments\run_fix7_pairwise_targetstop_runtime_reduction.py --benchmarks 60mer 64mer --resume
```

This keeps the pairwise-only QUBOs from step 1. For 60mer, it first creates an
in-directory 35-contact RB-full seed and then self-warms to the 36-contact
target. For 64mer, it uses target-aware stopping and reaches 42 contacts without
an external warm start.

### 3. Long Istrail rows

The 85mer, 100merA, and 100merB rows use staged RB-full-produced warm starts.
No contact-native search artifact is required.

```powershell
python experiments\run_fix7_istrail_long_scalability.py --benchmarks 85mer 100merA 100merB --resume
python experiments\run_fix7_istrail_long_warm_budget.py --benchmarks 85mer 100merA 100merB --resume
python experiments\run_fix7_istrail_long_16w_big_budget.py --benchmarks 85mer 100merA 100merB --resume
python experiments\run_fix7_istrail_100mer_L13_target_refine.py --benchmarks 100merA 100merB --resume
```

The first three scripts establish and refine L=12 long-chain warm states. The
last script exports L=13 pairwise QUBOs for 100merA and 100merB, remaps the
L=12 RB-full warm starts by coordinate, and reaches the reported targets.

## Expected Final Table

The expected final target-hit rows are recorded in:

```text
results/final_benchmark_results.csv
```

The curated final folds are recorded in:

```text
results/ground_folds/index.csv
```

## Command Snapshots

Every final fold folder contains both the original absolute command and a
repository-relative command:

```text
results/ground_folds/<benchmark>/command.txt
results/ground_folds/<benchmark>/command_relative.txt
```

The aggregate command table is:

```text
results/commands/final_run_commands.csv
```

For warm-started final rows, `command_relative.txt` points to the curated
`warm_start_best.json` artifact in the same benchmark folder. For full
end-to-end reproduction, prefer the experiment wrappers above because they
regenerate the warm-start chain in the same order used during the final study.

## QUBO JSON Payloads

The public repository tracks the QUBO manifest and the code that regenerates the
QUBO JSON files. After a wrapper exports a QUBO, its path should match the
`local_qubo` column in `results/qubos/final_qubo_manifest.csv`. For archival
publication where the generated JSON payloads must be downloadable without
rerunning the exporter, attach the files as GitHub Release assets or store them
with Git LFS using those same paths.

## Notes on Runtime

The runtime values in the CSV tables are wrapper wall-clock seconds on the
original local Ryzen 7 9700X workstation. They are not hardware-independent
complexity measures. Different CPUs, Python builds, and Numba cache states will
change the wall-clock values even when the same seed and command are used.




