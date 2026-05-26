# Ground Fold Artifacts

This directory contains the curated decoded folds used for the final RB-full
target-contact claims.

Use `index.csv` as the authoritative inventory. Each benchmark folder contains:

- `best_overall.json`: decoded self-avoiding fold used for the reported result;
- `warm_start_best.json`: final-stage warm seed, only for warm-started rows;
- `fold.png`: quick-view fold plot;
- `fold.pdf`: publication-quality fold plot;
- `metadata.json`: QUBO metadata, run settings, and command array;
- `summary.csv`: one-row run summary;
- `command.txt`: repository-relative command snapshot.
- `command_relative.txt`: same command retained under an explicit name.

Large QUBO files are not stored here. Recreate them with the experiment wrappers
documented in `../../docs/reproduce_final_results.md`.




