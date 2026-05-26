$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$seq = "HPHPPHHPHPPHPHHPPHPH"
$outdir = Join-Path $repo "runs\20merA\example_smoke"
$qubo = Join-Path $outdir "qubo_20merA_L6_pairwise.json"

New-Item -ItemType Directory -Force -Path $outdir | Out-Null

python (Join-Path $repo "src\HP_export_manybody_ising.py") `
  --seq $seq `
  --l_size 6 `
  --hp_reward -1.4 `
  --penalty_mode auto `
  --out $qubo

python (Join-Path $repo "src\HP_ssa_manybody_parallel_runner_cpu_fast_passthru.py") `
  --ising $qubo `
  --solver_script (Join-Path $repo "src\HP_ssa_manybody_ising_cpu_numba_saw_pivot_fix7_contactguided.py") `
  --total_trials 4 `
  --workers 4 `
  --steps 20000 `
  --t_init 8.0 `
  --t_final 0.0006 `
  --seed 910020001 `
  --device cpu `
  --dtype float64 `
  --move_mode residue `
  --block_size 0 `
  --reheat_every 35000 `
  --reheat_factor 2.4 `
  --outdir $outdir `
  --reseed_each_trial `
  --target_contacts 9 `
  --stop_on_target `
  -- `
  --pivot_prob 0.5 `
  --pivot_max_tail 0 `
  --pull_prob 0.08 `
  --reptation_prob 0.12 `
  --regrow_prob 0.05 `
  --frag_regrow_prob 0.16 `
  --regrow_max_len 10 `
  --init_mode seq_greedy `
  --archive_size 24 `
  --archive_min_hamming_frac 0.24 `
  --archive_contact_slack 5 `
  --contact_priority_best `
  --contact_check_every 50 `
  --contact_guided_accept `
  --contact_bias 0.4 `
  --contact_bias_final_frac 0.10 `
  --contact_paving_weight 0.0008 `
  --qubo_polish_frac 0.3

Write-Host "Smoke test output: $outdir"
