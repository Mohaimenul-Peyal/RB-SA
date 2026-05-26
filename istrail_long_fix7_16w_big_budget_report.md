# Istrail-Long Fix7 16-Worker Big-Budget Target Attempt

This run uses all 16 local workers and larger annealing budgets with RB-full-produced warm starts only.
It reuses the stronger pairwise-only `hp_reward=-3.0`, `L=12` QUBOs from the warm-budget experiment.

## Summary

| Benchmark | Target | Best | Target hit | Runtime s to selected best | Attempted runtime s | Hit rate | Workers | Trials | Steps | BBox | Notes |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---|---|
| 85mer | 53 | 53 | yes | 1124.237 | 1124.237 | 0.0159 | 16 | 256 | 2000000 | 11x10 | Target reached by 16-worker RB-full big-budget run |
| 100merA | 48 | 47 | no | 4432.962 | 9979.267 | 0.0000 | 16 | 512 | 2500000 | 11x12 | Target not reached; 16-worker RB-full big-budget attempt |
| 100merB | 50 | 49 | no | 4351.282 | 9783.078 | 0.0000 | 16 | 512 | 2500000 | 12x12 | Target not reached; 16-worker RB-full big-budget attempt |

## Iterations

| Benchmark | Iter | Label | Warm contacts | Runtime s | Best | Trials | Target hit | Command file |
|---|---:|---|---:|---:|---:|---:|:---:|---|
| 85mer | 1 | `warm256_2M_16w` | 52 | 1124.237 | 53 | 63 | yes | `runs\85mer\local\Final_run\fix7_istrail_long_16w_big_budget\iter01_warm256_2M_16w\command.txt` |
| 100merA | 1 | `warm256_2M_16w` | 46 | 1323.024 | 46 | 256 | no | `runs\100merA\local\Final_run\fix7_istrail_long_16w_big_budget\iter01_warm256_2M_16w\command.txt` |
| 100merA | 2 | `warm512_2p5M_16w_retry` | 46 | 3109.938 | 47 | 512 | no | `runs\100merA\local\Final_run\fix7_istrail_long_16w_big_budget\iter02_warm512_2p5M_16w_retry\command.txt` |
| 100merA | 3 | `warm768_3M_16w_final_refine` | 47 | 5546.305 | 47 | 768 | no | `runs\100merA\local\Final_run\fix7_istrail_long_16w_big_budget\iter03_warm768_3M_16w_final_refine\command.txt` |
| 100merB | 1 | `warm256_2M_16w` | 46 | 1280.477 | 47 | 256 | no | `runs\100merB\local\Final_run\fix7_istrail_long_16w_big_budget\iter01_warm256_2M_16w\command.txt` |
| 100merB | 2 | `warm512_2p5M_16w_retry` | 47 | 3070.805 | 49 | 512 | no | `runs\100merB\local\Final_run\fix7_istrail_long_16w_big_budget\iter02_warm512_2p5M_16w_retry\command.txt` |
| 100merB | 3 | `warm768_3M_16w_final_refine` | 49 | 5431.797 | 49 | 768 | no | `runs\100merB\local\Final_run\fix7_istrail_long_16w_big_budget\iter03_warm768_3M_16w_final_refine\command.txt` |




