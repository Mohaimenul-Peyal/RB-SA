# Istrail 100mer L=13 Fix7 Target Refinement

This experiment targets only `100merA` and `100merB`. It exports L=13 pairwise-only QUBOs, remaps the best RB-full L=12 warm starts into L=13, and runs 16-worker fix7 refinement.
No contact-native warm starts are used.

## Summary

| Benchmark | Target | Best | Target hit | Runtime s to selected best | Attempted runtime s | Hit rate | Workers | Trials | Steps | BBox | Notes |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---|---|
| 100merA | 48 | 48 | yes | 7285.698 | 7285.698 | 0.0526 | 16 | 768 | 3000000 | 13x10 | Target reached by L=13 RB-full refinement |
| 100merB | 50 | 50 | yes | 6698.349 | 6698.349 | 0.0667 | 16 | 768 | 3000000 | 13x11 | Target reached by L=13 RB-full refinement |

## Iterations

| Benchmark | Iter | Label | Warm contacts | Runtime s | Best | Trials | Target hit | Command file |
|---|---:|---|---:|---:|---:|---:|:---:|---|
| 100merA | 1 | `L13_hp4_warm512_2p5M_16w` | 47 | 4172.403 | 47 | 512 | no | `runs\100merA\local\Final_run\fix7_istrail_100mer_L13_target_refine\iter01_L13_hp4_warm512_2p5M_16w\command.txt` |
| 100merA | 2 | `L13_hp4_warm768_3M_16w_retry` | 47 | 3113.296 | 48 | 19 | yes | `runs\100merA\local\Final_run\fix7_istrail_100mer_L13_target_refine\iter02_L13_hp4_warm768_3M_16w_retry\command.txt` |
| 100merB | 1 | `L13_hp4_warm512_2p5M_16w` | 49 | 4164.937 | 49 | 512 | no | `runs\100merB\local\Final_run\fix7_istrail_100mer_L13_target_refine\iter01_L13_hp4_warm512_2p5M_16w\command.txt` |
| 100merB | 2 | `L13_hp4_warm768_3M_16w_retry` | 49 | 2533.412 | 50 | 15 | yes | `runs\100merB\local\Final_run\fix7_istrail_100mer_L13_target_refine\iter02_L13_hp4_warm768_3M_16w_retry\command.txt` |




