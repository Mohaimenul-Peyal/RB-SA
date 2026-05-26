# Table 2. Ground-State Validation On Standard And Extended Benchmarks

Full benchmark validation from `20merA` through `100merB`. RB-full uses the current pairwise-only target-stop and Istrail-long target-refinement evidence. VA uses the corrected full-loop local reference-code run with seed `0` and is available through `60mer`; no VA local run was executed for `64mer`, `85mer`, `100merA`, or `100merB`.

| Benchmark | N | Target contacts | Target HP energy | RB-full contacts | RB-full hit | RB-full runtime s | VA contacts | VA hit | VA runtime s | Interpretation |
|---|---:|---:|---:|---:|:---:|---:|---:|:---:|---:|---|
| 20merA | 20 | 9 | -9 | 9 | yes | 1.336 | 9 | yes | 965.509 | both reproduce target |
| 20merB | 20 | 10 | -10 | 10 | yes | 1.356 | 10 | yes | 1000.652 | both reproduce target |
| 24mer | 24 | 9 | -9 | 9 | yes | 1.466 | 9 | yes | 1321.199 | both reproduce target |
| 25mer | 25 | 8 | -8 | 8 | yes | 1.710 | 8 | yes | 1411.220 | both reproduce target |
| 36mer | 36 | 14 | -14 | 14 | yes | 3.376 | 14 | yes | 3295.744 | both reproduce target |
| 48mer | 48 | 23 | -23 | 23 | yes | 37.103 | 20 | no | 5512.697 | RB-full reproduces target; local VA seed-0 run does not |
| 50mer | 50 | 21 | -21 | 21 | yes | 24.862 | 20 | no | 5854.871 | RB-full reproduces target; local VA seed-0 corrected 10M run remains one contact below target |
| 60mer | 60 | 36 | -36 | 36 | yes | 746.805 | 35 | no | 35603.110 | RB-full reproduces target; local VA seed-0 run remains one contact below target |
| 64mer | 64 | 42 | -42 | 42 | yes | 282.110 | not run | not run | not run | RB-full extends the validated set beyond the reproduced VA benchmark range |
| 85mer | 85 | 53 | -53 | 53 | yes | 1124.237 | not run | not run | not run | RB-full reproduces the Istrail-long target with a 16-worker warm-budget run |
| 100merA | 100 | 48 | -48 | 48 | yes | 7285.698 | not run | not run | not run | RB-full reproduces the Istrail-long target after L=13 target refinement |
| 100merB | 100 | 50 | -50 | 50 | yes | 6698.349 | not run | not run | not run | RB-full reproduces the Istrail-long target after L=13 target refinement |

Primary sources: RB-full rows from `results/final_benchmark_results.csv`; VA rows from the local VA experiment summary retained in the paper workspace.




