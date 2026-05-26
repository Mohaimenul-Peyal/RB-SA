# Complete Istrail 2D HP Benchmark Suite

This directory groups the benchmark definitions used in the manuscript.
The covered set is:

```text
20merA, 20merB, 24mer, 25mer, 36mer, 48mer, 50mer, 60mer,
64mer, 85mer, 100merA, 100merB
```

`istrail_complete_suite.csv` records the sequence, chain length, target contact
count, target HP energy, final QUBO manifest path, curated fold folder, and
final command file for each benchmark.

The runnable configuration used by the experiment wrappers is also kept in
`../configs/benchmarks.csv`. The two files intentionally overlap: this directory
is the human-readable benchmark index, while `configs/benchmarks.csv` is the
machine-readable configuration consumed by the reproduction scripts.

