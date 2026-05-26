# Final QUBO Artifacts

The manuscript reproducibility section states that each reported run is tied to
an exported QUBO/Ising file. In this public repository, the exact final QUBO
definitions are represented by `final_qubo_manifest.csv` and by the experiment
wrappers that regenerate the JSON payloads under the paths listed in that
manifest.

The full generated JSON payloads are large: the final 48mer, 50mer, 60mer,
64mer, 85mer, 100merA, and 100merB QUBOs range from roughly 114 MB to 704 MB
each, and the complete final set is about 2.5 GB. They are therefore not part
of the normal Git checkout. This avoids making the public repository difficult
to clone while preserving the exact export recipe.

To recreate the final QUBO files, run the experiment wrappers documented in
`../../docs/reproduce_final_results.md`. The generated JSON files will be placed
under the `local_qubo` paths recorded in `final_qubo_manifest.csv`.

If an archival release requires the generated JSON payloads themselves, attach
them as GitHub Release assets or store them with Git LFS using the same relative
paths listed in the manifest.

