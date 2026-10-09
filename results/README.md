# Result files

See [results](../docs/RESULTS.md) for error tables and
[architecture settings](../docs/ARCHITECTURE.md) for activation selection.

| Profile | Raw JSON directory | Summary |
|---|---|---|
| CIFAR-C, shuffled | `gpu2_runs/` | `gpu2_reproduction_summary.csv` / `.json` |
| ImageNet architecture/settings, shuffled | `architecture_runs/` | `architecture_audit_summary.csv` / `.json` |
| ImageNet depth, fixed image-ID order | `depth_sweep_runs/` | `depth_sweep_summary.csv` / `.json` |
| RN50 residual-output configuration, shuffled | `imagenet_runs/` | `historical_results.csv` |
| RN50 depth, shuffled | `depth_sweep_shuffled_pilot_runs/` | `historical_results.csv` |

Raw JSONs contain resolved configurations, checkpoint hashes, selected parameters,
per-corruption errors, input-order hashes, and environment details.
`benchmark_summary.csv` contains the TENT/AcTTA comparisons shown in the README.

The `*_per_corruption.csv` files contain corruption-level errors. Layer manifests
list selected activation paths and widths. The depth figure is available as
`depth_sweep_measured.png`, `.pdf`, and `.svg`.

`paper_targets.csv` contains paper reference values. Use `depth_sweep_summary.csv`
for the ordered depth comparison; architecture results use the shuffled stream.
`reproduction_ledger.json` indexes the recorded configurations.

Source manifests record executed-file hashes. `publication_manifest.json`
contains file checksums; `reporting_manifest.json` records plotting/reporting
files and the original plotter path under `reference/reporting_v1/`.

Regenerate summaries and figures with
[the documented commands](../docs/REPRODUCIBILITY.md#output-checks-and-aggregation).
