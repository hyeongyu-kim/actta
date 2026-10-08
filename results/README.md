# Result files

This directory contains 42 complete full evaluations and their reporting artifacts:
34 current reproduction-suite runs and eight earlier historical measurements.
Read [the measured report](../docs/RESULTS.md) and
[configuration distinctions](../docs/ARCHITECTURE.md) before comparing numbers.

| Profile | Raw JSON directory | Full runs | Summary |
|---|---|---:|---|
| CIFAR-C, shuffled | `gpu2_runs/` | 12 | `gpu2_reproduction_summary.csv` / `.json` |
| ImageNet architecture/settings, shuffled | `architecture_runs/` | 12 | `architecture_audit_summary.csv` / `.json` |
| ImageNet depth, original fixed order | `depth_sweep_runs/` | 10 | `depth_sweep_summary.csv` / `.json` |
| Earlier ImageNet RN50, shuffled | `imagenet_runs/` | 3 | `historical_results.csv` |
| Earlier RN50 depth pilot, shuffled | `depth_sweep_shuffled_pilot_runs/` | 5 | `historical_results.csv` |

Each raw JSON records the complete resolved configuration, checkpoint hash,
trainable parameters, per-corruption errors, image order hashes, environment,
and full-versus-smoke scope. ImageNet additionally records image-content hashes.
Original raw records are unchanged. Physical GPU IDs 2/3 are provenance, not
requirements for a different machine.

`benchmark_summary.csv` gives the five main TENT/AcTTA comparisons shown in the
README. `historical_results.csv` reports the earlier runner/stream settings;
these records do not enter the current ordered depth curve.

The `*_per_corruption.csv` files provide corruption-level errors. Layer manifests
list actual selected activation paths and widths. The depth figure is available
as PNG, PDF, and SVG. Source manifests identify exact executed files; the
publication manifest includes SHA-256 checksums for this public subset.
`reproduction_ledger.json` indexes all 42 included runs and their profiles.

The figure shows the measured depth curve only. The original comparison plotter
is retained byte for byte under `reference/reporting_v1/`; the original workflow
manifest is unchanged. `reporting_manifest.json` identifies the current reporter
and resolves the historical plotter path when checking that original manifest.

`paper_targets.csv` transcribes paper targets for context, including configurations
outside this subset's measured scope. Architecture summary Table 4 columns are
retained as numerical references only; use the ordered depth summary for the
protocol-specific comparison. Single-seed results do not verify three-seed means.

The original parity and input-order records are described in
[VALIDATION.md](../docs/VALIDATION.md). Recompute summaries with
[the documented commands](../docs/REPRODUCIBILITY.md#output-checks-and-aggregation).
