# Result files

This directory contains 34 complete full evaluations and their reporting artifacts.
Read [the measured report](../docs/RESULTS.md) and
[configuration distinctions](../docs/ARCHITECTURE.md) before comparing numbers.

| Profile | Raw JSON directory | Full runs | Summary |
|---|---|---:|---|
| CIFAR-C, shuffled | `gpu2_runs/` | 12 | `gpu2_reproduction_summary.csv` / `.json` |
| ImageNet architecture/settings, shuffled | `architecture_runs/` | 12 | `architecture_audit_summary.csv` / `.json` |
| ImageNet depth, original fixed order | `depth_sweep_runs/` | 10 | `depth_sweep_summary.csv` / `.json` |

Each raw JSON records the complete resolved configuration, checkpoint hash,
trainable parameters, per-corruption errors, image order hashes, environment,
and full-versus-smoke scope. ImageNet additionally records image-content hashes.
Original raw records are unchanged. Physical GPU IDs 2/3 are provenance, not
requirements for a different machine.

The `*_per_corruption.csv` files provide corruption-level errors. Layer manifests
list actual selected activation paths and widths. The depth figure is available
as PNG, PDF, and SVG. Source manifests identify exact executed files; the
publication manifest includes SHA-256 checksums for this public subset.
`reproduction_ledger.json` indexes all 34 included runs and their profiles.

`paper_targets.csv` transcribes paper targets for context, including configurations
outside this subset's measured scope. Architecture summary Table 4 columns are
retained as numerical references only; use the ordered depth summary for the
protocol-specific comparison. Single-seed results do not verify three-seed means.

The original parity and input-order records are described in
[VALIDATION.md](../docs/VALIDATION.md). Recompute summaries with
[the documented commands](../docs/REPRODUCIBILITY.md#output-checks-and-aggregation).
