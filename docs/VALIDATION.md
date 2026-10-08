# Validation and scope

This public release contains 42 complete, previously executed evaluations:
34 current reproduction-suite runs and eight historical measurements.
Inference sources, measured YAMLs, and raw result JSONs are copied byte for
byte from the preserved release. The original local archive also retains all 42
complete runs. Earlier runner/stream profiles are documented in the results
appendix. No incomplete run is included. The plotting/reporting changes do not
alter measured errors or inference sources.

## Checks

- The 16 unit tests passed again in the publication checkout. They check
  original activation output/gradient parity, selection boundaries, shared
  ReLU splitting, trainable parameter selection, optimizer updates/resets,
  zero-depth BN behavior, and rejection of incomplete results.
- All 42 full records include all 15 corruptions and the required number of
  images: 10,000 each for CIFAR and 5,000 each for ImageNet. Means are checked
  against per-corruption errors. Independent summation checks confirm their
  error counts and prediction totals.
- The original 5,000 ImageNet paths/labels match the recovered dataset
  selection. Ordered runs preserve indices 0..4999 for every corruption.
  The paired order/content hashes match across all depths and both backbones.
- Before the full depth runs, both architectures' 0%/100% endpoints passed
  eight-image execution checks on real data.
- The cleaned 100%-depth RN50 and ViT implementations each matched the original
  full model bitwise for two CPU entropy steps: logits, every trainable
  gradient, and every SGD update. The only constructor override redirected
  the original hardcoded CUDA tensor allocations to CPU. Inputs were two
  synthetic RGB tensors, not the full benchmark.
- The six-block joint-LN ViT candidate also matched the original implementation
  bitwise for two CPU steps. Official pretrained loading and strict offline
  loading produced identical state tensors and post-construction RNG state.

The saved parity reports are:

- [RN50 100%](../results/depth_sweep_original_resnet_step_parity.json)
- [ViT 100%, frozen LN](../results/depth_sweep_original_vit_frozen_step_parity.json)
- [ViT six-block joint LN](../results/architecture_original_vit_step_parity.json)
- [ViT checkpoint loading](../results/architecture_pretrained_loading_parity.json)
- [Original input-order provenance](../results/depth_sweep_input_order_provenance.json)
- [ViT zero-depth versus Source](../results/depth_sweep_vit_zero_source_check.json)

Parity reports record completed checks, rather than claiming that unit tests
download and rerun those full-model experiments.

## Source and result integrity

`results/gpu2_executed_source_manifest.json`,
`results/architecture_executed_source_manifest.json`, and
`results/depth_sweep_executed_source_manifest.json` preserve the source hashes
captured before evaluation. `results/depth_sweep_workflow_manifest.json`
captures the initial reporting workflow and remains unchanged. Its original
plotter is preserved under `reference/reporting_v1/tools/plot_depth_sweep.py`;
`results/reporting_manifest.json` resolves that historical path and records
the current measurement-only reporter. Inference-source manifest paths still
resolve directly to the active files with their original hashes.
`results/publication_manifest.json` records all
included code/config/reference/result/doc files, excluding itself and the
unchanged website.

`.gitattributes` preserves original line endings in the reproducibility files
so checkout conversion does not invalidate recorded SHA-256 checksums. Original
reference/vendor whitespace is retained as part of byte-for-byte source evidence.

Check the publication manifest from the repository root:

```bash
python tools/verify_publication.py
```

For numerical regeneration and figure export, use the commands in
[the reproduction guide](REPRODUCIBILITY.md#output-checks-and-aggregation).

## Limits

The release checks implementation consistency and the published measurements.
It does not certify that the recovered candidates are the original final paper
run manifests. ImageNet measurements are single-seed; only CIFAR-10-C
TENT/AcTTA have three-seed coverage. Original main/depth profiles differ in
input order, normalization adaptation, and LR. The exact and tanh GELUs differ
at zero initialization. Keep these distinctions when making fair comparisons.
