# Original reference snapshots

These files preserve selected original research implementations for comparison.
They are not a second runnable package; their complete benchmark dependencies
are not vendored. Use the evaluators at the repository root for the measured
Source, TENT, and AcTTA_TENT configurations.

- `methods/`: baseline/AcTTA pairs for TENT, EATA/ETA, SAR, ROID, CMF, and DeYO.
  Full evaluations in this release cover the TENT objective only.
- `models/gs_model.py`: original GS2 activation and CIFAR/stage selectors.
- `architecture_audit/`: original launch commands, GS2/layer selection,
  optimizer setup, and ViT configurations that substantiate recovered options.
- `depth_sweep_order/`: original dataset/loader and main/depth evaluators,
  substantiating `shuffle=True` versus `shuffle=False` and image-ID ordering.
- `data/`: original RobustBench 5,000 ImageNet image IDs and class map.
- `reporting_v1/`: the unchanged initial comparison plotter, preserved for
  checking the original reporting-workflow hash after the figure was simplified.

Historical defaults, comments, or paths in these unchanged snapshots are not
instructions for the active evaluators. Main/depth candidates are distinguished
in [the architecture notes](../docs/ARCHITECTURE.md); portable commands are in
[the reproduction guide](../docs/REPRODUCIBILITY.md). Each file's hash is
recorded in the publication manifest, alongside the original executed-source
manifests in `results/`.
