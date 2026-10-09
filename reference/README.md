# Reference implementations

Original research source for adaptation objectives and architecture settings.
Use the repository-root evaluators and
[reproduction commands](../docs/REPRODUCIBILITY.md) to run Source, TENT,
and AcTTA_TENT.

- `methods/`: baseline/AcTTA pairs for TENT, EATA/ETA, SAR, ROID, CMF, and DeYO.
- `models/gs_model.py`: GS2 activation and CIFAR/stage selectors.
- `architecture_audit/`: launch commands, activation/layer selection,
  optimizer setup, and ViT configurations.
- `depth_sweep_order/`: dataset/loader and main/depth evaluators with
  `shuffle=True` and `shuffle=False` input profiles.
- `data/`: RobustBench 5,000-image ImageNet list and class map.
- `reporting_v1/`: original reporting-workflow plotter.

See [architecture settings](../docs/ARCHITECTURE.md) for activation boundaries,
normalization selection, and optimizer profiles.
