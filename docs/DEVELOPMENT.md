# Repository layout and maintenance

| Path | Responsibility |
|---|---|
| `actta/activation.py` | GS2 activation and activation replacement |
| `actta/adaptation.py` | Entropy adaptation and optimizer/reset behavior |
| `actta/models.py`, `actta/data.py` | CIFAR models and corruption data |
| `actta/imagenet.py` | ImageNet normalization, data loading, and residual-output selection |
| `actta/architecture_audit.py` | ResNet call splitting, ViT loading, activation/LN selection |
| `actta/depth_sweep.py` | Depth counts and frozen-normalization depth adapter |
| `evaluate*.py` | Dataset-specific evaluation and raw JSON output |
| `configs/` | Complete experiment YAMLs; see the [configuration index](../configs/README.md) |
| `scripts/` | Experiment suites and their aggregation commands |
| `tools/` | Suite launcher, result summaries, plots, and development checks |
| `tests/` | Activation, architecture, adaptation, and launcher tests |
| `results/` | Recorded results; see the [file index](../results/README.md) |
| `reference/`, `vendor/` | Original source, ImageNet metadata, and upstream model implementations |
| `docs/` | Setup, settings, results, and maintenance notes |
| `index.html`, `static/` | Project website |

## Run a suite

`tools/run_suite.py` resolves input/output paths relative to the calling
directory and runs the existing suite script from the repository root. It uses
the current Python interpreter, or `ACTTA_PYTHON` when set, and requires a new
or empty output directory.

```bash
python tools/run_suite.py imagenet-depth \
  --data-root /path/to/ImageNet-C \
  --resnet-checkpoint /path/to/resnet50-0676ba61.pth \
  --vit-checkpoint /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  --output output/depth_sweep --gpu 3
```

Choose `imagenet-settings` for architecture comparisons, or `cifar` with
`--data-root DATA_ROOT --checkpoint-root CHECKPOINT_ROOT`. Add `--dry-run`
to inspect the command. For depth runs, `--depths 0 25 50 75 100` selects
points explicitly; otherwise the script uses `DEPTH_POINTS` or its defaults.
From another directory, invoke the launcher by its absolute path.
Existing shell commands remain available in
[the reproduction guide](REPRODUCIBILITY.md).

## Add or modify an experiment

1. Add a complete YAML with a distinct filename in `configs/`; use
   `configs/depth_sweep_ordered/` for the ordered depth protocol.
2. Run the relevant evaluator with `--corruption` and `--limit` for a small
   execution check, writing to a fresh directory under `output/`.
3. Add a suite entry in the corresponding `scripts/reproduce_*.sh` when the
   experiment should be included in that suite. Update the configuration index.
4. Update the relevant summarizer when adding a new result schema or evaluation
   protocol. Keep shuffled and ordered profiles separate.

`actta/depth_sweep.py` uses model loading and call splitting from
`actta/architecture_audit.py`, which uses ImageNet normalization/loading from
`actta/imagenet.py`. Changes to these shared helpers affect both ImageNet
evaluators. Changes to activation math affect every AcTTA configuration.

Use new output paths for follow-up experiments. Keep recorded raw JSONs in
`results/`, original snapshots in `reference/`, and upstream licenses with
their source files. Runtime locks define the ImageNet JPEG decoder and CUDA
libraries; see [environment details](FRESH_CLONE_VALIDATION.md).

When publishing changes, update file entries in `results/publication_manifest.json`.
Executed-source manifests belong to the recorded experiments; new measurements
should have their own source manifest.

Development commands are in [VALIDATION.md](VALIDATION.md).
