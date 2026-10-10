# Development checks

Run from the repository root:

```bash
python tools/check_environment.py
python -m unittest discover -s tests -v
python tools/verify_publication.py
```

The unit tests cover activation behavior, layer selection, trainable parameters,
optimizer updates/resets, zero-depth behavior, and result aggregation. They
require no datasets or checkpoints. `verify_publication.py` checks file hashes
against `results/publication_manifest.json`.

Commands for regenerating summaries and figures are in
[the reproduction guide](REPRODUCIBILITY.md#output-checks-and-aggregation).
See [repository maintenance](DEVELOPMENT.md) for module responsibilities and
adding experiments.
