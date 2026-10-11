# Benchmark code

Run benchmark commands from this directory. Setup and checkpoint downloads are
in [Setup and execution](../docs/REPRODUCIBILITY.md).

| Path | Contents |
|---|---|
| `actta/` | Activation functions, adapters, models, and data loaders |
| `configs/` | CIFAR-C and ImageNet-C configurations |
| `evaluate*.py` | Evaluation entry points |
| `scripts/`, `tools/` | Experiment suites, summaries, and plots |
| `results/` | Recorded results and figures |
| `tests/` | Implementation and launcher tests |
| `reference/`, `vendor/` | Original implementations and upstream model code |
| `requirements*.txt`, `environment-conda-linux-64.lock.txt` | Environment specifications |

```bash
python tools/run_suite.py --help
```
