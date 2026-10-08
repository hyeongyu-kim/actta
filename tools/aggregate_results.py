"""Summarize full benchmark runs; reject incomplete, duplicate, or mixed settings."""

import argparse
import json
from pathlib import Path
import statistics


CORRUPTIONS = {"gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
               "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
               "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"}


def summarize(paths):
    groups = {}
    for path in paths:
        row = json.loads(Path(path).read_text())
        if not row.get("complete") or row.get("scope") != "full_benchmark":
            raise ValueError(f"Not a completed full benchmark: {path}")
        metrics = row.get("per_corruption", [])
        if (len(metrics) != 15 or {m["corruption"] for m in metrics} != CORRUPTIONS
                or any(m["samples"] != 10000 or not 0 <= m["error_percent"] <= 100 for m in metrics)):
            raise ValueError(f"Missing full corruption/sample coverage: {path}")
        if abs(statistics.mean(m["error_percent"] for m in metrics) - row["mean_error_percent"]) > 1e-6:
            raise ValueError(f"Mean does not match per-corruption errors: {path}")
        key = (row["dataset"], row["architecture"], row["method"], row["batch_size"],
               row["lr"], row["checkpoint_sha256"], row["setting"], row["severity"],
               row.get("shuffle"), row.get("legacy_rng_preamble"),
               json.dumps(row["activation_manifest"], sort_keys=True),
               json.dumps(row["config"], sort_keys=True))
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, runs in groups.items():
        seeds = [r["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Repeated seed in {key[:3]}: {seeds}")
        errors = [r["mean_error_percent"] for r in runs]
        summaries.append({"dataset": key[0], "architecture": key[1], "method": key[2],
                          "batch_size": key[3], "lr": key[4], "seeds": sorted(seeds),
                          "n_runs": len(runs), "mean_error_percent": statistics.mean(errors),
                          "std_population_percent": statistics.pstdev(errors),
                          "std_sample_percent": statistics.stdev(errors) if len(errors) > 1 else None,
                          "three_run_coverage": len(runs) == 3,
                          "note": "Paper standard-deviation convention is unspecified; both supplied."})
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(summarize(args.results), indent=2) + "\n")
