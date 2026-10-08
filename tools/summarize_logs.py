"""Audit legacy experiment logs without treating copies as additional seeds."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import yaml


CORRUPTIONS = {"gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
               "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
               "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"}
TIMESTAMP = re.compile(r"^\[\d\d/\d\d/\d\d ")
METRIC = re.compile(r"(\w+) error % \[([a-z_]+)([1-5])\]\[#samples=(\d+)\]: ([\d.]+)%")


def parse_log(path):
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    config = {}
    for i, line in enumerate(lines):
        if "[conf.py:" in line and "]: ADACONTRAST:" in line:
            stop = i + 1
            while stop < len(lines) and not TIMESTAMP.match(lines[stop]):
                stop += 1
            config = yaml.safe_load("ADACONTRAST:\n" + "\n".join(lines[i + 1:stop]))
            break
    metrics = [{"dataset": m[0], "corruption": m[1], "severity": int(m[2]),
                "samples": int(m[3]), "error_percent": float(m[4])}
               for m in METRIC.findall(text)]
    means = re.findall(r"mean error: ([\d.]+)%", text)
    final_mean = float(means[-1]) if means else None
    complete = (len(metrics) == 15 and {m["corruption"] for m in metrics} == CORRUPTIONS
                and all(m["severity"] == 5 and m["samples"] == 10000 for m in metrics)
                and final_mean is not None)
    calculated = sum(m["error_percent"] for m in metrics) / len(metrics) if metrics else None
    method = config.get("MODEL", {}).get("ADAPTATION", "unknown")
    row = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "paths": [str(path)], "method": method,
           "family": "AcTTA" if method.endswith("_gs") else "alternative_activation"
                     if any(k in method for k in ("pppp", "acon", "pau")) else "baseline",
           "dataset": config.get("CORRUPTION", {}).get("DATASET"),
           "architecture": config.get("MODEL", {}).get("ARCH"),
           "seed": config.get("RNG_SEED"), "setting": config.get("SETTING"),
           "batch_size": config.get("TEST", {}).get("BATCH_SIZE"),
           "lr": config.get("OPTIM", {}).get("LR"), "with_bn": config.get("with_bn"),
           "mean_error_percent": final_mean, "observed_corruptions": len(metrics),
           "complete": complete, "mean_consistent": complete and abs(final_mean - calculated) < .011,
           "metrics": metrics, "config": config}
    return row


def collect(roots):
    unique = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if path.is_file() and path.suffix in {".txt", ".log"}:
                row = parse_log(path)
                if row["sha256"] in unique:
                    unique[row["sha256"]]["paths"].append(str(path))
                else:
                    unique[row["sha256"]] = row
    return list(unique.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = collect(args.roots)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    fields = ["sha256", "method", "family", "dataset", "architecture", "seed", "setting",
              "batch_size", "lr", "with_bn", "mean_error_percent", "observed_corruptions",
              "complete", "mean_consistent"]
    with output.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    print(f"{len(rows)} distinct logs; {sum(r['complete'] for r in rows)} completed full CIFAR-C runs")


if __name__ == "__main__":
    main()
