"""Report the supplied current-suite and historical completed measurements."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


GROUPS = {
    "gpu2_runs": "current_reproduction_suite",
    "architecture_runs": "current_reproduction_suite",
    "depth_sweep_runs": "current_reproduction_suite",
    "imagenet_runs": "historical_earlier_runner",
    "depth_sweep_shuffled_pilot_runs": "historical_shuffled_depth_pilot",
}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(results_root, output):
    records, ledger, historical = {}, [], []
    for group, profile in GROUPS.items():
        for path in sorted((results_root / group).glob("*.json")):
            row = json.loads(path.read_text())
            metrics = row["per_corruption"]
            expected_samples = 10000 if group == "gpu2_runs" else 5000
            if not row.get("complete") or row.get("scope") != "full_benchmark":
                raise ValueError(f"Incomplete result: {path}")
            if (len(metrics) != 15 or len({m["corruption"] for m in metrics}) != 15
                    or any(m["samples"] != expected_samples for m in metrics)):
                raise ValueError(f"Incomplete corruption coverage: {path}")
            errors = [m["error_percent"] * m["samples"] / 100. for m in metrics]
            if any(abs(e - round(e)) > 1e-7 for e in errors):
                raise ValueError(f"Error percentage is inconsistent with integer counts: {path}")
            predictions = sum(m["samples"] for m in metrics)
            if abs(sum(round(e) for e in errors) / predictions * 100.
                   - row["mean_error_percent"]) > 1e-9:
                raise ValueError(f"Mean error mismatch: {path}")
            relative = f"results/{group}/{path.name}"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            records[(group, path.stem)] = row
            ledger.append({"path": relative, "profile": group, "reporting_role": profile,
                           "architecture": row["architecture"], "method": row["method"],
                           "seed": row["seed"], "batch_size": row["batch_size"],
                           "shuffle": row["shuffle"], "error_percent": row["mean_error_percent"],
                           "predictions": predictions, "sha256": digest})
            if profile.startswith("historical"):
                optimizer = row["method"] != "source" and row.get("optimizer_applied", True)
                historical.append({"run": path.stem, "profile": profile,
                                   "method": row["method"], "seed": row["seed"],
                                   "batch_size": row["batch_size"],
                                   "nominal_depth_percent": row.get("nominal_depth_percent"),
                                   "selected_activations": len(row["activation_manifest"]),
                                   "lr": row["lr"] if optimizer else None,
                                   "nesterov": (False if group == "imagenet_runs"
                                                else row["sgd_nesterov"]) if optimizer else None,
                                   "shuffle": row["shuffle"],
                                   "error_percent": row["mean_error_percent"],
                                   "predictions": predictions, "result_path": relative,
                                   "result_sha256": digest})

    def cifar_runs(config, method, seeds):
        return [("gpu2_runs", f"{config}_{method}_seed{seed}") for seed in seeds]

    cases = [
        ("CIFAR-10-C", "WRN-28-10", 128,
         cifar_runs("cifar10_wrn28_bs128", "source", [1]),
         cifar_runs("cifar10_wrn28_bs128", "tent", [1, 2, 3]),
         cifar_runs("cifar10_wrn28_bs128", "actta_tent", [1, 2, 3]), "first WRN stage"),
        ("CIFAR-100-C", "WRN-40-2", 128,
         cifar_runs("cifar100_wrn40_bs128", "source", [1]),
         cifar_runs("cifar100_wrn40_bs128", "tent", [1]),
         cifar_runs("cifar100_wrn40_bs128", "actta_tent", [1]), "first two WRN stages"),
        ("CIFAR-100-C", "WRN-40-2", 4, [],
         cifar_runs("cifar100_wrn40_bs4", "tent", [1]),
         cifar_runs("cifar100_wrn40_bs4", "actta_tent", [1]), "first two WRN stages"),
        ("ImageNet-C", "ResNet-50", 128,
         [("imagenet_runs", "imagenet_resnet50_bs128_source_seed1")],
         [("architecture_runs", "resnet50_nesterov_tent_seed1")],
         [("architecture_runs", "resnet50_calls25_main_lr_actta_seed1")],
         "first 25/49 ReLU calls; frozen BN affine"),
        ("ImageNet-C", "ViT-B/16", 128,
         [("architecture_runs", "vit_b16_source_seed1")],
         [("architecture_runs", "vit_b16_tent_seed1")],
         [("architecture_runs", "vit_b16_blocks6_joint_ln_actta_seed1")],
         "first 6/12 MLP GELUs and all LN; tanh GELU"),
    ]
    overview = []
    for dataset, backbone, batch, sources, tents, acttas, selection in cases:
        item = {"dataset": dataset, "backbone": backbone, "batch_size": batch,
                "shuffle": True, "actta_selection": selection}
        for method, keys in [("source", sources), ("tent", tents), ("actta_tent", acttas)]:
            group = [records[key] for key in keys]
            values = [r["mean_error_percent"] for r in group]
            if any(r["method"] != method or r["batch_size"] != batch
                   or not r["shuffle"] for r in group):
                raise ValueError(f"Overview profile mismatch: {dataset}/{backbone}/{method}")
            item[f"{method}_error_percent"] = statistics.mean(values) if values else None
            item[f"{method}_seeds"] = ",".join(str(r["seed"]) for r in group)
            item[f"{method}_std_population_percent"] = statistics.pstdev(values) if len(values) > 1 else None
            item[f"{method}_std_sample_percent"] = statistics.stdev(values) if len(values) > 1 else None
            item[f"{method}_lr"] = group[0]["lr"] if method != "source" and group else None
            item[f"{method}_result_paths"] = ";".join(f"results/{g}/{n}.json" for g, n in keys)
        item["source_provenance"] = ("earlier Source evaluator; no optimizer"
                                     if sources and sources[0][0] == "imagenet_runs"
                                     else "current suite" if sources else "not measured at this batch")
        overview.append(item)
    current = sum(r["reporting_role"] == "current_reproduction_suite" for r in ledger)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "benchmark_summary.csv", overview)
    write_csv(output / "historical_results.csv", historical)
    (output / "reproduction_ledger.json").write_text(json.dumps({
        "public_full_runs": len(ledger), "current_reproduction_suite_full_runs": current,
        "historical_full_runs": len(historical),
        "total_predictions": sum(r["predictions"] for r in ledger),
        "original_local_full_runs_preserved": 42, "excluded_historical_full_runs": 0,
        "final_paper_configuration_certified": False, "runs": ledger,
    }, indent=2) + "\n")
    print(f"Reported {len(ledger)} full runs: {current} current-suite and {len(historical)} historical.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=root / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.results_root, args.output)
