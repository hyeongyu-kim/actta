"""Validate architecture-audit results without merging distinct configurations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from actta.data import CORRUPTIONS


PAPER = {
    "resnet50": {"source": 82.03, "tent": 66.50, "actta_tent": 64.95},
    "vit_base_patch16_224.orig_in21k_ft_in1k":
        {"source": 62.10, "tent": 53.85, "actta_tent": 51.79},
}


def validate_result(result):
    if not result["complete"] or result["scope"] != "full_benchmark":
        raise ValueError("Incomplete or smoke result cannot enter the audit report")
    metrics = result["per_corruption"]
    if len(metrics) != 15 or {m["corruption"] for m in metrics} != set(CORRUPTIONS):
        raise ValueError("Expected all 15 unique corruptions")
    if any(m["samples"] != 5000 for m in metrics):
        raise ValueError("Expected 5,000 samples per corruption")
    if result["severity"] != 5 or result["setting"] != "reset_each_shift":
        raise ValueError("Unexpected evaluation protocol")
    if not result["shuffle"] or not result["legacy_rng_preamble"]:
        raise ValueError("Unexpected evaluation order protocol")
    if result["batch_size"] != 128 or result["seed"] != 1:
        raise ValueError("This report describes only the planned BS128 / seed1 audit")
    cfg = result["config"]
    if result["sgd_nesterov"] != cfg["nesterov"] or result["checkpoint_sha256"] != cfg["checkpoint_sha256"]:
        raise ValueError("Optimizer or checkpoint differs from the audit configuration")
    measured = float(np.mean([m["error_percent"] for m in metrics]))
    if abs(measured - result["mean_error_percent"]) > 1e-10:
        raise ValueError("Inconsistent mean error")
    if result["method"] == "actta_tent":
        paths = [m["path"] for m in result["activation_manifest"]]
        if paths != result["activation_selection"]["selected_paths"]:
            raise ValueError("Actual activation selection differs from configuration")
        names = result["trainable_parameters"]
        norm = [n for n in names if not n.endswith("_gsL")]
        if bool(norm) != cfg["adapt_norm"]:
            raise ValueError("Normalization update setting differs from actual trainables")
    return measured


def summarize(paths):
    rows, records, seen = [], [], set()
    for path in paths:
        path = Path(path)
        r = json.loads(path.read_text())
        value = validate_result(r)
        c = r["config"]
        key = (r["architecture"], r["method"], json.dumps(c, sort_keys=True), r["seed"])
        if key in seen:
            raise ValueError("Duplicate configuration/seed result")
        seen.add(key)
        target = PAPER[r["architecture"]][r["method"]]
        s = r["activation_selection"]
        half_depth = c["selection_mode"] != "legacy_residual_outputs"
        depth_target = (65.89 if r["architecture"] == "resnet50" else 52.47) if r["method"] == "actta_tent" and half_depth else None
        rows.append({
            "run": path.stem, "architecture": r["architecture"], "method": r["method"],
            "seed": r["seed"], "batch_size": r["batch_size"],
            "lr": r["lr"] if r["method"] != "source" else None,
            "nesterov": c["nesterov"] if r["method"] != "source" else None,
            "adapt_norm": c["adapt_norm"] if r["method"] == "actta_tent" else r["method"] == "tent",
            "selection_mode": c["selection_mode"],
            "activation_selection_applied": r["method"] == "actta_tent",
            "activation_count": len(r["activation_manifest"]),
            "selection_total": s["total_count"],
            "activation_depth_ratio": s["ratio"] if r["method"] == "actta_tent" else None,
            "gelu_approximation": (c.get("gelu_approximation") if r["method"] == "actta_tent"
                                    else "none" if r["architecture"] != "resnet50" else None),
            "trainable_parameters": sum(r["trainable_parameters"].values()),
            "measured_error_percent": value, "paper_table5_error_percent": target,
            "delta_table5_percentage_points": value - target,
            "same_table5_rounded_2dp": round(value, 2) == round(target, 2),
            "paper_table4_50pct_error_percent": depth_target,
            "delta_table4_percentage_points": value - depth_target if depth_target is not None else None,
            "same_table4_rounded_2dp": round(value, 2) == round(depth_target, 2) if depth_target is not None else None,
            "statistical_comparison": "single_seed_only",
            "checkpoint_sha256": r["checkpoint_sha256"],
            "result_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            "evaluation_seconds": r["evaluation_seconds"],
        })
        records.append(r)
    # Pair input identities within each architecture, including differing AcTTA
    # selections. These hashes compare both sample order and compressed image bytes.
    comparisons = []
    for arch in {r["architecture"] for r in records}:
        group = [r for r in records if r["architecture"] == arch]
        for r in group[1:]:
            baseline = group[0]
            for field in ["checkpoint_sha256", "image_ids_sha256", "class_map_sha256"]:
                if r[field] != baseline[field]:
                    raise ValueError(f"Paired identity mismatch: {arch} {field}")
            left = {m["corruption"]: m for m in baseline["per_corruption"]}
            for m in r["per_corruption"]:
                for field in ["sample_order_sha256", "sample_content_sha256"]:
                    if m[field] != left[m["corruption"]][field]:
                        raise ValueError(f"Paired input mismatch: {arch} {m['corruption']} {field}")
        comparisons.append({"architecture": arch, "runs": len(group),
                            "paired_input_order_and_content_match": True})
    return {"runs": rows, "paired_checks": comparisons,
            "all_single_seed": True,
            "final_original_paper_main_table_config_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--output", required=True, help="Summary prefix")
    args = parser.parse_args()
    summary = summarize(args.results)
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["runs"][0]))
        writer.writeheader()
        writer.writerows(summary["runs"])
    for row in summary["runs"]:
        print(f"{row['run']}: {row['measured_error_percent']:.4f}% "
              f"(Table5 delta {row['delta_table5_percentage_points']:+.4f} pp)")


if __name__ == "__main__":
    main()
