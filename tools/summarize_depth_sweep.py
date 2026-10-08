"""Check complete fixed-setting depth curves and retain every measured point."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from actta.data import CORRUPTIONS
from actta.depth_sweep import DEPTH_COUNTS


PAPER = {
    "resnet50": {10: 67.67, 25: 67.02, 50: 65.89, 75: 63.45, 100: 63.47},
    "vit_base_patch16_224.orig_in21k_ft_in1k":
        {10: 56.29, 25: 51.40, 50: 52.47, 75: 57.81, 100: 65.84},
}


def summarize(paths, required_depths=(0, 25, 50, 75, 100), expected_gpu=None,
              expected_shuffle=False):
    records, rows, seen = [], [], set()
    for path in paths:
        path = Path(path)
        result = json.loads(path.read_text())
        config = result["config"]
        architecture, nominal = result["architecture"], result["nominal_depth_percent"]
        if (architecture, nominal) in seen:
            raise ValueError("Duplicate architecture/depth point")
        seen.add((architecture, nominal))
        if not result["complete"] or result["scope"] != "full_benchmark":
            raise ValueError("Incomplete/smoke outputs cannot enter a depth curve")
        if result["method"] != "actta_tent":
            raise ValueError("Depth curve must use the entropy-minimization AcTTA_TENT objective")
        metrics = result["per_corruption"]
        if len(metrics) != 15 or {m["corruption"] for m in metrics} != set(CORRUPTIONS):
            raise ValueError("Expected all fifteen unique corruptions")
        if any(m["samples"] != 5000 for m in metrics):
            raise ValueError("Expected 5000 images per corruption")
        if not expected_shuffle:
            order_digest = hashlib.sha256(np.arange(5000, dtype="<i8").tobytes()).hexdigest()
            if any(m["sample_order_sha256"] != order_digest for m in metrics):
                raise ValueError("Depth runner must retain the original image-ID file order")
            if config.get("shuffle") is not False:
                raise ValueError("Ordered depth configuration must explicitly disable shuffle")
        if result["seed"] != 1 or result["batch_size"] != 128 or result["severity"] != 5:
            raise ValueError("Unexpected seed/batch/severity")
        if (result["setting"] != "reset_each_shift" or result["shuffle"] is not expected_shuffle
                or not result["legacy_rng_preamble"]):
            raise ValueError("Unexpected evaluation stream/reset")
        if config["adapt_norm"] or config["lr"]["actta_tent"] != .0025:
            raise ValueError("Normalization or LR varies from the fixed archived profile")
        if (config["momentum"], config["nesterov"], config["dampening"], config["weight_decay"]) != (.9, True, 0., 0.):
            raise ValueError("Unexpected optimizer configuration")
        if result["checkpoint_sha256"] != config["checkpoint_sha256"]:
            raise ValueError("Wrong checkpoint")
        visible_gpu = result["cuda_visible_devices"]
        if not isinstance(visible_gpu, str) or not visible_gpu.isdigit():
            raise ValueError("Expected exactly one physical GPU in CUDA_VISIBLE_DEVICES")
        if expected_gpu is not None and visible_gpu != str(expected_gpu):
            raise ValueError(f"This audit was requested on physical GPU {expected_gpu}")
        expected_count = DEPTH_COUNTS[architecture][nominal]
        manifest = result["activation_manifest"]
        if len(manifest) != expected_count or [m["path"] for m in manifest] != result["activation_selection"]["selected_paths"]:
            raise ValueError("Unexpected actual depth selection")
        names = result["trainable_parameters"]
        if any(not name.endswith("_gsL") for name in names):
            raise ValueError("Backbone/normalization affine parameters became trainable")
        if bool(names) != (nominal > 0) or result["optimizer_applied"] != (nominal > 0):
            raise ValueError("Zero-depth behavior differs from the defined control")
        if result["sgd_nesterov"] != (True if nominal > 0 else None):
            raise ValueError("Optimizer does not match the actual depth branch")
        measured = float(np.mean([m["error_percent"] for m in metrics]))
        if abs(measured - result["mean_error_percent"]) > 1e-10:
            raise ValueError("Inconsistent stored mean")
        target = PAPER[architecture].get(nominal)
        rows.append({
            "run": path.stem, "architecture": architecture, "method": "actta_tent",
            "nominal_depth_percent": nominal,
            "selected_activations": expected_count,
            "total_activations": result["activation_selection"]["total_count"],
            "actual_depth_percent": result["activation_depth_actual_ratio"] * 100.,
            "last_selected_path": manifest[-1]["path"] if manifest else None,
            "trainable_values": sum(names.values()), "normalization_affine_trainable": False,
            "lr": .0025 if nominal > 0 else None,
            "nesterov": True if nominal > 0 else None,
            "measured_error_percent": measured, "paper_table4_error_percent": target,
            "delta_percentage_points": measured - target if target is not None else None,
            "measured_no_worse_than_table4": measured <= target if target is not None else None,
            "matches_table4_at_two_decimals":
                f"{measured:.2f}" == f"{target:.2f}" if target is not None else None,
            "no_worse_than_table4_at_two_decimals":
                float(f"{measured:.2f}") <= target if target is not None else None,
            "seed": 1, "physical_gpu": int(visible_gpu), "predictions": 75000,
            "shuffle": result["shuffle"],
            "input_order_policy": result.get("input_order_policy", "main_shuffled_stream"),
            "zero_depth_note": result["baseline_kind"] if nominal == 0 else None,
            "result_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        })
        records.append(result)
    expected = {(architecture, depth) for architecture in DEPTH_COUNTS for depth in required_depths}
    if seen != expected:
        raise ValueError(f"Incomplete depth curves: missing {expected - seen}, extra {seen - expected}")
    checks = []
    for architecture in DEPTH_COUNTS:
        group = sorted([r for r in records if r["architecture"] == architecture],
                       key=lambda r: r["nominal_depth_percent"])
        baseline = group[0]
        base_metrics = {m["corruption"]: m for m in baseline["per_corruption"]}
        for record in group[1:]:
            for key in ["checkpoint_sha256", "image_ids_sha256", "class_map_sha256"]:
                if record[key] != baseline[key]:
                    raise ValueError(f"Input identity mismatch: {key}")
            for metric in record["per_corruption"]:
                for key in ["sample_order_sha256", "sample_content_sha256"]:
                    if metric[key] != base_metrics[metric["corruption"]][key]:
                        raise ValueError(f"Input stream mismatch: {architecture} {key}")
        checks.append({"architecture": architecture, "points": len(group),
                       "input_order_and_image_content_match": True})
    return {"runs": sorted(rows, key=lambda r: (r["architecture"], r["nominal_depth_percent"])),
            "paired_checks": checks, "all_single_seed": True,
            "shuffle": expected_shuffle,
            "paper_table4_has_10_percent_not_0_percent": True,
            "zero_point_has_no_published_table4_target": True,
            "all_requested_depth_points_complete": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--depths", nargs="+", type=int, default=[0, 25, 50, 75, 100])
    parser.add_argument("--expected-gpu", type=int)
    parser.add_argument("--input-order", choices=["ordered", "shuffled"], default="ordered")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = summarize(args.results, tuple(args.depths), args.expected_gpu,
                        expected_shuffle=args.input_order == "shuffled")
    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["runs"][0]))
        writer.writeheader()
        writer.writerows(summary["runs"])
    for row in summary["runs"]:
        print(f"{row['run']}: {row['measured_error_percent']:.4f}%")


if __name__ == "__main__":
    main()
