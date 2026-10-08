"""Evaluate activation depths in the original test_time_hgh fixed file order."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import yaml

from actta.data import CORRUPTIONS
from actta.imagenet import ImageNetCorruption
from actta.depth_sweep import prepare_depth_model


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/depth_sweep_ordered/resnet50_d050.yaml")
    parser.add_argument("--data-dir", required=True, help="ImageNet-C directory")
    parser.add_argument("--checkpoint", required=True, help="Exact checkpoint named by the configuration")
    parser.add_argument("--image-ids", default="reference/data/imagenet_test_image_ids.txt")
    parser.add_argument("--class-map", default="reference/data/imagenet_class_to_id_map.json")
    parser.add_argument("--method", choices=["actta_tent"], default="actta_tent")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--corruption", choices=CORRUPTIONS, nargs="+")
    parser.add_argument("--limit", type=int, help="Smoke subset, excluded from full benchmark reporting")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if config.get("shuffle") is not False:
        parser.error("Original depth-ablation protocol requires shuffle: false")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Result already exists: {output}")
    corruptions = args.corruption or CORRUPTIONS
    if len(set(corruptions)) != len(corruptions):
        parser.error("Each corruption must appear only once")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != config["checkpoint_sha256"]:
        parser.error("Checkpoint SHA-256 does not match the configuration")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", True)
    torch.set_num_threads(4)
    preamble = float(torch.rand(1))
    model, manifest, selection = prepare_depth_model(config, args.checkpoint, args.device)
    lr = config["lr"][args.method]
    trainable = model.trainable_parameters() if args.method != "source" else {}
    import timm
    backbone = model.model if args.method == "source" else model.model.model
    fused_attention = bool(backbone.blocks[0].attn.fused_attn) if "vit" in config["architecture"] else None
    result = {"dataset": "imagenet_c", "architecture": config["architecture"], "weights": config["weights"],
              "method": args.method, "seed": args.seed, "batch_size": config["batch_size"],
              "lr": lr if trainable else None, "setting": "reset_each_shift", "severity": 5, "device": args.device,
              "shuffle": False, "input_order_policy": "original_hgh_image_id_file_order",
              "legacy_rng_preamble": True, "preamble_draw": preamble,
              "workers": args.workers, "config": config, "checkpoint_sha256": checkpoint_sha,
              "image_ids_sha256": sha256_file(args.image_ids), "class_map_sha256": sha256_file(args.class_map),
              "preprocessing": "PIL RGB -> ToTensor; in-model backbone-specific mean/std; no resize/crop",
              "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
              "numpy_version": np.__version__, "cuda_version": torch.version.cuda,
              "cudnn_version": torch.backends.cudnn.version(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "activation_manifest": manifest, "trainable_parameters": trainable,
              "activation_selection": selection, "timm_version": timm.__version__,
              "sgd_nesterov": config["nesterov"] if trainable else None,
              "requested_sgd_nesterov": config["nesterov"],
              "nominal_depth_percent": config["nominal_depth_percent"],
              "activation_depth_actual_ratio": selection["ratio"],
              "optimizer_applied": bool(trainable),
              "norm_affine_adapted": False,
              "depth_profile": "archived_norm_frozen",
              "gelu_approximation_actual": (config["gelu_approximation"] if manifest else "none") if "vit" in config["architecture"] else None,
              "baseline_kind": "activation_adaptation" if trainable else "frozen_affine_target_BN_or_fixed_LN",
              "attention_fused": fused_attention,
              "per_corruption": [], "complete": False, "scope": "smoke_subset"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for corruption in corruptions:
        if args.method != "source":
            model.reset()
        random.seed(args.seed)
        np.random.seed(args.seed)
        dataset = ImageNetCorruption(args.data_dir, corruption, args.image_ids, args.class_map, limit=args.limit)
        loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                            num_workers=args.workers, pin_memory=(args.device == "cuda"))
        correct = total = 0
        order_hash, content_hash = hashlib.sha256(), hashlib.sha256()
        if args.device == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        for images, labels, indices, image_hashes in loader:
            index_bytes = indices.numpy().astype("<i8").tobytes()
            order_hash.update(index_bytes)
            for index, digest in zip(indices.tolist(), image_hashes):
                content_hash.update(int(index).to_bytes(8, "little"))
                content_hash.update(bytes.fromhex(digest))
            with torch.no_grad():
                logits = model(images.to(args.device, non_blocking=True))
            correct += int((logits.argmax(1).cpu() == labels).sum())
            total += len(labels)
        if args.device == "cuda":
            torch.cuda.synchronize()
        metric = {"corruption": corruption, "samples": total,
                  "error_percent": 100. * (1. - correct / total),
                  "sample_order_sha256": order_hash.hexdigest(),
                  "sample_content_sha256": content_hash.hexdigest(),
                  "seconds": time.monotonic() - started}
        result["per_corruption"].append(metric)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{corruption}: {metric['error_percent']:.2f}% ({total} samples)", flush=True)
    result["complete"] = True
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["evaluation_seconds"] = sum(m["seconds"] for m in result["per_corruption"])
    result["mean_error_percent"] = float(np.mean([m["error_percent"] for m in result["per_corruption"]]))
    result["scope"] = "full_benchmark" if set(corruptions) == set(CORRUPTIONS) and args.limit is None else "smoke_subset"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Mean error: {result['mean_error_percent']:.4f}% [{result['scope']}]", flush=True)


if __name__ == "__main__":
    main()
