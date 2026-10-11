"""Evaluate source, TENT, and AcTTA_TENT on CIFAR-C.

All saved results distinguish full benchmark evaluations from smoke subsets.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from datetime import datetime, timezone
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from actta import EntropyAdapter, replace_activations
from actta.data import CIFARCorruption, CORRUPTIONS, IndexedDataset
from actta.models import load_cifar_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True, help="CIFAR-10-C or CIFAR-100-C directory")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=["source", "tent", "actta_tent"], default="actta_tent")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float, help="Override LR explicitly; no automatic tuning")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--corruption", choices=CORRUPTIONS, nargs="+")
    parser.add_argument("--limit", type=int, help="Smoke subset per corruption; never a paper result")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True,
                        help="Original test_time.py shuffles each corruption stream")
    parser.add_argument("--legacy-rng-preamble", action=argparse.BooleanOptionalAction, default=True,
                        help="Match conf.py's torch.rand(1) logging draw before model initialization")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    batch_size = args.batch_size or config["batch_size"]
    if batch_size < 2:
        parser.error("This batch runner supports batch_size >= 2")
    dataset = config["dataset"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; choose --device cpu")
    output = Path(args.output)
    if output.exists():
        parser.error(f"Result already exists: {output}; choose a new output")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", True)
    preamble = float(torch.rand(1)) if args.legacy_rng_preamble else None
    model = load_cifar_model(config["architecture"], dataset, args.checkpoint, device)
    manifest = []
    if args.method == "actta_tent":
        manifest = replace_activations(
            model, torch.zeros(1, 3, 32, 32, device=device), config["activation_prefixes"],
            share=config.get("share", "channel"), learn_beta=False)
    lr = args.lr if args.lr is not None else config["lr"][args.method]
    if args.method != "source":
        model = EntropyAdapter(model, lr, config["optimizer"],
                               adapt_norm=(args.method == "tent"),
                               adapt_activation=(args.method == "actta_tent"))
        trainable = model.trainable_parameters()
    else:
        trainable = {}
    with open(args.checkpoint, "rb") as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    result = {"method": args.method, "dataset": dataset, "architecture": config["architecture"],
              "seed": args.seed, "batch_size": batch_size, "lr": lr,
              "setting": "reset_each_shift", "severity": 5, "device": device,
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "cudnn_version": torch.backends.cudnn.version(),
              "cuda_version": torch.version.cuda, "shuffle": args.shuffle,
              "legacy_rng_preamble": args.legacy_rng_preamble, "preamble_draw": preamble,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "activation_manifest": manifest, "trainable_parameters": trainable,
              "config": config, "checkpoint_sha256": checkpoint_sha256,
              "per_corruption": [], "complete": False, "scope": "smoke_subset"}
    corruptions = args.corruption or CORRUPTIONS
    if len(set(corruptions)) != len(corruptions):
        parser.error("Each corruption must appear only once")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Create an initial record: an interrupted run must remain explicitly incomplete.
    output.write_text(json.dumps(result, indent=2) + "\n")
    for corruption in corruptions:
        if args.method != "source":
            model.reset()
        random.seed(args.seed)
        np.random.seed(args.seed)
        test_set = IndexedDataset(CIFARCorruption(args.data_dir, corruption, limit=args.limit))
        loader = DataLoader(test_set, batch_size=batch_size, shuffle=args.shuffle, num_workers=0)
        correct = total = 0
        order_hash = hashlib.sha256()
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.monotonic()
        for images, labels, indices in loader:
            order_hash.update(indices.numpy().astype("<i8").tobytes())
            with torch.no_grad():
                logits = model(images.to(device))
            correct += int((logits.argmax(1).cpu() == labels).sum())
            total += len(labels)
        error = 100. * (1. - correct / total)
        if device == "cuda":
            torch.cuda.synchronize()
        result["per_corruption"].append({"corruption": corruption, "samples": total,
                                          "error_percent": error,
                                          "sample_order_sha256": order_hash.hexdigest(),
                                          "seconds": time.monotonic() - started})
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{corruption}: {error:.2f}% ({total} samples)", flush=True)
    result["complete"] = True
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["evaluation_seconds"] = sum(row["seconds"] for row in result["per_corruption"])
    result["mean_error_percent"] = float(np.mean([r["error_percent"] for r in result["per_corruption"]]))
    result["scope"] = "full_benchmark" if set(corruptions) == set(CORRUPTIONS) and args.limit is None else "smoke_subset"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Mean error: {result['mean_error_percent']:.2f}% [{result['scope']}]", flush=True)


if __name__ == "__main__":
    main()
