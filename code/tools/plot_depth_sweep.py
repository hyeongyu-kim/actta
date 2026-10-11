"""Standalone activation-depth curves from the completed measurements."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5), sharey=False)
    titles = {"resnet50": "ResNet-50 (BN)",
              "vit_base_patch16_224.orig_in21k_ft_in1k": "ViT-B/16"}
    for ax, architecture in zip(axes, titles):
        rows = sorted([row for row in summary["runs"] if row["architecture"] == architecture],
                      key=lambda row: row["nominal_depth_percent"])
        ax.plot([row["nominal_depth_percent"] for row in rows],
                [row["measured_error_percent"] for row in rows],
                "o-", color="#155EAD", lw=2, ms=5)
        for row in rows:
            ax.annotate(f"{row['measured_error_percent']:.2f}",
                        (row["nominal_depth_percent"], row["measured_error_percent"]),
                        xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8,
                        color="#155EAD")
        ax.set_title(titles[architecture], fontsize=12)
        ax.set_xlabel("Nominal activation depth (%)")
        ax.set_ylabel("Mean corruption error (%) — lower is better")
        ax.set_xticks([row["nominal_depth_percent"] for row in rows])
        ax.grid(alpha=.22)
        ax.margins(x=.05, y=.2)
    fig.suptitle("AcTTA activation depth: fixed LR 0.0025, normalization affine frozen", fontsize=12)
    fig.text(.5, .025, "Seed 1; fixed image-ID order. 0% has no optimizer; CNN uses target BN statistics.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .06, 1, .94))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output.with_suffix("."+suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(json.loads(Path(args.summary).read_text()), args.output)
