"""Run an experiment suite using the existing reproduction scripts."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "cifar": "reproduce_selected.sh",
    "imagenet-settings": "reproduce_architecture_audit.sh",
    "imagenet-depth": "reproduce_depth_sweep.sh",
}


def build_command(args):
    """Resolve caller-relative paths before running from the repository root."""
    data = str(Path(args.data_root).expanduser().resolve())
    output = str(Path(args.output).expanduser().resolve())
    command = ["bash", str(ROOT / "scripts" / SCRIPTS[args.suite]), data]
    if args.suite == "cifar":
        command.append(str(Path(args.checkpoint_root).expanduser().resolve()))
    else:
        command.extend(str(Path(path).expanduser().resolve())
                       for path in (args.resnet_checkpoint, args.vit_checkpoint))
    command.extend([output, str(args.gpu)])
    environment = os.environ.copy()
    environment.setdefault("ACTTA_PYTHON", sys.executable)
    if args.depths is not None:
        environment["DEPTH_POINTS"] = " ".join(map(str, args.depths))
    return command, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=SCRIPTS)
    parser.add_argument("--data-root", required=True,
                        help="Parent of CIFAR-10-C/CIFAR-100-C, or ImageNet-C directory")
    parser.add_argument("--checkpoint-root", help="CIFAR checkpoint root")
    parser.add_argument("--resnet-checkpoint", help="ResNet-50 IMAGENET1K_V1 checkpoint")
    parser.add_argument("--vit-checkpoint", help="ViT orig_in21k_ft_in1k checkpoint")
    parser.add_argument("--output", required=True, help="New or empty output directory")
    parser.add_argument("--gpu", type=int, default=3, help="Physical GPU index (default: 3)")
    parser.add_argument("--depths", type=int, nargs="+", choices=[0, 10, 25, 50, 75, 100],
                        help="ImageNet depth points; otherwise use the script defaults")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running")
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if args.suite == "cifar":
        if not args.checkpoint_root:
            parser.error("cifar requires --checkpoint-root")
        if args.resnet_checkpoint or args.vit_checkpoint:
            parser.error("cifar uses --checkpoint-root")
    else:
        if not args.resnet_checkpoint or not args.vit_checkpoint:
            parser.error("ImageNet suites require --resnet-checkpoint and --vit-checkpoint")
        if args.checkpoint_root:
            parser.error("ImageNet suites use individual checkpoint paths")
    if args.depths is not None:
        if args.suite != "imagenet-depth":
            parser.error("--depths applies only to imagenet-depth")
        if len(set(args.depths)) != len(args.depths):
            parser.error("--depths must contain unique points")
    command, environment = build_command(args)
    if args.dry_run:
        assignments = [f"ACTTA_PYTHON={shlex.quote(environment['ACTTA_PYTHON'])}"]
        if args.suite == "imagenet-depth" and "DEPTH_POINTS" in environment:
            assignments.append(f"DEPTH_POINTS={shlex.quote(environment['DEPTH_POINTS'])}")
        print(f"cd {shlex.quote(str(ROOT))} && "
              f"{' '.join(assignments)} {shlex.join(command)}")
        return 0
    data = Path(command[2])
    if not data.is_dir():
        parser.error(f"Data directory does not exist: {data}")
    if args.suite == "cifar":
        if not Path(command[3]).is_dir():
            parser.error(f"Checkpoint directory does not exist: {command[3]}")
    else:
        for checkpoint in command[3:5]:
            if not Path(checkpoint).is_file():
                parser.error(f"Checkpoint file does not exist: {checkpoint}")
    output = Path(command[-2])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output directory must be new or empty: {output}")
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    sys.exit(main())
