"""Check the binary runtime used for the recorded ImageNet-C measurements."""

import importlib.metadata
import json
import platform

from PIL import features
import torch


def main():
    actual = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "numpy": importlib.metadata.version("numpy"),
        "Pillow": importlib.metadata.version("Pillow"),
        "timm": importlib.metadata.version("timm"),
        "compiled_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "jpeg_codec": features.version_codec("jpg"),
        "libjpeg_turbo": features.version_feature("libjpeg_turbo"),
    }
    expected = {
        "python": "3.12.2", "torch": "2.2.1", "torchvision": "0.17.1",
        "numpy": "1.26.4", "Pillow": "10.2.0", "timm": "1.0.15",
        "compiled_cuda": "12.1", "cudnn": 8902,
        "jpeg_codec": "9.0", "libjpeg_turbo": None,
    }
    mismatches = {key: {"actual": actual[key], "expected": value}
                  for key, value in expected.items() if actual[key] != value}
    print(json.dumps({"matches_measured_runtime": not mismatches,
                      "actual": actual, "mismatches": mismatches}, indent=2))
    if mismatches:
        print("Use environment-conda-linux-64.lock.txt and the pinned Python requirements "
              "to recreate the measured binary runtime.")
        return 1
    print("Recorded ImageNet-C runtime fingerprint matched. "
          "The explicit Conda lock pins the complete native package builds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
