"""ImageNet-C's original 5,000-image protocol and ResNet-50 V1 backbone.

The research GS2 replaces a shared bottleneck ReLU with expansion-width
parameters. Its shape fallback leaves the two internal ReLU calls unchanged.
Here those calls are explicit, preserving the effective original computation
without catching arbitrary runtime errors. The bottleneck forward follows
torchvision (BSD-3-Clause; see vendor/TORCHVISION_LICENSE).
"""

from collections import OrderedDict
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck
from torchvision.transforms import ToTensor


class ImageNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images):
        return (images - self.mean) / self.std


class OutputActivationBottleneck(Bottleneck):
    """Expose only the shared ReLU call that can use expansion-width GS2."""

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def expose_legacy_resnet_calls(model):
    """Keep state-dict paths while separating the effective GS2 call site."""
    count = 0
    for module in model.modules():
        if isinstance(module, Bottleneck):
            if module.conv1.out_channels != module.conv2.out_channels:
                raise ValueError("Unexpected bottleneck widths")
            if module.conv1.out_channels == module.conv3.out_channels:
                raise ValueError("This router requires a standard expansion bottleneck")
            module.__class__ = OutputActivationBottleneck
            count += 1
    return count


def load_imagenet_resnet50(checkpoint, device="cpu", expose_output_calls=False):
    backbone = resnet50(weights=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    backbone.load_state_dict(state, strict=True)
    if expose_output_calls:
        if expose_legacy_resnet_calls(backbone) != 16:
            raise ValueError("Expected 16 ResNet-50 bottlenecks")
    model = nn.Sequential(OrderedDict([("normalize", ImageNormalizer()), ("model", backbone)]))
    return model.to(device).eval()


def resnet_activation_paths(model, stages):
    if not 1 <= stages <= 4:
        raise ValueError("stages must be between 1 and 4")
    paths = ["model.relu"]
    for stage in range(1, stages + 1):
        paths += [f"model.layer{stage}.{i}.relu"
                  for i in range(len(model.model.get_submodule(f"layer{stage}")))]
    return paths


class ImageNetCorruption(Dataset):
    """Load exactly the ordered RobustBench list; normalization stays in-model."""

    def __init__(self, data_dir, corruption, image_ids, class_map, severity=5, limit=None):
        self.root = Path(data_dir) / corruption / str(severity)
        self.ids = Path(image_ids).read_text().splitlines()
        self.class_to_idx = json.loads(Path(class_map).read_text())
        if len(self.ids) != 5000 or len(set(self.ids)) != 5000:
            raise ValueError("Expected exactly 5,000 unique original ImageNet test ids")
        if len(self.class_to_idx) != 1000 or set(self.class_to_idx.values()) != set(range(1000)):
            raise ValueError("Expected the original 1,000-class mapping")
        for name in self.ids:
            parts = Path(name).parts
            if len(parts) != 2 or parts[0] not in self.class_to_idx or ".." in parts:
                raise ValueError(f"Invalid ImageNet image id: {name}")
        if limit is not None:
            if not 1 <= limit <= 5000:
                raise ValueError("limit must be between 1 and 5000")
            self.ids = self.ids[:limit]
        missing = [name for name in self.ids if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} ImageNet-C images; first: {missing[0]}")
        self.transform = ToTensor()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        name = self.ids[index]
        raw = (self.root / name).read_bytes()
        with Image.open(BytesIO(raw)) as image:
            if image.size != (224, 224):
                raise ValueError(f"Expected pre-cropped 224x224 ImageNet-C image: {name}, {image.size}")
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.class_to_idx[name.split("/")[0]], index, hashlib.sha256(raw).hexdigest()
