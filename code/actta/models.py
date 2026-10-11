"""CIFAR backbones matching the local RobustBench checkpoint definitions."""

from pathlib import Path

import torch

from vendor.wide_resnet import WideResNet
from vendor.resnext import CifarResNeXt, ResNeXtBottleneck


class AugMixWRN(WideResNet):
    def __init__(self):
        super().__init__(depth=40, widen_factor=2, num_classes=100, sub_block1=False)
        self.register_buffer("mu", torch.tensor([0.5] * 3).reshape(1, 3, 1, 1))
        self.register_buffer("sigma", torch.tensor([0.5] * 3).reshape(1, 3, 1, 1))

    def forward(self, x):
        return super().forward((x - self.mu) / self.sigma)


class AugMixResNeXt(CifarResNeXt):
    def __init__(self, num_classes):
        super().__init__(ResNeXtBottleneck, depth=29, cardinality=4, base_width=32,
                         num_classes=num_classes)
        self.register_buffer("mu", torch.tensor([0.5] * 3).reshape(1, 3, 1, 1))
        self.register_buffer("sigma", torch.tensor([0.5] * 3).reshape(1, 3, 1, 1))

    def forward(self, x):
        return super().forward((x - self.mu) / self.sigma)


def load_cifar_model(architecture, dataset, checkpoint, device):
    if architecture == "Standard" and dataset == "cifar10_c":
        model = WideResNet(depth=28, widen_factor=10)
    elif architecture == "Hendrycks2020AugMix_WRN" and dataset == "cifar100_c":
        model = AugMixWRN()
    elif architecture == "Hendrycks2020AugMix_ResNeXt" and dataset in {"cifar10_c", "cifar100_c"}:
        model = AugMixResNeXt(10 if dataset == "cifar10_c" else 100)
    else:
        raise ValueError(f"Unsupported CIFAR backbone: {dataset}/{architecture}")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Download the original RobustBench checkpoint to {checkpoint}")
    # These original checkpoints include training metadata, not just tensors.
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data.get("state_dict", checkpoint_data)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if isinstance(model, (AugMixResNeXt, AugMixWRN)):
        # The local runner inserts these fixed normalization buffers after
        # loading a plain AugMix checkpoint. Their values match model_zoo.
        state.setdefault("mu", model.mu)
        state.setdefault("sigma", model.sigma)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
