"""Explicit architecture settings recovered during the paper-depth audit.

This is separate from the earlier measured runners, whose source hashes and
results must remain reproducible. The split bottleneck follows torchvision and
the archived custom_standard_act.py (see vendor/TORCHVISION_LICENSE).
"""

from collections import OrderedDict
from copy import deepcopy

import torch
from torch import nn
from torchvision.models.resnet import Bottleneck

from .activation import replace_activations
from .adaptation import EntropyAdapter
from .imagenet import ImageNormalizer, load_imagenet_resnet50, resnet_activation_paths


class SplitActivationBottleneck(nn.Module):
    """Three independent ReLUs in execution and module-registration order."""

    expansion = 4

    def __init__(self, block):
        super().__init__()
        for i in (1, 2, 3):
            setattr(self, f"conv{i}", getattr(block, f"conv{i}"))
            setattr(self, f"bn{i}", getattr(block, f"bn{i}"))
            setattr(self, f"relu{i}", nn.ReLU(inplace=block.relu.inplace))
        self.downsample = block.downsample
        self.stride = block.stride

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu3(out)


def split_resnet_calls(module):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Bottleneck):
            setattr(module, name, SplitActivationBottleneck(child))
            count += 1
        else:
            count += split_resnet_calls(child)
    return count


def selection_paths(model, config):
    mode = config["selection_mode"]
    if mode == "first_activation_calls":
        all_paths = [name for name, module in model.named_modules()
                     if isinstance(module, nn.ReLU)]
        if len(all_paths) != 49:
            raise ValueError(f"Expected 49 independent ResNet-50 calls, got {len(all_paths)}")
        count = config["activation_count"]
        if not 1 <= count <= 49:
            raise ValueError("activation_count must be in 1..49")
        return all_paths[:count], len(all_paths)
    if mode == "legacy_residual_outputs":
        return resnet_activation_paths(model, config["activation_stages"]), 17
    if mode == "first_transformer_blocks":
        if len(model.model.blocks) != 12:
            raise ValueError("Expected 12 ViT-B/16 blocks")
        count = config["activation_blocks"]
        if not 1 <= count <= 12:
            raise ValueError("activation_blocks must be in 1..12")
        return [f"model.blocks.{i}.mlp.act" for i in range(count)], 12
    raise ValueError(f"Unknown selection_mode: {mode}")


def load_audit_model(config, checkpoint, device="cpu"):
    if config["architecture"] == "resnet50":
        legacy = config["selection_mode"] == "legacy_residual_outputs"
        model = load_imagenet_resnet50(checkpoint, device, expose_output_calls=legacy)
        if not legacy and split_resnet_calls(model) != 16:
            raise ValueError("Expected 16 split bottlenecks")
        return model
    if config["architecture"] == "vit_base_patch16_224.orig_in21k_ft_in1k":
        import timm
        backbone = timm.create_model(config["architecture"], pretrained=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
        if backbone.blocks[0].mlp.fc1.out_features != 3072:
            raise ValueError("Unexpected ViT MLP width")
        mean, std = backbone.pretrained_cfg["mean"], backbone.pretrained_cfg["std"]
        if tuple(mean) != (.5, .5, .5) or tuple(std) != (.5, .5, .5):
            raise ValueError("Unexpected orig_in21k_ft_in1k normalization")
        normalize = ImageNormalizer()
        normalize.mean.fill_(.5)
        normalize.std.fill_(.5)
        return nn.Sequential(OrderedDict([("normalize", normalize),
                                          ("model", backbone)])).to(device).eval()
    raise ValueError("Unsupported architecture")


class AuditEntropyAdapter(EntropyAdapter):
    """Use every original SGD option, including Nesterov momentum."""

    def __init__(self, model, config, method):
        super().__init__(model, config["lr"][method], optimizer="SGD",
                         momentum=config["momentum"],
                         adapt_norm=(method == "tent" or config["adapt_norm"]),
                         adapt_activation=(method == "actta_tent"))
        parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            parameters, lr=config["lr"][method], momentum=config["momentum"],
            dampening=config["dampening"], weight_decay=config["weight_decay"],
            nesterov=config["nesterov"],
        )
        self._initial_optimizer = deepcopy(self.optimizer.state_dict())


def prepare_audit_model(config, checkpoint, method, device="cpu"):
    model = load_audit_model(config, checkpoint, device)
    paths, total = selection_paths(model, config)
    manifest = []
    if method == "actta_tent":
        manifest = replace_activations(
            model, torch.zeros(1, 3, 224, 224, device=device), paths,
            share="channel", learn_beta=False,
            gelu_approximation=config.get("gelu_approximation", "tanh"),
        )
        if [m["path"] for m in manifest] != paths:
            raise ValueError("Replacement manifest differs from requested selection")
    if method != "source":
        model = AuditEntropyAdapter(model, config, method)
    return model, manifest, {"selected_paths": paths, "selected_count": len(paths),
                              "total_count": total, "ratio": len(paths) / total,
                              "denominator": config["selection_mode"]}
