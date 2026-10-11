"""Fixed-setting activation depth sweep, including a well-defined zero point.

Earlier executed architecture_audit.py remains unchanged. Positive depths reuse
its exact replacement and adaptation path. Zero depth freezes all parameters
and retains target-batch BN statistics, avoiding an empty optimizer.
"""

from copy import deepcopy

import torch
from torch import nn

from .activation import replace_activations
from .architecture_audit import AuditEntropyAdapter, load_audit_model, selection_paths


DEPTH_COUNTS = {
    "resnet50": {0: 0, 10: 5, 25: 13, 50: 25, 75: 38, 100: 49},
    "vit_base_patch16_224.orig_in21k_ft_in1k":
        {0: 0, 10: 1, 25: 3, 50: 6, 75: 9, 100: 12},
}


def depth_paths(model, config):
    architecture = config["architecture"]
    nominal = config["nominal_depth_percent"]
    if architecture not in DEPTH_COUNTS or nominal not in DEPTH_COUNTS[architecture]:
        raise ValueError("Unsupported architecture or depth point")
    field = "activation_count" if architecture == "resnet50" else "activation_blocks"
    if config[field] != DEPTH_COUNTS[architecture][nominal]:
        raise ValueError("Selected count differs from the archived depth mapping")
    if config["adapt_norm"]:
        raise ValueError("The archived-depth sweep freezes normalization affine parameters")
    if config[field] == 0:
        total = 49 if architecture == "resnet50" else 12
        return [], total
    return selection_paths(model, config)


class FrozenDepthAdapter(nn.Module):
    """Zero activation depth, frozen affine parameters, target BN statistics."""

    def __init__(self, model):
        super().__init__()
        self.model = model.train()
        self.model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
        self.optimizer = None
        self._initial_model = deepcopy(model.state_dict())

    @torch.no_grad()
    def forward(self, images):
        return self.model(images)

    def reset(self):
        self.model.load_state_dict(self._initial_model, strict=True)

    def trainable_parameters(self):
        return {}


def prepare_depth_model(config, checkpoint, device="cpu"):
    model = load_audit_model(config, checkpoint, device)
    paths, total = depth_paths(model, config)
    manifest = []
    if paths:
        manifest = replace_activations(
            model, torch.zeros(1, 3, 224, 224, device=device), paths,
            share="channel", learn_beta=False,
            gelu_approximation=config.get("gelu_approximation", "tanh"),
        )
        if [entry["path"] for entry in manifest] != paths:
            raise ValueError("Requested and actual activation selections differ")
        adapted = AuditEntropyAdapter(model, config, "actta_tent")
    else:
        adapted = FrozenDepthAdapter(model)
    return adapted, manifest, {
        "selected_paths": paths, "selected_count": len(paths), "total_count": total,
        "ratio": len(paths) / total, "nominal_depth_percent": config["nominal_depth_percent"],
        "denominator": config["selection_mode"],
        "zero_depth_behavior": "frozen affine; target-batch BN statistics; no optimizer",
    }
