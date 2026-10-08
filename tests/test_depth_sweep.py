"""Zero-point semantics and archived depth boundaries for the full sweep."""

from collections import OrderedDict
import unittest

import torch
from torch import nn
from torchvision.models import resnet50

from actta.architecture_audit import split_resnet_calls
from actta.depth_sweep import DEPTH_COUNTS, FrozenDepthAdapter, depth_paths


class DepthSweepTests(unittest.TestCase):
    def test_resnet_all_archived_boundaries(self):
        model = nn.Sequential(OrderedDict([("model", resnet50(weights=None))]))
        split_resnet_calls(model)
        previous = []
        last_paths = {0: None, 10: "model.layer1.1.relu1",
                      25: "model.layer2.0.relu3", 50: "model.layer3.0.relu3",
                      75: "model.layer3.5.relu1", 100: "model.layer4.2.relu3"}
        for nominal, count in DEPTH_COUNTS["resnet50"].items():
            config = {"architecture": "resnet50", "nominal_depth_percent": nominal,
                      "selection_mode": "first_activation_calls", "activation_count": count,
                      "adapt_norm": False}
            paths, total = depth_paths(model, config)
            self.assertEqual(total, 49)
            self.assertEqual(len(paths), count)
            self.assertEqual(paths[-1] if paths else None, last_paths[nominal])
            self.assertEqual(paths[:len(previous)], previous)
            previous = paths

    def test_vit_all_depth_boundaries_and_invalid_mapping(self):
        architecture = "vit_base_patch16_224.orig_in21k_ft_in1k"
        backbone = nn.Module()
        backbone.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        model = nn.Sequential(OrderedDict([("model", backbone)]))
        for nominal, count in DEPTH_COUNTS[architecture].items():
            config = {"architecture": architecture, "nominal_depth_percent": nominal,
                      "selection_mode": "first_transformer_blocks", "activation_blocks": count,
                      "adapt_norm": False}
            paths, total = depth_paths(model, config)
            self.assertEqual(total, 12)
            self.assertEqual(paths, [f"model.blocks.{i}.mlp.act" for i in range(count)])
        config["activation_blocks"] = 11
        with self.assertRaises(ValueError):
            depth_paths(model, config)

    def test_zero_preserves_batch_statistics_without_updates(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        reference = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        reference.load_state_dict(model.state_dict())
        reference.train()
        reference[1].track_running_stats = False
        reference[1].running_mean = reference[1].running_var = None
        adapter = FrozenDepthAdapter(model)
        images = torch.randn(4, 3, 5, 5)
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        self.assertIsNone(adapter.optimizer)
        self.assertFalse(any(param.requires_grad for param in model.parameters()))
        self.assertEqual(adapter.trainable_parameters(), {})
        torch.testing.assert_close(adapter(images), reference(images), rtol=0, atol=0)
        adapter.reset()
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
