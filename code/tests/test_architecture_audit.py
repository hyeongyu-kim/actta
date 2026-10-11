"""Depth boundaries, independent call sites, and joint LN/SGD update checks."""

import unittest
from copy import deepcopy

import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck

from actta.activation import AcTTAActivation
from actta.architecture_audit import (AuditEntropyAdapter, SplitActivationBottleneck,
                                     selection_paths, split_resnet_calls)


class ArchitectureAuditTests(unittest.TestCase):
    def test_resnet_half_depth_boundary(self):
        model = nn.Sequential()
        model.add_module("model", resnet50(weights=None))
        self.assertEqual(split_resnet_calls(model), 16)
        paths, total = selection_paths(model, {"selection_mode": "first_activation_calls",
                                               "activation_count": 25})
        self.assertEqual(total, 49)
        self.assertEqual(paths[0], "model.relu")
        self.assertEqual(paths[-1], "model.layer3.0.relu3")
        self.assertEqual(len(paths), 25)
        self.assertEqual(sum(p.startswith("model.layer1.") for p in paths), 9)
        self.assertEqual(sum(p.startswith("model.layer2.") for p in paths), 12)
        self.assertEqual(sum(p.startswith("model.layer3.") for p in paths), 3)

    def test_split_bottleneck_preserves_logits_and_gradient(self):
        source = Bottleneck(8, 2).double().eval()
        split = SplitActivationBottleneck(deepcopy(source)).double().eval()
        x = torch.randn(2, 8, 5, 5, dtype=torch.double, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a, b = source(x), split(y)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        for (an, ap), (bn, bp) in zip(source.named_parameters(), split.named_parameters()):
            self.assertEqual(an, bn)
            torch.testing.assert_close(ap.grad, bp.grad, rtol=0, atol=0)

    def test_joint_ln_and_nesterov_update_and_reset(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4),
                              AcTTAActivation(4, preset="gelu"), nn.Linear(4, 3))
        config = {"lr": {"actta_tent": .001}, "momentum": .9, "nesterov": True,
                  "dampening": 0., "weight_decay": 0., "adapt_norm": True}
        adapter = AuditEntropyAdapter(model, config, "actta_tent")
        names = set(adapter.trainable_parameters())
        self.assertEqual(names, {"1.weight", "1.bias", "2.neg_gsL", "2.pos_gsL", "2.shift_gsL"})
        initial = deepcopy(model.state_dict())
        manual = deepcopy(model)
        optimizer = torch.optim.SGD([p for p in manual.parameters() if p.requires_grad],
                                    lr=.001, momentum=.9, nesterov=True)
        x = torch.randn(8, 4)
        logits = manual(x)
        loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        optimizer.step()
        observed = adapter(x)
        torch.testing.assert_close(observed, logits.detach(), rtol=0, atol=0)
        for name, value in manual.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(initial[n], model.state_dict()[n]) for n in names))
        adapter.reset()
        for name, value in initial.items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(adapter.optimizer.state_dict()["state"], {})


if __name__ == "__main__":
    unittest.main()
