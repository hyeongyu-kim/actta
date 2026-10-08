"""Independent parity checks for the research ResNet shared-ReLU behavior."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import torch
from torchvision.models.resnet import Bottleneck

from actta.activation import AcTTAActivation
from actta.imagenet import expose_legacy_resnet_calls


class ImageNetTests(unittest.TestCase):
    def test_nonzero_original_shared_relu_output_and_gradient_parity(self):
        spec = importlib.util.spec_from_file_location(
            "legacy_imagenet_gs", Path(__file__).resolve().parents[1] / "reference/models/gs_model.py")
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        torch.manual_seed(13)
        original = Bottleneck(8, 2).double().eval()
        original.relu = legacy.GS2(8, preset="relu", learn_beta=False).double()
        for p in original.relu.parameters():
            with torch.no_grad():
                p.uniform_(-.15, .15)
        cleaned = deepcopy(original)
        self.assertEqual(expose_legacy_resnet_calls(cleaned), 1)
        cleaned.relu = AcTTAActivation(8).double()
        cleaned.relu.load_state_dict(original.relu.state_dict(), strict=True)
        x = torch.randn(2, 8, 5, 5, dtype=torch.float64, requires_grad=True)
        x2 = x.detach().clone().requires_grad_()
        a, b = original(x), cleaned(x2)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_close(x.grad, x2.grad, rtol=0, atol=0)
        for name, p in original.named_parameters():
            torch.testing.assert_close(p.grad, dict(cleaned.named_parameters())[name].grad,
                                       rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
