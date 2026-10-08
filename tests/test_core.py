"""Scientific invariants for the release candidate, independent of benchmark scores."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from actta import AcTTAActivation, EntropyAdapter, replace_activations
from actta.data import CIFARCorruption
from tools.aggregate_results import summarize
from tools.summarize_logs import collect


torch.set_num_threads(2)


class CoreTests(unittest.TestCase):
    def test_legacy_forward_and_gradient_parity(self):
        path = Path(__file__).resolve().parents[1] / "reference/models/gs_model.py"
        spec = importlib.util.spec_from_file_location("legacy_gs", path)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for preset in ["relu", "gelu"]:
            for share in ["channel", "layer"]:
                for shape in [(3, 4), (2, 3, 4), (2, 4, 5, 5)]:
                    torch.manual_seed(11)
                    old = legacy.GS2(4, share, preset, learn_beta=False).double()
                    new = AcTTAActivation(4, share, preset).double()
                    for p in old.parameters():
                        with torch.no_grad():
                            p.uniform_(-.2, .2)
                    new.load_state_dict(old.state_dict(), strict=True)
                    x1 = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                    x2 = x1.detach().clone().requires_grad_()
                    y1, y2 = old(x1), new(x2)
                    torch.testing.assert_close(y1, y2, rtol=0, atol=0)
                    y1.square().sum().backward()
                    y2.square().sum().backward()
                    torch.testing.assert_close(x1.grad, x2.grad, rtol=0, atol=0)
                    for a, b in zip(old.parameters(), new.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_initialization_preserves_base(self):
        x = torch.randn(2, 4, 6, 6)
        torch.testing.assert_close(AcTTAActivation(4)(x), nn.ReLU()(x), rtol=0, atol=0)
        a = AcTTAActivation(4, preset="gelu", gelu_approximation="none")
        torch.testing.assert_close(a(x), nn.GELU()(x), rtol=0, atol=0)
        self.assertNotIn("beta_gsL", dict(a.named_parameters()))

    def test_shape_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "channel mismatch"):
            AcTTAActivation(4)(torch.randn(2, 5, 3, 3))

    def test_replacement_preserves_model_and_modes(self):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(),
                              nn.Flatten(), nn.Linear(4 * 4 * 4, 3))
        model.train()
        model[1].eval()
        original = deepcopy(model).eval()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        x = torch.randn(2, 3, 4, 4)
        manifest = replace_activations(model, x, ["2"])
        self.assertEqual(manifest[0]["channels"], 4)
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        for k, v in state.items():
            torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
        model.eval()
        torch.testing.assert_close(model(x), original(x), rtol=0, atol=0)

    def test_shared_activation_rejected_before_edit(self):
        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(x[:, :2]).sum() + self.relu(x).sum()

        model = Shared()
        with self.assertRaisesRegex(ValueError, "per call site"):
            replace_activations(model, torch.randn(2, 4), ["relu"])
        self.assertIsInstance(model.relu, nn.ReLU)

    def test_activation_only_update_and_reset(self):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4),
                              AcTTAActivation(4), nn.Flatten(), nn.Linear(64, 3))
        adapter = EntropyAdapter(model, .01)
        original = {k: p.clone() for k, p in model.named_parameters()}
        x = torch.randn(4, 3, 4, 4)
        before = model(x).detach()
        output = adapter(x)
        torch.testing.assert_close(output, before, rtol=0, atol=0)
        changed = []
        for name, p in model.named_parameters():
            if "gsL" not in name:
                torch.testing.assert_close(p, original[name], rtol=0, atol=0)
            elif not torch.equal(p, original[name]):
                changed.append(name)
        self.assertEqual(len(changed), 3)
        self.assertIsNone(model[1].running_mean)
        self.assertFalse(model[1].weight.requires_grad)
        self.assertTrue(adapter.optimizer.state)
        adapter.reset()
        self.assertFalse(adapter.optimizer.state)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, original[name], rtol=0, atol=0)
        torch.testing.assert_close(adapter(x), output, rtol=0, atol=0)

    def test_cifar_severity_and_image_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            images = np.lib.format.open_memmap(p / "gaussian_noise.npy", mode="w+",
                                               dtype=np.uint8, shape=(50000, 32, 32, 3))
            images[40000] = 255
            images[40001] = 127
            del images
            np.save(p / "labels.npy", np.arange(50000) % 10)
            data = CIFARCorruption(p, "gaussian_noise", limit=2)
            x, label = data[0]
            self.assertEqual(label, 0)
            self.assertEqual(tuple(x.shape), (3, 32, 32))
            self.assertEqual(float(x.mean()), 1.)
            self.assertEqual(data[1][1], 1)
            torch.testing.assert_close(data[1][0], torch.full((3, 32, 32), 127 / 255))

    def test_log_copies_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            text = "[26/04/15 13:25:29] [conf.py: 432]: ADACONTRAST:\n  ALPHA: 1.0\nMODEL:\n  ADAPTATION: tent_gs\nRNG_SEED: 1\n"
            (p / "a.txt").write_text(text)
            (p / "b.txt").write_text(text)
            rows = collect([p])
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["paths"]), 2)
            self.assertEqual(rows[0]["family"], "AcTTA")
            self.assertFalse(rows[0]["complete"])

    def test_smoke_result_is_not_aggregated(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "smoke.json"
            p.write_text('{"complete": true, "scope": "smoke_subset"}')
            with self.assertRaisesRegex(ValueError, "full benchmark"):
                summarize([p])


if __name__ == "__main__":
    unittest.main()
