"""Stage-9 gate: the gated B-scan encoder must start as RTM-only and be trainable."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rtm_inv.models import GatedBscanUpdateNet, RTMInvNet  # noqa: E402


class StubRTM(nn.Module):
    def forward(self, observed_bscan, current_model, **kwargs):
        view = observed_bscan.reshape(observed_bscan.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1)
        return current_model + view

    def synthesize_data(self, model, n_shots, nt, sample_config=None, **kwargs):
        return model.reshape(model.shape[0], 1, -1)[..., :nt].repeat(1, n_shots, 1)


def build(mode: str) -> RTMInvNet:
    torch.manual_seed(0)
    return RTMInvNet(
        rtm_operator=StubRTM(),
        input_mode=mode,
        unet_base_channels=8,
        unet_depth=2,
        num_stages=2,
    )


def batch():
    g = torch.Generator().manual_seed(1)
    return torch.randn(1, 6, 64, generator=g), torch.rand(1, 1, 20, 24, generator=g)


class GatedFusionTest(unittest.TestCase):
    def test_uses_gated_net_and_two_input_channels(self) -> None:
        net = build("m0_rtm_bscan_gated")
        self.assertIsInstance(net.update_net, GatedBscanUpdateNet)
        self.assertEqual(net.update_net.inc.block[0].in_channels, 2)

    def test_starts_as_rtm_only_network(self) -> None:
        bscan, m0 = batch()
        gated = build("m0_rtm_bscan_gated")
        outputs = gated(bscan, m0)
        # Zero-initialised output conv: no update yet, whatever the B-scan is.
        self.assertTrue(torch.equal(outputs["final_model"], m0))

    def test_bscan_changes_prediction_only_through_the_gate(self) -> None:
        bscan, m0 = batch()
        net = build("m0_rtm_bscan_gated")
        for p in net.update_net.parameters():
            nn.init.normal_(p, std=0.05)
        base = net(bscan, m0)["final_model"]
        other = net(torch.randn_like(bscan) * 3, m0)["final_model"]
        self.assertFalse(torch.allclose(base, other))
        net.update_net.proj.weight.data.zero_()
        net.update_net.proj.bias.data.zero_()
        a = net(bscan, m0)["final_model"]
        b = net(torch.randn_like(bscan), m0)["final_model"]
        # RTM stub depends on the B-scan mean, so compare with the same mean.
        b = net(bscan.flip(-1), m0)["final_model"]
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_bscan_branch_receives_gradient(self) -> None:
        bscan, m0 = batch()
        net = build("m0_rtm_bscan_gated")
        for p in net.update_net.outc.parameters():
            nn.init.normal_(p, std=0.05)
        for p in net.update_net.proj.parameters():
            nn.init.normal_(p, std=0.05)
        net(bscan, m0)["final_model"].sum().backward()
        for name in ("bscan_encoder", "gate", "proj"):
            grads = [p.grad for p in getattr(net.update_net, name).parameters()]
            self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads), name)

    def test_concat_mode_unchanged(self) -> None:
        net = build("m0_rtm_bscan")
        self.assertEqual(net.update_net.inc.block[0].in_channels, 3)


if __name__ == "__main__":
    unittest.main()
