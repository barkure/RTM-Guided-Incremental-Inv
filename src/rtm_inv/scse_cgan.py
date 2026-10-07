"""Paper-derived SCSE cGAN adaptation, NOT an official/faithful reproduction.

All discretionary choices are in docs/stage7_scse_adaptation.json. This module
does not modify the RTM model or its checkpoints and has no physics/RTM calls.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SCSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1),
                                     nn.ReLU(), nn.Conv2d(hidden, channels, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel(x) + x * self.spatial(x)


class ResidualSCSE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.InstanceNorm2d(channels),
                                  nn.LeakyReLU(0.2), nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.InstanceNorm2d(channels))
        self.attention = SCSE(channels)

    def forward(self, x):
        return self.attention(F.leaky_relu(x + self.body(x), 0.2))


class SCSEGenerator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        channels = [base_channels * n for n in (1, 2, 4, 8, 8)]
        self.encoders = nn.ModuleList()
        previous = 1
        for index, channel in enumerate(channels):
            layers = [nn.Conv2d(previous, channel, 4, stride=2, padding=1),
                      nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2)]
            if index:
                layers.append(ResidualSCSE(channel))
            self.encoders.append(nn.Sequential(*layers))
            previous = channel
        self.bottleneck = ResidualSCSE(previous)
        self.decoders = nn.ModuleList()
        for channel in reversed(channels[:-1]):
            self.decoders.append(nn.Sequential(nn.Conv2d(previous + channel, channel, 3, padding=1),
                                               nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2), ResidualSCSE(channel)))
            previous = channel
        self.output = nn.Conv2d(previous, 1, 3, padding=1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1:] != (1, 256, 256):
            raise ValueError("expected Bx1x256x256 input")
        features = []
        for encoder in self.encoders:
            x = encoder(x)
            features.append(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat((x, skip), dim=1))
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        return (torch.tanh(self.output(x)) + 1) * 0.5


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.branches = nn.ModuleList()
        for _ in range(3):
            layers, previous = [], 2
            for index, channel in enumerate([base_channels, base_channels * 2, base_channels * 4, base_channels * 8, 1]):
                layers.append(nn.Conv2d(previous, channel, 4, stride=2 if index < 3 else 1, padding=1))
                if index < 4:
                    layers.extend([nn.InstanceNorm2d(channel), nn.LeakyReLU(0.2)])
                previous = channel
            self.branches.append(nn.Sequential(*layers))

    def forward(self, observed, model):
        if observed.shape != model.shape:
            raise ValueError("condition and model images must have identical shapes")
        pair = torch.cat((observed, model), dim=1)
        return [branch(pair if factor == 1 else F.avg_pool2d(pair, factor))
                for factor, branch in zip((1, 2, 4), self.branches)]


def gradient_difference(prediction, target):
    return (F.l1_loss(prediction[..., 1:, :] - prediction[..., :-1, :], target[..., 1:, :] - target[..., :-1, :])
            + F.l1_loss(prediction[..., :, 1:] - prediction[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]))


def adversarial(logits, real):
    return sum(F.binary_cross_entropy_with_logits(x, torch.ones_like(x) if real else torch.zeros_like(x)) for x in logits)


def train_step(generator, discriminator, optimizer_g, optimizer_d, observed, target):
    """One detached D update then one non-saturating G update, explicit ownership."""
    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)
    fake = generator(observed)
    loss_d = 0.5 * (adversarial(discriminator(observed, target), True)
                    + adversarial(discriminator(observed, fake.detach()), False))
    if not torch.isfinite(loss_d):
        raise FloatingPointError("nonfinite SCSE discriminator loss")
    loss_d.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in discriminator.parameters()):
        raise FloatingPointError("nonfinite SCSE discriminator gradient")
    optimizer_d.step()
    optimizer_d.zero_grad(set_to_none=True)
    flags = [p.requires_grad for p in discriminator.parameters()]
    try:
        discriminator.requires_grad_(False)
        gan = adversarial(discriminator(observed, fake), True)
        l1 = F.l1_loss(fake, target)
        gradient = gradient_difference(fake, target)
        loss_g = gan + 100 * l1 + 5 * gradient
        if not torch.isfinite(loss_g):
            raise FloatingPointError("nonfinite SCSE generator loss")
        loss_g.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in generator.parameters()):
            raise FloatingPointError("nonfinite SCSE generator gradient")
        optimizer_g.step()
    finally:
        for parameter, flag in zip(discriminator.parameters(), flags):
            parameter.requires_grad_(flag)
    return {"generator": float(loss_g.detach()), "discriminator": float(loss_d.detach()),
            "l1": float(l1.detach()), "gradient": float(gradient.detach())}
