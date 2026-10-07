"""FusionInv-GAN paper-derived adaptation; see docs/stage7_fusion_design.md."""
import torch
from torch import nn
from torch.nn import functional as F


def _positive(value, like):
    value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if not torch.isfinite(value).all() or (value <= 0).any():
        raise ValueError("permittivity must be finite and positive")
    return value


def to_reflection(permittivity, reference):
    eps = _positive(permittivity, permittivity).sqrt()
    ref = _positive(reference, permittivity).sqrt()
    return (eps - ref) / (eps + ref)


def from_reflection(reflection, reference):
    """Strict inverse: no silent clipping or target-dependent background."""
    ref = _positive(reference, reflection)
    if not torch.isfinite(reflection).all() or (reflection.abs() >= 1).any():
        raise ValueError("reflection must be finite and strictly inside (-1, 1)")
    result = ref * ((1 + reflection) / (1 - reflection)).square()
    if not torch.isfinite(result).all():
        raise ValueError("non-finite inverse reflection transform")
    return result


def conv_bn(inputs, outputs, kernel=1):
    # Explicit asymmetric SAME padding for the even 4x4 spatial kernels.
    pad = kernel - 1
    return nn.Sequential(nn.ZeroPad2d((pad // 2, pad - pad // 2) * 2),
                         nn.Conv2d(inputs, outputs, kernel), nn.BatchNorm2d(outputs))


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(channels, max(1, channels // reduction), 1),
                                 nn.ReLU(), nn.Conv2d(max(1, channels // reduction), channels, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        channel = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        x = x * channel.sigmoid()
        spatial = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), 1)
        return x * self.spatial(spatial).sigmoid()


class FusionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.ModuleList([
            nn.Sequential(conv_bn(channels, channels), ChannelSpatialAttention(channels),
                          conv_bn(channels, channels)) for _ in range(2)])
        self.attention_merge = conv_bn(2 * channels, channels)
        self.spatial = nn.ModuleList([conv_bn(channels, channels) for _ in range(2)])
        self.spatial_merge = nn.Sequential(conv_bn(2 * channels, channels),
                                            conv_bn(channels, channels, 4),
                                            conv_bn(channels, channels, 4))

    def forward(self, observed, rtm):
        inputs = (observed, rtm)
        attention = self.attention_merge(torch.cat([
            x + layer(x) for x, layer in zip(inputs, self.attention)], 1))
        spatial = self.spatial_merge(torch.cat([
            layer(x) for x, layer in zip(inputs, self.spatial)], 1))
        return attention + spatial


class FusionGenerator(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        channels = [base_channels * n for n in (1, 2, 4, 8)]
        self.encoders = nn.ModuleList()
        for _ in range(2):
            layers, previous = nn.ModuleList(), 1
            for channel in channels:
                layers.append(nn.Sequential(nn.Conv2d(previous, channel, 4, 2, 1),
                                             nn.BatchNorm2d(channel)))
                previous = channel
            self.encoders.append(layers)
        self.fusions = nn.ModuleList([FusionModule(c) for c in channels])
        self.ups = nn.ModuleList()
        self.merges = nn.ModuleList()
        for previous, channel in zip(reversed(channels[1:]), reversed(channels[:-1])):
            self.ups.append(nn.Sequential(nn.Conv2d(previous, channel * 4, 3, padding=1),
                                           nn.PixelShuffle(2), nn.BatchNorm2d(channel), nn.PReLU()))
            self.merges.append(nn.Sequential(conv_bn(channel * 2, channel), nn.LeakyReLU(.2)))
        self.head = nn.Sequential(nn.Conv2d(channels[0], channels[0] * 4, 3, padding=1),
                                  nn.PixelShuffle(2), nn.BatchNorm2d(channels[0]), nn.PReLU(),
                                  nn.Conv2d(channels[0], 1, 1), nn.Tanh())

    def forward(self, observed, rtm):
        if observed.shape != rtm.shape or observed.ndim != 4 or observed.shape[1:] != (1, 256, 256):
            raise ValueError("both inputs must be matching Bx1x256x256 tensors")
        a, b, fused = observed, rtm, []
        for left, right, fusion in zip(*self.encoders, self.fusions):
            a, b = left(a), right(b)
            fused.append(fusion(a, b))
        x = fused[-1]
        for up, merge, skip in zip(self.ups, self.merges, reversed(fused[:-1])):
            x = merge(torch.cat((up(x), skip), 1))
        return self.head(x)


class FusionDiscriminator(nn.Module):
    """Three-channel conditional PatchGAN; channel widths are declared defaults."""
    def __init__(self, base_channels=64):
        super().__init__()
        layers, previous = [], 3
        for index, channel in enumerate([base_channels, base_channels * 2,
                                         base_channels * 4, base_channels * 8, 1]):
            layers.append(nn.Conv2d(previous, channel, 4, 2 if index < 3 else 1, 1))
            if 0 < index < 4:
                layers.append(nn.BatchNorm2d(channel))
            if index < 4:
                layers.append(nn.LeakyReLU(.2))
            previous = channel
        self.layers = nn.Sequential(*layers)

    def forward(self, observed, rtm, reflection):
        if observed.shape != rtm.shape or observed.shape != reflection.shape:
            raise ValueError("all three discriminator images must have matching shapes")
        return self.layers(torch.cat((observed, rtm, reflection), 1))


def reflection_ssim_loss(prediction, target):
    """Differentiable Gaussian SSIM in reflection units, range=2, valid window."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("expected matching Bx1xHxW images")
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("images must be at least 11x11")
    coordinates = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    gaussian = torch.exp(-coordinates.square() / (2 * 1.5 ** 2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian)[None, None]
    mx, my = F.conv2d(prediction, window), F.conv2d(target, window)
    vx = F.conv2d(prediction.square(), window) - mx.square()
    vy = F.conv2d(target.square(), window) - my.square()
    cov = F.conv2d(prediction * target, window) - mx * my
    numerator = (2 * mx * my + .02 ** 2) * (2 * cov + .06 ** 2)
    denominator = (mx.square() + my.square() + .02 ** 2) * (vx + vy + .06 ** 2)
    return 1 - (numerator / denominator.clamp_min(1e-12)).mean()


def train_step(generator, discriminator, optimizer_g, optimizer_d, observed, rtm, target):
    """One D then non-saturating G update. D buffers are frozen during G update."""
    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)
    fake = generator(observed, rtm)
    real_logits = discriminator(observed, rtm, target)
    fake_logits = discriminator(observed, rtm, fake.detach())
    loss_d = .5 * (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                  + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
    if not torch.isfinite(loss_d):
        raise FloatingPointError("non-finite discriminator loss")
    loss_d.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in discriminator.parameters()):
        raise FloatingPointError("non-finite discriminator gradient")
    optimizer_d.step()
    optimizer_d.zero_grad(set_to_none=True)
    flags = [p.requires_grad for p in discriminator.parameters()]
    modes = [(module, module.training) for module in discriminator.modules()]
    try:
        discriminator.requires_grad_(False)
        discriminator.eval()
        logits = discriminator(observed, rtm, fake)
        gan = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        ssim = reflection_ssim_loss(fake, target)
        mse = F.mse_loss(fake, target)
        loss_g = gan + 50 * ssim + 50 * mse
        if not torch.isfinite(loss_g):
            raise FloatingPointError("non-finite generator loss")
        loss_g.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in generator.parameters()):
            raise FloatingPointError("non-finite generator gradient")
        optimizer_g.step()
    finally:
        for parameter, flag in zip(discriminator.parameters(), flags):
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
    return {"generator": float(loss_g.detach()), "discriminator": float(loss_d.detach()),
            "adversarial": float(gan.detach()), "reflection_ssim_loss": float(ssim.detach()),
            "reflection_mse": float(mse.detach())}
