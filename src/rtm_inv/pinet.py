"""PINet paper-derived implementation candidate (not verified author code).

The fixed dimensions follow Fig. 2; discretionary choices are documented in
docs/stage7_pinet_design.md. No RTM calls or changes to the main RTM model.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ChannelWiseLinear(nn.Module):
    """An independent dense map per channel, not a shared Linear over all channels."""
    def __init__(self, channels, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, inputs, outputs))
        self.bias = nn.Parameter(torch.empty(channels, outputs))
        bound = 1 / math.sqrt(inputs)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != self.weight.shape[:2]:
            raise ValueError("expected Bxchannelsxfeatures matching channel-wise weights")
        return torch.einsum("bci,cio->bco", x, self.weight) + self.bias


def conv_block(inputs, outputs, kernel, stride, padding):
    return nn.Sequential(nn.Conv2d(inputs, outputs, kernel, stride, padding),
                         nn.BatchNorm2d(outputs), nn.ReLU())


class PINet(nn.Module):
    def __init__(self, channels=(16, 32, 64, 64, 128, 128),
                 decoder_channels=(128, 64, 32, 32, 32, 1),
                 fc_hidden=(1024, 512, 256), dropout=.2):
        super().__init__()
        if len(channels) != 6 or len(decoder_channels) != 6 or decoder_channels[-1] != 1 or len(fc_hidden) != 3:
            raise ValueError("PINet requires six encoder/decoder blocks and four CwFC layers")
        self.encoders = nn.ModuleList()
        previous = 1
        for index, channel in enumerate(channels):
            kernel, stride, padding = ((3, 1), (2, 1), (1, 0)) if index < 4 else (3, 2, 1)
            self.encoders.append(nn.Sequential(conv_block(previous, channel, kernel, stride, padding),
                                                conv_block(channel, channel, 3, 1, 1)))
            previous = channel
        fc_sizes = (1350, *fc_hidden, 338)
        layers = []
        for inputs, outputs in zip(fc_sizes[:-1], fc_sizes[1:]):
            layers.extend([ChannelWiseLinear(previous, inputs, outputs), nn.ReLU(), nn.Dropout(dropout)])
        self.global_encoder = nn.Sequential(*layers)
        self.decoders = nn.ModuleList()
        for index, channel in enumerate(decoder_channels):
            if index < 4:
                block = nn.Sequential(nn.ConvTranspose2d(previous, channel, 3, 2, 1, output_padding=1),
                                      nn.BatchNorm2d(channel), nn.ReLU(), conv_block(channel, channel, 3, 1, 1))
            else:
                block = conv_block(previous, channel, 3, 1, 1)
            self.decoders.append(block)
            previous = channel

    def forward(self, observed):
        if observed.ndim != 4 or observed.shape[1:] != (1, 1700, 199):
            raise ValueError("expected Bx1x1700x199, with time before trace")
        x = observed
        for encoder in self.encoders:
            x = encoder(x)
        if x.shape[-2:] != (27, 50):
            raise RuntimeError("paper compression shape violated")
        batch, channels = x.shape[:2]
        x = self.global_encoder(x.flatten(2)).reshape(batch, channels, 13, 26)
        for decoder in self.decoders:
            x = decoder(x)
        return F.interpolate(x, (220, 420), mode="bilinear", align_corners=False)
