from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtm import RTMOperator
from .stages import validate_schedule


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        group_count = min(8, out_channels)
        while out_channels % group_count != 0:
            group_count -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GroupNorm(group_count, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
            ),
            nn.GroupNorm(group_count, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UpdateNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = int(depth)
        self.inc = ConvBlock(in_channels, base_channels)
        channels = [base_channels * (2**level) for level in range(self.depth + 1)]
        self.downs = nn.ModuleList(
            DownBlock(channels[level], channels[level + 1])
            for level in range(self.depth)
        )
        self.ups = nn.ModuleList(
            UpBlock(channels[level + 1], channels[level], channels[level])
            for level in reversed(range(self.depth))
        )
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # Backward compatibility for checkpoints saved before UpdateNet became
        # depth-configurable. Old depth=2 modules were named down1/down2/up1/up2.
        legacy_map = {
            "down1.": "downs.0.",
            "down2.": "downs.1.",
            "up1.": "ups.0.",
            "up2.": "ups.1.",
        }
        for old, new in legacy_map.items():
            old_prefix = prefix + old
            new_prefix = prefix + new
            for key in list(state_dict.keys()):
                if key.startswith(old_prefix):
                    state_dict[new_prefix + key[len(old_prefix):]] = state_dict[key]
                    state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)
        return self.outc(x)


class UNetPlusPlusUpdateNet(nn.Module):
    """UNet++ style update network with dense decoder skip connections."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        depth: int = 3,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = int(depth)
        channels = [base_channels * (2**level) for level in range(self.depth + 1)]
        self.encoder = nn.ModuleList()
        for level in range(self.depth + 1):
            level_in = in_channels if level == 0 else channels[level - 1]
            self.encoder.append(ConvBlock(level_in, channels[level]))
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.decoder = nn.ModuleDict()
        for nested_level in range(1, self.depth + 1):
            for level in range(self.depth - nested_level + 1):
                in_ch = channels[level] * nested_level + channels[level + 1]
                self.decoder[f"{level}_{nested_level}"] = ConvBlock(in_ch, channels[level])

        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes: dict[tuple[int, int], torch.Tensor] = {}
        current = x
        for level, encoder in enumerate(self.encoder):
            if level > 0:
                current = self.pool(current)
            current = encoder(current)
            nodes[(level, 0)] = current

        for nested_level in range(1, self.depth + 1):
            for level in range(self.depth - nested_level + 1):
                lateral = [nodes[(level, prev_level)] for prev_level in range(nested_level)]
                up = self._resize_like(nodes[(level + 1, nested_level - 1)], lateral[0])
                nodes[(level, nested_level)] = self.decoder[f"{level}_{nested_level}"](
                    torch.cat([*lateral, up], dim=1)
                )
        return self.outc(nodes[(0, self.depth)])


def _valid_attention_heads(channels: int, max_heads: int = 8) -> int:
    for heads in range(min(max_heads, channels), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class TransUNetUpdateNet(nn.Module):
    """Lightweight TransUNet-style update network.

    A convolutional encoder builds local features, a transformer encoder mixes
    global context at the bottleneck, and a U-Net decoder restores resolution.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        depth: int = 3,
        transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = int(depth)
        self.inc = ConvBlock(in_channels, base_channels)
        channels = [base_channels * (2**level) for level in range(self.depth + 1)]
        self.downs = nn.ModuleList(
            DownBlock(channels[level], channels[level + 1])
            for level in range(self.depth)
        )
        bottleneck_channels = channels[-1]
        self.pos_proj = nn.Conv2d(2, bottleneck_channels, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bottleneck_channels,
            nhead=_valid_attention_heads(bottleneck_channels),
            dim_feedforward=bottleneck_channels * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))
        self.ups = nn.ModuleList(
            UpBlock(channels[level + 1], channels[level], channels[level])
            for level in reversed(range(self.depth))
        )
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    @staticmethod
    def _position_grid(
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([yy, xx], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)

        batch_size, channels, height, width = x.shape
        pos = self._position_grid(batch_size, height, width, x.device, x.dtype)
        x = x + self.pos_proj(pos)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        x = tokens.transpose(1, 2).reshape(batch_size, channels, height, width)

        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)
        return self.outc(x)


class GatedBscanUpdateNet(UpdateNet):
    """RTM/m0 U-Net whose stem is gated by an independent B-scan encoder.

    The B-scan is encoded in its native (time, shot) coordinates and only then
    resampled onto the model grid, instead of being resized and concatenated as if
    time and depth were the same axis. The encoded features enter after the first
    conv block through a learned gate. The projection is zero-initialised, so the
    network starts as the RTM-only update network.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        depth: int = 3,
        bscan_channels: int = 32,
    ) -> None:
        super().__init__(in_channels=in_channels, base_channels=base_channels, depth=depth)
        self.bscan_channels = int(bscan_channels)

        def block(cin: int, cout: int, stride: tuple[int, int]) -> nn.Sequential:
            groups = min(8, cout)
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, padding_mode="replicate"),
                nn.GroupNorm(groups, cout),
                nn.GELU(),
            )

        c = self.bscan_channels
        self.bscan_encoder = nn.Sequential(
            block(1, c // 2, (2, 1)),
            block(c // 2, c, (2, 1)),
            block(c, c, (2, 1)),
        )
        self.bscan_refine = block(c, c, (1, 1))
        self.gate = nn.Conv2d(base_channels + c, base_channels, kernel_size=1)
        self.proj = nn.Conv2d(c, base_channels, kernel_size=1)
        nn.init.constant_(self.gate.bias, -3.0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.last_gate_mean: float | None = None

    def forward(self, x: torch.Tensor, bscan: torch.Tensor) -> torch.Tensor:
        stem = self.inc(x)
        features = self.bscan_encoder(bscan)
        features = F.interpolate(
            features, size=stem.shape[-2:], mode="bilinear", align_corners=False
        )
        features = self.bscan_refine(features)
        gate = torch.sigmoid(self.gate(torch.cat([stem, features], dim=1)))
        self.last_gate_mean = float(gate.detach().mean())
        x = stem + gate * self.proj(features)
        skips = [x]
        for down in self.downs:
            x = down(x)
            skips.append(x)
        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)
        return self.outc(x)


def build_update_net(
    backbone: str,
    in_channels: int,
    base_channels: int,
    depth: int,
) -> nn.Module:
    if backbone == "unet":
        return UpdateNet(in_channels=in_channels, base_channels=base_channels, depth=depth)
    if backbone == "unetpp":
        return UNetPlusPlusUpdateNet(in_channels=in_channels, base_channels=base_channels, depth=depth)
    if backbone == "transunet":
        return TransUNetUpdateNet(in_channels=in_channels, base_channels=base_channels, depth=depth)
    raise ValueError("update_backbone must be one of ['unet', 'unetpp', 'transunet']")


class RTMInvNet(nn.Module):
    """Closed-loop RTM inversion network.

    The network operates in *normalised* model space (values in [0, 1]).
    Before calling the RTM / forward-modelling operator, models are
    denormalised back to physical permittivity scale using fixed global
    bounds (model_min, model_max).
    """

    # Global normalisation bounds (permittivity scale).
    MODEL_MIN = 2.0
    MODEL_MAX = 10.0
    MODEL_SCALE = MODEL_MAX - MODEL_MIN  # 8.0
    IMAGE_SCALE_QUANTILE = 0.95
    IMAGE_CLAMP = 5.0
    INPUT_MODES = {"m0_bscan", "m0_rtm", "m0_rtm_bscan", "m0_rtm_bscan_gated", "rtm"}
    UPDATE_BACKBONES = {"unet", "unetpp", "transunet"}

    def __init__(
        self,
        rtm_operator: RTMOperator,
        compute_synthetic_data: bool = False,
        input_mode: str = "m0_rtm",
        unet_base_channels: int = 64,
        unet_depth: int = 3,
        num_stages: int = 2,
        update_backbone: str = "unet",
        recompute_rtm_between_stages: bool = True,
    ) -> None:
        super().__init__()
        if input_mode not in self.INPUT_MODES:
            raise ValueError(
                f"input_mode must be one of {sorted(self.INPUT_MODES)}, "
                f"got {input_mode!r}"
            )
        if update_backbone not in self.UPDATE_BACKBONES:
            raise ValueError(
                f"update_backbone must be one of {sorted(self.UPDATE_BACKBONES)}, "
                f"got {update_backbone!r}"
            )
        if num_stages < 1:
            raise ValueError(f"num_stages must be at least 1, got {num_stages}")
        if num_stages > 1:
            validate_schedule(int(num_stages))
        self.rtm_operator = rtm_operator
        self.compute_synthetic_data = compute_synthetic_data
        self.input_mode = input_mode
        self.unet_base_channels = int(unet_base_channels)
        self.unet_depth = int(unet_depth)
        self.num_stages = int(num_stages)
        self.update_backbone = update_backbone
        self.recompute_rtm_between_stages = bool(recompute_rtm_between_stages)
        input_channels = {
            "m0_rtm": 2,
            "rtm": 1,
            "m0_bscan": 2,
            "m0_rtm_bscan": 3,
            "m0_rtm_bscan_gated": 2,
        }
        in_channels = input_channels[input_mode]
        if input_mode == "m0_rtm_bscan_gated":
            if update_backbone != "unet":
                raise ValueError("m0_rtm_bscan_gated requires update_backbone='unet'")
            self.update_net = GatedBscanUpdateNet(
                in_channels=in_channels,
                base_channels=self.unet_base_channels,
                depth=self.unet_depth,
            )
        else:
            self.update_net = build_update_net(
                backbone=self.update_backbone,
                in_channels=in_channels,
                base_channels=self.unet_base_channels,
                depth=self.unet_depth,
            )

    @classmethod
    def _denormalise(cls, model: torch.Tensor) -> torch.Tensor:
        """Convert normalised [0, 1] model → physical permittivity scale."""
        return model * cls.MODEL_SCALE + cls.MODEL_MIN

    @classmethod
    def _normalise(cls, model: torch.Tensor) -> torch.Tensor:
        """Convert physical permittivity scale → normalised [0, 1] model."""
        return (model - cls.MODEL_MIN) / cls.MODEL_SCALE

    @classmethod
    def _normalise_image(cls, image: torch.Tensor) -> torch.Tensor:
        """Robustly scale each RTM image before it enters the update network."""
        if image.ndim != 4:
            raise ValueError(f"Expected RTM image shape (batch, channel, z, x), got {tuple(image.shape)}")
        flat = image.detach().abs().reshape(image.shape[0], -1)
        scale = torch.quantile(flat, cls.IMAGE_SCALE_QUANTILE, dim=1)
        scale = scale.clamp_min(1e-12).view(image.shape[0], 1, 1, 1)
        image = torch.clamp(image / scale, min=-cls.IMAGE_CLAMP, max=cls.IMAGE_CLAMP)
        return image / cls.IMAGE_CLAMP

    @classmethod
    def _bscan_to_model_image(cls, observed_bscan: torch.Tensor, target_size: tuple[int, int] | None) -> torch.Tensor:
        if observed_bscan.ndim == 2:
            # B-scan storage is (shot, time). Model images are (depth, x),
            # so map time -> depth and shot -> x before resizing.
            image = observed_bscan.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        elif observed_bscan.ndim == 3:
            # Batched B-scan without the singleton receiver dimension:
            # (batch, shot, time) -> (batch, channel, time, shot).
            image = observed_bscan.transpose(-2, -1).unsqueeze(1)
        elif observed_bscan.ndim == 4:
            # Standard layout is (batch, shot, receiver, time). Average any
            # receiver channels, then orient the image as (time, shot).
            image = observed_bscan.mean(dim=2).transpose(-2, -1).unsqueeze(1)
        else:
            raise ValueError(f"Expected B-scan tensor with 2-4 dims, got {tuple(observed_bscan.shape)}")

        image = image.float()
        flat = image.detach().abs().reshape(image.shape[0], -1)
        scale = torch.quantile(flat, cls.IMAGE_SCALE_QUANTILE, dim=1)
        scale = scale.clamp_min(1e-12).view(image.shape[0], 1, 1, 1)
        image = torch.clamp(image / scale, min=-cls.IMAGE_CLAMP, max=cls.IMAGE_CLAMP)
        image = image / cls.IMAGE_CLAMP
        if target_size is not None and image.shape[-2:] != target_size:
            image = F.interpolate(image, size=target_size, mode="bilinear", align_corners=False)
        return image

    def _make_update_input(
        self,
        image: torch.Tensor,
        model: torch.Tensor,
        bscan_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_mode in {"m0_rtm", "m0_rtm_bscan_gated"}:
            return torch.cat([image, model], dim=1)
        if self.input_mode == "rtm":
            return image
        if self.input_mode == "m0_bscan":
            if bscan_image is None:
                raise RuntimeError("bscan_image is required for m0_bscan mode")
            return torch.cat([bscan_image, model], dim=1)
        if self.input_mode == "m0_rtm_bscan":
            if bscan_image is None:
                raise RuntimeError("bscan_image is required for m0_rtm_bscan mode")
            return torch.cat([image, bscan_image, model], dim=1)
        raise RuntimeError(f"Unsupported input_mode={self.input_mode!r}")

    def _run_stage(
        self,
        observed_bscan: torch.Tensor,
        current_model: torch.Tensor,
        sample_config: torch.Tensor | None = None,
        observed_is_processed: bool | torch.Tensor = False,
        compute_synthetic_data: bool | None = None,
        precomputed_image: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Denormalise → physical scale for RTM / forward modelling
        current_phys = self._denormalise(current_model).detach()
        needs_rtm = self.input_mode in {"m0_rtm", "rtm", "m0_rtm_bscan", "m0_rtm_bscan_gated"}
        needs_bscan = self.input_mode in {"m0_bscan", "m0_rtm_bscan", "m0_rtm_bscan_gated"}
        if precomputed_image is not None:
            if not needs_rtm:
                raise ValueError("precomputed_image is only valid for RTM input modes")
            image = precomputed_image
        elif not needs_rtm:
            image = torch.zeros_like(current_model)
        else:
            with torch.no_grad():
                image = self.rtm_operator(
                    observed_bscan,
                    current_phys,
                    observed_is_processed=observed_is_processed,
                    sample_config=sample_config,
                )
            image = self._normalise_image(image)
        bscan_image = None
        if needs_bscan:
            native = self.input_mode == "m0_rtm_bscan_gated"
            bscan_image = self._bscan_to_model_image(
                observed_bscan,
                target_size=None if native else current_model.shape[-2:],
            )
        update_input = self._make_update_input(image, current_model, bscan_image=bscan_image)
        if self.input_mode == "m0_rtm_bscan_gated":
            delta = self.update_net(update_input, bscan_image)
        else:
            delta = self.update_net(update_input)
        updated_model = torch.clamp(current_model + delta, min=0.0, max=1.0)
        result: dict[str, torch.Tensor] = {
            "image": image,
            "delta": delta,
            "updated_model": updated_model,
        }
        should_compute_synthetic = self.compute_synthetic_data if compute_synthetic_data is None else compute_synthetic_data
        if should_compute_synthetic:
            n_shots = int(observed_bscan.shape[1]) if observed_bscan.ndim == 4 else int(observed_bscan.shape[0])
            nt = int(observed_bscan.shape[-1])
            updated_phys = self._denormalise(updated_model)
            synth = self.rtm_operator.synthesize_data(
                updated_phys,
                n_shots=n_shots,
                nt=nt,
                sample_config=sample_config,
            )
            result["synthetic_data"] = synth
        return result

    def forward(
        self,
        observed_bscan: torch.Tensor,
        initial_model: torch.Tensor,
        sample_config: torch.Tensor | None = None,
        observed_is_processed: bool | torch.Tensor = False,
    ) -> dict[str, torch.Tensor | int | None]:
        current_model = initial_model
        outputs: dict[str, torch.Tensor | int | None] = {"num_stages": self.num_stages}
        first_stage_image = None
        for stage_idx in range(1, self.num_stages + 1):
            is_final = stage_idx == self.num_stages
            reuse_first_image = (
                stage_idx > 1
                and not self.recompute_rtm_between_stages
                and self.input_mode in {"m0_rtm", "rtm", "m0_rtm_bscan", "m0_rtm_bscan_gated"}
                and first_stage_image is not None
            )
            stage = self._run_stage(
                observed_bscan,
                current_model,
                sample_config=sample_config,
                observed_is_processed=observed_is_processed,
                compute_synthetic_data=self.compute_synthetic_data and is_final,
                precomputed_image=first_stage_image if reuse_first_image else None,
            )
            if stage_idx == 1:
                first_stage_image = stage["image"]
            current_model = stage["updated_model"]
            outputs[f"stage{stage_idx}_image"] = stage["image"]
            outputs[f"stage{stage_idx}_delta"] = stage["delta"]
            outputs[f"stage{stage_idx}_model"] = current_model
            outputs[f"stage{stage_idx}_synthetic_data"] = stage.get("synthetic_data")
            if not is_final and torch.cuda.is_available():
                torch.cuda.empty_cache()

        outputs["final_model"] = current_model
        outputs["final_synthetic_data"] = outputs.get(f"stage{self.num_stages}_synthetic_data")
        return outputs
