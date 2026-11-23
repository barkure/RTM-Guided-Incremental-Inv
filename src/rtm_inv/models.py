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
    INPUT_MODES = {"m0_rtm"}
    UPDATE_BACKBONES = {"unet"}

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
        self.update_net = UpdateNet(
            in_channels=2,
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

    def _make_update_input(self, image: torch.Tensor, model: torch.Tensor) -> torch.Tensor:
        return torch.cat([image, model], dim=1)

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
        if precomputed_image is not None:
            image = precomputed_image
        else:
            with torch.no_grad():
                image = self.rtm_operator(
                    observed_bscan,
                    current_phys,
                    observed_is_processed=observed_is_processed,
                    sample_config=sample_config,
                )
            image = self._normalise_image(image)
        delta = self.update_net(self._make_update_input(image, current_model))
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
