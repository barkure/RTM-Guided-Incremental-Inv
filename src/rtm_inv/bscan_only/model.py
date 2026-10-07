from __future__ import annotations

import torch
import torch.nn as nn

from ..models import RTMInvNet, UpdateNet


class BscanOnlyNet(nn.Module):
    """Pure B-scan to model baseline.

    The network input is only the processed B-scan. The initial model is not
    used as an input feature or as a residual-update anchor; it is only a shape
    template so the shared training loop knows the desired output size.
    """

    def __init__(
        self,
        unet_base_channels: int = 64,
        unet_depth: int = 3,
    ) -> None:
        super().__init__()
        self.unet_base_channels = int(unet_base_channels)
        self.unet_depth = int(unet_depth)
        self.net = UpdateNet(
            in_channels=1,
            base_channels=self.unet_base_channels,
            depth=self.unet_depth,
        )

    @staticmethod
    def _bscan_to_image(observed_bscan: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        return RTMInvNet._bscan_to_model_image(observed_bscan, target_size)

    def forward(
        self,
        observed_bscan: torch.Tensor,
        initial_model: torch.Tensor,
        sample_config: torch.Tensor | None = None,
        observed_is_processed: bool | torch.Tensor = False,
    ) -> dict[str, torch.Tensor | int | None | bool]:
        del sample_config, observed_is_processed
        bscan_image = self._bscan_to_image(observed_bscan, target_size=initial_model.shape[-2:])
        predicted_model = torch.sigmoid(self.net(bscan_image))
        zero_delta = torch.zeros_like(predicted_model)
        return {
            "stage1_image": bscan_image,
            "stage1_delta": zero_delta,
            "stage1_model": predicted_model,
            "stage1_synthetic_data": None,
            "stage2_image": bscan_image,
            "stage2_delta": zero_delta,
            "stage2_model": predicted_model,
            "stage2_synthetic_data": None,
            "final_model": predicted_model,
            "final_synthetic_data": None,
            "num_stages": 1,
            "direct_model_prediction": True,
        }
