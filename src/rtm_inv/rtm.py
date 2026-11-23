from __future__ import annotations

from typing import Protocol

import torch

from .deepwave_rtm import DeepwaveClosedLoopRTM


class RTMOperator(Protocol):
    def __call__(
        self,
        observed_bscan: torch.Tensor,
        model: torch.Tensor,
        observed_is_processed: bool | torch.Tensor = False,
        sample_config: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def synthesize_data(
        self,
        model: torch.Tensor,
        n_shots: int | None = None,
        nt: int | None = None,
        sample_config: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


__all__ = ["RTMOperator", "DeepwaveClosedLoopRTM"]
