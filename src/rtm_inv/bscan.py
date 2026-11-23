from __future__ import annotations

import torch


def preprocess_bscan(data: torch.Tensor) -> torch.Tensor:
    """Apply B-scan preprocessing used before RTM/data-loss comparisons.

    Supported tensor layouts are ``(shot, receiver, time)`` and
    ``(batch, shot, receiver, time)``. Processing removes each trace DC
    component and then subtracts the horizontal median trace at every time
    sample.
    """
    if data.ndim not in (3, 4):
        raise ValueError(f"Expected a 3D/4D B-scan tensor, got {tuple(data.shape)}")

    out = data.float()
    out = out - out.mean(dim=-1, keepdim=True)
    shot_dim = 1 if out.ndim == 4 else 0
    median_trace = out.median(dim=shot_dim, keepdim=True).values
    return out - median_trace


def robust_bscan_scale(
    data: torch.Tensor,
    quantile: float = 0.95,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return q-percentile absolute amplitude scale per sample."""
    if data.ndim == 4:
        batch_size = data.shape[0]
        flat = data.reshape(batch_size, -1)
        view_shape = (batch_size, 1, 1, 1)
    elif data.ndim == 3:
        flat = data.reshape(1, -1)
        view_shape = (1, 1, 1)
    else:
        raise ValueError(f"Expected a 3D/4D B-scan tensor, got {tuple(data.shape)}")

    scale = torch.quantile(flat.abs(), float(quantile), dim=1)
    return scale.clamp_min(float(eps)).view(*view_shape)


def normalise_bscan(
    data: torch.Tensor,
    scale: torch.Tensor | None = None,
    *,
    quantile: float = 0.95,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalise a B-scan and return ``(normalised, scale)``."""
    if scale is None:
        scale = robust_bscan_scale(data, quantile=quantile, eps=eps)
    return data / scale, scale
