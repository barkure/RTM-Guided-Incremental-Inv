from __future__ import annotations

import math

import torch
import torch.nn.functional as F


SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5


def peak_signal_noise_ratio(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
) -> float:
    """Return PSNR using a fixed, explicitly supplied signal range."""
    if data_range <= 0.0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    mse = float(F.mse_loss(pred.float(), target.float()).item())
    if mse == 0.0:
        return float("inf")
    return float(20.0 * math.log10(data_range) - 10.0 * math.log10(mse))


def structural_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float,
    window_size: int = SSIM_WINDOW_SIZE,
    sigma: float = SSIM_SIGMA,
) -> float:
    """Return single-scale SSIM with the standard Gaussian local window.

    The implementation follows the conventional Wang et al. formulation with
    an 11 x 11 Gaussian window (sigma 1.5) and population local moments.  A
    fixed ``data_range`` keeps values comparable across samples.
    """
    if data_range <= 0.0:
        raise ValueError(f"data_range must be positive, got {data_range}")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(
            f"window_size must be a positive odd integer, got {window_size}"
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if pred.shape != target.shape:
        raise ValueError(
            "pred and target must have the same shape, got "
            f"{tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.ndim != 2:
        raise ValueError(f"Expected 2-D model images, got shape {tuple(pred.shape)}")
    if min(pred.shape) < window_size:
        raise ValueError(
            f"Image shape {tuple(pred.shape)} is smaller than the "
            f"{window_size} x {window_size} SSIM window"
        )

    pred_4d = pred.float().unsqueeze(0).unsqueeze(0)
    target_4d = target.float().unsqueeze(0).unsqueeze(0)
    coordinates = torch.arange(
        window_size,
        device=pred.device,
        dtype=pred_4d.dtype,
    )
    coordinates = coordinates - (window_size - 1) / 2.0
    gaussian_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    window = torch.outer(gaussian_1d, gaussian_1d).view(1, 1, window_size, window_size)

    mu_x = F.conv2d(pred_4d, window)
    mu_y = F.conv2d(target_4d, window)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(pred_4d.square(), window) - mu_x_sq
    sigma_y = F.conv2d(target_4d.square(), window) - mu_y_sq
    sigma_xy = F.conv2d(pred_4d * target_4d, window) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / denominator.clamp_min(1e-12)).mean().item())
