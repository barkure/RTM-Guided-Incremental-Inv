"""Declared data/loss adapters for external baselines; no changes to RTM inputs."""
import torch
from torch.nn import functional as F

from .models import RTMInvNet


def pinet_inputs(observed):
    """Batched shot/receiver/time data → paper time/trace dimensions."""
    if observed.ndim != 4:
        raise ValueError("expected Bxshotxreceiverxtime")
    return RTMInvNet._bscan_to_model_image(observed, (1700, 199))


def pinet_targets(normalized_model):
    if normalized_model.ndim != 4 or normalized_model.shape[1] != 1:
        raise ValueError("expected Bx1xdepthxwidth")
    return F.interpolate(normalized_model, (220, 420), mode="bilinear", align_corners=False)


def normalized_dssim(prediction, target):
    """1−SSIM (NOT divided by 2), fixed normalized range 1, Gaussian 11/1.5."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError("expected matching Bx1xHxW images")
    if min(prediction.shape[-2:]) < 11:
        raise ValueError("images smaller than the SSIM window")
    coordinates = torch.arange(11, dtype=prediction.dtype, device=prediction.device) - 5
    gaussian = torch.exp(-coordinates.square() / (2 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    window = torch.outer(gaussian, gaussian)[None, None]
    mx, my = F.conv2d(prediction, window), F.conv2d(target, window)
    vx = F.conv2d(prediction.square(), window) - mx.square()
    vy = F.conv2d(target.square(), window) - my.square()
    cov = F.conv2d(prediction * target, window) - mx * my
    numerator = (2 * mx * my + .01**2) * (2 * cov + .03**2)
    denominator = (mx.square() + my.square() + .01**2) * (vx + vy + .03**2)
    return 1 - (numerator / denominator.clamp_min(1e-12)).mean()
