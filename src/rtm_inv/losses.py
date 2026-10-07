"""Loss functions for RTM closed-loop inversion.

Provides:

- ``PermittivityLoss``: composite model-space loss (L1 + MSE + SSIM + gradient),
  with residual weighting based on ``|target - initial_model|``.
- ``BscanLoss``: composite data-space loss (L1 + MSE + gradient) for comparing
  synthetic and observed B-scan receiver data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bscan import preprocess_bscan, robust_bscan_scale


# ---------------------------------------------------------------------------
#  Helper components
# ---------------------------------------------------------------------------


def _gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 difference of horizontal and vertical spatial gradients.

    This term penalises blurry interfaces and encourages sharp material-property
    discontinuities at layer boundaries.
    """
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    tgt_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dz = pred[..., 1:, :] - pred[..., :-1, :]
    tgt_dz = target[..., 1:, :] - target[..., :-1, :]
    return torch.mean(torch.abs(pred_dx - tgt_dx)) + torch.mean(
        torch.abs(pred_dz - tgt_dz)
    )


def _build_residual_weight(
    target: torch.Tensor,
    initial_model: torch.Tensor,
    alpha: float,
    eps: float,
    max_weight: float | None = None,
) -> torch.Tensor:
    """Build a mean-normalised weight map from the required model correction."""
    residual = (target - initial_model).abs().detach()
    residual_mean = residual.mean(dim=(-2, -1), keepdim=True)
    residual_norm = residual / (residual_mean + eps)
    weight = 1.0 + alpha * residual_norm
    if max_weight is not None and max_weight > 0:
        weight = torch.clamp(weight, min=1.0, max=float(max_weight))
    return weight / (weight.mean(dim=(-2, -1), keepdim=True) + eps)


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Differentiable structural-similarity index (3x3 average-pool approximation).

    Returns a scalar in [0, 1]; 1 means identical structure.
    """
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)

    sigma_x = F.relu(F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x)
    sigma_y = F.relu(F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y)
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-12
    )
    return ssim_map.mean()


# ---------------------------------------------------------------------------
#  Composite model-space loss
# ---------------------------------------------------------------------------


class PermittivityLoss(nn.Module):
    """Weighted composite loss for permittivity-field prediction.

    The gradient term directly penalises differences in spatial gradients,
    helping the network produce sharp layer interfaces instead of over-
    smoothed transitions. SSIM (when enabled) adds a perceptually
    motivated structural constraint. The pixel-space L1 and MSE terms are
    residually weighted using ``|target - initial_model|``.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.3,
        ssim_weight: float = 0.0,
        grad_weight: float = 0.2,
        ssim_data_range: float = 1.0,
        residual_weight_alpha: float = 1.0,
        residual_weight_max: float | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.mse_weight = float(mse_weight)
        self.ssim_weight = float(ssim_weight)
        self.grad_weight = float(grad_weight)
        self.ssim_data_range = float(ssim_data_range)
        self.residual_weight_alpha = float(residual_weight_alpha)
        self.residual_weight_max = None if residual_weight_max is None else float(residual_weight_max)
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        initial_model: torch.Tensor | None = None,
        return_breakdown: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if initial_model is None:
            l1 = F.l1_loss(pred, target)
            mse = F.mse_loss(pred, target)
        else:
            weight = _build_residual_weight(
                target=target,
                initial_model=initial_model,
                alpha=self.residual_weight_alpha,
                eps=self.eps,
                max_weight=self.residual_weight_max,
            )
            diff = pred - target
            l1 = (weight * diff.abs()).mean()
            mse = (weight * diff.square()).mean()
        ssim = 1.0 - _ssim(pred, target, self.ssim_data_range)
        grad = _gradient_l1_loss(pred, target)

        w_l1 = self.l1_weight * l1
        w_mse = self.mse_weight * mse
        w_ssim = self.ssim_weight * ssim
        w_grad = self.grad_weight * grad
        total = w_l1 + w_mse + w_ssim + w_grad

        if return_breakdown:
            info: dict[str, torch.Tensor] = {
                "l1": w_l1.detach(),
                "mse": w_mse.detach(),
                "grad": w_grad.detach(),
            }
            if self.ssim_weight > 0:
                info["ssim"] = w_ssim.detach()
            return total, info
        return total


# ---------------------------------------------------------------------------
#  Composite data-space loss (B-scan)
# ---------------------------------------------------------------------------


class BscanLoss(nn.Module):
    """Weighted composite loss for B-scan receiver data.

    Compares synthetic and observed B-scan traces using L1, MSE, and
    spatial-gradient consistency.  The singleton receiver-count dimension
    (size 1) is squeezed so that gradient penalties are computed along the
    shot and time axes.

    The observed target defines the amplitude scale. Synthetic predictions
    are preprocessed in the same way as the observed data and divided by the
    observed robust scale, so amplitude errors are not hidden by normalising
    each prediction independently.

    Unlike ``PermittivityLoss`` there is no residual weighting or SSIM
    term --- the data-space loss is meant to be a simple, physically
    motivated constraint.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 0.3,
        grad_weight: float = 0.2,
        scale_quantile: float = 0.95,
        scale_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.mse_weight = float(mse_weight)
        self.grad_weight = float(grad_weight)
        self.scale_quantile = float(scale_quantile)
        self.scale_eps = float(scale_eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_scale: torch.Tensor | None = None,
        target_is_processed: bool | torch.Tensor = False,
        return_breakdown: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pred_processed = preprocess_bscan(pred)

        if torch.is_tensor(target_is_processed):
            target_is_processed_bool = bool(torch.all(target_is_processed).item())
        else:
            target_is_processed_bool = bool(target_is_processed)

        if target_is_processed_bool:
            target_scaled = target.float()
            if target_scale is None:
                target_scale = torch.ones(
                    (target.shape[0], 1, 1, 1) if target.ndim == 4 else (1, 1, 1),
                    dtype=target.dtype,
                    device=target.device,
                )
            pred_scaled = pred_processed / target_scale.to(pred_processed.device)
        else:
            target_processed = preprocess_bscan(target)
            if target_scale is None:
                target_scale = robust_bscan_scale(
                    target_processed.detach(),
                    quantile=self.scale_quantile,
                    eps=self.scale_eps,
                )
            target_scale = target_scale.to(target_processed.device)
            pred_scaled = pred_processed / target_scale
            target_scaled = target_processed / target_scale

        # Squeeze the singleton receiver-count dimension so that the
        # gradient loss operates on (shot, time) axes.
        if pred_scaled.shape[-2] == 1:
            pred_scaled = pred_scaled.squeeze(-2)
        if target_scaled.shape[-2] == 1:
            target_scaled = target_scaled.squeeze(-2)

        l1 = F.l1_loss(pred_scaled, target_scaled)
        mse = F.mse_loss(pred_scaled, target_scaled)
        grad = _gradient_l1_loss(pred_scaled, target_scaled)

        w_l1 = self.l1_weight * l1
        w_mse = self.mse_weight * mse
        w_grad = self.grad_weight * grad
        total = w_l1 + w_mse + w_grad

        if return_breakdown:
            info: dict[str, torch.Tensor] = {
                "l1": w_l1.detach(),
                "mse": w_mse.detach(),
                "grad": w_grad.detach(),
            }
            return total, info
        return total
