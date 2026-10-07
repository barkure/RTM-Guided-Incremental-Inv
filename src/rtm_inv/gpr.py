from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class GPRSurveyConfig:
    dx: float = 0.04
    dz: float = 0.04
    dt: float = 1.0e-10 * 1e6
    freq: float = 400e6 / 1e6
    src_x_start: float = 0.2
    src_x_step: float = 0.1
    src_z: float = 0.04
    rec_offset_x: float = 0.05
    rec_z: float = 0.04
    pml_width: int = 20
    accuracy: int = 4
    scale: float = 1e6
    model_kind: str = "permittivity"
    bscan_layout: str = "time_by_trace"


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def blackharrispulse(freq: float, t: np.ndarray) -> np.ndarray:
    coeffs = [0.35322222, -0.488, 0.145, -0.010222222]
    duration = 1.14 / freq
    window = np.zeros_like(t)
    mask = t < duration
    t_active = t[mask]

    values = np.zeros_like(t_active)
    for order, coeff in enumerate(coeffs):
        values += coeff * np.cos(2 * order * np.pi * t_active / duration)

    window[mask] = values
    shifted = np.concatenate([window[1:], [0]])
    pulse = shifted - window
    max_abs = np.max(np.abs(pulse))
    if max_abs != 0:
        pulse = pulse / max_abs
    return pulse


def permittivity_to_velocity(
    model: torch.Tensor,
    scale: float = 1e6,
    relative_mu: float = 1.0,
) -> torch.Tensor:
    c0 = 299_792_458.0
    return (c0 / torch.sqrt(model * relative_mu)) / scale


def model_to_velocity(model: torch.Tensor, config: GPRSurveyConfig) -> torch.Tensor:
    if config.model_kind == "velocity":
        return model
    if config.model_kind == "permittivity":
        # epsilon_r=1 represents air in acquisition-geometry perturbation
        # simulations. Training targets still use the configured model range.
        return permittivity_to_velocity(torch.clamp(model, min=1.0, max=20.0), scale=config.scale)
    raise ValueError(f"Unsupported model_kind: {config.model_kind}")


def prepare_observed_data(bscan: torch.Tensor, layout: str) -> torch.Tensor:
    if bscan.ndim == 3 and bscan.shape[0] == 1:
        bscan = bscan.squeeze(0)
    if bscan.ndim != 2:
        raise ValueError(f"Expected a 2D bscan tensor, got shape {tuple(bscan.shape)}")

    if layout == "time_by_trace":
        return bscan.transpose(0, 1).unsqueeze(1).contiguous()
    if layout == "trace_by_time":
        return bscan.unsqueeze(1).contiguous()
    raise ValueError(f"Unsupported bscan_layout: {layout}")


def build_wavelet(nt: int, config: GPRSurveyConfig, device: torch.device) -> torch.Tensor:
    t = np.arange(0, nt * config.dt, config.dt)
    pulse = blackharrispulse(config.freq, t)
    return torch.tensor(pulse, dtype=torch.float32, device=device)


def build_geometry(
    n_shots: int,
    nt: int,
    nx: int,
    nz: int,
    config: GPRSurveyConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_positions = config.src_x_start + np.arange(n_shots) * config.src_x_step
    src_x_idx = np.round(x_positions / config.dx).astype(int)
    rec_x_idx = np.round((x_positions + config.rec_offset_x) / config.dx).astype(int)
    src_z_idx = int(round(config.src_z / config.dz))
    rec_z_idx = int(round(config.rec_z / config.dz))

    src_x_idx = np.clip(src_x_idx, 0, nx - 1)
    rec_x_idx = np.clip(rec_x_idx, 0, nx - 1)
    src_z_idx = int(np.clip(src_z_idx, 0, nz - 1))
    rec_z_idx = int(np.clip(rec_z_idx, 0, nz - 1))

    source_locations = torch.zeros((n_shots, 1, 2), dtype=torch.long, device=device)
    receiver_locations = torch.zeros((n_shots, 1, 2), dtype=torch.long, device=device)

    source_locations[:, 0, 0] = src_z_idx
    source_locations[:, 0, 1] = torch.tensor(src_x_idx, dtype=torch.long, device=device)
    receiver_locations[:, 0, 0] = rec_z_idx
    receiver_locations[:, 0, 1] = torch.tensor(rec_x_idx, dtype=torch.long, device=device)

    wavelet = build_wavelet(nt, config, device)
    source_amplitudes = wavelet.view(1, 1, -1).repeat(n_shots, 1, 1)
    return source_amplitudes, source_locations, receiver_locations
