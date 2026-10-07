from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from deepwave import scalar
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtm_inv.bscan import preprocess_bscan
from rtm_inv.gpr import GPRSurveyConfig, build_geometry, model_to_velocity

ARTIFACTS_ROOT = PROJECT_ROOT / "data" / "artifacts"
ROCK_DATA_DIR = PROJECT_ROOT / "data" / "data"


@dataclass(frozen=True)
class ForwardConfig:
    dx: float
    dz: float
    dt: float
    freq_mhz: float
    nt: int
    src_x_start: float
    src_x_step: float
    src_z: float
    rec_offset_x: float
    rec_z: float
    pml_width: int
    accuracy: int
    scale: float = 1e6

    def survey(self) -> GPRSurveyConfig:
        return GPRSurveyConfig(
            dx=self.dx,
            dz=self.dz,
            dt=self.dt,
            freq=self.freq_mhz,
            src_x_start=self.src_x_start,
            src_x_step=self.src_x_step,
            src_z=self.src_z,
            rec_offset_x=self.rec_offset_x,
            rec_z=self.rec_z,
            pml_width=self.pml_width,
            accuracy=self.accuracy,
            scale=self.scale,
            model_kind="permittivity",
            bscan_layout="time_by_trace",
        )


ROCK_CONFIG = ForwardConfig(
    dx=0.025,
    dz=0.025,
    dt=1.0e-4,
    freq_mhz=400.0,
    nt=512,
    src_x_start=0.2,
    src_x_step=0.075,
    src_z=0.025,
    rec_offset_x=0.025,
    rec_z=0.025,
    pml_width=20,
    accuracy=4,
)

REBAR_CONFIG = ForwardConfig(
    dx=0.01,
    dz=0.004,
    dt=1.953125e-5,
    freq_mhz=1500.0,
    nt=512,
    src_x_start=0.0,
    src_x_step=0.01,
    src_z=0.004,
    rec_offset_x=0.0,
    rec_z=0.004,
    pml_width=32,
    accuracy=4,
)

REBAR_1600_CONFIG = ForwardConfig(
    dx=0.005,
    dz=0.005,
    dt=1.465e-5,
    freq_mhz=1600.0,
    nt=1024,
    src_x_start=0.0,
    src_x_step=0.005,
    src_z=0.005,
    rec_offset_x=0.0,
    rec_z=0.005,
    pml_width=32,
    accuracy=4,
)

REBAR_2600_CONFIG = ForwardConfig(
    dx=0.004,
    dz=0.004,
    dt=1.953125e-5,
    freq_mhz=2600.0,
    nt=512,
    src_x_start=0.0,
    src_x_step=0.004,
    src_z=0.004,
    rec_offset_x=0.0,
    rec_z=0.004,
    pml_width=32,
    accuracy=4,
)


SANDBOX_CONFIG = ForwardConfig(
    dx=0.01,
    dz=0.01,
    dt=9.7754e-6,
    freq_mhz=1600.0,
    nt=1024,
    src_x_start=0.0,
    src_x_step=0.01,
    src_z=0.01,
    rec_offset_x=0.0,
    rec_z=0.01,
    pml_width=20,
    accuracy=4,
)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def output_root(kind: str) -> Path:
    return ARTIFACTS_ROOT / kind


def resolve_output_dir(args: argparse.Namespace, kind: str) -> Path:
    return args.output_dir if args.output_dir is not None else output_root(kind)


def sample_dir(root: Path, index: int) -> Path:
    return root / f"sample_{index:04d}"


def collect_model_files(data_dir: Path) -> list[Path]:
    return sorted(
        data_dir.glob("model_*.mat"), key=lambda path: int(path.stem.split("_")[-1])
    )


def load_mat_model(
    path: Path, config: ForwardConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = loadmat(path)
    permittivity = np.asarray(mat["model"], dtype=np.float32)
    if "x" in mat and "z" in mat:
        x_orig = np.squeeze(mat["x"]).astype(np.float32)
        z_orig = np.squeeze(mat["z"]).astype(np.float32)
    else:
        nz, nx = permittivity.shape
        x_orig = np.linspace(0.0, 8.0, nx, dtype=np.float32)
        z_orig = np.linspace(0.0, 3.0, nz, dtype=np.float32)

    x = np.arange(0.0, 8.0 + config.dx / 10.0, config.dx, dtype=np.float32)
    z = np.arange(0.0, 3.0 + config.dz / 10.0, config.dz, dtype=np.float32)
    grid = RegularGridInterpolator(
        (z_orig, x_orig),
        permittivity,
        method="nearest",
        bounds_error=False,
        fill_value=None,
    )
    xx, zz = np.meshgrid(x, z, indexing="xy")
    model = (
        grid(np.stack([zz.ravel(), xx.ravel()], axis=-1))
        .reshape(len(z), len(x))
        .astype(np.float32)
    )
    return model, x, z


def smooth_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    sigma: tuple[float, float],
    amp: float,
) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    return (gaussian_filter(noise, sigma=sigma) * amp).astype(np.float32)


def make_cross_model(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    config = ROCK_CONFIG
    x = np.arange(0.0, 8.0 + config.dx / 10.0, config.dx, dtype=np.float32)
    z = np.arange(0.0, 3.0 + config.dz / 10.0, config.dz, dtype=np.float32)
    model = np.full((len(z), len(x)), rng.uniform(6.0, 9.0), dtype=np.float32)
    model += smooth_noise(rng, model.shape, sigma=(5, 9), amp=0.35)
    model = np.clip(model, 6.0, 9.0)
    n_cross = int(rng.integers(2, 4))
    xx, zz = np.meshgrid(x, z, indexing="xy")
    params = []
    for _ in range(n_cross):
        cx = float(rng.uniform(0.8, 7.2))
        cz = float(rng.uniform(0.45, 2.4))
        arm_x = float(rng.uniform(0.18, 0.42))
        arm_z = float(rng.uniform(0.18, 0.42))
        width = float(rng.uniform(0.035, 0.08))
        if rng.random() < 0.7:
            eps = float(rng.uniform(10.0, 16.0))
        else:
            eps = float(rng.uniform(2.0, 4.0))
        horiz = (np.abs(zz - cz) < width) & (np.abs(xx - cx) < arm_x)
        vert = (np.abs(xx - cx) < width) & (np.abs(zz - cz) < arm_z)
        mask = horiz | vert
        model[mask] = eps
        params.append(
            {
                "x_m": cx,
                "z_m": cz,
                "arm_x_m": arm_x,
                "arm_z_m": arm_z,
                "width_m": width,
                "eps": eps,
            }
        )
    return np.clip(model, 2.0, 18.0), x, z, {"num_cross": n_cross, "objects": params}


def make_layered_model(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    config = ROCK_CONFIG
    x = np.arange(0.0, 8.0 + config.dx / 10.0, config.dx, dtype=np.float32)
    z = np.arange(0.0, 3.0 + config.dz / 10.0, config.dz, dtype=np.float32)
    zz = np.broadcast_to(z[:, None], (len(z), len(x)))

    n_layers = int(rng.integers(2, 4))
    layer_eps = rng.uniform(4.0, 10.0, size=n_layers).astype(np.float32)
    # Keep neighbouring layers visibly different.
    for i in range(1, n_layers):
        if abs(float(layer_eps[i] - layer_eps[i - 1])) < 1.2:
            layer_eps[i] = np.clip(
                layer_eps[i - 1] + rng.choice([-1.8, 1.8]), 3.0, 11.5
            )

    base_depths = [float(rng.uniform(1.0, 1.65))]
    if n_layers == 3:
        base_depths.append(float(rng.uniform(2.0, 2.45)))

    interfaces = []
    for base in base_depths:
        phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
        phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
        interface = (
            base
            + float(rng.uniform(0.12, 0.32))
            * np.sin(2.0 * np.pi * x / float(rng.uniform(4.0, 8.0)) + phase1)
            + float(rng.uniform(0.03, 0.10))
            * np.sin(2.0 * np.pi * x / float(rng.uniform(1.8, 3.4)) + phase2)
        )
        interfaces.append(interface.astype(np.float32))

    if len(interfaces) == 2:
        interfaces[1] = np.maximum(interfaces[1], interfaces[0] + 0.45)

    model = np.full((len(z), len(x)), float(layer_eps[0]), dtype=np.float32)
    transition_width = float(rng.uniform(0.025, 0.055))
    for i, interface in enumerate(interfaces):
        transition = 0.5 * (1.0 + np.tanh((zz - interface[None, :]) / transition_width))
        model = model * (1.0 - transition) + float(layer_eps[i + 1]) * transition
    model += smooth_noise(rng, model.shape, sigma=(10, 18), amp=0.06)
    return (
        np.clip(model, 2.0, 18.0).astype(np.float32),
        x,
        z,
        {
            "num_layers": n_layers,
            "layer_eps": layer_eps.astype(float).tolist(),
            "interface_min_m": [float(interface.min()) for interface in interfaces],
            "interface_max_m": [float(interface.max()) for interface in interfaces],
        },
    )


def make_complex_model(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    config = ROCK_CONFIG
    x = np.arange(0.0, 8.0 + config.dx / 10.0, config.dx, dtype=np.float32)
    z = np.arange(0.0, 3.0 + config.dz / 10.0, config.dz, dtype=np.float32)
    xx, zz = np.meshgrid(x, z, indexing="xy")

    phase1 = float(rng.uniform(0.0, 2.0 * np.pi))
    phase2 = float(rng.uniform(0.0, 2.0 * np.pi))
    interface = (
        float(rng.uniform(1.0, 1.45))
        + float(rng.uniform(0.18, 0.38))
        * np.sin(2.0 * np.pi * x / float(rng.uniform(4.5, 8.0)) + phase1)
        + float(rng.uniform(0.05, 0.16))
        * np.sin(2.0 * np.pi * x / float(rng.uniform(1.8, 3.2)) + phase2)
    )
    interface = np.clip(interface, 0.65, 2.15).astype(np.float32)

    top_eps = float(rng.uniform(4.5, 7.0))
    bottom_eps = float(rng.uniform(8.5, 13.5))
    transition = 0.5 * (
        1.0 + np.tanh((zz - interface[None, :]) / float(rng.uniform(0.035, 0.075)))
    )
    model = top_eps * (1.0 - transition) + bottom_eps * transition
    model += smooth_noise(rng, model.shape, sigma=(6, 14), amp=0.22)

    if rng.random() < 0.45:
        interface2 = (
            interface
            + float(rng.uniform(0.45, 0.85))
            + float(rng.uniform(0.05, 0.15))
            * np.sin(
                2.0 * np.pi * x / float(rng.uniform(3.0, 6.0))
                + float(rng.uniform(0.0, 2.0 * np.pi))
            )
        )
        middle_eps = float(rng.uniform(6.5, 10.5))
        transition2 = 0.5 * (
            1.0 + np.tanh((zz - interface2[None, :]) / float(rng.uniform(0.04, 0.08)))
        )
        model = model * (1.0 - transition2) + middle_eps * transition2

    objects = []
    for _ in range(int(rng.integers(2, 4))):
        cx = float(rng.uniform(0.6, 7.4))
        cz = float(rng.uniform(0.55, 1.85))
        sx = float(rng.uniform(0.10, 0.24))
        sz = float(rng.uniform(0.10, 0.24))
        eps = float(
            rng.uniform(10.5, 16.0) if rng.random() < 0.55 else rng.uniform(2.2, 4.0)
        )
        radius = ((xx - cx) / sx) ** 2 + ((zz - cz) / sz) ** 2
        weight = 1.0 / (1.0 + np.exp((radius - 1.0) / float(rng.uniform(0.08, 0.18))))
        model = model * (1.0 - weight) + eps * weight
        objects.append(
            {"x_m": cx, "z_m": cz, "sigma_x_m": sx, "sigma_z_m": sz, "eps": eps}
        )
    meta = {
        "top_eps": top_eps,
        "bottom_eps": bottom_eps,
        "interface_min_m": float(interface.min()),
        "interface_max_m": float(interface.max()),
        "objects": objects,
    }
    return np.clip(model, 2.0, 16.0).astype(np.float32), x, z, meta


def make_rebar_model(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ForwardConfig, dict]:
    config = REBAR_1600_CONFIG if rng.random() < 0.5 else REBAR_2600_CONFIG
    length = float(rng.choice([1.2, 1.5, 2.0], p=[0.35, 0.35, 0.30]))
    x = np.arange(0.0, length, config.dx, dtype=np.float32)
    depth = 0.70 if config.freq_mhz == 1600.0 else 0.36
    z = np.arange(0.0, depth, config.dz, dtype=np.float32)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    base = float(rng.uniform(5.0, 10.0))
    model = np.full((len(z), len(x)), base, dtype=np.float32)
    model += smooth_noise(
        rng,
        model.shape,
        sigma=(max(1.0, 0.035 / config.dz), max(1.0, 0.080 / config.dx)),
        amp=float(rng.uniform(0.05, 0.35)),
    )
    spacing = float(rng.uniform(0.17, 0.23))
    first = float(rng.uniform(0.06, 0.14))
    positions = []
    x_pos = first
    while x_pos < length - 0.05:
        positions.append(x_pos + float(rng.normal(0.0, 0.010)))
        x_pos += spacing + float(rng.normal(0.0, 0.025))
    objects = []
    rebar_depth = float(rng.uniform(0.075, 0.135))
    for cx in positions:
        cz = float(rebar_depth + rng.normal(0.0, 0.004))
        if cz >= depth - 0.03:
            continue
        diameter = float(rng.uniform(0.015, 0.025))
        eps = 20.0
        radius = diameter / 2.0
        mask = (xx - cx) ** 2 + (zz - cz) ** 2 <= radius**2
        model[mask] = eps
        objects.append(
            {
                "type": "rebar",
                "x_m": float(cx),
                "z_m": cz,
                "diameter_m": diameter,
                "eps": eps,
            }
        )

    foams = []

    model = np.clip(model, 2.0, 20.0).astype(np.float32)
    return (
        model,
        x,
        z,
        config,
        {
            "length_m": length,
            "depth_m": depth,
            "background_eps": base,
            "frequency_family": f"{config.freq_mhz:g}MHz",
            "rebar_layout": "single_row_nearly_constant_depth",
            "rebar_depth_m": rebar_depth,
            "foam_layout": "disabled_pure_rebar",
            "objects": objects + foams,
        },
    )


def make_sandbox_model(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ForwardConfig, dict]:
    """Sandbox: sand background + double-row rebars + foam defect blocks.

    Mirrors the VIBWNet sandbox experiment (1.6 GHz, ~1.36 m x 0.75 m).
    """
    config = SANDBOX_CONFIG
    length = 1.37  # 137 traces x 0.01 m
    depth = 0.76
    x = np.arange(0.0, length, config.dx, dtype=np.float32)
    z = np.arange(0.0, depth, config.dz, dtype=np.float32)
    xx, zz = np.meshgrid(x, z, indexing="xy")

    base = float(rng.uniform(3.7, 4.3))
    model = np.full((len(z), len(x)), base, dtype=np.float32)
    model += smooth_noise(
        rng, model.shape,
        sigma=(max(1.0, 0.04 / config.dz), max(1.0, 0.08 / config.dx)),
        amp=float(rng.uniform(0.03, 0.15)),
    )

    objects = []
    # Double-row rebars, using an effective permittivity contrast rather than
    # a near-metal value.  A value of 20 creates unrealistically persistent
    # crossing tails in the lossless scalar simulation; eps=6 better matches
    # the measured 049 B-scan and stays inside the network's training range.
    rebar_eps = 6.0
    rebar_depths = [
        float(rng.uniform(0.10, 0.14)),
        float(rng.uniform(0.24, 0.30)),
    ]
    spacing = float(rng.uniform(0.17, 0.23))
    for rebar_depth in rebar_depths:
        x_pos = float(rng.uniform(0.05, 0.12))
        while x_pos < length - 0.05:
            cx = x_pos + float(rng.normal(0.0, 0.008))
            cz = float(rebar_depth + rng.normal(0.0, 0.005))
            diameter = float(rng.uniform(0.012, 0.022))
            radius = diameter / 2.0
            mask = (xx - cx) ** 2 + (zz - cz) ** 2 <= radius**2
            model[mask] = rebar_eps
            objects.append(
                {"type": "rebar", "x_m": float(cx), "z_m": cz,
                 "diameter_m": diameter, "eps": rebar_eps}
            )
            x_pos += spacing + float(rng.normal(0.0, 0.02))

    # Three separated foam targets follow the approximate 049 layout.  Fully
    # random, overlapping large blocks produce unrealistically dense late-time
    # crossings.  Their effective permittivity is kept inside the network's
    # [2, 10] model range and near the 049 FWI anomaly range.
    foam_templates = [
        (0.31, 0.45, 0.35),
        (0.67, 0.53, 0.30),
        (1.10, 0.55, 0.15),
    ]
    for base_x, base_z, base_w in foam_templates:
        cx = float(base_x + rng.normal(0.0, 0.02))
        cz = float(base_z + rng.normal(0.0, 0.015))
        w = float(base_w * rng.uniform(0.85, 1.15))
        h = float(rng.uniform(0.06, 0.10))
        eps = float(rng.uniform(2.4, 3.1))
        mask = (np.abs(xx - cx) <= w / 2) & (np.abs(zz - cz) <= h / 2)
        model[mask] = eps
        objects.append(
            {"type": "foam", "x_m": cx, "z_m": cz, "w_m": w, "h_m": h, "eps": eps}
        )

    model = np.clip(model, 2.0, 10.0).astype(np.float32)
    return (
        model,
        x,
        z,
        config,
        {
            "length_m": float(length),
            "depth_m": float(depth),
            "background_eps": base,
            "frequency_family": f"{config.freq_mhz:g}MHz",
            "rebar_layout": "double_row",
            "rebar_depths_m": rebar_depths,
            "foam_layout": "three_separated_049_like_blocks",
            "objects": objects,
        },
    )


def simulate(
    model: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    config: ForwardConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    permittivity = torch.tensor(model, dtype=torch.float32, device=device)
    survey = config.survey()
    velocity = model_to_velocity(permittivity, survey)
    nshot = int(round((float(x[-1]) - config.src_x_start) / config.src_x_step)) + 1
    source_amplitudes, source_locations, receiver_locations = build_geometry(
        nshot, config.nt, len(x), len(z), survey, device
    )
    out = scalar(
        velocity,
        [survey.dz, survey.dx],
        dt=survey.dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=[survey.pml_width] * 4,
        pml_freq=survey.freq,
        accuracy=survey.accuracy,
    )
    return out[-1].contiguous(), source_amplitudes, source_locations, receiver_locations


def process_bscan(raw: torch.Tensor) -> torch.Tensor:
    return preprocess_bscan(raw.detach().cpu()).contiguous()


def save_data_pt(
    path: Path,
    model: np.ndarray,
    raw: torch.Tensor,
    source_amplitudes: torch.Tensor,
    source_locations: torch.Tensor,
    receiver_locations: torch.Tensor,
    x: np.ndarray,
    z: np.ndarray,
) -> None:
    processed = process_bscan(raw)
    torch.save(
        {
            "permittivity": torch.tensor(model, dtype=torch.float32),
            "bscan_raw": raw.detach().cpu(),
            "bscan_processed": processed,
            "source_amplitudes": source_amplitudes.detach().cpu(),
            "source_locations": source_locations.detach().cpu(),
            "receiver_locations": receiver_locations.detach().cpu(),
            "x": torch.tensor(x, dtype=torch.float32),
            "z": torch.tensor(z, dtype=torch.float32),
        },
        path,
    )


def save_meta(
    path: Path,
    sample_id: str,
    kind: str,
    source: str,
    config: ForwardConfig,
    x: np.ndarray,
    z: np.ndarray,
    extra: dict,
) -> None:
    payload = {
        "sample_id": sample_id,
        "kind": kind,
        "source": source,
        "dx": config.dx,
        "dz": config.dz,
        "dt": config.dt,
        "freq_mhz": config.freq_mhz,
        "nt": config.nt,
        "nshot": int(round((float(x[-1]) - config.src_x_start) / config.src_x_step))
        + 1,
        "length_m": float(x[-1] - x[0] + config.dx),
        "depth_m": float(z[-1] - z[0] + config.dz),
        "src_x_start": config.src_x_start,
        "src_x_step": config.src_x_step,
        "src_z": config.src_z,
        "rec_offset_x": config.rec_offset_x,
        "rec_z": config.rec_z,
        "pml_width": config.pml_width,
        "accuracy": config.accuracy,
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2))


def bscan_panel(bscan: torch.Tensor) -> np.ndarray:
    arr = bscan.detach().cpu()
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    return arr.T.numpy()


def save_preview(
    path: Path,
    model: np.ndarray,
    raw: torch.Tensor,
    processed: torch.Tensor,
    x: np.ndarray,
    z: np.ndarray,
    config: ForwardConfig,
) -> None:
    panels = [bscan_panel(raw), bscan_panel(processed)]
    raw_v = float(np.percentile(np.abs(panels[0]), 95.0)) or 1.0
    proc_v = float(np.percentile(np.abs(panels[1]), 95.0)) or 1.0
    t_ns = config.nt * config.dt / config.scale * 1e9
    extent_model = [
        float(x[0]),
        float(x[-1] + config.dx),
        float(z[-1] + config.dz),
        float(z[0]),
    ]
    extent_bscan = [float(x[0]), float(x[-1] + config.dx), t_ns, 0.0]
    model_vmax = max(6.0, min(20.0, float(np.ceil(np.max(model)))))
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.0), dpi=140)
    im0 = axes[0].imshow(
        model,
        cmap="turbo",
        aspect="equal",
        extent=extent_model,
        vmin=2.0,
        vmax=model_vmax,
    )
    axes[0].set_title("Permittivity")
    axes[0].set_ylabel("z (m)")
    fig.colorbar(im0, ax=axes[0], fraction=0.025, pad=0.015)
    axes[1].imshow(
        panels[0],
        cmap="gray",
        aspect="auto",
        extent=extent_bscan,
        vmin=-raw_v,
        vmax=raw_v,
    )
    axes[1].set_title("Raw B-scan")
    axes[1].set_ylabel("t (ns)")
    axes[2].imshow(
        panels[1],
        cmap="gray",
        aspect="auto",
        extent=extent_bscan,
        vmin=-proc_v,
        vmax=proc_v,
    )
    axes[2].set_title("Processed B-scan")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("t (ns)")
    fig.subplots_adjust(left=0.08, right=0.94, top=0.94, bottom=0.08, hspace=0.48)
    fig.savefig(path)
    plt.close(fig)


def write_manifest(root: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with (root / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_one(
    kind: str, index: int, root: Path, rng: np.random.Generator, device: torch.device
) -> dict:
    if kind == "cross":
        model, x, z, extra = make_cross_model(rng)
        config = ROCK_CONFIG
        source = "generated_cross_400mhz_v1"
    elif kind == "layered":
        model, x, z, extra = make_layered_model(rng)
        config = ROCK_CONFIG
        source = "generated_layered_400mhz_v1"
    elif kind == "complex":
        model, x, z, extra = make_complex_model(rng)
        config = ROCK_CONFIG
        source = "generated_complex_400mhz_v2"
    elif kind == "rebar":
        model, x, z, config, extra = make_rebar_model(rng)
        source = f"generated_rebar_highcontrast_{config.freq_mhz:g}mhz_v2"
    elif kind == "sandbox":
        model, x, z, config, extra = make_sandbox_model(rng)
        source = f"generated_sandbox_{config.freq_mhz:g}mhz_v1"
    else:
        raise ValueError(f"Unsupported generated kind: {kind}")
    raw, amps, srcs, recs = simulate(model, x, z, config, device)
    if kind in ("rebar", "sandbox"):
        raw = -raw
        amps = -amps
    processed = process_bscan(raw)
    dst = sample_dir(root, index)
    dst.mkdir(parents=True, exist_ok=True)
    save_data_pt(dst / "data.pt", model, raw, amps, srcs, recs, x, z)
    save_meta(dst / "meta.json", f"{index:04d}", kind, source, config, x, z, extra)
    save_preview(
        dst / "preview.png", model, raw.detach().cpu(), processed, x, z, config
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "sample_id": f"{index:04d}",
        "kind": kind,
        "source": source,
        "data": str(dst / "data.pt"),
        "meta": str(dst / "meta.json"),
        "preview": str(dst / "preview.png"),
    }


def run_rock(args: argparse.Namespace, device: torch.device) -> None:
    root = resolve_output_dir(args, "rock")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    model_files = collect_model_files(args.data_dir)
    if args.limit is not None:
        model_files = model_files[: args.limit]
    print(f"rock model_count={len(model_files)}")
    for out_index, model_path in enumerate(model_files):
        dst = sample_dir(root, out_index)
        if args.resume and (dst / "data.pt").exists():
            rows.append(
                {
                    "sample_id": f"{out_index:04d}",
                    "kind": "rock",
                    "source": model_path.name,
                    "data": str(dst / "data.pt"),
                    "meta": str(dst / "meta.json"),
                    "preview": str(dst / "preview.png"),
                }
            )
            continue
        model, x, z = load_mat_model(model_path, ROCK_CONFIG)
        raw, amps, srcs, recs = simulate(model, x, z, ROCK_CONFIG, device)
        processed = process_bscan(raw)
        dst.mkdir(parents=True, exist_ok=True)
        save_data_pt(dst / "data.pt", model, raw, amps, srcs, recs, x, z)
        save_meta(
            dst / "meta.json",
            f"{out_index:04d}",
            "rock",
            model_path.name,
            ROCK_CONFIG,
            x,
            z,
            {"model_path": str(model_path)},
        )
        save_preview(
            dst / "preview.png", model, raw.detach().cpu(), processed, x, z, ROCK_CONFIG
        )
        rows.append(
            {
                "sample_id": f"{out_index:04d}",
                "kind": "rock",
                "source": model_path.name,
                "data": str(dst / "data.pt"),
                "meta": str(dst / "meta.json"),
                "preview": str(dst / "preview.png"),
            }
        )
        print(f"[rock {out_index + 1}/{len(model_files)}] {dst}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_manifest(root, rows)


def run_generated(kind: str, args: argparse.Namespace, device: torch.device) -> None:
    root = resolve_output_dir(args, kind)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for index in range(args.count):
        dst = sample_dir(root, index)
        if args.resume and (dst / "data.pt").exists():
            rows.append(
                {
                    "sample_id": f"{index:04d}",
                    "kind": kind,
                    "source": f"existing_{kind}",
                    "data": str(dst / "data.pt"),
                    "meta": str(dst / "meta.json"),
                    "preview": str(dst / "preview.png"),
                }
            )
            continue
        row = generate_one(kind, index, root, rng, device)
        rows.append(row)
        print(f"[{kind} {index + 1}/{args.count}] {dst}")
    write_manifest(root, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate forward datasets using the unified sample schema."
    )
    parser.add_argument(
        "--kind",
        choices=["rock", "cross", "layered", "complex", "rebar", "sandbox"],
        required=True,
    )
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--data-dir", type=Path, default=ROCK_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Exact dataset output directory. Defaults to data/artifacts/<kind>.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.kind == "rock":
        run_rock(args, device)
    else:
        if args.count is None:
            raise ValueError("--count is required for generated datasets")
        run_generated(args.kind, args, device)


if __name__ == "__main__":
    main()
