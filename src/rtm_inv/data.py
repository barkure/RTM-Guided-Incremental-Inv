from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from .bscan import normalise_bscan
from .initialization import (
    MODE_ORACLE_MEAN,
    INITIALIZATION_MODES,
    InitializationSpec,
)
from .models import RTMInvNet


@dataclass(frozen=True)
class SamplePaths:
    sample_id: str
    sample_dir: Path
    data: Path
    meta: Path


class RTMDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        data_dir: str | Path,
        limit: int | None = None,
        normalize_models: bool = True,
        include_raw_bscan: bool = False,
        bscan_scale_quantile: float = 0.95,
        bscan_scale_eps: float = 1e-12,
        initial_model_mode: str = MODE_ORACLE_MEAN,
        initializer: InitializationSpec | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.normalize_models = normalize_models
        self.include_raw_bscan = bool(include_raw_bscan)
        self.bscan_scale_quantile = float(bscan_scale_quantile)
        self.bscan_scale_eps = float(bscan_scale_eps)
        if initial_model_mode not in INITIALIZATION_MODES:
            raise ValueError(
                f"initial_model_mode must be one of {list(INITIALIZATION_MODES)}, "
                f"got {initial_model_mode!r}"
            )
        if initial_model_mode == MODE_ORACLE_MEAN:
            if initializer is not None and initializer.mode != MODE_ORACLE_MEAN:
                raise ValueError(
                    "initial_model_mode='oracle_mean' conflicts with "
                    f"initializer.mode={initializer.mode!r}"
                )
        else:
            if initializer is None:
                raise ValueError(
                    f"initial_model_mode={initial_model_mode!r} requires an "
                    "InitializationSpec describing the background."
                )
            if initializer.mode != initial_model_mode:
                raise ValueError(
                    f"initial_model_mode={initial_model_mode!r} does not match "
                    f"initializer.mode={initializer.mode!r}"
                )
        self.initial_model_mode = initial_model_mode
        self.initializer = initializer
        self.samples = self._collect_samples(limit=limit)
        if not self.samples:
            raise ValueError(f"No data.pt samples found in {self.data_dir}")

    def _collect_samples(self, limit: int | None) -> Sequence[SamplePaths]:
        sample_dirs = sorted(
            [path for path in self.data_dir.glob("sample_*") if path.is_dir()],
            key=lambda path: int(path.name.split("_")[-1]),
        )
        samples: list[SamplePaths] = []
        for sample_dir in sample_dirs:
            data = sample_dir / "data.pt"
            meta = sample_dir / "meta.json"
            if data.exists() and meta.exists():
                samples.append(
                    SamplePaths(
                        sample_id=sample_dir.name.split("_")[-1],
                        sample_dir=sample_dir,
                        data=data,
                        meta=meta,
                    )
                )
            if limit is not None and len(samples) >= limit:
                break
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def sample_shape_key(self, index: int) -> tuple[int, ...]:
        sample = self.samples[index]
        meta = json.loads(sample.meta.read_text())
        nx = int(round(float(meta["length_m"]) / float(meta["dx"])))
        nz = int(round(float(meta["depth_m"]) / float(meta["dz"])))
        nt = int(meta["nt"])
        nshot = int(meta["nshot"])
        return nz, nx, nshot, nt

    def target_permittivity(self, index: int) -> torch.Tensor:
        """Raw (physical) relative-permittivity target, without normalisation.

        Used to derive ``c_train`` from the training split only.
        """
        sample = self.samples[index]
        data = torch.load(sample.data, map_location="cpu", weights_only=False)
        return data["permittivity"].float()

    def _make_initial_model(
        self, permittivity: torch.Tensor, sample_id: str
    ) -> torch.Tensor:
        """Constant initial model in physical permittivity, shape (1, nz, nx).

        ``oracle_mean`` reproduces the historical behaviour exactly. Every other mode
        takes its value from the :class:`InitializationSpec`, which is where the
        target-leakage accounting lives.
        """
        if self.initializer is None:
            return torch.full_like(permittivity, permittivity.mean()).unsqueeze(0)
        value = self.initializer.value_for_sample(sample_id, permittivity)
        return torch.full_like(permittivity, value).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        data = torch.load(sample.data, map_location="cpu", weights_only=False)
        meta = json.loads(sample.meta.read_text())

        permittivity = data["permittivity"].float()
        current_model = self._make_initial_model(permittivity, sample.sample_id)
        target_model = permittivity.unsqueeze(0)
        observed_data = data["bscan_processed"].float()

        observed_data, bscan_scale = normalise_bscan(
            observed_data,
            quantile=self.bscan_scale_quantile,
            eps=self.bscan_scale_eps,
        )

        if self.normalize_models:
            model_min = RTMInvNet.MODEL_MIN
            model_max = RTMInvNet.MODEL_MAX
            scale = model_max - model_min
            current_model = (current_model - model_min) / scale
            target_model = (target_model - model_min) / scale

        result: dict[str, torch.Tensor | str] = {
            "sample_id": sample.sample_id,
            "bscan": observed_data,
            "bscan_scale": bscan_scale.float(),
            "observed_is_processed": torch.tensor(True, dtype=torch.bool),
            "rtm_config": torch.tensor(
                [
                    float(meta["dx"]),
                    float(meta["dz"]),
                    float(meta["dt"]),
                    float(meta["freq_mhz"]),
                    float(meta["src_x_start"]),
                    float(meta["src_x_step"]),
                    float(meta["src_z"]),
                    float(meta["rec_offset_x"]),
                    float(meta["rec_z"]),
                    float(meta["pml_width"]),
                    float(meta["accuracy"]),
                ],
                dtype=torch.float32,
            ),
            "initial_model": current_model,
            "target_model": target_model,
        }
        if self.include_raw_bscan:
            result["bscan_raw"] = data["bscan_raw"].float()
        return result
