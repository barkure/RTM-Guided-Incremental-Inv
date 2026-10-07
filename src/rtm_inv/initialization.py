"""Realistic initial-model construction, with explicit target-leakage accounting.

The paper's baseline initialisation is the per-sample **spatial mean of that
sample's own ground-truth model**. That is an oracle-informed controlled
initialisation: it injects target information into the input and cannot be
obtained in the field. This module makes the distinction explicit and provides
initialisations that never read validation or test targets.

Modes
-----
``oracle_mean``
    Per-sample spatial mean of that sample's own target. Reproduces the historical
    behaviour byte-for-byte. **Oracle-informed**: it reads the target of whatever
    sample it is applied to, so on a test split it reads test targets. It is not a
    realistic initialisation and must not be presented as a performance ceiling.
``train_global_constant``
    One constant ``c_train`` = mean over all target pixels of the *training split
    only*, frozen and then reused for train/val/test. Validation and test targets
    never participate.
``biased_global_constant``
    ``c_train`` shifted by a pre-declared systematic relative bias, then clipped to
    the model range. Never starts from a per-sample target mean.
``random_biased_global_constant``
    ``c_train`` shifted by a per-sample draw from a pre-declared truncated normal.
    The draw depends only on ``(random_seed, sample_id)``, so every method that
    shares a spec sees the *same* realisation for a given sample, and the value does
    not depend on iteration order or on which method is being evaluated.
``calibration_derived``
    Reserved for a genuinely independent calibration (B-scan arrival time, field
    velocity, dielectric probe, ...). No such method is implemented here: the
    dataset ships no independent calibration measurements, and using a target mean
    as a stand-in would be target leakage wearing a different name. The mode raises
    rather than fabricating values.

Scale conventions
-----------------
Initial models are always defined in **physical relative permittivity** and only
afterwards converted to the normalized model scale, so a mode can never silently
change the normalisation contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from scipy import stats

from .protocol import MODEL_MAX_DEFAULT, MODEL_MIN_DEFAULT, hash_values

INITIALIZATION_VERSION = "1.0.0"

MODE_ORACLE_MEAN = "oracle_mean"
MODE_TRAIN_GLOBAL_CONSTANT = "train_global_constant"
MODE_BIASED_GLOBAL_CONSTANT = "biased_global_constant"
MODE_RANDOM_BIASED_GLOBAL_CONSTANT = "random_biased_global_constant"
MODE_CALIBRATION_DERIVED = "calibration_derived"

INITIALIZATION_MODES = (
    MODE_ORACLE_MEAN,
    MODE_TRAIN_GLOBAL_CONSTANT,
    MODE_BIASED_GLOBAL_CONSTANT,
    MODE_RANDOM_BIASED_GLOBAL_CONSTANT,
    MODE_CALIBRATION_DERIVED,
)

#: Modes that read the sample's own target model, i.e. oracle-informed.
ORACLE_MODES = frozenset({MODE_ORACLE_MEAN})

#: Realistic modes: they may read *training* targets but never validation/test ones.
REALISTIC_MODES = frozenset(
    {
        MODE_TRAIN_GLOBAL_CONSTANT,
        MODE_BIASED_GLOBAL_CONSTANT,
        MODE_RANDOM_BIASED_GLOBAL_CONSTANT,
        MODE_CALIBRATION_DERIVED,
    }
)

CONSTANT_MODES = frozenset(
    {
        MODE_TRAIN_GLOBAL_CONSTANT,
        MODE_BIASED_GLOBAL_CONSTANT,
        MODE_RANDOM_BIASED_GLOBAL_CONSTANT,
    }
)

DEFAULT_RANDOM_STD_PERCENT = 0.10
DEFAULT_RANDOM_CLIP_PERCENT = 0.30

#: Bias levels swept by the stage-3 OOD evaluation.
SYSTEMATIC_BIAS_LEVELS = (-0.20, -0.10, 0.10, 0.20)

#: Older checkpoints recorded an ``initial_model_mode`` under a different vocabulary.
#: ``mean_constant`` is the per-sample target mean, i.e. exactly ``oracle_mean``.
LEGACY_MODE_ALIASES = {
    "mean_constant": MODE_ORACLE_MEAN,
    "sample_spatial_mean_constant": MODE_ORACLE_MEAN,
    "oracle": MODE_ORACLE_MEAN,
}


def normalise_mode(value: object) -> str | None:
    """Map a recorded mode string onto this module's vocabulary.

    Returns ``None`` for anything unrecognised, so callers can warn and fall back
    explicitly instead of silently guessing.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in INITIALIZATION_MODES:
        return text
    return LEGACY_MODE_ALIASES.get(text)


#: Splits an initialisation may be evaluated on. Anything else is an error rather
#: than something to guess at.
EVALUATED_SPLITS = ("train", "validation", "test")


def normalise_split_name(value: object) -> str:
    """Accept the documented spellings only; raise on anything else."""
    text = str(value).strip().lower()
    if text in EVALUATED_SPLITS:
        return text
    alias = {"val": "validation", "valid": "validation"}.get(text)
    if alias is not None:
        return alias
    raise ValueError(
        f"unknown evaluated split {value!r}; expected one of {list(EVALUATED_SPLITS)}"
    )


def _sample_rng(seed: int, sample_id: str) -> np.random.Generator:
    """RNG that depends only on ``(seed, sample_id)``, never on iteration order."""
    digest = hashlib.sha256(f"{int(seed)}|{sample_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


@dataclass(frozen=True)
class TrainConstant:
    """``c_train`` plus the provenance that makes its leakage status auditable."""

    value: float
    train_samples: int
    train_pixels: int
    train_sample_ids: tuple[str, ...]
    train_sample_ids_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "train_constant_value": self.value,
            "train_constant_samples": self.train_samples,
            "train_constant_pixels": self.train_pixels,
            "train_constant_sample_ids_sha256": self.train_sample_ids_sha256,
        }


def compute_train_global_constant(
    dataset,
    train_indices: list[int],
    *,
    progress_every: int | None = None,
) -> TrainConstant:
    """Mean over every target pixel of the training split, and nothing else.

    Only ``train_indices`` are read. Validation and test targets cannot influence
    the result, which is what makes this a non-oracle initialisation.
    """
    if not train_indices:
        raise ValueError("train_indices must not be empty")
    total = 0.0
    pixels = 0
    sample_ids: list[str] = []
    for position, index in enumerate(train_indices):
        permittivity = dataset.target_permittivity(index)
        total += float(permittivity.double().sum().item())
        pixels += int(permittivity.numel())
        sample_ids.append(str(dataset.samples[index].sample_id))
        if progress_every and (position + 1) % progress_every == 0:
            print(f"c_train: {position + 1}/{len(train_indices)}", flush=True)
    if pixels == 0:
        raise ValueError("Training split contributed no pixels")
    return TrainConstant(
        value=total / pixels,
        train_samples=len(train_indices),
        train_pixels=pixels,
        train_sample_ids=tuple(sample_ids),
        train_sample_ids_sha256=hash_values(sample_ids),
    )


@dataclass(frozen=True)
class InitializationSpec:
    """Fully specified, reproducible initial-model recipe."""

    mode: str
    model_min: float = MODEL_MIN_DEFAULT
    model_max: float = MODEL_MAX_DEFAULT
    train_constant: float | None = None
    bias_percent: float = 0.0
    random_std_percent: float = DEFAULT_RANDOM_STD_PERCENT
    random_clip_percent: float = DEFAULT_RANDOM_CLIP_PERCENT
    random_seed: int | None = None
    train_constant_source: TrainConstant | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.mode not in INITIALIZATION_MODES:
            raise ValueError(
                f"mode must be one of {list(INITIALIZATION_MODES)}, got {self.mode!r}"
            )
        if self.model_max <= self.model_min:
            raise ValueError("model_max must exceed model_min")
        if self.mode in CONSTANT_MODES and self.train_constant is None:
            raise ValueError(f"mode {self.mode!r} requires a train_constant")
        if self.mode == MODE_RANDOM_BIASED_GLOBAL_CONSTANT:
            if self.random_seed is None:
                raise ValueError("random_biased_global_constant requires a random_seed")
            if self.random_std_percent <= 0.0:
                raise ValueError("random_std_percent must be positive")
            if self.random_clip_percent <= 0.0:
                raise ValueError("random_clip_percent must be positive")

    # -- leakage accounting ------------------------------------------------- #

    @property
    def reads_sample_target(self) -> bool:
        """True when the value depends on the target of the sample it is applied to."""
        return self.mode in ORACLE_MODES

    @property
    def is_realistic(self) -> bool:
        return self.mode in REALISTIC_MODES

    def leakage_flags(self, *, evaluated_split: str = "test") -> dict[str, bool]:
        """Record exactly which splits' targets the initialisation reads.

        Only ``train``, ``validation`` and ``test`` are accepted; an unknown split
        is an error, because inferring "everything that is not test is validation"
        is how a leakage flag silently becomes wrong.

        A shared-background mode reads the *training* split (and nothing else).
        ``oracle_mean`` reads the target of whatever split it is applied to, so the
        flag follows the split it is evaluated on.
        """
        split = normalise_split_name(evaluated_split)
        if self.mode not in ORACLE_MODES:
            return {
                "uses_train_target": self.mode in CONSTANT_MODES,
                "uses_validation_target": False,
                "uses_test_target": False,
                "reads_sample_target": False,
            }
        return {
            "uses_train_target": split == "train",
            "uses_validation_target": split == "validation",
            "uses_test_target": split == "test",
            "reads_sample_target": True,
        }

    # -- resolution --------------------------------------------------------- #

    def unclipped_background_value(self, sample_id: str) -> float:
        """The pre-clip value, so a sweep can show where the model bounds bind."""
        if self.mode == MODE_ORACLE_MEAN:
            raise ValueError(
                "oracle_mean has no shared background value; it is derived per sample"
            )
        if self.mode == MODE_CALIBRATION_DERIVED:
            raise NotImplementedError(
                "No independent calibration source is available for this dataset, so "
                "a calibration-derived background cannot be produced. Using a target "
                "mean here would be target leakage."
            )
        base = float(self.train_constant)  # type: ignore[arg-type]
        if self.mode == MODE_BIASED_GLOBAL_CONSTANT:
            delta = float(self.bias_percent)
        elif self.mode == MODE_RANDOM_BIASED_GLOBAL_CONSTANT:
            delta = self._random_delta(sample_id)
        else:
            delta = 0.0
        return base * (1.0 + delta)

    def background_value(self, sample_id: str) -> float:
        """Physical relative-permittivity background for one sample, clipped.

        Deterministic in ``(mode, train_constant, bias, seed, sample_id)`` and
        independent of iteration order, so two methods sharing a spec necessarily
        receive the same value for the same sample.
        """
        return float(
            np.clip(self.unclipped_background_value(sample_id), self.model_min, self.model_max)
        )

    def _random_delta(self, sample_id: str) -> float:
        """Truncated-normal relative bias for one sample."""
        rng = _sample_rng(int(self.random_seed), sample_id)  # type: ignore[arg-type]
        # Truncate at +/- clip_percent, expressed in units of the standard deviation.
        bound = self.random_clip_percent / self.random_std_percent
        return float(
            stats.truncnorm.rvs(
                -bound, bound, loc=0.0, scale=self.random_std_percent, random_state=rng
            )
        )

    def value_for_sample(self, sample_id: str, target_permittivity: torch.Tensor) -> float:
        if self.mode == MODE_ORACLE_MEAN:
            return float(target_permittivity.double().mean().item())
        return self.background_value(sample_id)

    def describe(self) -> str:
        if self.mode == MODE_ORACLE_MEAN:
            return "per-sample spatial mean of the target model (oracle-informed)"
        if self.mode == MODE_TRAIN_GLOBAL_CONSTANT:
            return f"global constant c_train={self.train_constant:.6f} from the training split"
        if self.mode == MODE_BIASED_GLOBAL_CONSTANT:
            return (
                f"c_train={self.train_constant:.6f} with systematic bias "
                f"{self.bias_percent:+.0%}, clipped to "
                f"[{self.model_min}, {self.model_max}]"
            )
        if self.mode == MODE_RANDOM_BIASED_GLOBAL_CONSTANT:
            return (
                f"c_train={self.train_constant:.6f} with per-sample truncated-normal bias "
                f"(sd {self.random_std_percent:.0%}, clipped at +/-{self.random_clip_percent:.0%}, "
                f"seed {self.random_seed}), clipped to [{self.model_min}, {self.model_max}]"
            )
        return "calibration-derived background (not implemented)"

    def as_dict(self, *, evaluated_split: str = "test") -> dict[str, object]:
        payload: dict[str, object] = {
            "initialization_version": INITIALIZATION_VERSION,
            "initial_model_mode": self.mode,
            "initialization_label": self.label,
            "initialization_description": self.describe(),
            "initialization_model_min": self.model_min,
            "initialization_model_max": self.model_max,
            "initialization_bias_percent": self.bias_percent,
            "initialization_random_std_percent": self.random_std_percent,
            "initialization_random_clip_percent": self.random_clip_percent,
            "initialization_random_seed": self.random_seed,
            "initialization_train_constant": self.train_constant,
            "initialization_is_realistic": self.is_realistic,
            "initialization_reads_sample_target": self.reads_sample_target,
            "initialization_clipping_applied": self.mode in CONSTANT_MODES,
        }
        if self.train_constant_source is not None:
            payload.update(self.train_constant_source.as_dict())
        payload.update(self.leakage_flags(evaluated_split=evaluated_split))
        return payload


def build_initialization_spec(
    mode: str,
    *,
    model_min: float = MODEL_MIN_DEFAULT,
    model_max: float = MODEL_MAX_DEFAULT,
    train_constant: TrainConstant | float | None = None,
    bias_percent: float = 0.0,
    random_std_percent: float = DEFAULT_RANDOM_STD_PERCENT,
    random_clip_percent: float = DEFAULT_RANDOM_CLIP_PERCENT,
    random_seed: int | None = None,
    label: str = "",
) -> InitializationSpec:
    """Convenience builder accepting either a :class:`TrainConstant` or a raw float."""
    if isinstance(train_constant, TrainConstant):
        value: float | None = train_constant.value
        source = train_constant
    else:
        value = None if train_constant is None else float(train_constant)
        source = None
    return InitializationSpec(
        mode=mode,
        model_min=model_min,
        model_max=model_max,
        train_constant=value,
        bias_percent=bias_percent,
        random_std_percent=random_std_percent,
        random_clip_percent=random_clip_percent,
        random_seed=random_seed,
        train_constant_source=source,
        label=label,
    )


def make_initial_model(target_permittivity: torch.Tensor, value: float) -> torch.Tensor:
    """Constant physical-permittivity initial model with the target's shape (1, nz, nx)."""
    if not np.isfinite(value):
        raise ValueError(f"initial-model value must be finite, got {value}")
    return torch.full_like(target_permittivity, float(value)).unsqueeze(0)


def realized_backgrounds(
    spec: InitializationSpec,
    sample_ids: list[str],
) -> list[dict[str, object]]:
    """``sample_id -> background value`` table, for auditing the realisation."""
    if spec.mode == MODE_ORACLE_MEAN:
        raise ValueError("oracle_mean has no shared realisation table")
    rows: list[dict[str, object]] = []
    for sample_id in sample_ids:
        unclipped = spec.unclipped_background_value(sample_id)
        value = spec.background_value(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "initial_model_mode": spec.mode,
                "background_value_physical": value,
                "background_value_unclipped": unclipped,
                "clipped": int(value != unclipped),
            }
        )
    return rows


def background_table(
    specs: Mapping[str, InitializationSpec],
    sample_ids: list[str],
) -> list[dict[str, object]]:
    """One row per (condition label, sample), covering every non-oracle spec."""
    rows: list[dict[str, object]] = []
    for label, spec in specs.items():
        if spec.mode == MODE_ORACLE_MEAN:
            continue
        for row in realized_backgrounds(spec, sample_ids):
            rows.append(
                {
                    "condition": label,
                    "mode": row["initial_model_mode"],
                    "sample_id": row["sample_id"],
                    "bias_percent": spec.bias_percent,
                    "random_seed": spec.random_seed,
                    "background_value_physical": row["background_value_physical"],
                    "background_value_unclipped": row["background_value_unclipped"],
                    "clipped": row["clipped"],
                }
            )
    return rows
