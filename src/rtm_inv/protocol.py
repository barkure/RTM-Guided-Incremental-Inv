"""Frozen evaluation, reproducibility and split-identity protocol.

This module is the single source of truth for how the paper-facing metrics are
defined, how the train/val/test split is reconstructed, how random seeds are
resolved, and what metadata is attached to every evaluation artefact.  Analysis
scripts must import from here instead of re-implementing metric formulas.

Metric protocol (version in ``METRIC_PROTOCOL_VERSION``)
-------------------------------------------------------
* All metrics are computed per sample on the **full model grid** (no cropping)
  and aggregated with a plain per-sample arithmetic mean.
* ``initial``, ``stage1`` and ``final`` are the three evaluated model stages.
* Canonical metrics are computed on the **normalized** model scale with a fixed
  ``data_range = 1.0``:

  - MAE  : mean absolute error
  - MSE  : mean squared error
  - SSIM : single-scale Wang et al. formulation, 11x11 Gaussian window,
           ``sigma = 1.5``
  - PSNR : ``20*log10(data_range) - 10*log10(MSE)``

* The physical relative-permittivity scale is an **affine** map of the
  normalized scale: ``eps = MODEL_SCALE * m + MODEL_MIN`` with
  ``MODEL_MIN = 2`` and ``MODEL_MAX = 10`` (``MODEL_SCALE = 8``). Therefore

  - ``MAE_physical = 8 * MAE_normalized``
  - ``MSE_physical = 64 * MSE_normalized``

  and only MAE/MSE are re-reported on the physical scale (columns suffixed
  ``_er_*``).  Physical MAE is the quantity quoted in the main text.

* SSIM and PSNR are **not** duplicated on the physical scale because:

  - PSNR is exactly invariant under the affine map when ``data_range`` is
    scaled by 8, so a second set would be numerically identical.
  - SSIM is invariant under a pure scaling with scaled ``data_range``, but the
    permittivity map has a non-zero offset (``+MODEL_MIN``), and the SSIM
    luminance term is not offset-invariant.  A physical-scale SSIM therefore
    differs from the normalized one (e.g. 0.9148 vs 0.8947 for RTM-Refresh on
    the historical 100-sample test set).  Reporting both would create two
    incompatible SSIM numbers, so SSIM is reported only once, on the canonical
    normalized scale.

Do not add a second SSIM/PSNR scale without updating this docstring and the
protocol version.

Two independent version axes
----------------------------
* ``METRIC_PROTOCOL_VERSION`` versions the metric definitions and the metadata
  contract documented above.
* ``SPLIT_IDENTITY_VERSION`` versions the contents of the ``split_sha256``
  payload.

They must never be linked: ``split_sha256`` identifies a data partition, so a
metric-side change must not move it.  See the notes on the constants below.
"""

from __future__ import annotations

import csv
import hashlib
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import peak_signal_noise_ratio, structural_similarity

METRIC_PROTOCOL_VERSION = "1.0.1"

# Version of the *split identity definition* (which fields go into ``split_sha256``
# and how they are serialised). It is deliberately independent of
# ``METRIC_PROTOCOL_VERSION``: changing metric definitions, metadata columns or
# documentation must never change a split hash, or already-published artefacts
# would look as though they used a different data partition.
#
# The value is frozen at the string that was hashed when split identity was first
# defined, so existing ``split_sha256`` values stay valid. Bump it only when the
# hashed payload itself changes (a new field, a different separator, ...).
SPLIT_IDENTITY_VERSION = "1.0.0"

# Version of the *evaluation artefact schema* (which columns a `summary.csv` /
# `metadata.json` pair is required to carry, and how coverage is declared). Kept
# separate from ``METRIC_PROTOCOL_VERSION`` so that adding provenance columns does not
# read as a change of metric definition, and separate from ``SPLIT_IDENTITY_VERSION``
# so that it can never move ``split_sha256``.
EVALUATION_ARTIFACT_SCHEMA_VERSION = "1.0.0"

MODEL_MIN_DEFAULT = 2.0
MODEL_MAX_DEFAULT = 10.0

NORMALIZED_DATA_RANGE = 1.0
EVALUATION_REGION = "full_model_grid"

SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5

MODEL_STAGES = ("initial", "stage1", "final")
MODEL_METRICS = ("mae", "mse", "ssim", "psnr")
PHYSICAL_ERROR_METRICS = ("mae", "mse")
PHYSICAL_SUFFIX = "er"

# Column names that are protocol metadata rather than measured quantities.
NON_METRIC_COLUMNS = frozenset(
    {"sample_id", "input_sample_id", "label", "condition"}
)


def _as_float(value: torch.Tensor) -> float:
    return float(value.item())


@dataclass(frozen=True)
class MetricProtocol:
    """Immutable description of the frozen metric definitions."""

    model_min: float = MODEL_MIN_DEFAULT
    model_max: float = MODEL_MAX_DEFAULT
    version: str = METRIC_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.model_max <= self.model_min:
            raise ValueError(
                f"model_max ({self.model_max}) must exceed model_min ({self.model_min})"
            )

    @property
    def model_scale(self) -> float:
        return float(self.model_max - self.model_min)

    @property
    def normalized_data_range(self) -> float:
        return NORMALIZED_DATA_RANGE

    @property
    def physical_data_range(self) -> float:
        return self.model_scale

    @property
    def mae_normalized_to_physical(self) -> float:
        return self.model_scale

    @property
    def mse_normalized_to_physical(self) -> float:
        return self.model_scale**2

    @property
    def evaluation_region(self) -> str:
        return EVALUATION_REGION

    def normalize(self, physical: torch.Tensor) -> torch.Tensor:
        return (physical - self.model_min) / self.model_scale

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized * self.model_scale + self.model_min

    def normalized_scores(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        """Canonical metrics on the normalized model scale (fixed unit range)."""
        return {
            "mae": _as_float(F.l1_loss(pred, target)),
            "mse": _as_float(F.mse_loss(pred, target)),
            "ssim": structural_similarity(
                pred, target, data_range=self.normalized_data_range
            ),
            "psnr": peak_signal_noise_ratio(
                pred, target, data_range=self.normalized_data_range
            ),
        }

    def physical_error_scores(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, float]:
        """MAE/MSE on the physical relative-permittivity scale.

        SSIM and PSNR are deliberately omitted; see the module docstring.
        """
        pred_physical = self.denormalize(pred)
        target_physical = self.denormalize(target)
        return {
            "mae": _as_float(F.l1_loss(pred_physical, target_physical)),
            "mse": _as_float(F.mse_loss(pred_physical, target_physical)),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "metric_protocol_version": self.version,
            "evaluation_artifact_schema_version": EVALUATION_ARTIFACT_SCHEMA_VERSION,
            "model_min": self.model_min,
            "model_max": self.model_max,
            "model_scale": self.model_scale,
            "normalized_data_range": self.normalized_data_range,
            "physical_data_range": self.physical_data_range,
            "evaluation_region": self.evaluation_region,
            "ssim_window_size": SSIM_WINDOW_SIZE,
            "ssim_sigma": SSIM_SIGMA,
            "aggregation": "per_sample_then_mean",
            "canonical_metric_scale": "normalized_model_unit_interval",
            "physical_metric_scale": "physical_relative_permittivity",
            "mae_normalized_to_physical_factor": self.mae_normalized_to_physical,
            "mse_normalized_to_physical_factor": self.mse_normalized_to_physical,
            "physical_scale_metrics": list(PHYSICAL_ERROR_METRICS),
            "ssim_psnr_duplicated_on_physical_scale": False,
        }


def metric_fieldnames(protocol: MetricProtocol) -> list[str]:
    """Ordered metric column names produced by :func:`compute_metric_row`."""
    names: list[str] = []
    for metric in MODEL_METRICS:
        for stage in MODEL_STAGES:
            names.append(f"{stage}_{metric}")
    for metric in PHYSICAL_ERROR_METRICS:
        for stage in MODEL_STAGES:
            names.append(f"{stage}_{PHYSICAL_SUFFIX}_{metric}")
    return names


def compute_metric_row(
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    protocol: MetricProtocol,
) -> dict[str, str]:
    """Per-sample metric row for one model stage.

    Returns formatted strings so rows can be written straight to ``metrics.csv``.
    """
    normalized = protocol.normalized_scores(pred, target)
    physical = protocol.physical_error_scores(pred, target)
    row = {f"{prefix}_{name}": f"{value:.7g}" for name, value in normalized.items()}
    row.update(
        {f"{prefix}_{PHYSICAL_SUFFIX}_{name}": f"{value:.7g}" for name, value in physical.items()}
    )
    return row


def apply_metric_row(
    row: dict[str, object],
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    protocol: MetricProtocol,
) -> None:
    """In-place :func:`compute_metric_row` for dicts that mix metrics and metadata."""
    row.update(compute_metric_row(prefix, pred, target, protocol))


def aggregate_metric_rows(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> dict[str, float]:
    """Per-sample arithmetic mean over the given metric columns."""
    if not rows:
        raise ValueError("Cannot aggregate an empty metrics table")
    return {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in fieldnames
        if name not in NON_METRIC_COLUMNS
    }


# --------------------------------------------------------------------------- #
# Randomness
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeedConfig:
    """Resolved seeds plus an audit trail of any legacy fallbacks."""

    split_seed: int
    training_seed: int
    sampler_seed: int
    statistics_seed: int | None
    legacy_seed: int | None
    fallback_notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "split_seed": self.split_seed,
            "training_seed": self.training_seed,
            "sampler_seed": self.sampler_seed,
            "statistics_seed": self.statistics_seed,
            "legacy_seed": self.legacy_seed,
            "seed_fallbacks": "; ".join(self.fallback_notes) if self.fallback_notes else "",
        }

    def log_lines(self) -> list[str]:
        lines = [
            f"seeds: split={self.split_seed} training={self.training_seed} "
            f"sampler={self.sampler_seed} statistics={self.statistics_seed} "
            f"legacy={self.legacy_seed}"
        ]
        for note in self.fallback_notes:
            lines.append(f"seed_fallback: {note}")
        return lines


def resolve_seeds(
    *,
    legacy_seed: int | None,
    split_seed: int | None = None,
    training_seed: int | None = None,
    sampler_seed: int | None = None,
    statistics_seed: int | None = None,
) -> SeedConfig:
    """Resolve the split/training/sampler seed trio.

    Explicit per-purpose seeds win.  Any seed left unset falls back to the legacy
    single ``seed`` value, which keeps historical checkpoints and CLI invocations
    valid.  Every fallback is recorded so the log/artefacts state clearly that a
    legacy value was reused.
    """
    notes: list[str] = []

    def _resolve(name: str, value: int | None) -> int:
        if value is not None:
            return int(value)
        if legacy_seed is None:
            raise ValueError(
                f"--{name.replace('_', '-')} is required when no legacy --seed is given"
            )
        notes.append(
            f"{name} not supplied, reusing legacy seed={int(legacy_seed)} "
            "(split/sampler/global RNG provenance is therefore not independent)"
        )
        return int(legacy_seed)

    return SeedConfig(
        split_seed=_resolve("split_seed", split_seed),
        training_seed=_resolve("training_seed", training_seed),
        sampler_seed=_resolve("sampler_seed", sampler_seed),
        statistics_seed=None if statistics_seed is None else int(statistics_seed),
        legacy_seed=None if legacy_seed is None else int(legacy_seed),
        fallback_notes=tuple(notes),
    )


def set_global_seed(seed: int, *, deterministic: bool = False) -> dict[str, object]:
    """Seed every RNG that can influence model initialisation or training.

    Call this **before** constructing the model.  Historical runs never did this,
    so their ``seed`` only pinned the split and the batch sampler; the model was
    initialised from an uncontrolled global RNG state.

    The backend flags are set explicitly in *both* modes, so calling this with
    ``deterministic=True`` and then ``False`` in one process really does leave the
    process in the non-deterministic configuration instead of inheriting the
    earlier setting.  ``cudnn.benchmark`` is pinned off in both modes: enabling it
    would let cuDNN pick algorithms by timing, which reintroduces run-to-run
    variability in the opposite direction to what this protocol is for.
    """
    seed = int(seed)
    deterministic = bool(deterministic)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "deterministic_requested": deterministic,
        "python_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": True,
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def argument_snapshot(args: object) -> dict[str, object]:
    """JSON-ready view of a resolved ``argparse.Namespace``.

    ``Path`` values become strings; everything else is passed through and left to
    the caller's JSON encoder.  This replaces the ambiguous practice of dumping
    ``vars(args).values()`` as an ordered list with no key names.
    """
    snapshot: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            snapshot[key] = str(value)
        elif isinstance(value, (list, tuple)):
            snapshot[key] = [
                str(item) if isinstance(item, Path) else item for item in value
            ]
        else:
            snapshot[key] = value
    return snapshot


# --------------------------------------------------------------------------- #
# Split identity
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SplitSpec:
    """Everything needed to deterministically rebuild the in-memory split."""

    dataset_size: int
    train_split: float
    val_split: float
    split_seed: int

    def __post_init__(self) -> None:
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1")
        if not 0.0 < self.val_split < 1.0:
            raise ValueError("val_split must be between 0 and 1")
        if self.train_split + self.val_split >= 1.0:
            raise ValueError("train_split + val_split must be less than 1")

    @property
    def train_size(self) -> int:
        return int(self.dataset_size * self.train_split)

    @property
    def val_size(self) -> int:
        return int(self.dataset_size * self.val_split)

    @property
    def test_size(self) -> int:
        return self.dataset_size - self.train_size - self.val_size

    @property
    def sizes(self) -> tuple[int, int, int]:
        return (self.train_size, self.val_size, self.test_size)

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_size": self.dataset_size,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "split_seed": self.split_seed,
            "split_train": self.train_size,
            "split_val": self.val_size,
            "split_test": self.test_size,
        }


def reconstruct_split_indices(spec: SplitSpec) -> tuple[list[int], list[int], list[int]]:
    """Rebuild the split exactly as ``torch.utils.data.random_split`` does.

    ``random_split`` permutes ``range(n)`` with a seeded generator and then slices
    the permutation; replicating that here avoids depending on the Dataset object
    and keeps every script bit-identical.
    """
    sizes = spec.sizes
    if min(sizes) <= 0:
        raise ValueError(
            f"One reconstructed split is empty: train/val/test={sizes} for "
            f"dataset_size={spec.dataset_size}. Adjust --limit, --train-split, or "
            "--val-split."
        )
    permutation = torch.randperm(
        spec.dataset_size,
        generator=torch.Generator().manual_seed(spec.split_seed),
    ).tolist()
    train_end = spec.train_size
    val_end = train_end + spec.val_size
    return (
        permutation[:train_end],
        permutation[train_end:val_end],
        permutation[val_end:],
    )


def hash_values(values: Iterable[str]) -> str:
    """Ordered, unambiguous SHA-256 over a sequence of strings."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitIdentity:
    """Split specification plus the identity of the samples it actually selects."""

    spec: SplitSpec
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    test_sample_ids: tuple[str, ...]
    test_sample_ids_sha256: str
    split_sha256: str

    @property
    def test_size(self) -> int:
        return len(self.test_sample_ids)

    def as_dict(self) -> dict[str, object]:
        payload = self.spec.as_dict()
        payload.update(
            {
                "test_sample_ids_sha256": self.test_sample_ids_sha256,
                "split_sha256": self.split_sha256,
                "split_identity_version": SPLIT_IDENTITY_VERSION,
            }
        )
        return payload


def build_split_identity(spec: SplitSpec, sample_ids: Sequence[str]) -> SplitIdentity:
    """Compute the split and a stable hash over its config and test sample IDs."""
    if len(sample_ids) != spec.dataset_size:
        raise ValueError(
            f"sample_ids length {len(sample_ids)} does not match dataset_size {spec.dataset_size}"
        )
    train_indices, val_indices, test_indices = reconstruct_split_indices(spec)
    test_ids = tuple(str(sample_ids[index]) for index in test_indices)
    test_ids_hash = hash_values(test_ids)
    payload = "|".join(
        [
            f"split_identity_version={SPLIT_IDENTITY_VERSION}",
            f"dataset_size={spec.dataset_size}",
            f"train_split={spec.train_split!r}",
            f"val_split={spec.val_split!r}",
            f"split_seed={spec.split_seed}",
            f"split_train={spec.train_size}",
            f"split_val={spec.val_size}",
            f"split_test={spec.test_size}",
            f"test_sample_ids_sha256={test_ids_hash}",
        ]
    )
    split_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return SplitIdentity(
        spec=spec,
        train_indices=tuple(train_indices),
        val_indices=tuple(val_indices),
        test_indices=tuple(test_indices),
        test_sample_ids=test_ids,
        test_sample_ids_sha256=test_ids_hash,
        split_sha256=split_hash,
    )


def split_spec_from_saved_args(
    saved_args: object, dataset_size: int
) -> tuple[SplitSpec, tuple[str, ...]]:
    """Build a :class:`SplitSpec` from checkpoint args, recording seed fallback."""
    legacy = getattr(saved_args, "seed", None)
    split_seed = getattr(saved_args, "split_seed", None)
    notes: list[str] = []
    if split_seed is None:
        if legacy is None:
            raise ValueError("Checkpoint args contain neither split_seed nor seed")
        split_seed = legacy
        notes.append(
            f"split_seed missing in checkpoint args, reusing legacy seed={legacy} "
            "(checkpoint predates the frozen split-seed protocol)"
        )
    spec = SplitSpec(
        dataset_size=int(dataset_size),
        train_split=float(getattr(saved_args, "train_split", 0.8)),
        val_split=float(getattr(saved_args, "val_split", 0.1)),
        split_seed=int(split_seed),
    )
    return spec, tuple(notes)


def assert_matching_test_identity(
    identities: Sequence[tuple[str, SplitIdentity]],
) -> SplitIdentity:
    """Raise unless every labelled run selects the same test samples in order."""
    if not identities:
        raise ValueError("No split identities supplied")
    reference_label, reference = identities[0]
    for label, identity in identities[1:]:
        assert_same_sample_sequence(
            label, identity.test_sample_ids, reference_label, reference.test_sample_ids
        )
        if identity.spec.split_seed != reference.spec.split_seed:
            raise ValueError(
                f"split_seed mismatch between {reference_label!r} and {label!r}: "
                f"{reference.spec.split_seed} vs {identity.spec.split_seed}"
            )
    return reference


def assert_same_sample_sequence(
    label: str,
    sample_ids: Sequence[str],
    reference_label: str,
    reference_sample_ids: Sequence[str],
) -> None:
    """Raise unless two runs scored exactly the same samples in the same order.

    Comparing seeds and split ratios is not sufficient: the same configuration can
    resolve to different samples when the dataset content changes, so every
    cross-method comparison goes through the actual sample ID sequence.
    """
    if hash_values(sample_ids) == hash_values(reference_sample_ids):
        return
    raise ValueError(
        f"{label} was evaluated on a different test sample sequence than "
        f"{reference_label}: "
        f"{_first_identity_difference(reference_sample_ids, sample_ids)}"
    )


def _first_identity_difference(left: Sequence[str], right: Sequence[str]) -> str:
    if len(left) != len(right):
        return f"different sample counts: {len(left)} vs {len(right)}"
    for position, (left_id, right_id) in enumerate(zip(left, right, strict=True)):
        if left_id != right_id:
            return f"first difference at test position {position + 1}: {left_id} vs {right_id}"
    return "identical sample ID sequences"


# --------------------------------------------------------------------------- #
# Evaluation coverage: full test split vs the subset actually scored
# --------------------------------------------------------------------------- #

FULL_TEST_MODE = "full_test"
MAX_SAMPLES_MODE = "max_samples"
SINGLE_SAMPLE_MODE = "single_sample"


@dataclass(frozen=True)
class EvaluationCoverage:
    """The full test split and the subset that was actually written to metrics.csv."""

    test_split_size: int
    test_split_sample_ids: tuple[str, ...]
    test_split_sample_ids_sha256: str
    split_sha256: str
    evaluated_samples: int
    evaluated_sample_ids: tuple[str, ...]
    evaluated_sample_ids_sha256: str
    evaluation_mode: str
    is_full_test_evaluation: bool
    coverage_verified: bool
    max_samples: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "test_split_size": self.test_split_size,
            "test_split_sample_ids_sha256": self.test_split_sample_ids_sha256,
            "split_sha256": self.split_sha256,
            "evaluated_samples": self.evaluated_samples,
            "evaluated_sample_ids_sha256": self.evaluated_sample_ids_sha256,
            "evaluation_mode": self.evaluation_mode,
            "is_full_test_evaluation": self.is_full_test_evaluation,
            "coverage_verified": self.coverage_verified,
            "max_samples": self.max_samples,
        }


def infer_evaluation_mode(
    evaluated_sample_ids: Sequence[str],
    test_sample_ids: Sequence[str],
    *,
    max_samples: int | None = None,
) -> str:
    """Classify a run from how it was requested, not from the number of samples.

    An explicit ``max_samples`` wins even when it happens to select one sample: asking
    for ``--max-samples 1`` is a prefix evaluation of length one, and must not be
    confused with an explicit single-sample query, which may select any test position.
    """
    evaluated = list(evaluated_sample_ids)
    test = list(test_sample_ids)
    if max_samples is not None:
        return MAX_SAMPLES_MODE
    if evaluated == test:
        return FULL_TEST_MODE
    if len(evaluated) == 1 and evaluated[0] in test:
        return SINGLE_SAMPLE_MODE
    return MAX_SAMPLES_MODE


def build_evaluation_coverage(
    identity: SplitIdentity,
    evaluated_sample_ids: Sequence[str],
    *,
    max_samples: int | None = None,
) -> EvaluationCoverage:
    """Summarise what was evaluated, separating split identity from run coverage.

    ``coverage_verified`` records that the evaluated samples really are what the run
    declares, checked per mode rather than by count:

    * ``full_test``: the sequence equals the frozen test split exactly.
    * ``max_samples``: the first ``min(max_samples, test split size)`` entries of the
      frozen test split.  A ``--max-samples`` budget larger than the split selects the
      whole split, which is a legitimate request and stays a ``max_samples`` run.
    * ``single_sample``: exactly one ID, drawn from anywhere in the frozen test split.

    Duplicate IDs, samples outside the split, reordering, or a sequence that disagrees
    with the declaration are reported as unverified.
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError(f"max_samples must be positive, got {max_samples!r}")
    evaluated = tuple(str(sample_id) for sample_id in evaluated_sample_ids)
    test_ids = tuple(identity.test_sample_ids)
    mode = infer_evaluation_mode(evaluated, test_ids, max_samples=max_samples)
    unique = len(set(evaluated)) == len(evaluated)
    if mode == FULL_TEST_MODE:
        verified = unique and evaluated == test_ids
    elif mode == SINGLE_SAMPLE_MODE:
        verified = unique and len(evaluated) == 1 and evaluated[0] in set(test_ids)
    else:
        # A budget larger than the split selects the whole split, so the expected length
        # is capped rather than compared against the raw request.
        expected_n = None if max_samples is None else min(max_samples, identity.test_size)
        verified = unique and expected_n is not None and evaluated == test_ids[:expected_n]
    return EvaluationCoverage(
        test_split_size=identity.test_size,
        test_split_sample_ids=test_ids,
        test_split_sample_ids_sha256=identity.test_sample_ids_sha256,
        split_sha256=identity.split_sha256,
        evaluated_samples=len(evaluated),
        evaluated_sample_ids=evaluated,
        evaluated_sample_ids_sha256=hash_values(evaluated),
        evaluation_mode=mode,
        # What the artefact covers, derived from the recorded sequence and not from the
        # declared mode, matching how check_evaluation_coverage() reads it back.
        is_full_test_evaluation=unique and evaluated == test_ids,
        coverage_verified=verified,
        max_samples=max_samples,
    )


def parse_optional_bool(value: object) -> bool | None:
    """Parse a metadata boolean, returning ``None`` when the field is absent."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "":
        return None
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean")


@dataclass(frozen=True)
class CoverageCheck:
    """Outcome of validating one evaluation artefact against the frozen split."""

    is_full: bool
    legacy: bool
    mode: str


@dataclass(frozen=True)
class TestSplitReference:
    """The full test split as known to a validator.

    Built either from a checkpoint's reconstructed :class:`SplitIdentity` or from
    the recorded ``test_split_sample_ids.csv`` plus summary hashes, so scripts that
    never touch the dataset can still validate coverage.
    """

    sample_ids: tuple[str, ...]
    sample_ids_sha256: str
    split_sha256: str | None = None

    @property
    def size(self) -> int:
        return len(self.sample_ids)


def test_split_reference_from_identity(identity: SplitIdentity) -> TestSplitReference:
    return TestSplitReference(
        sample_ids=identity.test_sample_ids,
        sample_ids_sha256=identity.test_sample_ids_sha256,
        split_sha256=identity.split_sha256,
    )


def check_evaluation_coverage(
    label: str,
    summary: Mapping[str, object],
    metrics_sample_ids: Sequence[str],
    reference: TestSplitReference,
    *,
    declared_evaluated_sample_ids: Sequence[str] | None = None,
    allow_partial: bool = False,
) -> CoverageCheck:
    """Validate that an evaluation artefact's coverage matches its own claim.

    Cross-method comparisons must not infer a full-test evaluation from the sample
    count alone.  This checks the declared flag, the recorded hashes, the declared
    evaluated ID file and the sample sequence actually present in ``metrics.csv``;
    the sample count is never used as evidence on its own.
    """
    metrics_ids = [str(sample_id) for sample_id in metrics_sample_ids]
    actual_full = metrics_ids == list(reference.sample_ids)
    declared = parse_optional_bool(summary.get("is_full_test_evaluation"))
    legacy = declared is None

    if declared is True and not actual_full:
        raise ValueError(
            f"{label} declares is_full_test_evaluation=true but its metrics.csv holds "
            f"{len(metrics_ids)} samples that do not match the full test split "
            f"({reference.size} samples)."
        )
    if declared is False and actual_full:
        raise ValueError(
            f"{label} declares is_full_test_evaluation=false but its metrics.csv covers "
            "the entire test split. The recorded metadata is inconsistent."
        )

    if declared is not None:
        declared_count = summary.get("evaluated_samples")
        if declared_count is not None and str(declared_count).strip() != "":
            if int(declared_count) != len(metrics_ids):
                raise ValueError(
                    f"{label} records evaluated_samples={declared_count} but metrics.csv "
                    f"contains {len(metrics_ids)} rows."
                )

    if declared_evaluated_sample_ids is not None:
        assert_same_sample_sequence(
            f"{label} evaluated_sample_ids.csv",
            declared_evaluated_sample_ids,
            f"{label} metrics.csv",
            metrics_ids,
        )

    recorded_test_hash = summary.get("test_split_sample_ids_sha256")
    if recorded_test_hash and str(recorded_test_hash) != reference.sample_ids_sha256:
        raise ValueError(
            f"{label} records test_split_sample_ids_sha256={str(recorded_test_hash)[:16]} "
            f"but the reference test split hashes to {reference.sample_ids_sha256[:16]}."
        )
    if reference.split_sha256 is not None:
        recorded_split_hash = summary.get("split_sha256")
        if recorded_split_hash and str(recorded_split_hash) != reference.split_sha256:
            raise ValueError(
                f"{label} records split_sha256={str(recorded_split_hash)[:16]} but the "
                f"reference split hashes to {reference.split_sha256[:16]}."
            )
    recorded_evaluated_hash = summary.get("evaluated_sample_ids_sha256")
    if recorded_evaluated_hash:
        actual_hash = hash_values(metrics_ids)
        if str(recorded_evaluated_hash) != actual_hash:
            raise ValueError(
                f"{label} records evaluated_sample_ids_sha256="
                f"{str(recorded_evaluated_hash)[:16]} but its metrics.csv hashes to "
                f"{actual_hash[:16]}."
            )

    if not actual_full and not allow_partial:
        raise ValueError(
            f"{label} is a partial evaluation ({len(metrics_ids)} of {reference.size} test "
            f"samples, mode={infer_evaluation_mode(metrics_ids, reference.sample_ids)}). "
            "Formal cross-method tables require full-test evaluations; pass "
            "--allow-partial-evaluation only for exploratory subset runs."
        )

    return CoverageCheck(
        is_full=actual_full,
        legacy=legacy,
        mode=_declared_or_inferred_mode(label, summary, metrics_ids, reference),
    )


def _declared_or_inferred_mode(
    label: str,
    summary: Mapping[str, object],
    metrics_ids: Sequence[str],
    reference: TestSplitReference,
) -> str:
    """Return the artefact's own ``evaluation_mode``, checked against its samples.

    The mode records *how the samples were selected* (full split, a ``--max-samples``
    budget, a single ``--sample-id``); it is not a restatement of the sample count.
    Re-inferring it here would silently rewrite a ``--max-samples <split size>`` run
    into ``full_test`` downstream, so the declared value is kept and merely verified.
    Artefacts that predate the field fall back to inference.
    """
    test_ids = [str(sample_id) for sample_id in reference.sample_ids]
    declared = str(summary.get("evaluation_mode") or "").strip()
    if declared not in {FULL_TEST_MODE, MAX_SAMPLES_MODE, SINGLE_SAMPLE_MODE}:
        if declared:
            raise ValueError(
                f"{label} declares an unknown evaluation_mode={declared!r}; expected one "
                f"of {FULL_TEST_MODE!r}, {MAX_SAMPLES_MODE!r}, {SINGLE_SAMPLE_MODE!r}."
            )
        return infer_evaluation_mode(metrics_ids, test_ids)

    if declared == FULL_TEST_MODE and list(metrics_ids) != test_ids:
        raise ValueError(
            f"{label} declares evaluation_mode={FULL_TEST_MODE!r} but its metrics.csv "
            f"holds {len(metrics_ids)} samples that do not match the full test split "
            f"({reference.size} samples)."
        )
    if declared == SINGLE_SAMPLE_MODE and not (
        len(metrics_ids) == 1 and metrics_ids[0] in set(test_ids)
    ):
        raise ValueError(
            f"{label} declares evaluation_mode={SINGLE_SAMPLE_MODE!r} but its metrics.csv "
            f"holds {len(metrics_ids)} samples."
        )
    if declared == MAX_SAMPLES_MODE:
        raw_max = summary.get("max_samples")
        budget = int(raw_max) if raw_max is not None and str(raw_max).strip() != "" else None
        expected_n = reference.size if budget is None else min(budget, reference.size)
        if list(metrics_ids) != test_ids[:expected_n]:
            raise ValueError(
                f"{label} declares evaluation_mode={MAX_SAMPLES_MODE!r} with "
                f"max_samples={budget!r}, which expects the first {expected_n} test "
                f"samples, but its metrics.csv holds {len(metrics_ids)} sample(s)."
            )
    return declared


def read_sample_ids_file(path: str | Path) -> list[str]:
    """Read the ``sample_id`` column of a ``*_sample_ids.csv`` artefact."""
    with Path(path).open() as handle:
        return [row["sample_id"] for row in csv.DictReader(handle)]


def test_split_reference_from_summary(
    summary: Mapping[str, object],
    test_split_ids_path: str | Path,
) -> TestSplitReference:
    """Rebuild the reference test split from recorded files instead of a checkpoint."""
    sample_ids = tuple(read_sample_ids_file(test_split_ids_path))
    recorded_hash = str(summary.get("test_split_sample_ids_sha256", "") or "")
    computed_hash = hash_values(sample_ids)
    if recorded_hash and recorded_hash != computed_hash:
        raise ValueError(
            f"{test_split_ids_path} hashes to {computed_hash[:16]} but the summary records "
            f"test_split_sample_ids_sha256={recorded_hash[:16]}."
        )
    split_hash = summary.get("split_sha256")
    return TestSplitReference(
        sample_ids=sample_ids,
        sample_ids_sha256=computed_hash,
        split_sha256=str(split_hash) if split_hash else None,
    )


# --------------------------------------------------------------------------- #
# Run metadata
# --------------------------------------------------------------------------- #


def git_revision(repo_root: Path | None = None) -> dict[str, object]:
    """Best-effort Git revision, never raises on a non-repository checkout."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    info: dict[str, object] = {"git_commit": None, "git_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info["git_commit"] = commit
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info["git_dirty"] = bool(status)
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Total and trainable parameter counts (``requires_grad=True`` only)."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"parameters": trainable, "trainable_parameters": trainable, "total_parameters": total}
