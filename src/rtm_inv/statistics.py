"""Paired statistics for the frozen evaluation protocol.

Everything here is deliberately explicit about the degenerate cases that a
100-sample paired comparison actually hits:

* zero differences for every pair (two identical prediction sets),
* zero-variance differences (all pairs shifted by the same amount),
* constant inputs, where a correlation coefficient is undefined,
* zero-length or zero-norm vectors.

Rules used throughout
---------------------
``ZERO_TOLERANCE = 1e-12``
    A norm is treated as zero when it is ``<= ZERO_TOLERANCE``. Any quantity that
    divides by it is reported as ``NaN`` and flagged, never as ``inf``.

``CONSTANT_TOLERANCE = 1e-12``
    A vector is treated as constant (zero variance) when its standard deviation
    is ``<= CONSTANT_TOLERANCE``. Pearson correlation is then undefined and is
    reported as ``NaN`` with flag ``constant_vector``; Spearman correlation is
    also reported as ``NaN`` because the rank vector carries no information.

``all_differences_zero``
    Every paired difference is exactly zero. The permutation and signed-rank
    tests cannot produce a more extreme statistic, so ``p = 1.0``. Cohen's
    ``d_z`` is ``0.0`` (a genuine null effect), the confidence interval is
    ``[0, 0]``, and the rank-biserial correlation is ``0.0``.

``zero_variance_nonzero_differences``
    All differences are equal and non-zero: Cohen's ``d_z`` is undefined
    (``NaN``) and the rank-biserial correlation is exactly ``+/-1`` (complete
    dominance). The permutation p-value is still computed; because a sign flip of
    an all-equal vector is either "all flipped" or "none flipped", it is
    resolution-limited by the Monte Carlo repeat count, which is reported
    alongside it.

Bootstrap and permutation tests are Monte Carlo, seeded from a single
``statistics_seed``. The permutation test is **not** an exhaustive enumeration of
``2**n`` sign patterns, and is never described as one.

A relative improvement is a descriptive percentage, not a standardised effect
size; the standardised effect sizes reported here are Cohen's ``d_z`` (parametric)
and the rank-biserial correlation (non-parametric).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats

from .protocol import assert_same_sample_sequence

STATISTICS_VERSION = "1.0.0"

DEFAULT_STATISTICS_SEED = 20260713
DEFAULT_BOOTSTRAP_REPEATS = 10_000
DEFAULT_PERMUTATION_REPEATS = 10_000

ZERO_TOLERANCE = 1e-12
CONSTANT_TOLERANCE = 1e-12

LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"

# Which direction counts as an improvement, per protocol metric.
METRIC_DIRECTIONS = {
    "mae": LOWER_IS_BETTER,
    "mse": LOWER_IS_BETTER,
    "ssim": HIGHER_IS_BETTER,
    "psnr": HIGHER_IS_BETTER,
}

METRIC_UNITS = {
    "mae": "normalized_model_units",
    "mse": "normalized_model_units_squared",
    "ssim": "unitless",
    "psnr": "dB",
}


def direction_sign(direction: str) -> float:
    if direction == LOWER_IS_BETTER:
        return -1.0
    if direction == HIGHER_IS_BETTER:
        return 1.0
    raise ValueError(f"Unknown direction {direction!r}")


def paired_differences(
    reference: Sequence[float],
    method: Sequence[float],
    direction: str,
) -> np.ndarray:
    """Differences oriented so that a **positive** value means ``method`` improves."""
    reference_array = np.asarray(reference, dtype=float)
    method_array = np.asarray(method, dtype=float)
    if reference_array.shape != method_array.shape:
        raise ValueError("reference and method must have the same number of samples")
    if reference_array.size == 0:
        raise ValueError("paired comparison needs at least one sample")
    return direction_sign(direction) * (method_array - reference_array)


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_STATISTICS_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the paired mean difference.

    Resampling is over sample indices (paired), so the pairing is preserved.
    """
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("paired bootstrap needs at least one difference")
    if repeats < 1:
        raise ValueError("--bootstrap-repeats must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if values.size == 1 or np.all(values == values[0]):
        # Every resample reproduces the same mean; the interval is degenerate.
        mean = float(values.mean())
        return mean, mean
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(repeats), values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def paired_sign_flip_permutation_p(
    differences: Sequence[float],
    *,
    repeats: int = DEFAULT_PERMUTATION_REPEATS,
    seed: int = DEFAULT_STATISTICS_SEED,
) -> float:
    """Two-sided Monte Carlo sign-flip permutation p-value.

    Each replicate multiplies a random subset of the differences by ``-1`` and
    compares ``|mean|`` against the observed ``|mean|``. This is a Monte Carlo
    estimate with the usual ``+1`` correction, **not** an exhaustive enumeration
    of the ``2**n`` sign patterns.
    """
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        raise ValueError("permutation test needs at least one difference")
    if repeats < 1:
        raise ValueError("--permutation-repeats must be at least 1")
    observed = abs(float(values.mean()))
    if np.all(values == 0.0) or observed <= ZERO_TOLERANCE:
        return 1.0
    rng = np.random.default_rng(int(seed) + 1)
    signs = rng.integers(0, 2, size=(int(repeats), values.size)) * 2 - 1
    permuted = np.abs((signs * values).mean(axis=1))
    exceeded = int(np.count_nonzero(permuted >= observed - ZERO_TOLERANCE))
    return float((exceeded + 1) / (int(repeats) + 1))


def cohens_dz(differences: Sequence[float]) -> float:
    """Paired Cohen's ``d_z`` = mean(difference) / sd(difference), sd with ddof=1."""
    values = np.asarray(differences, dtype=float)
    if values.size < 2:
        return float("nan")
    sd = float(values.std(ddof=1))
    if not math.isfinite(sd) or sd <= ZERO_TOLERANCE:
        return float("nan")
    return float(values.mean() / sd)


def rank_biserial_correlation(differences: Sequence[float]) -> float:
    """Rank-biserial correlation ``(R+ - R-) / (R+ + R-)`` over signed ranks.

    Zero differences carry no rank information and are excluded. When every
    difference is zero the correlation is defined as ``0.0`` (no dominance).
    """
    values = np.asarray(differences, dtype=float)
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    if total <= ZERO_TOLERANCE:
        return 0.0
    return float((positive - negative) / total)


def wilcoxon_signed_rank_p(differences: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value; ``1.0`` when all differences are 0.

    scipy warns when the statistic has zero variance (all differences equal and
    non-zero). In that case the p-value is reported as ``NaN`` rather than as the
    spuriously small number the normal approximation produces, and the caller is
    expected to flag the degenerate case.
    """
    values = np.asarray(differences, dtype=float)
    if values.size < 2:
        return float("nan")
    if np.all(values == 0.0):
        return 1.0
    if np.all(values == values[0]):
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        result = stats.wilcoxon(values, alternative="two-sided", method="auto")
    return float(result.pvalue)


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjustment, returned in the original order."""
    values = np.asarray(pvalues, dtype=float)
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    count = values.size
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        candidate = (count - position) * float(values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def pearson_with_p(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, str]:
    """Pearson r with a two-sided p-value; NaN plus a flag when undefined."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.size != y_array.size:
        raise ValueError("correlation inputs must have the same length")
    if x_array.size < 3:
        return float("nan"), float("nan"), "insufficient_pairs"
    if not (np.all(np.isfinite(x_array)) and np.all(np.isfinite(y_array))):
        return float("nan"), float("nan"), "non_finite_input"
    if x_array.std() <= CONSTANT_TOLERANCE or y_array.std() <= CONSTANT_TOLERANCE:
        return float("nan"), float("nan"), "constant_vector"
    with np.errstate(invalid="ignore", divide="ignore"):
        result = stats.pearsonr(x_array, y_array)
    return float(result.statistic), float(result.pvalue), ""


def spearman_with_p(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, str]:
    """Spearman rho with a two-sided p-value; NaN plus a flag when undefined."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.size != y_array.size:
        raise ValueError("correlation inputs must have the same length")
    if x_array.size < 3:
        return float("nan"), float("nan"), "insufficient_pairs"
    if not (np.all(np.isfinite(x_array)) and np.all(np.isfinite(y_array))):
        return float("nan"), float("nan"), "non_finite_input"
    if y_array.std() <= CONSTANT_TOLERANCE or x_array.std() <= CONSTANT_TOLERANCE:
        return float("nan"), float("nan"), "constant_vector"
    with np.errstate(invalid="ignore", divide="ignore"):
        result = stats.spearmanr(x_array, y_array)
    return float(result.statistic), float(result.pvalue), ""


@dataclass(frozen=True)
class PairedComparison:
    """Result of one reference-vs-method paired comparison on one metric."""

    label: str
    metric: str
    direction: str
    samples: int
    reference_mean: float
    method_mean: float
    paired_mean_difference: float
    paired_median_difference: float
    relative_improvement_percent: float
    ci95_lower: float
    ci95_upper: float
    permutation_p: float
    wilcoxon_p: float
    cohens_dz: float
    rank_biserial: float
    win_rate_percent: float
    tie_rate_percent: float
    loss_rate_percent: float
    raw_p: float
    raw_p_source: str
    holm_p: float | None
    holm_family: str
    holm_family_size: int
    bootstrap_repeats: int
    permutation_repeats: int
    statistics_seed: int
    degenerate: str
    difference_unit: str

    def as_dict(self) -> dict[str, object]:
        return {
            "comparison": self.label,
            "metric": self.metric,
            "direction": self.direction,
            "samples": self.samples,
            "reference_mean": self.reference_mean,
            "method_mean": self.method_mean,
            "paired_mean_difference": self.paired_mean_difference,
            "paired_median_difference": self.paired_median_difference,
            "relative_improvement_percent": self.relative_improvement_percent,
            "ci95_lower": self.ci95_lower,
            "ci95_upper": self.ci95_upper,
            "permutation_p": self.permutation_p,
            "wilcoxon_p": self.wilcoxon_p,
            "cohens_dz": self.cohens_dz,
            "rank_biserial": self.rank_biserial,
            "win_rate_percent": self.win_rate_percent,
            "tie_rate_percent": self.tie_rate_percent,
            "loss_rate_percent": self.loss_rate_percent,
            "raw_p": self.raw_p,
            "raw_p_source": self.raw_p_source,
            "holm_p": self.holm_p,
            "holm_family": self.holm_family,
            "holm_family_size": self.holm_family_size,
            "bootstrap_repeats": self.bootstrap_repeats,
            "permutation_repeats": self.permutation_repeats,
            "statistics_seed": self.statistics_seed,
            "degenerate": self.degenerate,
            "difference_unit": self.difference_unit,
        }


def compare_paired(
    label: str,
    metric: str,
    reference: Mapping[str, float],
    method: Mapping[str, float],
    *,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    permutation_repeats: int = DEFAULT_PERMUTATION_REPEATS,
    statistics_seed: int = DEFAULT_STATISTICS_SEED,
    holm_family: str = "",
) -> PairedComparison:
    """Full paired statistics for one metric of one comparison."""
    if metric not in METRIC_DIRECTIONS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {sorted(METRIC_DIRECTIONS)}")
    assert_same_sample_sequence(
        f"{label} ({metric}) method",
        list(method.keys()),
        f"{label} ({metric}) reference",
        list(reference.keys()),
    )
    sample_ids = list(reference.keys())
    reference_values = [float(reference[sample_id]) for sample_id in sample_ids]
    method_values = [float(method[sample_id]) for sample_id in sample_ids]
    direction = METRIC_DIRECTIONS[metric]
    differences = paired_differences(reference_values, method_values, direction)

    reference_mean = float(np.mean(reference_values))
    method_mean = float(np.mean(method_values))
    paired_mean = float(differences.mean())
    paired_median = float(np.median(differences))
    if abs(reference_mean) > ZERO_TOLERANCE:
        relative = paired_mean / abs(reference_mean) * 100.0
    else:
        relative = float("nan")

    lower, upper = paired_bootstrap_ci(
        differences, repeats=bootstrap_repeats, seed=statistics_seed
    )
    permutation_p = paired_sign_flip_permutation_p(
        differences, repeats=permutation_repeats, seed=statistics_seed
    )
    wilcoxon_p = wilcoxon_signed_rank_p(differences)
    dz = cohens_dz(differences)
    rank_biserial = rank_biserial_correlation(differences)

    wins = int(np.count_nonzero(differences > ZERO_TOLERANCE))
    ties = int(np.count_nonzero(np.abs(differences) <= ZERO_TOLERANCE))
    losses = int(differences.size - wins - ties)

    degenerate = ""
    if np.all(differences == 0.0):
        degenerate = "all_differences_zero"
        # Documented rule: two identical prediction sets are a genuine null
        # effect, so d_z is 0 rather than the 0/0 that the formula would give.
        dz = 0.0
        rank_biserial = 0.0
    elif np.all(differences == differences[0]):
        degenerate = "zero_variance_nonzero_differences"

    return PairedComparison(
        label=label,
        metric=metric,
        direction=direction,
        samples=int(differences.size),
        reference_mean=reference_mean,
        method_mean=method_mean,
        paired_mean_difference=paired_mean,
        paired_median_difference=paired_median,
        relative_improvement_percent=relative,
        ci95_lower=lower,
        ci95_upper=upper,
        permutation_p=permutation_p,
        wilcoxon_p=wilcoxon_p,
        cohens_dz=dz,
        rank_biserial=rank_biserial,
        win_rate_percent=wins / differences.size * 100.0,
        tie_rate_percent=ties / differences.size * 100.0,
        loss_rate_percent=losses / differences.size * 100.0,
        raw_p=permutation_p,
        raw_p_source="paired_sign_flip_permutation",
        holm_p=None,
        holm_family=holm_family,
        holm_family_size=0,
        bootstrap_repeats=int(bootstrap_repeats),
        permutation_repeats=int(permutation_repeats),
        statistics_seed=int(statistics_seed),
        degenerate=degenerate,
        difference_unit=METRIC_UNITS[metric],
    )


def apply_holm_family(
    comparisons: Sequence[PairedComparison],
    *,
    family: str,
) -> list[PairedComparison]:
    """Holm-adjust the raw p-values of one declared hypothesis family.

    The family is passed in explicitly so that exploratory analyses (for example
    the refresh-mechanism correlations) can never leak into the confirmatory
    correction by accident.
    """
    from dataclasses import replace

    if not comparisons:
        return []
    adjusted = holm_adjust([comparison.raw_p for comparison in comparisons])
    return [
        replace(
            comparison,
            holm_p=adjusted[index],
            holm_family=family,
            holm_family_size=len(comparisons),
        )
        for index, comparison in enumerate(comparisons)
    ]


def metric_values_from_rows(
    rows: Sequence[Mapping[str, str]],
    key: str,
) -> dict[str, float]:
    """Extract one metric column from ``metrics.csv``-style rows, keyed by sample id."""
    values: dict[str, float] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        raw = row[key]
        if raw is None or str(raw).strip() == "":
            continue
        values[sample_id] = float(raw)
    return values


DESCRIBE_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def describe_distribution(
    values: Sequence[float],
    *,
    quantiles: Sequence[float] = DESCRIBE_QUANTILES,
) -> dict[str, float | int]:
    """Summarise one per-sample quantity, reporting NaN/inf counts explicitly.

    Undefined values (``NaN`` from a zero denominator, a constant image, ...) are
    excluded from the statistics and counted in ``n_non_finite`` instead of being
    silently dropped.
    """
    array = np.asarray(values, dtype=float)
    total = int(array.size)
    finite = array[np.isfinite(array)]
    summary: dict[str, float | int] = {
        "n_total": total,
        "n_valid": int(finite.size),
        "n_non_finite": int(total - finite.size),
        "n_zero": int(np.count_nonzero(finite == 0.0)),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
    }
    if finite.size:
        for quantile, value in zip(quantiles, np.quantile(finite, quantiles)):
            summary[f"p{int(round(quantile * 100)):02d}"] = float(value)
    else:
        for quantile in quantiles:
            summary[f"p{int(round(quantile * 100)):02d}"] = float("nan")
    return summary
