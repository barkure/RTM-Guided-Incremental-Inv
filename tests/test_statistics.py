"""Unit tests for the paired-statistics primitives.

Coverage is deliberately aimed at the cases a 100-sample paired comparison really
hits: known directions, identical prediction sets, zero-variance differences,
constant inputs and mismatched sample order.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rtm_inv.statistics import (  # noqa: E402
    DEFAULT_STATISTICS_SEED,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    apply_holm_family,
    cohens_dz,
    compare_paired,
    holm_adjust,
    metric_values_from_rows,
    paired_bootstrap_ci,
    paired_differences,
    paired_sign_flip_permutation_p,
    pearson_with_p,
    rank_biserial_correlation,
    spearman_with_p,
    wilcoxon_signed_rank_p,
)

FEW_REPEATS = 2_000


def _rows(**metrics: list[float]) -> list[dict[str, str]]:
    """Build metrics.csv-style rows with zero-padded sample ids."""
    size = len(next(iter(metrics.values())))
    rows = []
    for index in range(size):
        row = {"sample_id": f"{index:04d}"}
        row.update({name: str(values[index]) for name, values in metrics.items()})
        rows.append(row)
    return rows


class DifferenceOrientationTests(unittest.TestCase):
    def test_lower_is_better_improvement_is_positive(self) -> None:
        # MAE dropped from 0.5 to 0.4 -> improvement -> positive difference.
        differences = paired_differences([0.5, 0.6], [0.4, 0.5], LOWER_IS_BETTER)
        np.testing.assert_allclose(differences, [0.1, 0.1])

    def test_higher_is_better_improvement_is_positive(self) -> None:
        # PSNR rose from 20 to 25 -> improvement -> positive difference.
        differences = paired_differences([20.0, 21.0], [25.0, 26.0], HIGHER_IS_BETTER)
        np.testing.assert_allclose(differences, [5.0, 5.0])

    def test_regression_is_negative(self) -> None:
        differences = paired_differences([0.4], [0.5], LOWER_IS_BETTER)
        np.testing.assert_allclose(differences, [-0.1])

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_differences([0.1, 0.2], [0.1], LOWER_IS_BETTER)

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_differences([], [], LOWER_IS_BETTER)


class BootstrapTests(unittest.TestCase):
    def test_seed_is_reproducible(self) -> None:
        values = list(np.linspace(-0.2, 0.4, 60))
        first = paired_bootstrap_ci(values, repeats=FEW_REPEATS, seed=20260713)
        second = paired_bootstrap_ci(values, repeats=FEW_REPEATS, seed=20260713)
        self.assertEqual(first, second)

    def test_interval_brackets_the_mean_for_a_clear_improvement(self) -> None:
        values = list(np.linspace(0.05, 0.15, 100))
        lower, upper = paired_bootstrap_ci(values, repeats=FEW_REPEATS, seed=1)
        self.assertLess(lower, float(np.mean(values)))
        self.assertLess(float(np.mean(values)), upper)
        self.assertGreater(lower, 0.0)

    def test_all_zero_differences_give_a_degenerate_interval(self) -> None:
        self.assertEqual(
            paired_bootstrap_ci([0.0] * 10, repeats=FEW_REPEATS, seed=1), (0.0, 0.0)
        )

    def test_zero_variance_differences_give_a_degenerate_interval(self) -> None:
        lower, upper = paired_bootstrap_ci([0.3] * 10, repeats=FEW_REPEATS, seed=1)
        self.assertAlmostEqual(lower, 0.3, places=12)
        self.assertAlmostEqual(upper, 0.3, places=12)
        self.assertEqual(lower, upper)

    def test_invalid_repeats_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([0.1, 0.2], repeats=0, seed=1)


class PermutationTests(unittest.TestCase):
    def test_seed_is_reproducible(self) -> None:
        values = list(np.linspace(0.01, 0.3, 40))
        first = paired_sign_flip_permutation_p(values, repeats=FEW_REPEATS, seed=7)
        second = paired_sign_flip_permutation_p(values, repeats=FEW_REPEATS, seed=7)
        self.assertEqual(first, second)

    def test_one_sided_consistent_improvement_is_significant(self) -> None:
        values = list(np.linspace(0.02, 0.2, 30))
        p = paired_sign_flip_permutation_p(values, repeats=FEW_REPEATS, seed=7)
        self.assertLess(p, 0.01)

    def test_symmetric_noise_is_not_significant(self) -> None:
        rng = np.random.default_rng(0)
        values = list(rng.normal(0.0, 0.1, 80))
        p = paired_sign_flip_permutation_p(values, repeats=FEW_REPEATS, seed=7)
        self.assertGreater(p, 0.05)

    def test_all_zero_differences_give_p_one(self) -> None:
        self.assertEqual(
            paired_sign_flip_permutation_p([0.0] * 20, repeats=FEW_REPEATS, seed=1),
            1.0,
        )

    def test_p_is_never_zero_thanks_to_the_correction(self) -> None:
        p = paired_sign_flip_permutation_p([0.5] * 20, repeats=FEW_REPEATS, seed=1)
        self.assertGreaterEqual(p, 1.0 / (FEW_REPEATS + 1))
        self.assertLessEqual(p, 1.0)

    def test_monte_carlo_is_not_an_exhaustive_claim(self) -> None:
        # With few repeats the resolution floor must follow the repeat count.
        p = paired_sign_flip_permutation_p([0.5] * 20, repeats=20, seed=1)
        self.assertGreaterEqual(p, 1.0 / 21)
        self.assertLessEqual(p, 1.0)


class EffectSizeTests(unittest.TestCase):
    def test_cohens_dz_known_value(self) -> None:
        # mean 1, sd(ddof=1) of [0,1,2] is 1 -> dz = 1
        self.assertAlmostEqual(cohens_dz([0.0, 1.0, 2.0]), 1.0, places=12)

    def test_cohens_dz_is_signed(self) -> None:
        self.assertAlmostEqual(cohens_dz([0.0, -1.0, -2.0]), -1.0, places=12)

    def test_cohens_dz_is_nan_for_zero_variance(self) -> None:
        self.assertTrue(math.isnan(cohens_dz([0.5] * 5)))
        self.assertTrue(math.isnan(cohens_dz([0.0] * 5)))

    def test_cohens_dz_needs_two_samples(self) -> None:
        self.assertTrue(math.isnan(cohens_dz([0.3])))

    def test_rank_biserial_complete_dominance(self) -> None:
        self.assertAlmostEqual(rank_biserial_correlation([1.0, 2.0, 3.0]), 1.0)
        self.assertAlmostEqual(rank_biserial_correlation([-1.0, -2.0, -3.0]), -1.0)

    def test_rank_biserial_ignores_zero_differences(self) -> None:
        self.assertAlmostEqual(rank_biserial_correlation([1.0, 2.0, 0.0]), 1.0)

    def test_rank_biserial_all_zero_is_zero(self) -> None:
        self.assertEqual(rank_biserial_correlation([0.0] * 4), 0.0)

    def test_rank_biserial_mixed_signs(self) -> None:
        value = rank_biserial_correlation([1.0, -1.0])
        self.assertAlmostEqual(value, 0.0)


class WilcoxonTests(unittest.TestCase):
    def test_all_zero_differences_give_p_one(self) -> None:
        self.assertEqual(wilcoxon_signed_rank_p([0.0] * 8), 1.0)

    def test_zero_variance_nonzero_is_nan(self) -> None:
        self.assertTrue(math.isnan(wilcoxon_signed_rank_p([0.4] * 8)))

    def test_consistent_improvement_is_significant(self) -> None:
        values = [0.1 * (index + 1) for index in range(12)]
        self.assertLess(wilcoxon_signed_rank_p(values), 0.01)

    def test_single_sample_is_nan(self) -> None:
        self.assertTrue(math.isnan(wilcoxon_signed_rank_p([0.2])))


class CorrelationTests(unittest.TestCase):
    def test_pearson_known_positive(self) -> None:
        r, p, flag = pearson_with_p([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        self.assertAlmostEqual(r, 1.0, places=12)
        self.assertLess(p, 0.01)
        self.assertEqual(flag, "")

    def test_pearson_known_negative(self) -> None:
        r, _p, _flag = pearson_with_p([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        self.assertAlmostEqual(r, -1.0, places=12)

    def test_pearson_constant_input_is_flagged(self) -> None:
        r, _p, flag = pearson_with_p([1.0] * 5, [1, 2, 3, 4, 5])
        self.assertTrue(math.isnan(r))
        self.assertEqual(flag, "constant_vector")

    def test_pearson_non_finite_is_flagged(self) -> None:
        r, _p, flag = pearson_with_p([1.0, math.nan, 3.0], [1.0, 2.0, 3.0])
        self.assertTrue(math.isnan(r))
        self.assertEqual(flag, "non_finite_input")

    def test_pearson_needs_three_pairs(self) -> None:
        r, _p, flag = pearson_with_p([1.0, 2.0], [2.0, 1.0])
        self.assertTrue(math.isnan(r))
        self.assertEqual(flag, "insufficient_pairs")

    def test_spearman_detects_monotone_nonlinearity(self) -> None:
        rho, p, flag = spearman_with_p([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
        self.assertAlmostEqual(rho, 1.0, places=12)
        self.assertLess(p, 0.05)
        self.assertEqual(flag, "")

    def test_spearman_constant_input_is_flagged(self) -> None:
        rho, _p, flag = spearman_with_p([2.0] * 5, [1, 2, 3, 4, 5])
        self.assertTrue(math.isnan(rho))
        self.assertEqual(flag, "constant_vector")

    def test_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pearson_with_p([1.0, 2.0, 3.0], [1.0, 2.0])


class HolmTests(unittest.TestCase):
    def test_known_example(self) -> None:
        # Holm step-down for 0.01, 0.02, 0.03, 0.04 with m=4:
        # 4*0.01=0.04, 3*0.02=0.06, 2*0.03=0.06, 1*0.04=0.04 -> cumulative max,
        # so the last one is carried up to 0.06 by the monotonicity constraint.
        adjusted = holm_adjust([0.01, 0.02, 0.03, 0.04])
        self.assertAlmostEqual(adjusted[0], 0.04)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.06)
        self.assertAlmostEqual(adjusted[3], 0.06)

    def test_preserves_input_order(self) -> None:
        adjusted = holm_adjust([0.04, 0.01, 0.03, 0.02])
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[0], 0.06)

    def test_is_monotone_in_sorted_order(self) -> None:
        pvalues = [0.001, 0.004, 0.02, 0.03, 0.2, 0.6]
        adjusted = holm_adjust(pvalues)
        ordered = [value for _, value in sorted(zip(pvalues, adjusted))]
        for left, right in zip(ordered, ordered[1:]):
            self.assertLessEqual(left, right + 1e-15)

    def test_is_never_below_the_raw_pvalue(self) -> None:
        pvalues = [0.001, 0.02, 0.03, 0.5]
        for raw, adjusted in zip(pvalues, holm_adjust(pvalues)):
            self.assertGreaterEqual(adjusted, raw - 1e-15)

    def test_is_bounded_by_one(self) -> None:
        self.assertTrue(all(value <= 1.0 for value in holm_adjust([0.9, 0.95, 0.99])))

    def test_single_hypothesis_is_unchanged(self) -> None:
        self.assertAlmostEqual(holm_adjust([0.037])[0], 0.037)

    def test_empty_input(self) -> None:
        self.assertEqual(holm_adjust([]), [])


class ComparePairedTests(unittest.TestCase):
    @staticmethod
    def _varied_drop(size: int, base: float = 0.05) -> dict[str, float]:
        """A per-sample improvement that varies, so differences have variance."""
        return {f"{index:04d}": base * (0.5 + (index % 5) / 5.0) for index in range(size)}

    def test_lower_is_better_reports_a_positive_improvement(self) -> None:
        reference = {f"{index:04d}": 0.5 - 0.001 * index for index in range(30)}
        drops = self._varied_drop(30)
        method = {key: value - drops[key] for key, value in reference.items()}
        comparison = compare_paired(
            "ref vs method",
            "mae",
            reference,
            method,
            bootstrap_repeats=FEW_REPEATS,
            permutation_repeats=FEW_REPEATS,
        )
        self.assertGreater(comparison.paired_mean_difference, 0.0)
        self.assertLess(comparison.permutation_p, 0.01)
        self.assertEqual(comparison.samples, 30)
        self.assertEqual(comparison.win_rate_percent, 100.0)
        self.assertEqual(comparison.loss_rate_percent, 0.0)
        self.assertEqual(comparison.difference_unit, "normalized_model_units")
        self.assertEqual(comparison.degenerate, "")
        self.assertTrue(math.isfinite(comparison.cohens_dz))

    def test_higher_is_better_reports_a_positive_improvement(self) -> None:
        reference = {f"{index:04d}": 20.0 + 0.1 * index for index in range(30)}
        method = {key: value + 3.0 for key, value in reference.items()}
        comparison = compare_paired(
            "ref vs method",
            "psnr",
            reference,
            method,
            bootstrap_repeats=FEW_REPEATS,
            permutation_repeats=FEW_REPEATS,
        )
        self.assertGreater(comparison.paired_mean_difference, 0.0)
        self.assertEqual(comparison.difference_unit, "dB")
        self.assertEqual(comparison.win_rate_percent, 100.0)

    def test_identical_predictions_are_a_zero_effect(self) -> None:
        reference = {f"{index:04d}": 0.2 + 0.01 * index for index in range(20)}
        comparison = compare_paired(
            "same vs same",
            "mae",
            reference,
            dict(reference),
            bootstrap_repeats=FEW_REPEATS,
            permutation_repeats=FEW_REPEATS,
        )
        self.assertEqual(comparison.paired_mean_difference, 0.0)
        self.assertEqual(comparison.permutation_p, 1.0)
        self.assertEqual(comparison.wilcoxon_p, 1.0)
        self.assertEqual(comparison.cohens_dz, 0.0)
        self.assertEqual(comparison.rank_biserial, 0.0)
        self.assertEqual((comparison.ci95_lower, comparison.ci95_upper), (0.0, 0.0))
        self.assertEqual(comparison.win_rate_percent, 0.0)
        self.assertEqual(comparison.tie_rate_percent, 100.0)
        self.assertEqual(comparison.degenerate, "all_differences_zero")

    def test_zero_variance_nonzero_differences_are_flagged(self) -> None:
        reference = {f"{index:04d}": 0.3 for index in range(12)}
        method = {key: 0.25 for key in reference}
        comparison = compare_paired(
            "ref vs method",
            "mae",
            reference,
            method,
            bootstrap_repeats=FEW_REPEATS,
            permutation_repeats=FEW_REPEATS,
        )
        self.assertEqual(comparison.degenerate, "zero_variance_nonzero_differences")
        self.assertTrue(math.isnan(comparison.wilcoxon_p))
        self.assertAlmostEqual(comparison.rank_biserial, 1.0)

    def test_sample_order_mismatch_is_rejected(self) -> None:
        reference = {f"{index:04d}": 0.5 for index in range(10)}
        method = {f"{index:04d}": 0.4 for index in range(10)}
        method = dict(reversed(list(method.items())))
        with self.assertRaises(ValueError) as context:
            compare_paired("ref vs method", "mae", reference, method)
        self.assertIn("different test sample sequence", str(context.exception))

    def test_unknown_metric_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compare_paired("ref vs method", "iou", {"0000": 1.0}, {"0000": 2.0})

    def test_raw_p_is_the_permutation_p_and_is_labelled(self) -> None:
        reference = {f"{index:04d}": 0.5 for index in range(12)}
        method = {key: value - 0.02 * (index + 1) % 0.05 for index, (key, value) in enumerate(reference.items())}
        comparison = compare_paired(
            "ref vs method",
            "mae",
            reference,
            method,
            bootstrap_repeats=100,
            permutation_repeats=100,
        )
        self.assertEqual(comparison.raw_p, comparison.permutation_p)
        self.assertEqual(comparison.raw_p_source, "paired_sign_flip_permutation")

    def test_as_dict_exposes_every_required_field(self) -> None:
        reference = {f"{index:04d}": 0.5 for index in range(10)}
        method = {key: value - 0.01 for key, value in reference.items()}
        payload = compare_paired(
            "ref vs method",
            "mae",
            reference,
            method,
            bootstrap_repeats=100,
            permutation_repeats=100,
        ).as_dict()
        for field in (
            "reference_mean",
            "method_mean",
            "paired_mean_difference",
            "relative_improvement_percent",
            "ci95_lower",
            "ci95_upper",
            "permutation_p",
            "wilcoxon_p",
            "cohens_dz",
            "rank_biserial",
            "win_rate_percent",
            "tie_rate_percent",
            "loss_rate_percent",
            "raw_p",
            "holm_p",
        ):
            self.assertIn(field, payload)


class HolmFamilyTests(unittest.TestCase):
    def _comparison(self, label: str, metric: str, drop: float):
        reference = {f"{index:04d}": 0.5 + 0.001 * index for index in range(24)}
        method = {key: value - drop for key, value in reference.items()}
        return compare_paired(
            label,
            metric,
            reference,
            method,
            bootstrap_repeats=200,
            permutation_repeats=200,
        )

    def test_family_size_and_label_are_recorded(self) -> None:
        family = [self._comparison("a", "mae", 0.05), self._comparison("a", "mse", 0.01)]
        adjusted = apply_holm_family(family, family="main")
        self.assertTrue(all(comparison.holm_family == "main" for comparison in adjusted))
        self.assertTrue(all(comparison.holm_family_size == 2 for comparison in adjusted))
        self.assertTrue(all(comparison.holm_p is not None for comparison in adjusted))

    def test_holm_is_monotone_and_never_below_raw(self) -> None:
        family = [
            self._comparison("a", "mae", 0.05),
            self._comparison("a", "mse", 0.03),
            self._comparison("b", "mae", 0.02),
            self._comparison("b", "ssim", 0.001),
        ]
        adjusted = apply_holm_family(family, family="main")
        for comparison in adjusted:
            self.assertGreaterEqual(comparison.holm_p, comparison.raw_p - 1e-15)
            self.assertLessEqual(comparison.holm_p, 1.0)

    def test_empty_family(self) -> None:
        self.assertEqual(apply_holm_family([], family="main"), [])


class MetricExtractionTests(unittest.TestCase):
    def test_extracts_by_sample_id(self) -> None:
        rows = _rows(final_mae=[0.1, 0.2, 0.3])
        values = metric_values_from_rows(rows, "final_mae")
        self.assertEqual(list(values), ["0000", "0001", "0002"])
        self.assertAlmostEqual(values["0002"], 0.3)

    def test_skips_blank_values(self) -> None:
        rows = [{"sample_id": "0000", "final_mae": ""}, {"sample_id": "0001", "final_mae": "0.2"}]
        self.assertEqual(list(metric_values_from_rows(rows, "final_mae")), ["0001"])


if __name__ == "__main__":
    unittest.main()
