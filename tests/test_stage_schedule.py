"""Stage-schedule tests (decision D1 for stage 6).

The load-bearing property is that generalising the stage count leaves the existing
two-stage behaviour untouched, so the accepted stage-3/4 results stay comparable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for _path in (PROJECT_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from rtm_inv.stages import (  # noqa: E402
    DEFAULT_TARGET_ALPHA,
    FINAL_WEIGHT,
    NON_FINAL_TOTAL_WEIGHT,
    schedule_as_dict,
    stage_loss_weights,
    stage_target_lambdas,
    stage_targets,
    validate_schedule,
)


class TwoStageCompatibilityTest(unittest.TestCase):
    """N = 2 must reproduce the pre-stage-6 targets and weights exactly."""

    def test_default_alpha_is_the_frozen_value(self) -> None:
        self.assertEqual(DEFAULT_TARGET_ALPHA, 0.5)
        self.assertEqual(NON_FINAL_TOTAL_WEIGHT, 0.3)
        self.assertEqual(FINAL_WEIGHT, 0.7)

    def test_two_stage_lambdas_match_the_historical_targets(self) -> None:
        self.assertEqual(stage_target_lambdas(2, 0.5), (0.5, 1.0))

    def test_two_stage_weights_match_the_historical_weights(self) -> None:
        self.assertEqual(stage_loss_weights(2), (0.3, 0.7))

    def test_two_stage_targets_are_bit_identical_to_the_old_formula(self) -> None:
        generator = torch.Generator().manual_seed(20260713)
        initial = torch.randn(2, 1, 6, 5, generator=generator)
        target = torch.randn(2, 1, 6, 5, generator=generator)
        targets = stage_targets(initial, target, 2, 0.5)
        # The historical stage-1 target, verbatim.
        self.assertTrue(torch.equal(targets[0], initial + 0.5 * (target - initial)))
        # The historical final target was `target` itself, not a recomputed blend.
        self.assertIs(targets[1], target)

    def test_legacy_one_stage_supervises_the_true_model(self) -> None:
        initial = torch.zeros(1, 1, 3, 3)
        target = torch.ones(1, 1, 3, 3)
        targets = stage_targets(initial, target, 1, 0.5)
        self.assertIs(targets[0], target)
        self.assertEqual(stage_loss_weights(1), (1.0,))


class GeneralStageTest(unittest.TestCase):
    def test_lambdas_start_at_alpha_and_end_at_one(self) -> None:
        for num_stages in (2, 3, 4, 5, 8):
            lambdas = stage_target_lambdas(num_stages, 0.5)
            self.assertEqual(len(lambdas), num_stages)
            self.assertAlmostEqual(lambdas[0], 0.5, places=12)
            self.assertAlmostEqual(lambdas[-1], 1.0, places=12)

    def test_three_stage_inserts_the_midpoint(self) -> None:
        self.assertEqual(stage_target_lambdas(3, 0.5), (0.5, 0.75, 1.0))

    def test_four_stage_is_evenly_spaced(self) -> None:
        lambdas = stage_target_lambdas(4, 0.5)
        expected = (0.5, 0.5 + 0.5 / 3, 0.5 + 1.0 / 3, 1.0)
        for got, want in zip(lambdas, expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_lambdas_are_monotonic(self) -> None:
        for num_stages in (2, 3, 4, 7):
            lambdas = stage_target_lambdas(num_stages, 0.5)
            self.assertEqual(list(lambdas), sorted(lambdas))

    def test_non_final_stages_share_the_frozen_total(self) -> None:
        for num_stages in (2, 3, 4, 6):
            weights = stage_loss_weights(num_stages)
            self.assertAlmostEqual(sum(weights), 1.0, places=12)
            self.assertAlmostEqual(weights[-1], FINAL_WEIGHT, places=12)
            self.assertAlmostEqual(
                sum(weights[:-1]), NON_FINAL_TOTAL_WEIGHT, places=12
            )
            for weight in weights[:-1]:
                self.assertAlmostEqual(
                    weight, NON_FINAL_TOTAL_WEIGHT / (num_stages - 1), places=12
                )

    def test_targets_are_between_initial_and_the_true_model(self) -> None:
        initial = torch.zeros(1, 1, 4, 4)
        target = torch.full((1, 1, 4, 4), 4.0)
        for num_stages in (2, 3, 4):
            for current, following in zip(
                stage_targets(initial, target, num_stages, 0.5)[:-1],
                stage_targets(initial, target, num_stages, 0.5)[1:],
            ):
                self.assertTrue(bool((current <= following).all()))
                self.assertTrue(bool((current >= initial).all()))
                self.assertTrue(bool((current <= target).all()))


class ScheduleValidationTest(unittest.TestCase):
    def test_validate_accepts_the_frozen_schedule(self) -> None:
        for num_stages in (2, 3, 4, 5):
            validate_schedule(num_stages, 0.5)

    def test_invalid_stage_counts_and_alphas_are_rejected(self) -> None:
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                stage_target_lambdas(bad)
            with self.assertRaises(ValueError):
                stage_loss_weights(bad)
        for bad_alpha in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                stage_target_lambdas(3, bad_alpha)

    def test_provenance_record_carries_the_schedule(self) -> None:
        record = schedule_as_dict(3, 0.5)
        self.assertEqual(record["num_stages"], 3)
        self.assertEqual(record["target_lambdas"], [0.5, 0.75, 1.0])
        self.assertEqual(record["stage_loss_weights"], [0.15, 0.15, 0.7])
        self.assertEqual(record["data_loss_stage"], "final")

    def test_two_stage_record_matches_the_historical_numbers(self) -> None:
        record = schedule_as_dict(2, 0.5)
        self.assertEqual(record["target_lambdas"], [0.5, 1.0])
        self.assertEqual(record["stage_loss_weights"], [0.3, 0.7])


class ComputeLossTwoStageCompatibilityTest(unittest.TestCase):
    """Exercise the real loss function, not just the schedule helpers."""

    @staticmethod
    def _outputs(initial, stage1, final, num_stages):
        return {
            "num_stages": num_stages,
            "stage1_model": stage1,
            "final_model": final,
            "direct_model_prediction": False,
        }

    def test_two_stage_model_loss_equals_the_legacy_expression(self) -> None:
        import torch.nn.functional as F

        import train

        generator = torch.Generator().manual_seed(7)
        initial = torch.randn(2, 1, 5, 4, generator=generator)
        target = torch.randn(2, 1, 5, 4, generator=generator)
        stage1 = torch.randn(2, 1, 5, 4, generator=generator)
        final = torch.randn(2, 1, 5, 4, generator=generator)

        total, info = train.compute_loss(
            self._outputs(initial, stage1, final, 2),
            target,
            initial,
            stage1_target_alpha=0.5,
        )

        # The pre-stage-6 implementation, verbatim.
        legacy_stage1_target = initial + 0.5 * (target - initial)
        legacy = 0.3 * F.l1_loss(stage1, legacy_stage1_target) + 0.7 * F.l1_loss(final, target)

        self.assertTrue(torch.equal(total, legacy))
        self.assertTrue(torch.equal(total, torch.tensor(float(legacy))))
        self.assertAlmostEqual(info["m_s1_l1"], float(legacy_stage1_target.sub(stage1).abs().mean()), places=12)

    def test_three_stage_loss_uses_evenly_split_and_final_targets(self) -> None:
        import torch.nn.functional as F

        import train

        generator = torch.Generator().manual_seed(11)
        initial = torch.randn(1, 1, 4, 4, generator=generator)
        target = torch.randn(1, 1, 4, 4, generator=generator)
        stage1 = torch.randn(1, 1, 4, 4, generator=generator)
        stage2 = torch.randn(1, 1, 4, 4, generator=generator)
        final = torch.randn(1, 1, 4, 4, generator=generator)

        outputs = {
            "num_stages": 3,
            "stage1_model": stage1,
            "stage2_model": stage2,
            "final_model": final,
            "direct_model_prediction": False,
        }
        total, info = train.compute_loss(
            outputs, target, initial, stage1_target_alpha=0.5
        )

        expected = (
            0.15 * F.l1_loss(stage1, initial + 0.5 * (target - initial))
            + 0.15 * F.l1_loss(stage2, initial + 0.75 * (target - initial))
            + 0.7 * F.l1_loss(final, target)
        )
        self.assertTrue(torch.equal(total, expected))
        self.assertEqual(info["lambda_s1"], 0.5)
        self.assertEqual(info["lambda_s3"], 1.0)
        self.assertAlmostEqual(info["w_s2"], 0.15, places=12)


class NonFinalTotalWeightTest(unittest.TestCase):
    """The stage-6.4 screen needs this knob; its default must not move any result."""

    def test_default_reproduces_the_frozen_weights(self) -> None:
        self.assertEqual(stage_loss_weights(2), (0.3, 0.7))
        self.assertEqual(stage_loss_weights(3), (0.15, 0.15, 0.7))
        for got, expected in zip(stage_loss_weights(4), (0.1, 0.1, 0.1, 0.7)):
            self.assertAlmostEqual(got, expected, places=12)

    def test_default_final_weight_is_the_literal_constant_not_a_derived_one(self) -> None:
        """The default must return the frozen constants themselves.

        Complement arithmetic can differ by an ulp in general (`1.0 - 0.7 != 0.3`), so
        the implementation returns the constants directly on the default path instead of
        relying on `1.0 - total` happening to be exact. This test pins that.
        """
        self.assertNotEqual(1.0 - 0.7, 0.3)
        for num_stages in (2, 3, 4):
            self.assertEqual(stage_loss_weights(num_stages)[-1], FINAL_WEIGHT)
            self.assertEqual(
                stage_loss_weights(num_stages)[0], NON_FINAL_TOTAL_WEIGHT / (num_stages - 1)
            )

    def test_screened_values_split_the_total_and_keep_the_final_stage(self) -> None:
        for got, expected in zip(stage_loss_weights(2, 0.5), (0.5, 0.5)):
            self.assertAlmostEqual(got, expected, places=12)
        for got, expected in zip(stage_loss_weights(2, 0.7), (0.7, 0.3)):
            self.assertAlmostEqual(got, expected, places=12)
        for num_stages in (2, 3, 4):
            for total in (0.0, 0.5, 0.7, 1.0):
                weights = stage_loss_weights(num_stages, total)
                self.assertAlmostEqual(sum(weights), 1.0, places=12)
                self.assertAlmostEqual(weights[-1], 1.0 - total, places=12)

    def test_out_of_range_totals_are_rejected(self) -> None:
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                stage_loss_weights(2, bad)

    def test_schedule_record_tracks_the_parameter(self) -> None:
        self.assertEqual(schedule_as_dict(3, 0.5)["non_final_total_weight"], 0.3)
        self.assertEqual(schedule_as_dict(3, 0.5, 0.5)["stage_loss_weights"], [0.25, 0.25, 0.5])
        self.assertEqual(schedule_as_dict(3, 0.5, 0.5)["final_weight"], 0.5)

    def test_default_loss_stays_bit_identical_to_the_legacy_expression(self) -> None:
        import torch.nn.functional as F

        import train

        generator = torch.Generator().manual_seed(21)
        initial = torch.randn(2, 1, 5, 4, generator=generator)
        target = torch.randn(2, 1, 5, 4, generator=generator)
        stage1 = torch.randn(2, 1, 5, 4, generator=generator)
        final = torch.randn(2, 1, 5, 4, generator=generator)
        outputs = {
            "num_stages": 2,
            "stage1_model": stage1,
            "final_model": final,
            "direct_model_prediction": False,
        }
        total, _ = train.compute_loss(
            outputs, target, initial, stage1_target_alpha=0.5
        )
        legacy = 0.3 * F.l1_loss(stage1, initial + 0.5 * (target - initial)) + 0.7 * F.l1_loss(
            final, target
        )
        self.assertTrue(torch.equal(total, legacy))

    def test_changing_the_total_changes_the_loss(self) -> None:
        import train

        generator = torch.Generator().manual_seed(23)
        initial = torch.randn(2, 1, 5, 4, generator=generator)
        target = torch.randn(2, 1, 5, 4, generator=generator)
        stage1 = torch.randn(2, 1, 5, 4, generator=generator)
        final = torch.randn(2, 1, 5, 4, generator=generator)
        outputs = {
            "num_stages": 2,
            "stage1_model": stage1,
            "final_model": final,
            "direct_model_prediction": False,
        }
        baseline, _ = train.compute_loss(outputs, target, initial, stage1_target_alpha=0.5)
        shifted, _ = train.compute_loss(
            outputs, target, initial, stage1_target_alpha=0.5, non_final_total_weight=0.5
        )
        self.assertFalse(torch.equal(baseline, shifted))


class StageCountIsNoLongerRestrictedTest(unittest.TestCase):
    def test_model_accepts_three_and_four_stages(self) -> None:
        from rtm_inv.models import RTMInvNet

        for num_stages in (1, 2, 3, 4):
            net = RTMInvNet(num_stages=num_stages, rtm_operator=None)
            self.assertEqual(net.num_stages, num_stages)

    def test_model_rejects_a_non_positive_stage_count(self) -> None:
        from rtm_inv.models import RTMInvNet

        for bad in (0, -2):
            with self.assertRaises(ValueError):
                RTMInvNet(num_stages=bad, rtm_operator=None)

    def test_cli_accepts_an_arbitrary_stage_count(self) -> None:
        import train

        self.assertEqual(train.positive_stage_count("1"), 1)
        self.assertEqual(train.positive_stage_count("3"), 3)
        self.assertEqual(train.positive_stage_count("4"), 4)
        for bad in ("0", "-1", "two"):
            with self.assertRaises(Exception):
                train.positive_stage_count(bad)

    def test_cli_parser_accepts_three_and_four_stages(self) -> None:
        import train

        original = sys.argv
        try:
            for num_stages in (1, 2, 3, 4):
                sys.argv = [
                    "train.py",
                    "--data-dir",
                    str(PROJECT_ROOT / "data" / "rock_1000"),
                    "--num-stages",
                    str(num_stages),
                ]
                self.assertEqual(train.parse_args().num_stages, num_stages)
        finally:
            sys.argv = original

    def test_cli_parser_still_rejects_zero_stages(self) -> None:
        import train

        original = sys.argv
        try:
            sys.argv = [
                "train.py",
                "--data-dir",
                str(PROJECT_ROOT / "data" / "rock_1000"),
                "--num-stages",
                "0",
            ]
            with self.assertRaises(SystemExit):
                train.parse_args()
        finally:
            sys.argv = original


if __name__ == "__main__":
    unittest.main()
