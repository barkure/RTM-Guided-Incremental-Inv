"""Unit tests for realistic initial-model construction and its leakage accounting.

The dataset fixtures are deliberately tiny synthetic ``sample_*/data.pt`` trees so
the tests can assert *exactly* which targets a mode reads.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rtm_inv.data import RTMDataset  # noqa: E402
from rtm_inv.initialization import (  # noqa: E402
    DEFAULT_RANDOM_CLIP_PERCENT,
    DEFAULT_RANDOM_STD_PERCENT,
    INITIALIZATION_MODES,
    MODE_BIASED_GLOBAL_CONSTANT,
    MODE_CALIBRATION_DERIVED,
    MODE_ORACLE_MEAN,
    MODE_RANDOM_BIASED_GLOBAL_CONSTANT,
    MODE_TRAIN_GLOBAL_CONSTANT,
    InitializationSpec,
    TrainConstant,
    build_initialization_spec,
    compute_train_global_constant,
    make_initial_model,
)
from rtm_inv.protocol import hash_values  # noqa: E402

SIZE = 30


def _write_sample(root: Path, index: int, *, mean: float, constant: bool = False) -> float:
    """Create one synthetic sample whose target mean is ``mean``.

    Returns the value that ``oracle_mean`` should produce for it.
    """
    sample_dir = root / f"sample_{index:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(index)
    if constant:
        permittivity = torch.full((6, 8), mean)
    else:
        permittivity = mean + 0.5 * torch.rand((6, 8), generator=generator) - 0.25
    torch.save(
        {
            "permittivity": permittivity,
            # B-scans are (shot, receiver, time) in this repository.
            "bscan_processed": torch.zeros((4, 1, 8)),
            "bscan_raw": torch.zeros((4, 1, 8)),
        },
        sample_dir / "data.pt",
    )
    (sample_dir / "meta.json").write_text(
        json.dumps(
            {
                "sample_id": f"{index:04d}",
                "dx": 0.025,
                "dz": 0.025,
                "dt": 0.0001,
                "freq_mhz": 400.0,
                "nt": 8,
                "nshot": 4,
                "length_m": 0.2,
                "depth_m": 0.15,
                "src_x_start": 0.0,
                "src_x_step": 0.05,
                "src_z": 0.025,
                "rec_offset_x": 0.025,
                "rec_z": 0.025,
                "pml_width": 5,
                "accuracy": 4,
            }
        )
    )
    return float(permittivity.mean().item())


class InitializationFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rtm-init-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.means: dict[int, float] = {}
        for index in range(SIZE):
            self.means[index] = _write_sample(self.tmp, index, mean=3.0 + 0.05 * index)
        self.train_indices = list(range(0, 20))
        self.val_indices = list(range(20, 25))
        self.test_indices = list(range(25, 30))

    def dataset(self, **kwargs) -> RTMDataset:
        return RTMDataset(self.tmp, normalize_models=True, **kwargs)


class TrainConstantTests(InitializationFixture):
    def test_c_train_uses_only_training_indices(self) -> None:
        dataset = self.dataset()
        constant = compute_train_global_constant(dataset, self.train_indices)
        expected = float(np.mean([self.means[i] for i in self.train_indices]))
        self.assertAlmostEqual(constant.value, expected, places=5)
        self.assertEqual(constant.train_samples, len(self.train_indices))

    def test_changing_validation_and_test_targets_does_not_change_c_train(self) -> None:
        dataset = self.dataset()
        before = compute_train_global_constant(dataset, self.train_indices)

        for index in self.val_indices + self.test_indices:
            _write_sample(self.tmp, index, mean=9.5, constant=True)
        after = compute_train_global_constant(dataset, self.train_indices)

        self.assertEqual(before.value, after.value)
        self.assertEqual(before.train_sample_ids_sha256, after.train_sample_ids_sha256)

    def test_changing_a_training_target_does_change_c_train(self) -> None:
        before = compute_train_global_constant(self.dataset(), self.train_indices)
        _write_sample(self.tmp, self.train_indices[0], mean=9.5, constant=True)
        after = compute_train_global_constant(self.dataset(), self.train_indices)
        self.assertNotEqual(before.value, after.value)

    def test_train_sample_ids_are_hashed(self) -> None:
        dataset = self.dataset()
        constant = compute_train_global_constant(dataset, self.train_indices)
        expected_ids = [dataset.samples[i].sample_id for i in self.train_indices]
        self.assertEqual(constant.train_sample_ids, tuple(expected_ids))
        self.assertEqual(constant.train_sample_ids_sha256, hash_values(expected_ids))

    def test_empty_training_indices_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_train_global_constant(self.dataset(), [])


class OracleMeanTests(InitializationFixture):
    def test_oracle_mean_tracks_the_sample_target(self) -> None:
        dataset = self.dataset()
        for index in self.test_indices[:3]:
            item = dataset[index]
            expected = (dataset.target_permittivity(index).mean() - 2.0) / 8.0
            self.assertAlmostEqual(float(item["initial_model"].mean()), float(expected), places=6)

    def test_oracle_mean_changes_when_one_target_changes(self) -> None:
        dataset = self.dataset()
        before = float(dataset[self.test_indices[0]]["initial_model"].mean())
        _write_sample(self.tmp, self.test_indices[0], mean=9.0, constant=True)
        after = float(dataset[self.test_indices[0]]["initial_model"].mean())
        self.assertNotEqual(before, after)

    def test_oracle_mean_leakage_flags(self) -> None:
        spec = InitializationSpec(mode=MODE_ORACLE_MEAN)
        flags = spec.leakage_flags(evaluated_split="test")
        self.assertTrue(flags["uses_test_target"])
        self.assertTrue(flags["reads_sample_target"])
        self.assertFalse(flags["uses_validation_target"])
        self.assertFalse(spec.is_realistic)


class GlobalConstantTests(InitializationFixture):
    def setUp(self) -> None:
        super().setUp()
        self.constant = compute_train_global_constant(self.dataset(), self.train_indices)

    def _spec(self, **kwargs) -> InitializationSpec:
        return build_initialization_spec(
            MODE_TRAIN_GLOBAL_CONSTANT, train_constant=self.constant, **kwargs
        )

    def test_every_sample_gets_the_same_background(self) -> None:
        spec = self._spec()
        values = {spec.background_value(f"{index:04d}") for index in range(SIZE)}
        self.assertEqual(len(values), 1)
        self.assertAlmostEqual(values.pop(), self.constant.value, places=10)

    def test_initial_model_is_constant_physical_value(self) -> None:
        dataset = self.dataset(
            initial_model_mode=MODE_TRAIN_GLOBAL_CONSTANT, initializer=self._spec()
        )
        for index in (0, 17, 29):
            physical = dataset[index]["initial_model"] * 8.0 + 2.0
            self.assertAlmostEqual(float(physical.min()), self.constant.value, places=5)
            self.assertAlmostEqual(float(physical.max()), self.constant.value, places=5)

    def test_same_value_for_train_val_and_test_samples(self) -> None:
        dataset = self.dataset(
            initial_model_mode=MODE_TRAIN_GLOBAL_CONSTANT, initializer=self._spec()
        )
        means = {
            float(dataset[index]["initial_model"].mean())
            for index in self.train_indices[:2] + self.val_indices[:2] + self.test_indices[:2]
        }
        self.assertEqual(len(means), 1)

    def test_leakage_flags_exclude_validation_and_test(self) -> None:
        flags = self._spec().leakage_flags(evaluated_split="test")
        self.assertTrue(flags["uses_train_target"])
        self.assertFalse(flags["uses_validation_target"])
        self.assertFalse(flags["uses_test_target"])
        self.assertFalse(flags["reads_sample_target"])
        self.assertTrue(self._spec().is_realistic)

    def test_missing_constant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            InitializationSpec(mode=MODE_TRAIN_GLOBAL_CONSTANT)

    def test_dataset_requires_an_initializer_for_non_oracle_modes(self) -> None:
        with self.assertRaises(ValueError):
            self.dataset(initial_model_mode=MODE_TRAIN_GLOBAL_CONSTANT)

    def test_dataset_rejects_mismatched_initializer(self) -> None:
        with self.assertRaises(ValueError):
            self.dataset(
                initial_model_mode=MODE_BIASED_GLOBAL_CONSTANT, initializer=self._spec()
            )


class BiasedConstantTests(InitializationFixture):
    def setUp(self) -> None:
        super().setUp()
        self.constant = compute_train_global_constant(self.dataset(), self.train_indices)

    def _spec(self, bias: float) -> InitializationSpec:
        return build_initialization_spec(
            MODE_BIASED_GLOBAL_CONSTANT,
            train_constant=self.constant,
            bias_percent=bias,
        )

    def test_bias_values_are_exact(self) -> None:
        for bias in (-0.20, -0.10, 0.10, 0.20):
            with self.subTest(bias=bias):
                spec = self._spec(bias)
                expected = self.constant.value * (1.0 + bias)
                self.assertAlmostEqual(
                    spec.background_value("0000"), expected, places=10
                )

    def test_bias_does_not_start_from_a_per_sample_mean(self) -> None:
        """The biased value must be identical for samples with different targets."""
        spec = self._spec(0.1)
        values = {spec.background_value(dataset_id) for dataset_id in ("0000", "0012", "0029")}
        self.assertEqual(len(values), 1)

    def test_clipping_to_the_model_range(self) -> None:
        spec = InitializationSpec(
            mode=MODE_BIASED_GLOBAL_CONSTANT,
            model_min=2.0,
            model_max=10.0,
            train_constant=9.5,
            bias_percent=0.5,
        )
        self.assertGreater(spec.unclipped_background_value("0000"), 10.0)
        self.assertAlmostEqual(spec.background_value("0000"), 10.0, places=10)

        low = InitializationSpec(
            mode=MODE_BIASED_GLOBAL_CONSTANT,
            model_min=2.0,
            model_max=10.0,
            train_constant=2.2,
            bias_percent=-0.5,
        )
        self.assertLess(low.unclipped_background_value("0000"), 2.0)
        self.assertAlmostEqual(low.background_value("0000"), 2.0, places=10)


class RandomBiasedConstantTests(InitializationFixture):
    def setUp(self) -> None:
        super().setUp()
        self.constant = compute_train_global_constant(self.dataset(), self.train_indices)
        self.sample_ids = [f"{index:04d}" for index in range(SIZE)]

    def _spec(self, seed: int = 20260713, **kwargs) -> InitializationSpec:
        return build_initialization_spec(
            MODE_RANDOM_BIASED_GLOBAL_CONSTANT,
            train_constant=self.constant,
            random_seed=seed,
            **kwargs,
        )

    def test_same_seed_is_reproducible(self) -> None:
        first = [self._spec().background_value(s) for s in self.sample_ids]
        second = [self._spec().background_value(s) for s in self.sample_ids]
        self.assertEqual(first, second)

    def test_different_seed_changes_the_realisation(self) -> None:
        first = [self._spec(seed=1).background_value(s) for s in self.sample_ids]
        second = [self._spec(seed=2).background_value(s) for s in self.sample_ids]
        self.assertNotEqual(first, second)

    def test_realisation_is_identical_across_methods(self) -> None:
        """Two specs with the same parameters must agree per sample, whatever order
        they are queried in -- that is what makes cross-method comparison valid."""
        spec_a = self._spec()
        spec_b = self._spec()
        forward = {s: spec_a.background_value(s) for s in self.sample_ids}
        backward = {s: spec_b.background_value(s) for s in reversed(self.sample_ids)}
        self.assertEqual(forward, backward)

    def test_values_are_centred_around_the_constant(self) -> None:
        values = np.asarray([self._spec().background_value(s) for s in self.sample_ids])
        self.assertAlmostEqual(float(values.mean()), self.constant.value, delta=0.05)
        self.assertGreater(float(values.std()), 0.0)

    def test_truncation_keeps_deviations_within_the_declared_range(self) -> None:
        spec = self._spec()
        deviations = [
            abs(spec.background_value(s) / self.constant.value - 1.0)
            for s in self.sample_ids * 5
        ]
        self.assertLessEqual(max(deviations), DEFAULT_RANDOM_CLIP_PERCENT + 1e-9)

    def test_standard_deviation_setting_is_honoured(self) -> None:
        narrow = self._spec(random_std_percent=0.01, random_clip_percent=0.10)
        wide = self._spec(random_std_percent=0.20, random_clip_percent=0.90)
        narrow_spread = np.std([narrow.background_value(s) for s in self.sample_ids])
        wide_spread = np.std([wide.background_value(s) for s in self.sample_ids])
        self.assertLess(narrow_spread, wide_spread)

    def test_seed_is_required(self) -> None:
        with self.assertRaises(ValueError):
            InitializationSpec(
                mode=MODE_RANDOM_BIASED_GLOBAL_CONSTANT,
                train_constant=self.constant.value,
            )

    def test_defaults_are_the_documented_ones(self) -> None:
        spec = self._spec()
        self.assertEqual(spec.random_std_percent, DEFAULT_RANDOM_STD_PERCENT)
        self.assertEqual(spec.random_clip_percent, DEFAULT_RANDOM_CLIP_PERCENT)


class CalibrationModeTests(InitializationFixture):
    def test_calibration_mode_raises_rather_than_fabricating(self) -> None:
        spec = InitializationSpec(mode=MODE_CALIBRATION_DERIVED)
        with self.assertRaises(NotImplementedError) as context:
            spec.background_value("0000")
        self.assertIn("target leakage", str(context.exception))
        self.assertTrue(spec.is_realistic)


class ScaleConversionTests(InitializationFixture):
    def test_initial_model_is_defined_in_physical_then_normalised(self) -> None:
        constant = compute_train_global_constant(self.dataset(), self.train_indices)
        spec = build_initialization_spec(MODE_TRAIN_GLOBAL_CONSTANT, train_constant=constant)
        dataset = self.dataset(
            initial_model_mode=MODE_TRAIN_GLOBAL_CONSTANT, initializer=spec
        )
        normalized = dataset[0]["initial_model"]
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        expected = (constant.value - 2.0) / 8.0
        self.assertAlmostEqual(float(normalized.mean()), expected, places=6)

    def test_make_initial_model_shape_and_value(self) -> None:
        target = torch.full((6, 8), 4.0)
        model = make_initial_model(target, 3.5)
        self.assertEqual(tuple(model.shape), (1, 6, 8))
        self.assertTrue(torch.allclose(model, torch.full((1, 6, 8), 3.5)))

    def test_make_initial_model_rejects_non_finite(self) -> None:
        with self.assertRaises(ValueError):
            make_initial_model(torch.zeros((2, 2)), float("nan"))


class CompatibilityTests(InitializationFixture):
    def test_default_mode_is_oracle_mean(self) -> None:
        dataset = self.dataset()
        self.assertEqual(dataset.initial_model_mode, MODE_ORACLE_MEAN)
        self.assertIsNone(dataset.initializer)

    def test_oracle_mean_matches_the_legacy_formula(self) -> None:
        """Backward compatibility: the inline mean-of-target must be reproduced."""
        dataset = self.dataset()
        for index in (0, 12, 29):
            permittivity = dataset.target_permittivity(index)
            legacy = torch.full_like(permittivity, permittivity.mean()).unsqueeze(0)
            legacy = (legacy - 2.0) / 8.0
            self.assertTrue(
                torch.allclose(dataset[index]["initial_model"], legacy, atol=1e-6)
            )

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RTMDataset(self.tmp, initial_model_mode="not_a_mode")

    def test_all_documented_modes_are_accepted_by_the_spec(self) -> None:
        for mode in INITIALIZATION_MODES:
            with self.subTest(mode=mode):
                if mode in (MODE_TRAIN_GLOBAL_CONSTANT, MODE_BIASED_GLOBAL_CONSTANT):
                    spec = InitializationSpec(mode=mode, train_constant=4.0)
                elif mode == MODE_RANDOM_BIASED_GLOBAL_CONSTANT:
                    spec = InitializationSpec(
                        mode=mode, train_constant=4.0, random_seed=1
                    )
                else:
                    spec = InitializationSpec(mode=mode)
                self.assertIn(spec.mode, INITIALIZATION_MODES)


class MetadataTests(InitializationFixture):
    def setUp(self) -> None:
        super().setUp()
        self.constant = TrainConstant(
            value=4.0,
            train_samples=20,
            train_pixels=960,
            train_sample_ids=tuple(f"{index:04d}" for index in range(20)),
            train_sample_ids_sha256=hash_values([f"{index:04d}" for index in range(20)]),
        )

    def test_realistic_spec_metadata_marks_no_test_leakage(self) -> None:
        spec = build_initialization_spec(
            MODE_BIASED_GLOBAL_CONSTANT, train_constant=self.constant, bias_percent=0.1
        )
        payload = spec.as_dict(evaluated_split="test")
        self.assertFalse(payload["uses_test_target"])
        self.assertFalse(payload["uses_validation_target"])
        self.assertTrue(payload["uses_train_target"])
        self.assertTrue(payload["initialization_is_realistic"])
        self.assertFalse(payload["initialization_reads_sample_target"])
        self.assertEqual(payload["initialization_train_constant"], 4.0)
        self.assertEqual(payload["train_constant_pixels"], 960)
        self.assertEqual(payload["initialization_bias_percent"], 0.1)

    def test_oracle_spec_metadata_marks_test_leakage(self) -> None:
        payload = InitializationSpec(mode=MODE_ORACLE_MEAN).as_dict(evaluated_split="test")
        self.assertTrue(payload["uses_test_target"])
        self.assertFalse(payload["initialization_is_realistic"])

    def test_random_spec_metadata_records_seed_and_distribution(self) -> None:
        spec = build_initialization_spec(
            MODE_RANDOM_BIASED_GLOBAL_CONSTANT,
            train_constant=self.constant,
            random_seed=7,
            random_std_percent=0.05,
            random_clip_percent=0.15,
        )
        payload = spec.as_dict(evaluated_split="test")
        self.assertEqual(payload["initialization_random_seed"], 7)
        self.assertEqual(payload["initialization_random_std_percent"], 0.05)
        self.assertEqual(payload["initialization_random_clip_percent"], 0.15)
        self.assertFalse(payload["uses_test_target"])


class SplitNameStrictnessTests(unittest.TestCase):
    """`leakage_flags` must not guess which split it is looking at."""

    def test_oracle_flags_follow_the_named_split(self) -> None:
        spec = InitializationSpec(mode=MODE_ORACLE_MEAN)
        expected = {
            "train": "uses_train_target",
            "validation": "uses_validation_target",
            "test": "uses_test_target",
        }
        for split, flag in expected.items():
            with self.subTest(split=split):
                flags = spec.leakage_flags(evaluated_split=split)
                self.assertTrue(flags[flag])
                self.assertTrue(flags["reads_sample_target"])
                for other in set(expected.values()) - {flag}:
                    self.assertFalse(flags[other])

    def test_val_is_accepted_as_validation(self) -> None:
        spec = InitializationSpec(mode=MODE_ORACLE_MEAN)
        self.assertEqual(
            spec.leakage_flags(evaluated_split="val"),
            spec.leakage_flags(evaluated_split="validation"),
        )

    def test_unknown_split_raises_instead_of_inferring(self) -> None:
        spec = InitializationSpec(mode=MODE_ORACLE_MEAN)
        for bad in ("holdout", "", "test_set", "eval"):
            with self.subTest(split=bad):
                with self.assertRaises(ValueError):
                    spec.leakage_flags(evaluated_split=bad)

    def test_realistic_modes_never_flag_validation_or_test(self) -> None:
        spec = InitializationSpec(mode=MODE_TRAIN_GLOBAL_CONSTANT, train_constant=4.0)
        for split in ("train", "validation", "test"):
            with self.subTest(split=split):
                flags = spec.leakage_flags(evaluated_split=split)
                self.assertTrue(flags["uses_train_target"])
                self.assertFalse(flags["uses_validation_target"])
                self.assertFalse(flags["uses_test_target"])
                self.assertFalse(flags["reads_sample_target"])

    def test_unknown_split_raises_even_for_realistic_modes(self) -> None:
        spec = InitializationSpec(mode=MODE_TRAIN_GLOBAL_CONSTANT, train_constant=4.0)
        with self.assertRaises(ValueError):
            spec.leakage_flags(evaluated_split="everything_else")


class InitializationLogLineTests(unittest.TestCase):
    """The `initial_model=` log line must describe the mode actually in use.

    It used to be hard-coded to the oracle description, so every
    train_global_constant run was mislabelled in train.log.
    """

    def test_oracle_line_says_oracle(self) -> None:
        import train

        line = train.initialization_log_line(MODE_ORACLE_MEAN, None)
        self.assertIn("initial_model=oracle_mean", line)
        self.assertIn("oracle-informed", line)

    def test_realistic_line_names_the_mode_and_constant(self) -> None:
        import train

        spec = InitializationSpec(mode=MODE_TRAIN_GLOBAL_CONSTANT, train_constant=4.0462517)
        line = train.initialization_log_line(MODE_TRAIN_GLOBAL_CONSTANT, spec)
        self.assertIn("initial_model=train_global_constant", line)
        self.assertIn("c_train=4.046252", line)
        self.assertNotIn("sample_spatial_mean_constant", line)

    def test_missing_spec_for_realistic_mode_is_rejected(self) -> None:
        import train

        with self.assertRaises(ValueError):
            train.initialization_log_line(MODE_TRAIN_GLOBAL_CONSTANT, None)

    def test_mode_spec_disagreement_is_rejected(self) -> None:
        import train

        spec = InitializationSpec(mode=MODE_TRAIN_GLOBAL_CONSTANT, train_constant=4.0)
        with self.assertRaises(ValueError):
            train.initialization_log_line(MODE_BIASED_GLOBAL_CONSTANT, spec)


class DeadInitialModelFieldGuardTests(unittest.TestCase):
    """The dead legacy `initial_model` key must never reach a new run's artifacts.

    `main()` used to assign it unconditionally as the oracle description, so every
    realistic run's config.json and checkpoint advertised themselves as oracle-
    initialised. It has no reader, so it changed nothing numerically — but a post-fix
    run carrying it is a provenance regression and must be refused, while resuming one
    of the three documented pre-fix seed-42 runs must stay possible.
    """

    def test_parsed_arguments_carrying_the_key_are_refused(self) -> None:
        import train

        with self.assertRaises(RuntimeError) as caught:
            train.assert_no_legacy_initial_model_field(
                {"initial_model": "sample_spatial_mean_constant", "epochs": 100}
            )
        self.assertIn("refusing to run", str(caught.exception))

    def test_clean_parsed_arguments_are_accepted(self) -> None:
        import train

        train.assert_no_legacy_initial_model_field(
            {"initial_model_mode": MODE_TRAIN_GLOBAL_CONSTANT, "epochs": 100}
        )

    def test_resuming_the_documented_seed_42_value_warns_without_failing(self) -> None:
        import train

        # Must not raise: resuming our own pre-fix seed-42 run is legitimate.
        train.warn_on_legacy_field_in_resumed_checkpoint(
            {"initial_model": "sample_spatial_mean_constant"}, "last.pt"
        )

    def test_resuming_an_undocumented_value_is_refused(self) -> None:
        import train

        with self.assertRaises(RuntimeError):
            train.warn_on_legacy_field_in_resumed_checkpoint(
                {"initial_model": "per_sample_spatial_mean"}, "last.pt"
            )

    def test_resuming_a_clean_checkpoint_is_silent(self) -> None:
        import train

        train.warn_on_legacy_field_in_resumed_checkpoint({}, "last.pt")


class TinyTrainingFixture(unittest.TestCase):
    """A tiny synthetic dataset plus helpers to run train.py end to end."""

    NZ, NX, NT, NSHOT, PML, DX = 16, 24, 16, 4, 4, 0.05

    # Big enough that the U-Net's three downsampling levels survive (z >= 8).
    NZ, NX, NT, NSHOT, PML, DX = 16, 24, 16, 4, 4, 0.05

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = Path(tempfile.mkdtemp(prefix="rtm-init-e2e-"))
        cls.dataset_dir = cls._tmp / "data"
        cls.dataset_dir.mkdir(parents=True)
        for index in range(12):
            cls._write_sample(index, mean=3.5 + 0.05 * index)

    @classmethod
    def _write_sample(cls, index: int, *, mean: float) -> None:
        sample_dir = cls.dataset_dir / f"sample_{index:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(index)
        permittivity = mean + 0.4 * torch.rand((cls.NZ, cls.NX), generator=generator) - 0.2
        torch.save(
            {
                "permittivity": permittivity,
                "bscan_processed": torch.zeros((cls.NSHOT, 1, cls.NT)),
                "bscan_raw": torch.zeros((cls.NSHOT, 1, cls.NT)),
            },
            sample_dir / "data.pt",
        )
        (sample_dir / "meta.json").write_text(
            json.dumps(
                {
                    "sample_id": f"{index:04d}",
                    "dx": cls.DX,
                    "dz": cls.DX,
                    "dt": 0.0001,
                    "freq_mhz": 400.0,
                    "nt": cls.NT,
                    "nshot": cls.NSHOT,
                    "length_m": (cls.NX - 1) * cls.DX,
                    "depth_m": (cls.NZ - 1) * cls.DX,
                    "src_x_start": 0.0,
                    "src_x_step": cls.DX,
                    "src_z": cls.DX,
                    "rec_offset_x": cls.DX,
                    "rec_z": cls.DX,
                    "pml_width": cls.PML,
                    "accuracy": 4,
                }
            )
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _train(self, command: list[str]) -> None:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900
        )
        if completed.returncode != 0:
            self.fail(
                f"training failed:\n{completed.stdout[-3000:]}\n{completed.stderr[-3000:]}"
            )


class TrainingProvenanceEndToEndTests(TinyTrainingFixture):
    """config.json, the checkpoint args and train.log must agree on the mode."""

    def _run_training(self, mode: str, run_name: str) -> Path:
        run_dir = self._tmp / run_name
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--data-dir",
            str(self.dataset_dir),
            "--run-dir",
            str(run_dir),
            "--model-input-mode",
            "m0_rtm",
            "--num-stages",
            "1",
            "--epochs",
            "1",
            "--warmup-epochs",
            "0",
            "--batch-size",
            "2",
            "--initial-model-mode",
            mode,
            "--early-stopping-patience",
            "2",
            "--data-loss-weight",
            "0",
            "--device",
            "cpu",
        ]
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=900
        )
        if completed.returncode != 0:
            self.fail(
                f"training failed for {mode}:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}"
            )
        return run_dir

    def _assert_artifacts_agree(self, run_dir: Path, mode: str) -> None:
        config = json.loads((run_dir / "config.json").read_text())
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        log = (run_dir / "train.log").read_text()

        self.assertEqual(config["initial_model_mode"], mode)
        self.assertEqual(checkpoint["args"]["initial_model_mode"], mode)

        # A dead legacy key used to be written as the oracle description regardless of
        # the resolved mode, so every realistic run's config.json and checkpoint carried
        # an oracle label. It had no reader, but it contradicted the run's own
        # provenance, so nothing may reintroduce it.
        self.assertNotIn("initial_model", config)
        self.assertNotIn("initial_model", checkpoint["args"])

        model_lines = [line for line in log.splitlines() if line.startswith("initial_model=")]
        self.assertEqual(len(model_lines), 1, model_lines)
        self.assertIn(f"initial_model={mode}", model_lines[0])
        if mode == MODE_ORACLE_MEAN:
            self.assertNotIn("c_train=", model_lines[0])
        else:
            self.assertIn("c_train=", model_lines[0])
            self.assertNotIn("sample_spatial_mean_constant", model_lines[0])

    def test_oracle_mean_run_is_self_consistent(self) -> None:
        run_dir = self._run_training(MODE_ORACLE_MEAN, "oracle")
        self._assert_artifacts_agree(run_dir, MODE_ORACLE_MEAN)
        log = (run_dir / "train.log").read_text()
        self.assertNotIn("initial_model=train_global_constant", log)

    def test_train_global_constant_run_is_self_consistent(self) -> None:
        run_dir = self._run_training(MODE_TRAIN_GLOBAL_CONSTANT, "global_constant")
        self._assert_artifacts_agree(run_dir, MODE_TRAIN_GLOBAL_CONSTANT)
        log = (run_dir / "train.log").read_text()
        # The oracle description must not leak into a realistic run.
        self.assertNotIn("sample_spatial_mean_constant", log)
        self.assertIn("uses_validation_target=False", log)
        self.assertIn("uses_test_target=False", log)


class TrainResumeContinuityTests(TinyTrainingFixture):
    """`--resume` must continue the absolute epoch, LR schedule and patience count."""

    def _base_command(self, run_dir: Path, resume: Path | None) -> list[str]:
        command = [
            sys.executable,
            str(PROJECT_ROOT / "train.py"),
            "--data-dir", str(self.dataset_dir),
            "--run-dir", str(run_dir),
            "--model-input-mode", "m0_rtm",
            "--num-stages", "1",
            "--epochs", "2",
            "--warmup-epochs", "0",
            "--batch-size", "2",
            "--initial-model-mode", MODE_TRAIN_GLOBAL_CONSTANT,
            "--early-stopping-patience", "5",
            "--data-loss-weight", "0",
            "--device", "cpu",
        ]
        if resume is not None:
            command.extend(["--resume", str(resume)])
        return command

    def test_resume_continues_the_absolute_epoch_and_scheduler(self) -> None:
        run_dir = self._tmp / "resume_run"
        # Fresh run capped at one epoch.
        self._train(self._first_epoch_command(run_dir))
        first = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(first["epoch"], 1)
        self.assertEqual(first["scheduler_state_dict"]["last_epoch"], 1)

        # Resume and run the second epoch.
        self._train(self._base_command(run_dir, run_dir / "last.pt"))
        second = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(second["epoch"], 2)
        self.assertEqual(second["scheduler_state_dict"]["last_epoch"], 2)

        log = (run_dir / "train.log").read_text()
        self.assertIn("resumed_from=", log)
        self.assertIn("start_epoch=001", log)
        # Exactly one epoch-1 row and one epoch-2 row: the resume continued rather
        # than restarting, and the log was appended to, not overwritten.
        with (run_dir / "loss_log.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["epoch"] for row in rows], ["1", "2"])
        self.assertIn("uses_validation_target=False", log)
        self.assertNotIn("sample_spatial_mean_constant", log)

    def _first_epoch_command(self, run_dir: Path) -> list[str]:
        command = self._base_command(run_dir, None)
        index = command.index("--epochs") + 1
        command[index] = "1"
        return command


if __name__ == "__main__":
    unittest.main()
