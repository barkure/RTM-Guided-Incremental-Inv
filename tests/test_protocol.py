"""Protocol self-checks for the frozen evaluation and split-identity contract.

Run with::

    pixi run python -m unittest discover -s tests -v

These tests deliberately avoid Deepwave and GPU work so they stay fast; they only
exercise the metric protocol, the seed resolution, the split reconstruction and
the identity checks that every analysis script now shares.
"""

from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rtm_inv.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: E402
from rtm_inv.protocol import (  # noqa: E402
    EVALUATION_ARTIFACT_SCHEMA_VERSION,
    FULL_TEST_MODE,
    MAX_SAMPLES_MODE,
    METRIC_PROTOCOL_VERSION,
    MODEL_MAX_DEFAULT,
    MODEL_MIN_DEFAULT,
    SINGLE_SAMPLE_MODE,
    SPLIT_IDENTITY_VERSION,
    MetricProtocol,
    SplitSpec,
    TestSplitReference,
    aggregate_metric_rows,
    argument_snapshot,
    assert_matching_test_identity,
    assert_same_sample_sequence,
    build_evaluation_coverage,
    build_split_identity,
    check_evaluation_coverage,
    compute_metric_row,
    hash_values,
    infer_evaluation_mode,
    metric_fieldnames,
    parameter_counts,
    parse_optional_bool,
    reconstruct_split_indices,
    resolve_seeds,
    set_global_seed,
    split_spec_from_saved_args,
    test_split_reference_from_identity,
)

PROTOCOL = MetricProtocol()


def _sample_pair(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    target = torch.rand((32, 48), generator=generator) * 0.5 + 0.2
    noise = torch.randn((32, 48), generator=generator) * 0.03
    return target + noise, target


class MetricScaleTests(unittest.TestCase):
    def test_physical_mae_is_eight_times_normalized(self) -> None:
        pred, target = _sample_pair()
        normalized = PROTOCOL.normalized_scores(pred, target)
        physical = PROTOCOL.physical_error_scores(pred, target)
        self.assertAlmostEqual(
            physical["mae"] / normalized["mae"],
            PROTOCOL.mae_normalized_to_physical,
            places=5,
        )
        self.assertAlmostEqual(
            physical["mse"] / normalized["mse"],
            PROTOCOL.mse_normalized_to_physical,
            places=5,
        )
        self.assertEqual(PROTOCOL.mae_normalized_to_physical, 8.0)
        self.assertEqual(PROTOCOL.mse_normalized_to_physical, 64.0)

    def test_physical_scores_do_not_duplicate_ssim_or_psnr(self) -> None:
        self.assertEqual(set(PROTOCOL.physical_error_scores(*_sample_pair())), {"mae", "mse"})

    def test_psnr_is_invariant_under_the_affine_permittivity_map(self) -> None:
        pred, target = _sample_pair()
        normalized = PROTOCOL.normalized_scores(pred, target)
        physical = PROTOCOL.denormalize(pred), PROTOCOL.denormalize(target)
        self.assertAlmostEqual(
            peak_signal_noise_ratio(*physical, data_range=PROTOCOL.physical_data_range),
            normalized["psnr"],
            places=6,
        )

    def test_ssim_is_invariant_under_pure_scaling_only(self) -> None:
        """SSIM survives a pure rescale but not the offset affine permittivity map.

        This is why the protocol reports SSIM once, on the normalized scale.
        """
        pred, target = _sample_pair()
        scaled = structural_similarity(
            pred * 8.0, target * 8.0, data_range=8.0
        )
        self.assertAlmostEqual(
            scaled,
            structural_similarity(pred, target, data_range=1.0),
            places=6,
        )
        offset = structural_similarity(
            PROTOCOL.denormalize(pred),
            PROTOCOL.denormalize(target),
            data_range=PROTOCOL.physical_data_range,
        )
        self.assertNotAlmostEqual(
            offset,
            structural_similarity(pred, target, data_range=1.0),
            places=6,
        )

    def test_ssim_uses_eleven_pixel_gaussian_window(self) -> None:
        from rtm_inv.metrics import SSIM_SIGMA, SSIM_WINDOW_SIZE

        self.assertEqual(SSIM_WINDOW_SIZE, 11)
        self.assertEqual(SSIM_SIGMA, 1.5)


class MetricRowTests(unittest.TestCase):
    def test_row_keys_match_declared_fieldnames(self) -> None:
        pred, target = _sample_pair()
        row = compute_metric_row("final", pred, target, PROTOCOL)
        self.assertEqual(
            sorted(row),
            sorted(name for name in metric_fieldnames(PROTOCOL) if name.startswith("final_")),
        )
        self.assertNotIn("final_er_ssim", row)
        self.assertNotIn("final_er_psnr", row)
        self.assertIn("final_er_mae", row)
        self.assertIn("final_er_mse", row)

    def test_row_values_match_the_metric_functions(self) -> None:
        # Rows are formatted to 7 significant digits, so compare to 5 decimals.
        pred, target = _sample_pair()
        row = compute_metric_row("final", pred, target, PROTOCOL)
        self.assertAlmostEqual(
            float(row["final_ssim"]),
            structural_similarity(pred, target, data_range=1.0),
            places=5,
        )
        self.assertAlmostEqual(
            float(row["final_psnr"]),
            peak_signal_noise_ratio(pred, target, data_range=1.0),
            places=5,
        )

    def test_aggregation_is_per_sample_then_mean(self) -> None:
        rows = [
            {"sample_id": "0001", "final_mae": "0.1", "final_mse": "0.01"},
            {"sample_id": "0002", "final_mae": "0.3", "final_mse": "0.05"},
        ]
        aggregated = aggregate_metric_rows(rows, ["sample_id", "final_mae", "final_mse"])
        self.assertEqual(set(aggregated), {"final_mae", "final_mse"})
        self.assertAlmostEqual(aggregated["final_mae"], 0.2, places=12)
        self.assertAlmostEqual(aggregated["final_mse"], 0.03, places=12)

    def test_aggregation_uses_per_sample_values_not_a_pooled_error(self) -> None:
        # A pooled MSE over concatenated pixels would give 0.05 for this input;
        # the protocol requires the mean of the two per-sample MSEs.
        rows = [
            {"sample_id": "0001", "final_mse": "0.0"},
            {"sample_id": "0002", "final_mse": "0.1"},
        ]
        aggregated = aggregate_metric_rows(rows, ["final_mse"])
        self.assertAlmostEqual(aggregated["final_mse"], 0.05, places=12)


class SplitIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_ids = [f"{index:04d}" for index in range(1000)]
        self.spec = SplitSpec(
            dataset_size=len(self.sample_ids),
            train_split=0.8,
            val_split=0.1,
            split_seed=42,
        )

    def test_reconstruction_matches_random_split(self) -> None:
        for size, train_split, val_split, seed in (
            (1000, 0.8, 0.1, 42),
            (137, 0.7, 0.2, 7),
            (53, 0.6, 0.15, 20260713),
        ):
            spec = SplitSpec(size, train_split, val_split, seed)
            expected = random_split(
                range(size),
                [spec.train_size, spec.val_size, spec.test_size],
                generator=torch.Generator().manual_seed(seed),
            )
            train_ids, val_ids, test_ids = reconstruct_split_indices(spec)
            self.assertEqual(train_ids, list(expected[0].indices))
            self.assertEqual(val_ids, list(expected[1].indices))
            self.assertEqual(test_ids, list(expected[2].indices))

    def test_split_sizes_are_unchanged(self) -> None:
        self.assertEqual(self.spec.sizes, (800, 100, 100))

    def test_hashes_are_stable_across_repeated_calls(self) -> None:
        first = build_split_identity(self.spec, self.sample_ids)
        second = build_split_identity(self.spec, self.sample_ids)
        self.assertEqual(first.split_sha256, second.split_sha256)
        self.assertEqual(first.test_sample_ids_sha256, second.test_sample_ids_sha256)
        self.assertEqual(first.test_sample_ids, second.test_sample_ids)

    def test_hash_changes_with_split_seed(self) -> None:
        other = build_split_identity(
            SplitSpec(self.spec.dataset_size, 0.8, 0.1, 43), self.sample_ids
        )
        reference = build_split_identity(self.spec, self.sample_ids)
        self.assertNotEqual(reference.split_sha256, other.split_sha256)
        self.assertNotEqual(reference.test_sample_ids_sha256, other.test_sample_ids_sha256)

    def test_hash_detects_a_renamed_sample(self) -> None:
        renamed = list(self.sample_ids)
        reference = build_split_identity(self.spec, self.sample_ids)
        renamed[reference.test_indices[0]] = "9999"
        changed = build_split_identity(self.spec, renamed)
        self.assertNotEqual(reference.test_sample_ids_sha256, changed.test_sample_ids_sha256)

    def test_hash_values_is_order_sensitive(self) -> None:
        self.assertNotEqual(hash_values(["a", "b"]), hash_values(["b", "a"]))


class SampleIdentityGuardTests(unittest.TestCase):
    def test_identical_sequences_pass(self) -> None:
        assert_same_sample_sequence("a", ["1", "2"], "b", ["1", "2"])

    def test_reordered_sequences_are_rejected(self) -> None:
        with self.assertRaises(ValueError) as context:
            assert_same_sample_sequence("a", ["2", "1"], "b", ["1", "2"])
        self.assertIn("different test sample sequence", str(context.exception))

    def test_different_sample_sets_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_same_sample_sequence("a", ["1", "3"], "b", ["1", "2"])

    def test_benchmark_style_identity_mismatch_is_rejected(self) -> None:
        spec = SplitSpec(100, 0.8, 0.1, 42)
        reference = build_split_identity(spec, [f"{index:04d}" for index in range(100)])
        shifted = build_split_identity(spec, [f"{index:04d}" for index in range(1, 101)])
        self.assertNotEqual(reference.split_sha256, shifted.split_sha256)
        with self.assertRaises(ValueError):
            assert_matching_test_identity([("reference", reference), ("shifted", shifted)])

    def test_matching_identities_return_the_reference(self) -> None:
        spec = SplitSpec(100, 0.8, 0.1, 42)
        ids = [f"{index:04d}" for index in range(100)]
        first = build_split_identity(spec, ids)
        second = build_split_identity(spec, list(ids))
        self.assertEqual(
            assert_matching_test_identity([("a", first), ("b", second)]), first
        )


class SeedFallbackTests(unittest.TestCase):
    def test_legacy_seed_fills_every_purpose(self) -> None:
        seeds = resolve_seeds(legacy_seed=42)
        self.assertEqual(
            (seeds.split_seed, seeds.training_seed, seeds.sampler_seed), (42, 42, 42)
        )
        self.assertEqual(len(seeds.fallback_notes), 3)
        self.assertIn("split_seed not supplied, reusing legacy seed=42", seeds.fallback_notes[0])

    def test_explicit_seeds_win_and_record_no_fallback(self) -> None:
        seeds = resolve_seeds(
            legacy_seed=42, split_seed=1, training_seed=2, sampler_seed=3
        )
        self.assertEqual(
            (seeds.split_seed, seeds.training_seed, seeds.sampler_seed), (1, 2, 3)
        )
        self.assertEqual(seeds.fallback_notes, ())

    def test_partial_override_logs_only_the_missing_purposes(self) -> None:
        seeds = resolve_seeds(legacy_seed=42, training_seed=7)
        self.assertEqual(seeds.split_seed, 42)
        self.assertEqual(seeds.training_seed, 7)
        self.assertEqual(len(seeds.fallback_notes), 2)

    def test_legacy_checkpoint_args_fall_back_to_seed(self) -> None:
        legacy_args = SimpleNamespace(seed=42, train_split=0.8, val_split=0.1)
        spec, notes = split_spec_from_saved_args(legacy_args, dataset_size=1000)
        self.assertEqual(spec.split_seed, 42)
        self.assertEqual(spec.sizes, (800, 100, 100))
        self.assertTrue(any("reusing legacy seed=42" in note for note in notes))

    def test_new_checkpoint_args_use_split_seed(self) -> None:
        new_args = SimpleNamespace(
            seed=42, split_seed=123, training_seed=7, sampler_seed=9,
            train_split=0.8, val_split=0.1,
        )
        spec, notes = split_spec_from_saved_args(new_args, dataset_size=1000)
        self.assertEqual(spec.split_seed, 123)
        self.assertEqual(notes, ())

    def test_legacy_and_new_args_agree_when_split_seed_equals_seed(self) -> None:
        legacy = SimpleNamespace(seed=42, train_split=0.8, val_split=0.1)
        new = SimpleNamespace(
            seed=42, split_seed=42, train_split=0.8, val_split=0.1
        )
        legacy_spec, _ = split_spec_from_saved_args(legacy, 1000)
        new_spec, _ = split_spec_from_saved_args(new, 1000)
        self.assertEqual(legacy_spec, new_spec)

    def test_set_global_seed_reports_configuration(self) -> None:
        info = set_global_seed(11)
        self.assertEqual(info["seed"], 11)
        self.assertTrue(info["torch_seeded"])
        self.assertIn("cudnn_deterministic", info)
        first = torch.rand(3)
        set_global_seed(11)
        self.assertTrue(torch.equal(first, torch.rand(3)))


class ParameterCountTests(unittest.TestCase):
    def test_only_trainable_parameters_are_counted_as_parameters(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3), torch.nn.Linear(3, 2, bias=False)
        )
        # Linear(4, 3) = 12 weights + 3 biases = 15 trainable.
        for parameter in model[1].parameters():
            parameter.requires_grad_(False)
        counts = parameter_counts(model)
        self.assertEqual(counts["trainable_parameters"], 15)
        self.assertEqual(counts["total_parameters"], 21)
        self.assertEqual(counts["parameters"], counts["trainable_parameters"])

    def test_all_parameters_counted_when_nothing_is_frozen(self) -> None:
        model = torch.nn.Linear(4, 3)
        counts = parameter_counts(model)
        self.assertEqual(counts["trainable_parameters"], 15)
        self.assertEqual(counts["total_parameters"], 15)


class ProtocolMetadataTests(unittest.TestCase):
    def test_metadata_records_both_scales_explicitly(self) -> None:
        metadata = PROTOCOL.as_dict()
        self.assertEqual(metadata["canonical_metric_scale"], "normalized_model_unit_interval")
        self.assertEqual(metadata["physical_metric_scale"], "physical_relative_permittivity")
        self.assertEqual(metadata["mae_normalized_to_physical_factor"], 8.0)
        self.assertEqual(metadata["mse_normalized_to_physical_factor"], 64.0)
        self.assertEqual(metadata["evaluation_region"], "full_model_grid")
        self.assertEqual(metadata["aggregation"], "per_sample_then_mean")
        self.assertFalse(metadata["ssim_psnr_duplicated_on_physical_scale"])

    def test_defaults_match_the_network_bounds(self) -> None:
        self.assertEqual((MODEL_MIN_DEFAULT, MODEL_MAX_DEFAULT), (2.0, 10.0))

    def test_invalid_bounds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MetricProtocol(model_min=10.0, model_max=2.0)


class EvaluateScriptCompatibilityTests(unittest.TestCase):
    """The evaluation entry point must keep accepting legacy checkpoint args."""

    def test_split_subset_accepts_legacy_args(self) -> None:
        import evaluate

        class _FakeSample:
            def __init__(self, sample_id: str) -> None:
                self.sample_id = sample_id

        class _FakeDataset:
            def __init__(self, count: int) -> None:
                self.samples = [_FakeSample(f"{index:04d}") for index in range(count)]

            def __len__(self) -> int:
                return len(self.samples)

        dataset = _FakeDataset(1000)
        legacy_args = SimpleNamespace(seed=42, train_split=0.8, val_split=0.1)
        subset, sizes, split_seed = evaluate.split_subset(dataset, legacy_args)
        self.assertEqual(sizes, (800, 100, 100))
        self.assertEqual(split_seed, 42)
        self.assertEqual(len(subset.indices), 100)

        identity, notes = evaluate.split_identity(dataset, legacy_args)
        self.assertTrue(any("reusing legacy seed=42" in note for note in notes))
        self.assertEqual(identity.test_sample_ids, identity.test_sample_ids)
        self.assertEqual(len(identity.test_sample_ids_sha256), 64)

    def test_split_seed_override_is_honoured_and_logged(self) -> None:
        import evaluate

        class _FakeSample:
            def __init__(self, sample_id: str) -> None:
                self.sample_id = sample_id

        class _FakeDataset:
            def __init__(self, count: int) -> None:
                self.samples = [_FakeSample(f"{index:04d}") for index in range(count)]

            def __len__(self) -> int:
                return len(self.samples)

        dataset = _FakeDataset(1000)
        legacy_args = SimpleNamespace(seed=42, train_split=0.8, val_split=0.1)
        identity, notes = evaluate.split_identity(
            dataset, legacy_args, split_seed_override=7
        )
        self.assertEqual(identity.spec.split_seed, 7)
        self.assertTrue(any("overridden" in note for note in notes))

    def test_metric_fieldnames_have_no_duplicate_ssim_or_psnr_scales(self) -> None:
        import evaluate

        names = evaluate.metric_fieldnames()
        self.assertEqual(len(names), len(set(names)))
        self.assertNotIn("final_er_ssim", names)
        self.assertNotIn("final_er_psnr", names)
        self.assertIn("final_er_mae", names)


class _OrderDataset(torch.utils.data.Dataset):
    """Minimal dataset exposing the shape key the batch sampler buckets on."""

    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"value": torch.tensor([float(index)])}

    def sample_shape_key(self, index: int) -> tuple[int, ...]:
        return (1,)


def _first_epoch_order(
    size: int, *, batch_size: int, sampler_seed: int, epoch: int = 0
) -> list[float]:
    import train

    dataset = _OrderDataset(size)
    loader = train.make_shape_loader(
        dataset, list(range(size)), batch_size, True, sampler_seed
    )
    train.set_loader_epoch(loader, epoch)
    order: list[float] = []
    for batch in loader:
        order.extend(batch["value"].flatten().tolist())
    return order


class SamplerSeedTests(unittest.TestCase):
    """The sampler seed, not the training seed, must determine the batch order.

    Regression guard: `make_shape_loader` used to special-case `batch_size <= 1` as
    `DataLoader(subset, batch_size=1, shuffle=True)` with no generator, which drew
    from the global torch RNG -- and every paper checkpoint used `batch_size=1`.
    """

    def setUp(self) -> None:
        original_state = {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }

        def restore() -> None:
            if original_state["deterministic"]:
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = original_state["cudnn_deterministic"]
            torch.backends.cudnn.benchmark = original_state["cudnn_benchmark"]

        self.addCleanup(restore)

    def test_same_sampler_seed_gives_same_first_epoch_order(self) -> None:
        first = _first_epoch_order(64, batch_size=1, sampler_seed=11)
        second = _first_epoch_order(64, batch_size=1, sampler_seed=11)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_different_sampler_seed_gives_different_first_epoch_order(self) -> None:
        first = _first_epoch_order(64, batch_size=1, sampler_seed=11)
        other = _first_epoch_order(64, batch_size=1, sampler_seed=12)
        self.assertNotEqual(first, other)

    def test_training_seed_does_not_change_the_order(self) -> None:
        set_global_seed(1)
        first = _first_epoch_order(64, batch_size=1, sampler_seed=11)
        first_random_draw = torch.rand(4)

        set_global_seed(999)
        second = _first_epoch_order(64, batch_size=1, sampler_seed=11)
        second_random_draw = torch.rand(4)

        self.assertEqual(first, second)
        # The global stream did move: this is the coupling that used to leak in.
        self.assertFalse(torch.equal(first_random_draw, second_random_draw))

    def test_batch_size_one_and_larger_batches_both_follow_the_seed(self) -> None:
        for batch_size in (1, 2, 8):
            with self.subTest(batch_size=batch_size):
                first = _first_epoch_order(32, batch_size=batch_size, sampler_seed=5)
                second = _first_epoch_order(32, batch_size=batch_size, sampler_seed=5)
                other = _first_epoch_order(32, batch_size=batch_size, sampler_seed=6)
                self.assertEqual(first, second)
                self.assertNotEqual(first, other)

    def test_epoch_pins_the_permutation(self) -> None:
        epoch_five = _first_epoch_order(64, batch_size=1, sampler_seed=11, epoch=5)
        again = _first_epoch_order(64, batch_size=1, sampler_seed=11, epoch=5)
        epoch_six = _first_epoch_order(64, batch_size=1, sampler_seed=11, epoch=6)
        self.assertEqual(epoch_five, again)
        self.assertNotEqual(epoch_five, epoch_six)

    def test_epoch_matches_an_uninterrupted_run_after_resume(self) -> None:
        """A run resumed at epoch k emits the same permutation at epoch k."""
        import train

        dataset = _OrderDataset(48)
        loader = train.make_shape_loader(dataset, list(range(48)), 1, True, 11)
        uninterrupted: dict[int, list[float]] = {}
        for epoch in range(1, 5):
            train.set_loader_epoch(loader, epoch)
            uninterrupted[epoch] = [
                float(value) for batch in loader for value in batch["value"].flatten()
            ]

        resumed_loader = train.make_shape_loader(dataset, list(range(48)), 1, True, 11)
        for epoch in (3, 4, 5):
            train.set_loader_epoch(resumed_loader, epoch)
            order = [
                float(value)
                for batch in resumed_loader
                for value in batch["value"].flatten()
            ]
            if epoch in uninterrupted:
                self.assertEqual(order, uninterrupted[epoch])
        resumed_loader = None


class GlobalSeedStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }

        def restore() -> None:
            if self.original["deterministic"]:
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.use_deterministic_algorithms(False)
            torch.backends.cudnn.deterministic = self.original["cudnn_deterministic"]
            torch.backends.cudnn.benchmark = self.original["cudnn_benchmark"]

        self.addCleanup(restore)

    def test_deterministic_true_sets_every_flag(self) -> None:
        info = set_global_seed(3, deterministic=True)
        self.assertTrue(info["torch_deterministic_algorithms"])
        self.assertTrue(info["cudnn_deterministic"])
        self.assertFalse(info["cudnn_benchmark"])

    def test_deterministic_false_clears_previously_enabled_state(self) -> None:
        set_global_seed(3, deterministic=True)
        info = set_global_seed(3, deterministic=False)
        self.assertFalse(info["torch_deterministic_algorithms"])
        self.assertFalse(info["cudnn_deterministic"])
        self.assertFalse(info["cudnn_benchmark"])
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.deterministic)

    def test_requested_flag_is_recorded(self) -> None:
        self.assertTrue(
            set_global_seed(3, deterministic=True)["deterministic_requested"]
        )
        self.assertFalse(
            set_global_seed(3, deterministic=False)["deterministic_requested"]
        )


class EvaluationCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        # 100 samples at 0.6/0.2 gives a 20-sample test split, large enough that a
        # truncated subset is unambiguously partial.
        self.sample_ids = [f"{index:04d}" for index in range(100)]
        self.identity = build_split_identity(
            SplitSpec(len(self.sample_ids), 0.6, 0.2, 42), self.sample_ids
        )
        self.assertEqual(self.identity.test_size, 20)
        self.reference = test_split_reference_from_identity(self.identity)

    def test_full_coverage_matches_the_split_hash(self) -> None:
        coverage = build_evaluation_coverage(
            self.identity, self.identity.test_sample_ids
        )
        self.assertTrue(coverage.is_full_test_evaluation)
        self.assertEqual(coverage.evaluation_mode, FULL_TEST_MODE)
        self.assertEqual(coverage.test_split_size, self.identity.test_size)
        self.assertEqual(
            coverage.evaluated_sample_ids_sha256, self.identity.test_sample_ids_sha256
        )
        self.assertEqual(coverage.evaluated_samples, self.identity.test_size)

    def test_max_samples_coverage_is_not_full(self) -> None:
        coverage = build_evaluation_coverage(
            self.identity, self.identity.test_sample_ids[:2], max_samples=2
        )
        self.assertFalse(coverage.is_full_test_evaluation)
        self.assertEqual(coverage.evaluation_mode, MAX_SAMPLES_MODE)
        self.assertEqual(coverage.evaluated_samples, 2)
        self.assertNotEqual(
            coverage.evaluated_sample_ids_sha256, self.identity.test_sample_ids_sha256
        )
        # The full split is still recorded alongside the subset.
        self.assertEqual(coverage.test_split_size, self.identity.test_size)

    def test_single_sample_coverage_is_not_full(self) -> None:
        coverage = build_evaluation_coverage(
            self.identity, [self.identity.test_sample_ids[3]]
        )
        self.assertFalse(coverage.is_full_test_evaluation)
        self.assertEqual(coverage.evaluation_mode, SINGLE_SAMPLE_MODE)
        self.assertEqual(coverage.evaluated_samples, 1)

    def test_mode_inference_rejects_a_single_sample_outside_the_split(self) -> None:
        self.assertEqual(
            infer_evaluation_mode(["nope"], self.identity.test_sample_ids),
            MAX_SAMPLES_MODE,
        )

    def test_full_coverage_is_verified(self) -> None:
        coverage = build_evaluation_coverage(
            self.identity, self.identity.test_sample_ids
        )
        self.assertTrue(coverage.coverage_verified)
        self.assertEqual(coverage.evaluation_mode, FULL_TEST_MODE)

    def test_max_samples_prefix_is_verified(self) -> None:
        coverage = build_evaluation_coverage(
            self.identity, self.identity.test_sample_ids[:3], max_samples=3
        )
        self.assertTrue(coverage.coverage_verified)
        self.assertEqual(coverage.evaluation_mode, MAX_SAMPLES_MODE)

    def test_max_samples_one_is_recorded_as_max_samples(self) -> None:
        # A one-entry prefix is a max_samples run; it must not be inferred as an
        # explicit single-sample query from its length alone.
        coverage = build_evaluation_coverage(
            self.identity, self.identity.test_sample_ids[:1], max_samples=1
        )
        self.assertEqual(coverage.evaluation_mode, MAX_SAMPLES_MODE)
        self.assertTrue(coverage.coverage_verified)

    def test_max_samples_covering_the_whole_split_is_still_full(self) -> None:
        # The mode records how the run was requested; the covered range is derived
        # from the recorded sequence, so asking for the whole split is a full run.
        coverage = build_evaluation_coverage(
            self.identity,
            self.identity.test_sample_ids,
            max_samples=self.identity.test_size,
        )
        self.assertEqual(coverage.evaluation_mode, MAX_SAMPLES_MODE)
        self.assertTrue(coverage.is_full_test_evaluation)
        self.assertTrue(coverage.coverage_verified)

    def test_non_prefix_max_samples_is_unverified(self) -> None:
        ids = self.identity.test_sample_ids
        coverage = build_evaluation_coverage(
            self.identity, [ids[2], ids[1], ids[0]], max_samples=3
        )
        self.assertFalse(coverage.coverage_verified)

    def test_max_samples_count_mismatch_is_unverified(self) -> None:
        coverage = build_evaluation_coverage(
            self.identity, self.identity.test_sample_ids[:2], max_samples=3
        )
        self.assertFalse(coverage.coverage_verified)

    def test_max_samples_equal_to_the_split_size_is_verified(self) -> None:
        coverage = build_evaluation_coverage(
            self.identity,
            self.identity.test_sample_ids,
            max_samples=self.identity.test_size,
        )
        self.assertEqual(coverage.evaluation_mode, MAX_SAMPLES_MODE)
        self.assertTrue(coverage.is_full_test_evaluation)
        self.assertTrue(coverage.coverage_verified)

    def test_max_samples_above_the_split_size_is_verified(self) -> None:
        # A budget larger than the split selects the whole split; the expected count is
        # capped, so this must not be mistaken for a declaration mismatch.
        coverage = build_evaluation_coverage(
            self.identity,
            self.identity.test_sample_ids,
            max_samples=self.identity.test_size * 2,
        )
        self.assertEqual(coverage.evaluation_mode, MAX_SAMPLES_MODE)
        self.assertEqual(coverage.max_samples, self.identity.test_size * 2)
        self.assertTrue(coverage.is_full_test_evaluation)
        self.assertTrue(coverage.coverage_verified)

    def test_non_positive_max_samples_is_rejected(self) -> None:
        for bad in (0, -1):
            with self.subTest(max_samples=bad):
                with self.assertRaises(ValueError):
                    build_evaluation_coverage(
                        self.identity, self.identity.test_sample_ids, max_samples=bad
                    )

    def test_duplicate_ids_are_unverified(self) -> None:
        ids = self.identity.test_sample_ids
        coverage = build_evaluation_coverage(
            self.identity, [ids[0], ids[0]], max_samples=2
        )
        self.assertFalse(coverage.coverage_verified)

    def test_single_sample_from_the_middle_is_verified(self) -> None:
        # An explicit --sample-id may select any test position, not only the first.
        middle = self.identity.test_sample_ids[self.identity.test_size // 2]
        coverage = build_evaluation_coverage(self.identity, [middle])
        self.assertEqual(coverage.evaluation_mode, SINGLE_SAMPLE_MODE)
        self.assertFalse(coverage.is_full_test_evaluation)
        self.assertTrue(coverage.coverage_verified)

    def test_single_sample_outside_the_split_is_unverified(self) -> None:
        coverage = build_evaluation_coverage(self.identity, ["nope"])
        self.assertFalse(coverage.coverage_verified)

    def test_artifact_schema_version_is_separate_from_the_metric_version(self) -> None:
        payload = PROTOCOL.as_dict()
        self.assertEqual(
            payload["evaluation_artifact_schema_version"],
            EVALUATION_ARTIFACT_SCHEMA_VERSION,
        )
        self.assertEqual(payload["metric_protocol_version"], METRIC_PROTOCOL_VERSION)
        self.assertNotEqual(
            EVALUATION_ARTIFACT_SCHEMA_VERSION, METRIC_PROTOCOL_VERSION
        )


class CoverageValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_ids = [f"{index:04d}" for index in range(100)]
        self.identity = build_split_identity(
            SplitSpec(len(self.sample_ids), 0.6, 0.2, 42), self.sample_ids
        )
        self.assertEqual(self.identity.test_size, 20)
        self.reference = test_split_reference_from_identity(self.identity)
        self.full_ids = list(self.identity.test_sample_ids)

    def _full_summary(self) -> dict[str, object]:
        coverage = build_evaluation_coverage(self.identity, self.full_ids)
        return {**coverage.as_dict(), **self.identity.as_dict()}

    def test_declared_max_samples_mode_is_not_rewritten_as_full_test(self) -> None:
        # A --max-samples run that happens to cover the whole split stays a max_samples
        # run: the mode records the selection, not the resulting sample count.
        coverage = build_evaluation_coverage(
            self.identity, self.full_ids, max_samples=self.identity.test_size
        )
        summary = {**coverage.as_dict(), **self.identity.as_dict()}
        check = check_evaluation_coverage("method", summary, self.full_ids, self.reference)
        self.assertEqual(check.mode, MAX_SAMPLES_MODE)
        self.assertTrue(check.is_full)

    def test_declared_max_samples_mode_above_the_split_size_is_accepted(self) -> None:
        coverage = build_evaluation_coverage(
            self.identity, self.full_ids, max_samples=self.identity.test_size * 3
        )
        summary = {**coverage.as_dict(), **self.identity.as_dict()}
        check = check_evaluation_coverage("method", summary, self.full_ids, self.reference)
        self.assertEqual(check.mode, MAX_SAMPLES_MODE)
        self.assertTrue(check.is_full)

    def test_declared_single_sample_mode_is_preserved(self) -> None:
        middle = self.identity.test_sample_ids[self.identity.test_size // 2]
        coverage = build_evaluation_coverage(self.identity, [middle])
        summary = {**coverage.as_dict(), **self.identity.as_dict()}
        check = check_evaluation_coverage(
            "method", summary, [middle], self.reference, allow_partial=True
        )
        self.assertEqual(check.mode, SINGLE_SAMPLE_MODE)
        self.assertFalse(check.is_full)

    def test_declared_full_test_mode_with_a_partial_sequence_is_rejected(self) -> None:
        summary = self._full_summary()
        summary["evaluation_mode"] = FULL_TEST_MODE
        with self.assertRaises(ValueError):
            check_evaluation_coverage(
                "method", summary, self.full_ids[:-1], self.reference, allow_partial=True
            )

    def test_declared_max_samples_mode_with_a_non_prefix_sequence_is_rejected(self) -> None:
        coverage = build_evaluation_coverage(self.identity, self.full_ids[:3], max_samples=3)
        summary = {**coverage.as_dict(), **self.identity.as_dict()}
        with self.assertRaises(ValueError):
            check_evaluation_coverage(
                "method",
                summary,
                list(reversed(self.full_ids[:3])),
                self.reference,
                allow_partial=True,
            )

    def test_unknown_declared_mode_is_rejected(self) -> None:
        summary = self._full_summary()
        summary["evaluation_mode"] = "made_up_mode"
        with self.assertRaises(ValueError):
            check_evaluation_coverage("method", summary, self.full_ids, self.reference)

    def test_artefact_without_a_declared_mode_falls_back_to_inference(self) -> None:
        coverage = build_evaluation_coverage(self.identity, self.full_ids)
        summary = {**coverage.as_dict(), **self.identity.as_dict()}
        del summary["evaluation_mode"]
        check = check_evaluation_coverage("method", summary, self.full_ids, self.reference)
        self.assertEqual(check.mode, FULL_TEST_MODE)

    def test_full_evaluation_passes(self) -> None:
        check = check_evaluation_coverage(
            "method", self._full_summary(), self.full_ids, self.reference
        )
        self.assertTrue(check.is_full)
        self.assertFalse(check.legacy)

    def test_partial_evaluation_is_rejected_by_default(self) -> None:
        subset = self.full_ids[:3]
        coverage = build_evaluation_coverage(self.identity, subset, max_samples=3)
        with self.assertRaises(ValueError) as context:
            check_evaluation_coverage(
                "method", coverage.as_dict(), subset, self.reference
            )
        self.assertIn("partial evaluation", str(context.exception))
        self.assertIn("--allow-partial-evaluation", str(context.exception))

    def test_partial_evaluation_is_accepted_when_explicitly_allowed(self) -> None:
        subset = self.full_ids[:3]
        coverage = build_evaluation_coverage(self.identity, subset, max_samples=3)
        check = check_evaluation_coverage(
            "method",
            coverage.as_dict(),
            subset,
            self.reference,
            allow_partial=True,
        )
        self.assertFalse(check.is_full)
        self.assertEqual(check.mode, MAX_SAMPLES_MODE)

    def test_declared_full_but_partial_rows_is_rejected(self) -> None:
        coverage = build_evaluation_coverage(self.identity, self.full_ids)
        summary = coverage.as_dict()
        with self.assertRaises(ValueError) as context:
            check_evaluation_coverage(
                "method", summary, self.full_ids[:3], self.reference, allow_partial=True
            )
        self.assertIn("declares is_full_test_evaluation=true", str(context.exception))

    def test_declared_partial_but_full_rows_is_rejected(self) -> None:
        coverage = build_evaluation_coverage(self.identity, self.full_ids)
        summary = coverage.as_dict()
        summary["is_full_test_evaluation"] = False
        with self.assertRaises(ValueError) as context:
            check_evaluation_coverage("method", summary, self.full_ids, self.reference)
        self.assertIn("inconsistent", str(context.exception))

    def test_declared_evaluated_file_must_match_metrics_order(self) -> None:
        coverage = build_evaluation_coverage(self.identity, self.full_ids)
        with self.assertRaises(ValueError) as context:
            check_evaluation_coverage(
                "method",
                coverage.as_dict(),
                self.full_ids,
                self.reference,
                declared_evaluated_sample_ids=list(reversed(self.full_ids)),
            )
        self.assertIn("different test sample sequence", str(context.exception))

    def test_evaluated_hash_mismatch_is_rejected(self) -> None:
        summary = self._full_summary()
        summary["evaluated_sample_ids_sha256"] = hash_values(["bogus"])
        with self.assertRaises(ValueError) as context:
            check_evaluation_coverage("method", summary, self.full_ids, self.reference)
        self.assertIn("evaluated_sample_ids_sha256", str(context.exception))

    def test_test_split_hash_mismatch_is_rejected(self) -> None:
        summary = self._full_summary()
        summary["test_split_sample_ids_sha256"] = hash_values(["bogus"])
        with self.assertRaises(ValueError) as context:
            check_evaluation_coverage("method", summary, self.full_ids, self.reference)
        self.assertIn("test_split_sample_ids_sha256", str(context.exception))

    def test_split_hash_mismatch_is_rejected(self) -> None:
        summary = self._full_summary()
        summary["split_sha256"] = hash_values(["bogus"])
        with self.assertRaises(ValueError) as context:
            check_evaluation_coverage("method", summary, self.full_ids, self.reference)
        self.assertIn("split_sha256", str(context.exception))

    def test_evaluated_samples_count_mismatch_is_rejected(self) -> None:
        summary = self._full_summary()
        summary["evaluated_samples"] = len(self.full_ids) + 1
        with self.assertRaises(ValueError) as context:
            check_evaluation_coverage("method", summary, self.full_ids, self.reference)
        self.assertIn("evaluated_samples", str(context.exception))

    def test_legacy_summary_without_the_flag_is_flagged(self) -> None:
        legacy_summary: dict[str, object] = {"samples": len(self.full_ids)}
        check = check_evaluation_coverage(
            "legacy", legacy_summary, self.full_ids, self.reference
        )
        self.assertTrue(check.legacy)
        self.assertTrue(check.is_full)

    def test_legacy_summary_offering_only_a_count_is_still_validated(self) -> None:
        """A bare sample count must never be accepted as proof of full coverage."""
        legacy_summary: dict[str, object] = {"samples": len(self.full_ids)}
        with self.assertRaises(ValueError):
            check_evaluation_coverage(
                "legacy-truncated", legacy_summary, self.full_ids[:3], self.reference
            )

    def test_parse_optional_bool_handles_csv_and_missing_values(self) -> None:
        self.assertIsNone(parse_optional_bool(""))
        self.assertIsNone(parse_optional_bool(None))
        self.assertTrue(parse_optional_bool("True"))
        self.assertTrue(parse_optional_bool("1"))
        self.assertFalse(parse_optional_bool("False"))
        self.assertFalse(parse_optional_bool("0"))
        with self.assertRaises(ValueError):
            parse_optional_bool("maybe")


class FinetuneLoaderTests(unittest.TestCase):
    """`finetune.py` builds its loaders through the same shape-batched helper."""

    def test_mixed_domain_loader_is_seeded_and_order_stable(self) -> None:
        import finetune
        import train

        target = _OrderDataset(8)
        replay = _OrderDataset(6)
        entries = [("target", target, index) for index in range(8)] + [
            ("replay", replay, index) for index in range(6)
        ]

        def order(seed: int) -> list[float]:
            dataset = finetune.MixedDomainDataset(entries)
            loader = train.make_shape_loader(
                dataset, list(range(len(dataset))), 1, True, seed
            )
            train.set_loader_epoch(loader, 0)
            return [
                float(value) for batch in loader for value in batch["value"].flatten()
            ]

        self.assertEqual(order(17), order(17))
        self.assertNotEqual(order(17), order(18))

    def test_domain_loader_has_no_shuffling(self) -> None:
        import finetune
        import train

        dataset = _OrderDataset(5)
        first = train.make_shape_loader(dataset, [3, 1, 4], 1, False, 7)
        second = train.make_shape_loader(dataset, [3, 1, 4], 1, False, 7)
        first_order = [
            float(value) for batch in first for value in batch["value"].flatten()
        ]
        second_order = [
            float(value) for batch in second for value in batch["value"].flatten()
        ]
        self.assertEqual(first_order, [3.0, 1.0, 4.0])
        self.assertEqual(first_order, second_order)
        self.assertIsInstance(finetune.MixedDomainDataset, type)


class ArgumentSnapshotTests(unittest.TestCase):
    def test_paths_become_strings_and_keys_are_preserved(self) -> None:
        args = SimpleNamespace(
            data_dir=Path("/tmp/data"),
            output_dir=Path("/tmp/out"),
            max_samples=5,
            flag=True,
        )
        snapshot = argument_snapshot(args)
        self.assertEqual(
            snapshot,
            {
                "data_dir": "/tmp/data",
                "output_dir": "/tmp/out",
                "max_samples": 5,
                "flag": True,
            },
        )

    def test_path_lists_become_string_lists(self) -> None:
        args = SimpleNamespace(extra=[Path("/a"), Path("/b")])
        self.assertEqual(argument_snapshot(args)["extra"], ["/a", "/b"])


class ProtocolVersionTests(unittest.TestCase):
    def test_version_is_the_frozen_1_0_1(self) -> None:
        self.assertEqual(METRIC_PROTOCOL_VERSION, "1.0.1")

    def test_split_identity_version_is_a_separate_constant(self) -> None:
        self.assertEqual(SPLIT_IDENTITY_VERSION, "1.0.0")
        # The two version axes must not be aliases of each other.
        self.assertNotEqual(SPLIT_IDENTITY_VERSION, METRIC_PROTOCOL_VERSION)
        self.assertNotIn(METRIC_PROTOCOL_VERSION, SPLIT_IDENTITY_VERSION)

    def test_split_hash_payload_reports_the_split_identity_version(self) -> None:
        identity = build_split_identity(
            SplitSpec(100, 0.8, 0.1, 42), [f"{index:04d}" for index in range(100)]
        )
        self.assertEqual(
            identity.as_dict()["split_identity_version"], SPLIT_IDENTITY_VERSION
        )

    def test_coverage_dict_exposes_both_sides(self) -> None:
        spec = SplitSpec(100, 0.6, 0.2, 42)
        identity = build_split_identity(spec, [f"{index:04d}" for index in range(100)])
        coverage = build_evaluation_coverage(identity, identity.test_sample_ids[:4])
        payload = coverage.as_dict()
        for field in (
            "test_split_size",
            "test_split_sample_ids_sha256",
            "split_sha256",
            "evaluated_samples",
            "evaluated_sample_ids_sha256",
            "evaluation_mode",
            "is_full_test_evaluation",
            "max_samples",
        ):
            self.assertIn(field, payload)
        self.assertEqual(payload["test_split_size"], 20)
        self.assertEqual(payload["evaluated_samples"], 4)
        self.assertFalse(payload["is_full_test_evaluation"])


class SplitHashVersionIndependenceTests(unittest.TestCase):
    """`split_sha256` must depend on the split identity version only.

    Regression guard: the hash payload used to embed ``METRIC_PROTOCOL_VERSION``,
    so bumping the metric protocol silently changed every split hash even though
    the dataset, sample IDs, split seed and fractions were untouched.
    """

    SAMPLE_IDS = [f"{index:04d}" for index in range(1000)]
    SPEC = SplitSpec(1000, 0.8, 0.1, 42)
    # Frozen value for the rock_1000 configuration, computed before the split
    # identity version was decoupled from the metric protocol version.
    STABLE_ROCK_1000_SHA256 = (
        "cf6f1094b2bc374a760e814c2ac5a6ba9c50fd02c87e356310fe098c109db6cf"
    )

    def test_metric_protocol_version_does_not_affect_the_split_hash(self) -> None:
        import rtm_inv.protocol as protocol

        baseline = build_split_identity(self.SPEC, self.SAMPLE_IDS)
        with unittest.mock.patch.object(
            protocol, "METRIC_PROTOCOL_VERSION", "9.9.9"
        ):
            changed = build_split_identity(self.SPEC, self.SAMPLE_IDS)
        self.assertEqual(baseline.split_sha256, changed.split_sha256)
        self.assertEqual(
            baseline.test_sample_ids_sha256, changed.test_sample_ids_sha256
        )

    def test_split_identity_version_does_affect_the_split_hash(self) -> None:
        import rtm_inv.protocol as protocol

        baseline = build_split_identity(self.SPEC, self.SAMPLE_IDS)
        with unittest.mock.patch.object(protocol, "SPLIT_IDENTITY_VERSION", "2"):
            bumped = build_split_identity(self.SPEC, self.SAMPLE_IDS)
        self.assertNotEqual(baseline.split_sha256, bumped.split_sha256)

    def test_rock_1000_split_hash_is_stable(self) -> None:
        identity = build_split_identity(self.SPEC, self.SAMPLE_IDS)
        self.assertEqual(identity.split_sha256, self.STABLE_ROCK_1000_SHA256)
        self.assertEqual(
            identity.test_sample_ids_sha256,
            "68eef2e37648a63bcd495e4fd9a10f6d6535c68dc028067465004350ef0a9cee",
        )

    def test_dataset_size_change_moves_the_split_hash(self) -> None:
        baseline = build_split_identity(self.SPEC, self.SAMPLE_IDS)
        smaller = build_split_identity(
            SplitSpec(999, 0.8, 0.1, 42), self.SAMPLE_IDS[:-1]
        )
        self.assertNotEqual(baseline.split_sha256, smaller.split_sha256)

    def test_split_fraction_change_moves_the_split_hash(self) -> None:
        baseline = build_split_identity(self.SPEC, self.SAMPLE_IDS)
        for spec in (
            SplitSpec(1000, 0.7, 0.1, 42),
            SplitSpec(1000, 0.8, 0.15, 42),
        ):
            with self.subTest(spec=spec):
                self.assertNotEqual(
                    baseline.split_sha256,
                    build_split_identity(spec, self.SAMPLE_IDS).split_sha256,
                )

    def test_split_seed_change_moves_the_split_hash(self) -> None:
        baseline = build_split_identity(self.SPEC, self.SAMPLE_IDS)
        other = build_split_identity(
            SplitSpec(1000, 0.8, 0.1, 43), self.SAMPLE_IDS
        )
        self.assertNotEqual(baseline.split_sha256, other.split_sha256)

    def test_sample_id_change_moves_the_split_hash(self) -> None:
        baseline = build_split_identity(self.SPEC, self.SAMPLE_IDS)
        renamed = list(self.SAMPLE_IDS)
        renamed[baseline.test_indices[0]] = "9999"
        self.assertNotEqual(
            baseline.split_sha256,
            build_split_identity(self.SPEC, renamed).split_sha256,
        )


if __name__ == "__main__":
    unittest.main()
