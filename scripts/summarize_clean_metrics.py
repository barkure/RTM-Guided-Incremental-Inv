from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtm_inv.protocol import (  # noqa: E402
    METRIC_PROTOCOL_VERSION,
    TestSplitReference,
    assert_same_sample_sequence,
    check_evaluation_coverage,
    hash_values,
    read_sample_ids_file,
    test_split_reference_from_summary,
)
from rtm_inv.statistics import (  # noqa: E402
    DEFAULT_PERMUTATION_REPEATS,
    PairedComparison,
    apply_holm_family,
    compare_paired,
    metric_values_from_rows,
)

PROTOCOL_FIELDS = (
    "metric_protocol_version",
    "evaluation_region",
    "model_min",
    "model_max",
    "split_sha256",
    "test_split_sample_ids_sha256",
    "split_seed",
    "canonical_metric_scale",
    "physical_metric_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine Clean evaluation summaries and paired statistics."
    )
    parser.add_argument(
        "--method",
        action="append",
        nargs=2,
        metavar=("LABEL", "SUMMARY"),
        required=True,
        help="Method label and its evaluate.py summary.csv; repeat per method.",
    )
    parser.add_argument(
        "--paired-comparison",
        action="append",
        nargs=3,
        metavar=("LABEL", "REFERENCE_METRICS", "METHOD_METRICS"),
        default=[],
        help="Optional paired Clean comparison; positive effects favor METHOD.",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=DEFAULT_PERMUTATION_REPEATS,
        help=(
            "Monte Carlo sign-flip permutation replicates. This is a Monte Carlo "
            "estimate, not an exhaustive 2**100 enumeration."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260713,
        help="Legacy resampling seed; used when --statistics-seed is not given.",
    )
    parser.add_argument(
        "--statistics-seed",
        type=int,
        default=None,
        help="Seed for every resampling/permutation step. Default: --bootstrap-seed.",
    )
    parser.add_argument(
        "--allow-partial-evaluation",
        action="store_true",
        help=(
            "Accept summaries/metrics that cover only part of the test split. The "
            "evaluated sequence must still be identical across methods. Off by "
            "default: the formal clean table requires full-test evaluations."
        ),
    )
    parser.add_argument(
        "--allow-legacy-artifacts",
        action="store_true",
        help=(
            "Accept pre-protocol summaries lacking is_full_test_evaluation / "
            "test_split_sample_ids.csv. The resulting table is explicitly unverifiable "
            "and must not be quoted as a protocol-v1 result. Off by default."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, str]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}, got {len(rows)}")
    return rows[0]


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def protocol_values(summary: dict[str, str]) -> dict[str, str]:
    return {field: summary.get(field, "") for field in PROTOCOL_FIELDS}


def evaluated_sample_count(summary: dict[str, str]) -> int:
    """Rows actually scored: ``evaluated_samples`` (v1.0.1) or legacy ``samples``."""
    for field in ("evaluated_samples", "samples"):
        value = summary.get(field)
        if value is not None and str(value).strip() != "":
            return int(value)
    raise ValueError("Summary records neither evaluated_samples nor samples")


def assert_single_protocol(
    summaries: list[tuple[str, Path, dict[str, str]]],
    *,
    allow_legacy: bool,
) -> tuple[dict[str, str], bool]:
    """Every method must share one frozen protocol and one split identity.

    Summaries written before the frozen protocol lack these columns. They are only
    accepted with ``--allow-legacy-artifacts``, and are reported as such: their
    SSIM/PSNR provenance and their full/partial coverage cannot be verified from the
    file alone.
    """
    reference_label, _reference_path, reference_summary = summaries[0]
    reference = protocol_values(reference_summary)
    missing = [field for field, value in reference.items() if not value]
    legacy = bool(missing)
    if legacy:
        if not allow_legacy:
            raise ValueError(
                f"{reference_label} lacks the frozen-protocol columns ({', '.join(missing)}). "
                "Regenerate it with the current evaluate.py, or pass "
                "--allow-legacy-artifacts to build an exploratory table from pre-protocol "
                "artefacts (which cannot be verified)."
            )
        print(
            "WARNING: running in legacy-artifact mode "
            f"(missing protocol columns: {', '.join(missing)}). SSIM/PSNR provenance and "
            "full-test coverage cannot be verified from these files; do not quote the "
            "resulting table as a protocol-v1 result.",
            flush=True,
        )
    for label, path, summary in summaries[1:]:
        values = protocol_values(summary)
        mismatches = [
            field
            for field in PROTOCOL_FIELDS
            if values[field] and reference[field] and values[field] != reference[field]
        ]
        if mismatches:
            raise ValueError(
                f"{label} ({path}) disagrees with {reference_label} on {mismatches}. "
                "All methods must be evaluated with the same frozen protocol and split."
            )
    return reference, legacy


def validate_coverage(
    summaries: list[tuple[str, Path, dict[str, str]]],
    *,
    allow_partial: bool,
    allow_legacy: bool,
) -> tuple[TestSplitReference, list[dict[str, object]]]:
    """Reject partial evaluations and verify the evaluated sequence of every method."""
    reference_label, reference_path, reference_summary = summaries[0]
    test_ids_path = reference_path.with_name("test_split_sample_ids.csv")
    if test_ids_path.exists():
        reference = test_split_reference_from_summary(reference_summary, test_ids_path)
    else:
        if not allow_legacy:
            raise ValueError(
                f"{test_ids_path} is missing, so the full test split cannot be "
                "reconstructed from files. Regenerate the evaluation with the current "
                "evaluate.py, or pass --allow-legacy-artifacts to fall back to the "
                "first method's own sample list (coverage then cannot be verified)."
            )
        first_ids = read_sample_ids_file(reference_path.with_name("metrics.csv"))
        reference = TestSplitReference(
            sample_ids=tuple(first_ids),
            sample_ids_sha256=hash_values(first_ids),
            split_sha256=reference_summary.get("split_sha256") or None,
        )
        print(
            "WARNING: legacy-artifact mode infers the reference test split from "
            f"{reference_label}'s own metrics.csv ({len(first_ids)} samples). Whether "
            "that coverage is the full test split CANNOT be verified from these files; "
            "it must not be reported as a protocol-v1 full-test result.",
            flush=True,
        )

    records: list[dict[str, object]] = []
    for label, path, summary in summaries:
        metrics_path = path.with_name("metrics.csv")
        metrics_ids = read_sample_ids_file(metrics_path)
        evaluated_path = path.with_name("evaluated_sample_ids.csv")
        declared_ids = (
            read_sample_ids_file(evaluated_path) if evaluated_path.exists() else None
        )
        if declared_ids is None and not allow_legacy:
            raise ValueError(
                f"{evaluated_path} is missing; the evaluated subset cannot be verified. "
                "Regenerate the evaluation with the current evaluate.py."
            )
        check = check_evaluation_coverage(
            label,
            summary,
            metrics_ids,
            reference,
            declared_evaluated_sample_ids=declared_ids,
            allow_partial=allow_partial,
        )
        if check.is_full:
            print(f"{label}: full-test evaluation verified ({reference.size} samples)", flush=True)
        else:
            print(
                f"{label}: PARTIAL evaluation accepted via --allow-partial-evaluation "
                f"(mode={check.mode}, {len(metrics_ids)} samples)",
                flush=True,
            )
        records.append(
            {
                "evaluation_mode": check.mode,
                "is_full_test_evaluation": check.is_full,
                "evaluated_samples": len(metrics_ids),
                "evaluated_sample_ids_sha256": hash_values(metrics_ids),
                "test_split_size": reference.size,
                "test_split_sample_ids_sha256": reference.sample_ids_sha256,
                "coverage_verified": not check.legacy,
            }
        )
    return reference, records


def initial_record(
    summary: dict[str, str],
    source: Path,
    protocol: dict[str, str],
    coverage: dict[str, object],
) -> dict[str, object]:
    return {
        "method": "Mean-constant initial model",
        "evaluated_samples": evaluated_sample_count(summary),
        "checkpoint_epoch": "",
        "checkpoint_val_loss": "",
        "mae_normalized": float(summary["initial_mae"]),
        "mse_normalized": float(summary["initial_mse"]),
        "ssim": float(summary["initial_ssim"]),
        "psnr": float(summary["initial_psnr"]),
        "physical_permittivity_mae": float(summary["initial_er_mae"]),
        "physical_permittivity_mse": float(summary["initial_er_mse"]),
        **coverage,
        **protocol,
        "source_summary": str(source),
    }


def method_record(
    label: str,
    summary: dict[str, str],
    source: Path,
    protocol: dict[str, str],
    coverage: dict[str, object],
) -> dict[str, object]:
    return {
        "method": label,
        "evaluated_samples": evaluated_sample_count(summary),
        "checkpoint_epoch": int(summary["checkpoint_epoch"]),
        "checkpoint_val_loss": float(summary["checkpoint_val_loss"]),
        "mae_normalized": float(summary["final_mae"]),
        "mse_normalized": float(summary["final_mse"]),
        "ssim": float(summary["final_ssim"]),
        "psnr": float(summary["final_psnr"]),
        "physical_permittivity_mae": float(summary["final_er_mae"]),
        "physical_permittivity_mse": float(summary["final_er_mse"]),
        **coverage,
        **protocol,
        "source_summary": str(source),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


METRIC_COLUMNS = (
    ("mae", "final_mae"),
    ("mse", "final_mse"),
    ("ssim", "final_ssim"),
    ("psnr", "final_psnr"),
)

HOLM_FAMILY_NAME = "main_clean_comparisons"


def write_paired_statistics(
    path: Path,
    comparisons: list[list[str]],
    repeats: int,
    permutation_repeats: int,
    statistics_seed: int,
) -> list[dict[str, object]]:
    """Complete paired statistics for every comparison x protocol metric.

    The confirmatory Holm family is the full cross-product of the supplied
    comparisons and the four protocol metrics (4 comparisons x 4 metrics = 16 in
    the paper's configuration). Exploratory analyses are separate artefacts and
    are never folded into this correction.
    """
    if repeats < 1:
        raise ValueError("--bootstrap-repeats must be at least 1")
    if permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be at least 1")

    collected: list[PairedComparison] = []
    for label, reference_path, method_path in comparisons:
        reference_rows = read_metrics(Path(reference_path))
        method_rows = read_metrics(Path(method_path))
        assert_same_sample_sequence(
            f"{label} method", list(method_rows), f"{label} reference", list(reference_rows)
        )
        for metric, column in METRIC_COLUMNS:
            collected.append(
                compare_paired(
                    label,
                    metric,
                    metric_values_from_rows(list(reference_rows.values()), column),
                    metric_values_from_rows(list(method_rows.values()), column),
                    bootstrap_repeats=repeats,
                    permutation_repeats=permutation_repeats,
                    statistics_seed=statistics_seed,
                )
            )

    family = apply_holm_family(collected, family=HOLM_FAMILY_NAME)
    if len(comparisons) != 4:
        print(
            f"NOTE: Holm family '{HOLM_FAMILY_NAME}' adjusted over "
            f"{len(family)} hypotheses ({len(comparisons)} comparisons x 4 metrics). "
            "The paper's confirmatory family is exactly 4 comparisons x 4 metrics = 16.",
            flush=True,
        )
    write_rows(path, [comparison.as_dict() for comparison in family])
    return [comparison.as_dict() for comparison in family]


def main() -> None:
    args = parse_args()
    statistics_seed = (
        args.bootstrap_seed if args.statistics_seed is None else args.statistics_seed
    )
    summaries = [
        (label, Path(path), read_summary(Path(path))) for label, path in args.method
    ]
    sample_counts = {evaluated_sample_count(summary) for _, _, summary in summaries}
    if len(sample_counts) != 1:
        raise ValueError(f"Inconsistent evaluated sample counts: {sorted(sample_counts)}")
    protocol, _legacy = assert_single_protocol(
        summaries, allow_legacy=args.allow_legacy_artifacts
    )
    reference, coverage_records = validate_coverage(
        summaries,
        allow_partial=args.allow_partial_evaluation,
        allow_legacy=args.allow_legacy_artifacts,
    )

    _, first_path, first_summary = summaries[0]
    records: list[dict[str, object]] = [
        initial_record(first_summary, first_path, protocol, coverage_records[0])
    ]
    records.extend(
        method_record(label, summary, path, protocol, coverage)
        for (label, path, summary), coverage in zip(summaries, coverage_records, strict=True)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    write_rows(summary_path, records)
    print(f"saved_summary={summary_path}")
    print(f"statistics_seed={statistics_seed}")
    print(f"metric_protocol_version={METRIC_PROTOCOL_VERSION}")
    print(f"test_split_sample_ids_sha256={reference.sample_ids_sha256}")
    if args.paired_comparison:
        paired_path = args.output_dir / "paired_statistics.csv"
        rows = write_paired_statistics(
            paired_path,
            args.paired_comparison,
            args.bootstrap_repeats,
            args.permutation_repeats,
            statistics_seed,
        )
        print(
            f"saved_paired_statistics={paired_path} "
            f"hypotheses={len(rows)} holm_family={HOLM_FAMILY_NAME} "
            f"bootstrap_repeats={args.bootstrap_repeats} "
            f"permutation_repeats={args.permutation_repeats}"
        )


if __name__ == "__main__":
    main()
