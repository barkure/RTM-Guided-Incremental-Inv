from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtm_inv import BscanOnlyNet, DeepwaveClosedLoopRTM, RTMDataset, RTMInvNet
from rtm_inv.protocol import (
    MetricProtocol,
    SplitIdentity,
    aggregate_metric_rows,
    argument_snapshot,
    build_evaluation_coverage,
    build_split_identity,
    compute_metric_row as _protocol_metric_row,
    file_sha256,
    git_revision,
    metric_fieldnames as _protocol_fieldnames,
    parameter_counts,
    resolve_seeds,
    split_spec_from_saved_args,
)
from rtm_inv.initialization import (
    DEFAULT_RANDOM_CLIP_PERCENT,
    DEFAULT_RANDOM_STD_PERCENT,
    INITIALIZATION_MODES,
    MODE_ORACLE_MEAN,
    InitializationSpec,
    build_initialization_spec,
    compute_train_global_constant,
    normalise_mode,
)

EVAL_SPLIT = "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the reconstructed test split."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Evaluate one sample ID from the reconstructed test split.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--plot-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rtm-shot-batch-size", type=int, default=None)
    parser.add_argument(
        "--initial-model-scale",
        type=float,
        default=1.0,
        help="Multiply the per-sample mean-constant initial permittivity by this factor.",
    )
    parser.add_argument(
        "--bscan-input-perturbation",
        choices=["none", "sample_shuffle", "zero"],
        default="none",
        help="Negative-control perturbation for a B-scan-only checkpoint.",
    )
    parser.add_argument(
        "--perturbation-seed",
        type=int,
        default=42,
        help="Seed used to create the sample-level B-scan derangement.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Override the split seed recorded in the checkpoint. Defaults to the "
            "checkpoint's split_seed, then to its legacy seed."
        ),
    )
    parser.add_argument(
        "--initial-model-mode",
        choices=INITIALIZATION_MODES,
        default=None,
        help=(
            "Initial-model recipe. Defaults to the checkpoint's recorded mode, then "
            "to 'oracle_mean' (the historical behaviour) with a warning. Non-oracle "
            "modes derive their background from the training split only."
        ),
    )
    parser.add_argument(
        "--initial-bias-percent",
        type=float,
        default=0.0,
        help="Systematic relative bias for biased_global_constant.",
    )
    parser.add_argument(
        "--initial-random-seed",
        type=int,
        default=None,
        help="Seed for random_biased_global_constant realisations.",
    )
    parser.add_argument(
        "--initial-random-std-percent",
        type=float,
        default=DEFAULT_RANDOM_STD_PERCENT,
        help="Truncated-normal standard deviation for the random background.",
    )
    parser.add_argument(
        "--initial-random-clip-percent",
        type=float,
        default=DEFAULT_RANDOM_CLIP_PERCENT,
        help="Truncation bound for the random background.",
    )
    return parser.parse_args()


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device(default_device())
    return torch.device(device_arg)


def checkpoint_args(checkpoint: dict) -> SimpleNamespace:
    return SimpleNamespace(**checkpoint.get("args", {}))


def pick(cli_value, checkpoint_value, fallback):
    if cli_value is not None:
        return cli_value
    if checkpoint_value is not None:
        return checkpoint_value
    return fallback


def build_dataset(
    args: argparse.Namespace,
    saved_args: SimpleNamespace,
    initial_model_mode: str = MODE_ORACLE_MEAN,
    initializer: InitializationSpec | None = None,
) -> Dataset:
    limit = getattr(saved_args, "limit", None)
    return RTMDataset(
        args.data_dir,
        limit=limit,
        normalize_models=True,
        bscan_scale_quantile=getattr(saved_args, "bscan_scale_quantile", 0.95),
        bscan_scale_eps=getattr(saved_args, "bscan_scale_eps", 1e-12),
        initial_model_mode=initial_model_mode,
        initializer=initializer,
    )


def resolve_initialization(
    args: argparse.Namespace,
    saved_args: SimpleNamespace,
    dataset: Dataset,
    identity: SplitIdentity,
) -> tuple[str, InitializationSpec | None, list[str]]:
    """Decide the initial-model recipe and, when needed, derive it from train only.

    Precedence: explicit ``--initial-model-mode``, then the checkpoint's recorded
    mode, then ``oracle_mean`` with an explicit warning. ``c_train`` is computed
    from the reconstructed **training** indices only, so validation and test targets
    cannot take part.
    """
    notes: list[str] = []
    cli_mode = getattr(args, "initial_model_mode", None)
    recorded_raw = getattr(saved_args, "initial_model_mode", None)
    if cli_mode is not None:
        mode = cli_mode
    elif recorded_raw is not None:
        recorded_mode = normalise_mode(recorded_raw)
        if recorded_mode is None:
            mode = MODE_ORACLE_MEAN
            notes.append(
                f"checkpoint records an unrecognised initial_model_mode="
                f"{recorded_raw!r}; falling back to 'oracle_mean'. Confirm this is "
                "intended before quoting the result."
            )
        else:
            mode = recorded_mode
            if str(recorded_raw) != recorded_mode:
                notes.append(
                    f"checkpoint's legacy initial_model_mode={recorded_raw!r} maps to "
                    f"{recorded_mode!r}"
                )
            else:
                notes.append(
                    f"initial_model_mode not given on the CLI; using the checkpoint's {mode!r}"
                )
    else:
        mode = MODE_ORACLE_MEAN
        notes.append(
            "initial_model_mode missing from both the CLI and the checkpoint args; "
            "falling back to 'oracle_mean' (the historical oracle-informed behaviour). "
            "This is NOT a realistic initialisation."
        )
    if mode == MODE_ORACLE_MEAN:
        return mode, None, notes

    model_min = float(getattr(saved_args, "model_min", RTMInvNet.MODEL_MIN))
    model_max = float(getattr(saved_args, "model_max", RTMInvNet.MODEL_MAX))
    if mode == "calibration_derived":
        raise NotImplementedError(
            "calibration_derived requires independent calibration measurements "
            "(arrival times, field velocity, dielectric probe) that this dataset does "
            "not provide; using a target mean instead would be target leakage."
        )
    train_constant = compute_train_global_constant(
        dataset, list(identity.train_indices), progress_every=None
    )
    notes.append(
        f"c_train={train_constant.value:.6f} computed from {train_constant.train_samples} "
        f"training samples ({train_constant.train_pixels} pixels), "
        f"ids_sha256={train_constant.train_sample_ids_sha256[:16]}"
    )
    spec = build_initialization_spec(
        mode,
        model_min=model_min,
        model_max=model_max,
        train_constant=train_constant,
        bias_percent=float(getattr(args, "initial_bias_percent", 0.0)),
        random_std_percent=float(getattr(args, "initial_random_std_percent", DEFAULT_RANDOM_STD_PERCENT)),
        random_clip_percent=float(getattr(args, "initial_random_clip_percent", DEFAULT_RANDOM_CLIP_PERCENT)),
        random_seed=getattr(args, "initial_random_seed", None),
    )
    notes.append(f"initialization: {spec.describe()}")
    return mode, spec, notes


def build_model(
    args: argparse.Namespace | SimpleNamespace,
    saved_args: SimpleNamespace,
    device: torch.device,
) -> nn.Module:
    shot_batch_size = pick(
        getattr(args, "rtm_shot_batch_size", None),
        getattr(saved_args, "rtm_shot_batch_size", None),
        105,
    )
    input_mode = getattr(saved_args, "model_input_mode", "m0_rtm")
    base_channels = getattr(saved_args, "unet_base_channels", 64)
    unet_depth = getattr(saved_args, "unet_depth", 3)
    num_stages = getattr(saved_args, "num_stages", 2)
    update_backbone = getattr(saved_args, "update_backbone", "unet")
    recompute_rtm_between_stages = getattr(saved_args, "recompute_rtm_between_stages", True)
    rtm_operator = DeepwaveClosedLoopRTM(shot_batch_size=shot_batch_size, device=device)
    if input_mode == "bscan":
        return BscanOnlyNet(
            unet_base_channels=base_channels,
            unet_depth=unet_depth,
        ).to(device)
    return RTMInvNet(
        rtm_operator=rtm_operator,
        compute_synthetic_data=False,
        input_mode=input_mode,
        unet_base_channels=base_channels,
        unet_depth=unet_depth,
        num_stages=num_stages,
        update_backbone=update_backbone,
        recompute_rtm_between_stages=recompute_rtm_between_stages,
    ).to(device)


def compute_metric_row(
    prefix: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    protocol: MetricProtocol | None = None,
) -> dict[str, str]:
    """Per-sample metric row; delegates to the frozen protocol definition.

    ``protocol`` defaults to the bounds currently installed on ``RTMInvNet`` so
    that plotting helpers can keep calling this without threading the object.
    """
    if protocol is None:
        protocol = MetricProtocol(RTMInvNet.MODEL_MIN, RTMInvNet.MODEL_MAX)
    return _protocol_metric_row(prefix, pred, target, protocol)


def denormalise_model(model: torch.Tensor) -> torch.Tensor:
    return model * RTMInvNet.MODEL_SCALE + RTMInvNet.MODEL_MIN


def move_batch(
    batch: dict,
    device: torch.device,
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    sample_id = str(batch.get("sample_id", "unknown"))
    sample_config = batch.get("rtm_config")
    if sample_config is not None:
        sample_config = sample_config.unsqueeze(0).to(device)
    observed_is_processed = batch.get("observed_is_processed")
    if observed_is_processed is not None:
        observed_is_processed = observed_is_processed.unsqueeze(0).to(device)
    return (
        sample_id,
        batch["bscan"].unsqueeze(0).to(device),
        batch["initial_model"].unsqueeze(0).to(device),
        batch["target_model"].unsqueeze(0).to(device),
        sample_config,
        observed_is_processed,
    )


def plot_prediction(
    output_path: Path,
    sample_id: str,
    position: int,
    input_mode: str,
    stage_input: torch.Tensor,
    initial_model: torch.Tensor,
    target_model: torch.Tensor,
    outputs: dict[str, torch.Tensor],
) -> None:
    observed = stage_input[0].detach().cpu()
    initial = RTMInvNet._denormalise(initial_model)[0, 0].detach().cpu()
    rtm_i0 = outputs["stage1_image"][0, 0].detach().cpu()
    target = RTMInvNet._denormalise(target_model)[0, 0].detach().cpu()
    stage1 = RTMInvNet._denormalise(outputs["stage1_model"])[0, 0].detach().cpu()
    final = RTMInvNet._denormalise(outputs["final_model"])[0, 0].detach().cpu()
    num_stages = int(outputs.get("num_stages", 1))
    rtm_i1 = None
    if num_stages > 1 and "stage2_image" in outputs:
        rtm_i1 = outputs["stage2_image"][0, 0].detach().cpu()

    initial_mae = F.l1_loss(initial, target).item()
    final_mae = F.l1_loss(final, target).item()

    vmin = 2.0
    vmax = 10.0
    input_panel = observed.squeeze()
    if input_panel.ndim == 3:
        input_panel = input_panel.squeeze(1)
    if input_panel.ndim == 2 and input_panel.shape[0] < input_panel.shape[1]:
        input_panel = input_panel.T

    bscan_abs_max = float(torch.quantile(input_panel.abs().flatten(), 0.90).item())
    if bscan_abs_max <= 0.0:
        bscan_abs_max = 1.0

    rtm_abs_max = float(torch.quantile(rtm_i0.abs().flatten(), 0.90).item())
    if rtm_abs_max <= 0.0:
        rtm_abs_max = 1.0
    rtm_i1_abs_max = None
    if rtm_i1 is not None:
        rtm_i1_abs_max = float(torch.quantile(rtm_i1.abs().flatten(), 0.90).item())
        if rtm_i1_abs_max <= 0.0:
            rtm_i1_abs_max = 1.0

    is_bscan_only = bool(outputs.get("direct_model_prediction", False))
    if is_bscan_only:
        title_metrics = f"MAE final {final_mae:.4f}"
        panels = [
            ("Bscan", input_panel, "gray", -bscan_abs_max, bscan_abs_max),
            ("Final", final, "turbo", vmin, vmax),
            ("Target", target, "turbo", vmin, vmax),
        ]
    elif input_mode == "m0_bscan":
        title_metrics = f"MAE init {initial_mae:.4f} -> final {final_mae:.4f}"
        panels = [
            ("Bscan", input_panel, "gray", -bscan_abs_max, bscan_abs_max),
            ("Initial", initial, "turbo", vmin, vmax),
            ("Stage1", stage1, "turbo", vmin, vmax),
            ("Final", final, "turbo", vmin, vmax),
            ("Target", target, "turbo", vmin, vmax),
        ]
    elif input_mode == "rtm":
        title_metrics = f"MAE final {final_mae:.4f}"
        panels = [
            ("Bscan", input_panel, "gray", -bscan_abs_max, bscan_abs_max),
            ("RTM 1 (M0)", rtm_i0, "gray", -rtm_abs_max, rtm_abs_max),
            ("Stage1", stage1, "turbo", vmin, vmax),
            ("Final", final, "turbo", vmin, vmax),
            ("Target", target, "turbo", vmin, vmax),
        ]
    else:
        title_metrics = f"MAE init {initial_mae:.4f} -> final {final_mae:.4f}"
        panels = [
            ("Bscan", input_panel, "gray", -bscan_abs_max, bscan_abs_max),
            ("Initial", initial, "turbo", vmin, vmax),
            ("RTM 1 (M0)", rtm_i0, "gray", -rtm_abs_max, rtm_abs_max),
            ("Stage1", stage1, "turbo", vmin, vmax),
            ("Final", final, "turbo", vmin, vmax),
            ("Target", target, "turbo", vmin, vmax),
        ]
    if num_stages == 1:
        panels = [panel for panel in panels if panel[0] != "Stage1"]
    elif rtm_i1 is not None and rtm_i1_abs_max is not None:
        final_position = next(
            index for index, panel in enumerate(panels) if panel[0] == "Final"
        )
        panels.insert(
            final_position,
            (
                "RTM 2 (M1)",
                rtm_i1,
                "gray",
                -rtm_i1_abs_max,
                rtm_i1_abs_max,
            ),
        )

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4.8), dpi=150)
    if len(panels) == 1:
        axes = [axes]
    fig.suptitle(
        f"sample {sample_id} (test pos {position + 1}) | {title_metrics}",
        fontsize=12,
    )
    for ax, (title, panel, cmap, pvmin, pvmax) in zip(axes, panels, strict=True):
        ax.set_box_aspect(0.5)
        image = ax.imshow(panel, cmap=cmap, aspect="auto", vmin=pvmin, vmax=pvmax)
        ax.set_title(title)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def split_identity(
    dataset: Dataset,
    saved_args: SimpleNamespace,
    *,
    split_seed_override: int | None = None,
) -> tuple[SplitIdentity, tuple[str, ...]]:
    """Rebuild the split and return its stable identity (IDs + hashes).

    The split is reconstructed in memory from ``train_split``/``val_split`` and
    the resolved split seed; no ``split.json`` is involved.
    """
    spec, notes = split_spec_from_saved_args(saved_args, len(dataset))
    if split_seed_override is not None:
        spec = type(spec)(
            dataset_size=spec.dataset_size,
            train_split=spec.train_split,
            val_split=spec.val_split,
            split_seed=int(split_seed_override),
        )
        notes = notes + (
            f"split_seed overridden on the command line to {int(split_seed_override)}",
        )
    sample_ids = [sample.sample_id for sample in dataset.samples]
    return build_split_identity(spec, sample_ids), notes


def split_subset(
    dataset: Dataset,
    saved_args: SimpleNamespace,
) -> tuple[Subset, tuple[int, int, int], int]:
    """Backwards-compatible helper returning ``(test_subset, sizes, split_seed)``."""
    identity, _notes = split_identity(dataset, saved_args)
    return (
        Subset(dataset, list(identity.test_indices)),
        identity.spec.sizes,
        identity.spec.split_seed,
    )


def write_split_ids(path: Path, identity: SplitIdentity) -> None:
    """Persist the exact, full test-sample order so runs can be cross-checked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_position", "dataset_index", "sample_id"])
        for position, (dataset_index, sample_id) in enumerate(
            zip(identity.test_indices, identity.test_sample_ids, strict=True), start=1
        ):
            writer.writerow([position, dataset_index, sample_id])


def write_evaluated_ids(
    path: Path,
    entries: list[tuple[int, int, str]],
) -> None:
    """Persist the samples actually scored, in the order metrics.csv lists them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["evaluation_position", "test_position", "dataset_index", "sample_id"]
        )
        for index, (test_position, dataset_index, sample_id) in enumerate(entries, start=1):
            writer.writerow([index, test_position + 1, dataset_index, sample_id])


def derange_indices(indices: list[int], seed: int) -> list[int]:
    """Return a deterministic permutation with no sample left in place."""
    if len(indices) < 2:
        raise ValueError("At least two test samples are required for sample_shuffle.")
    generator = torch.Generator().manual_seed(int(seed))
    positions = torch.arange(len(indices))
    for _ in range(10_000):
        permutation = torch.randperm(len(indices), generator=generator)
        if not torch.any(permutation == positions):
            return [indices[int(position)] for position in permutation]
    raise RuntimeError("Unable to construct a derangement for the test split.")


def metric_fieldnames(include_physical: bool = True) -> list[str]:
    protocol = MetricProtocol(RTMInvNet.MODEL_MIN, RTMInvNet.MODEL_MAX)
    names = _protocol_fieldnames(protocol)
    if not include_physical:
        names = [name for name in names if "_er_" not in name]
    return ["sample_id"] + names


def main() -> None:
    args = parse_args()
    if args.initial_model_scale <= 0.0:
        raise ValueError("--initial-model-scale must be positive.")
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint_args(checkpoint)

    model_min = getattr(saved_args, "model_min", RTMInvNet.MODEL_MIN)
    model_max = getattr(saved_args, "model_max", RTMInvNet.MODEL_MAX)
    if model_max <= model_min:
        raise ValueError("model_max must be greater than model_min.")
    RTMInvNet.MODEL_MIN = float(model_min)
    RTMInvNet.MODEL_MAX = float(model_max)
    RTMInvNet.MODEL_SCALE = float(model_max - model_min)
    protocol = MetricProtocol(model_min=float(model_min), model_max=float(model_max))

    cli_split_seed = args.split_seed
    resolved_split_seed = (
        cli_split_seed
        if cli_split_seed is not None
        else getattr(saved_args, "split_seed", None)
    )
    seeds = resolve_seeds(
        legacy_seed=getattr(saved_args, "seed", None),
        split_seed=resolved_split_seed,
        training_seed=getattr(saved_args, "training_seed", None),
        sampler_seed=getattr(saved_args, "sampler_seed", None),
    )
    for line in seeds.log_lines():
        print(line, flush=True)

    dataset = build_dataset(args, saved_args)
    identity, identity_notes = split_identity(
        dataset, saved_args, split_seed_override=cli_split_seed
    )
    for note in identity_notes:
        print(f"split_note: {note}", flush=True)

    # Resolve the initial-model recipe against the reconstructed split, then rebuild
    # the dataset with it. c_train (when needed) is derived from the training indices
    # of this same identity, so evaluation and training agree on the background.
    initial_model_mode, initializer, init_notes = resolve_initialization(
        args, saved_args, dataset, identity
    )
    for note in init_notes:
        print(f"initialization_note: {note}", flush=True)
    if initializer is not None:
        dataset = build_dataset(args, saved_args, initial_model_mode, initializer)
    print(
        f"initial_model_mode={initial_model_mode} "
        f"realistic={bool(initializer and initializer.is_realistic)}",
        flush=True,
    )
    split_sizes = identity.spec.sizes
    print(
        f"split_identity: split_seed={identity.spec.split_seed} "
        f"test_samples={identity.test_size} "
        f"test_sample_ids_sha256={identity.test_sample_ids_sha256[:16]} "
        f"split_sha256={identity.split_sha256[:16]}",
        flush=True,
    )
    # Plan the exact evaluated sequence up front: the full test split, its first
    # --max-samples entries, or the single --sample-id entry. The evaluated identity
    # written below comes from this list, so it can never drift from metrics.csv.
    planned: list[tuple[int, int]] = list(enumerate(identity.test_indices))
    if args.sample_id is not None:
        requested_id = args.sample_id.removeprefix("sample_")
        if requested_id.isdigit():
            requested_id = requested_id.zfill(4)
        matches = [
            (test_position, dataset_index)
            for test_position, dataset_index in planned
            if dataset.samples[dataset_index].sample_id == requested_id
        ]
        if not matches:
            raise ValueError(
                f"Sample {args.sample_id!r} is not in the reconstructed test split."
            )
        planned = [matches[0]]
        print(
            f"selected_test_sample={requested_id} test_position={matches[0][0] + 1}",
            flush=True,
        )
    elif args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive.")
        planned = planned[: args.max_samples]

    model = build_model(args, saved_args, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    input_mode = getattr(saved_args, "model_input_mode", "m0_rtm")
    if args.bscan_input_perturbation != "none" and input_mode != "bscan":
        raise ValueError(
            "--bscan-input-perturbation is currently restricted to B-scan-only checkpoints."
        )

    test_indices = [dataset_index for _, dataset_index in planned]
    shuffled_input_by_target: dict[int, int] = {}
    if args.bscan_input_perturbation == "sample_shuffle":
        shuffled_indices = derange_indices(test_indices, args.perturbation_seed)
        shuffled_input_by_target = dict(zip(test_indices, shuffled_indices, strict=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict[str, str]] = []
    evaluated_entries: list[tuple[int, int, str]] = []
    plotted = 0
    total = len(planned)
    print(f"test: samples={total} device={device}", flush=True)

    with torch.no_grad():
        for position, (test_position, dataset_index) in enumerate(planned):
            batch = dataset[dataset_index]
            (
                sample_id,
                stage_input,
                initial_model,
                target_model,
                sample_config,
                observed_is_processed,
            ) = move_batch(batch, device)
            input_sample_id = sample_id
            if args.bscan_input_perturbation == "sample_shuffle":
                input_dataset_index = shuffled_input_by_target[dataset_index]
                input_batch = dataset[input_dataset_index]
                input_sample_id = str(input_batch.get("sample_id", "unknown"))
                stage_input = input_batch["bscan"].unsqueeze(0).to(device)
            elif args.bscan_input_perturbation == "zero":
                input_sample_id = "zero"
                stage_input = torch.zeros_like(stage_input)
            if args.initial_model_scale != 1.0:
                initial_er = denormalise_model(initial_model)
                initial_er = torch.clamp(
                    initial_er * args.initial_model_scale,
                    min=RTMInvNet.MODEL_MIN,
                    max=RTMInvNet.MODEL_MAX,
                )
                initial_model = (
                    (initial_er - RTMInvNet.MODEL_MIN) / RTMInvNet.MODEL_SCALE
                )
            outputs = model(
                stage_input,
                initial_model,
                sample_config=sample_config,
                observed_is_processed=observed_is_processed,
            )

            initial = initial_model[0, 0]
            stage1 = outputs["stage1_model"][0, 0]
            final = outputs["final_model"][0, 0]
            target = target_model[0, 0]
            row: dict[str, str] = {"sample_id": sample_id}
            if args.bscan_input_perturbation != "none":
                row["input_sample_id"] = input_sample_id
            row.update(compute_metric_row("initial", initial, target, protocol))
            row.update(compute_metric_row("stage1", stage1, target, protocol))
            row.update(compute_metric_row("final", final, target, protocol))
            metrics_rows.append(row)
            evaluated_entries.append((test_position, dataset_index, sample_id))

            if plotted < args.plot_samples:
                plot_prediction(
                    args.output_dir / f"sample_{sample_id}.png",
                    sample_id,
                    test_position,
                    input_mode,
                    stage_input,
                    initial_model,
                    target_model,
                    outputs,
                )
                plotted += 1

            if (position + 1) % 10 == 0 or position + 1 == total:
                print(f"test: {position + 1}/{total}", flush=True)

            del (
                outputs,
                stage_input,
                initial_model,
                target_model,
                sample_config,
                observed_is_processed,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fieldnames = metric_fieldnames(include_physical=True)
    if args.bscan_input_perturbation != "none":
        fieldnames.insert(1, "input_sample_id")
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    coverage = build_evaluation_coverage(
        identity,
        [sample_id for _, _, sample_id in evaluated_entries],
        max_samples=args.max_samples,
    )
    if len(coverage.evaluated_sample_ids) != len(metrics_rows):
        raise RuntimeError(
            "Evaluated sample identity does not match the written metrics rows "
            f"({len(coverage.evaluated_sample_ids)} vs {len(metrics_rows)})."
        )

    test_ids_path = args.output_dir / "test_split_sample_ids.csv"
    write_split_ids(test_ids_path, identity)
    evaluated_ids_path = args.output_dir / "evaluated_sample_ids.csv"
    write_evaluated_ids(evaluated_ids_path, evaluated_entries)

    checkpoint_sha256 = file_sha256(args.checkpoint)
    if initializer is not None:
        initialization_fields = initializer.as_dict(evaluated_split=EVAL_SPLIT)
    else:
        initialization_fields = build_initialization_spec(
            MODE_ORACLE_MEAN,
            model_min=RTMInvNet.MODEL_MIN,
            model_max=RTMInvNet.MODEL_MAX,
        ).as_dict(evaluated_split=EVAL_SPLIT)
    summary: dict[str, object] = {
        **protocol.as_dict(),
        **initialization_fields,
        # Split spec (no hashes) so the columns below stay unambiguous.
        **identity.spec.as_dict(),
        **coverage.as_dict(),
        "test_split_sample_ids_path": str(test_ids_path),
        "evaluated_sample_ids_path": str(evaluated_ids_path),
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "device": str(device),
        "rtm_shot_batch_size": getattr(args, "rtm_shot_batch_size", None),
        "eval_split": EVAL_SPLIT,
        **seeds.as_dict(),
        "initial_model_scale": args.initial_model_scale,
        "bscan_input_perturbation": args.bscan_input_perturbation,
        "perturbation_seed": args.perturbation_seed,
        **parameter_counts(model),
        **git_revision(PROJECT_ROOT),
    }
    summary.update(aggregate_metric_rows(metrics_rows, fieldnames))

    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    metadata = {
        **summary,
        "argv": list(sys.argv),
        "resolved_arguments": argument_snapshot(args),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "split_train_indices": list(identity.train_indices),
        "split_val_indices": list(identity.val_indices),
        "split_test_indices": list(identity.test_indices),
        "test_split_sample_ids": list(identity.test_sample_ids),
        "evaluated_sample_entries": [
            {"evaluation_position": index + 1, "test_position": test_position + 1,
             "dataset_index": dataset_index, "sample_id": sample_id}
            for index, (test_position, dataset_index, sample_id) in enumerate(evaluated_entries)
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    print(
        f"checkpoint_epoch={checkpoint.get('epoch')} checkpoint_val_loss={checkpoint.get('val_loss')}"
    )
    print(
        f"split_sizes train={split_sizes[0]} val={split_sizes[1]} test={split_sizes[2]} "
        f"split_seed={identity.spec.split_seed}"
    )
    print(
        f"metric_protocol_version={protocol.version} "
        f"checkpoint_sha256={checkpoint_sha256[:16]}"
    )
    print(
        f"coverage mode={coverage.evaluation_mode} "
        f"is_full_test_evaluation={coverage.is_full_test_evaluation} "
        f"evaluated_samples={coverage.evaluated_samples} test_split_size={coverage.test_split_size}"
    )
    print(
        f"evaluated_sample_ids_sha256={coverage.evaluated_sample_ids_sha256[:16]} "
        f"test_split_sample_ids_sha256={coverage.test_split_sample_ids_sha256[:16]}"
    )
    print(f"saved_metrics={args.output_dir / 'metrics.csv'}")
    print(f"saved_summary={args.output_dir / 'summary.csv'}")
    print(f"saved_metadata={args.output_dir / 'metadata.json'}")
    print(f"saved_test_split_sample_ids={test_ids_path}")
    print(f"saved_evaluated_sample_ids={evaluated_ids_path}")


if __name__ == "__main__":
    main()
