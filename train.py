from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Sampler, Subset

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtm_inv import BscanLoss, DeepwaveClosedLoopRTM, PermittivityLoss, RTMDataset, RTMInvNet
from rtm_inv.stages import (
    DEFAULT_TARGET_ALPHA,
    NON_FINAL_TOTAL_WEIGHT,
    schedule_as_dict,
    stage_loss_weights,
    stage_target_lambdas,
    stage_targets,
)
from rtm_inv.initialization import (
    DEFAULT_RANDOM_CLIP_PERCENT,
    DEFAULT_RANDOM_STD_PERCENT,
    INITIALIZATION_MODES,
    MODE_ORACLE_MEAN,
    InitializationSpec,
    build_initialization_spec,
    compute_train_global_constant,
)
from rtm_inv.protocol import (
    METRIC_PROTOCOL_VERSION,
    SeedConfig,
    SplitIdentity,
    SplitSpec,
    build_split_identity,
    resolve_seeds,
    set_global_seed,
)


MODEL_INPUT_MODES = ("m0_rtm",)
UPDATE_BACKBONES = ("unet",)
BSCAN_SCALE_QUANTILE = 0.95
BSCAN_SCALE_EPS = 1e-12
# A dead legacy key that `main()` used to assign unconditionally as the oracle
# description, so every realistic run's config.json and checkpoint args carried an
# oracle label. It had no reader anywhere, so it never reached the loss, the optimizer,
# the model or any evaluation — but it contradicted the run's own provenance. Three
# seed-42 runs predate its removal and still carry it; that is recorded as a documented
# exception in `evaluation/Realistic initialization/Retraining/PROVENANCE_EXCEPTIONS.json`.
# Nothing may reintroduce it, and a checkpoint may not propagate it into a new run.
LEGACY_DEAD_INITIAL_MODEL_FIELD = "initial_model"
LEGACY_DEAD_INITIAL_MODEL_VALUE = "sample_spatial_mean_constant"


def positive_stage_count(value: str) -> int:
    """Stage counts are no longer restricted to {1, 2}; any N >= 1 is valid."""
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"num_stages must be an integer, got {value!r}") from error
    if count < 1:
        raise argparse.ArgumentTypeError(f"num_stages must be at least 1, got {count}")
    return count


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is None or device_arg == "auto":
        return torch.device(default_device())
    return torch.device(device_arg)


def assert_no_legacy_initial_model_field(namespace: dict) -> None:
    """Refuse to run if the dead legacy key reappears in the parsed arguments.

    The key is what made realistic runs advertise themselves as oracle-initialised in
    their config and checkpoint. It is never read, so its return would be a pure
    provenance regression; failing loudly keeps new runs (seeds 43/44) clean.
    """
    if LEGACY_DEAD_INITIAL_MODEL_FIELD in namespace:
        raise RuntimeError(
            f"refusing to run: the dead legacy key "
            f"{LEGACY_DEAD_INITIAL_MODEL_FIELD!r} is present in the parsed arguments "
            f"(value {namespace[LEGACY_DEAD_INITIAL_MODEL_FIELD]!r}). It has no reader "
            "and used to mislabel realistic runs as oracle-initialised. Remove the "
            "assignment rather than letting it reach config.json and the checkpoint."
        )


def warn_on_legacy_field_in_resumed_checkpoint(checkpoint_args: dict, resume_path) -> None:
    """Warn (do not fail) when resuming a documented pre-fix run.

    The three seed-42 runs carry the dead key. Resuming one of them is legitimate and
    must stay possible, but the key is dropped rather than propagated, so the warning
    is explicit instead of silent.
    """
    if LEGACY_DEAD_INITIAL_MODEL_FIELD not in checkpoint_args:
        return
    value = checkpoint_args[LEGACY_DEAD_INITIAL_MODEL_FIELD]
    if value != LEGACY_DEAD_INITIAL_MODEL_VALUE:
        raise RuntimeError(
            f"refusing to resume {resume_path}: legacy "
            f"{LEGACY_DEAD_INITIAL_MODEL_FIELD}={value!r} is not the documented "
            f"{LEGACY_DEAD_INITIAL_MODEL_VALUE!r} seed-42 exception value."
        )
    print(
        f"warning: {resume_path} carries the dead legacy "
        f"{LEGACY_DEAD_INITIAL_MODEL_FIELD}={value!r} key from the pre-fix code. It "
        "has no reader and is NOT carried into this run (see the documented exception "
        "in PROVENANCE_EXCEPTIONS.json).",
        flush=True,
    )


def initialization_log_line(
    mode: str,
    initializer: InitializationSpec | None,
) -> str:
    """``initial_model=`` log line describing the mode actually in use.

    This used to be a hard-coded ``sample_spatial_mean_constant`` string, which
    mislabelled every realistic-initialisation run as the oracle one. The line is
    now derived from the resolved mode so the log cannot drift from the config and
    the checkpoint.
    """
    if initializer is None:
        if mode != MODE_ORACLE_MEAN:
            raise ValueError(
                f"mode {mode!r} requires an InitializationSpec; refusing to describe it "
                "as the oracle mean."
            )
        return (
            "initial_model=oracle_mean "
            "per-sample target spatial mean (oracle-informed; reads the sample's own target)"
        )
    if initializer.mode != mode:
        raise ValueError(
            f"mode {mode!r} disagrees with initializer.mode={initializer.mode!r}"
        )
    return f"initial_model={mode} {initializer.describe()}"


class ShapeBatchSampler(Sampler[list[int]]):
    """Batch subset positions only when their tensors have matching shapes.

    The iteration order is a pure function of ``(seed, epoch)``: it uses a private
    ``random.Random`` instance and never touches the global PyTorch/Python RNG.
    That keeps the sampler seed independent of the training seed, and makes the
    order reproducible after a resume as long as the same epoch number is passed
    to :meth:`set_epoch`.
    """

    def __init__(
        self,
        dataset: RTMDataset,
        original_indices: list[int],
        batch_size: int,
        shuffle: bool,
        seed: int,
        epoch: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = int(epoch)
        buckets: dict[tuple[int, ...], list[int]] = {}
        for subset_position, original_index in enumerate(original_indices):
            key = dataset.sample_shape_key(original_index)
            buckets.setdefault(key, []).append(subset_position)
        self.buckets = buckets

    def set_epoch(self, epoch: int) -> None:
        """Select which epoch's permutation to emit."""
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for positions in self.buckets.values():
            bucket_positions = list(positions)
            if self.shuffle:
                rng.shuffle(bucket_positions)
            for start in range(0, len(bucket_positions), self.batch_size):
                batches.append(bucket_positions[start : start + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return sum(math.ceil(len(positions) / self.batch_size) for positions in self.buckets.values())


def make_shape_loader(
    dataset: RTMDataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a loader whose order depends only on ``seed`` and the epoch.

    ``batch_size=1`` takes the same seeded path as larger batches; it must not fall
    back to ``DataLoader(shuffle=True)``, which draws from the global PyTorch RNG
    and would make the sample order depend on the training seed instead.
    """
    subset = Subset(dataset, indices)
    batch_sampler = ShapeBatchSampler(
        dataset,
        indices,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        seed=seed,
    )
    return DataLoader(subset, batch_sampler=batch_sampler)


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Pin a shape-batched loader to a training epoch before iterating it."""
    batch_sampler = getattr(loader, "batch_sampler", None)
    if isinstance(batch_sampler, ShapeBatchSampler):
        batch_sampler.set_epoch(epoch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Deepwave RTM closed-loop inversion network.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum total epoch. A resumed run continues up to this epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup, then cosine annealing to zero.")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--skip-final-test", action="store_true",
                        help="Skip final test evaluation for validation-only screening; training is unchanged.")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Legacy single seed. Used as the fallback for any of "
            "--split-seed/--training-seed/--sampler-seed that is not given, so "
            "historical invocations and checkpoints stay valid."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed that determines the train/val/test partition. Default: --seed.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help=(
            "Seed for model initialisation and global RNG state. Default: --seed. "
            "Set explicitly to make training-seed repeats genuinely independent."
        ),
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Seed for batch shuffling. Default: --seed.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Request deterministic algorithms and cuDNN determinism. Recorded in "
            "the config either way; may slow training down."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help=(
            "Use only the first N training indices. The val/test split, the split seed "
            "and c_train are unaffected, so this is safe for reduced-budget pilots."
        ),
    )
    parser.add_argument("--model-min", type=float, default=2.0)
    parser.add_argument("--model-max", type=float, default=10.0)
    parser.add_argument("--rtm-shot-batch-size", type=int, default=105)
    parser.add_argument("--num-stages", type=positive_stage_count, default=2)
    parser.add_argument(
        "--recompute-rtm-between-stages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute RTM from the updated model before stage 2 (disable for fixed-RTM ablation).",
    )
    parser.add_argument(
        "--model-input-mode",
        choices=MODEL_INPUT_MODES,
        default="m0_rtm",
        help="Main network input: current model and RTM image.",
    )
    parser.add_argument("--unet-base-channels", type=int, default=64)
    parser.add_argument("--unet-depth", type=int, default=3)
    parser.add_argument(
        "--update-backbone",
        choices=UPDATE_BACKBONES,
        default="unet",
        help="Update network backbone inside rtm_inv.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=Path("train_runs"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Exact checkpoint/log directory. If set, disables timestamp subdirectory creation.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--data-loss-weight",
        type=float,
        default=0.0,
        help="Synthetic B-scan loss weight. The paper checkpoints use model-space supervision only.",
    )
    parser.add_argument("--data-l1-weight", type=float, default=0.2)
    parser.add_argument("--data-mse-weight", type=float, default=0.3)
    parser.add_argument("--data-grad-weight", type=float, default=0.2)
    parser.add_argument("--model-l1-weight", type=float, default=0.5)
    parser.add_argument("--model-mse-weight", type=float, default=0.5)
    parser.add_argument("--model-grad-weight", type=float, default=0.5)
    parser.add_argument("--model-ssim-weight", type=float, default=0.1)
    parser.add_argument(
        "--model-residual-weight-alpha",
        type=float,
        default=1.0,
        help="Strength of loss weighting from abs(target - M0); 0 disables it.",
    )
    parser.add_argument(
        "--model-residual-weight-max",
        type=float,
        default=5.0,
        help="Maximum model loss weight from abs(target - M0).",
    )
    parser.add_argument(
        "--stage1-target-alpha",
        type=float,
        default=0.5,
        help="Stage-1 supervision target: initial + alpha * (target - initial).",
    )
    parser.add_argument(
        "--non-final-total-weight",
        type=float,
        default=NON_FINAL_TOTAL_WEIGHT,
        help=(
            "Share of the model loss given to all non-final stages together; the final "
            "stage gets the remainder. The frozen default 0.3 reproduces the accepted "
            "stage-3/4 runs; the stage-6.4 screen varies it."
        ),
    )
    parser.add_argument(
        "--initial-model-mode",
        choices=INITIALIZATION_MODES,
        default=MODE_ORACLE_MEAN,
        help=(
            "Initial-model recipe. The default reproduces the historical oracle-informed "
            "per-sample target mean. Realistic modes derive their background from the "
            "training split only."
        ),
    )
    parser.add_argument("--initial-bias-percent", type=float, default=0.0)
    parser.add_argument("--initial-random-seed", type=int, default=None)
    parser.add_argument(
        "--initial-random-std-percent", type=float, default=DEFAULT_RANDOM_STD_PERCENT
    )
    parser.add_argument(
        "--initial-random-clip-percent", type=float, default=DEFAULT_RANDOM_CLIP_PERCENT
    )
    return parser.parse_args()


def make_loaders(
    args: argparse.Namespace,
    seeds: SeedConfig,
) -> tuple[DataLoader, DataLoader, DataLoader, SplitIdentity, InitializationSpec | None]:
    dataset = RTMDataset(
        args.data_dir,
        limit=args.limit,
        normalize_models=True,
        bscan_scale_quantile=BSCAN_SCALE_QUANTILE,
        bscan_scale_eps=BSCAN_SCALE_EPS,
    )
    if not (0.0 < args.train_split < 1.0):
        raise ValueError("--train-split must be between 0 and 1.")
    if not (0.0 < args.val_split < 1.0):
        raise ValueError("--val-split must be between 0 and 1.")
    if args.train_split + args.val_split >= 1.0:
        raise ValueError("--train-split + --val-split must be less than 1.")

    identity = build_split_identity(
        SplitSpec(
            dataset_size=len(dataset),
            train_split=args.train_split,
            val_split=args.val_split,
            split_seed=seeds.split_seed,
        ),
        [sample.sample_id for sample in dataset.samples],
    )

    initializer: InitializationSpec | None = None
    if args.initial_model_mode != MODE_ORACLE_MEAN:
        if args.initial_model_mode == "calibration_derived":
            raise NotImplementedError(
                "calibration_derived requires independent calibration measurements that "
                "this dataset does not provide; using a target mean would be leakage."
            )
        train_constant = compute_train_global_constant(
            dataset, list(identity.train_indices), progress_every=200
        )
        initializer = build_initialization_spec(
            args.initial_model_mode,
            model_min=args.model_min,
            model_max=args.model_max,
            train_constant=train_constant,
            bias_percent=args.initial_bias_percent,
            random_std_percent=args.initial_random_std_percent,
            random_clip_percent=args.initial_random_clip_percent,
            random_seed=args.initial_random_seed,
        )
        dataset = RTMDataset(
            args.data_dir,
            limit=args.limit,
            normalize_models=True,
            bscan_scale_quantile=BSCAN_SCALE_QUANTILE,
            bscan_scale_eps=BSCAN_SCALE_EPS,
            initial_model_mode=args.initial_model_mode,
            initializer=initializer,
        )

    train_indices = list(identity.train_indices)
    if args.train_limit is not None:
        if args.train_limit < 1:
            raise ValueError("--train-limit must be positive.")
        if args.train_limit > len(train_indices):
            raise ValueError(
                f"--train-limit={args.train_limit} exceeds the {len(train_indices)} "
                "training samples in the reconstructed split."
            )
        train_indices = train_indices[: args.train_limit]
    val_indices = list(identity.val_indices)
    test_indices = list(identity.test_indices)
    train_loader = make_shape_loader(
        dataset, train_indices, args.batch_size, shuffle=True, seed=seeds.sampler_seed
    )
    val_loader = make_shape_loader(
        dataset, val_indices, args.batch_size, shuffle=False, seed=seeds.sampler_seed
    )
    test_loader = make_shape_loader(
        dataset, test_indices, args.batch_size, shuffle=False, seed=seeds.sampler_seed
    )
    return train_loader, val_loader, test_loader, identity, initializer


def compute_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    initial_model: torch.Tensor,
    observed_bscan: torch.Tensor | None = None,
    bscan_scale: torch.Tensor | None = None,
    observed_is_processed: torch.Tensor | bool = False,
    data_loss_weight: float = 0.0,
    data_loss_fn: nn.Module | None = None,
    model_loss_fn: nn.Module | None = None,
    stage1_target_alpha: float = 1.0,
    non_final_total_weight: float = NON_FINAL_TOTAL_WEIGHT,
) -> tuple[torch.Tensor, dict[str, float]]:
    info: dict[str, float] = {}

    num_stages = int(outputs.get("num_stages", 2))
    final_model = outputs["final_model"]

    # Stage supervision follows the frozen D1 schedule: stage k targets
    # M0 + lambda_k (Mtrue - M0) with lambda_k = alpha + (1-alpha)(k-1)/(N-1), the
    # non-final stages share 0.3 evenly and the final stage carries 0.7. For N = 2 this
    # is exactly the historical "stage1_target_alpha blend + 0.3/0.7" behaviour.
    stage_weights = stage_loss_weights(num_stages, non_final_total_weight)
    stage_target_list = stage_targets(initial_model, target, num_stages, stage1_target_alpha)

    # --- model-space loss (permittivity) ---
    # Pure direct-prediction baselines such as bscan must not use M0 in
    # the loss weighting. Their supervision is exactly Bscan -> U-Net -> Mtrue.
    model_loss_initial = None if bool(outputs.get("direct_model_prediction", False)) else initial_model
    if model_loss_fn is not None:
        if num_stages == 1:
            final_total, final_info = model_loss_fn(final_model, target, model_loss_initial, return_breakdown=True)
            model_loss = final_total
            for k, v in final_info.items():
                info[f"m_final_{k}"] = float(v)
        else:
            model_loss = torch.tensor(0.0, device=final_model.device)
            for stage_idx in range(1, num_stages + 1):
                is_final = stage_idx == num_stages
                stage_prediction = (
                    final_model if is_final else outputs[f"stage{stage_idx}_model"]
                )
                stage_total, stage_info = model_loss_fn(
                    stage_prediction,
                    stage_target_list[stage_idx - 1],
                    model_loss_initial,
                    return_breakdown=True,
                )
                model_loss = model_loss + stage_weights[stage_idx - 1] * stage_total
                prefix = "m_final" if is_final else f"m_s{stage_idx}"
                for k, v in stage_info.items():
                    info[f"{prefix}_{k}"] = float(v)
                info[f"w_s{stage_idx}"] = float(stage_weights[stage_idx - 1])
    else:
        if num_stages == 1:
            model_loss = F.l1_loss(final_model, target)
            info["m_final_l1"] = float(model_loss.detach())
        else:
            model_loss = torch.tensor(0.0, device=final_model.device)
            for stage_idx in range(1, num_stages + 1):
                is_final = stage_idx == num_stages
                stage_prediction = (
                    final_model if is_final else outputs[f"stage{stage_idx}_model"]
                )
                stage_total = F.l1_loss(stage_prediction, stage_target_list[stage_idx - 1])
                model_loss = model_loss + stage_weights[stage_idx - 1] * stage_total
                prefix = "m_final" if is_final else f"m_s{stage_idx}"
                info[f"{prefix}_l1"] = float(stage_total.detach())
                info[f"w_s{stage_idx}"] = float(stage_weights[stage_idx - 1])

    if num_stages > 1:
        info["stage1_target_alpha"] = float(stage1_target_alpha)
        for stage_idx, lam in enumerate(stage_target_lambdas(num_stages, stage1_target_alpha), start=1):
            info[f"lambda_s{stage_idx}"] = float(lam)
    info["model_loss"] = float(model_loss.detach())

    total = model_loss

    # --- data-space loss (B-scan) ---
    if data_loss_weight > 0.0 and observed_bscan is not None and data_loss_fn is not None:
        final_synth = outputs.get("final_synthetic_data")
        data_loss = torch.tensor(0.0, device=total.device)
        if final_synth is not None:
            d_total, d_info = data_loss_fn(
                final_synth,
                observed_bscan,
                target_scale=bscan_scale,
                target_is_processed=observed_is_processed,
                return_breakdown=True,
            )
            data_loss = data_loss + data_loss_weight * d_total
            for k, v in d_info.items():
                info[f"d_final_{k}"] = float(v)
        total = total + data_loss
        info["data_loss"] = float(data_loss.detach())

    info["total_loss"] = float(total.detach())
    return total, info


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    data_loss_weight: float = 0.0,
    data_loss_fn: nn.Module | None = None,
    model_loss_fn: nn.Module | None = None,
    stage1_target_alpha: float = 1.0,
    non_final_total_weight: float = NON_FINAL_TOTAL_WEIGHT,
    label: str = "train",
    log_fn: callable = print,
    progress_every: int = 1,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_total = len(loader.dataset)
    n_processed = 0
    report_every = max(1, int(progress_every))
    t_start = time.perf_counter()

    # Accumulate per-term breakdown across batches (weighted by batch_size).
    accum: dict[str, float] = {}

    for batch_idx, batch in enumerate(loader):
        observed_bscan = batch["bscan"].to(device)
        bscan_scale = batch.get("bscan_scale")
        if bscan_scale is not None:
            bscan_scale = bscan_scale.to(device)
        initial_model = batch["initial_model"].to(device)
        target_model = batch["target_model"].to(device)
        observed_is_processed = batch.get("observed_is_processed")
        if observed_is_processed is not None:
            observed_is_processed = observed_is_processed.to(device)
        sample_config = batch.get("rtm_config")
        if sample_config is not None:
            sample_config = sample_config.to(device)
        batch_size = initial_model.size(0)

        with torch.set_grad_enabled(is_train):
            outputs = model(
                observed_bscan,
                initial_model,
                sample_config=sample_config,
                observed_is_processed=observed_is_processed,
            )
            loss, info = compute_loss(
                outputs, target_model, initial_model, observed_bscan,
                bscan_scale, observed_is_processed,
                data_loss_weight, data_loss_fn, model_loss_fn,
                stage1_target_alpha, non_final_total_weight,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size
        for k, v in info.items():
            accum[k] = accum.get(k, 0.0) + v * batch_size
        n_processed += batch_size
        del outputs, loss, info, observed_bscan, bscan_scale, initial_model, target_model, sample_config, observed_is_processed
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (batch_idx + 1) % report_every == 0 or n_processed >= n_total:
            elapsed = time.perf_counter() - t_start
            avg_loss = total_loss / n_processed
            log_fn(f"  {label:5s} [{n_processed:>4d}/{n_total:>4d}] "
                   f"loss={avg_loss:.6f}  elapsed={elapsed:7.1f}s  "
                   f"speed={n_processed/elapsed:.2f} samp/s")

    avg_loss = total_loss / n_processed
    avg_info = {k: v / n_processed for k, v in accum.items()}

    # Build a compact one-line breakdown.
    parts = [f"loss={avg_loss:.6f}"]

    # model terms
    model_parts = []
    for stage in ("s1", "final"):
        stage_parts = []
        for term in ("l1", "mse", "grad", "ssim"):
            key = f"m_{stage}_{term}"
            if key in avg_info:
                stage_parts.append(f"{term}={avg_info[key]:.4f}")
        if stage_parts:
            model_parts.append(f"m_{stage}({', '.join(stage_parts)})")
    if model_parts:
        parts.append("model: " + " ".join(model_parts))

    # data terms
    data_parts = []
    for stage in ("final",):
        stage_parts = []
        for term in ("l1", "mse", "grad"):
            key = f"d_{stage}_{term}"
            if key in avg_info:
                stage_parts.append(f"{term}={avg_info[key]:.4f}")
        if stage_parts:
            data_parts.append(f"d_{stage}({', '.join(stage_parts)})")
    if data_parts:
        parts.append("data: " + " ".join(data_parts))

    log_fn(f"  {label:5s} breakdown | {' | '.join(parts)}")

    return avg_loss, avg_info


def best_state_for_resume(model: nn.Module, run_dir: Path) -> dict[str, torch.Tensor]:
    """Keep the historical best across resume, even if no later epoch improves it."""
    best_path = run_dir / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location="cpu", weights_only=False)["model_state_dict"]
    else:
        state = model.state_dict()
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def main() -> None:
    args = parse_args()
    assert_no_legacy_initial_model_field(vars(args))
    seeds = resolve_seeds(
        legacy_seed=args.seed,
        split_seed=args.split_seed,
        training_seed=args.training_seed,
        sampler_seed=args.sampler_seed,
    )
    args.split_seed = seeds.split_seed
    args.training_seed = seeds.training_seed
    args.sampler_seed = seeds.sampler_seed
    args.seed_fallbacks = "; ".join(seeds.fallback_notes)
    device = resolve_device(args.device)
    if args.model_max <= args.model_min:
        raise ValueError("--model-max must be greater than --model-min.")
    RTMInvNet.MODEL_MIN = float(args.model_min)
    RTMInvNet.MODEL_MAX = float(args.model_max)
    RTMInvNet.MODEL_SCALE = float(args.model_max - args.model_min)

    if args.run_dir is not None:
        run_dir = args.run_dir
        args.output = run_dir
    else:
        ckpt_dir = args.output.parent if args.output.suffix else args.output
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = ckpt_dir / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "train.log"

    log_fh = None
    def _log(msg: str = "") -> None:
        print(msg, flush=True)
        if log_fh is not None:
            log_fh.write(msg + "\n")
            log_fh.flush()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "a")

    _log(f"# Training log")
    _log(f"# cmd: {__file__}")
    _log(f"# device={device}  data_dir={args.data_dir}  limit={args.limit}")
    _log(f"# model_range=[{args.model_min}, {args.model_max}]")
    _log(f"# epochs={args.epochs}  batch_size={args.batch_size}  lr={args.learning_rate}  weight_decay={args.weight_decay}")
    _log(
        f"# rtm_shot_batch_size={args.rtm_shot_batch_size}  "
        f"imaging=rtm  data_loss_weight={args.data_loss_weight}"
    )
    _log(
        f"# model_input_mode={args.model_input_mode}  "
        f"num_stages={args.num_stages}  "
        f"recompute_rtm_between_stages={args.recompute_rtm_between_stages}  "
        f"unet_base_channels={args.unet_base_channels}  "
        f"unet_depth={args.unet_depth}  update_backbone={args.update_backbone}"
    )
    _log(
        "# observed_bscan=data.pt:bscan_processed  synthetic_preprocess=trace_mean+shot_median  "
        f"bscan_scale=q{BSCAN_SCALE_QUANTILE:g}_percentile eps={BSCAN_SCALE_EPS}"
    )
    _log(f"# warmup={args.warmup_epochs}  epochs={args.epochs}")
    _log(f"# output={args.output}")
    _log(f"# run_dir={run_dir}")
    _log(f"# log={log_path}")
    _log(f"# metric_protocol_version={METRIC_PROTOCOL_VERSION}")
    for line in seeds.log_lines():
        _log(f"# {line}")
    _log(f"# deterministic={args.deterministic}")
    if args.num_stages == 1:
        _log("# stage_supervision single_stage_target=true_model data_loss_stage=final")
    else:
        _schedule = schedule_as_dict(
            args.num_stages, args.stage1_target_alpha, args.non_final_total_weight
        )
        _log(
            "# stage_supervision "
            f"schedule={_schedule['stage_schedule_version']} "
            f"alpha={args.stage1_target_alpha} "
            f"lambdas={_schedule['target_lambdas']} "
            f"weights={_schedule['stage_loss_weights']} "
            "data_loss_stage=final"
        )
    _log(f"using_device={device}")
    train_loader, val_loader, test_loader, split_identity, initializer = make_loaders(
        args, seeds
    )
    args.initialization = (
        initializer.as_dict(evaluated_split="train") if initializer is not None else {}
    )
    _log(
        f"split_sizes train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} test={len(test_loader.dataset)} "
        f"split_seed={split_identity.spec.split_seed} "
        f"sampler_seed={seeds.sampler_seed}"
    )
    _log(
        f"split_identity test_sample_ids_sha256={split_identity.test_sample_ids_sha256} "
        f"split_sha256={split_identity.split_sha256} dataset_size={split_identity.spec.dataset_size}"
    )
    if args.train_limit is not None:
        _log(
            f"train_limit={args.train_limit} of {len(split_identity.train_indices)} training "
            "samples (val/test split and c_train unchanged)"
        )
    _log(f"initial_model_mode={args.initial_model_mode}")
    _log(initialization_log_line(args.initial_model_mode, initializer))
    if initializer is not None:
        _log(
            "initialization_provenance "
            + " ".join(f"{key}={value}" for key, value in initializer.as_dict(evaluated_split="train").items())
        )

    seed_environment = set_global_seed(
        seeds.training_seed, deterministic=args.deterministic
    )
    _log(
        "global_rng "
        + " ".join(f"{key}={value}" for key, value in seed_environment.items())
    )

    args.seed_environment = seed_environment
    args.split_sha256 = split_identity.split_sha256
    args.test_sample_ids_sha256 = split_identity.test_sample_ids_sha256

    rtm_operator = DeepwaveClosedLoopRTM(
        shot_batch_size=args.rtm_shot_batch_size,
        device=device,
    )

    model = RTMInvNet(
        rtm_operator=rtm_operator,
        compute_synthetic_data=args.data_loss_weight > 0.0,
        input_mode=args.model_input_mode,
        unet_base_channels=args.unet_base_channels,
        unet_depth=args.unet_depth,
        num_stages=args.num_stages,
        update_backbone=args.update_backbone,
        recompute_rtm_between_stages=args.recompute_rtm_between_stages,
    ).to(device)
    model_loss_fn = PermittivityLoss(
        l1_weight=args.model_l1_weight,
        mse_weight=args.model_mse_weight,
        grad_weight=args.model_grad_weight,
        ssim_weight=args.model_ssim_weight,
        residual_weight_alpha=args.model_residual_weight_alpha,
        residual_weight_max=args.model_residual_weight_max,
    )
    data_loss_fn = None
    if args.data_loss_weight > 0.0:
        data_loss_fn = BscanLoss(
            l1_weight=args.data_l1_weight,
            mse_weight=args.data_mse_weight,
            grad_weight=args.data_grad_weight,
            scale_quantile=BSCAN_SCALE_QUANTILE,
            scale_eps=BSCAN_SCALE_EPS,
        )
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    warmup = args.warmup_epochs if args.warmup_epochs > 0 else 0
    total = args.epochs
    eta_min = 1e-6
    if warmup > 0 and warmup < total:
        schedulers = [LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup)]
        if warmup < total:
            schedulers.append(CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=eta_min))
        scheduler = SequentialLR(optimizer, schedulers=schedulers, milestones=[warmup])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total, eta_min=eta_min)

    best_val_loss = float("inf")
    best_state_dict = None
    start_epoch = 0
    epochs_without_improvement = 0

    # Save training config
    # Record the frozen stage schedule (D1) next to the arguments, so a run's own
    # config states which targets and weights supervised it. For N = 2 this reproduces
    # the historical lambda=(0.5, 1.0) / weight=(0.3, 0.7) exactly.
    config_record = dict(vars(args))
    config_record["stage_schedule"] = schedule_as_dict(
        int(args.num_stages),
        float(args.stage1_target_alpha),
        float(args.non_final_total_weight),
    )
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2, default=str))

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        warn_on_legacy_field_in_resumed_checkpoint(
            checkpoint.get("args") or {}, args.resume
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", best_val_loss)))
            start_epoch = int(checkpoint.get("epoch", 0))
            if "epochs_without_improvement" in checkpoint:
                epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
            else:
                best_path = run_dir / "best.pt"
                if best_path.exists():
                    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
                    best_epoch = int(best_checkpoint.get("epoch", start_epoch))
                    epochs_without_improvement = max(0, start_epoch - best_epoch)
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in best_checkpoint["model_state_dict"].items()
                    }
            if best_state_dict is None:
                best_state_dict = best_state_for_resume(model, run_dir)
        current_lr = optimizer.param_groups[0]["lr"]
        _log(
            f"resumed_from={args.resume} start_epoch={start_epoch:03d} "
            f"best_val_loss={best_val_loss:.6f} lr={current_lr:.6g}"
        )

    _log(f"{'epoch':>5} {'train_loss':>12} {'val_loss':>12} {'lr':>10} {'delta':>10}")
    _log(f"{'-----':>5} {'----------':>12} {'--------':>12} {'--':>10} {'-----':>10}")

    # CSV loss log
    loss_csv = run_dir / "loss_log.csv"
    csv_header_written = loss_csv.exists() and loss_csv.stat().st_size > 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Pin the batch order to the absolute epoch number so a resumed run emits
        # exactly the same permutation as an uninterrupted one.
        set_loader_epoch(train_loader, epoch)
        train_loss, train_info = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.data_loss_weight,
            data_loss_fn,
            model_loss_fn,
            args.stage1_target_alpha,
            non_final_total_weight=args.non_final_total_weight,
            label="train",
            log_fn=_log,
        )
        val_loss, val_info = run_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            data_loss_weight=args.data_loss_weight,
            data_loss_fn=data_loss_fn,
            model_loss_fn=model_loss_fn,
            stage1_target_alpha=args.stage1_target_alpha,
            non_final_total_weight=args.non_final_total_weight,
            label="val",
            log_fn=_log,
        )

        if not csv_header_written:
            keys = sorted(set(list(train_info.keys()) + list(val_info.keys())))
            with open(loss_csv, "w") as f:
                f.write("epoch," + ",".join(f"t_{k}" for k in keys) + "," + ",".join(f"v_{k}" for k in keys) + "\n")
            csv_header_written = True
        elif "keys" not in locals():
            with open(loss_csv) as f:
                header = [name.strip() for name in f.readline().strip().split(",")]
            keys = [name[2:] for name in header[1:] if name.startswith("t_")]

        with open(loss_csv, "a") as f:
            if loss_csv.exists() and loss_csv.stat().st_size > 0:
                with open(loss_csv, "rb") as check_f:
                    check_f.seek(-1, 2)
                    if check_f.read(1) != b"\n":
                        f.write("\n")
            row = [str(epoch)]
            for k in keys:
                row.append(f"{train_info.get(k, 0):.6f}")
            for k in keys:
                row.append(f"{val_info.get(k, 0):.6f}")
            f.write(",".join(row) + "\n")
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        delta = best_val_loss - val_loss
        _log(f"{epoch:>5d} {train_loss:>12.6f} {val_loss:>12.6f} {current_lr:>10.6g} {delta:>+10.6f}")

        checkpoint_best_val_loss = min(best_val_loss, val_loss)
        improved = val_loss < best_val_loss - args.min_delta
        next_epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_loss": val_loss,
                "best_val_loss": checkpoint_best_val_loss,
                "epochs_without_improvement": next_epochs_without_improvement,
            },
            run_dir / "last.pt",
        )

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                run_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience is not None and epochs_without_improvement >= args.early_stopping_patience:
            _log(f"early_stopping epoch={epoch:03d} best_val_loss={best_val_loss:.6f} patience={args.early_stopping_patience}")
            break

    if best_state_dict is None:
        best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if args.skip_final_test:
        _log(f"best_val_loss={best_val_loss:.6f} test_skipped=True training_complete=True")
        return

    model.load_state_dict(best_state_dict)
    test_loss, _ = run_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        data_loss_weight=args.data_loss_weight,
        data_loss_fn=data_loss_fn,
        model_loss_fn=model_loss_fn,
        stage1_target_alpha=args.stage1_target_alpha,
        non_final_total_weight=args.non_final_total_weight,
        label="test",
        log_fn=_log,
    )
    _log(f"best_val_loss={best_val_loss:.6f} test_loss={test_loss:.6f}")


if __name__ == "__main__":
    main()
