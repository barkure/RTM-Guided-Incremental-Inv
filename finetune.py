from __future__ import annotations

"""Domain fine-tuning with rotating source-domain replay.

The target sandbox samples adapt the pretrained rtm_inv network to the 049
acquisition geometry. A rotating subset of the original rock training split
is replayed every epoch to limit catastrophic forgetting. Target and replay
validation losses are tracked independently.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rtm_inv import DeepwaveClosedLoopRTM, PermittivityLoss, RTMDataset, RTMInvNet
from rtm_inv.protocol import resolve_seeds, set_global_seed
from train import make_shape_loader, resolve_device, run_epoch, set_loader_epoch


@dataclass(frozen=True)
class DomainSplit:
    train: list[int]
    val: list[int]
    test: list[int]


class MixedDomainDataset(Dataset):
    """Index selected samples from multiple RTMDataset instances."""

    def __init__(self, entries: list[tuple[str, RTMDataset, int]]) -> None:
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        domain, dataset, sample_index = self.entries[index]
        item = dict(dataset[sample_index])
        item["domain"] = domain
        return item

    def sample_shape_key(self, index: int) -> tuple[int, ...]:
        _domain, dataset, sample_index = self.entries[index]
        return dataset.sample_shape_key(sample_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune rtm_inv on sandbox data with original-data replay."
    )
    parser.add_argument("--target-data-dir", type=Path, required=True)
    parser.add_argument("--replay-data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--replay-fraction",
        type=float,
        default=0.30,
        help="Fraction of training samples per epoch drawn from the replay domain.",
    )
    parser.add_argument(
        "--forgetting-tolerance",
        type=float,
        default=0.05,
        help="Maximum allowed replay validation loss increase relative to baseline.",
    )
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Legacy single seed; fallback for the split/training/sampler seeds.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for both domain splits. Default: --seed.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help="Seed for model initialisation and global RNG state. Default: --seed.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=None,
        help="Seed for replay rotation and batch shuffling. Default: --seed.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic algorithms and cuDNN determinism.",
    )
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument("--replay-limit", type=int, default=None)
    parser.add_argument("--rtm-shot-batch-size", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate the supplied checkpoint on both domain splits without training.",
    )
    return parser.parse_args()


def split_indices(
    size: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> DomainSplit:
    train_size = int(size * train_fraction)
    val_size = int(size * val_fraction)
    test_size = size - train_size - val_size
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError(
            f"Empty split for dataset size={size}: "
            f"train={train_size}, val={val_size}, test={test_size}"
        )
    permutation = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()
    return DomainSplit(
        train=permutation[:train_size],
        val=permutation[train_size : train_size + val_size],
        test=permutation[train_size + val_size :],
    )


def rotating_replay_indices(
    indices: list[int],
    count: int,
    epoch: int,
    seed: int,
) -> list[int]:
    if not indices or count <= 0:
        return []
    rng = random.Random(seed)
    order = list(indices)
    rng.shuffle(order)
    start = ((epoch - 1) * count) % len(order)
    return [order[(start + offset) % len(order)] for offset in range(count)]


def mixed_loader(
    target_dataset: RTMDataset,
    target_indices: list[int],
    replay_dataset: RTMDataset,
    replay_indices: list[int],
    batch_size: int,
    seed: int,
) -> torch.utils.data.DataLoader:
    entries = [
        ("target", target_dataset, index) for index in target_indices
    ] + [
        ("replay", replay_dataset, index) for index in replay_indices
    ]
    dataset = MixedDomainDataset(entries)
    return make_shape_loader(
        dataset,
        list(range(len(dataset))),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )


def domain_loader(
    dataset: RTMDataset,
    indices: list[int],
    batch_size: int,
    seed: int,
) -> torch.utils.data.DataLoader:
    return make_shape_loader(
        dataset,
        indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )


def build_model(
    checkpoint: dict,
    device: torch.device,
    shot_batch_size: int,
) -> tuple[RTMInvNet, argparse.Namespace]:
    saved_args = argparse.Namespace(**checkpoint.get("args", {}))
    model_min = float(getattr(saved_args, "model_min", 2.0))
    model_max = float(getattr(saved_args, "model_max", 10.0))
    RTMInvNet.MODEL_MIN = model_min
    RTMInvNet.MODEL_MAX = model_max
    RTMInvNet.MODEL_SCALE = model_max - model_min

    operator = DeepwaveClosedLoopRTM(
        shot_batch_size=shot_batch_size,
        device=device,
    )
    model = RTMInvNet(
        rtm_operator=operator,
        compute_synthetic_data=False,
        input_mode=getattr(saved_args, "model_input_mode", "m0_rtm"),
        unet_base_channels=int(getattr(saved_args, "unet_base_channels", 64)),
        unet_depth=int(getattr(saved_args, "unet_depth", 3)),
        num_stages=int(getattr(saved_args, "num_stages", 2)),
        update_backbone=getattr(saved_args, "update_backbone", "unet"),
        recompute_rtm_between_stages=bool(
            getattr(saved_args, "recompute_rtm_between_stages", True)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, saved_args


def make_model_loss(saved_args: argparse.Namespace) -> nn.Module:
    return PermittivityLoss(
        l1_weight=float(getattr(saved_args, "model_l1_weight", 0.5)),
        mse_weight=float(getattr(saved_args, "model_mse_weight", 0.5)),
        grad_weight=float(getattr(saved_args, "model_grad_weight", 0.5)),
        ssim_weight=float(getattr(saved_args, "model_ssim_weight", 0.1)),
        residual_weight_alpha=float(
            getattr(saved_args, "model_residual_weight_alpha", 1.0)
        ),
        residual_weight_max=float(
            getattr(saved_args, "model_residual_weight_max", 5.0)
        ),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    args: argparse.Namespace,
    saved_args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": {
                **vars(saved_args),
                "fine_tune": vars(args),
                "data_dir": str(args.target_data_dir),
                "replay_data_dir": str(args.replay_data_dir),
            },
            "epoch": epoch,
            **metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.replay_fraction < 1.0:
        raise ValueError("--replay-fraction must be between 0 and 1")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

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
    args.seed_environment = set_global_seed(
        seeds.training_seed, deterministic=args.deterministic
    )
    device = resolve_device(args.device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.run_dir / "train.log"
    log_fh = log_path.open("a")

    def log(message: str = "") -> None:
        print(message, flush=True)
        log_fh.write(message + "\n")
        log_fh.flush()

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, saved_args = build_model(checkpoint, device, args.rtm_shot_batch_size)
    model_loss = make_model_loss(saved_args)
    stage1_alpha = float(getattr(saved_args, "stage1_target_alpha", 0.5))

    target_dataset = RTMDataset(args.target_data_dir, limit=args.target_limit)
    replay_dataset = RTMDataset(args.replay_data_dir, limit=args.replay_limit)
    target_split = split_indices(
        len(target_dataset), args.train_split, args.val_split, seeds.split_seed
    )
    replay_split = split_indices(
        len(replay_dataset), args.train_split, args.val_split, seeds.split_seed
    )
    replay_per_epoch = int(
        math.ceil(
            len(target_split.train)
            * args.replay_fraction
            / (1.0 - args.replay_fraction)
        )
    )

    target_val_loader = domain_loader(
        target_dataset, target_split.val, args.batch_size, seeds.sampler_seed
    )
    replay_val_loader = domain_loader(
        replay_dataset, replay_split.val, args.batch_size, seeds.sampler_seed
    )
    target_test_loader = domain_loader(
        target_dataset, target_split.test, args.batch_size, seeds.sampler_seed
    )
    replay_test_loader = domain_loader(
        replay_dataset, replay_split.test, args.batch_size, seeds.sampler_seed
    )

    if args.evaluate_only:
        log("# Dual-domain validation/test evaluation only")
        evaluation: dict[str, float | int | str] = {
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        }
        for label, loader in [
            ("target_val", target_val_loader),
            ("replay_val", replay_val_loader),
            ("target_test", target_test_loader),
            ("replay_test", replay_test_loader),
        ]:
            loss, _ = run_epoch(
                model,
                loader,
                optimizer=None,
                device=device,
                model_loss_fn=model_loss,
                stage1_target_alpha=stage1_alpha,
                label=label,
                log_fn=log,
                progress_every=args.progress_every,
            )
            evaluation[f"{label}_loss"] = loss
        (args.run_dir / "final_evaluation.json").write_text(
            json.dumps(evaluation, indent=2),
            encoding="utf-8",
        )
        log(json.dumps(evaluation, indent=2))
        log_fh.close()
        return

    config_payload = {
        **vars(args),
        "target_size": len(target_dataset),
        "replay_size": len(replay_dataset),
        "target_split_sizes": {
            "train": len(target_split.train),
            "val": len(target_split.val),
            "test": len(target_split.test),
        },
        "replay_split_sizes": {
            "train": len(replay_split.train),
            "val": len(replay_split.val),
            "test": len(replay_split.test),
        },
        "replay_samples_per_epoch": replay_per_epoch,
        "pretrained_epoch": checkpoint.get("epoch"),
        "pretrained_val_loss": checkpoint.get("val_loss"),
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, default=str)
    )

    log("# Sandbox domain fine-tuning with rotating rock replay")
    log(f"# checkpoint={args.checkpoint}")
    log(f"# device={device} epochs={args.epochs} lr={args.learning_rate:g}")
    log(
        f"# target={len(target_dataset)} split="
        f"{len(target_split.train)}/{len(target_split.val)}/{len(target_split.test)}"
    )
    log(
        f"# replay={len(replay_dataset)} split="
        f"{len(replay_split.train)}/{len(replay_split.val)}/{len(replay_split.test)} "
        f"per_epoch={replay_per_epoch}"
    )
    actual_replay_fraction = replay_per_epoch / (
        len(target_split.train) + replay_per_epoch
    )
    log(f"# actual_replay_fraction={actual_replay_fraction:.4f}")

    log("Evaluating pretrained baseline on both validation domains...")
    baseline_target_val, _ = run_epoch(
        model,
        target_val_loader,
        optimizer=None,
        device=device,
        model_loss_fn=model_loss,
        stage1_target_alpha=stage1_alpha,
        label="target_baseline",
        log_fn=log,
        progress_every=args.progress_every,
    )
    baseline_replay_val, _ = run_epoch(
        model,
        replay_val_loader,
        optimizer=None,
        device=device,
        model_loss_fn=model_loss,
        stage1_target_alpha=stage1_alpha,
        label="replay_baseline",
        log_fn=log,
        progress_every=args.progress_every,
    )
    replay_limit = baseline_replay_val * (1.0 + args.forgetting_tolerance)
    log(
        f"baseline target_val={baseline_target_val:.6f} "
        f"replay_val={baseline_replay_val:.6f} replay_limit={replay_limit:.6f}"
    )

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=max(args.learning_rate * 0.05, 1e-7),
    )

    csv_path = args.run_dir / "loss_log.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "target_val_loss",
                "replay_val_loss",
                "replay_limit",
                "eligible",
                "learning_rate",
            ]
        )

    best_target_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_replay = rotating_replay_indices(
            replay_split.train,
            replay_per_epoch,
            epoch,
            seeds.sampler_seed,
        )
        train_loader = mixed_loader(
            target_dataset,
            target_split.train,
            replay_dataset,
            epoch_replay,
            args.batch_size,
            seeds.sampler_seed,
        )
        # Order is a pure function of (sampler_seed, epoch); pin the absolute epoch
        # so the source-domain replay rotation and the batch order stay reproducible.
        set_loader_epoch(train_loader, epoch)
        log(
            f"\nEpoch {epoch:03d}/{args.epochs:03d} "
            f"target={len(target_split.train)} replay={len(epoch_replay)}"
        )
        train_loss, _ = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            model_loss_fn=model_loss,
            stage1_target_alpha=stage1_alpha,
            label="train",
            log_fn=log,
            progress_every=args.progress_every,
        )
        target_val, _ = run_epoch(
            model,
            target_val_loader,
            optimizer=None,
            device=device,
            model_loss_fn=model_loss,
            stage1_target_alpha=stage1_alpha,
            label="target_val",
            log_fn=log,
            progress_every=args.progress_every,
        )
        replay_val, _ = run_epoch(
            model,
            replay_val_loader,
            optimizer=None,
            device=device,
            model_loss_fn=model_loss,
            stage1_target_alpha=stage1_alpha,
            label="replay_val",
            log_fn=log,
            progress_every=args.progress_every,
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        eligible = replay_val <= replay_limit
        improved = eligible and target_val < best_target_val - args.min_delta
        metrics = {
            "train_loss": train_loss,
            "target_val_loss": target_val,
            "replay_val_loss": replay_val,
            "baseline_target_val_loss": baseline_target_val,
            "baseline_replay_val_loss": baseline_replay_val,
            "replay_val_limit": replay_limit,
            "eligible": eligible,
        }
        save_checkpoint(
            args.run_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            args,
            saved_args,
            epoch,
            metrics,
        )
        if improved:
            best_target_val = target_val
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            save_checkpoint(
                args.run_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                args,
                saved_args,
                epoch,
                metrics,
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        with csv_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{target_val:.8f}",
                    f"{replay_val:.8f}",
                    f"{replay_limit:.8f}",
                    int(eligible),
                    f"{current_lr:.8g}",
                ]
            )
        log(
            f"epoch={epoch:03d} train={train_loss:.6f} "
            f"target_val={target_val:.6f} replay_val={replay_val:.6f} "
            f"eligible={eligible} lr={current_lr:.3g} "
            f"elapsed={time.perf_counter() - started:.1f}s"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            log(
                f"early_stopping epoch={epoch:03d} "
                f"patience={args.early_stopping_patience}"
            )
            break

    if best_state is None:
        log(
            "No epoch satisfied the replay constraint; retaining the pretrained "
            "checkpoint as best.pt."
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        fallback_metrics = {
            "target_val_loss": baseline_target_val,
            "replay_val_loss": baseline_replay_val,
            "baseline_target_val_loss": baseline_target_val,
            "baseline_replay_val_loss": baseline_replay_val,
            "replay_val_limit": replay_limit,
            "eligible": True,
            "fallback_to_pretrained": True,
        }
        fallback_optimizer = Adam(model.parameters(), lr=args.learning_rate)
        fallback_scheduler = CosineAnnealingLR(fallback_optimizer, T_max=args.epochs)
        save_checkpoint(
            args.run_dir / "best.pt",
            model,
            fallback_optimizer,
            fallback_scheduler,
            args,
            saved_args,
            0,
            fallback_metrics,
        )
    else:
        model.load_state_dict(best_state, strict=True)

    log("\nFinal tests using the best replay-safe checkpoint...")
    target_test, _ = run_epoch(
        model,
        target_test_loader,
        optimizer=None,
        device=device,
        model_loss_fn=model_loss,
        stage1_target_alpha=stage1_alpha,
        label="target_test",
        log_fn=log,
        progress_every=args.progress_every,
    )
    replay_test, _ = run_epoch(
        model,
        replay_test_loader,
        optimizer=None,
        device=device,
        model_loss_fn=model_loss,
        stage1_target_alpha=stage1_alpha,
        label="replay_test",
        log_fn=log,
        progress_every=args.progress_every,
    )
    log(
        f"final target_test={target_test:.6f} replay_test={replay_test:.6f} "
        f"wall={time.perf_counter() - started:.1f}s"
    )
    log_fh.close()


if __name__ == "__main__":
    main()
