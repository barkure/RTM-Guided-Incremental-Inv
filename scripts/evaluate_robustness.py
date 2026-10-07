from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from deepwave import scalar
from torch.utils.data import Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate import (  # noqa: E402
    RTMInvNet,
    build_model,
    checkpoint_args,
    compute_metric_row,
    plot_prediction,
    resolve_initialization,
    split_identity,
    write_evaluated_ids,
    write_split_ids,
)
from rtm_inv import RTMDataset, normalise_bscan, preprocess_bscan  # noqa: E402
from rtm_inv.initialization import MODE_TRAIN_GLOBAL_CONSTANT  # noqa: E402
from rtm_inv.gpr import GPRSurveyConfig, build_geometry, model_to_velocity  # noqa: E402
from rtm_inv.protocol import (  # noqa: E402
    SPLIT_IDENTITY_VERSION,
    MetricProtocol,
    SplitIdentity,
    aggregate_metric_rows,
    build_evaluation_coverage,
    git_revision,
    metric_fieldnames,
    parameter_counts,
    resolve_seeds,
)


@dataclass(frozen=True)
class Condition:
    name: str
    kind: str
    severity: float = 0.0
    width: int = 0
    height_m: float = 0.0
    initial_scale: float = 1.0
    freq_factor: float = 1.0
    shift_samples: int = 0
    atten_alpha_t: float = 0.0
    source_kind: str = ""


CONDITIONS = [
    Condition("clean", "clean"),
    Condition("snr30db", "noise", severity=30.0),
    Condition("snr25db", "noise", severity=25.0),
    Condition("snr20db", "noise", severity=20.0),
    Condition("snr15db", "noise", severity=15.0),
    # Main missing-trace experiment: traces are declared missing in the RAW record,
    # excluded from background and amplitude estimation, and zero-filled only after
    # the valid traces have been preprocessed.
    Condition("missing5_raw", "missing_raw", width=5),
    Condition("missing10_raw", "missing_raw", width=10),
    Condition("missing15_raw", "missing_raw", width=15),
    Condition("missing20_raw", "missing_raw", width=20),
    # Labelled control: the historical implementation, which zeroes the already
    # preprocessed and normalised tensor in place. Never pooled with the raw runs.
    Condition("missing5_post", "missing_post", width=5),
    Condition("missing10_post", "missing_post", width=10),
    Condition("missing15_post", "missing_post", width=15),
    Condition("missing20_post", "missing_post", width=20),
    Condition("liftoff_2p5cm", "liftoff", width=20, height_m=0.025),
    Condition("liftoff_5cm", "liftoff", width=20, height_m=0.050),
    Condition("liftoff_10cm", "liftoff", width=20, height_m=0.100),
    Condition("liftoff_20cm", "liftoff", width=20, height_m=0.200),
    Condition("initial_scale_0p8", "initial_model", initial_scale=0.8),
    Condition("initial_scale_0p9", "initial_model", initial_scale=0.9),
    Condition("initial_scale_1p1", "initial_model", initial_scale=1.1),
    Condition("initial_scale_1p2", "initial_model", initial_scale=1.2),
    # RTM centre-frequency mismatch: only the operator's wavelet changes. The observed
    # B-scan and the initial model stay bit-identical (gate 1).
    Condition("freq_m20pct", "frequency", freq_factor=0.80),
    Condition("freq_m10pct", "frequency", freq_factor=0.90),
    Condition("freq_m5pct", "frequency", freq_factor=0.95),
    Condition("freq_p5pct", "frequency", freq_factor=1.05),
    Condition("freq_p10pct", "frequency", freq_factor=1.10),
    Condition("freq_p20pct", "frequency", freq_factor=1.20),
    Condition("timezero_m10", "time_zero", shift_samples=-10),
    Condition("timezero_m5", "time_zero", shift_samples=-5),
    Condition("timezero_m2", "time_zero", shift_samples=-2),
    Condition("timezero_p2", "time_zero", shift_samples=2),
    Condition("timezero_p5", "time_zero", shift_samples=5),
    Condition("timezero_p10", "time_zero", shift_samples=10),
    # Empirical attenuation (NOT conductivity forward modelling): a two-way-travel-time
    # amplitude decay exp(-alpha*t) with alpha*T_max = 1, 2. Lossless-scalar Deepwave has
    # no conductivity/attenuation/Q argument, so this is the only defensible "lossy
    # medium" mismatch: amplitude decay only, no dispersion, arrival times unchanged.
    Condition("atten_e1", "attenuation", atten_alpha_t=1.0),
    Condition("atten_e2", "attenuation", atten_alpha_t=2.0),
    # Source / antenna response mismatch: injected on the OBSERVED traces, unlike the
    # centre-frequency condition which only touches the RTM operator.
    Condition("src_bw_50", "source", source_kind="bw_50"),
    Condition("src_phase_90", "source", source_kind="phase_90"),
]


def condition_directory_name(condition: Condition) -> str:
    if condition.kind == "clean":
        return "Clean"
    if condition.kind == "noise":
        return f"SNR {condition.severity:g} dB"
    if condition.kind == "missing_raw":
        return f"Missing {condition.width} traces (raw)"
    if condition.kind == "missing_post":
        return f"Missing {condition.width} traces (post-processing control)"
    if condition.kind == "liftoff":
        return f"Lift-off {condition.height_m * 100:g} cm"
    if condition.kind == "initial_model":
        return f"Initial model {condition.initial_scale:g}x"
    if condition.kind == "frequency":
        return f"RTM frequency {(condition.freq_factor - 1.0) * 100:+.0f}%"
    if condition.kind == "time_zero":
        return f"Time-zero {condition.shift_samples:+d} samples"
    if condition.kind == "attenuation":
        return f"Empirical attenuation alpha*Tmax={condition.atten_alpha_t:g}"
    if condition.kind == "source":
        return f"Source mismatch {condition.source_kind}"
    raise ValueError(f"Unknown condition kind: {condition.kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate acquisition and initial-model robustness."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--perturbation-seed", type=int, default=20260712)
    parser.add_argument("--plot-samples", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--initial-model-mode",
        default=None,
        help=(
            "Override the initial-model recipe. Defaults to the checkpoint's own "
            "mode; stage-5 runs must resolve to train_global_constant."
        ),
    )
    parser.add_argument(
        "--allow-non-realistic",
        action="store_true",
        help=(
            "Permit an initial_model_mode other than train_global_constant. Only for a "
            "deliberate oracle-prior comparison; stage-5 results require the realistic "
            "background the checkpoints were trained with."
        ),
    )
    parser.add_argument(
        "--gate-check",
        action="store_true",
        help=(
            "Run the stage-5 verification gates on a few samples instead of the full "
            "evaluation, and stop before writing results."
        ),
    )
    parser.add_argument(
        "--gate-samples",
        type=int,
        default=3,
        help="How many samples the gate checks use.",
    )
    parser.add_argument(
        "--gate-reference-metrics",
        default=None,
        help=(
            "metrics.csv from the accepted clean evaluation; gate 0 reproduces it "
            "sample by sample to prove the harness still evaluates the same "
            "configuration that produced the accepted numbers."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=[condition.name for condition in CONDITIONS],
        default=None,
        help="Evaluate only the selected conditions (default: all).",
    )
    return parser.parse_args()


def sample_seed(sample_id: str, base_seed: int, salt: int) -> int:
    try:
        number = int(sample_id)
    except ValueError:
        number = sum((index + 1) * ord(char) for index, char in enumerate(sample_id))
    return int(base_seed + 1009 * number + 7919 * salt)


def split_subset(dataset: RTMDataset, saved_args: SimpleNamespace) -> tuple[Subset, tuple[int, int, int], SplitIdentity]:
    identity, notes = split_identity(dataset, saved_args)
    for note in notes:
        print(f"split_note: {note}", flush=True)
    return Subset(dataset, list(identity.test_indices)), identity.spec.sizes, identity


def segment_weight(n_shots: int, width: int, start: int, edge_width: int = 4) -> torch.Tensor:
    weight = torch.zeros(n_shots)
    end = start + width
    weight[start:end] = 1.0
    edge = min(edge_width, width // 2)
    if edge:
        ramp = torch.linspace(0.0, 1.0, edge + 2)[1:-1]
        weight[start : start + edge] = ramp
        weight[end - edge : end] = torch.flip(ramp, dims=[0])
    return weight


def config_from_tensor(values: torch.Tensor) -> GPRSurveyConfig:
    flat = values.detach().cpu().flatten().tolist()
    if len(flat) < 11:
        raise ValueError(f"Expected 11 RTM config values, got {len(flat)}")
    return GPRSurveyConfig(
        dx=float(flat[0]),
        dz=float(flat[1]),
        dt=float(flat[2]),
        freq=float(flat[3]),
        src_x_start=float(flat[4]),
        src_x_step=float(flat[5]),
        src_z=float(flat[6]),
        rec_offset_x=float(flat[7]),
        rec_z=float(flat[8]),
        pml_width=int(round(float(flat[9]))),
        accuracy=int(round(float(flat[10]))),
        scale=1e6,
        model_kind="permittivity",
        bscan_layout="trace_by_time",
    )


def forward_liftoff_segment(
    target_normalized: torch.Tensor,
    raw_clean: torch.Tensor,
    config_values: torch.Tensor,
    start: int,
    width: int,
    height_m: float,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Forward model all traces with an air layer and a local lift-off segment.

    The source and receiver follow the same smooth height profile. Outside the
    selected segment they retain the survey's nominal acquisition geometry.
    """
    config = config_from_tensor(config_values)
    target = denormalise(target_normalized.squeeze(0)).to(device)
    n_shots, _n_receivers, nt = raw_clean.shape
    nz, nx = target.shape

    max_height_cells = max(1, int(round(height_m / config.dz)))
    air_cells = max_height_cells + 2
    air = torch.ones((air_cells, nx), dtype=target.dtype, device=device)
    velocity = model_to_velocity(torch.cat((air, target), dim=0), config)

    source_amplitudes, source_locations, receiver_locations = build_geometry(
        n_shots, nt, nx, nz + air_cells, config, device
    )
    profile = segment_weight(n_shots, width, start).to(device)
    source_nominal = air_cells + int(round(config.src_z / config.dz))
    receiver_nominal = air_cells + int(round(config.rec_z / config.dz))
    lifted_z = air_cells - max_height_cells
    source_z = source_nominal + profile * (lifted_z - source_nominal)
    receiver_z = receiver_nominal + profile * (lifted_z - receiver_nominal)
    source_locations[:, 0, 0] = torch.round(source_z).to(torch.long)
    receiver_locations[:, 0, 0] = torch.round(receiver_z).to(torch.long)

    simulated = scalar(
        velocity,
        [config.dz, config.dx],
        dt=config.dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        pml_width=[config.pml_width] * 4,
        pml_freq=config.freq,
        accuracy=config.accuracy,
    )[-1]
    return simulated.cpu(), air_cells


def preprocess_bscan_masked(
    raw: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    quantile: float = 0.95,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess + normalise a B-scan while EXCLUDING declared-missing traces.

    ``raw`` is ``(shot, receiver, time)``; ``valid_mask`` is ``(shot,)`` boolean.
    Both estimators are the ones ``preprocess_bscan``/``robust_bscan_scale`` use
    (horizontal median trace, q-quantile of |amplitude|), but computed over valid
    traces only, so a missing trace cannot leak the reflectivity it would have
    carried into another trace's background or into the global amplitude scale.
    Missing traces are zero-filled *after* the valid traces are processed.

    Returns ``(normalised, scale, background)``; the latter two are returned so the
    gate checks can prove the estimators ignored the missing traces.
    """
    if raw.ndim != 3:
        raise ValueError(f"Expected a 3D B-scan, got {tuple(raw.shape)}")
    keep = valid_mask.to(dtype=torch.bool).view(-1, 1, 1)
    if keep.shape[0] != raw.shape[0]:
        raise ValueError(
            f"valid_mask has {keep.shape[0]} entries for {raw.shape[0]} traces"
        )
    if not bool(keep.any()):
        raise ValueError("every trace is marked missing; a background cannot be estimated")

    centred = raw.float() - raw.float().mean(dim=-1, keepdim=True)
    masked = centred.masked_fill(~keep, float("nan"))
    background = torch.nanmedian(masked, dim=0, keepdim=True).values
    residual = centred - background
    valid_values = residual.masked_fill(~keep, float("nan")).abs()
    scale = torch.nanquantile(valid_values, float(quantile), dim=None)
    scale = torch.as_tensor(scale).clamp_min(float(eps))
    normalised = residual / scale
    normalised = normalised.masked_fill(~keep, 0.0)
    return normalised, scale.expand(()), background


def shift_traces_in_time(raw: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift every trace along the time axis with zero padding (no wraparound)."""
    if shift == 0:
        return raw
    out = torch.zeros_like(raw)
    if shift > 0:
        if shift < raw.shape[-1]:
            out[..., shift:] = raw[..., :-shift]
    elif -shift < raw.shape[-1]:
        out[..., :shift] = raw[..., -shift:]
    return out


def atten_sigmoid_wavelet(n: int, f0: float, dt: float, device: torch.device) -> torch.Tensor:
    """Ricker-like reference wavelet used only for bandwidth bookkeeping."""
    t = (torch.arange(n, device=device, dtype=torch.float32) - (n - 1) / 2.0) * dt
    arg = (torch.pi * f0 * t).square()
    return (1.0 - 2.0 * arg) * torch.exp(-arg)


def apply_empirical_attenuation(raw: torch.Tensor, alpha_t: float) -> torch.Tensor:
    """Two-way-travel-time amplitude decay ``exp(-alpha * t)`` with ``alpha*T_max`` set.

    Amplitude decay only: no dispersion, so arrival times and the wavespeed are
    untouched. This is NOT conductivity forward modelling (see the frozen design).
    """
    n = raw.shape[-1]
    ramp = torch.linspace(0.0, 1.0, n, device=raw.device, dtype=torch.float32)
    decay = torch.exp(-float(alpha_t) * ramp)
    return raw * decay.view(1, 1, -1)


def apply_source_mismatch(raw: torch.Tensor, kind: str, dt: float) -> torch.Tensor:
    """Time-domain antenna-response filter applied to the OBSERVED traces.

    ``bw_50`` halves the bandwidth with a zero-phase low-pass; ``phase_90`` rotates
    phase by 90 degrees and leaves the amplitude spectrum unchanged.
    """
    spectrum = torch.fft.rfft(raw, dim=-1)
    freqs = torch.fft.rfftfreq(raw.shape[-1], d=float(dt), device=raw.device)
    f0 = 400e6 / 1e6
    if kind == "bw_50":
        taper = torch.exp(-0.5 * (freqs / (0.5 * f0)).square())
        return torch.fft.irfft(spectrum * taper, n=raw.shape[-1], dim=-1)
    if kind == "phase_90":
        return torch.fft.irfft(spectrum * torch.tensor(-1j, dtype=spectrum.dtype), n=raw.shape[-1], dim=-1)
    raise ValueError(f"Unknown source mismatch kind: {kind}")


def perturb(
    batch: dict,
    condition: Condition,
    base_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float | int], dict[str, float] | None]:
    sample_id = str(batch["sample_id"])
    clean = batch["bscan"].float()
    raw = batch["bscan_raw"].float()
    metadata: dict[str, float | int] = {}
    if condition.kind == "clean":
        return clean, metadata, None
    if condition.kind == "initial_model":
        metadata["initial_model_scale"] = condition.initial_scale
        return clean, metadata, None
    if condition.kind == "frequency":
        # The RTM operator's wavelet frequency is the ONLY thing that changes; the
        # observation and the initial model are returned untouched (gate 1).
        base_freq = float(batch["rtm_config"].flatten()[3])
        metadata.update(
            {
                "base_frequency": base_freq,
                "frequency": base_freq * condition.freq_factor,
                "frequency_factor": condition.freq_factor,
            }
        )
        return clean, metadata, {"freq": base_freq * condition.freq_factor}
    if condition.kind == "noise":
        signal_rms = raw.square().mean().sqrt().clamp_min(1e-12)
        noise_rms = signal_rms / (10.0 ** (condition.severity / 20.0))
        generator = torch.Generator().manual_seed(sample_seed(sample_id, base_seed, 1))
        noise = torch.randn(raw.shape, generator=generator, dtype=raw.dtype) * noise_rms
        processed = preprocess_bscan(raw + noise)
        normalized, _scale = normalise_bscan(processed, quantile=0.95)
        metadata["target_snr_db"] = condition.severity
        return normalized, metadata, None
    if condition.kind == "attenuation":
        decayed = apply_empirical_attenuation(raw, condition.atten_alpha_t)
        processed = preprocess_bscan(decayed)
        normalized, _scale = normalise_bscan(processed, quantile=0.95)
        metadata.update(
            {
                "atten_alpha_t": condition.atten_alpha_t,
                "atten_model": "empirical_two_way_amplitude_decay",
                "atten_dispersive": 0,
                "atten_is_conductivity_forward": 0,
            }
        )
        return normalized, metadata, None
    if condition.kind == "source":
        trace_dt = float(batch["rtm_config"].flatten()[2])
        filtered = apply_source_mismatch(raw, condition.source_kind, dt=trace_dt)
        processed = preprocess_bscan(filtered)
        normalized, _scale = normalise_bscan(processed, quantile=0.95)
        metadata.update({"source_kind": condition.source_kind, "dt_s": trace_dt})
        return normalized, metadata, None
    generator = torch.Generator().manual_seed(sample_seed(sample_id, base_seed, 2))
    start = int(torch.randint(0, clean.shape[0] - condition.width + 1, (1,), generator=generator))
    metadata.update({"segment_start": start, "segment_width": condition.width})
    if condition.kind == "missing_post":
        # Labelled control: the historical implementation zeroes the already
        # preprocessed, normalised tensor in place.
        missing = clean.clone()
        missing[start : start + condition.width] = 0.0
        metadata["injection_stage"] = "post_processing"
        return missing, metadata, None
    if condition.kind == "missing_raw":
        valid = torch.ones(raw.shape[0], dtype=torch.bool)
        valid[start : start + condition.width] = False
        normalized, _scale, _background = preprocess_bscan_masked(raw, valid)
        metadata.update(
            {
                "injection_stage": "raw",
                "valid_traces": int(valid.sum()),
                "excluded_from_estimators": condition.width,
            }
        )
        return normalized, metadata, None
    if condition.kind == "time_zero":
        shifted = shift_traces_in_time(raw, condition.shift_samples)
        processed = preprocess_bscan(shifted)
        normalized, _scale = normalise_bscan(processed, quantile=0.95)
        metadata.update(
            {"shift_samples": condition.shift_samples, "shift_dt_s": 1.0e-4, "padding": "zeros"}
        )
        return normalized, metadata, None
    if condition.kind == "liftoff":
        lifted_raw, air_cells = forward_liftoff_segment(
            batch["target_model"],
            raw,
            batch["rtm_config"],
            start,
            condition.width,
            condition.height_m,
            device,
        )
        processed = preprocess_bscan(lifted_raw)
        normalized, _scale = normalise_bscan(processed, quantile=0.95)
        metadata.update(
            {"height_m": condition.height_m, "air_cells": air_cells, "forward_shots": raw.shape[0]}
        )
        return normalized, metadata, None
    raise ValueError(f"Unknown condition kind: {condition.kind}")


def _condition_by_name(name: str) -> Condition:
    for condition in CONDITIONS:
        if condition.name == name:
            return condition
    raise ValueError(f"unknown condition {name!r}")


def _forward(model, batch, condition, config_override, device):
    observed, metadata, _ = perturb(batch, condition, 20260712, device)
    observed = observed.unsqueeze(0).to(device)
    initial = batch["initial_model"].unsqueeze(0).to(device)
    target = batch["target_model"].unsqueeze(0).to(device)
    config = batch["rtm_config"].unsqueeze(0).to(device)
    if config_override:
        config = config.clone()
        config[..., 3] = float(config_override["freq"])
    processed = batch["observed_is_processed"].unsqueeze(0).to(device)
    outputs = model(observed, initial, sample_config=config, observed_is_processed=processed)
    return observed, initial, outputs, metadata


def run_gate_checks(
    model,
    subset,
    protocol: MetricProtocol,
    initializer,
    reference_metrics: Path | None,
    device: torch.device,
    samples: int = 3,
) -> dict[str, str]:
    """Stage-5 verification gates. Any failure stops the run before full evaluation."""
    import csv as _csv

    results: dict[str, str] = {}
    batches = [subset[i] for i in range(min(samples, len(subset)))]

    def check(name: str, ok: bool, detail: str) -> None:
        results[name] = "PASS" if ok else f"FAIL ({detail})"
        print(f"gate {name}: {results[name]} | {detail}", flush=True)

    # Gate 0 -- the initialisation really is the checkpoint's realistic one, and the
    # clean condition reproduces the accepted clean evaluation sample by sample.
    check(
        "0a_initial_model_mode",
        initializer is not None and initializer.is_realistic,
        f"initializer={None if initializer is None else initializer.mode} "
        f"c_train={None if initializer is None else getattr(initializer, 'train_constant', None)}",
    )
    if reference_metrics is not None and reference_metrics.exists():
        reference = {
            row["sample_id"]: row
            for row in _csv.DictReader(reference_metrics.open())
        }
        worst = 0.0
        worst_metric = ""
        missing_ids = []
        with torch.no_grad():
            for batch in batches:
                sample_id = str(batch["sample_id"])
                if sample_id not in reference:
                    missing_ids.append(sample_id)
                    continue
                observed, initial, outputs, _ = _forward(
                    model, batch, _condition_by_name("clean"), None, device
                )
                row = compute_metric_row(
                    "final",
                    outputs["final_model"][0, 0],
                    batch["target_model"][0].to(device),
                    protocol,
                )
                for key, value in row.items():
                    if key in reference[sample_id]:
                        diff = abs(float(value) - float(reference[sample_id][key]))
                        if diff > worst:
                            worst, worst_metric = diff, key
        check(
            "0b_clean_reproduction",
            worst == 0.0 and not missing_ids,
            f"max|diff|={worst:.3g} (worst metric={worst_metric}); missing_ids={missing_ids}",
        )
    else:
        check("0b_clean_reproduction", False, f"no reference metrics at {reference_metrics}")

    batch = batches[0]
    with torch.no_grad():
        clean_observed, clean_initial, clean_outputs, _ = _forward(
            model, batch, _condition_by_name("clean"), None, device
        )

        # Gate 1 -- centre-frequency mismatch may change ONLY the RTM operator.
        freq_condition = _condition_by_name("freq_p20pct")
        base_freq = float(batch["rtm_config"].flatten()[3])
        freq_observed, freq_initial, freq_outputs, freq_meta = _forward(
            model, batch, freq_condition, {"freq": base_freq * freq_condition.freq_factor}, device
        )
        obs_same = torch.equal(clean_observed, freq_observed)
        init_same = torch.equal(clean_initial, freq_initial)
        rtm_differs = not torch.equal(
            clean_outputs["final_model"], freq_outputs["final_model"]
        )
        check(
            "1_frequency_only_rtm",
            obs_same and init_same and rtm_differs,
            f"observed_identical={obs_same} initial_identical={init_same} "
            f"operator_effect_seen={rtm_differs}",
        )

        # Gate 2 -- time-zero shifts only the time axis and pads with information-free
        # zeros: shifting back reproduces the raw record away from the padded band, and
        # the band itself is exactly zero (no wraparound, no invented samples).
        raw = batch["bscan_raw"].float()
        shift = 5
        round_trip = shift_traces_in_time(shift_traces_in_time(raw, shift), -shift)
        inside = round_trip[..., shift:-shift]
        hit = torch.equal(inside, raw[..., shift:-shift])
        # +shift pads the first `shift` samples; -shift pads the last `shift` samples.
        delayed = shift_traces_in_time(raw, shift)
        advanced = shift_traces_in_time(raw, -shift)
        head_zero = bool((delayed[..., :shift] == 0).all())
        tail_zero = bool((advanced[..., -shift:] == 0).all())
        preserved_head = torch.equal(delayed[..., shift:], raw[..., :-shift])
        preserved_tail = torch.equal(advanced[..., :-shift], raw[..., shift:])
        check(
            "2_timezero_padding",
            hit and head_zero and tail_zero and preserved_head and preserved_tail,
            f"round_trip_identical_inside={hit} head_pad_zero={head_zero} "
            f"tail_pad_zero={tail_zero} delay_preserves_signal={preserved_head} "
            f"advance_preserves_signal={preserved_tail}",
        )

        # Gate 3 -- raw-injected missing traces bypass BOTH estimators. Dropping the
        # trace from the input entirely must give the same background and scale as
        # masking it, and the result must differ from the post-processing control.
        width = 5
        start = 7
        valid = torch.ones(raw.shape[0], dtype=torch.bool)
        valid[start : start + width] = False
        masked, masked_scale, masked_bg = preprocess_bscan_masked(raw, valid)
        kept = raw[valid]
        dropped, dropped_scale, dropped_bg = preprocess_bscan_masked(
            kept, torch.ones(kept.shape[0], dtype=torch.bool)
        )
        same_estimators = torch.equal(masked_scale, dropped_scale) and torch.equal(
            masked_bg, dropped_bg
        )
        same_valid_traces = torch.equal(masked[valid], dropped)
        post = normalise_bscan(preprocess_bscan(raw), quantile=0.95)[0].clone()
        post[start : start + width] = 0.0
        differs_from_control = not torch.equal(masked, post)
        check(
            "3_missing_raw_bypasses_estimators",
            same_estimators and same_valid_traces and differs_from_control,
            f"scale_and_background_match_dropped={same_estimators} "
            f"valid_traces_match_dropped={same_valid_traces} "
            f"differs_from_post_control={differs_from_control}",
        )

        # Gate 4 -- the post-processing control still reproduces the historical path.
        control_observed, _, _, _ = _forward(
            model, batch, _condition_by_name("missing5_post"), None, device
        )
        expected = batch["bscan"].float().clone().to(device)
        generator = torch.Generator().manual_seed(sample_seed(str(batch["sample_id"]), 20260712, 2))
        expected_start = int(
            torch.randint(0, expected.shape[0] - width + 1, (1,), generator=generator)
        )
        expected[expected_start : expected_start + width] = 0.0
        check(
            "4_post_control_historical",
            torch.equal(control_observed[0], expected),
            f"zeroed_segment={expected_start}..{expected_start + width}",
        )

    # Gate 5 -- every repetition is auditable: each non-trivial condition records its
    # realisation (positions/levels/seeds), and that realisation depends only on
    # (perturbation_seed, sample_id), so all three methods see the SAME perturbation
    # for a given sample and the degradation is genuinely paired.
    required = {
        "missing_raw": {"segment_start", "segment_width", "injection_stage", "valid_traces"},
        "missing_post": {"segment_start", "segment_width", "injection_stage"},
        "liftoff": {"segment_start", "segment_width", "height_m", "air_cells"},
        "noise": {"target_snr_db"},
        "frequency": {"frequency", "frequency_factor"},
        "time_zero": {"shift_samples", "padding"},
        "attenuation": {"atten_alpha_t", "atten_model", "atten_is_conductivity_forward"},
        "source": {"source_kind", "dt_s"},
    }
    gaps: list[str] = []
    unstable: list[str] = []
    for condition in CONDITIONS:
        if condition.kind in ("clean", "initial_model"):
            continue
        first, meta_one, _ = perturb(batch, condition, 20260712, device)
        second, meta_two, _ = perturb(batch, condition, 20260712, device)
        missing = required.get(condition.kind, set()) - set(meta_one)
        if missing:
            gaps.append(f"{condition.name} lacks {sorted(missing)}")
        if meta_one != meta_two or not torch.equal(first, second):
            unstable.append(condition.name)
    check(
        "5_realisation_metadata",
        not gaps and not unstable,
        f"missing_fields={gaps} non_reproducible={unstable}",
    )

    return results


def denormalise(model: torch.Tensor) -> torch.Tensor:
    return model * RTMInvNet.MODEL_SCALE + RTMInvNet.MODEL_MIN


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint_args(checkpoint)
    RTMInvNet.MODEL_MIN = float(getattr(saved_args, "model_min", 2.0))
    RTMInvNet.MODEL_MAX = float(getattr(saved_args, "model_max", 10.0))
    RTMInvNet.MODEL_SCALE = RTMInvNet.MODEL_MAX - RTMInvNet.MODEL_MIN
    protocol = MetricProtocol(RTMInvNet.MODEL_MIN, RTMInvNet.MODEL_MAX)
    seeds = resolve_seeds(
        legacy_seed=getattr(saved_args, "seed", None),
        split_seed=getattr(saved_args, "split_seed", None),
        training_seed=getattr(saved_args, "training_seed", None),
        sampler_seed=getattr(saved_args, "sampler_seed", None),
    )
    for line in seeds.log_lines():
        print(line, flush=True)
    dataset = RTMDataset(
        args.data_dir,
        limit=getattr(saved_args, "limit", None),
        normalize_models=True,
        include_raw_bscan=True,
    )
    subset, split_sizes, identity = split_subset(dataset, saved_args)

    # The initial-model recipe MUST come from the checkpoint. RTMDataset defaults to
    # `oracle_mean`, so evaluating a `train_global_constant` checkpoint without this
    # step would silently swap in an oracle prior and make every robustness number
    # meaningless. Same precedence as evaluate.py: CLI, then checkpoint, then a loud
    # oracle fallback.
    initial_model_mode, initializer, init_notes = resolve_initialization(
        args, saved_args, dataset, identity
    )
    for note in init_notes:
        print(f"initialization_note: {note}", flush=True)
    if initial_model_mode != MODE_TRAIN_GLOBAL_CONSTANT and not args.allow_non_realistic:
        raise SystemExit(
            f"refusing to run: the checkpoint's initial_model_mode is "
            f"{initial_model_mode!r}, not {MODE_TRAIN_GLOBAL_CONSTANT!r}. Stage-5 "
            "robustness numbers are only meaningful under the realistic background "
            "that the checkpoints were trained with; pass --allow-non-realistic only "
            "for a deliberate oracle-prior comparison."
        )
    if initializer is not None:
        # Rebuild with the checkpoint's recipe but keep `include_raw_bscan=True`:
        # evaluate.py's build_dataset() does not request the raw trace, and the
        # raw-injected conditions need it.
        dataset = RTMDataset(
            args.data_dir,
            limit=getattr(saved_args, "limit", None),
            normalize_models=True,
            include_raw_bscan=True,
            initial_model_mode=initial_model_mode,
            initializer=initializer,
        )
        rebuilt_subset, rebuilt_sizes, rebuilt_identity = split_subset(dataset, saved_args)
        if (rebuilt_sizes, rebuilt_identity.split_sha256) != (
            split_sizes,
            identity.split_sha256,
        ):
            raise SystemExit(
                "refusing to run: rebuilding the dataset with the checkpoint's "
                "initialisation changed the split identity, so evaluation and "
                "training would not share a background or a test sequence."
            )
        subset = rebuilt_subset
    print(
        f"initial_model_mode={initial_model_mode} "
        f"realistic={bool(initializer and initializer.is_realistic)} "
        f"c_train={getattr(initializer, 'train_constant', None)}",
        flush=True,
    )

    print(
        f"split_identity: split_sizes={split_sizes} split_seed={identity.spec.split_seed} "
        f"test_sample_ids_sha256={identity.test_sample_ids_sha256[:16]} "
        f"split_sha256={identity.split_sha256[:16]}",
        flush=True,
    )
    model = build_model(SimpleNamespace(rtm_shot_batch_size=None), saved_args, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    total = len(subset) if args.max_samples is None else min(args.max_samples, len(subset))

    # --max-samples truncates the test split in order, so the evaluated identity is
    # the first `total` entries of the reconstructed test sequence.
    coverage = build_evaluation_coverage(
        identity,
        list(identity.test_sample_ids[:total]),
        max_samples=args.max_samples,
    )
    test_ids_path = args.output_dir / "test_split_sample_ids.csv"
    write_split_ids(test_ids_path, identity)
    evaluated_ids_path = args.output_dir / "evaluated_sample_ids.csv"
    evaluated_records = [
        (test_position, dataset_index, sample_id)
        for test_position, (dataset_index, sample_id) in enumerate(
            zip(identity.test_indices, identity.test_sample_ids, strict=True)
        )
    ][:total]
    write_evaluated_ids(evaluated_ids_path, evaluated_records)
    run_metadata = {
        **protocol.as_dict(),
        "label": args.label,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        # Stage 5's first gate: the initial-model recipe must be the checkpoint's own
        # realistic one, so the artifact records which recipe was actually used.
        "initialization_mode": initial_model_mode,
        "initialization_spec": (
            initializer.as_dict() if initializer is not None else None
        ),
        "split_identity_version": SPLIT_IDENTITY_VERSION,
        **identity.spec.as_dict(),
        **coverage.as_dict(),
        "test_split_sample_ids_path": str(test_ids_path),
        "evaluated_sample_ids_path": str(evaluated_ids_path),
        **seeds.as_dict(),
        **parameter_counts(model),
        **git_revision(PROJECT_ROOT),
    }
    print(
        f"coverage mode={coverage.evaluation_mode} "
        f"is_full_test_evaluation={coverage.is_full_test_evaluation} "
        f"evaluated_samples={coverage.evaluated_samples} "
        f"test_split_size={coverage.test_split_size}",
        flush=True,
    )

    selected_conditions = CONDITIONS
    if args.conditions is not None:
        selected = set(args.conditions)
        selected_conditions = [condition for condition in CONDITIONS if condition.name in selected]

    if args.gate_check:
        reference = Path(args.gate_reference_metrics) if args.gate_reference_metrics else None
        print("== stage-5 gate checks ==", flush=True)
        gate_results = run_gate_checks(
            model,
            subset,
            protocol,
            initializer,
            reference,
            device,
            samples=args.gate_samples,
        )
        failed = {k: v for k, v in gate_results.items() if not v.startswith("PASS")}
        if failed:
            raise SystemExit(
                "refusing to run full evaluation: gate(s) failed -> "
                + "; ".join(f"{k}: {v}" for k, v in failed.items())
            )
        print("gate summary: all checks PASS", flush=True)
        return

    for condition in selected_conditions:
        condition_dir = args.output_dir / condition_directory_name(condition)
        condition_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        print(f"== {args.label} {condition.name}: {total} samples on {device} ==", flush=True)
        with torch.no_grad():
            for position, batch in enumerate(subset):
                if position >= total:
                    break
                sample_id = str(batch["sample_id"])
                observed, metadata, config_override = perturb(
                    batch, condition, args.perturbation_seed, device
                )
                observed = observed.unsqueeze(0).to(device)
                initial = batch["initial_model"].unsqueeze(0).to(device)
                if condition.kind == "initial_model":
                    initial_er = torch.clamp(
                        denormalise(initial) * condition.initial_scale,
                        min=RTMInvNet.MODEL_MIN,
                        max=RTMInvNet.MODEL_MAX,
                    )
                    initial = (
                        (initial_er - RTMInvNet.MODEL_MIN)
                        / RTMInvNet.MODEL_SCALE
                    )
                target = batch["target_model"].unsqueeze(0).to(device)
                config = batch["rtm_config"].unsqueeze(0).to(device)
                if config_override:
                    # Only the RTM operator's inputs are touched here; `observed` and
                    # `initial` were already produced without the override.
                    overridden = config.clone()
                    for field, value in config_override.items():
                        if field == "freq":
                            overridden[..., 3] = float(value)
                        else:
                            raise ValueError(f"unsupported RTM config override: {field}")
                    config = overridden
                processed = batch["observed_is_processed"].unsqueeze(0).to(device)
                outputs = model(observed, initial, sample_config=config, observed_is_processed=processed)
                row: dict[str, str | float | int] = {"sample_id": sample_id, **metadata}
                row.update(compute_metric_row("initial", initial[0, 0], target[0, 0], protocol))
                row.update(
                    compute_metric_row(
                        "stage1", outputs["stage1_model"][0, 0], target[0, 0], protocol
                    )
                )
                row.update(
                    compute_metric_row("final", outputs["final_model"][0, 0], target[0, 0], protocol)
                )
                rows.append(row)
                if position < args.plot_samples:
                    plot_prediction(
                        condition_dir / f"sample_{sample_id}.png",
                        sample_id,
                        position,
                        getattr(saved_args, "model_input_mode", "m0_rtm"),
                        observed,
                        initial,
                        target,
                        outputs,
                    )
                if (position + 1) % 20 == 0 or position + 1 == total:
                    print(f"{condition.name}: {position + 1}/{total}", flush=True)
                del outputs, observed, initial, target, config, processed
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if len(rows) != coverage.evaluated_samples:
            raise RuntimeError(
                f"{condition.name} produced {len(rows)} rows but the planned "
                f"evaluated coverage is {coverage.evaluated_samples} samples."
            )
        metric_keys = metric_fieldnames(protocol)
        fieldnames = ["sample_id"] + [
            key for key in dict.fromkeys(key for row in rows for key in row) if key != "sample_id"
        ]
        with (condition_dir / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        summary: dict[str, object] = {
            "condition": condition.name,
            "perturbation_seed": args.perturbation_seed,
            "initial_model_scale": (
                condition.initial_scale if condition.kind == "initial_model" else 1.0
            ),
            **run_metadata,
        }
        summary.update(aggregate_metric_rows(rows, metric_keys))
        with (condition_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)

        # Each condition carries its own split/evaluated id lists, so the aggregator
        # can verify one condition's coverage without trusting its parent directory.
        write_split_ids(condition_dir / "test_split_sample_ids.csv", identity)
        write_evaluated_ids(condition_dir / "evaluated_sample_ids.csv", evaluated_records)

        # Gate 5: the realisation of every repetition is auditable after the fact, so
        # the per-condition metadata file records where and how strongly each sample
        # was perturbed, alongside the provenance needed to compare conditions.
        realisation_keys = [
            key
            for row in rows
            for key in row
            if key != "sample_id" and key not in metric_keys
        ]
        realisation_keys = list(dict.fromkeys(realisation_keys))
        condition_metadata = {
            "condition": condition.name,
            "kind": condition.kind,
            "directory": condition_dir.name,
            "perturbation_seed": args.perturbation_seed,
            "level": {
                "severity_db": condition.severity,
                "width": condition.width,
                "height_m": condition.height_m,
                "initial_scale": condition.initial_scale,
                "freq_factor": condition.freq_factor,
                "shift_samples": condition.shift_samples,
                "atten_alpha_t": condition.atten_alpha_t,
                "source_kind": condition.source_kind,
            },
            "realisation_fields": realisation_keys,
            "realisations": [
                {"sample_id": row["sample_id"], **{k: row[k] for k in realisation_keys if k in row}}
                for row in rows
            ],
            **run_metadata,
        }
        (condition_dir / "metadata.json").write_text(
            json.dumps(condition_metadata, indent=2, default=str), encoding="utf-8"
        )
        summaries.append(summary)

    combined = args.output_dir / "summary.csv"
    with combined.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2, default=str), encoding="utf-8"
    )
    print(f"combined_summary={combined}", flush=True)


if __name__ == "__main__":
    main()
