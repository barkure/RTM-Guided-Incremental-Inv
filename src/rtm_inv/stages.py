"""Stage supervision schedule for an arbitrary number of update stages.

Frozen by decision D1 (stage 6): for an ``N``-stage model the target of stage ``k`` is

    M_k* = M_0 + lambda_k * (M_true - M_0),
    lambda_k = alpha + (1 - alpha) * (k - 1) / (N - 1),   k = 1..N

with ``alpha = 0.5`` by default, so the two-stage model keeps exactly the existing
0.5 / 1.0 targets and three- and four-stage models insert evenly spaced targets
between them. The stage-loss weights keep the existing convention: the non-final
stages share ``NON_FINAL_TOTAL_WEIGHT`` (0.3) evenly and the final stage carries
``FINAL_WEIGHT`` (0.7), so ``N = 2`` reproduces the historical 0.3 / 0.7 exactly.

The data-space (B-scan) loss stays on the final stage only, as before.
"""

from __future__ import annotations

DEFAULT_TARGET_ALPHA = 0.5
NON_FINAL_TOTAL_WEIGHT = 0.3
FINAL_WEIGHT = 0.7
STAGE_SCHEDULE_VERSION = "1.0.0"


def stage_target_lambdas(
    num_stages: int,
    alpha: float = DEFAULT_TARGET_ALPHA,
) -> tuple[float, ...]:
    """Interpolation factors ``lambda_k`` for ``k = 1..N``.

    ``N = 1`` has no interpolation span: the single stage is supervised directly on
    ``M_true``, which is what the existing one-stage model does.
    """
    if num_stages < 1:
        raise ValueError(f"num_stages must be at least 1, got {num_stages}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    if num_stages == 1:
        return (1.0,)
    span = num_stages - 1
    return tuple(float(alpha) + (1.0 - float(alpha)) * k / span for k in range(num_stages))


def stage_loss_weights(
    num_stages: int,
    non_final_total_weight: float = NON_FINAL_TOTAL_WEIGHT,
) -> tuple[float, ...]:
    """Per-stage model-loss weights, summing to 1.

    ``non_final_total_weight`` defaults to the frozen 0.3, so the accepted runs are
    unchanged; it exists so the stage-6.4 screen can vary the non-final/final balance
    without editing the frozen constants.
    """
    if num_stages < 1:
        raise ValueError(f"num_stages must be at least 1, got {num_stages}")
    if num_stages == 1:
        return (1.0,)
    total = float(non_final_total_weight)
    if not 0.0 <= total <= 1.0:
        raise ValueError(f"non_final_total_weight must lie in [0, 1], got {total}")
    # The default path must be bit-identical to the accepted runs, so the frozen values
    # are returned as their own constants rather than derived. (Complement arithmetic can
    # differ by an ulp in general -- 1.0 - 0.7 != 0.3 -- so this keeps the accepted
    # weights exact by construction instead of relying on the pairing being safe.)
    if total == NON_FINAL_TOTAL_WEIGHT:
        return tuple([NON_FINAL_TOTAL_WEIGHT / (num_stages - 1)] * (num_stages - 1) + [FINAL_WEIGHT])
    share = total / (num_stages - 1)
    return tuple([share] * (num_stages - 1) + [1.0 - total])


def stage_targets(
    initial_model,
    target_model,
    num_stages: int,
    alpha: float = DEFAULT_TARGET_ALPHA,
):
    """``M_k* = M_0 + lambda_k (M_true - M_0)`` for ``k = 1..N``, ordered.

    ``lambda = 1`` returns ``target_model`` and ``lambda = 0`` returns
    ``initial_model`` *unchanged*, rather than recomputing
    ``M_0 + 1.0 * (M_true - M_0)``. That keeps the final stage bit-identical to the
    historical implementation, so a two-stage run reproduces it exactly.
    """
    targets = []
    for lam in stage_target_lambdas(num_stages, alpha):
        if lam == 1.0:
            targets.append(target_model)
        elif lam == 0.0:
            targets.append(initial_model)
        else:
            targets.append(initial_model + lam * (target_model - initial_model))
    return targets


def schedule_as_dict(
    num_stages: int,
    alpha: float = DEFAULT_TARGET_ALPHA,
    non_final_total_weight: float = NON_FINAL_TOTAL_WEIGHT,
) -> dict[str, object]:
    """Provenance record for the frozen schedule, written into run configs."""
    lambdas = stage_target_lambdas(num_stages, alpha)
    weights = stage_loss_weights(num_stages, non_final_total_weight)
    return {
        "stage_schedule_version": STAGE_SCHEDULE_VERSION,
        "num_stages": int(num_stages),
        "target_alpha": float(alpha),
        "target_lambdas": [round(value, 12) for value in lambdas],
        "stage_loss_weights": [round(value, 12) for value in weights],
        "non_final_total_weight": float(non_final_total_weight),
        "final_weight": round(1.0 - float(non_final_total_weight), 12)
        if num_stages > 1
        else 1.0,
        "data_loss_stage": "final",
        "intermediate_targets": [f"M0 + {value:g}*(Mtrue - M0)" for value in lambdas],
    }


def validate_schedule(
    num_stages: int,
    alpha: float = DEFAULT_TARGET_ALPHA,
    non_final_total_weight: float = NON_FINAL_TOTAL_WEIGHT,
) -> None:
    """Raise if the schedule is not self-consistent."""
    weights = stage_loss_weights(num_stages, non_final_total_weight)
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError(f"stage weights must sum to 1, got {sum(weights)}")
    lambdas = stage_target_lambdas(num_stages, alpha)
    if abs(lambdas[-1] - 1.0) > 1e-12:
        raise ValueError("the final stage must target M_true")
    if num_stages > 1 and abs(lambdas[0] - float(alpha)) > 1e-12:
        raise ValueError("the first stage must target the configured alpha blend")
    if any(b < a for a, b in zip(lambdas, lambdas[1:])):
        raise ValueError("stage targets must increase monotonically towards M_true")
