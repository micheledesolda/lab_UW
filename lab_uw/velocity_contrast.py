"""Velocity-contrast parameterisation for the two DDS gouge layers.

The homogeneous velocity/damping search fixes the overall direct-arrival
travel time.  This module generates pairs ``(c1, c2)`` that change the
velocity contrast between the two gouge layers while preserving the travel
time through the two gouges and the four groove ramps.

The groove contribution matches ``VelocityModelHandler``: the density-
normalised stiffness ``c**2`` is varied linearly between the adjacent steel
block and the gouge.  In the continuum limit its one-way travel time is

    integral_0^h dx / sqrt((1-f)c_block**2 + f c_gouge**2)
        = 2 h / (c_block + c_gouge).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def stiffness_ramp_travel_time(
    thickness_cm: float,
    block_velocity_cm_us: float,
    gouge_velocity_cm_us: float,
) -> float:
    """Return one-way time through a groove with a linear ``c**2`` ramp."""

    thickness = float(thickness_cm)
    if not np.isfinite(thickness) or thickness < 0.0:
        raise ValueError("groove thickness must be finite and non-negative")
    block_velocity = _positive_finite(
        block_velocity_cm_us, "block_velocity_cm_us"
    )
    gouge_velocity = _positive_finite(
        gouge_velocity_cm_us, "gouge_velocity_cm_us"
    )
    return 2.0 * thickness / (block_velocity + gouge_velocity)


def dds_gouge_groove_travel_time(
    assembly_dict: Mapping[str, Any],
    gouge_velocity_1: float,
    gouge_velocity_2: float,
    *,
    wave_type: str | None = None,
) -> float:
    """One-way time through both gouges and all four groove ramps.

    Steel-only propagation is deliberately excluded: it is unchanged during
    the contrast search and therefore cancels from the constant-time
    constraint.
    """

    c1 = _positive_finite(gouge_velocity_1, "gouge_velocity_1")
    c2 = _positive_finite(gouge_velocity_2, "gouge_velocity_2")
    h1 = _positive_finite(assembly_dict["gouge_thickness_1"], "gouge_thickness_1")
    h2 = _positive_finite(assembly_dict["gouge_thickness_2"], "gouge_thickness_2")

    wt = wave_type or assembly_dict["wave_type"]
    side1 = assembly_dict["side1_params"]
    side2 = assembly_dict["side2_params"]
    central = assembly_dict["central_params"]

    vs1 = _positive_finite(side1["velocity" + wt], "side1 velocity")
    vs2 = _positive_finite(side2["velocity" + wt], "side2 velocity")
    vc = _positive_finite(central["velocity" + wt], "central velocity")

    return (
        h1 / c1
        + h2 / c2
        + stiffness_ramp_travel_time(side1["h_grooves"], vs1, c1)
        + stiffness_ramp_travel_time(central["h_grooves"], vc, c1)
        + stiffness_ramp_travel_time(central["h_grooves"], vc, c2)
        + stiffness_ramp_travel_time(side2["h_grooves"], vs2, c2)
    )


def constant_time_velocity_pair(
    assembly_dict: Mapping[str, Any],
    homogeneous_velocity: float,
    log_contrast: float,
    *,
    wave_type: str | None = None,
    target_time_us: float | None = None,
    relative_tolerance: float = 1.0e-12,
    max_iterations: int = 100,
) -> tuple[float, float, float, float]:
    r"""Generate ``c1`` and ``c2`` at fixed gouge-plus-groove travel time.

    The pair is parameterised as

    ``c1 = scale * exp(log_contrast)`` and
    ``c2 = scale * exp(-log_contrast)``.

    The common scale is solved by bisection so that the complete travel time
    through the gouges and the groove ramps equals ``target_time_us``.  When
    that argument is omitted, the target is the travel time of the homogeneous
    reference model on the supplied geometry.  The explicit target is used by
    the joint velocity/thickness search, whose candidate geometry differs from
    the homogeneous reference geometry.  The return values are ``c1, c2,
    target_time, achieved_time``.
    """

    c0 = _positive_finite(homogeneous_velocity, "homogeneous_velocity")
    delta = float(log_contrast)
    if not np.isfinite(delta):
        raise ValueError("log_contrast must be finite")
    if relative_tolerance <= 0.0 or not np.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be positive and finite")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    exp_delta = float(np.exp(delta))
    exp_minus_delta = float(np.exp(-delta))
    if target_time_us is None:
        target_time = dds_gouge_groove_travel_time(
            assembly_dict,
            c0,
            c0,
            wave_type=wave_type,
        )
    else:
        target_time = _positive_finite(target_time_us, "target_time_us")

    def time_for_scale(scale: float) -> float:
        return dds_gouge_groove_travel_time(
            assembly_dict,
            scale * exp_delta,
            scale * exp_minus_delta,
            wave_type=wave_type,
        )

    homogeneous_time = time_for_scale(c0)
    if delta == 0.0 and abs(homogeneous_time - target_time) <= (
        relative_tolerance * target_time
    ):
        return c0, c0, target_time, homogeneous_time

    lower = c0 * 0.25
    upper = c0 * 4.0
    while time_for_scale(lower) < target_time:
        lower *= 0.5
    while time_for_scale(upper) > target_time:
        upper *= 2.0

    scale = c0
    for _ in range(max_iterations):
        scale = 0.5 * (lower + upper)
        current_time = time_for_scale(scale)
        if abs(current_time - target_time) <= relative_tolerance * target_time:
            break
        if current_time > target_time:
            lower = scale
        else:
            upper = scale

    c1 = scale * exp_delta
    c2 = scale * exp_minus_delta
    achieved_time = dds_gouge_groove_travel_time(
        assembly_dict,
        c1,
        c2,
        wave_type=wave_type,
    )
    return c1, c2, target_time, achieved_time


def build_constant_time_contrast_candidates(
    assembly_dict: Mapping[str, Any],
    homogeneous_velocity: float,
    log_contrasts: Sequence[float],
    *,
    wave_type: str | None = None,
) -> list[dict[str, float]]:
    """Build serialisable contrast candidates for the orchestration layer."""

    candidates: list[dict[str, float]] = []
    for value in log_contrasts:
        delta = float(value)
        c1, c2, target, achieved = constant_time_velocity_pair(
            assembly_dict,
            homogeneous_velocity,
            delta,
            wave_type=wave_type,
        )
        candidates.append(
            {
                "log_contrast": delta,
                "velocity_ratio": c1 / c2,
                "gouge_velocity_1": c1,
                "gouge_velocity_2": c2,
                "target_stack_travel_time_us": target,
                "stack_travel_time_us": achieved,
            }
        )
    return candidates
