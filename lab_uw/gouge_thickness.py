"""Constant-total-thickness perturbations for the two DDS gouge layers.

The measured thicknesses are treated as a reference pair ``(h1, h2)``.  A
candidate is generated with

    h1_candidate = h1 + delta_h
    h2_candidate = h2 - delta_h

so that the total gouge thickness is unchanged.  When both gouges use the
same velocity, their total direct-arrival travel time is therefore unchanged
as well.  The perturbation only redistributes the internal layer thicknesses
and moves the central block relative to the external faces.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def constant_total_thickness_pair(
    reference_thickness_1_cm: float,
    reference_thickness_2_cm: float,
    offset_mm: float,
) -> tuple[float, float]:
    """Return ``h1, h2`` after an opposite perturbation in millimetres."""

    h1_reference = _positive_finite(
        reference_thickness_1_cm,
        "reference_thickness_1_cm",
    )
    h2_reference = _positive_finite(
        reference_thickness_2_cm,
        "reference_thickness_2_cm",
    )
    offset = float(offset_mm)
    if not np.isfinite(offset):
        raise ValueError("offset_mm must be finite")

    offset_cm = offset / 10.0
    h1 = h1_reference + offset_cm
    h2 = h2_reference - offset_cm
    if h1 <= 0.0 or h2 <= 0.0:
        raise ValueError(
            "the requested thickness offset makes one gouge layer non-positive"
        )
    return h1, h2


def build_constant_total_thickness_candidates(
    assembly_dict: Mapping[str, Any],
    offsets_mm: Sequence[float],
    *,
    common_velocity_cm_us: float,
) -> list[dict[str, float]]:
    """Build serialisable thickness candidates for the orchestration layer.

    ``common_velocity_cm_us`` is recorded to make the fixed direct-arrival
    travel time explicit and to reject accidental use with an invalid
    velocity.  The two gouges must use this same velocity during this search.
    """

    reference_h1 = _positive_finite(
        assembly_dict["gouge_thickness_1"],
        "gouge_thickness_1",
    )
    reference_h2 = _positive_finite(
        assembly_dict["gouge_thickness_2"],
        "gouge_thickness_2",
    )
    velocity = _positive_finite(common_velocity_cm_us, "common_velocity_cm_us")
    total = reference_h1 + reference_h2
    target_time = total / velocity

    candidates: list[dict[str, float]] = []
    for value in offsets_mm:
        offset = float(value)
        h1, h2 = constant_total_thickness_pair(
            reference_h1,
            reference_h2,
            offset,
        )
        candidates.append(
            {
                "thickness_offset_mm": offset,
                "gouge_thickness_1_cm": h1,
                "gouge_thickness_2_cm": h2,
                "total_gouge_thickness_cm": h1 + h2,
                "gouge_travel_time_us": (h1 + h2) / velocity,
                "target_gouge_travel_time_us": target_time,
            }
        )
    return candidates


def select_thickness_minimum(
    results: Sequence[Mapping[str, float]],
    *,
    metric_key: str = "misfit",
    relative_tolerance: float = 1.0e-6,
) -> tuple[dict[str, float], np.ndarray]:
    """Select reproducibly while retaining reciprocal equivalent minima.

    Opposite thickness transfers are indistinguishable in a reciprocal 1D
    stack when the remaining source/receiver geometry is mirror symmetric.
    Every candidate within the numerical tolerance of the minimum is returned
    as an equivalent offset. The representative is the one closest to zero;
    a positive offset wins only an otherwise exact +/- tie.
    """

    if not results:
        raise ValueError("results must be non-empty")
    tolerance = float(relative_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    metrics = np.asarray([float(item[metric_key]) for item in results])
    if np.any(~np.isfinite(metrics)):
        raise ValueError(f"{metric_key} values must be finite")
    minimum = float(np.min(metrics))
    absolute_tolerance = tolerance * max(1.0, abs(minimum))
    equivalent = [
        dict(item)
        for item, metric in zip(results, metrics)
        if metric <= minimum + absolute_tolerance
    ]
    representative = min(
        equivalent,
        key=lambda item: (
            abs(float(item["thickness_offset_mm"])),
            float(item["thickness_offset_mm"]) < 0.0,
        ),
    )
    offsets = np.sort(
        np.asarray(
            [item["thickness_offset_mm"] for item in equivalent], dtype=float
        )
    )
    return representative, offsets
