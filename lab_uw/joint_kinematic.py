"""Joint velocity/thickness perturbations at fixed DDS travel time.

The homogeneous search defines a reference model ``(c0, h1, h2)``.  This
module redistributes the total gouge thickness and changes the velocity
contrast simultaneously.  For every pair ``(log_contrast, delta_h)`` a common
velocity scale is solved so that the complete one-way time through the two
gouges and all four groove ramps is equal to the homogeneous reference time.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .gouge_thickness import constant_total_thickness_pair
from .velocity_contrast import (
    constant_time_velocity_pair,
    dds_gouge_groove_travel_time,
)


def build_joint_constant_time_candidates(
    assembly_dict: Mapping[str, Any],
    homogeneous_velocity: float,
    log_contrasts: Sequence[float],
    thickness_offsets_mm: Sequence[float],
    *,
    wave_type: str | None = None,
) -> list[dict[str, float]]:
    """Return the Cartesian ``(velocity contrast, thickness offset)`` grid.

    The total gouge thickness is held fixed.  Unlike two independent
    one-dimensional perturbations, the common velocity scale is recomputed for
    each two-dimensional candidate against the *same* homogeneous target time.
    """

    c0 = float(homogeneous_velocity)
    if not np.isfinite(c0) or c0 <= 0.0:
        raise ValueError("homogeneous_velocity must be positive and finite")

    reference_h1 = float(assembly_dict["gouge_thickness_1"])
    reference_h2 = float(assembly_dict["gouge_thickness_2"])
    target_time = dds_gouge_groove_travel_time(
        assembly_dict,
        c0,
        c0,
        wave_type=wave_type,
    )

    candidates: list[dict[str, float]] = []
    for offset_value in thickness_offsets_mm:
        offset_mm = float(offset_value)
        h1, h2 = constant_total_thickness_pair(
            reference_h1,
            reference_h2,
            offset_mm,
        )
        candidate_assembly = dict(assembly_dict)
        candidate_assembly["gouge_thickness_1"] = h1
        candidate_assembly["gouge_thickness_2"] = h2

        for contrast_value in log_contrasts:
            log_contrast = float(contrast_value)
            c1, c2, _, achieved_time = constant_time_velocity_pair(
                candidate_assembly,
                c0,
                log_contrast,
                wave_type=wave_type,
                target_time_us=target_time,
            )
            candidates.append(
                {
                    "log_contrast": log_contrast,
                    "thickness_offset_mm": offset_mm,
                    "velocity_ratio": c1 / c2,
                    "gouge_velocity_1": c1,
                    "gouge_velocity_2": c2,
                    "gouge_thickness_1_cm": h1,
                    "gouge_thickness_2_cm": h2,
                    "total_gouge_thickness_cm": h1 + h2,
                    "target_stack_travel_time_us": target_time,
                    "stack_travel_time_us": achieved_time,
                }
            )

    return candidates
