"""Grid-convergence diagnostics for the homogeneous reference forward model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

import numpy as np

from lab_uw.validation.reference.case import ReferenceCase
from lab_uw.validation.reference.forward import (
    ReferenceForwardConfig,
    ReferenceGeometry,
    run_reference_forward,
)
from lab_uw.validation.reference.objective import (
    bounded_lag_correlation,
    fit_amplitude_scale,
    normalized_l2,
    packet_indices,
    signed_correlation,
)


@dataclass(frozen=True)
class ConvergenceLevel:
    points_per_wavelength: int
    dx_cm: float
    dt_us: float
    cfl_max: float
    cfl_at_min_velocity: float
    theoretical_phase_velocity_ratio_at_fmax: float
    raw_relative_l2_to_finest: float
    scaled_relative_l2_to_finest: float
    scale_to_finest: float
    correlation_to_finest: float
    best_lag_us_to_finest: float


def phase_velocity_ratio_at_fmax(points_per_wavelength: int, cfl: float) -> float:
    """Second-order FD phase velocity divided by physical velocity at fmax."""
    ppw = int(points_per_wavelength)
    courant = float(cfl)
    if ppw < 2 or not (0 < courant <= 1):
        raise ValueError("ppw must be >= 2 and CFL must be in (0, 1]")
    kh_over_two = np.pi / ppw
    return float(
        np.arcsin(courant * np.sin(kh_over_two))
        / (courant * kh_over_two)
    )


def check_reference_grid_convergence(
    *,
    case: ReferenceCase,
    geometry: ReferenceGeometry,
    base_config: ReferenceForwardConfig,
    points_per_wavelength: Sequence[int] = (10, 15, 20, 30),
    packet_order: int = 0,
    max_diagnostic_lag_us: float = 0.25,
) -> dict[str, Any]:
    """Compare a packet against the finest requested spatial resolution."""
    levels = sorted(set(int(value) for value in points_per_wavelength))
    if len(levels) < 2 or levels[0] < 2:
        raise ValueError("provide at least two distinct PPW levels, all >= 2")
    arrivals = case.packet_arrivals_us
    if packet_order < 0 or packet_order >= arrivals.size:
        raise ValueError("packet_order is outside the calibrated arrival list")
    stop_absolute = arrivals[packet_order] + case.packet_span_us + max_diagnostic_lag_us
    stop_relative = float(stop_absolute - case.observed_time_us[0])

    runs = []
    for ppw in levels:
        config = replace(base_config, points_per_wavelength=ppw)
        result = run_reference_forward(
            case,
            geometry,
            config,
            stop_time_us=stop_relative,
        )
        runs.append(result)

    finest = runs[-1]
    idx = packet_indices(
        finest.observed_time_us,
        start_us=float(arrivals[packet_order]),
        span_us=case.packet_span_us,
    )
    if idx.size < 3:
        raise ValueError("selected convergence packet contains fewer than 3 samples")
    reference = finest.synthetic_waveform[idx]
    dt_observed = float(np.median(np.diff(finest.observed_time_us)))
    max_lag_samples = int(round(max_diagnostic_lag_us / dt_observed))

    diagnostics = []
    for result in runs:
        candidate = result.synthetic_waveform[idx]
        scale = fit_amplitude_scale(
            reference,
            candidate,
            np.arange(idx.size),
            allow_polarity_flip=False,
        )
        lag, _ = bounded_lag_correlation(
            reference,
            candidate,
            max_lag_samples=max_lag_samples,
        )
        dx = float(np.diff(result.spatial_axis_cm)[0])
        dt = float(np.diff(result.simulation_time_us)[0])
        cfl_max = float(np.max(result.velocity_model_cm_per_us) * dt / dx)
        cfl_min_velocity = float(np.min(result.velocity_model_cm_per_us) * dt / dx)
        diagnostics.append(
            ConvergenceLevel(
                points_per_wavelength=result.config.points_per_wavelength,
                dx_cm=dx,
                dt_us=dt,
                cfl_max=cfl_max,
                cfl_at_min_velocity=cfl_min_velocity,
                theoretical_phase_velocity_ratio_at_fmax=phase_velocity_ratio_at_fmax(
                    result.config.points_per_wavelength, cfl_min_velocity
                ),
                raw_relative_l2_to_finest=normalized_l2(reference, candidate),
                scaled_relative_l2_to_finest=normalized_l2(
                    reference, scale * candidate
                ),
                scale_to_finest=scale,
                correlation_to_finest=signed_correlation(reference, candidate),
                best_lag_us_to_finest=float(lag * dt_observed),
            )
        )

    return {
        "packet_order": int(packet_order),
        "reference_points_per_wavelength": levels[-1],
        "levels": [asdict(item) for item in diagnostics],
        "interpretation": (
            "Choose the coarsest PPW whose trace is acceptably close to the "
            "finest level; do not calibrate physical parameters at unconverged PPW."
        ),
    }
