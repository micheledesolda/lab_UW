"""Quantitative metrics for known-model synthetic inversion experiments."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.integrate import trapezoid

from lab_uw.forward_modeling import relative_waveform_objective
from lab_uw.spectral_diagnostics import compare_spectral_shapes


def relative_scalar_error(recovered: float, truth: float, eps: float = 1.0e-15) -> float:
    denominator = max(abs(float(truth)), eps)
    return abs(float(recovered) - float(truth)) / denominator


def normalized_arrival_time(
    time_us: np.ndarray,
    waveform: np.ndarray,
    *,
    relative_threshold: float = 0.05,
) -> float:
    """Return the first amplitude crossing relative to that trace's own peak.

    The threshold is dimensionless and is evaluated separately for each
    waveform, so it never assumes a calibrated acquisition amplitude.  Direct
    absolute amplitude is used rather than a Hilbert envelope because the
    latter is non-local and can leak late reverberation energy back to the
    beginning of a finite record.
    """

    time = np.asarray(time_us, dtype=float).reshape(-1)
    trace = np.asarray(waveform, dtype=float).reshape(-1)
    if time.size != trace.size or time.size < 3:
        raise ValueError("time and waveform must be matched 1D arrays")
    threshold = float(relative_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("relative_threshold must lie in (0, 1)")
    amplitude = np.abs(trace)
    peak = float(np.max(amplitude))
    if peak <= 0.0:
        return float("nan")
    indices = np.flatnonzero(amplitude >= threshold * peak)
    return float(time[indices[0]]) if indices.size else float("nan")


def integrated_travel_time_us(
    spatial_axis_cm: np.ndarray,
    velocity_cm_us: np.ndarray,
) -> float:
    x = np.asarray(spatial_axis_cm, dtype=float).reshape(-1)
    velocity = np.asarray(velocity_cm_us, dtype=float).reshape(-1)
    if x.size != velocity.size or x.size < 2 or np.any(velocity <= 0.0):
        raise ValueError("invalid spatial axis or velocity profile")
    return float(trapezoid(1.0 / velocity, x))


def profile_relative_l2(
    truth_x: np.ndarray,
    truth_values: np.ndarray,
    recovered_x: np.ndarray,
    recovered_values: np.ndarray,
    *,
    eps: float = 1.0e-15,
) -> float:
    """Compare profiles on the truth grid without assuming equal discretisation."""

    tx = np.asarray(truth_x, dtype=float).reshape(-1)
    tv = np.asarray(truth_values, dtype=float).reshape(-1)
    rx = np.asarray(recovered_x, dtype=float).reshape(-1)
    rv = np.asarray(recovered_values, dtype=float).reshape(-1)
    if tx.size != tv.size or rx.size != rv.size:
        raise ValueError("profile coordinates and values must have equal lengths")
    recovered_on_truth = np.interp(tx, rx, rv)
    denominator = max(float(np.linalg.norm(tv)), eps)
    return float(np.linalg.norm(recovered_on_truth - tv) / denominator)


def model_validation_metrics(
    *,
    truth_simulation: Any,
    recovered_simulation: Any,
    observed_time_us: np.ndarray,
    observed_waveform: np.ndarray,
    misfit_interval: np.ndarray,
    frequency_min_mhz: float,
    frequency_max_mhz: float,
) -> dict[str, float]:
    """Evaluate waveform, spectrum, arrival, profile and effective-property errors."""

    time = np.asarray(observed_time_us, dtype=float)
    observed = np.asarray(observed_waveform, dtype=float)
    recovered_trace = np.asarray(recovered_simulation.synthetic_waveform, dtype=float)
    objective = relative_waveform_objective(observed, recovered_trace, misfit_interval)
    indices = np.asarray(misfit_interval, dtype=int).reshape(-1)
    spectral = compare_spectral_shapes(
        time[indices],
        observed[indices],
        time[indices],
        recovered_trace[indices],
        frequency_min_mhz=float(frequency_min_mhz),
        frequency_max_mhz=float(frequency_max_mhz),
    )

    truth_model = truth_simulation.velocity_model_handler
    recovered_model = recovered_simulation.velocity_model_handler
    truth_x = truth_simulation.grid_handler.spatial_axis
    recovered_x = recovered_simulation.grid_handler.spatial_axis
    truth_travel = integrated_travel_time_us(truth_x, truth_model.velocity_array)
    recovered_travel = integrated_travel_time_us(
        recovered_x, recovered_model.velocity_array
    )
    truth_impedance = truth_model.density_array * truth_model.velocity_array
    recovered_impedance = (
        recovered_model.density_array * recovered_model.velocity_array
    )
    observed_arrival = normalized_arrival_time(time[indices], observed[indices])
    recovered_arrival = normalized_arrival_time(
        time[indices], recovered_trace[indices]
    )

    return {
        "waveform_relative_l2": float(objective["relative_l2"]),
        "waveform_correlation": float(objective["correlation"]),
        "spectral_shape_distance": float(spectral.shape_distance),
        "arrival_time_observed_us": observed_arrival,
        "arrival_time_recovered_us": recovered_arrival,
        "arrival_time_error_us": float(recovered_arrival - observed_arrival),
        "truth_integrated_travel_time_us": truth_travel,
        "recovered_integrated_travel_time_us": recovered_travel,
        "integrated_travel_time_error_us": float(recovered_travel - truth_travel),
        "velocity_profile_relative_l2": profile_relative_l2(
            truth_x, truth_model.velocity_array,
            recovered_x, recovered_model.velocity_array,
        ),
        "damping_profile_relative_l2": profile_relative_l2(
            truth_x, truth_model.damping_array,
            recovered_x, recovered_model.damping_array,
        ),
        "density_profile_relative_l2": profile_relative_l2(
            truth_x, truth_model.density_array,
            recovered_x, recovered_model.density_array,
        ),
        "impedance_profile_relative_l2": profile_relative_l2(
            truth_x, truth_impedance,
            recovered_x, recovered_impedance,
        ),
    }


def parameter_errors(
    truth_assembly: Mapping[str, Any],
    recovered_assembly: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    names = (
        "gouge_velocity_1", "gouge_velocity_2",
        "gouge_damping_1", "gouge_damping_2",
        "gouge_density_1", "gouge_density_2",
        "gouge_thickness_1", "gouge_thickness_2",
    )
    output: dict[str, dict[str, float]] = {}
    for name in names:
        truth = float(truth_assembly[name])
        recovered = float(recovered_assembly[name])
        output[name] = {
            "truth": truth,
            "recovered": recovered,
            "absolute_error": abs(recovered - truth),
            "relative_error": relative_scalar_error(recovered, truth),
        }
    return output
