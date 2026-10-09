"""Configuration helpers for controlled synthetic DDS validation cases.

The functions in this module deliberately contain no inversion logic.  They
only build a mechanically explicit DDS assembly, an effective source wavelet
and Cartesian parameter candidates.  Keeping this layer pure makes every
synthetic experiment reproducible from its JSON configuration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass
class ArraySTF:
    """Minimal source-time-function object accepted by ``UltrasonicModeler``."""

    waveform_data: np.ndarray
    metadata: dict[str, Any]


def analytic_stf(specification: Mapping[str, Any]) -> ArraySTF:
    """Build a deterministic damped sinusoid for self-contained validations."""

    dt = float(specification.get("dt_us", 0.005))
    duration = float(specification.get("duration_us", 1.5))
    frequency = float(specification.get("frequency_mhz", 1.5))
    decay = float(specification.get("decay_per_us", 2.5))
    phase = float(specification.get("phase_rad", 0.0))
    if dt <= 0.0 or duration <= dt or frequency <= 0.0 or decay < 0.0:
        raise ValueError("invalid analytic STF parameters")
    time = np.arange(0.0, duration + 0.5 * dt, dt, dtype=float)
    waveform = np.sin(2.0 * np.pi * frequency * time + phase)
    waveform *= np.exp(-decay * time)
    peak = float(np.max(np.abs(waveform)))
    if peak == 0.0:
        raise ValueError("the analytic STF is identically zero")
    waveform /= peak
    return ArraySTF(
        waveform_data=waveform,
        metadata={
            "time_ax_waveform": time,
            "number_of_samples": int(time.size),
            "number_of_waveforms": 1,
            "product_type": "synthetic_validation_stf",
        },
    )


def load_stf(specification: Mapping[str, Any], *, base_directory: Path) -> Any:
    """Load either the analytic validation STF or an experimental HDF5 STF."""

    kind = str(specification.get("kind", "analytic")).lower()
    if kind == "analytic":
        return analytic_stf(specification)
    if kind != "hdf5":
        raise ValueError("stf.kind must be either 'analytic' or 'hdf5'")

    from lab_uw.data_io.data_io import UltrasonicDataHandler

    raw_path = Path(str(specification["path"])).expanduser()
    path = raw_path if raw_path.is_absolute() else base_directory / raw_path
    handler = UltrasonicDataHandler.read_stf_hdf5(
        path,
        name=str(specification["name"]),
        cycle_index=int(specification.get("cycle_index", 0)),
    )
    waveform = np.asarray(handler.waveform_data, dtype=float).squeeze()
    time = np.asarray(handler.metadata["time_ax_waveform"], dtype=float).squeeze()
    if waveform.ndim != 1 or time.ndim != 1 or waveform.size != time.size:
        raise ValueError("the selected HDF5 STF must resolve to one 1D waveform")
    truncate_us = specification.get("truncate_us")
    if truncate_us is not None:
        keep = time <= time[0] + float(truncate_us)
        waveform = waveform[keep]
        time = time[keep]
    if waveform.size < 2:
        raise ValueError("the selected STF is too short")
    handler.waveform_data = waveform
    handler.metadata = dict(handler.metadata)
    handler.metadata["time_ax_waveform"] = time - time[0]
    handler.metadata["number_of_samples"] = int(time.size)
    return handler


def _pair(values: Iterable[float], name: str) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float).reshape(-1)
    if array.size != 2 or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly two finite values")
    return float(array[0]), float(array[1])


def build_dds_assembly(
    configuration: Mapping[str, Any],
    *,
    model_section: str,
) -> dict[str, Any]:
    """Build the exact assembly dictionary used by the effective-surface solver."""

    geometry = configuration["geometry"]
    material = configuration[model_section]
    dimensions = np.asarray(geometry["sample_dimensions_cm"], dtype=float).copy()
    if dimensions.shape != (5,) or np.any(dimensions <= 0.0):
        raise ValueError("geometry.sample_dimensions_cm must contain five positives")

    velocities = geometry["block_velocity_cm_us"]
    densities = geometry["block_density_g_cm3"]
    wave_type = str(configuration.get("wave_type", "_p"))
    if wave_type not in {"_p", "_s"}:
        raise ValueError("synthetic validation wave_type must be '_p' or '_s'")
    velocity_key = "velocity" + wave_type

    gouge_velocity = _pair(material["gouge_velocity_cm_us"], "gouge velocity")
    gouge_damping = _pair(material["gouge_damping_cm2_us"], "gouge damping")
    gouge_density = _pair(material["gouge_density_g_cm3"], "gouge density")
    gouge_thickness = _pair(
        material.get("gouge_thickness_cm", (dimensions[1], dimensions[3])),
        "gouge thickness",
    )
    dimensions[1], dimensions[3] = gouge_thickness

    side_groove = float(geometry["groove_side_cm"])
    central_groove = float(geometry["groove_central_cm"])
    if side_groove <= 0.0 or central_groove <= 0.0:
        raise ValueError("groove thicknesses must be positive")
    if side_groove >= min(dimensions[0], dimensions[4]):
        raise ValueError("side grooves are thicker than their steel blocks")
    if 2.0 * central_groove >= dimensions[2]:
        raise ValueError("central grooves are thicker than the central block")

    side1 = {
        "h_grooves": side_groove,
        velocity_key: float(velocities["side1"]),
        "pzt_layer_width": 0.0,
        "pla_layer_width": 0.0,
    }
    central = {
        "h_grooves": central_groove,
        velocity_key: float(velocities["central"]),
    }
    side2 = {
        "h_grooves": side_groove,
        velocity_key: float(velocities["side2"]),
        "pzt_layer_width": 0.0,
        "pla_layer_width": 0.0,
    }
    assembly = {
        "side1_params": side1,
        "side2_params": side2,
        "central_params": central,
        "wave_type": wave_type,
        "sample_dimensions": dimensions.tolist(),
        "gouge_thickness_1": gouge_thickness[0],
        "gouge_thickness_2": gouge_thickness[1],
        "gouge_velocity_1": gouge_velocity[0],
        "gouge_velocity_2": gouge_velocity[1],
        "gouge_damping_1": gouge_damping[0],
        "gouge_damping_2": gouge_damping[1],
        "density_model_enabled": True,
        "side1_density": float(densities["side1"]),
        "central_density": float(densities["central"]),
        "side2_density": float(densities["side2"]),
        "gouge_density_1": gouge_density[0],
        "gouge_density_2": gouge_density[1],
        "embedding_mode": "effective_surface",
        "rec_n": 0,
        "idx_wave": 0,
        "normal_stress": float("nan"),
        "shear_stress": float("nan"),
    }
    return assembly


COMMON_PARAMETERS = {
    "gouge_velocity_common": ("gouge_velocity_1", "gouge_velocity_2"),
    "gouge_damping_common": ("gouge_damping_1", "gouge_damping_2"),
    "gouge_density_common": ("gouge_density_1", "gouge_density_2"),
    "gouge_thickness_common": ("gouge_thickness_1", "gouge_thickness_2"),
}
DIRECT_PARAMETERS = {
    "gouge_velocity_1", "gouge_velocity_2",
    "gouge_damping_1", "gouge_damping_2",
    "gouge_density_1", "gouge_density_2",
    "gouge_thickness_1", "gouge_thickness_2",
}


def apply_candidate(
    base_assembly: Mapping[str, Any],
    candidate: Mapping[str, float],
) -> dict[str, Any]:
    """Apply one validation candidate while keeping thickness bookkeeping exact."""

    assembly = deepcopy(dict(base_assembly))
    for name, raw_value in candidate.items():
        value = float(raw_value)
        if not np.isfinite(value):
            raise ValueError(f"candidate parameter {name!r} is not finite")
        if name in COMMON_PARAMETERS:
            for target in COMMON_PARAMETERS[name]:
                assembly[target] = value
        elif name in DIRECT_PARAMETERS:
            assembly[name] = value
        else:
            raise ValueError(f"unsupported validation parameter: {name}")

    dimensions = list(map(float, assembly["sample_dimensions"]))
    dimensions[1] = float(assembly["gouge_thickness_1"])
    dimensions[3] = float(assembly["gouge_thickness_2"])
    assembly["sample_dimensions"] = dimensions
    positive_names = (
        "gouge_velocity_1", "gouge_velocity_2",
        "gouge_density_1", "gouge_density_2",
        "gouge_thickness_1", "gouge_thickness_2",
    )
    if any(float(assembly[name]) <= 0.0 for name in positive_names):
        raise ValueError("velocity, density and thickness candidates must be positive")
    if min(float(assembly["gouge_damping_1"]), float(assembly["gouge_damping_2"])) < 0.0:
        raise ValueError("damping candidates must be non-negative")
    return assembly


def cartesian_candidates(axes: Mapping[str, Iterable[float]]) -> list[dict[str, float]]:
    """Return deterministically ordered dictionaries from named search axes."""

    if not axes:
        return [{}]
    names = list(axes)
    values = []
    for name in names:
        if name not in COMMON_PARAMETERS and name not in DIRECT_PARAMETERS:
            raise ValueError(f"unsupported validation search axis: {name}")
        axis = np.unique(np.asarray(list(axes[name]), dtype=float).reshape(-1))
        if axis.size == 0 or np.any(~np.isfinite(axis)):
            raise ValueError(f"search axis {name!r} must be finite and non-empty")
        values.append(axis)
    return [
        {name: float(value) for name, value in zip(names, combination)}
        for combination in product(*values)
    ]


def refined_axes(
    coarse_axes: Mapping[str, Iterable[float]],
    best_candidate: Mapping[str, float],
    *,
    points_per_axis: int,
    bounds: Mapping[str, Iterable[float]] | None = None,
) -> dict[str, np.ndarray]:
    """Refine every active parameter by one coarse spacing around its minimum."""

    count = int(points_per_axis)
    if count < 1:
        raise ValueError("points_per_axis must be positive")
    bounds = {} if bounds is None else bounds
    output: dict[str, np.ndarray] = {}
    for name, raw_axis in coarse_axes.items():
        axis = np.unique(np.asarray(list(raw_axis), dtype=float).reshape(-1))
        if axis.size == 1 or count == 1:
            output[name] = np.asarray([float(best_candidate[name])])
            continue
        spacing = float(np.max(np.diff(axis)))
        low = float(best_candidate[name]) - spacing
        high = float(best_candidate[name]) + spacing
        if name in bounds:
            lower, upper = _pair(bounds[name], f"bounds for {name}")
            low, high = max(low, lower), min(high, upper)
        if not low < high:
            raise ValueError(f"empty fine-search interval for {name}")
        output[name] = np.linspace(low, high, count, dtype=float)
    return output

