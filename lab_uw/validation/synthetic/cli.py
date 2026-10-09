"""Known-model numerical validation of the 1D DDS inversion workflow.

The truth trace is generated on a finer grid than the inversion grid.  A
coarse/fine Cartesian search uses the production ``UltrasonicModeler`` and its
current relative waveform objective.  The optional final adjoint inversion is
the same ``run_local_inversion`` used for laboratory records.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lab_uw.forward_modeling import UltrasonicModeler
from lab_uw.spectral_diagnostics import (
    compare_spectral_shapes,
    plot_spectral_comparison,
)
from lab_uw.validation.synthetic.models import (
    COMMON_PARAMETERS,
    apply_candidate,
    build_dds_assembly,
    cartesian_candidates,
    load_stf,
    refined_axes,
)
from lab_uw.validation.synthetic.metrics import (
    integrated_travel_time_us,
    model_validation_metrics,
    parameter_errors,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        # Search axes define the dimensionality of a case.  Inheriting axes
        # that are absent from a child case would silently turn, for example,
        # a two-velocity experiment into a five-parameter inversion.
        if key == "coarse_axes":
            merged[key] = deepcopy(value)
            continue
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_configuration(path: Path | str) -> tuple[dict[str, Any], Path]:
    path = Path(path).expanduser().resolve()

    def load_one(current: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        if current in stack:
            raise ValueError("cyclic synthetic-validation configuration inheritance")
        with current.open("r", encoding="utf-8") as stream:
            content = json.load(stream)
        if not isinstance(content, dict):
            raise ValueError("the validation configuration must contain a JSON object")
        parent = content.pop("extends", None)
        if parent is None:
            return content
        parent_path = Path(str(parent)).expanduser()
        if not parent_path.is_absolute():
            parent_path = current.parent / parent_path
        inherited = load_one(parent_path.resolve(), (*stack, current))
        return _deep_merge(inherited, content)

    return load_one(path, ()), path.parent


def _time_axis(configuration: Mapping[str, Any]) -> np.ndarray:
    time = configuration["time"]
    dt = float(time["dt_us"])
    duration = float(time["duration_us"])
    if dt <= 0.0 or duration <= dt:
        raise ValueError("time.dt_us and time.duration_us must be positive")
    return np.arange(0.0, duration + 0.5 * dt, dt, dtype=float)


def _material_velocity_limits(assembly: Mapping[str, Any]) -> tuple[float, float]:
    wave_type = str(assembly["wave_type"])
    velocity_key = "velocity" + wave_type
    values = [
        float(assembly["gouge_velocity_1"]),
        float(assembly["gouge_velocity_2"]),
        float(assembly["side1_params"][velocity_key]),
        float(assembly["central_params"][velocity_key]),
        float(assembly["side2_params"][velocity_key]),
    ]
    return min(values), max(values)


def _forward(
    *,
    configuration: Mapping[str, Any],
    assembly: Mapping[str, Any],
    stf_handler: Any,
    observed_time: np.ndarray,
    observed_waveform: np.ndarray,
    misfit_interval: np.ndarray,
    points_per_wavelength: int,
    normalize_waveform: bool,
    store_wavefield: bool,
) -> UltrasonicModeler:
    physics = configuration["physics"]
    cmin, cmax = _material_velocity_limits(assembly)
    cmin = min(cmin, float(physics.get("minimum_velocity_cm_us", cmin)))
    cmax = max(cmax, float(physics.get("maximum_velocity_cm_us", cmax)))
    modeler = UltrasonicModeler()
    modeler.forward_simulation(
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=float(physics["frequency_cutoff_mhz"]),
        minimum_velocity=cmin,
        maximum_velocity=cmax,
        maximum_damping=max(
            float(assembly["gouge_damping_1"]),
            float(assembly["gouge_damping_2"]),
        ),
        assembly_dict=deepcopy(dict(assembly)),
        misfit_interval=misfit_interval,
        normalization_interval=misfit_interval,
        embedding_mode="effective_surface",
        boundary_condition=str(physics.get("boundary_condition", "free")),
        source_injection=str(physics.get("source_injection", "body_force")),
        receiver_quantity=str(physics.get("receiver_quantity", "velocity")),
        points_per_wavelength=int(points_per_wavelength),
        source_receiver_config=dict(physics.get(
            "source_receiver_config",
            {
                "interpolation_radius_cells": 4,
                "source_normalization": "integral",
                "receiver_normalization": "sum",
                "spatial_boundary_mode": "truncate",
            },
        )),
        normalize_waveform=bool(normalize_waveform),
        store_wavefield=bool(store_wavefield),
        enable_plotting=False,
        make_movie=False,
    )
    return modeler


def _direct_interval(
    configuration: Mapping[str, Any],
    truth_simulation: UltrasonicModeler,
    stf_handler: Any,
    observed_time: np.ndarray,
) -> np.ndarray:
    window = configuration.get("direct_window", {})
    if "start_us" in window and "end_us" in window:
        start, end = float(window["start_us"]), float(window["end_us"])
    else:
        travel_time = integrated_travel_time_us(
            truth_simulation.grid_handler.spatial_axis,
            truth_simulation.velocity_model_handler.velocity_array,
        )
        stf_time = np.asarray(stf_handler.metadata["time_ax_waveform"], dtype=float)
        duration = float(stf_time[-1] - stf_time[0])
        start = travel_time - float(window.get("pre_padding_us", 0.15))
        end = travel_time + duration + float(window.get("post_padding_us", 0.35))
    indices = np.flatnonzero((observed_time >= start) & (observed_time <= end))
    if indices.size < 8:
        raise ValueError("the direct-arrival validation interval is too short")
    return indices


def generate_observation(
    configuration: Mapping[str, Any],
    stf_handler: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], UltrasonicModeler]:
    """Generate a fine-grid truth trace and the exact inversion window."""

    observed_time = _time_axis(configuration)
    truth_assembly = build_dds_assembly(configuration, model_section="truth_model")
    complete_interval = np.arange(observed_time.size, dtype=int)
    truth_ppw = int(configuration["numerics"]["truth_points_per_wavelength"])
    inversion_ppw = int(configuration["numerics"]["inversion_points_per_wavelength"])
    if truth_ppw <= inversion_ppw:
        raise ValueError(
            "truth_points_per_wavelength must exceed inversion_points_per_wavelength"
        )
    truth = _forward(
        configuration=configuration,
        assembly=truth_assembly,
        stf_handler=stf_handler,
        observed_time=observed_time,
        observed_waveform=np.zeros_like(observed_time),
        misfit_interval=complete_interval,
        points_per_wavelength=truth_ppw,
        normalize_waveform=False,
        store_wavefield=False,
    )
    clean = np.asarray(truth.synthetic_waveform, dtype=float).copy()
    direct = _direct_interval(configuration, truth, stf_handler, observed_time)

    noise = configuration.get("noise", {})
    relative_rms = float(noise.get("relative_rms", 0.0))
    if relative_rms < 0.0:
        raise ValueError("noise.relative_rms cannot be negative")
    observed = clean.copy()
    if relative_rms > 0.0:
        rng = np.random.default_rng(int(noise.get("seed", 0)))
        scale = relative_rms * float(np.sqrt(np.mean(clean[direct] ** 2)))
        observed += rng.normal(0.0, scale, observed.size)
    truth.observed_waveform = observed
    truth.misfit_interval = direct
    return observed_time, observed, direct, truth_assembly, truth


def _evaluate_candidate(
    *,
    candidate: Mapping[str, float],
    base_assembly: Mapping[str, Any],
    configuration: Mapping[str, Any],
    stf_handler: Any,
    observed_time: np.ndarray,
    observed_waveform: np.ndarray,
    misfit_interval: np.ndarray,
) -> tuple[dict[str, Any], UltrasonicModeler]:
    assembly = apply_candidate(base_assembly, candidate)
    simulation = _forward(
        configuration=configuration,
        assembly=assembly,
        stf_handler=stf_handler,
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        misfit_interval=misfit_interval,
        points_per_wavelength=int(
            configuration["numerics"]["inversion_points_per_wavelength"]
        ),
        normalize_waveform=True,
        store_wavefield=False,
    )
    row = {
        **{name: float(value) for name, value in candidate.items()},
        "misfit": float(simulation.misfit),
        "correlation": float(simulation.objective_components["correlation"]),
        "trace_scale": float(simulation.trace_scale),
    }
    return row, simulation


def _run_search_stage(
    *,
    stage: str,
    axes: Mapping[str, Any],
    base_assembly: Mapping[str, Any],
    configuration: Mapping[str, Any],
    stf_handler: Any,
    observed_time: np.ndarray,
    observed_waveform: np.ndarray,
    misfit_interval: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], UltrasonicModeler]:
    candidates = cartesian_candidates(axes)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_simulation: UltrasonicModeler | None = None
    print(f"{stage}: {len(candidates)} candidate(s)")
    progress_stride = max(1, len(candidates) // 10)
    for index, candidate in enumerate(candidates, start=1):
        row, simulation = _evaluate_candidate(
            candidate=candidate,
            base_assembly=base_assembly,
            configuration=configuration,
            stf_handler=stf_handler,
            observed_time=observed_time,
            observed_waveform=observed_waveform,
            misfit_interval=misfit_interval,
        )
        rows.append(row)
        if best_row is None or row["misfit"] < best_row["misfit"]:
            best_row, best_simulation = row, simulation
        if index == len(candidates) or index % progress_stride == 0:
            print(f"  {stage}: {index}/{len(candidates)}")
    assert best_row is not None and best_simulation is not None
    return rows, best_row, best_simulation


def _write_search_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _equivalent_model_ranges(
    rows: list[dict[str, Any]],
    parameter_names: list[str],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    best = min(float(row["misfit"]) for row in rows)
    threshold = best * (1.0 + float(relative_tolerance)) + float(absolute_tolerance)
    accepted = [row for row in rows if float(row["misfit"]) <= threshold]
    ranges = {
        name: [
            float(min(row[name] for row in accepted)),
            float(max(row[name] for row in accepted)),
        ]
        for name in parameter_names
    }
    return {
        "best_misfit": best,
        "threshold": threshold,
        "number_of_equivalent_candidates": len(accepted),
        "parameter_ranges": ranges,
    }


def _run_local_inversion(
    *,
    configuration: Mapping[str, Any],
    recovered_assembly: dict[str, Any],
    stf_handler: Any,
    observed_time: np.ndarray,
    observed_waveform: np.ndarray,
    misfit_interval: np.ndarray,
) -> UltrasonicModeler:
    local = configuration["inversion"].get("local", {})
    simulation = _forward(
        configuration=configuration,
        assembly=recovered_assembly,
        stf_handler=stf_handler,
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        misfit_interval=misfit_interval,
        points_per_wavelength=int(
            configuration["numerics"]["inversion_points_per_wavelength"]
        ),
        normalize_waveform=True,
        store_wavefield=True,
    )
    mean_velocity = 0.5 * (
        float(recovered_assembly["gouge_velocity_1"])
        + float(recovered_assembly["gouge_velocity_2"])
    )
    mean_damping = 0.5 * (
        float(recovered_assembly["gouge_damping_1"])
        + float(recovered_assembly["gouge_damping_2"])
    )
    dc = float(local.get("velocity_step_cm_us", 0.01 * mean_velocity))
    da = float(local.get("damping_step_cm2_us", 0.05 * mean_damping))
    simulation.run_local_inversion(
        n_iterations=int(local.get("iterations", 0)),
        dc_max_start=dc,
        dc_threshold=float(local.get("velocity_threshold_cm_us", 0.01 * dc)),
        da_max_start=da,
        da_threshold=float(local.get("damping_threshold_cm2_us", 0.01 * da)),
        reduce_factor=float(local.get("reduce_factor", 2.0)),
        misfit_threshold=float(local.get("misfit_threshold", 1.0e-10)),
        normalize_waveform=False,
        amplitude_scale_mode="fixed",
        fixed_trace_scale=float(simulation.trace_scale),
        spectral_gamma=float(local.get("spectral_gamma", 1.0)),
        objective_taper_fraction=float(local.get("taper_fraction", 0.0)),
        enable_plotting=False,
        retain_wavefield=False,
    )
    idx = simulation.velocity_model_handler.idx_dict
    for layer in (1, 2):
        gouge = idx[f"gouge_{layer}"]
        recovered_assembly[f"gouge_velocity_{layer}"] = float(np.mean(
            simulation.velocity_model_handler.velocity_array[gouge]
        ))
        recovered_assembly[f"gouge_damping_{layer}"] = float(np.mean(
            simulation.velocity_model_handler.damping_array[gouge]
        ))
    return simulation


def _plot_waveforms(
    path: Path,
    time: np.ndarray,
    observed: np.ndarray,
    recovered: np.ndarray,
    interval: np.ndarray,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    axis.plot(time, observed, label="Synthetic observation", linewidth=1.3)
    axis.plot(time, recovered, label="Recovered model", linewidth=1.3)
    axis.axvspan(time[interval[0]], time[interval[-1]], color="0.85", alpha=0.5)
    axis.set_xlabel("Time [µs]")
    axis.set_ylabel("Amplitude")
    axis.set_title("Known-model waveform validation")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_profiles(path: Path, truth: UltrasonicModeler, recovered: UltrasonicModeler) -> None:
    truth_model = truth.velocity_model_handler
    recovered_model = recovered.velocity_model_handler
    tx = truth.grid_handler.spatial_axis
    rx = recovered.grid_handler.spatial_axis
    values = (
        (truth_model.velocity_array, recovered_model.velocity_array,
         r"Velocity $c$ [cm/$\mu$s]"),
        (truth_model.density_array, recovered_model.density_array,
         r"Density $\rho$ [g/cm$^3$]"),
        (truth_model.damping_array, recovered_model.damping_array,
         r"Damping $\nu$ [cm$^2$/$\mu$s]"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for axis, (truth_values, recovered_values, ylabel) in zip(axes, values):
        axis.plot(tx, truth_values, label="Truth", linewidth=2.0)
        axis.plot(rx, recovered_values, "--", label="Recovered", linewidth=1.6)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Position [cm]")
    fig.suptitle("True and recovered DDS material profiles")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_search(path: Path, coarse: list[dict[str, Any]], fine: list[dict[str, Any]]) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.semilogy(
        np.arange(len(coarse)),
        np.maximum([row["misfit"] for row in coarse], 1.0e-16),
        "o-", label="Coarse search", markersize=3,
    )
    offset = len(coarse)
    axis.semilogy(
        offset + np.arange(len(fine)),
        np.maximum([row["misfit"] for row in fine], 1.0e-16),
        "o-", label="Fine search", markersize=3,
    )
    axis.set_xlabel("Candidate evaluation")
    axis.set_ylabel("Relative waveform misfit")
    axis.set_title("Global validation search")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_local_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    iteration = [item["iteration"] for item in history]
    best = np.maximum([item["best_misfit"] for item in history], 1.0e-16)
    trial = np.maximum([item["trial_misfit"] for item in history], 1.0e-16)
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.semilogy(iteration, trial, "o--", label="Trial")
    axis.semilogy(iteration, best, "o-", label="Accepted best")
    axis.set_xlabel("Local iteration")
    axis.set_ylabel("Objective")
    axis.set_title("Local inversion convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_validation(
    configuration: Mapping[str, Any],
    *,
    base_directory: Path,
    output_directory: Path | None = None,
    disable_local: bool = False,
) -> dict[str, Any]:
    """Run one complete known-model validation and write an auditable report."""

    case_name = str(configuration["name"])
    if output_directory is None:
        raw_output = Path(str(configuration.get("output_directory", "synthetic_validation_results")))
        output_directory = raw_output if raw_output.is_absolute() else base_directory / raw_output
        output_directory = output_directory / case_name
    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    stf_handler = load_stf(configuration["stf"], base_directory=base_directory)
    time, observed, interval, truth_assembly, truth = generate_observation(
        configuration, stf_handler
    )
    inversion_assembly = build_dds_assembly(
        configuration, model_section="initial_model"
    )
    inversion = configuration["inversion"]
    coarse_axes = inversion["coarse_axes"]
    coarse_rows, coarse_best, _ = _run_search_stage(
        stage="coarse search",
        axes=coarse_axes,
        base_assembly=inversion_assembly,
        configuration=configuration,
        stf_handler=stf_handler,
        observed_time=time,
        observed_waveform=observed,
        misfit_interval=interval,
    )
    coarse_parameter_best = {name: coarse_best[name] for name in coarse_axes}
    fine_axes = refined_axes(
        coarse_axes,
        coarse_parameter_best,
        points_per_axis=int(inversion.get("fine_points_per_axis", 3)),
        bounds=inversion.get("bounds"),
    )
    fine_base = apply_candidate(inversion_assembly, coarse_parameter_best)
    fine_rows, fine_best, fine_simulation = _run_search_stage(
        stage="fine search",
        axes=fine_axes,
        base_assembly=fine_base,
        configuration=configuration,
        stf_handler=stf_handler,
        observed_time=time,
        observed_waveform=observed,
        misfit_interval=interval,
    )
    fine_parameter_best = {name: fine_best[name] for name in fine_axes}
    recovered_assembly = apply_candidate(fine_base, fine_parameter_best)
    local_enabled = bool(inversion.get("local", {}).get("enabled", True))
    if local_enabled and not disable_local:
        final_simulation = _run_local_inversion(
            configuration=configuration,
            recovered_assembly=recovered_assembly,
            stf_handler=stf_handler,
            observed_time=time,
            observed_waveform=observed,
            misfit_interval=interval,
        )
    else:
        final_simulation = fine_simulation

    physics = configuration["physics"]
    metrics = model_validation_metrics(
        truth_simulation=truth,
        recovered_simulation=final_simulation,
        observed_time_us=time,
        observed_waveform=observed,
        misfit_interval=interval,
        frequency_min_mhz=float(physics.get("frequency_min_mhz", 0.0)),
        frequency_max_mhz=float(physics["frequency_cutoff_mhz"]),
    )
    errors = parameter_errors(truth_assembly, recovered_assembly)
    active_parameters = list(coarse_axes)
    equivalent = _equivalent_model_ranges(
        fine_rows,
        active_parameters,
        relative_tolerance=float(inversion.get("equivalent_relative_tolerance", 0.02)),
        absolute_tolerance=float(inversion.get("equivalent_absolute_tolerance", 1.0e-8)),
    )

    tolerance = configuration.get("acceptance", {})
    active_targets = []
    for parameter_name in active_parameters:
        active_targets.extend(
            COMMON_PARAMETERS.get(parameter_name, (parameter_name,))
        )
    maximum_parameter_error = max(
        (errors[name]["relative_error"] for name in active_targets),
        default=0.0,
    )
    checks = {
        "waveform": metrics["waveform_relative_l2"] <= float(
            tolerance.get("max_waveform_relative_l2", np.inf)
        ),
        "travel_time": abs(metrics["integrated_travel_time_error_us"]) <= float(
            tolerance.get("max_travel_time_error_us", np.inf)
        ),
        "active_parameters": maximum_parameter_error <= float(
            tolerance.get("max_active_parameter_relative_error", np.inf)
        ),
    }
    require_parameter_recovery = bool(
        tolerance.get("require_active_parameter_recovery", True)
    )
    required_check_names = ["waveform", "travel_time"]
    if require_parameter_recovery:
        required_check_names.append("active_parameters")

    np.savez_compressed(
        output_directory / "synthetic_observation.npz",
        time_us=time,
        observed_waveform=observed,
        clean_truth_waveform=truth.synthetic_waveform,
        direct_interval=interval,
        recovered_waveform=final_simulation.synthetic_waveform,
    )
    _write_search_csv(output_directory / "coarse_search.csv", coarse_rows)
    _write_search_csv(output_directory / "fine_search.csv", fine_rows)
    _plot_search(output_directory / "global_search_convergence.png", coarse_rows, fine_rows)
    _plot_waveforms(
        output_directory / "observed_vs_recovered_waveform.png",
        time, observed, final_simulation.synthetic_waveform, interval,
    )
    _plot_profiles(
        output_directory / "true_vs_recovered_profiles.png", truth, final_simulation
    )
    spectral = compare_spectral_shapes(
        time[interval], observed[interval],
        time[interval], final_simulation.synthetic_waveform[interval],
        frequency_min_mhz=float(physics.get("frequency_min_mhz", 0.0)),
        frequency_max_mhz=float(physics["frequency_cutoff_mhz"]),
    )
    plot_spectral_comparison(
        spectral,
        first_label="Synthetic observation",
        second_label="Recovered model",
        title="Direct-arrival spectral validation",
        output_path=output_directory / "spectral_comparison.png",
        show_distance=True,
    )
    history = list(getattr(final_simulation, "local_inversion_history", []))
    _plot_local_history(output_directory / "local_objective_history.png", history)

    summary = {
        "name": case_name,
        "description": configuration.get("description", ""),
        "passed": bool(all(checks[key] for key in required_check_names)),
        "acceptance_checks": checks,
        "required_acceptance_checks": required_check_names,
        "active_parameter_recovery_required": require_parameter_recovery,
        "truth_points_per_wavelength": int(
            configuration["numerics"]["truth_points_per_wavelength"]
        ),
        "inversion_points_per_wavelength": int(
            configuration["numerics"]["inversion_points_per_wavelength"]
        ),
        "direct_window_us": [float(time[interval[0]]), float(time[interval[-1]])],
        "truth_assembly": truth_assembly,
        "recovered_assembly": recovered_assembly,
        "coarse_best": coarse_best,
        "fine_best": fine_best,
        "metrics": metrics,
        "parameter_errors": errors,
        "maximum_active_parameter_relative_error": maximum_parameter_error,
        "equivalent_model_analysis": equivalent,
        "local_inversion_history": history,
        "output_directory": output_directory,
    }
    with (output_directory / "validation_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(summary), stream, indent=2, allow_nan=True)
        stream.write("\n")
    print(
        f"{case_name}: {'PASS' if summary['passed'] else 'FAIL'}; "
        f"J={metrics['waveform_relative_l2']:.6g}, "
        f"Df={metrics['spectral_shape_distance']:.6g}, "
        f"dt={metrics['integrated_travel_time_error_us']:.6g} us"
    )
    print(f"wrote {output_directory / 'validation_summary.json'}")
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the 1D DDS inversion against known synthetic models."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config", action="append", type=Path,
        help="JSON case configuration; repeat to run several cases.",
    )
    source.add_argument(
        "--suite", type=Path,
        help="JSON file listing case configurations to run in order.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        help="Optional common output root (one subdirectory per case).",
    )
    parser.add_argument(
        "--no-local", action="store_true",
        help="Run only the coarse/fine global search.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    config_paths = arguments.config
    if arguments.suite is not None:
        suite_path = arguments.suite.expanduser().resolve()
        with suite_path.open("r", encoding="utf-8") as stream:
            suite = json.load(stream)
        listed = suite.get("configs") if isinstance(suite, Mapping) else None
        if not isinstance(listed, list) or not listed:
            raise ValueError("a validation suite must contain a non-empty configs list")
        config_paths = []
        for listed_path in listed:
            candidate = Path(str(listed_path)).expanduser()
            if not candidate.is_absolute():
                candidate = suite_path.parent / candidate
            config_paths.append(candidate)
    assert config_paths is not None
    failed = False
    completed_results = []
    for config_path in config_paths:
        configuration, base_directory = load_configuration(config_path)
        output = None
        if arguments.output_root is not None:
            output = arguments.output_root.expanduser().resolve() / str(configuration["name"])
        result = run_validation(
            configuration,
            base_directory=base_directory,
            output_directory=output,
            disable_local=arguments.no_local,
        )
        completed_results.append(result)
        failed = failed or not bool(result["passed"])
    if arguments.suite is not None:
        if arguments.output_root is not None:
            suite_output = arguments.output_root.expanduser().resolve()
        else:
            suite_output = Path(completed_results[0]["output_directory"]).parent
        suite_output.mkdir(parents=True, exist_ok=True)
        suite_summary = {
            "passed": not failed,
            "number_of_cases": len(completed_results),
            "cases": [
                {
                    "name": result["name"],
                    "passed": result["passed"],
                    "required_acceptance_checks": result[
                        "required_acceptance_checks"
                    ],
                    "acceptance_checks": result["acceptance_checks"],
                    "waveform_relative_l2": result["metrics"][
                        "waveform_relative_l2"
                    ],
                    "spectral_shape_distance": result["metrics"][
                        "spectral_shape_distance"
                    ],
                    "integrated_travel_time_error_us": result["metrics"][
                        "integrated_travel_time_error_us"
                    ],
                    "maximum_active_parameter_relative_error": result[
                        "maximum_active_parameter_relative_error"
                    ],
                    "output_directory": result["output_directory"],
                }
                for result in completed_results
            ],
        }
        suite_path = suite_output / "suite_summary.json"
        with suite_path.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(suite_summary), stream, indent=2, allow_nan=True)
            stream.write("\n")
        print(f"wrote {suite_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
