"""Additional local iterations on one saved waveform, with no global search."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle

import numpy as np


def enrich_local_checkpoint(state, simulation):
    """Add O(nx+nt) runtime data; the existing continuation fields are unchanged."""
    runtime = {
        "schema": 1,
        "grid": deepcopy(vars(simulation.grid_handler)),
        "simulation_time": deepcopy(vars(simulation.sim_time_handler)),
        "source_time_function": simulation.source_handler.time_function.copy(),
        "simulation": {},
    }
    for name in ("observed_time", "observed_waveform", "misfit_interval",
                 "normalization_interval", "embedding_mode", "boundary_condition",
                 "boundary_impedance", "source_injection", "receiver_quantity",
                 "frequency_cutoff", "absorbing", "maximum_damping",
                 "source_amplitude", "source_receiver_config", "geometry_type", "wave_type", "sample_dimensions",
                 "orchestration_optimizer_limits"):
        if hasattr(simulation, name):
            runtime["simulation"][name] = deepcopy(getattr(simulation, name))
    state["restart_runtime"] = runtime
    return state


def restart_requested(params):
    return bool(params.get("local_restart", {}).get("enabled", False))


def _matching_pair(saved, paired):
    """Match physical acquisition time, not the old downsampled waveform index."""
    times = paired["time_s"].to_numpy(dtype=float)
    target = float(saved["acquisition_time_s"])
    valid = np.isfinite(times)
    if not np.any(valid):
        raise ValueError("Nessun record meccanico associato al file ultrasonico.")
    positions = np.flatnonzero(valid)
    index = int(positions[np.argmin(np.abs(times[positions] - target))])
    error = abs(float(times[index]) - target)
    # Mechanical times in completed rows are exact samples of this same data.
    if error > 1e-7:
        raise ValueError(f"Il record salvato non è associato alla stessa onda: errore temporale {error:g} s.")
    candidates = np.flatnonzero(valid & (np.abs(times - target) <= 1e-7))
    if len(candidates) != 1:
        raise ValueError("Più onde sono associate al tempo salvato: selezione non univoca.")
    return next(paired.iloc[[index]].itertuples(index=False))


def merge_iteration_history(previous, current):
    old = deepcopy(previous)
    new = deepcopy(current)
    offset = max((int(row["iteration"]) for row in old), default=0)
    if not old:
        return new, offset
    for row in new:
        if int(row["iteration"]) == 0:
            continue
        row["iteration"] = offset + int(row["iteration"])
        old.append(row)
    return old, offset


def _model_trace(simulation, functions):
    """Recompute a trace with the same FD operator, source and receiver."""
    model = simulation.velocity_model_handler
    clock = simulation.sim_time_handler
    field = functions["pseudospectral_1D_damped"](
        num_x=simulation.grid_handler.total_grid_points,
        delta_x=simulation.grid_handler.dx, num_t=clock.num_t, delta_t=clock.dt,
        source_spatial_function=simulation.source_handler.spatial_function,
        source_time_function=simulation.source_handler.time_function,
        velocity_model=model.velocity_array, damping_model=model.damping_array,
        density_model=model.density_array, absorbing=simulation.absorbing,
        boundary_condition=simulation.boundary_condition,
        boundary_impedance=simulation.boundary_impedance,
        source_injection=simulation.source_injection,
    )
    trace = functions["sample_receiver"](
        field, simulation.receiver_handler.spatial_function,
        quantity=simulation.receiver_quantity, delta_t=clock.dt,
        delta_x=simulation.grid_handler.dx, velocity_model=model.velocity_array,
        damping_model=model.damping_array, density_model=model.density_array,
    )
    del field
    return np.interp(simulation.observed_time, clock.simulation_time, trace) * float(simulation.reference_trace_scale)


def _restore_runtime(simulation, state, observed, observed_time):
    runtime = state.get("restart_runtime")
    if runtime is None:
        return False
    saved = runtime["simulation"]
    if not np.array_equal(observed_time, saved["observed_time"]) or not np.array_equal(observed, saved["observed_waveform"]):
        raise ValueError("Dato o preprocessing diversi da quelli del checkpoint: non è la stessa inversione.")
    for name, value in runtime["grid"].items():
        setattr(simulation.grid_handler, name, deepcopy(value))
    for name, value in runtime["simulation_time"].items():
        setattr(simulation.sim_time_handler, name, deepcopy(value))
    for name, value in saved.items():
        setattr(simulation, name, deepcopy(value))
    simulation.source_handler.time_function = runtime["source_time_function"].copy()
    return True


def _legacy_window(saved, global_row, rows, assembly, simulation, params, functions, settings):
    """Recover the window used before layer means were updated for the next record."""
    explicit = settings.get("misfit_interval_us")
    time = np.asarray(simulation.observed_time)
    if explicit is not None:
        start, stop = map(float, explicit)
        if not start < stop:
            raise ValueError("misfit_interval_us deve avere inizio < fine.")
        mask = (time >= start) & (time <= stop)
        if np.count_nonzero(mask) < 3:
            raise ValueError("Finestra di misfit troppo corta.")
        return mask
    target_index = int(saved["idx_wave"])
    if target_index == int(global_row["idx_wave"]):
        c1, c2 = float(global_row["best_gouge_velocity_1"]), float(global_row["best_gouge_velocity_2"])
    else:
        stride = int(saved.get("local_anchor_stride", params.get("local_anchor_stride", 3)))
        candidates = [row for row in rows if "global_best_misfit" not in row
                      and int(row["idx_wave"]) < target_index
                      and (target_index - int(row["idx_wave"])) % stride == 0]
        if candidates:
            previous = max(candidates, key=lambda row: int(row["idx_wave"]))
            previous_assembly = previous.get("local_continuation_state", {}).get("assembly_dict", {})
            c1 = previous_assembly.get("gouge_velocity_1", previous.get("avg_gouge_velocity"))
            c2 = previous_assembly.get("gouge_velocity_2", previous.get("avg_gouge_velocity"))
        elif target_index - int(global_row["idx_wave"]) <= stride:
            c1, c2 = float(global_row["best_gouge_velocity_1"]), float(global_row["best_gouge_velocity_2"])
        else:
            raise ValueError("Il vecchio checkpoint non contiene la finestra di misfit e manca il predecessore. "
                             "Imposta local_restart['misfit_interval_us'] con gli estremi della run originale.")
    arrival = functions["compute_dds_travel_time"](assembly, float(c1), float(c2))
    stf = simulation.stf_handler
    stf_time = np.asarray(stf.metadata["time_ax_waveform"])
    return functions["direct_misfit_interval"](
        time, arrival_min_us=arrival, arrival_max_us=arrival,
        stf_duration_us=float(stf_time[-1] - stf_time[0]),
        padding_us=params["direct_window_padding_us"],
    )


def _limits(saved, simulation, params, settings):
    history = saved.get("local_iteration_history", [])
    known = getattr(simulation, "orchestration_optimizer_limits", None)
    initial = history[0] if history else {}
    last = history[-1] if history else {}
    assembly = simulation.assembly_dict
    initial_v = float(initial.get("velocity_step_limit", params.get("local_velocity_step_fraction", .02) * assembly["gouge_velocity_1"]))
    initial_d = float(initial.get("damping_step_limit", params.get("local_damping_step_fraction", .05) * assembly["gouge_damping_1"]))
    reuse = settings.get("reuse_last_step_limits", True)
    dc = float(last.get("velocity_step_limit", initial_v)) if reuse else initial_v
    da = float(last.get("damping_step_limit", initial_d)) if reuse else initial_d
    return {"dc_max_start": dc, "da_max_start": da,
            "dc_threshold": float(known["dc_threshold"] if known else initial_v * params.get("local_velocity_step_threshold_fraction", .01)),
            "da_threshold": float(known["da_threshold"] if known else initial_d * params.get("local_damping_step_threshold_fraction", .01))}


def continue_saved_waveform(functions, *, paired, observed_waveform_data,
                            waveform_metadata, stf_handler, params, base_assembly, infile_path):
    """Run one new local batch and write a separate global+local checkpoint stream."""
    settings = params["local_restart"]
    default_pickle = Path(params["outdir_path_l2norm"]) / (Path(infile_path).name.split(".")[0] + ".pkl")
    source_pickle = Path(settings.get("pickle_path") or default_pickle).expanduser().resolve()
    # Reading is non-mutating: never repair/truncate the source pickle here.
    rows = functions["load_pickle_stream"](source_pickle, repair_incomplete_tail=False)
    globals_ = [row for row in rows if "global_best_misfit" in row]
    if len(globals_) != 1:
        raise ValueError("Il pickle deve contenere una sola riga globale e almeno un risultato locale.")
    global_row = globals_[0]
    locals_ = [row for row in rows if "global_best_misfit" not in row]
    rec_n = settings.get("rec_n")
    candidates = locals_ if rec_n is None else [row for row in locals_ if int(row["rec_n"]) == int(rec_n)]
    if not candidates:
        raise ValueError(f"Nessun modello locale completato per rec_n={rec_n} in {source_pickle}.")
    saved = candidates[-1]
    state = deepcopy(saved.get("local_continuation_state"))
    if state is None:
        raise ValueError("Questo risultato non contiene local_continuation_state; servono i profili e il checkpoint del modello.")
    saved_mode = state["assembly_dict"].get("wave_type")
    if saved_mode != base_assembly["wave_type"]:
        raise ValueError("Il checkpoint appartiene all'altra polarizzazione.")
    paired_row = _matching_pair(saved, paired)
    observed = np.asarray(observed_waveform_data[int(paired_row.uw_idx)])
    observed_time = np.asarray(waveform_metadata["time_ax_waveform"])
    # The original global result rebuilds the original discretization and clock.
    # This is a single forward initialization, not another global parameter search.
    _, simulation = functions["_rebuild_simulation_from_saved_row"](
        saved_row=global_row, global_row=global_row, paired_row=paired_row,
        observed_waveform_data=observed_waveform_data, waveform_metadata=waveform_metadata,
        stf_handler=stf_handler, params=params, base_assembly=base_assembly,
    )
    assembly = functions["_restore_local_continuation_state"](simulation, state)
    exact_runtime = _restore_runtime(simulation, state, observed, observed_time)
    if not exact_runtime:
        # The old schema stored the nodal fields but not dt/window/source time.
        # Reconstruct from the original global seed, then check the stored trace.
        axis = np.asarray(state["grid_spatial_axis"])
        dx = float(axis[1] - axis[0])
        if not np.isclose(dx, simulation.grid_handler.dx, rtol=1e-9, atol=1e-12):
            raise ValueError("Passo spaziale ricostruito diverso da quello salvato. Ripristina i parametri della run originale.")
        simulation.grid_handler.dx = dx
        simulation.observed_time = observed_time.copy()
        simulation.observed_waveform = observed.copy()
        simulation.misfit_interval = _legacy_window(saved, global_row, rows, assembly, simulation, params, functions, settings)
    gamma = float(saved.get("spectral_gamma", params.get("local_spectral_gamma", 1.)))
    if not np.isclose(gamma, params.get("local_spectral_gamma", gamma), rtol=0, atol=0):
        raise ValueError("Gamma locale diverso da quello salvato: per continuare imposta lo stesso local_spectral_gamma.")
    taper = float(state.get("simulation_attributes", {}).get("objective_taper_fraction", params.get("local_objective_taper_fraction", 0.)))
    if not np.isclose(taper, params.get("local_objective_taper_fraction", taper), rtol=0, atol=0):
        raise ValueError("Taper diverso dalla run originale.")
    simulation.reference_trace_scale = float(saved["reference_trace_scale"])
    simulation.trace_scale = simulation.reference_trace_scale
    starting_trace = _model_trace(simulation, functions)
    stored_trace = state.get("simulation_attributes", {}).get("synthetic_waveform")
    if stored_trace is not None and not np.allclose(starting_trace, stored_trace, rtol=1e-7, atol=1e-9):
        raise ValueError("Il forward ricostruito non riproduce il sintetico salvato. "
                         "Ripristina STF, preprocessing e parametri di discretizzazione della run originale.")
    objective = functions["relative_waveform_objective"](
        simulation.observed_waveform, starting_trace, simulation.misfit_interval,
        spectral_gamma=gamma, taper_fraction=taper,
    )
    initial_J = float(objective["total"])
    if not np.isclose(initial_J, float(saved["misfit"]), rtol=1e-6, atol=1e-9):
        raise ValueError(f"Misfit iniziale {initial_J:.12g} diverso da quello salvato {float(saved['misfit']):.12g}. "
                         "Controlla finestra di misfit, STF e impostazioni originali; nessuna iterazione avviata.")
    simulation.synthetic_waveform = starting_trace
    simulation.misfit = initial_J
    additional = settings.get("additional_iterations", 200)
    if isinstance(additional, bool) or int(additional) != additional or int(additional) < 1:
        raise ValueError("additional_iterations deve essere un intero positivo.")
    limits = _limits(saved, simulation, params, settings)
    simulation.orchestration_optimizer_limits = deepcopy(limits)
    previous = saved.get("local_iteration_history", [])
    offset = max((int(row["iteration"]) for row in previous), default=0)
    output_root = Path(settings.get("output_directory") or source_pickle.parent / "local_continuations")
    output = output_root / f"{source_pickle.stem}_rec_n_{int(saved['rec_n'])}"
    batch = 1
    while (output / f"batch_{batch:03d}").exists():
        batch += 1
    output = output / f"batch_{batch:03d}"
    output.mkdir(parents=True)
    strategy = str(params.get("local_update_strategy", saved.get("local_update_strategy", "joint")))
    print(f"Continuazione rec_n={saved['rec_n']}: J iniziale verificato={initial_J:.9g}; "
          f"{int(additional)} iterazioni aggiuntive dopo {offset}, strategia={strategy}.", flush=True)
    simulation.run_local_inversion(
        n_iterations=int(additional), **limits, dw_max_start=0., dw_threshold=0., ds_max_start=0., ds_threshold=0.,
        reduce_factor=params["reduce_factor"], misfit_threshold=float(params.get("local_misfit_threshold", 1e-8)),
        normalize_waveform=False, amplitude_scale_mode="fixed", fixed_trace_scale=simulation.reference_trace_scale,
        spectral_gamma=gamma, objective_taper_fraction=taper, update_strategy=strategy,
        enable_plotting=True, make_movie=False, plot_output_path=output,
        retain_wavefield=False, plot_name_prefix=f"rec_n_{int(saved['rec_n'])}_additional_{int(additional)}",
    )
    if not simulation.local_inversion_history or not np.isclose(
        float(simulation.local_inversion_history[0]['best_misfit']), initial_J, rtol=1e-6, atol=1e-9
    ):
        raise RuntimeError('Il solver locale ha ricostruito un modello iniziale diverso dal checkpoint verificato.')
    history, _ = merge_iteration_history(previous, simulation.local_inversion_history)
    simulation.local_inversion_history = history
    boundary = previous[-1] if previous else history[0]
    if not np.isclose(float(boundary["best_misfit"]), initial_J, rtol=1e-6, atol=1e-9):
        raise RuntimeError("Cronologia del modello iniziale incoerente.")
    if float(simulation.misfit) > initial_J + 1e-10:
        raise RuntimeError("La continuazione ha aumentato il miglior misfit.")
    # Point the new result at the physical full-file waveform index and retain its record id.
    assembly["idx_wave"] = int(paired_row.uw_idx)
    simulation.assembly_dict = assembly
    result = {"simulation_handler": simulation, "local_spectral_diagnostics": None}
    if params.get("local_spectral_diagnostics_enabled", True):
        result["local_spectral_diagnostics"] = simulation.write_direct_arrival_spectral_diagnostics(
            output_directory=output, frequency_min_mhz=params.get("frequency_min", 0.),
            frequency_max_mhz=params["frequency_cutoff"],
        )
    result_row = functions["_build_local_result_row"](int(saved["rec_n"]), assembly, simulation, result)
    result_row["density_profile"] = simulation.velocity_model_handler.density_array.copy()
    result_row["local_continuation_state"] = enrich_local_checkpoint(result_row["local_continuation_state"], simulation)
    result_row["local_restart"] = {"source_pickle": str(source_pickle), "source_idx_wave": int(saved["idx_wave"]),
                                   "initial_misfit": initial_J, "previous_iterations": offset,
                                   "requested_additional_iterations": int(additional), "exact_runtime_checkpoint": exact_runtime}
    result_row["local_anchor_stride"] = int(saved.get("local_anchor_stride", params.get("local_anchor_stride", 3)))
    result_row["local_records_per_worker"] = int(saved.get("local_records_per_worker", params.get("local_records_per_worker", 3)))
    destination = output / "continued_results.pkl"
    with destination.open("wb") as stream:
        pickle.dump(deepcopy(global_row), stream, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(result_row, stream, protocol=pickle.HIGHEST_PROTOCOL)
    from .forward_modeling import _plot_local_misfit_history
    _plot_local_misfit_history(history, output / "cumulative_misfit_vs_iterations.png", spectral_gamma=gamma)
    summary = {**result_row["local_restart"], "rec_n": int(saved["rec_n"]), "final_misfit": float(simulation.misfit),
               "total_iterations": max((int(row["iteration"]) for row in history), default=0),
               "checkpoint": str(destination), "step_limits": limits,
               "restart_kind": "additional local batch; parameter bounds are recomputed by the unchanged optimizer"}
    (output / "continuation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Continuazione salvata: {destination}; J={simulation.misfit:.9g}.", flush=True)
    return result_row
