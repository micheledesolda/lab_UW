import copy
import json
import os
import pickle
import time as tm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from lab_uw.data_io.data_io import UltrasonicDataHandler
from lab_uw.forward_modeling import UltrasonicModeler
from lab_uw.plotting import Plotter

os.environ["MPLBACKEND"] = "Agg"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"


def _set_start_method():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


_set_start_method()


# ---------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------
def fit_trace_scale(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    misfit_interval,
    eps: float = 1e-12,
) -> float:
    if isinstance(misfit_interval, slice):
        d = observed_waveform[misfit_interval]
        s = synthetic_waveform[misfit_interval]
    else:
        idx = np.asarray(misfit_interval, dtype=int)
        d = observed_waveform[idx]
        s = synthetic_waveform[idx]

    return float(np.dot(d, s) / (np.dot(s, s) + eps))


def load_reference_calibration(calibration_path: Path) -> dict:
    with open(calibration_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_reflections_arrival_time(
    t_start_direct: float,
    reflection_spacing: float,
    direct_arrival_span: float,
    max_time: float
) -> list[float]:
    """
    Identifies reflection arrivals using a constant spacing between
    successive windows.

    Model:
        direct arrival start      = t_start_direct
        1st reflection start      = t_start_direct + reflection_spacing
        2nd reflection start      = t_start_direct + 2 * reflection_spacing
        ...

    where:
        reflection_spacing = 2 * one_way_travel_time
    """
    arrivals = [t_start_direct]
    tA = t_start_direct + reflection_spacing
    while tA + direct_arrival_span <= max_time:
        arrivals.append(tA)
        tA += reflection_spacing
    return arrivals


def build_multiwindow_misfit_interval(
    observed_time: np.ndarray,
    arrivals: list[float],
    direct_arrival_span: float,
) -> np.ndarray:
    idx_all = []
    for t_start in arrivals:
        t_end = t_start + direct_arrival_span
        i0 = np.searchsorted(observed_time, t_start)
        i1 = np.searchsorted(observed_time, t_end)
        i0 = max(0, i0)
        i1 = min(observed_time.size, i1)
        if i1 > i0:
            idx_all.append(np.arange(i0, i1, dtype=int))

    if len(idx_all) == 0:
        return np.array([], dtype=int)

    return np.unique(np.concatenate(idx_all))


def build_reference_misfit_interval_from_calibration(
    observed_time: np.ndarray,
    calibration_dict: dict,
) -> np.ndarray:
    direct_arrival_span = calibration_dict["refined_direct_arrival"]["direct_arrival_span_us"]
    arrival_times = calibration_dict["zero_angle_packets"]["arrival_times_us"]

    return build_multiwindow_misfit_interval(
        observed_time=observed_time,
        arrivals=arrival_times,
        direct_arrival_span=direct_arrival_span,
    )


def build_reference_compact_interval_from_calibration(
    observed_time: np.ndarray,
    calibration_dict: dict,
) -> np.ndarray:
    """
    First compact interval (direct arrival) from the saved reference calibration.
    """
    t_start_direct = calibration_dict["refined_direct_arrival"]["t_start_direct_final_us"]
    direct_arrival_span = calibration_dict["refined_direct_arrival"]["direct_arrival_span_us"]

    start = np.searchsorted(observed_time, t_start_direct)
    end = np.searchsorted(observed_time, t_start_direct + direct_arrival_span)

    start = max(0, start)
    end = min(observed_time.size, end)

    return np.arange(start, end, dtype=int)


def apply_empirical_decay_correction_from_calibration(
    observed_time: np.ndarray,
    synthetic_waveform: np.ndarray,
    calibration_dict: dict,
    exponent_p: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the empirical 1/L^p packet correction relative to the first compact interval,
    using the saved packet windows from the manual-identification script.
    """
    syn = np.asarray(synthetic_waveform, dtype=float).copy()
    time = np.asarray(observed_time, dtype=float)

    packet_starts = np.asarray(calibration_dict["zero_angle_packets"]["arrival_times_us"], dtype=float)
    direct_arrival_span = float(calibration_dict["refined_direct_arrival"]["direct_arrival_span_us"])
    distances_cm = np.asarray(calibration_dict["packet_decay_measurement"]["distances_cm"], dtype=float)

    if packet_starts.size != distances_cm.size:
        n = min(packet_starts.size, distances_cm.size)
        packet_starts = packet_starts[:n]
        distances_cm = distances_cm[:n]

    if packet_starts.size == 0:
        return syn, np.ones_like(syn, dtype=float)

    L0 = float(distances_cm[0])
    if L0 <= 0:
        raise ValueError("First saved propagation distance must be > 0.")

    relative_factors = (distances_cm / L0) ** (-float(exponent_p))
    relative_factors[0] = 1.0

    weight_trace = np.ones_like(syn, dtype=float)
    half_width = 0.5 * direct_arrival_span

    for t_start, factor in zip(packet_starts[1:], relative_factors[1:]):
        center = t_start + half_width
        window = raised_cosine_window(time, center=center, half_width=half_width)
        weight_trace += (factor - 1.0) * window

    syn *= weight_trace
    return syn, weight_trace


# ---------------------------------------------------------------------
# LOW-DIMENSIONAL STF MODEL
# ---------------------------------------------------------------------
def shift_trace_linear(t: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """
    Return y(t - tau) by interpolation on the original time axis t.
    tau > 0 delays the waveform.
    """
    return np.interp(t - tau, t, y, left=0.0, right=0.0)


def build_stf_basis(
    stf_time: np.ndarray,
    stf_waveform: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Build a small basis tied to the current reference STF:
      s0, ds/dt, d2s/dt2

    All arrays live on the original STF time axis.
    """
    t = np.asarray(stf_time, dtype=float).reshape(-1)
    s0 = np.asarray(stf_waveform, dtype=float)

    if s0.ndim == 2:
        if s0.shape[0] == 1:
            s0 = s0[0]
        elif s0.shape[1] == 1:
            s0 = s0[:, 0]
        else:
            raise ValueError(
                f"build_stf_basis expected a single STF waveform, got shape {s0.shape}"
            )
    elif s0.ndim != 1:
        raise ValueError(
            f"build_stf_basis expected 1-D or single-row 2-D waveform, got ndim={s0.ndim}"
        )

    if s0.size != t.size:
        raise ValueError(
            f"STF time and waveform length mismatch: len(t)={t.size}, len(s0)={s0.size}"
        )

    ds = np.gradient(s0, t)
    d2s = np.gradient(ds, t)

    return {
        "s0": s0,
        "ds": ds,
        "d2s": d2s,
    }


def build_trial_stf_waveform(
    stf_time: np.ndarray,
    stf_waveform: np.ndarray,
    *,
    multiplier_stf: float = 1.0,
    tau_shift_us: float = 0.0,
    c1_d1: float = 0.0,
    c2_d2: float = 0.0,
    normalize_energy_to_reference: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Low-dimensional STF expansion:

        s_trial(t) = m * s0(t - tau) + c1 * ds/dt + c2 * d2s/dt2

    Notes
    -----
    - tau_shift_us is in the same time units as stf_time (your code uses microseconds).
    - c1_d1 and c2_d2 are coefficients of the derivative basis.
    - normalize_energy_to_reference can be used to keep comparable source energy.
    """
    t = np.asarray(stf_time, dtype=float).reshape(-1)
    s0 = np.asarray(stf_waveform, dtype=float)

    if s0.ndim == 2:
        if s0.shape[0] == 1:
            s0 = s0[0]
        elif s0.shape[1] == 1:
            s0 = s0[:, 0]
        else:
            raise ValueError(
                f"build_trial_stf_waveform expected a single STF waveform, got shape {s0.shape}"
            )
    elif s0.ndim != 1:
        raise ValueError(
            f"build_trial_stf_waveform expected 1-D or single-row 2-D waveform, got ndim={s0.ndim}"
        )

    basis = build_stf_basis(t, s0)

    s_shift = shift_trace_linear(t, basis["s0"], tau_shift_us)
    s_trial = (
        multiplier_stf * s_shift
        + c1_d1 * basis["ds"]
        + c2_d2 * basis["d2s"]
    )

    if normalize_energy_to_reference:
        e_ref = np.linalg.norm(basis["s0"])
        e_new = np.linalg.norm(s_trial)
        if e_new > eps:
            s_trial *= e_ref / e_new

    return s_trial


def clone_stf_handler_with_waveform(
    stf_handler: UltrasonicDataHandler,
    new_waveform: np.ndarray,
) -> UltrasonicDataHandler:
    """
    Make a shallow copy of the STF handler and replace waveform_data only.
    """
    new_handler = copy.copy(stf_handler)
    new_handler.metadata = copy.deepcopy(stf_handler.metadata)
    new_handler.waveform_data = np.asarray(new_waveform, dtype=float).reshape(-1).copy()
    return new_handler


# ---------------------------------------------------------------------
# EMPIRICAL PACKET-DECAY CORRECTION
# ---------------------------------------------------------------------
def raised_cosine_window(
    time: np.ndarray,
    center: float,
    half_width: float,
) -> np.ndarray:
    """
    Smooth packet window centered at `center`, supported on
    [center-half_width, center+half_width].
    """
    if half_width <= 0:
        raise ValueError("half_width must be > 0")

    x = (time - center) / half_width
    w = np.zeros_like(time, dtype=float)
    mask = np.abs(x) <= 1.0
    w[mask] = 0.5 * (1.0 + np.cos(np.pi * x[mask]))
    return w


def compute_log_spectral_amplitude_misfit(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    observed_time: np.ndarray,
    misfit_interval,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
    eps: float = 1e-12,
) -> float:
    if isinstance(misfit_interval, slice):
        d = observed_waveform[misfit_interval]
        s = synthetic_waveform[misfit_interval]
        t = observed_time[misfit_interval]
    else:
        idx = np.asarray(misfit_interval, dtype=int)
        d = observed_waveform[idx]
        s = synthetic_waveform[idx]
        t = observed_time[idx]

    if d.size < 2:
        return 0.0

    dt = float(np.mean(np.diff(t)))
    freqs = np.fft.rfftfreq(d.size, d=dt)

    D = np.fft.rfft(d)
    S = np.fft.rfft(s)

    amp_obs = np.abs(D)
    amp_syn = np.abs(S)

    band = np.ones_like(freqs, dtype=bool)
    if fmin is not None:
        band &= (freqs >= fmin)
    if fmax is not None:
        band &= (freqs <= fmax)

    amp_obs = amp_obs[band]
    amp_syn = amp_syn[band]

    log_obs = np.log(amp_obs + eps)
    log_syn = np.log(amp_syn + eps)

    return float(np.sum((log_obs - log_syn) ** 2))


def compute_reference_objective(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    observed_time: np.ndarray,
    misfit_interval,
    reference_calibration: dict,
    *,
    beta_spec: float = 0.0,
    spec_fmin: float | None = None,
    spec_fmax: float | None = None,
    apply_empirical_decay_correction: bool = False,
    decay_exponent_p: float | None = None,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Returns:
      total_misfit, time_misfit, spec_misfit, time_scale, scaled_synthetic
    """
    syn = synthetic_waveform.copy()

    compact_interval = build_reference_compact_interval_from_calibration(
        observed_time=observed_time,
        calibration_dict=reference_calibration,
    )

    time_scale = fit_trace_scale(
        observed_waveform=observed_waveform,
        synthetic_waveform=syn,
        misfit_interval=compact_interval,
    )
    syn *= time_scale

    if apply_empirical_decay_correction:
        if decay_exponent_p is None:
            raise ValueError("decay_exponent_p is required when apply_empirical_decay_correction=True")

        syn, _ = apply_empirical_decay_correction_from_calibration(
            observed_time=observed_time,
            synthetic_waveform=syn,
            calibration_dict=reference_calibration,
            exponent_p=decay_exponent_p,
        )

    if isinstance(misfit_interval, slice):
        time_misfit = float(np.linalg.norm(observed_waveform[misfit_interval] - syn[misfit_interval]))
    else:
        idx = np.asarray(misfit_interval, dtype=int)
        time_misfit = float(np.linalg.norm(observed_waveform[idx] - syn[idx]))

    if beta_spec > 0.0:
        spec_misfit = compute_log_spectral_amplitude_misfit(
            observed_waveform=observed_waveform,
            synthetic_waveform=syn,
            observed_time=observed_time,
            misfit_interval=misfit_interval,
            fmin=spec_fmin,
            fmax=spec_fmax,
        )
    else:
        spec_misfit = 0.0

    total = time_misfit + beta_spec * spec_misfit
    return total, time_misfit, spec_misfit, time_scale, syn


# ---------------------------------------------------------------------
# SINGLE CANDIDATE EVALUATION
# ---------------------------------------------------------------------
def process_reference_candidate(args):
    (
        spreading_factor_transmitter,
        spreading_factor_receiver,
        position2edge_transmitter,
        position2edge_receiver,
        radius_factor_transmitter,
        radius_factor_receiver,
        steel_velocity2simulate,
        pzt_velocity2simulate,
        multiplier_STF,
        tau_shift_us,
        c1_d1,
        c2_d2,
        observed_waveform,
        observed_time,
        misfit_interval,
        stf_handler,
        params,
        assembly_dict,
    ) = args

    wave_type = assembly_dict["wave_type"]

    assembly_trial = assembly_dict.copy()
    assembly_trial["velocity" + wave_type] = steel_velocity2simulate
    assembly_trial["pzt_velocity" + wave_type] = pzt_velocity2simulate

    stf_trial_waveform = build_trial_stf_waveform(
        stf_time=np.asarray(stf_handler.metadata["time_ax_waveform"], dtype=float),
        stf_waveform=np.asarray(stf_handler.waveform_data, dtype=float),
        multiplier_stf=multiplier_STF,
        tau_shift_us=tau_shift_us,
        c1_d1=c1_d1,
        c2_d2=c2_d2,
        normalize_energy_to_reference=False,
    )

    stf_trial_handler = clone_stf_handler_with_waveform(
        stf_handler=stf_handler,
        new_waveform=stf_trial_waveform,
    )

    montecarlo = {
        "spreading_factor_transmitter": spreading_factor_transmitter,
        "spreading_factor_receiver": spreading_factor_receiver,
        "position2edge_transmitter": position2edge_transmitter,
        "position2edge_receiver": position2edge_receiver,
        "radius_factor_transmitter": radius_factor_transmitter,
        "radius_factor_receiver": radius_factor_receiver,
        "multiplier_STF": 1.0,
    }

    sim = UltrasonicModeler()
    sim.forward_simulation(
        geometry_type="block",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_trial_handler,
        frequency_cutoff=params["frequency_cutoff"],
        minimum_velocity=params["min_velocity2simulate"],
        maximum_velocity=params["max_velocity2simulate"],
        assembly_dict=assembly_trial,
        misfit_interval=misfit_interval,
        absorbing=params.get("absorbing", False),
        maximum_damping=0.0,
        montecarlo=montecarlo,
        normalize_waveform=params.get("normalize_waveform_global", True),
        enable_plotting=False,
        make_movie=False,
        plot_output_path=None,
        movie_output_path=None,
    )

    return {
        "pzt_velocity": pzt_velocity2simulate,
        "steel_velocity": steel_velocity2simulate,
        "total_misfit": float(sim.misfit),
        "spread_tx": spreading_factor_transmitter,
        "spread_rx": spreading_factor_receiver,
        "pos_tx": position2edge_transmitter,
        "pos_rx": position2edge_receiver,
        "radius_tx": radius_factor_transmitter,
        "radius_rx": radius_factor_receiver,
        "multiplier_stf": multiplier_STF,
        "tau_shift_us": tau_shift_us,
        "c1_d1": c1_d1,
        "c2_d2": c2_d2,
        "stf_waveform_trial": stf_trial_waveform,
        "candidate_synthetic_waveform": sim.synthetic_waveform.copy(),
    }

# ---------------------------------------------------------------------
# LOCAL SOURCE/RECEIVER INVERSION ORCHESTRATION
# ---------------------------------------------------------------------
def run_local_source_stage(
    simulation: UltrasonicModeler,
    params: Dict[str, Any],
    plot_output_path: Path | None,
) -> Dict[str, Any]:
    if not params.get("run_local_source_receiver_inversion", False):
        return {
            "ran_local_source_receiver_inversion": False,
            "local_source_receiver_initial_misfit": float(simulation.misfit),
            "local_source_receiver_final_misfit": float(simulation.misfit),
            "final_synthetic_waveform": simulation.synthetic_waveform.copy(),
            "final_source_time_function": simulation.source_handler.time_function.copy(),
            "final_source_spatial_function": simulation.source_handler.spatial_function.copy(),
            "final_receiver_spatial_function": simulation.receiver_handler.spatial_function.copy(),
        }

    local_cfg = params["local_source_receiver_inversion"]
    initial_misfit = float(simulation.misfit)

    simulation.run_local_stf_inversion(
        n_iterations=local_cfg["n_iterations"],
        dw_max_start=local_cfg["dw_max_start"],
        dw_threshold=local_cfg["dw_threshold"],
        stf_duration=local_cfg["stf_duration"],
        reduce_factor=local_cfg["reduce_factor"],
        misfit_threshold=local_cfg["misfit_threshold"],
        normalize_waveform=local_cfg["normalize_waveform"],
        smooth_gradient=local_cfg.get("smooth_gradient", True),
        gradient_smoothing_halfwidth=local_cfg.get("gradient_smoothing_halfwidth", 2),
        enable_plotting=local_cfg["enable_plotting"],
        make_movie=False,
        plot_output_path=plot_output_path,
        movie_output_path=None,
    )

    return {
        "ran_local_source_receiver_inversion": True,
        "local_source_receiver_initial_misfit": initial_misfit,
        "local_source_receiver_final_misfit": float(simulation.misfit),
        "final_synthetic_waveform": simulation.synthetic_waveform.copy(),
        "final_source_time_function": simulation.source_handler.time_function.copy(),
        "final_source_spatial_function": simulation.source_handler.spatial_function.copy(),
        "final_receiver_spatial_function": simulation.receiver_handler.spatial_function.copy(),
    }


# ---------------------------------------------------------------------
# MAIN PER-WAVEFORM PROCESSING
# ---------------------------------------------------------------------
def process_waveform(
    idx_waveform: int,
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    outfile_name: str,
    stf_handler: UltrasonicDataHandler,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    global_search_space: Dict[str, Any],
) -> Dict[str, Any]:

    if isinstance(params["outdir_path_image"], list):
        outdir_path_image = Path(params["outdir_path_image"][0])
    else:
        outdir_path_image = Path(params["outdir_path_image"])

    misfit_interval = np.arange(observed_time.size, dtype=int)

    num_iteration = global_search_space["num_iterations"]

    steel_array = np.random.uniform(
        global_search_space["steel_velocity_low"],
        global_search_space["steel_velocity_high"],
        num_iteration,
    )
    pzt_array = np.random.uniform(
        global_search_space["pzt_velocity_low"],
        global_search_space["pzt_velocity_high"],
        num_iteration,
    )

    spread_tx_array = np.random.uniform(
        global_search_space["spreading_factor_low"],
        global_search_space["spreading_factor_high"],
        num_iteration,
    )
    spread_rx_array = np.random.uniform(
        global_search_space["spreading_factor_low"],
        global_search_space["spreading_factor_high"],
        num_iteration,
    )

    pos_tx_array = np.random.uniform(
        global_search_space["position2edge_low"],
        global_search_space["position2edge_high"],
        num_iteration,
    )
    pos_rx_array = np.random.uniform(
        global_search_space["position2edge_low"],
        global_search_space["position2edge_high"],
        num_iteration,
    )

    radius_tx_array = np.random.uniform(
        global_search_space["radius_factor_low"],
        global_search_space["radius_factor_high"],
        num_iteration,
    )
    radius_rx_array = np.random.uniform(
        global_search_space["radius_factor_low"],
        global_search_space["radius_factor_high"],
        num_iteration,
    )

    multiplier_array = np.geomspace(
        global_search_space["min_multiplier"],
        global_search_space["max_multiplier"],
        num_iteration,
    )

    tau_shift_array = np.random.uniform(
        global_search_space.get("tau_shift_low_us", 0.0),
        global_search_space.get("tau_shift_high_us", 0.0),
        num_iteration,
    )
    c1_array = np.random.uniform(
        global_search_space.get("c1_d1_low", 0.0),
        global_search_space.get("c1_d1_high", 0.0),
        num_iteration,
    )
    c2_array = np.random.uniform(
        global_search_space.get("c2_d2_low", 0.0),
        global_search_space.get("c2_d2_high", 0.0),
        num_iteration,
    )

    args_list = [
        (
            spread_tx_array[i],
            spread_rx_array[i],
            pos_tx_array[i],
            pos_rx_array[i],
            radius_tx_array[i],
            radius_rx_array[i],
            steel_array[i],
            pzt_array[i],
            multiplier_array[i],
            tau_shift_array[i],
            c1_array[i],
            c2_array[i],
            observed_waveform,
            observed_time,
            misfit_interval,
            stf_handler,
            params,
            assembly_dict,
        )
        for i in range(num_iteration)
    ]

    ncpu = min(cpu_count(), params.get("max_workers", cpu_count()))
    nproc = max(1, ncpu)
    with Pool(processes=nproc) as pool:
        results = pool.map(process_reference_candidate, args_list)

    results = sorted(results, key=lambda r: r["total_misfit"])
    best = results[0]

    total_array = np.array([r["total_misfit"] for r in results], dtype=float)

    steel_sorted = np.array([r["steel_velocity"] for r in results], dtype=float)
    pzt_sorted = np.array([r["pzt_velocity"] for r in results], dtype=float)
    spread_tx_sorted = np.array([r["spread_tx"] for r in results], dtype=float)
    spread_rx_sorted = np.array([r["spread_rx"] for r in results], dtype=float)
    pos_tx_sorted = np.array([r["pos_tx"] for r in results], dtype=float)
    pos_rx_sorted = np.array([r["pos_rx"] for r in results], dtype=float)
    radius_tx_sorted = np.array([r["radius_tx"] for r in results], dtype=float)
    radius_rx_sorted = np.array([r["radius_rx"] for r in results], dtype=float)
    multiplier_sorted = np.array([r["multiplier_stf"] for r in results], dtype=float)
    tau_sorted = np.array([r["tau_shift_us"] for r in results], dtype=float)
    c1_sorted = np.array([r["c1_d1"] for r in results], dtype=float)
    c2_sorted = np.array([r["c2_d2"] for r in results], dtype=float)

    save_plot = (idx_waveform % params["plot_save_interval"] == 0)
    plot_output_path = outdir_path_image / f"{outfile_name}_best_simulation_global_misfit_{best["total_misfit"]}" if save_plot else None

    wave_type = assembly_dict["wave_type"]
    assembly_best = assembly_dict.copy()
    assembly_best["velocity" + wave_type] = best["steel_velocity"]
    assembly_best["pzt_velocity" + wave_type] = best["pzt_velocity"]

    best_stf_handler = clone_stf_handler_with_waveform(
        stf_handler=stf_handler,
        new_waveform=best["stf_waveform_trial"],
    )

    best_montecarlo = {
        "spreading_factor_transmitter": best["spread_tx"],
        "spreading_factor_receiver": best["spread_rx"],
        "position2edge_transmitter": best["pos_tx"],
        "position2edge_receiver": best["pos_rx"],
        "radius_factor_transmitter": best["radius_tx"],
        "radius_factor_receiver": best["radius_rx"],
        "multiplier_STF": 1.0,
    }

    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        geometry_type="block",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=best_stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_best,
        montecarlo=best_montecarlo,
        misfit_interval=misfit_interval,
        minimum_velocity=params["min_velocity2simulate"],
        maximum_velocity=params["max_velocity2simulate"],
        maximum_damping=0.0,
        normalize_waveform=params.get("normalize_waveform_global", True),
        enable_plotting=True,
        make_movie=False,
        plot_output_path=plot_output_path,
        movie_output_path=None,
    )

    print("best returned misfit:", best["total_misfit"])
    print("rebuilt misfit:", simulation.misfit)
    print("candidate vs rebuilt synthetic L2:",
        np.linalg.norm(best["candidate_synthetic_waveform"] - simulation.synthetic_waveform))
    max_stf = max(best_stf_handler.waveform_data)
    print(f"max stf handler: {max_stf}")
    params["local_source_receiver_inversion"]["dw_max_start"] = 100*max_stf
    params["local_source_receiver_inversion"]["dw_threshold"] = 0.0001*max_stf

    pre_local_result = {
        "best_steel_velocity": best["steel_velocity"],
        "best_pzt_velocity": best["pzt_velocity"],
        "best_total_misfit": float(simulation.misfit),
        "best_spread_tx": best["spread_tx"],
        "best_spread_rx": best["spread_rx"],
        "best_position2edge_tx": best["pos_tx"],
        "best_position2edge_rx": best["pos_rx"],
        "best_radius_factor_tx": best["radius_tx"],
        "best_radius_factor_rx": best["radius_rx"],
        "best_multiplier_stf": best["multiplier_stf"],
        "best_tau_shift_us": best["tau_shift_us"],
        "best_c1_d1": best["c1_d1"],
        "best_c2_d2": best["c2_d2"],
        "best_stf_waveform": best["stf_waveform_trial"],
        "best_synthetic_waveform": simulation.synthetic_waveform.copy(),
    }

    local_result = run_local_source_stage(
        simulation=simulation,
        params=params,
        plot_output_path=plot_output_path,
    )

    scatter_title = f"Reference calibration objective vs parameters\n{outfile_name}"
    scatter_outpath = outdir_path_image / f"{outfile_name}_param_vs_objective.png"
    plotter = Plotter()
    plotter.plot_scatter_l2_vs_parameters(
        param_list=[
            ("Steel Velocity (cm/µs)", steel_sorted),
            ("PZT Velocity (cm/µs)", pzt_sorted),
            ("Spread Tx", spread_tx_sorted),
            ("Spread Rx", spread_rx_sorted),
            ("Pos2Edge Tx", pos_tx_sorted),
            ("Pos2Edge Rx", pos_rx_sorted),
            ("Radius Factor Tx", radius_tx_sorted),
            ("Radius Factor Rx", radius_rx_sorted),
            ("STF Multiplier", multiplier_sorted),
            ("STF Shift (µs)", tau_sorted),
            ("STF c1·d/dt", c1_sorted),
            ("STF c2·d²/dt²", c2_sorted),
        ],
        l2_values=total_array,
        title=scatter_title,
        best_index=0,
        outfile_path=scatter_outpath,
    )

    return {
        "misfit_interval": misfit_interval,
        "objective_array": total_array,
        "steel_velocity_array": steel_sorted,
        "pzt_velocity_array": pzt_sorted,
        "best_observed_waveform": observed_waveform.copy(),
        "global_search_best": pre_local_result,
        "local_source_receiver_result": local_result,
    }

# ---------------------------------------------------------------------
# TOP-LEVEL FILE PROCESSING
# ---------------------------------------------------------------------
def process_uw_file(
    infile_path: Path,
    stf_handler: UltrasonicDataHandler,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    global_search_space: Dict[str, Any],
) -> None:
    print(f"PROCESSING UW DATA IN {infile_path.name}:")

    outdir_path_l2norm = params["outdir_path_l2norm"]
    params = params.copy()

    freqmin = params.get("frequency_lowcut", 1.0)

    uw_data_handler = (
        UltrasonicDataHandler.read_tsv(infile_path)
        .remove_mean()
        .downsample_waveforms(params["number_of_waveforms2process"])
        .truncate_time(params["maxtime2simulate"])
        .bandpass(freqmin=freqmin, freqmax=params["frequency_cutoff"])
    )

    observed_waveform_data = uw_data_handler.waveform_data
    observed_time = uw_data_handler.metadata["time_ax_waveform"]
    observed_waveform = np.mean(observed_waveform_data, axis=0)

    start_time = tm.time()
    outfile_name = infile_path.stem.split(".")[0]

    result = process_waveform(
        idx_waveform=0,
        observed_waveform=observed_waveform,
        observed_time=observed_time,
        stf_handler=stf_handler,
        params=params,
        assembly_dict=assembly_dict,
        global_search_space=global_search_space,
        outfile_name=outfile_name,
    )

    print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---")

    results_pkl = (outdir_path_l2norm / outfile_name).with_suffix(".pkl")
    with open(results_pkl, "wb") as f:
        pickle.dump(
            {"reference_calibration_result": result, "params": params},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    if params.get("save_calibrated_STF", False):
        stf_out_path = Path(params["calibrated_stf_outpath"])
        out_name = params.get("calibrated_stf_name", f"{outfile_name}_effective_stf")

        stf_save_handler = clone_stf_handler_with_waveform(
            stf_handler=stf_handler,
            new_waveform=result["local_source_receiver_result"]["final_source_time_function"],
        )

        stf_save_handler.write_stf_hdf5(
            out_path=stf_out_path,
            stf_name=out_name,
            cycle_index=0,
            include_active_stub=False,
            include_passive_stub=False,
            t0_unix_s=None,
        )


################### MAIN ############################
if __name__ == "__main__":

    from pathlib import Path
    from lab_uw.directory_manager import DirectoryManager
    from lab_uw.data_io.data_io import UltrasonicDataHandler, BlockMetadataHandler

    dir_manager = DirectoryManager()

    # ------------------------------------------------------------
    # BASIC EXPERIMENT INFO
    # ------------------------------------------------------------
    machine_name = "on_bench"
    experiment_name = "STF_cilinder_vs_transversal_pp5_05"
    wave_type = "_p"
    data_type_uw = f"uw_data/data_tsv_files{wave_type}"
    outfolder_name = "2025-05-03_reference_calibration_6_with_local_SR_94"

    # ------------------------------------------------------------
    # OUTPUT DIRECTORIES
    # ------------------------------------------------------------
    outdir_path_l2norm = dir_manager.make_data_analysis_folders(
        machine_name=machine_name,
        experiment_name=experiment_name,
        data_types=[outfolder_name]
    )
    outdir_path_image = dir_manager.make_data_analysis_folders(
        machine_name=machine_name,
        experiment_name=experiment_name,
        data_types=[outfolder_name + "_images"]
    )

    # ------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------
    params = {
        "absorbing": False,
        "maxtime2simulate": 25,
        "frequency_lowcut": 1.0,
        "frequency_cutoff": 6.0,
        "min_velocity2simulate": 0.4,
        "max_velocity2simulate": 0.6,
        "plot_save_interval": 1,
        "movie_save_interval": 1000,
        "number_of_waveforms2process": 1,
        "outdir_path_l2norm": outdir_path_l2norm[0],
        "outdir_path_image": outdir_path_image[0],
        "max_workers": max(1, cpu_count() - 1),

        # weak spectral tie-breaker for reference calibration
        "beta_spec_reference": 0.001,
        "spec_fmin_reference": None,
        "spec_fmax_reference": None,

        # empirical packet-decay correction calibrated from observation
        "apply_empirical_decay_correction_reference": True,
        "reference_calibration_dir": str(
            dir_manager.paths.analysis_root("on_bench", "STF_cilinder_vs_transversal_pp5_05")
            / f"stf_reference_calibration{wave_type}"
        ),

        # save best effective STF
        "save_calibrated_STF": True,
        "calibrated_stf_outpath": str(Path(outdir_path_l2norm[0]) / "calibrated_stf.h5"),
        "calibrated_stf_name": f"{experiment_name}{wave_type}_effective_stf",

        # new stage
        "run_local_source_receiver_inversion": True,
        "local_source_receiver_inversion": {
            "n_iterations": 100,
            "dw_max_start": 200,
            "dw_threshold": 0.01,
            "ds_max_start": 0.02,
            "ds_threshold": 0,
            "stf_duration": 4.0,
            "reduce_factor": 5/4,
            "misfit_threshold": 1.0,
            "smooth_gradient": True,
            "gradient_smoothing_halfwidth": 2,
            "normalize_waveform": True,
            "enable_plotting": True,
        },
    }

    # ------------------------------------------------------------
    # GLOBAL SEARCH SPACE
    # ------------------------------------------------------------
    global_search_space = {
        "num_iterations": 50000,

        "steel_velocity_low": 0.5750,
        "steel_velocity_high": 0.6000,
        "pzt_velocity_low": 0.4070,
        "pzt_velocity_high": 0.6070,

        "spreading_factor_low": 0.1,
        "spreading_factor_high": 1.0,
        "position2edge_low": -0.5,
        "position2edge_high": 0.5,
        "radius_factor_low": 1,
        "radius_factor_high": 1.0,

        "min_multiplier": 1e-4,
        "max_multiplier": 1e-2,

        "tau_shift_low_us": 0,
        "tau_shift_high_us": 0.,
        "c1_d1_low":  0.,
        "c1_d1_high": 0.,
        "c2_d2_low": 0.,
        "c2_d2_high": 0.,
    }

    # ------------------------------------------------------------
    # BLOCK METADATA
    # ------------------------------------------------------------
    blocks_json = Path(dir_manager.base_dir) / "metadata" / "blocks_metadata.json"
    block = BlockMetadataHandler.load_blocks_metadata(
        blocks_json,
        ("on_bench_STF2",)
    )

    assembly_dict = block[0]
    assembly_dict["wave_type"] = wave_type
    assembly_dict["transmitter_position"] = 0.0
    assembly_dict["receiver_position"] = assembly_dict["z"]
    assembly_dict["sample_dimensions"] = [assembly_dict["z"]]

    # ------------------------------------------------------------
    # STF TO PERTURB AROUND
    # ------------------------------------------------------------
    stf_h5_path = (
        dir_manager.paths.analysis_root("on_bench", "STF_cilinder_vs_transversal_pp5_05")
        / f"source_time_functions{wave_type}"
        / "stf.h5"
    )

    stf_handler = UltrasonicDataHandler.read_stf_hdf5(
        stf_h5_path,
        name="STF_cilinder_vs_transversal_pp5_05",
        cycle_index=0,
    )
    stf_handler.waveform_data = np.asarray(stf_handler.waveform_data).squeeze(0)

    # ------------------------------------------------------------
    # INPUT UW FILES
    # ------------------------------------------------------------
    uw_dir = (
        dir_manager.paths.experiment_root(machine_name, experiment_name)
        / data_type_uw
    )

    infile_paths = sorted(uw_dir.glob("*.tsv"))

    if not infile_paths:
        raise FileNotFoundError(f"No TSV files found in {uw_dir}")

    for infile_path in infile_paths:
        process_uw_file(
            infile_path=infile_path,
            stf_handler=stf_handler,
            params=params,
            assembly_dict=assembly_dict.copy(),
            global_search_space=global_search_space,
        )