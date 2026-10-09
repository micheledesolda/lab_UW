import copy
import json
import os
import pickle
import time as tm
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, Tuple

import numpy as np

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


def split_into_contiguous_segments(idxs: np.ndarray) -> list[np.ndarray]:
    idxs = np.sort(np.unique(np.asarray(idxs, dtype=int)))
    if idxs.size == 0:
        return []
    gaps = np.where(np.diff(idxs) > 1)[0]
    return np.split(idxs, gaps + 1)


def fit_trace_scale_equal_packet_weight(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    misfit_interval,
    eps: float = 1e-12,
    skip_first_segment: bool = False,
) -> float:
    """
    Global scalar fit with equal packet weight.
    Each contiguous packet window contributes equally after normalization
    by the observed packet energy.
    """
    if isinstance(misfit_interval, slice):
        idx = np.arange(*misfit_interval.indices(observed_waveform.size), dtype=int)
    else:
        idx = np.asarray(misfit_interval, dtype=int)

    segments = split_into_contiguous_segments(idx)
    if skip_first_segment and len(segments) > 1:
        segments = segments[1:]

    if len(segments) == 0:
        return 1.0

    num = 0.0
    den = 0.0

    for seg in segments:
        d = observed_waveform[seg]
        s = synthetic_waveform[seg]
        w = 1.0 / (np.dot(d, d) + eps)
        num += w * np.dot(d, s)
        den += w * np.dot(s, s)

    return float(num / (den + eps))


def compute_packet_balanced_time_misfit(
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    misfit_interval,
    eps: float = 1e-12,
) -> float:
    """
    Equal-packet-weight L2 misfit.
    Each contiguous packet contributes one normalized residual norm.
    """
    if isinstance(misfit_interval, slice):
        idx = np.arange(*misfit_interval.indices(observed_waveform.size), dtype=int)
    else:
        idx = np.asarray(misfit_interval, dtype=int)

    segments = split_into_contiguous_segments(idx)
    if len(segments) == 0:
        return np.inf

    packet_terms = []
    for seg in segments:
        d = observed_waveform[seg]
        s = synthetic_waveform[seg]
        packet_terms.append(np.linalg.norm(d - s) / (np.linalg.norm(d) + eps))

    return float(np.mean(packet_terms))


# ---------------------------------------------------------------------
# REFERENCE CALIBRATION IO
# ---------------------------------------------------------------------
def load_reference_calibration(calibration_path: Path) -> dict:
    with open(calibration_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
# OPTIONAL: WEAK COMBINED OBJECTIVE FOR GLOBAL CALIBRATION
# ---------------------------------------------------------------------
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
    *,
    beta_spec: float = 0.0,
    spec_fmin: float | None = None,
    spec_fmax: float | None = None,
    skip_first_segment_in_scale: bool = False,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    Returns:
      total_misfit, time_misfit, spec_misfit, time_scale, scaled_synthetic

    Important
    ---------
    The misfit is evaluated on the union of the direct-arrival and reflection
    windows loaded from the reference calibration. No empirical multiplicative
    decay correction is applied to the synthetic trace here.
    """
    syn = synthetic_waveform.copy()

    time_scale = fit_trace_scale_equal_packet_weight(
        observed_waveform=observed_waveform,
        synthetic_waveform=syn,
        misfit_interval=misfit_interval,
        skip_first_segment=skip_first_segment_in_scale,
    )
    syn *= time_scale

    time_misfit = compute_packet_balanced_time_misfit(
        observed_waveform=observed_waveform,
        synthetic_waveform=syn,
        misfit_interval=misfit_interval,
    )

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
        normalize_waveform=False,
        enable_plotting=False,
        make_movie=False,
        plot_output_path=None,
        movie_output_path=None,
    )

    total_misfit, time_misfit, spec_misfit, time_scale, scaled_synthetic = compute_reference_objective(
        observed_waveform=observed_waveform,
        synthetic_waveform=sim.synthetic_waveform,
        observed_time=observed_time,
        misfit_interval=misfit_interval,
        beta_spec=params.get("beta_spec_reference", 0.0),
        spec_fmin=params.get("spec_fmin_reference", None),
        spec_fmax=params.get("spec_fmax_reference", None),
        skip_first_segment_in_scale=params.get("skip_first_segment_in_scale_reference", False),
    )

    return {
        "pzt_velocity": pzt_velocity2simulate,
        "steel_velocity": steel_velocity2simulate,
        "total_misfit": total_misfit,
        "time_misfit": time_misfit,
        "spec_misfit": spec_misfit,
        "time_scale": time_scale,
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
        "synthetic_waveform_scaled": scaled_synthetic,
        "stf_waveform_trial": stf_trial_waveform,
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
    reference_calibration: dict,
) -> Dict[str, Any]:

    if isinstance(params["outdir_path_image"], list):
        outdir_path_image = Path(params["outdir_path_image"][0])
    else:
        outdir_path_image = Path(params["outdir_path_image"])

    misfit_interval = build_reference_misfit_interval_from_calibration(
        observed_time=observed_time,
        calibration_dict=reference_calibration,
    )

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
    time_array = np.array([r["time_misfit"] for r in results], dtype=float)
    spec_array = np.array([r["spec_misfit"] for r in results], dtype=float)

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
    plot_output_path = outdir_path_image / f"{outfile_name}_best_simulation" if save_plot else None

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
        normalize_waveform=False,
        enable_plotting=False,
        make_movie=False,
        plot_output_path=plot_output_path,
        movie_output_path=None,
    )

    simulation.synthetic_waveform = best["synthetic_waveform_scaled"]
    simulation.misfit = best["total_misfit"]

    if save_plot and plot_output_path is not None:
        simulation.plotter.plot_simulation_waveform(
            t=observed_time,
            sp_simulated=best["synthetic_waveform_scaled"],
            sp_recorded=observed_waveform,
            misfit_interval=misfit_interval,
            outfile_path=plot_output_path,
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
        "time_misfit_array": time_array,
        "spec_misfit_array": spec_array,
        "steel_velocity_array": steel_sorted,
        "pzt_velocity_array": pzt_sorted,
        "best_steel_velocity": best["steel_velocity"],
        "best_pzt_velocity": best["pzt_velocity"],
        "best_total_misfit": best["total_misfit"],
        "best_time_misfit": best["time_misfit"],
        "best_spec_misfit": best["spec_misfit"],
        "best_time_scale": best["time_scale"],
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
        "best_synthetic_waveform": best["synthetic_waveform_scaled"],
        "best_observed_waveform": observed_waveform.copy(),
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

    uw_data_handler = (
        UltrasonicDataHandler.read_tsv(infile_path)
        .remove_mean()
        .downsample_waveforms(params["number_of_waveforms2process"])
        .truncate_time(params["maxtime2simulate"])
        .bandpass(freqmin=1, freqmax=params["frequency_cutoff"])
    )

    observed_waveform_data = uw_data_handler.waveform_data
    observed_time = uw_data_handler.metadata["time_ax_waveform"]

    observed_waveform = np.mean(observed_waveform_data, axis=0)

    calibration_dir = Path(params["reference_calibration_dir"])
    calibration_path = calibration_dir / f"{infile_path.stem.split('.')[0]}_reference_calibration.json"
    if not calibration_path.exists():
        raise FileNotFoundError(f"Reference calibration file not found: {calibration_path}")
    reference_calibration = load_reference_calibration(calibration_path)

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
        reference_calibration=reference_calibration,
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
            new_waveform=result["best_stf_waveform"],
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
    from lab_uw.data_io.data_io import BlockMetadataHandler

    dir_manager = DirectoryManager()

    machine_name = "on_bench"
    experiment_name = "STF_cilinder_vs_transversal_pp5_05"
    wave_type = "_p"
    data_type_uw = f"uw_data/data_tsv_files{wave_type}"
    outfolder_name = "2025-reference_calibration"

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

    params = {
        "absorbing": False,
        "maxtime2simulate": None,
        "frequency_cutoff": 6.0,
        "min_velocity2simulate": 0.2,
        "max_velocity2simulate": 0.6,
        "plot_save_interval": 1,
        "movie_save_interval": 1000,
        "number_of_waveforms2process": 10,
        "outdir_path_l2norm": outdir_path_l2norm[0],
        "outdir_path_image": outdir_path_image[0],
        "max_workers": max(1, cpu_count() - 1),

        "beta_spec_reference": 0.0,
        "spec_fmin_reference": None,
        "spec_fmax_reference": None,
        "skip_first_segment_in_scale_reference": False,

        "save_calibrated_STF": True,
        "calibrated_stf_outpath": str(Path(outdir_path_l2norm[0]) / "calibrated_stf.h5"),
        "calibrated_stf_name": f"{experiment_name}{wave_type}_effective_stf",

        "reference_calibration_dir": str(
            dir_manager.paths.analysis_root(machine_name, experiment_name)
            / f"stf_reference_calibration{wave_type}"
        ),
    }

    global_search_space = {
        "num_iterations": 200,
        "steel_velocity_low": 0.5600,
        "steel_velocity_high": 0.5800,
        "pzt_velocity_low": 0.40,
        "pzt_velocity_high": 0.45,
        "spreading_factor_low": 0.05,
        "spreading_factor_high": 1.50,
        "position2edge_low": -0.25,
        "position2edge_high": 0.25,
        "radius_factor_low": 0.50,
        "radius_factor_high": 1.50,
        "min_multiplier": 0.5,
        "max_multiplier": 2.0,
        "tau_shift_low_us": -0.05,
        "tau_shift_high_us": 0.05,
        "c1_d1_low": -0.02,
        "c1_d1_high": 0.02,
        "c2_d2_low": -0.002,
        "c2_d2_high": 0.002,
    }

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

    stf_h5_path = (
        dir_manager.paths.analysis_root(machine_name, experiment_name)
        / f"source_time_functions{wave_type}"
        / "stf.h5"
    )

    stf_handler = UltrasonicDataHandler.read_stf_hdf5(
        stf_h5_path,
        name="STF_cilinder_vs_transversal_pp5_05",
        cycle_index=0,
    )
    stf_handler.waveform_data = np.asarray(stf_handler.waveform_data).squeeze(0)

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
