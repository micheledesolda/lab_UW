# identify_effective_source_wavelet.py
import os
os.environ["MPLBACKEND"] = "TkAgg"   # or "Qt5Agg" if you installed PyQt5

import matplotlib
matplotlib.use("TkAgg", force=True)

import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from lab_uw.directory_manager import DirectoryManager
from lab_uw.data_io.data_io import UltrasonicDataHandler, BlockMetadataHandler
from lab_uw.data_io.schema import UWFrame
from lab_uw.data_io.writers.hdf5_writer import write_stf_hdf5
from lab_uw.plotting import InteractivePlotter, Plotter
import pickle

def pick_direct_arrival(observed_time: np.ndarray, waveform: np.ndarray) -> tuple[float, float]:
    """
    Lets the user pick two times: start and end of the direct arrival.

    Parameters
    ----------
    observed_time : np.ndarray
        1D array of time samples.
    waveform : np.ndarray
        Single 1D waveform.

    Returns
    -------
    (t_start_direct, t_end_direct) : (float, float)
        The two picked times in ascending order.
    """
    import matplotlib
    print("Matplotlib backend:", matplotlib.get_backend())

    picks = InteractivePlotter().manual_pick_arrival_times(
        observed_time=observed_time,
        observed_waveform=waveform,
        start_time=0.0,
        outfile_path=None
    )
    if len(picks) < 2:
        sys.exit("Not enough picks made. Please re-run and pick at least 2 times.")
    a, b = sorted(picks)
    return a, b

def compute_reflections_arrival_time(
    t_start_direct: float,
    direct_arrival_span: float,
    max_time: float
) -> list[float]:
    """
    Identifies the direct arrival and its zero-offset multiples.

    Model:
        direct arrival start = 1 * t_start_direct
        1st reflection start = 3 * t_start_direct
        2nd reflection start = 5 * t_start_direct
        ...
    """
    arrivals = [t_start_direct]
    tA = 3.0 * t_start_direct
    while tA + direct_arrival_span <= max_time:
        arrivals.append(tA)
        tA += 2.0 * t_start_direct
    return arrivals

def compute_reflections_correlation(
    direct_wave_data: np.ndarray,
    idx_Dstart: int,
    idx_Dend: int,
    arrivals: list[float],
    observed_time: np.ndarray,
    waveform: np.ndarray
) -> list[dict]:
    """
    Computes reflection snippets for each reflection arrival, then builds
    a cross-correlation matrix among the direct wave (index 0) and
    all reflection wave snippets (indices 1..N).

    Parameters
    ----------
    direct_wave_data : np.ndarray
        The snippet of the direct wave (1D).
    idx_Dstart : int
        The start index of the direct arrival in 'observed_time'.
    idx_Dend : int
        The end index of the direct arrival in 'observed_time'.
    arrivals : list[float]
        The arrival times (index 0 is direct or reference, so we skip it for reflection).
        Typically, arrivals[0] is the direct arrival time, but we treat that as "reference."
    observed_time : np.ndarray
        The full time axis for the entire waveform.
    waveform : np.ndarray
        The entire 1D waveform data.

    Returns
    -------
    reflection_info_list : list of dict
        Each dict has:
          {
            'arrival_time': float,
            'reflection_time': np.ndarray,  # snippet time array
            'reflection_data': np.ndarray,  # snippet wave array
            'corr_coeff': float             # correlation direct wave-reflection

          }

    """

    reflection_info_list = []
    direct_len = idx_Dend - idx_Dstart
    for i, arr_time in enumerate(arrivals):
        if i == 0:
            # The arrivals[0] is the direct arrival time -> skip
            continue

        ref_start_idx = np.searchsorted(observed_time, arr_time)
        ref_end_idx   = ref_start_idx + direct_len
        if ref_end_idx > len(observed_time):
            # reflection snippet out of bounds; skip
            break
        reflection_time = observed_time[ref_start_idx:ref_end_idx]
        reflection_data = waveform[ref_start_idx:ref_end_idx]

        # compute correlation
        if len(direct_wave_data) == len(reflection_data):
            cmat = np.corrcoef(direct_wave_data, reflection_data)
            corr_coeff = cmat[0, 1]
        else:
            corr_coeff = float('nan')

        reflection_info_list.append({
            'arrival_time': arr_time,
            'reflection_time': reflection_time,
            'reflection_data': reflection_data,
            'corr_coeff': corr_coeff
        })

    return reflection_info_list


def iterate_direct_arrival_times(
    observed_time: np.ndarray,
    waveform: np.ndarray,
    t_start_nominal: float,
    t_end_fixed: float,
    search_margin: float,
    step_size: float,
    max_time: float
) -> dict:
    """
    Search around a nominal direct-arrival start time to maximize correlation
    with the first reflection. Reflection starts are constrained to 3t, 5t, ...

    Returns
    -------
    dict
        {
          "best_t_start": float,
          "best_t_end": float,
          "best_reflection_spacing": float,
          "best_corr_score": float,
          "best_reflection_info_list": list[dict],
        }
    """
    from numpy import inf

    best_t_start = t_start_nominal
    best_t_end = t_end_fixed
    best_reflection_spacing = 2.0 * t_start_nominal
    best_corr_score = -inf
    best_reflection_info_list = []

    # Candidate direct starts
    t_min = max(0.0, t_start_nominal - search_margin)
    t_max = min(t_end_fixed - step_size, t_start_nominal + search_margin)
    candidate_t_starts = np.arange(t_min, t_max, step_size)
    if len(candidate_t_starts) == 0:
        candidate_t_starts = np.array([t_start_nominal])

    evaluated_candidates = []

    direct_arrival_span = t_end_fixed - t_start_nominal

    for candidate_t_start in candidate_t_starts:
        candidate_t_end = candidate_t_start + direct_arrival_span
        candidate_span = direct_arrival_span
        if candidate_span <= 0:
            continue
        if candidate_t_end > max_time:
            continue

        idxD_start = np.searchsorted(observed_time, candidate_t_start)
        idxD_end = np.searchsorted(observed_time, candidate_t_end)
        direct_data = waveform[idxD_start:idxD_end]

        if direct_data.size == 0:
            continue

        arrivals = compute_reflections_arrival_time(
            t_start_direct=candidate_t_start,
            direct_arrival_span=candidate_span,
            max_time=max_time
        )

        reflection_info_list = compute_reflections_correlation(
            direct_wave_data=direct_data,
            idx_Dstart=idxD_start,
            idx_Dend=idxD_end,
            arrivals=arrivals,
            observed_time=observed_time,
            waveform=waveform
        )

        usable = [r['corr_coeff'] for r in reflection_info_list if not np.isnan(r['corr_coeff'])]
        if len(usable) == 0:
            continue

        # The first reflection at 3t is the least contaminated and has the
        # highest signal-to-noise ratio. Later packets are diagnostics only.
        corr_score = float(usable[0])

        evaluated_candidates.append({
            "t_start": candidate_t_start,
            "t_end": candidate_t_end,
            "spacing": 2.0 * candidate_t_start,
            "corr_score": corr_score,
            "reflection_info_list": reflection_info_list,
        })

    if len(evaluated_candidates) == 0:
        return {
            "best_t_start": best_t_start,
            "best_t_end": best_t_end,
            "best_reflection_spacing": best_reflection_spacing,
            "best_corr_score": best_corr_score,
            "best_reflection_info_list": best_reflection_info_list,
        }

    best_candidate = max(
        evaluated_candidates,
        key=lambda c: c["corr_score"],
    )

    best_t_start = best_candidate["t_start"]
    best_t_end = best_candidate["t_end"]
    best_reflection_spacing = best_candidate["spacing"]
    best_corr_score = best_candidate["corr_score"]
    best_reflection_info_list = best_candidate["reflection_info_list"]

    return {
        "best_t_start": best_t_start,
        "best_t_end": best_t_end,
        "best_reflection_spacing": best_reflection_spacing,
        "best_corr_score": best_corr_score,
        "best_reflection_info_list": best_reflection_info_list,
    }

@dataclass
class ReflectionDecayResult:
    arrival_times: np.ndarray
    picked_times: np.ndarray
    amplitudes_signed: np.ndarray
    amplitudes_envelope: np.ndarray
    distances_cm: np.ndarray
    order: np.ndarray

def extract_packet_amplitudes_from_arrivals(
    observed_time: np.ndarray,
    waveform: np.ndarray,
    arrivals: list[float],
    thickness_cm: float,
    *,
    window_span: float,
) -> ReflectionDecayResult:
    """
    For each arrival time, search from the arrival onset to onset + span and extract:
    - signed waveform peak at the envelope maximum location
    - envelope amplitude
    - traveled distance assuming through-transmission zero-angle sequence

    Parameters
    ----------
    arrivals : list[float]
        Output of your existing compute_reflections_arrival_time().
        Must include the direct arrival as arrivals[0].
    window_span : float
        Full width of the search window [same units as observed_time].
    """
    observed_time = np.asarray(observed_time, dtype=float)
    waveform = np.asarray(waveform, dtype=float)
    env = np.abs(hilbert(waveform))

    picked_times = []
    amp_signed = []
    amp_env = []

    for tarr in arrivals:
        i0 = np.searchsorted(observed_time, tarr)
        i1 = np.searchsorted(observed_time, tarr + window_span)

        i0 = max(0, i0)
        i1 = min(observed_time.size, i1)

        if i1 <= i0:
            picked_times.append(np.nan)
            amp_signed.append(np.nan)
            amp_env.append(np.nan)
            continue

        iloc = np.argmax(env[i0:i1]) + i0

        picked_times.append(observed_time[iloc])
        amp_signed.append(waveform[iloc])
        amp_env.append(env[iloc])

    picked_times = np.asarray(picked_times, dtype=float)
    amp_signed = np.asarray(amp_signed, dtype=float)
    amp_env = np.asarray(amp_env, dtype=float)

    order = np.arange(len(arrivals), dtype=int)
    distances_cm = (2 * order + 1) * float(thickness_cm)

    return ReflectionDecayResult(
        arrival_times=np.asarray(arrivals, dtype=float),
        picked_times=picked_times,
        amplitudes_signed=amp_signed,
        amplitudes_envelope=amp_env,
        distances_cm=distances_cm,
        order=order,
    )

def fit_geometric_decay(order: np.ndarray, amplitudes: np.ndarray, eps: float = 1e-12):
    mask = np.isfinite(order) & np.isfinite(amplitudes) & (amplitudes > eps)
    x = np.asarray(order[mask], dtype=float)
    y = np.log(np.asarray(amplitudes[mask], dtype=float))

    if x.size < 2:
        return np.nan, np.nan

    m, b = np.polyfit(x, y, 1)
    A0 = np.exp(b)
    g = np.exp(m)
    return A0, g

def fit_power_law_decay(distances_cm: np.ndarray, amplitudes: np.ndarray, eps: float = 1e-12):
    mask = np.isfinite(distances_cm) & np.isfinite(amplitudes) & (distances_cm > 0) & (amplitudes > eps)
    x = np.log(np.asarray(distances_cm[mask], dtype=float))
    y = np.log(np.asarray(amplitudes[mask], dtype=float))

    if x.size < 2:
        return np.nan, np.nan

    m, b = np.polyfit(x, y, 1)
    C = np.exp(b)
    p = -m
    return C, p

def fit_exponential_decay(distances_cm: np.ndarray, amplitudes: np.ndarray, eps: float = 1e-12):
    mask = np.isfinite(distances_cm) & np.isfinite(amplitudes) & (amplitudes > eps)
    x = np.asarray(distances_cm[mask], dtype=float)
    y = np.log(np.asarray(amplitudes[mask], dtype=float))

    if x.size < 2:
        return np.nan, np.nan

    m, b = np.polyfit(x, y, 1)
    A0 = np.exp(b)
    alpha = -m
    return A0, alpha

def plot_arrivals_and_picked_amplitudes(
    observed_time: np.ndarray,
    waveform: np.ndarray,
    decay_result: ReflectionDecayResult,
    *,
    window_span: float,
    outfile_path: Path | None = None,
):
    env = np.abs(hilbert(waveform))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(observed_time, waveform, label="Waveform")
    ax.plot(observed_time, env, label="Envelope", alpha=0.8)

    for tarr, tpick in zip(decay_result.arrival_times, decay_result.picked_times):
        ax.axvline(tarr, linestyle="--", alpha=0.35)
        ax.axvspan(tarr, tarr + window_span, alpha=0.08)
        if np.isfinite(tpick):
            ax.axvline(tpick, linestyle="-", alpha=0.6)

    ax.scatter(
        decay_result.picked_times,
        decay_result.amplitudes_envelope,
        s=35,
        zorder=5,
        label="Picked envelope peaks",
    )

    ax.set_xlabel("Time [$\\mu$s]")
    ax.set_ylabel("Amplitude [a.u.]")
    ax.set_title("Arrival windows and picked packet amplitudes")
    ax.legend()

    if outfile_path is not None:
        fig.savefig(outfile_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

def plot_decay_fits(
    decay_result: ReflectionDecayResult,
    outfile_path: Path | None = None,
):
    order = decay_result.order
    dist = decay_result.distances_cm
    amp = decay_result.amplitudes_envelope

    A0_geo, g = fit_geometric_decay(order, amp)
    C_pow, p = fit_power_law_decay(dist, amp)
    A0_exp, alpha = fit_exponential_decay(dist, amp)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].scatter(order, amp, s=40, label="Measured")
    if np.isfinite(A0_geo) and np.isfinite(g):
        nfit = np.linspace(order.min(), order.max(), 200)
        axs[0].plot(nfit, A0_geo * (g ** nfit), label=f"A0 g^n, g={g:.4f}")
    axs[0].set_xlabel("Packet order")
    axs[0].set_ylabel("Envelope amplitude [a.u.]")
    axs[0].set_title("Decay vs order")
    axs[0].legend()

    axs[1].scatter(dist, amp, s=40, label="Measured")
    if np.isfinite(C_pow) and np.isfinite(p):
        dfit = np.linspace(dist.min(), dist.max(), 300)
        axs[1].plot(dfit, C_pow * dfit**(-p), label=f"C L^(-p), p={p:.4f}")
    if np.isfinite(A0_exp) and np.isfinite(alpha):
        dfit = np.linspace(dist.min(), dist.max(), 300)
        axs[1].plot(dfit, A0_exp * np.exp(-alpha * dfit), label=f"A0 exp(-αL), α={alpha:.4f}")
    axs[1].set_xlabel("Traveled distance [cm]")
    axs[1].set_ylabel("Envelope amplitude [a.u.]")
    axs[1].set_title("Decay vs traveled distance")
    axs[1].legend()

    if outfile_path is not None:
        fig.savefig(outfile_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def make_json_serializable(obj):
    """
    Recursively convert numpy objects into JSON-serializable Python objects.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def save_reference_calibration(
    out_path: Path,
    calibration_dict: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_dict = make_json_serializable(calibration_dict)

    if out_path.suffix.lower() == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(calibration_dict, f, indent=2)
    elif out_path.suffix.lower() == ".pkl":
        with open(out_path, "wb") as f:
            pickle.dump(calibration_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        raise ValueError(f"Unsupported calibration file suffix: {out_path.suffix}")
    
def build_reference_calibration_dict(
    *,
    infile_stem: str,
    t_start_direct_picked: float,
    t_end_direct_picked: float,
    best_t_start: float,
    best_t_end: float,
    effective_wavelet_start: float,
    best_reflection_spacing: float,
    final_arrivals: list[float],
    final_span: float,
    thickness_cm: float,
    v_est: float,
    decay_result: ReflectionDecayResult,
    best_corr_score: float,
    reflection_info_list: list[dict],
    geo_stats: dict | None,
    pow_stats: dict | None,
    exp_stats: dict | None,
) -> dict:
    comparable_decay_models = [
        stats for stats in (pow_stats, exp_stats)
        if stats is not None and np.isfinite(stats.get("aicc", np.nan))
    ]
    selected_decay = (
        min(comparable_decay_models, key=lambda stats: stats["aicc"])
        if comparable_decay_models else None
    )

    if selected_decay is None:
        selected_decay_parameters = None
    elif selected_decay["model"] == "power_law":
        selected_decay_parameters = {
            "model": "power_law",
            "exponent_p": float(selected_decay["p"]),
            "coefficient_C": float(selected_decay["C"]),
        }
    else:
        selected_decay_parameters = {
            "model": "exponential",
            "alpha_per_cm": float(selected_decay["alpha"]),
            "amplitude_A0": float(selected_decay["A0"]),
        }

    return {
        "infile_stem": infile_stem,

        "picked_direct_arrival": {
            "t_start_direct_picked_us": float(t_start_direct_picked),
            "t_end_direct_picked_us": float(t_end_direct_picked),
        },

        "refined_direct_arrival": {
            "best_t_start_us": float(best_t_start),
            "best_t_end_us": float(best_t_end),
            "effective_wavelet_start_us": float(effective_wavelet_start),
            "direct_arrival_span_us": float(final_span),
            "reflection_spacing_us": float(best_reflection_spacing),
        },

        "zero_angle_packets": {
            "arrival_times_us": [float(x) for x in final_arrivals],
            "n_packets": int(len(final_arrivals)),
            "packet_window_span_us": float(final_span),
            "packet_half_width_us": float(0.5 * final_span),
        },

        "correlation": {
            "selection_basis": "first_reflection_at_3t",
            "best_score": float(best_corr_score),
            "per_reflection": [
                {
                    "arrival_time_us": float(info["arrival_time"]),
                    "corr_coeff": float(info["corr_coeff"]),
                }
                for info in reflection_info_list
            ],
        },

        "reference_block": {
            "thickness_cm": float(thickness_cm),
            "estimated_velocity_cm_per_us": float(v_est),
        },

        "packet_decay_measurement": {
            "picked_times_us": decay_result.picked_times.tolist(),
            "amplitudes_signed": decay_result.amplitudes_signed.tolist(),
            "amplitudes_envelope": decay_result.amplitudes_envelope.tolist(),
            "distances_cm": decay_result.distances_cm.tolist(),
            "order": decay_result.order.tolist(),
        },

        "decay_models": {
            "geometric": geo_stats,
            "power_law": pow_stats,
            "exponential": exp_stats,
        },

        "decay_model_comparison_note": (
            "geometric and exponential are equivalent here because traveled "
            "distance is affine in packet order; model selection compares "
            "power_law with exponential only"
        ),
        "selected_decay_model": (
            selected_decay["model"] if selected_decay is not None else None
        ),
        "selected_decay_parameters": selected_decay_parameters,

        "proposed_misfit_construction": {
            "use_multiwindow_zero_angle_packets": True,
            "apply_empirical_packet_decay_correction": True,
            "fit_trace_scale_after_decay_correction": True,
        },
    }

#################################################################################################
def main():
    base_dir = "/home/michele/Desktop/Dottorato/active_source_implementation"
    machine_name = "on_bench"
    experiment_name = "STF_pfake500_s250_multichannel"
    data_type = "uw_data/data_tsv_files/only"
    wave_type = "_s"
    block_metadata_filename = "blocks_metadata.json"
    block_id = "on_bench_STF2"
    # Guess or define a margin, step_size
    search_margin = 1    # [mus] +/- interval search around the user pick
    step_size = 0.01     # [mus] increments step. At best should be the sampling frequency
    freqmin = 0.1
    freqmax = 6.0
    
    dir_manager = DirectoryManager(base_dir=base_dir)

    # block metadata (thickness etc.)
    blocks_json = Path(dir_manager.base_dir) / "metadata" / block_metadata_filename
    block_params, = BlockMetadataHandler.load_blocks_metadata(
    blocks_json, (block_id,)
    )

    # input files
    infile_path_list = sorted(dir_manager.make_infile_path_list(
        machine_name, experiment_name, data_type=data_type + wave_type
    ))

    # Use only one file for manual picking, then the same guess will be used for all the others

    # output roots
    out_types = [
        "source_time_functions" + wave_type,
        "stf_images_reflection_windows" + wave_type,
        "stf_images_direct-reflections_correlation" + wave_type,
        "stf_images_decay_analysis" + wave_type,
        "stf_reference_calibration" + wave_type,
    ]
    outdir_data, outdir_win, outdir_overlay, outdir_decay, outdir_calibration = dir_manager.make_data_analysis_folders(
        machine_name=machine_name, experiment_name=experiment_name, data_types=out_types
    )

    # single HDF5 for all STFs of this experiment+wave
    stf_h5_path = Path(outdir_data) / "stf.h5"

    # do the manual pick once on a single randomly chosen file
    file_manual_pick_idx = int(np.random.choice(len(infile_path_list)))
    manual_infile_path = infile_path_list[file_manual_pick_idx]

    uw_pick = UltrasonicDataHandler.read_tsv(manual_infile_path).bandpass(
        freqmin=freqmin,
        freqmax=freqmax,
    )
    data_pick, meta_pick = uw_pick.waveform_data, uw_pick.metadata
    observed_time_pick = meta_pick["time_ax_waveform"]  # [µs]
    waveform_pick = data_pick.mean(axis=0)              # average all waveforms

    t_start_direct_picked, t_end_direct_picked = pick_direct_arrival(
        observed_time_pick, waveform_pick
    )
    direct_arrival_span = t_end_direct_picked - t_start_direct_picked
    
    print(
        f"Picked direct arrival on {manual_infile_path.stem}: "
        f"start={t_start_direct_picked:.2f}, end={t_end_direct_picked:.2f}"
    )

    for infile_path in infile_path_list:
        
        # Load the single file and get the wave/time
        uw = UltrasonicDataHandler.read_tsv(infile_path).bandpass(
            freqmin=freqmin,
            freqmax=freqmax,
        )
        data, meta = uw.waveform_data, uw.metadata
        observed_time = meta["time_ax_waveform"]  # [µs]
        waveform = data.mean(axis=0)              # average all waveforms
        
        results = iterate_direct_arrival_times(
            observed_time=observed_time,
            waveform=waveform,
            t_start_nominal=t_start_direct_picked,
            t_end_fixed=t_end_direct_picked,
            search_margin=search_margin,
            step_size=step_size,
            max_time=observed_time[-1]
        )

        best_t_start = results["best_t_start"]
        best_t_end = results["best_t_end"]
        best_reflection_spacing = results["best_reflection_spacing"]
        best_reflection_info_list = results["best_reflection_info_list"]
        best_corr_score = results["best_corr_score"]

        # Zero-offset geometry: reflections start at 3t, 5t, 7t, ... .
        effective_wavelet_start = best_t_start
        final_span = best_t_end - effective_wavelet_start

        print(
            f"{infile_path.stem}: "
            f"best t_start = {best_t_start:.3f} µs; "
            f"effective wavelet start = {effective_wavelet_start:.3f} µs; "
            f"spacing = {best_reflection_spacing:.3f} µs; "
            f"corr = {best_corr_score:.4f}"
        )
        
        # Effective source wavelet: measured direct-arrival packet.  The HDF5
        # writer still uses the historical ``stf`` schema for compatibility.
        i0 = int(np.searchsorted(observed_time, effective_wavelet_start))
        i1 = int(np.searchsorted(observed_time, best_t_end))
        effective_wavelet_data = waveform[i0:i1]
        effective_wavelet_time = observed_time[i0:i1]

        final_arrivals = compute_reflections_arrival_time(
            t_start_direct=effective_wavelet_start,
            direct_arrival_span=final_span,
            max_time=observed_time[-1]
        )

        best_reflection_info_list = compute_reflections_correlation(
            direct_wave_data=effective_wavelet_data,
            idx_Dstart=i0,
            idx_Dend=i1,
            arrivals=final_arrivals,
            observed_time=observed_time,
            waveform=waveform
        )

        # Build the frame using the legacy STF schema expected downstream.
        stf_meta = dict(meta)  # copy
        stf_meta["number_of_samples"] = int(effective_wavelet_data.size)
        stf_meta["number_of_waveforms"] = 1
        stf_meta["time_ax_waveform"] = effective_wavelet_time
        stf_meta["product_type"] = "effective_source_wavelet"
        stf_meta["wavelet_start_us"] = float(effective_wavelet_start)
        stf_meta["reflection_spacing_us"] = float(best_reflection_spacing)

        stf_frame = UWFrame(effective_wavelet_data[None, :], stf_meta)

        # --- save STF into HDF5 under /stf/<name>/cycle_000000 ---
        original_infile_path = infile_path
        while infile_path.suffix:
            infile_path = infile_path.with_suffix('')
        write_stf_hdf5(
            stf=stf_frame,
            out_path=stf_h5_path,
            stf_name=infile_path.stem,
            cycle_index=0,
            include_active_stub=False,
            include_passive_stub=False,
            t0_unix_s=None,
        )

        # Plot final reflection windows
        plotter = Plotter()
        try:
            plotter.plot_reflection_windows(
                observed_time=observed_time,
                waveform=waveform,
                t_start_direct=effective_wavelet_start,
                t_end_direct=best_t_end,
                reflection_info_list=best_reflection_info_list,
                idx_Dstart=i0,
                idx_Dend=i1,
                title=f"Reflection windows: {infile_path.stem}",
                outfile_path=Path(outdir_win) / infile_path.stem,
            )
        except Exception as e:
            print("Window plot error:", e)

        try:
            plotter.plot_direct_and_reflections(
                direct_wave_time=effective_wavelet_time,
                direct_wave_data=effective_wavelet_data,
                reflection_info_list=best_reflection_info_list,
                outfile_path=Path(outdir_overlay) / infile_path.stem,
            )
        except Exception as e:
            print("Overlay plot error:", e)

        # velocity (simple estimate)
        thickness_cm = float(block_params["z"])
        one_way_time = 0.5 * best_reflection_spacing
        v_est = thickness_cm / one_way_time
        print(
            f"Estimated velocity for {infile_path.stem}: {v_est:.4f} cm/µs "
            f"(one-way time = {one_way_time:.3f} µs)"
        )

        decay_result = extract_packet_amplitudes_from_arrivals(
            observed_time=observed_time,
            waveform=waveform,
            arrivals=final_arrivals,
            thickness_cm=thickness_cm,
            window_span=final_span,
        )

        print("arrival times [µs]:", decay_result.arrival_times)
        print("picked times [µs]: ", decay_result.picked_times)
        print("envelope amplitudes:", decay_result.amplitudes_envelope)
        print("distances [cm]:", decay_result.distances_cm)

        A0_geo, g = fit_geometric_decay(decay_result.order, decay_result.amplitudes_envelope)
        C_pow, p = fit_power_law_decay(decay_result.distances_cm, decay_result.amplitudes_envelope)
        A0_exp, alpha = fit_exponential_decay(decay_result.distances_cm, decay_result.amplitudes_envelope)

        print(f"Geometric fit:   A_n = A0 g^n      -> A0={A0_geo:.6g}, g={g:.6g}")
        print(f"Power-law fit:   A(L) = C L^(-p)   -> C={C_pow:.6g}, p={p:.6g}")
        print(f"Exponential fit: A(L) = A0 e^(-aL) -> A0={A0_exp:.6g}, alpha={alpha:.6g}")

        geo_stats = evaluate_geometric_decay_fit(
            decay_result.order,
            decay_result.amplitudes_envelope
        )
        pow_stats = evaluate_power_law_decay_fit(
            decay_result.distances_cm,
            decay_result.amplitudes_envelope
        )
        exp_stats = evaluate_exponential_decay_fit(
            decay_result.distances_cm,
            decay_result.amplitudes_envelope
        )

        for stats in [geo_stats, pow_stats, exp_stats]:
            if stats is None:
                continue
            print(
                f"{stats['model']}: "
                f"R2={stats['r2']:.5f}, "
                f"RMSE={stats['rmse']:.5e}, "
                f"AICc={stats['aicc']:.5f}, "
                f"BIC={stats['bic']:.5f}"
            )
        calibration_dict = build_reference_calibration_dict(
            infile_stem=infile_path.stem,
            t_start_direct_picked=t_start_direct_picked,
            t_end_direct_picked=t_end_direct_picked,
            best_t_start=best_t_start,
            best_t_end=best_t_end,
            effective_wavelet_start=effective_wavelet_start,
            best_reflection_spacing=best_reflection_spacing,
            final_arrivals=final_arrivals,
            final_span=final_span,
            thickness_cm=thickness_cm,
            v_est=v_est,
            decay_result=decay_result,
            best_corr_score=best_corr_score,
            reflection_info_list=best_reflection_info_list,
            geo_stats=geo_stats,
            pow_stats=pow_stats,
            exp_stats=exp_stats,
        )

        save_reference_calibration(
            out_path=Path(outdir_calibration) / f"{infile_path.stem}_reference_calibration.json",
            calibration_dict=calibration_dict,
        )
        try:
            plot_arrivals_and_picked_amplitudes(
                observed_time=observed_time,
                waveform=waveform,
                decay_result=decay_result,
                window_span=final_span,
                outfile_path=Path(outdir_decay) / f"{infile_path.stem}_arrivals.png",
            )
        except Exception as e:
            print("Arrival/amplitude plot error:", e)

        try:
            plot_decay_fits(
                decay_result=decay_result,
                outfile_path=Path(outdir_decay) / f"{infile_path.stem}_decay_fit.png",
            )
        except Exception as e:
            print("Decay fit plot error:", e)

        infile_path = original_infile_path

def evaluate_linearized_fit(x: np.ndarray, y: np.ndarray, m: float, b: float):
    """
    Evaluate fit quality for y = b + m x.

    Returns
    -------
    dict with:
      rss   : residual sum of squares
      rmse  : root mean square error
      r2    : coefficient of determination
      aic   : Akaike information criterion
      aicc  : small-sample corrected AIC
      bic   : Bayesian information criterion
      yhat  : fitted values
      resid : residuals
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    yhat = b + m * x
    resid = y - yhat
    rss = float(np.sum(resid**2))
    n = y.size
    k = 2  # slope + intercept

    rmse = float(np.sqrt(rss / n)) if n > 0 else np.nan

    tss = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan

    if n > 0 and rss > 0:
        aic = float(n * np.log(rss / n) + 2 * k)
        bic = float(n * np.log(rss / n) + k * np.log(n))
    else:
        aic = np.nan
        bic = np.nan

    if n > k + 1 and np.isfinite(aic):
        aicc = float(aic + (2 * k * (k + 1)) / (n - k - 1))
    else:
        aicc = np.nan

    return {
        "rss": rss,
        "rmse": rmse,
        "r2": r2,
        "aic": aic,
        "aicc": aicc,
        "bic": bic,
        "yhat": yhat,
        "resid": resid,
    }

def evaluate_geometric_decay_fit(order: np.ndarray, amplitudes: np.ndarray, eps: float = 1e-12):
    mask = np.isfinite(order) & np.isfinite(amplitudes) & (amplitudes > eps)
    x = np.asarray(order[mask], dtype=float)
    y = np.log(np.asarray(amplitudes[mask], dtype=float))

    if x.size < 3:
        return None

    m, b = np.polyfit(x, y, 1)
    stats = evaluate_linearized_fit(x, y, m, b)
    stats["model"] = "geometric"
    stats["A0"] = float(np.exp(b))
    stats["g"] = float(np.exp(m))
    return stats


def evaluate_power_law_decay_fit(distances_cm: np.ndarray, amplitudes: np.ndarray, eps: float = 1e-12):
    mask = np.isfinite(distances_cm) & np.isfinite(amplitudes) & (distances_cm > 0) & (amplitudes > eps)
    x = np.log(np.asarray(distances_cm[mask], dtype=float))
    y = np.log(np.asarray(amplitudes[mask], dtype=float))

    if x.size < 3:
        return None

    m, b = np.polyfit(x, y, 1)
    stats = evaluate_linearized_fit(x, y, m, b)
    stats["model"] = "power_law"
    stats["C"] = float(np.exp(b))
    stats["p"] = float(-m)
    return stats


def evaluate_exponential_decay_fit(distances_cm: np.ndarray, amplitudes: np.ndarray, eps: float = 1e-12):
    mask = np.isfinite(distances_cm) & np.isfinite(amplitudes) & (amplitudes > eps)
    x = np.asarray(distances_cm[mask], dtype=float)
    y = np.log(np.asarray(amplitudes[mask], dtype=float))

    if x.size < 3:
        return None

    m, b = np.polyfit(x, y, 1)
    stats = evaluate_linearized_fit(x, y, m, b)
    stats["model"] = "exponential"
    stats["A0"] = float(np.exp(b))
    stats["alpha"] = float(-m)
    return stats

if __name__ == "__main__":
    main()
