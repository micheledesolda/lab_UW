# lab_uw/local_optimization_velocity_homogeneus.py

import sys
from pathlib import Path
import pickle
import time as tm
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from typing import Any, Dict
from itertools import product
import matplotlib.pyplot as plt

from lab_uw.data_io.data_io import UltrasonicDataHandler, BlockMetadataHandler, MechanicalDataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.simulation_setup import *
from lab_uw.forward_modeling import *
from lab_uw.plotting import Plotter
from lab_uw.forward_modeling import UltrasonicModeler
from lab_uw.utils import *

import os

# 1) Make Matplotlib non-interactive (no X server use)
os.environ["MPLBACKEND"] = "Agg"

# 2) Prevent Numba from using TBB (which is the one complaining)
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
# optional: cap threads to avoid oversubscription
# os.environ["NUMBA_NUM_THREADS"] = "4"

def _set_start_method():
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)  # avoid fork
    except RuntimeError:
        pass  # already set

_set_start_method()

def load_pickle_stream(pkl_path: Path) -> list[dict]:
    """Read a stream of pickled dicts (multiple pickle.dump calls in one file)."""
    rows = []
    with open(pkl_path, "rb") as f:
        while True:
            try:
                rows.append(pickle.load(f))
            except EOFError:
                break
    return rows


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def plot_shear_and_avg_velocity_from_pickle(pkl_path: Path, out_png: Path) -> None:
    rows = load_pickle_stream(pkl_path)

    # Keep rows that have time + shear stress + avg velocity
    rows2 = []
    for r in rows:
        if ("acquisition_time_s" in r) and ("shear_stress_MPa" in r):
            # avg_gouge_velocity exists only after local inversion in your schema
            if "avg_gouge_velocity" in r:
                if np.isfinite(_safe_float(r["avg_gouge_velocity"])):
                    rows2.append(r)

    if not rows2:
        raise RuntimeError("No rows with avg_gouge_velocity found in pickle. Did local inversion run/save?")

    rows2.sort(key=lambda r: _safe_float(r["acquisition_time_s"]))

    t = np.array([_safe_float(r["acquisition_time_s"]) for r in rows2])
    shear = np.array([_safe_float(r["shear_stress_MPa"]) for r in rows2])
    vavg = np.array([_safe_float(r["avg_gouge_velocity"]) for r in rows2])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(t, shear)
    ax2.plot(t, vavg)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Shear stress (MPa)")
    ax2.set_ylabel("Average gouge velocity (cm/µs)")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def build_common_x_ref(rows_with_profiles: list[dict], *, dx_target: float | None = None) -> np.ndarray:
    """
    Build a common physical x-axis (cm).
    Strategy: use the earliest (typically largest) domain as reference.
    If later grids shrink, we will fill outside-domain with NaNs.
    """
    # Sort by time
    rows_sorted = sorted(rows_with_profiles, key=lambda r: _safe_float(r["acquisition_time_s"]))
    x0 = np.asarray(rows_sorted[0]["x_axis_cm"], dtype=float)

    if dx_target is None:
        # estimate dx from first axis
        if x0.size < 2:
            raise ValueError("x_axis_cm too short to infer dx.")
        dx_target = float(np.median(np.diff(x0)))

    # Build uniform reference axis spanning the first record
    x_ref = np.arange(x0.min(), x0.max() + 0.5 * dx_target, dx_target, dtype=float)
    return x_ref


def resample_profile_to_x_ref(x: np.ndarray, y: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """
    Interpolate y(x) onto x_ref.
    Outside the original domain: NaN (so plots show missing where grid shrank).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Ensure strictly increasing for interpolation
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    y_ref = np.full_like(x_ref, np.nan, dtype=float)
    inside = (x_ref >= x[0]) & (x_ref <= x[-1])
    if np.any(inside):
        y_ref[inside] = np.interp(x_ref[inside], x, y)
    return y_ref


def plot_profile_evolution_from_pickle(
    pkl_path: Path,
    out_vel_png: Path,
    out_damp_png: Path,
) -> None:
    rows = load_pickle_stream(pkl_path)

    # Keep only rows that contain profiles and x axis
    prof_rows = []
    for r in rows:
        if (r.get("velocity_profile", None) is not None) and (r.get("x_axis_cm", None) is not None):
            if "acquisition_time_s" in r:
                prof_rows.append(r)

    if not prof_rows:
        raise RuntimeError("No rows with velocity_profile + x_axis_cm found in pickle.")

    prof_rows.sort(key=lambda r: _safe_float(r["acquisition_time_s"]))

    # Common physical x reference (cm)
    x_ref = build_common_x_ref(prof_rows)

    # Build matrices
    t = np.array([_safe_float(r["acquisition_time_s"]) for r in prof_rows], dtype=float)

    vel_mat = []
    damp_mat = []
    have_damp = True

    for r in prof_rows:
        x = np.asarray(r["x_axis_cm"], dtype=float)
        v = np.asarray(r["velocity_profile"], dtype=float)
        vel_mat.append(resample_profile_to_x_ref(x, v, x_ref))

        d = r.get("damping_profile", None)
        if d is None:
            have_damp = False
        else:
            d = np.asarray(d, dtype=float)
            damp_mat.append(resample_profile_to_x_ref(x, d, x_ref))

    vel_mat = np.vstack(vel_mat)  # shape: (nt, nx_ref)

    # ---- Velocity heatmap ----
    fig, ax = plt.subplots()
    im = ax.imshow(
        vel_mat,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[x_ref[0], x_ref[-1], t[0], t[-1]],
    )
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Velocity profile evolution")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_vel_png, dpi=200)
    plt.close(fig)

    # ---- Damping heatmap (if present) ----
    if have_damp and damp_mat:
        damp_mat = np.vstack(damp_mat)

        fig, ax = plt.subplots()
        im = ax.imshow(
            damp_mat,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[x_ref[0], x_ref[-1], t[0], t[-1]],
        )
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("Time (s)")
        ax.set_title("Damping profile evolution")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(out_damp_png, dpi=200)
        plt.close(fig)

def build_time_paired_waveform_mech_table(
    observed_waveform_data: np.ndarray,
    waveform_metadata: Dict[str, Any],
    mechanical_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pair UW waveforms to mechanical samples using time in seconds.

    Assumptions:
      - waveform_metadata["time_ax_acquisition"] is in seconds
      - mechanical_dataframe["time_s"] is in seconds

    The UW axis is treated as relative to the UW file start; we shift it by the first
    mechanical time (start of synced slice) to get absolute seconds.
    """

    n_uw = len(observed_waveform_data)

    if "time_s" not in mechanical_dataframe.columns:
        raise ValueError("mechanical_dataframe must contain a 'time_s' column in seconds.")

    mech_df = (
        mechanical_dataframe.sort_values("time_s")
        .reset_index()
        .rename(columns={"index": "Index"})
        .reset_index(drop=True)
    )

    t_uw_rel_s = waveform_metadata.get("time_ax_acquisition", None)
    if t_uw_rel_s is None:
        # Fallback: distribute across mech window
        mech_time = mech_df["time_s"].to_numpy()
        if mech_time.size < 2:
            raise ValueError("Mechanical dataframe has insufficient time_s samples for pairing.")
        t_uw_abs_s = np.linspace(mech_time[0], mech_time[-1], n_uw)
    else:
        t_uw_rel_s = np.asarray(t_uw_rel_s, dtype=float)
        if t_uw_rel_s.size != n_uw:
            raise ValueError(
                f"time_ax_acquisition length ({t_uw_rel_s.size}) != number of waveforms ({n_uw})."
            )
        t0_mech_s = float(mech_df["time_s"].iloc[0])
        t_uw_abs_s = t_uw_rel_s + t0_mech_s

    uw_df = (
        pd.DataFrame({"uw_time_s": t_uw_abs_s, "uw_idx": np.arange(n_uw, dtype=int)})
        .sort_values("uw_time_s")
        .reset_index(drop=True)
    )

    # Tolerance: half the mechanical sampling interval (this is the right scale)
    mech_time = mech_df["time_s"].to_numpy()
    if mech_time.size > 2:
        mech_dt = float(np.median(np.diff(mech_time)))
        tol_s = 0.5 * mech_dt if np.isfinite(mech_dt) and mech_dt > 0 else None
    else:
        tol_s = None

    paired = pd.merge_asof(
        uw_df,
        mech_df,
        left_on="uw_time_s",
        right_on="time_s",
        direction="nearest",
        tolerance=tol_s,   # optional; can remove if you prefer never skipping
    )

    paired["match_error_s"] = np.abs(paired["uw_time_s"] - paired["time_s"])
    return paired

def load_pickle_log(path):
    out = []
    if not path.exists():
        return out
    with open(path, "rb") as f:
        while True:
            try:
                out.append(pickle.load(f))
            except EOFError:
                break
    return out

def min_assembly_velocity(assembly_dict: Dict[str, Any]):
    wt = assembly_dict["wave_type"]
    return min(
        assembly_dict["gouge_velocity_1"],
        assembly_dict["gouge_velocity_2"],
        assembly_dict["side1_params"]["velocity" + wt],
        assembly_dict["side1_params"]["pzt_velocity" + wt],
        assembly_dict["side1_params"]["pla_velocity" + wt],
    )

def max_assembly_velocity(assembly_dict: Dict[str, Any]):
    wt = assembly_dict["wave_type"]
    return max(
        assembly_dict["gouge_velocity_1"],
        assembly_dict["gouge_velocity_2"],
        assembly_dict["side1_params"]["velocity" + wt],
        assembly_dict["side1_params"]["pzt_velocity" + wt],
        assembly_dict["side1_params"]["pla_velocity" + wt],
    )
        
def update_assembly_dict_with_mech_data(
    assembly_dict: Dict[str, Any],
    mech_data: Tuple,
) -> None:
    """
    Update assembly_dict with per-waveform mechanical data entry
    """
    assembly_dict["gouge_thickness_1"]      = mech_data.rgt_lt_mm / 10.0     # thickness in mm -> convert to cm
    assembly_dict["gouge_thickness_2"]      = mech_data.rgt_lt_mm / 10.0     # for now, layers are assumed to have same thickness
    assembly_dict["normal_stress"]          = mech_data.normal_stress_MPa
    assembly_dict["shear_stress"]           = mech_data.shear_stress_MPa
    assembly_dict["ec_disp_mm"]             = mech_data.ec_disp_mm
    assembly_dict["acquisition_time"]       = mech_data.time_s
    assembly_dict["rec_n"]                  = int(mech_data.records_na)
    assembly_dict['idx_wave']               = mech_data.Index
    # define assembly sample_dimensions all together. It is usefull for handling forward modeling 
    assembly_dict["sample_dimensions"] = [
                                            side1_params["z_pzt2grove"],
                                            assembly_dict["gouge_thickness_1"] ,
                                            central_params["z"],
                                            assembly_dict["gouge_thickness_2"],
                                            side2_params["z_pzt2grove"],
                                        ]
        
def set_assembly_dict_guessed_with_gouge_velocity_and_damping(
    assembly_dict: Dict[str, Any],
    gouge_velocity_tuple: Union[Tuple[float],Tuple[np.ndarray]],
    gouge_damping_tuple: Union[Tuple[float],Tuple[np.ndarray]],

) -> None:
    assembly_dict_guessed = assembly_dict.copy()
    assembly_dict_guessed["gouge_velocity_1"], assembly_dict_guessed["gouge_velocity_2"] = gouge_velocity_tuple
    assembly_dict_guessed["gouge_damping_1"], assembly_dict_guessed["gouge_damping_2"] = gouge_damping_tuple
    return assembly_dict_guessed

def mechdata_slice4uw_processing(mech_data, sync_peaks, chosen_uw_file, num_waveform2process=None):      
    from math import ceil  
    start_sync      = sync_peaks[2 * chosen_uw_file]
    end_sync        = sync_peaks[2 * chosen_uw_file + 1]
    mech_data_slice = mech_data.iloc[start_sync : end_sync].copy()
    if num_waveform2process:
        downsampling    = max(1, ceil(len(mech_data_slice)/ num_waveform2process)) 
        return mech_data_slice.iloc[:: downsampling].reset_index(drop=True)
    else:
        return mech_data_slice

def predict_event_windows(assembly_dict, v_gouge_list, wave_type="_p"):
    vg = np.asarray(v_gouge_list, dtype=float)
    vg = vg[np.isfinite(vg) & (vg > 0)]
    if vg.size == 0:
        raise ValueError("v_gouge_list must contain positive finite velocities (cm/us).")

    vg_min, vg_max = float(vg.min()), float(vg.max())

    h1 = float(assembly_dict["gouge_thickness_1"])
    h2 = float(assembly_dict["gouge_thickness_2"])
    hmin = min(h1, h2)

    # direct arrival bounds (using corrected travel time)
    t_direct_min = compute_dds_travel_time(assembly_dict, vg_max, vg_max, wave_type)  # fastest => earliest
    t_direct_max = compute_dds_travel_time(assembly_dict, vg_min, vg_min, wave_type)  # slowest => latest

    # first gouge internal multiple: direct + 2*hmin/vg
    # earliest multiple occurs with fastest vg; latest with slowest vg
    t_rt_min = t_direct_min + (h1+h2) / vg_max
    t_rt_max = t_direct_max + (h1+h2) / vg_min

    # side-block round trip scale (additive relative to direct):
    # round trip to first groove interface on one side ~ 2*z_pzt2groove/v_side
    side1 = assembly_dict["side1_params"]
    z = float(side1["z_pzt2grove"])
    v_side = float(side1["velocity" + wave_type])
    t_side_rt = t_direct_min + 2.0 * z / v_side  # weak vg dependence; use earliest direct as reference

    windows = {
        "direct": (t_direct_min, t_direct_max),
        "gouge_RT1": (t_rt_min, t_rt_max),
        "side_block_RT": (t_side_rt, t_side_rt),
    }
    return windows


def plot_with_windows(time_us, waveform, windows, title="Waveform with predicted event windows"):
    time_us = np.asarray(time_us, dtype=float)
    waveform = np.asarray(waveform, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_us, waveform, lw=1)

    for name, (tmin, tmax) in windows.items():
        tmin, tmax = float(tmin), float(tmax)
        if np.isclose(tmin, tmax):
            ax.axvline(tmin, linestyle="--", linewidth=1, label=f"{name}: {tmin:.3f} µs")
        else:
            lo, hi = (min(tmin, tmax), max(tmin, tmax))
            ax.axvspan(lo, hi, alpha=0.18, label=f"{name}: [{lo:.3f}, {hi:.3f}] µs")
            ax.axvline(lo, linestyle="--", linewidth=1)
            ax.axvline(hi, linestyle="--", linewidth=1)

    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Amplitude [a.u.]")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def compute_dds_travel_time(assembly_dict, v_gouge_1=None, v_gouge_2=None, wave_type=None):
    side1 = assembly_dict["side1_params"]
    side2 = assembly_dict["side2_params"]
    cent  = assembly_dict["central_params"]
    wave_type = wave_type or assembly_dict["wave_type"]

    v_s1 = side1["velocity" + wave_type]
    v_s2 = side2["velocity" + wave_type]
    v_sc = cent["velocity" + wave_type]

    # steel-only distances exclude groove thickness (as you designed)
    t_steel = (
        (side1["z_pzt2grove"] - side1["h_grooves"]) / v_s1 +
        (side2["z_pzt2grove"] - side2["h_grooves"]) / v_s2 +
        (cent["z"] - 2 * cent["h_grooves"]) / v_sc
    )

    if v_gouge_1 is None or v_gouge_2 is None:
        return t_steel

    h1 = assembly_dict["gouge_thickness_1"]
    h2 = assembly_dict["gouge_thickness_2"]

    # v^2 ramp implies dt = 2h/(v0+v1)
    def groove_dt(h, v0, v1):
        return 2.0 * h / (v0 + v1)

    t_gouge = h1 / v_gouge_1 + h2 / v_gouge_2

    t_grooves = (
        groove_dt(side1["h_grooves"], v_s1, v_gouge_1) +
        groove_dt(cent ["h_grooves"], v_sc, v_gouge_1) +
        groove_dt(cent ["h_grooves"], v_sc, v_gouge_2) +
        groove_dt(side2["h_grooves"], v_s2, v_gouge_2)
    )

    # PZT: if source and receiver are at the centers of the PZTs, total is (w/2 + w/2)/vp = w/vp
    t_pzt = side1["pzt_layer_width"] / side1["pzt_velocity" + wave_type]

    return t_steel + t_gouge + t_grooves + t_pzt

# Function Definitions
def process_uw_file(
    infile_path: Path,
    uw_data_handler: UltrasonicDataHandler,
    mechanical_dataframe: MechanicalDataHandler,
    stf_handler: UltrasonicDataHandler,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
) -> None:
    """
    Process a single UW data file (ultrasonic waveforms).
    Saves:
      - global-search arrays + best misfit (only when global search is run)
      - local inversion averages + full inverted profiles (velocity & damping)
      - mechanical metadata for later analysis/plotting
    """

    outdir_path_l2norm = params["outdir_path_l2norm"]
    start_time = tm.time()
    outfile_name = infile_path.name.split(".")[0]
    outfile_path = outdir_path_l2norm / outfile_name

    last_good_result = {"simulation_handler": None}

    observed_waveform_data = uw_data_handler.waveform_data
    waveform_metadata = uw_data_handler.metadata

    paired = build_time_paired_waveform_mech_table(
        observed_waveform_data=observed_waveform_data,
        waveform_metadata=waveform_metadata,
        mechanical_dataframe=mechanical_dataframe,
    )

    results_pkl = outfile_path.with_suffix(".pkl")
    results_summary = []      # list of dicts for pandas
    results_by_rec_n = {}     # rec_n -> dict

    with open(results_pkl, "ab") as f_results:

        for row in paired.itertuples(index=False):

            if not np.isfinite(getattr(row, "Index", np.nan)):
                print(
                    f"WARNING: No mechanical match within tolerance for UW idx={row.uw_idx} "
                    f"at t_us={row.uw_time_s}. Skipping."
                )
                continue

            observed_waveform = observed_waveform_data[int(row.uw_idx)]
            mech_data = row

            update_assembly_dict_with_mech_data(assembly_dict=assembly_dict, mech_data=mech_data)
            rec_n = int(assembly_dict["rec_n"])
            outfile_name_wave = outfile_name + f"_rec_n_{rec_n}"

            # -----------------------------
            # 1) If no prior model: GLOBAL SEARCH (once) + store global outputs
            # -----------------------------
            if last_good_result.get("simulation_handler", None) is None:

                result_global = global_search_homogeneus(
                    observed_waveform=observed_waveform,
                    waveform_metadata=waveform_metadata,
                    outfile_name=outfile_name_wave,
                    stf_handler=stf_handler,
                    params=params,
                    assembly_dict=assembly_dict,
                )

                if not result_global.get("simulation_handler", None):
                    continue

                # Update state
                last_good_result = result_global

                # Save EXACT global-search outputs + best misfit from the final forward run
                sim_g = result_global["simulation_handler"]
                global_row = {
                    # keys for recall
                    "rec_n": rec_n,
                    "idx_wave": int(assembly_dict["idx_wave"]),
                    "acquisition_time_s": float(assembly_dict["acquisition_time"]),
                    # mechanical state
                    "normal_stress_MPa": float(assembly_dict["normal_stress"]),
                    "shear_stress_MPa": float(assembly_dict["shear_stress"]),
                    "gouge_thickness_1_cm": float(assembly_dict["gouge_thickness_1"]),
                    "gouge_thickness_2_cm": float(assembly_dict["gouge_thickness_2"]),

                    # GLOBAL SEARCH arrays (as requested)
                    "x_axis_cm": sim_g.grid_handler.spatial_axis.copy(),
                    "gouge_velocity_list": result_global["gouge_velocity_list"],
                    "best_gouge_velocity": float(result_global["best_gouge_velocity"]),
                    "gouge_damping_list": result_global["gouge_damping_list"],
                    "best_gouge_damping": float(result_global["best_gouge_damping"]),
                    "L2norm_waveform_list": result_global["L2norm_waveform_list"],

                    # Best misfit from the FINAL forward run of global search
                    "global_best_misfit": float(sim_g.misfit),

                    # Profiles at the global-search best model (useful baseline)
                    "velocity_profile": sim_g.velocity_model_handler.velocity_array.copy(),
                    "damping_profile": sim_g.velocity_model_handler.damping_array.copy() if hasattr(sim_g.velocity_model_handler, "damping_array") else None,
                }

                results_summary.append(global_row)
                results_by_rec_n[rec_n] = global_row
                pickle.dump(global_row, f_results, protocol=pickle.HIGHEST_PROTOCOL)
                f_results.flush()

            # -----------------------------
            # 2) LOCAL refine on subsequent waveforms
            # -----------------------------
            last_good_result["simulation_handler"].observed_waveform = observed_waveform

            # IMPORTANT FIX: pass the prior simulation handler, not "result[...]"
            result_local = local_refine_waveform(
                outfile_name=outfile_name_wave,
                params=params,
                assembly_dict=assembly_dict,
                simulation=last_good_result["simulation_handler"],
            )

            if not result_local.get("simulation_handler", None):
                continue

            last_good_result = result_local
            sim = last_good_result["simulation_handler"]

            local_row = {
                # keys for recall
                "rec_n": rec_n,
                "idx_wave": int(assembly_dict["idx_wave"]),
                "acquisition_time_s": float(assembly_dict["acquisition_time"]),

                # mechanical state
                "normal_stress_MPa": float(assembly_dict["normal_stress"]),
                "shear_stress_MPa": float(assembly_dict["shear_stress"]),
                "ec_disp_mm": float(assembly_dict["ec_disp_mm"]),
                "gouge_thickness_1_cm": float(assembly_dict["gouge_thickness_1"]),
                "gouge_thickness_2_cm": float(assembly_dict["gouge_thickness_2"]),

                # LOCAL inversion outputs
                "x_axis_cm": sim.grid_handler.spatial_axis.copy(),
                "misfit": float(sim.misfit),
                "avg_gouge_velocity": float(getattr(sim, "average_gouge_velocity", np.nan)),
                "avg_gouge_damping": float(getattr(sim, "average_gouge_damping", np.nan)),

                # FULL inverted profiles (requested)
                "velocity_profile": sim.velocity_model_handler.velocity_array.copy(),
                "damping_profile": sim.velocity_model_handler.damping_array.copy() if hasattr(sim.velocity_model_handler, "damping_array") else None,
            }

            results_summary.append(local_row)
            results_by_rec_n[rec_n] = local_row
            pickle.dump(local_row, f_results, protocol=pickle.HIGHEST_PROTOCOL)
            f_results.flush()

    pkl_path = results_pkl

    out1 = pkl_path.with_name(pkl_path.stem + "_shear_and_vavg_vs_time.png")
    plot_shear_and_avg_velocity_from_pickle(pkl_path, out1)
    print(f"Saved: {out1}")

    out_vel = pkl_path.with_name(pkl_path.stem + "_velocity_profiles_vs_time.png")
    out_dmp = pkl_path.with_name(pkl_path.stem + "_damping_profiles_vs_time.png")
    plot_profile_evolution_from_pickle(pkl_path, out_vel, out_dmp)
    print(f"Saved: {out_vel}")
    print(f"Saved: {out_dmp}")
    print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---")


def global_search_homogeneus(
    observed_waveform: np.ndarray,
    waveform_metadata: Dict[str,Any],
    outfile_name: str,
    stf_handler : UltrasonicDataHandler,
    params: Dict[str, Any],
    assembly_dict: Dict[str,Any],
) -> Dict[str, Union[float, np.ndarray, None]]:

    # Unpack needed dictionaries entries
    observed_time          = waveform_metadata['time_ax_waveform']         # time axes for the samples of a single waveform acquisition
    frequency_cutoff       = params['frequency_cutoff']
    minimum_SNR            = params['minimum_SNR']
    plot_save_interval     = params['plot_save_interval']
    movie_save_interval    = params['movie_save_interval']
    l2norm_plot   = params['l2norm_plot']
    outdir_path_image      = params['outdir_path_image']

    rec_n = assembly_dict["rec_n"]
              
    # Evaluate SNR    
    waveform_snr= evaluate_snr(observed_time,observed_waveform, assembly_dict["steel_only_time_p_wave"])
    if  waveform_snr < minimum_SNR:
        normal_stress = assembly_dict["normal_stress"]
        shear_stress = assembly_dict["shear_stress"]
        print(f"SNR {waveform_snr:.2f} < {minimum_SNR}. Skipping rec n: {rec_n}, Normal Stress: {normal_stress:.2f}, Shear Stress: {shear_stress:.2f}.")
        return {"simulation_handler": None}

    # Compute windows
    windows = predict_event_windows(assembly_dict, params["velocity_initial_list"], wave_type="_p")
    # plot_with_windows(observed_time, observed_waveform, windows, title=f"rec_n={assembly_dict['rec_n']} predicted events")
    # print(windows)

    # Generate velocity array
    gouge_velocity_list = params["velocity_initial_list"]
    gouge_damping_list  = params["damping_initial_list"]

    # Misfit_interval is choosen according to STF len
    stf_time = stf_handler.metadata["time_ax_waveform"]
    stf_duration = stf_time[-1]-stf_time[0]  

    # Determine if we save plots and/or movies
    save_plot  = (assembly_dict['idx_wave'] % plot_save_interval == 0) 
    save_movie = (assembly_dict['idx_wave'] % movie_save_interval == 0) 

    # Construct output paths
    # if save_plot:
    plot_output_name = outfile_name
    plot_output_path = Path(outdir_path_image / plot_output_name)
    plot_output_path.mkdir(parents=True, exist_ok=True)
    plot_output_global = plot_output_path / "global_search_homoegeneus"
    plot_output_global.mkdir(parents=True, exist_ok=True)
    # else:
    #     plot_output_path = None

    if save_movie:
        movie_output_name = f"{outfile_name}.mp4"
        movie_output_path = outdir_path_image / movie_output_name
    else:
        movie_output_path = None

    # Prepare arguments for multiprocessing
    num_processes = cpu_count()-1
    def _build_args(gouge_velocity: float, gouge_damping: float):

        guessed_arrival_time = compute_dds_travel_time(
            assembly_dict = assembly_dict,
            v_gouge_1     = gouge_velocity,
            v_gouge_2     = gouge_velocity
        )
        
        misfit_interval = np.where(
            (observed_time > guessed_arrival_time) & 
            (observed_time < guessed_arrival_time + stf_duration)
        )[0]
        
        assembly_dict_guessed = set_assembly_dict_guessed_with_gouge_velocity_and_damping(
            assembly_dict, (gouge_velocity, gouge_velocity), (gouge_damping, gouge_damping)
        )
        
        # Return all arguments in a single tuple
        return (
            observed_waveform,
            observed_time,
            stf_handler,
            misfit_interval,
            params,
            assembly_dict_guessed,
            plot_output_global,
        )

    # args_list = [_build_args(gv) for gv in gouge_velocity_list]
    args_list = [_build_args(v, d) for v, d in product(gouge_velocity_list, gouge_damping_list)]
    # Multiprocessing over velocities and dampipng
    with Pool(processes=num_processes) as pool:
        results = pool.map(global_search_run, args_list)

    # Sort results by velocity
    results_sorted = sorted(results, key=lambda x: x[0])
    gouge_velocity_list  = np.array([res[0] for res in results_sorted])
    L2norm_waveform_list = np.array([res[1] for res in results_sorted])
    misfit_interval_list = np.array([res[2] for res in results_sorted])
    gouge_damping_list   = np.array([res[3] for res in results_sorted])

    # Possibly plot L2 norm vs. velocity
    if l2norm_plot:
        unique_vels = np.unique(gouge_velocity_list)
        unique_damps = np.unique(gouge_damping_list)
        misfit_grid = np.full((len(unique_vels), len(unique_damps)), np.nan)

        vel_to_row = {vel: idx for idx, vel in enumerate(unique_vels)}
        damp_to_col = {damp: idx for idx, damp in enumerate(unique_damps)}

        for (v, misfit, _, a) in results_sorted:
            row = vel_to_row[v]
            col = damp_to_col[a]
            misfit_grid[row, col] = misfit

        l2norm_plot_name = f"{outfile_name}_L2norm"
        plot_output_dir = plot_output_path / "global_search_homoegeneus"
        l2norm_plot_path = plot_output_dir / l2norm_plot_name

        plotter = Plotter()
        plotter.misfit_map(
            misfit_grid=misfit_grid,
            unique_damps=unique_damps,
            unique_vels=unique_vels,
            outfile_path=l2norm_plot_path
        )

    #####################################################
    # RE RUN WITH THE BEST MISFIT PARAMETERS
    ###################################################
    min_idx = np.argmin(L2norm_waveform_list)
    best_gouge_velocity = gouge_velocity_list[min_idx]
    best_gouge_damping = gouge_damping_list[min_idx]

    misfit_interval = misfit_interval_list[min_idx]
    assembly_dict["gouge_velocity_1"] = best_gouge_velocity
    assembly_dict["gouge_velocity_2"] = best_gouge_velocity
    assembly_dict["gouge_damping_1"]  = best_gouge_damping
    assembly_dict["gouge_damping_2"]  = best_gouge_damping
    gouge_thickness = assembly_dict["gouge_thickness_1"] 
    print(f"rec n: {rec_n}: min misfit at velocity = {best_gouge_velocity:.4f} cm/μs, damping: {best_gouge_damping:.5f} arrival time: {observed_time[misfit_interval][0]:.2f}, thickness: {gouge_thickness:.2f}")

    # Check boundary
    if min_idx in (0, len(L2norm_waveform_list) - 1):
        print(f"Minimum misfit is on the boundary!")
        # params["velocity_range"] = 2*params["velocity_range"]
    
    minimum_velocity = params["min_velocity2simulate"] if params["min_velocity2simulate"] else min_assembly_velocity(assembly_dict)
    maximum_velocity = params["max_velocity2simulate"] if params["max_velocity2simulate"] else max_assembly_velocity(assembly_dict)
    maximum_damping  = max(assembly_dict["gouge_damping_1"],assembly_dict["gouge_damping_2"])

    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing               = params["absorbing"],
        geometry_type           ="dds",
        observed_time           = observed_time,
        observed_waveform       = observed_waveform,
        frequency_cutoff        = frequency_cutoff,
        assembly_dict           = assembly_dict,
        stf_handler             = stf_handler,
        misfit_interval         = misfit_interval,
        minimum_velocity        = minimum_velocity,
        maximum_velocity        = maximum_velocity,  
        maximum_damping         = maximum_damping,
        normalize_waveform      = True,
        enable_plotting         = save_plot,
        make_movie              = False,
        plot_output_path        = plot_output_path,
        movie_output_path       = movie_output_path
    )

    return {
        'gouge_velocity_list' : gouge_velocity_list,
        'best_gouge_velocity' : best_gouge_velocity,
        'gouge_damping_list'  : gouge_damping_list,
        'best_gouge_damping'  : best_gouge_damping,
        'L2norm_waveform_list': L2norm_waveform_list,
        "simulation_handler": simulation
        }

def global_search_run(args):
    """
    Function to process a single velocity value in multiprocessing.
    """
    (
        observed_waveform,
        observed_time,
        stf_handler,
        misfit_interval,
        params,
        assembly_dict_guessed,
        plot_output_global
    ) = args

    # Unpack parameters
    frequency_cutoff = params['frequency_cutoff']
    minimum_velocity = params["min_velocity2simulate"] if params["min_velocity2simulate"] else min_assembly_velocity(assembly_dict_guessed)
    maximum_velocity = params["max_velocity2simulate"] if params["max_velocity2simulate"] else max_assembly_velocity(assembly_dict_guessed)
    maximum_damping  = max(assembly_dict_guessed["gouge_damping_1"],assembly_dict_guessed["gouge_damping_2"])

    # Call DDS_UW_simulation with gouge_velocity_tuple
    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing               = params["absorbing"],
        geometry_type           ="dds",
        observed_time           = observed_time,
        observed_waveform       = observed_waveform,
        stf_handler             = stf_handler,
        frequency_cutoff        = frequency_cutoff,
        assembly_dict           = assembly_dict_guessed,
        misfit_interval         = misfit_interval,
        minimum_velocity        = minimum_velocity,
        maximum_velocity        = maximum_velocity, 
        maximum_damping         = maximum_damping, 
        normalize_waveform      = True,
        enable_plotting         = True,
        plot_output_path        = plot_output_global
    )
    
    L2norm_new = simulation.misfit
    gouge_velocity = assembly_dict_guessed["gouge_velocity_1"]
    gouge_damping  = assembly_dict_guessed["gouge_damping_1"]
    # theo_arrival_time = observed_time[misfit_interval][0]
    # print(f"\tVelocity: {1e4*gouge_velocity:.0f}, damping:{gouge_damping:.5f}, Theo arrival time: {theo_arrival_time:.2f} => Misfit: {L2norm_new:.1f}")

    return gouge_velocity, L2norm_new, misfit_interval, gouge_damping

def local_refine_waveform(
    outfile_name: str,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    simulation: UltrasonicDataHandler,
) -> Dict[str, Union[float, np.ndarray, None]]:
    """
    Local-only continuation step: forward_simulation + inject prior model + run_local_inversion.
    No global grid search.

    Does not change I/O of forward_simulation or run_local_inversion; it only sets initial model arrays.
    """

    plot_save_interval = params["plot_save_interval"]
    movie_save_interval = params["movie_save_interval"]
    outdir_path_image = params["outdir_path_image"]
    save_plot  = (assembly_dict["idx_wave"] % plot_save_interval == 0)
    save_movie = (assembly_dict["idx_wave"] % movie_save_interval == 0)

    if save_plot: 
        plot_output_path = Path(outdir_path_image / f"{outfile_name}")
        plot_output_path.mkdir(parents=True,exist_ok=True) 
    else: plot_output_path = None
    if save_movie:
        movie_output_path = outdir_path_image / f"{outfile_name}.mp4"
    else: movie_output_path = None

    try:
        guessed_arrival_time = compute_dds_travel_time(
            assembly_dict = assembly_dict,
            v_gouge_1     = simulation.average_gouge_velocity,
            v_gouge_2     = simulation.average_gouge_velocity,
        )
    except:
        guessed_arrival_time = compute_dds_travel_time(
            assembly_dict = assembly_dict,
            v_gouge_1     = assembly_dict["gouge_velocity_1"],
            v_gouge_2     = assembly_dict["gouge_velocity_1"]
    )
    # Misfit_interval is choosen according to STF len
    stf_time = stf_handler.metadata["time_ax_waveform"]
    stf_duration = stf_time[-1]-stf_time[0]  

    misfit_interval = np.where(
        (simulation.observed_time > guessed_arrival_time) & 
        (simulation.observed_time < guessed_arrival_time + stf_duration)
    )[0]

    simulation.misfit_interval = misfit_interval
    simulation.assembly_dict = assembly_dict

    # Local inversion hyperparameters
    dw_max_start = np.amax(simulation.source_handler.time_function)
    dw_threshold = 0.01 * dw_max_start
    ds_max_start = np.amax(simulation.source_handler.spatial_function)
    ds_threshold = 0.01 * ds_max_start

    dc_max_start = 0.1 * assembly_dict["gouge_velocity_1"]
    dc_threshold = 0.01 * dc_max_start

    da_max_start = 0.1 * assembly_dict["gouge_damping_1"]
    da_threshold = 0.01 * da_max_start

    simulation.run_local_inversion(
        n_iterations         = params["n_iterations"],
        dc_max_start         = dc_max_start,
        dc_threshold         = dc_threshold,
        da_max_start         = da_max_start,
        da_threshold         = da_threshold,
        dw_max_start         = dw_max_start,
        dw_threshold         = dw_threshold,
        stf_duration         = stf_duration,
        ds_max_start         = ds_max_start,
        ds_threshold         = ds_threshold,
        reduce_factor        = params["reduce_factor"],
        normalize_waveform   = True,
        enable_plotting      = save_plot,
        make_movie           = False,
        plot_output_path     = plot_output_path,
        movie_output_path    = movie_output_path
    )

    # Return in the same schema as global_search_homogeneus
    return {
        "gouge_velocity_list": None,
        "best_gouge_velocity": assembly_dict["gouge_velocity_1"],
        "gouge_damping_list": None,
        "best_gouge_damping": assembly_dict["gouge_damping_1"],
        "L2norm_waveform_list": None,
        "simulation_handler": simulation
    }

###############################################################################################################
# Main Execution
if __name__ == "__main__":

    import os, matplotlib
    if os.getenv("DISPLAY", "") == "":          # inside screen / batch job
        matplotlib.use("Agg")                  # non-GUI backend

    # Initialize directory manager
    dir_manager = DirectoryManager()
    
    # Basic experiment info
    machine_name    = "Brava_2"
    experiment_name = "s0273suw03anhdol_5030"
    wave_type       = "_p"    # that "_" is ugly, but needed
    data_type_uw    = "uw_data/data_tsv_files" # + wave_type
    data_type_mech  = "mechanical_data"
    mech_file_name  = f"{experiment_name}_data_rp"
    outfolder_name  = "test_christmass"
    # Create output directories
    outdir_path_l2norm = dir_manager.make_data_analysis_folders(
        machine_name    = machine_name,
        experiment_name = experiment_name,
        data_types      = [outfolder_name]
    )

    outdir_path_image = dir_manager.make_data_analysis_folders(
        machine_name    = machine_name,
        experiment_name = experiment_name,
        data_types      = [outfolder_name + "_images_and_movie"]
    )

    # Basic simulation parameters 
    params = {
        "absorbing"                 : False,        # set absorbing bounday, merdoso silicone maledetto
        "num_waveform2process"      : 10,         # int, equespatially waveforms to sample for processing. Of "None", process all
        "maxtime2simulate"          : 20,           # [mus]
        "frequency_cutoff"          : 4,            # [MHz] low pass onserved data and simulate up to this frequency
        "minimum_SNR"               : 20,            # skip computation until time interval where signal should be is above SNR times surely-only-noise part 
        "velocity_initial_list"     : np.linspace(0.3000, 0.6000, 30),  # [cm/mus] first guess of best velocity. There is a visual tool for it, if needed
        "min_velocity2simulate"     : 0.25,         # [cm/mus] if not passed, computed by assembly and gouge velocity range
        "max_velocity2simulate"     : 0.61,         # [cm/mus]
        "damping_initial_list"      : np.linspace(0.0001,0.001, 10),
        "plot_save_interval"        : 1000,
        "movie_save_interval"       : 10000,
        "l2norm_plot"               : True,
        "outdir_path_l2norm"        : outdir_path_l2norm[0],
        "outdir_path_image"         : outdir_path_image[0],
        "n_iterations"              : 10,
        "reduce_factor"             : 10/9
    }

    # Load the Source Time Function (HDF5)
    stf_h5_path = (# FAKE FIX
        dir_manager.paths.analysis_root("on_bench", "STF_ss10_05")
        # / f"source_time_functions{wave_type}"
#################################################
# # FAKE FIX !!!!!!!!!!!!!!!!!!!!!!! WRONG SOURCE TIME FUNCTION
#########################################################
        / f"source_time_functions_s" 
        / "stf.h5"
    )

    # read the STF saved under /stf/width250_volt70/cycle_000000/source_waveform
    stf_handler = UltrasonicDataHandler.read_stf_h5(
        stf_h5_path,
        name="width250_volt70",   # "width250_volt70_local_inversion" when available
        cycle_index=0,
    )

    stf_handler.waveform_data = -(stf_handler.waveform_data).squeeze(0)   # shape (L,)
    stf_handler.lowpass(cutoff_mhz=params['frequency_cutoff'])
    # stf_handler.waveform_data[:19] = stf_handler.waveform_data[6:25]*np.hanning(len(stf_handler.waveform_data[6:25]))
    # stf_handler.waveform_data[40:] = 0
    stf_handler.waveform_data[0] = 1
    stf_handler.waveform_data[1:] = 0
    stf_handler.waveform_data = stf_handler.waveform_data[:100] #*np.hanning(len(stf_handler.waveform_data[:100]))
    stf_handler.metadata['time_ax_waveform'] =     stf_handler.metadata['time_ax_waveform'][:100]
    frequencies, amplitude_spectrum, phase_spectrum = stf_handler.compute_amplitude_phase_spectrum()
    # plt.plot(frequencies, amplitude_spectrum)
    # plt.plot(stf_handler.metadata['time_ax_waveform'], stf_handler.waveform_data)
    # plt.show()

    # Load Mechanical Data
    mech_folder = Path(dir_manager.base_dir) / f"experiments_{machine_name}" / experiment_name / data_type_mech
    mech_data, sync_data, sync_peaks = MechanicalDataHandler.find_and_load(mech_folder, mech_file_name)
    sync_peaks = np.sort(np.append(sync_peaks, sync_peaks[2]+13000))
    # print(mech_data)
    # print(mech_data.time_s[sync_peaks[2]+13000]-mech_data.time_s[sync_peaks[2]])
    # print(sync_peaks)
    # f, ax = plt.subplots()
    # ax.plot(mech_data.shear_stress_MPa, c='r')
    # ax.plot(mech_data.normal_stress_MPa, c='g')
    # ax1 = ax.twinx()
    # ax1.vlines(sync_peaks[2]+13000,ymin=0,ymax=5)
    # ax1.plot(mech_data.sync, c='k')
    # plt.show()
    # sys.exit()

    ########################
    # sync_peaks miss one pick: the second file, that is a runin, was left to end without clicking again 
    # on the synchronization: must find a duct taper for this problem

    # Build a dictionary containing all the relevant assembly parameters
    blocks_json = Path(dir_manager.base_dir) / "metadata" / "blocks_metadata.json"
    side1_params, side2_params, central_params = BlockMetadataHandler.load_blocks_metadata(
    blocks_json, ("guglielmi_side1","guglielmi_side2","central_block_titanium")
    )

    assembly_dict = {
        "side1_params"        : side1_params,
        "side2_params"        : side2_params,
        "central_params"      : central_params,
        "wave_type"           : wave_type,
        "transmitter_position": side1_params["z_pzt2grove"],
        "receiver_position"   : side2_params["z_pzt2grove"],
    }
    assembly_dict["velocity" + wave_type] = side1_params["velocity" + wave_type]
    # Duct-taper to check mininimum possible arrival time. TO BE REMOVED!!!
    assembly_dict["steel_only_time_p_wave"] = compute_dds_travel_time(assembly_dict=assembly_dict,wave_type="_p")
    
    # Make UW path list
    infile_path_list_uw = sorted( dir_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw))
 
    # Process each UW file
    for chosen_uw_file, infile_path in enumerate(infile_path_list_uw):
        infile_name = infile_path.name.split(".")[0]

        print(infile_name)
        # if infile_name != "001_compaction":
        #     continue

        mech_data_slice = mechdata_slice4uw_processing(mech_data,
                                                       sync_peaks, 
                                                       chosen_uw_file, 
                                                       num_waveform2process=params["num_waveform2process"])

        # Load & process ultrasonic data and metadata from TSV, with preprocessing
        uw_data_handler = (
            UltrasonicDataHandler.read_tsv(infile_path)
                .remove_mean()
                .downsample_waveforms(params["num_waveform2process"])     # keep N waveforms (evenly sampled)
                .truncate_time(params["maxtime2simulate"])                # limit to maxtime [µs]
                .zero_before(assembly_dict["steel_only_time_p_wave"] / 5) # zero early window [µs]
                .lowpass(params["frequency_cutoff"])                    # optional, cutoff in MHz
        )

        # import matplotlib.pyplot as plt
        # import matplotlib.cm as cm
        # import matplotlib.colors as colors
        # import numpy as np

        # t_acq = uw_data_handler.metadata.get("time_ax_acquisition", None)
        # print("mech time start/end =", mech_data_slice["time_s"].iloc[0], mech_data_slice["time_s"].iloc[-1])
        # print(f"num waves: {uw_data_handler.metadata.get("number_of_waveforms")}")

        # time = uw_data_handler.metadata['time_ax_waveform']
        # data = uw_data_handler.waveform_data[400:800] # (n_waveforms, n_samples)
        # print(mech_data_slice.rgt_lt_mm[800]-mech_data_slice.rgt_lt_mm[400])
        # n_waveforms = data.shape[0]
        # norm = colors.Normalize(vmin=0, vmax=n_waveforms - 1)
        # cmap = cm.winter

        # fig, ax = plt.subplots(figsize=(8, 5))

        # for i in range(n_waveforms):
        #     ax.plot(
        #         time,
        #         data[i, :],
        #         color=cmap(norm(i)),
        #         alpha=0.2,
        #         linewidth=0.7
        #     )

        # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        # sm.set_array([])

        # cbar = fig.colorbar(sm, ax=ax, label="Waveform index (time order)")
        # cbar.set_ticks([0, n_waveforms//2, n_waveforms-1])
        # cbar.set_ticklabels(["0", f"{n_waveforms//2}", f"{n_waveforms-1}"])

        # # fig.colorbar(sm, ax=ax, label="Waveform index (time order)")

        # ax.set_xlabel("Time")
        # ax.set_ylabel("Amplitude")
        # ax.set_title("Waveform Evolution Over Time")

        # plt.show()
        # continue

        process_uw_file(
            infile_path             = infile_path,
            uw_data_handler         = uw_data_handler,
            mechanical_dataframe    = mech_data_slice,
            stf_handler             = stf_handler,
            params                  = params,
            assembly_dict           = assembly_dict
        )