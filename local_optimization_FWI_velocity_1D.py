# lab_uw/local_optimization_velocity_homogeneus.py

import os

# 1) Make Matplotlib non-interactive (no X server use)
os.environ["MPLBACKEND"] = "Agg"

# 2) Prevent Numba from using TBB (which is the one complaining)
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
# optional: cap threads to avoid oversubscription
# os.environ["NUMBA_NUM_THREADS"] = "4"

import sys
from pathlib import Path
from copy import deepcopy
from collections import Counter, deque
import pickle
import queue
import time as tm
import traceback
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, get_context
from typing import Dict, Any, Tuple, Union
from itertools import product
import matplotlib.pyplot as plt

from lab_uw.data_io.data_io import UltrasonicDataHandler, BlockMetadataHandler, MechanicalDataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.simulation_setup import *
from lab_uw.forward_modeling import *
from lab_uw.plotting import Plotter
from lab_uw.forward_modeling import UltrasonicModeler
from lab_uw.inversion_windows import direct_misfit_interval, sample_interval_mask
from lab_uw.impedance_search import material_candidates, centered_refined_axis
from lab_uw.gouge_thickness import (
    build_constant_total_thickness_candidates,
    select_thickness_minimum,
)
from lab_uw.joint_kinematic import build_joint_constant_time_candidates
from lab_uw.velocity_contrast import build_constant_time_contrast_candidates
from lab_uw.utils import *

def resolve_worker_count(
    params: Dict[str, Any],
    parameter_name: str,
    number_of_tasks: int,
) -> int:
    """Return a safe number of worker processes for independent tasks."""

    if number_of_tasks <= 0:
        return 0

    cpu_reserve = max(0, int(params.get("parallel_cpu_reserve", 1)))
    available_workers = max(1, cpu_count() - cpu_reserve)
    requested_workers = params.get(parameter_name)
    if requested_workers is not None:
        available_workers = min(
            available_workers,
            max(1, int(requested_workers)),
        )
    return min(available_workers, number_of_tasks)


def extract_gouge_profiles(sim) -> Dict[str, Any]:
    """
    Extract gouge-only velocity & damping profiles from the simulation handler.
    Saves a per-layer coordinate 'eta' in [0,1] (dimensionless) and thickness (cm).
    """
    idx = sim.velocity_model_handler.idx_dict

    def _layer_payload(layer_key: str, thickness_cm_key: str) -> Dict[str, Any]:
        gidx = np.asarray(idx[layer_key], dtype=int)

        v = sim.velocity_model_handler.velocity_array[gidx].copy()

        d = None
        if hasattr(sim.velocity_model_handler, "damping_array") and sim.velocity_model_handler.damping_array is not None:
            d = sim.velocity_model_handler.damping_array[gidx].copy()

        # normalized coordinate along the layer (0..1)
        eta = np.linspace(0.0, 1.0, gidx.size, dtype=float)

        # physical thickness (cm) from assembly_dict stored in sim (or fallback to NaN)
        # In your code, assembly_dict is inside the modeler instance as self.assembly_dict
        thickness_cm = float(sim.assembly_dict.get(thickness_cm_key, np.nan))

        return {
            "eta": eta,
            "velocity": v,
            "damping": d,
            "npts": int(gidx.size),
            "thickness_cm": thickness_cm,
        }

    return {
        "gouge_1": _layer_payload("gouge_1", "gouge_thickness_1"),
        "gouge_2": _layer_payload("gouge_2", "gouge_thickness_2"),
    }

def load_pickle_stream(
    pkl_path: Path,
    *,
    repair_incomplete_tail: bool = False,
) -> list[dict]:
    """Load an append-only pickle stream.

    A process interruption can leave only the final pickle object truncated.
    When ``repair_incomplete_tail`` is true, that incomplete tail is removed
    while every previously completed object is preserved.
    """

    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        return []

    rows = []
    file_size = pkl_path.stat().st_size
    last_complete_offset = 0
    damaged_offset = None
    damaged_error = None

    with open(pkl_path, "rb") as stream:
        while stream.tell() < file_size:
            object_offset = stream.tell()
            try:
                row = pickle.load(stream)
            except (EOFError, pickle.UnpicklingError) as error:
                damaged_offset = object_offset
                damaged_error = error
                break
            if not isinstance(row, dict):
                raise TypeError(
                    f"pickle row at byte {object_offset} is not a dictionary"
                )
            rows.append(row)
            last_complete_offset = stream.tell()

    if damaged_offset is not None:
        if not repair_incomplete_tail:
            raise RuntimeError(
                f"incomplete pickle object at byte {damaged_offset} in "
                f"{pkl_path}: {damaged_error}"
            ) from damaged_error
        with open(pkl_path, "rb+") as stream:
            stream.truncate(last_complete_offset)
        print(
            "resume: removed incomplete pickle tail at byte "
            f"{damaged_offset}; preserved {len(rows)} complete rows",
            flush=True,
        )

    return rows

def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _resample_1d(x: np.ndarray, y: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """
    Interpolate y(x) onto x_ref. Outside: NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    out = np.full_like(x_ref, np.nan, dtype=float)
    inside = (x_ref >= x[0]) & (x_ref <= x[-1])
    if np.any(inside):
        out[inside] = np.interp(x_ref[inside], x, y)
    return out

def plot_gouge_layer_colormap_with_shear(
    pkl_path: Path,
    out_png: Path,
    *,
    layer: int,            # 1 or 2
    n_eta: int = 128,      # vertical resolution for the colormap
) -> None:
    rows = load_pickle_stream(pkl_path)

    # collect rows that have the required gouge payload
    key_eta = f"gouge_{layer}_eta"
    key_v   = f"gouge_{layer}_velocity"
    key_d   = f"gouge_{layer}_damping"

    rows2 = []
    for r in rows:
        if ("acquisition_time_s" in r) and ("shear_stress_MPa" in r) and (key_eta in r) and (key_v in r):
            if r.get(key_v, None) is not None:
                rows2.append(r)

    if not rows2:
        raise RuntimeError(f"No gouge_{layer} profile rows found in {pkl_path}.")

    # sort by acquisition time
    rows2.sort(key=lambda r: _safe_float(r["acquisition_time_s"]))

    t = np.array([_safe_float(r["acquisition_time_s"]) for r in rows2], dtype=float)
    shear = np.array([_safe_float(r["shear_stress_MPa"]) for r in rows2], dtype=float)

    # reference vertical coordinate: eta in [0,1]
    eta_ref = np.linspace(0.0, 1.0, n_eta, dtype=float)

    vel_mat = []
    damp_mat = []
    have_damp = True

    thickness_cm = np.array([_safe_float(r.get(f"gouge_{layer}_thickness_cm", np.nan)) for r in rows2], dtype=float)

    for r in rows2:
        eta = np.asarray(r[key_eta], dtype=float)
        v   = np.asarray(r[key_v], dtype=float)

        vel_mat.append(_resample_1d(eta, v, eta_ref))

        d = r.get(key_d, None)
        if d is None:
            have_damp = False
            damp_mat.append(np.full_like(eta_ref, np.nan, dtype=float))
        else:
            d = np.asarray(d, dtype=float)
            damp_mat.append(_resample_1d(eta, d, eta_ref))

    vel_mat = np.vstack(vel_mat)   # (nt, n_eta)
    damp_mat = np.vstack(damp_mat)

    # ---- Plot: 2 panels (velocity, damping), time on x, eta on y ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Velocity colormap
    ax = axes[0]
    im = ax.imshow(
        vel_mat.T,                         # transpose -> y=eta, x=time
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[t[0], t[-1], eta_ref[0], eta_ref[-1]],
    )
    ax.set_ylabel("Gouge coordinate η (0→1)")
    ax.set_title(f"Gouge {layer}: velocity evolution (colormap) + shear stress (line)")
    fig.colorbar(im, ax=ax, label="Velocity (cm/µs)")

    ax_s = ax.twinx()
    ax_s.plot(t, shear)
    ax_s.set_ylabel("Shear stress (MPa)")
    ax.grid(True, linestyle="--", linewidth=0.5)

    # Damping colormap
    ax = axes[1]
    im = ax.imshow(
        damp_mat.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[t[0], t[-1], eta_ref[0], eta_ref[-1]],
    )
    ax.set_xlabel("Acquisition time (s)")
    ax.set_ylabel("Gouge coordinate η (0→1)")
    ax.set_title(f"Gouge {layer}: damping evolution (colormap) + shear stress (line)")
    fig.colorbar(im, ax=ax, label="Damping (cm^2/µs)")

    ax_s = ax.twinx()
    ax_s.plot(t, shear)
    ax_s.set_ylabel("Shear stress (MPa)")
    ax.grid(True, linestyle="--", linewidth=0.5)

    # Optional: annotate thickness evolution (not as axis, but as info)
    if np.isfinite(thickness_cm).any():
        # show min/max thickness in figure footer
        th_ok = thickness_cm[np.isfinite(thickness_cm)]
        fig.text(
            0.01, 0.01,
            f"Thickness (cm) over plotted records: min={th_ok.min():.3f}, max={th_ok.max():.3f}. "
            f"Colormap y-axis uses normalized η; thickness shrink is encoded in the saved thickness values.",
            fontsize=9
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

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
    velocities = [
        assembly_dict["gouge_velocity_1"],
        assembly_dict["gouge_velocity_2"],
        assembly_dict["side1_params"]["velocity" + wt],
        assembly_dict["side2_params"]["velocity" + wt],
        assembly_dict["central_params"]["velocity" + wt],
    ]
    if assembly_dict.get("embedding_mode", "legacy_stack") == "legacy_stack":
        velocities.extend([
            assembly_dict["side1_params"]["pzt_velocity" + wt],
            assembly_dict["side1_params"]["pla_velocity" + wt],
        ])
    return min(velocities)

def max_assembly_velocity(assembly_dict: Dict[str, Any]):
    wt = assembly_dict["wave_type"]
    velocities = [
        assembly_dict["gouge_velocity_1"],
        assembly_dict["gouge_velocity_2"],
        assembly_dict["side1_params"]["velocity" + wt],
        assembly_dict["side2_params"]["velocity" + wt],
        assembly_dict["central_params"]["velocity" + wt],
    ]
    if assembly_dict.get("embedding_mode", "legacy_stack") == "legacy_stack":
        velocities.extend([
            assembly_dict["side1_params"]["pzt_velocity" + wt],
            assembly_dict["side1_params"]["pla_velocity" + wt],
        ])
    return max(velocities)
        
def update_assembly_dict_with_mech_data(
    assembly_dict: Dict[str, Any],
    mech_data: Tuple,
) -> None:
    """
    Update assembly_dict with per-waveform mechanical data entry
    """

    try: 
        assembly_dict["gouge_thickness_1"]      = mech_data.rgt_lt_mm / 10.0     # one layer thickness in mm -> convert to cm
        assembly_dict["gouge_thickness_2"]      = mech_data.rgt_lt_mm / 10.0     # for now, layers are assumed to have same thickness
        assembly_dict["ec_disp_mm"]             = mech_data.ec_disp_mm
        assembly_dict["rec_n"]                  = int(mech_data.records_na)

    except AttributeError:
        assembly_dict["gouge_thickness_1"]      = mech_data.lt_ec_mm / (10.0)     # entire layer thickness in mm -> divide by 2 (2 layer) and convert to cm
        assembly_dict["gouge_thickness_2"]      = mech_data.lt_ec_mm / (10.0)     # for now, layers are assumed to have same thickness
        assembly_dict["ec_disp_mm"]             = mech_data.lp_disp_mm
        assembly_dict["rec_n"]                  = mech_data.Index

    assembly_dict["normal_stress"]          = mech_data.normal_stress_MPa
    assembly_dict["shear_stress"]           = mech_data.shear_stress_MPa
    assembly_dict["acquisition_time"]       = mech_data.time_s
    assembly_dict['idx_wave']               = mech_data.uw_idx
    # define assembly sample_dimensions all together. It is usefull for handling forward modeling 
    assembly_dict["sample_dimensions"] = [
                                            assembly_dict["side1_params"]["z_pzt2grove"],
                                            assembly_dict["gouge_thickness_1"] ,
                                            assembly_dict["central_params"]["z"],
                                            assembly_dict["gouge_thickness_2"],
                                            assembly_dict["side2_params"]["z_pzt2grove"],
                                        ]
        
def set_assembly_dict_guessed_with_gouge_velocity_and_damping(
    assembly_dict: Dict[str, Any],
    gouge_velocity_tuple: Union[Tuple[float],Tuple[np.ndarray]],
    gouge_damping_tuple: Union[Tuple[float],Tuple[np.ndarray]],

) -> Dict[str, Any]:
    assembly_dict_guessed = assembly_dict.copy()
    assembly_dict_guessed["gouge_velocity_1"], assembly_dict_guessed["gouge_velocity_2"] = gouge_velocity_tuple
    assembly_dict_guessed["gouge_damping_1"], assembly_dict_guessed["gouge_damping_2"] = gouge_damping_tuple
    return assembly_dict_guessed


def set_assembly_dict_guessed_with_gouge_thicknesses(
    assembly_dict: Dict[str, Any],
    gouge_thickness_1_cm: float,
    gouge_thickness_2_cm: float,
) -> Dict[str, Any]:
    """Copy an assembly and keep its DDS geometry fields consistent."""

    h1 = float(gouge_thickness_1_cm)
    h2 = float(gouge_thickness_2_cm)
    if not np.isfinite(h1) or not np.isfinite(h2) or h1 <= 0.0 or h2 <= 0.0:
        raise ValueError("gouge thicknesses must be positive and finite")

    assembly_dict_guessed = assembly_dict.copy()
    sample_dimensions = list(assembly_dict["sample_dimensions"])
    if len(sample_dimensions) != 5:
        raise ValueError("DDS sample_dimensions must contain five entries")

    assembly_dict_guessed["gouge_thickness_1"] = h1
    assembly_dict_guessed["gouge_thickness_2"] = h2
    sample_dimensions[1] = h1
    sample_dimensions[3] = h2
    assembly_dict_guessed["sample_dimensions"] = sample_dimensions
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

    # t_grooves = (
    #     groove_dt(side1["h_grooves"], v_s1, v_gouge_1) +
    #     groove_dt(cent ["h_grooves"], v_sc, v_gouge_1) +
    #     groove_dt(cent ["h_grooves"], v_sc, v_gouge_2) +
    #     groove_dt(side2["h_grooves"], v_s2, v_gouge_2)
    # )
    t_grooves = (
        side1["h_grooves"]/v_s1 +
        cent["h_grooves"]/ v_sc +
        cent["h_grooves"]/ v_sc +
        side2["h_grooves"]/ v_s2
    )

    # PZT: if source and receiver are at the centers of the PZTs, total is (w/2 + w/2)/vp = w/vp
    # t_pzt = side1["pzt_layer_width"] / side1["pzt_velocity" + wave_type]
    t_pzt = 0

    return t_steel + t_gouge + t_grooves + t_pzt


# Read-only state inherited by the local-inversion worker processes.  Only the
# initializer writes this dictionary; every lane deep-copies its own model and
# assembly before starting the inversions.
_LOCAL_INVERSION_WORKER_CONTEXT: Dict[str, Any] = {}


def _build_local_result_row(
    rec_n: int,
    assembly_dict: Dict[str, Any],
    simulation: UltrasonicModeler,
    result_local: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the serialisable payload saved for one local inversion."""

    gouge_payload = extract_gouge_profiles(simulation)
    objective = getattr(simulation, "objective_components", {})
    local_spectral = result_local.get("local_spectral_diagnostics")

    result_row = {
        "rec_n": rec_n,
        "idx_wave": int(assembly_dict["idx_wave"]),
        "acquisition_time_s": float(assembly_dict["acquisition_time"]),
        "normal_stress_MPa": float(assembly_dict["normal_stress"]),
        "shear_stress_MPa": float(assembly_dict["shear_stress"]),
        "ec_disp_mm": float(assembly_dict["ec_disp_mm"]),
        "gouge_thickness_1_cm": float(assembly_dict["gouge_thickness_1"]),
        "gouge_thickness_2_cm": float(assembly_dict["gouge_thickness_2"]),
        "x_axis_cm": simulation.grid_handler.spatial_axis.copy(),
        "misfit": float(simulation.misfit),
        "relative_l2_misfit": objective.get("relative_l2"),
        "amplitude_misfit": objective.get("amplitude_misfit"),
        "spectral_shape_misfit": objective.get("spectral_shape_misfit"),
        "phase_misfit": objective.get("phase_misfit"),
        "synthetic_to_recorded_amplitude_ratio": objective.get(
            "amplitude_ratio"
        ),
        "waveform_correlation": objective.get("correlation"),
        "spectral_gamma": objective.get("spectral_gamma"),
        "local_update_strategy": getattr(
            simulation,
            "local_update_strategy",
            None,
        ),
        "local_iteration_history": list(
            getattr(simulation, "local_inversion_history", [])
        ),
        "reference_trace_scale": float(simulation.reference_trace_scale),
        "local_spectral_shape_distance": (
            None
            if local_spectral is None
            else local_spectral["recorded_vs_simulated_distance"]
        ),
        "local_stf_recorded_spectral_shape_distance": (
            None
            if local_spectral is None
            else local_spectral["stf_vs_recorded_distance"]
        ),
        "local_spectral_frequency_min_mhz": (
            None
            if local_spectral is None
            else local_spectral["frequency_min_mhz"]
        ),
        "local_spectral_frequency_max_mhz": (
            None
            if local_spectral is None
            else local_spectral["frequency_max_mhz"]
        ),
        "avg_gouge_velocity": float(
            getattr(simulation, "average_gouge_velocity", np.nan)
        ),
        "avg_gouge_damping": float(
            getattr(simulation, "average_gouge_damping", np.nan)
        ),
        "velocity_profile": (
            simulation.velocity_model_handler.velocity_array.copy()
        ),
        "damping_profile": (
            simulation.velocity_model_handler.damping_array.copy()
            if hasattr(simulation.velocity_model_handler, "damping_array")
            else None
        ),
        "gouge_1_eta": gouge_payload["gouge_1"]["eta"],
        "gouge_1_velocity": gouge_payload["gouge_1"]["velocity"],
        "gouge_1_damping": gouge_payload["gouge_1"]["damping"],
        "gouge_1_thickness_cm": gouge_payload["gouge_1"]["thickness_cm"],
        "gouge_2_eta": gouge_payload["gouge_2"]["eta"],
        "gouge_2_velocity": gouge_payload["gouge_2"]["velocity"],
        "gouge_2_damping": gouge_payload["gouge_2"]["damping"],
        "gouge_2_thickness_cm": gouge_payload["gouge_2"]["thickness_cm"],
    }
    # From this version onward every completed record is also a restart
    # checkpoint.  The state deliberately excludes the full space-time
    # wavefield, so its size is O(nx + nt), not O(nx * nt).
    result_row["local_continuation_state"] = (
        _capture_local_continuation_state(assembly_dict, simulation)
    )
    return result_row


def _initialize_local_inversion_worker(
    paired_rows,
    observed_waveform_data: np.ndarray,
    outfile_name: str,
    params: Dict[str, Any],
    assembly_seed: Dict[str, Any],
    simulation_seed: UltrasonicModeler,
    stf_handler: UltrasonicDataHandler,
) -> None:
    """Install the read-only context inherited by every local worker."""

    global _LOCAL_INVERSION_WORKER_CONTEXT
    _LOCAL_INVERSION_WORKER_CONTEXT = {
        "paired_rows": paired_rows,
        "observed_waveform_data": observed_waveform_data,
        "outfile_name": outfile_name,
        "params": params,
        "assembly_seed": assembly_seed,
        "simulation_seed": simulation_seed,
        "stf_handler": stf_handler,
    }


def _capture_local_continuation_state(
    assembly_dict: Dict[str, Any],
    simulation: UltrasonicModeler,
) -> Dict[str, Any]:
    """Return the compact state required to continue a local inversion lane.

    The complete ``UltrasonicModeler`` is intentionally not sent through a
    multiprocessing queue: it may retain large transient allocations.  The
    next process starts from the common seed and receives only the accepted
    material/grid state and the scalar inversion metadata.
    """

    simulation_attributes = {}
    for attribute_name in (
        "synthetic_waveform",
        "trace_scale",
        "reference_trace_scale",
        "misfit",
        "average_gouge_velocity",
        "average_gouge_damping",
        "objective_components",
        "local_update_strategy",
        "local_inversion_history",
        "amplitude_scale_mode",
        "spectral_gamma",
        "objective_taper_fraction",
    ):
        if hasattr(simulation, attribute_name):
            simulation_attributes[attribute_name] = deepcopy(
                getattr(simulation, attribute_name)
            )

    velocity_handler = simulation.velocity_model_handler
    return {
        "assembly_dict": deepcopy(assembly_dict),
        "grid_spatial_axis": simulation.grid_handler.spatial_axis.copy(),
        "grid_total_grid_points": int(
            simulation.grid_handler.total_grid_points
        ),
        "velocity_array": velocity_handler.velocity_array.copy(),
        "damping_array": velocity_handler.damping_array.copy(),
        "density_array": velocity_handler.density_array.copy(),
        "velocity_x": velocity_handler.x.copy(),
        "idx_dict": deepcopy(velocity_handler.idx_dict),
        "source_spatial_function": (
            simulation.source_handler.spatial_function.copy()
        ),
        "receiver_spatial_function": (
            simulation.receiver_handler.spatial_function.copy()
        ),
        "simulation_attributes": simulation_attributes,
    }


def _restore_local_continuation_state(
    simulation: UltrasonicModeler,
    continuation_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Restore a compact lane checkpoint into a fresh worker-local model."""

    simulation.grid_handler.spatial_axis = continuation_state[
        "grid_spatial_axis"
    ].copy()
    simulation.grid_handler.total_grid_points = int(
        continuation_state["grid_total_grid_points"]
    )

    velocity_handler = simulation.velocity_model_handler
    velocity_handler.velocity_array = continuation_state[
        "velocity_array"
    ].copy()
    velocity_handler.damping_array = continuation_state[
        "damping_array"
    ].copy()
    velocity_handler.density_array = continuation_state[
        "density_array"
    ].copy()
    velocity_handler.x = continuation_state["velocity_x"].copy()
    velocity_handler.idx_dict = deepcopy(continuation_state["idx_dict"])

    simulation.source_handler.spatial_function = continuation_state[
        "source_spatial_function"
    ].copy()
    simulation.receiver_handler.spatial_function = continuation_state[
        "receiver_spatial_function"
    ].copy()
    simulation.wavefield_forward = None

    for attribute_name, value in continuation_state[
        "simulation_attributes"
    ].items():
        setattr(simulation, attribute_name, deepcopy(value))

    assembly_dict = deepcopy(continuation_state["assembly_dict"])
    simulation.assembly_dict = assembly_dict
    return assembly_dict


def _run_local_inversion_chunk(
    row_positions: list[int],
    continuation_state: Dict[str, Any] | None = None,
):
    """Invert a short consecutive part of one interlaced lane."""

    worker_pid = os.getpid()
    chunk_label = list(row_positions)

    print(
        f"[worker {worker_pid}] starting chunk {chunk_label}",
        flush=True,
    )

    context = _LOCAL_INVERSION_WORKER_CONTEXT
    simulation = deepcopy(context["simulation_seed"])
    if continuation_state is None:
        assembly_lane = deepcopy(context["assembly_seed"])
    else:
        assembly_lane = _restore_local_continuation_state(
            simulation,
            continuation_state,
        )
    chunk_results = []

    for lane_index, row_position in enumerate(row_positions, start=1):
        row = context["paired_rows"][row_position]

        update_assembly_dict_with_mech_data(
            assembly_dict=assembly_lane,
            mech_data=row,
        )

        rec_n = int(assembly_lane["rec_n"])

        print(
            f"[worker {worker_pid}] chunk {chunk_label}: "
            f"starting record {lane_index}/{len(row_positions)}, "
            f"rec_n={rec_n}",
            flush=True,
        )

        outfile_name_wave = (
            context["outfile_name"] + f"_rec_n_{rec_n}"
        )

        simulation.observed_waveform = context[
            "observed_waveform_data"
        ][int(row.uw_idx)]

        # Counter persistent within this continuation lane.
        # It is saved inside local_continuation_state and therefore survives
        # worker recycling and later resume from the pickle.
        local_plot_counter = int(
            assembly_lane.get("_local_plot_counter", 0)
        )
        assembly_lane["_local_plot_counter"] = local_plot_counter

        result_local = local_refine_waveform(
            outfile_name=outfile_name_wave,
            params=context["params"],
            assembly_dict=assembly_lane,
            simulation=simulation,
            stf_handler=context["stf_handler"],
        )

        # Prepare the counter for the next record in the same lane.
        assembly_lane["_local_plot_counter"] = local_plot_counter + 1

        accepted_simulation = result_local.get("simulation_handler")

        if accepted_simulation is None:
            print(
                f"[worker {worker_pid}] chunk {chunk_label}: "
                f"record rec_n={rec_n} skipped",
                flush=True,
            )
            continue

        simulation = accepted_simulation

        local_row = _build_local_result_row(
            rec_n=rec_n,
            assembly_dict=assembly_lane,
            simulation=simulation,
            result_local=result_local,
        )
        local_row["local_anchor_stride"] = int(
            context["params"].get("local_anchor_stride", 10)
        )
        local_row["local_records_per_worker"] = int(
            context["params"].get("local_records_per_worker", 3)
        )
        chunk_results.append((row_position, local_row))

        print(
            f"[worker {worker_pid}] chunk {chunk_label}: "
            f"completed rec_n={rec_n}, "
            f"misfit={float(simulation.misfit):.6f}",
            flush=True,
        )

    print(
        f"[worker {worker_pid}] completed chunk {chunk_label}",
        flush=True,
    )

    continuation_state = _capture_local_continuation_state(
        assembly_lane,
        simulation,
    )
    return chunk_label, chunk_results, continuation_state


def _run_local_inversion_chunk_process(
    result_queue,
    task_id: int,
    lane_id: int,
    chunk_index: int,
    row_positions: list[int],
    continuation_state: Dict[str, Any] | None,
) -> None:
    """Child-process wrapper that always reports Python failures to parent."""

    try:
        chunk_label, chunk_results, next_state = _run_local_inversion_chunk(
            row_positions,
            continuation_state,
        )
        message = {
            "status": "ok",
            "task_id": task_id,
            "lane_id": lane_id,
            "chunk_index": chunk_index,
            "chunk_label": chunk_label,
            "chunk_results": chunk_results,
            "continuation_state": next_state,
        }
    except BaseException:
        message = {
            "status": "error",
            "task_id": task_id,
            "lane_id": lane_id,
            "chunk_index": chunk_index,
            "chunk_label": list(row_positions),
            "traceback": traceback.format_exc(),
        }

    result_queue.put(message)
    # Ensure the feeder has written the complete message before this short-
    # lived process exits.  This avoids a successful exit with a late queue
    # payload being mistaken for a lost task by the parent.
    result_queue.close()
    result_queue.join_thread()

def _build_local_inversion_lanes(
    paired_rows,
    remaining_positions: list[int],
    anchor_stride: int,
) -> list[list[int]]:
    """Split valid records into interlaced chains separated by a fixed stride."""

    if anchor_stride < 1:
        raise ValueError("local_anchor_stride must be at least 1")

    valid_positions = [
        position
        for position in remaining_positions
        if np.isfinite(getattr(paired_rows[position], "Index", np.nan))
    ]
    return [
        valid_positions[offset::anchor_stride]
        for offset in range(min(anchor_stride, len(valid_positions)))
        if valid_positions[offset::anchor_stride]
    ]


def _chunk_local_inversion_lane(
    row_positions: list[int],
    records_per_worker: int,
) -> list[list[int]]:
    """Split a logical continuation lane into bounded process-lifetime jobs."""

    if records_per_worker < 1:
        raise ValueError("local_records_per_worker must be at least 1")
    return [
        row_positions[start:start + records_per_worker]
        for start in range(0, len(row_positions), records_per_worker)
    ]


def _run_parallel_local_inversions(
    paired_rows,
    remaining_positions: list[int],
    observed_waveform_data: np.ndarray,
    outfile_name: str,
    params: Dict[str, Any],
    assembly_seed: Dict[str, Any],
    simulation_seed: UltrasonicModeler,
    stf_handler: UltrasonicDataHandler,
    lane_specs: list[Dict[str, Any]] | None = None,
):
    """Yield completed rows while recycling workers after short chunks.

    ``local_anchor_stride`` defines the scientific continuation chains.
    ``local_records_per_worker`` independently limits how many records one
    operating-system process may invert before its memory is released.
    """

    anchor_stride = int(params.get("local_anchor_stride", 10))
    records_per_worker = int(params.get("local_records_per_worker", 3))

    if lane_specs is None:
        lanes = _build_local_inversion_lanes(
            paired_rows,
            remaining_positions,
            anchor_stride,
        )
        lane_specs = [
            {
                "lane_id": lane_id,
                "positions": lane,
                "continuation_state": None,
            }
            for lane_id, lane in enumerate(lanes)
        ]
    else:
        lane_specs = [
            {
                "lane_id": int(spec["lane_id"]),
                "positions": list(spec["positions"]),
                "continuation_state": deepcopy(
                    spec.get("continuation_state")
                ),
            }
            for spec in lane_specs
            if spec.get("positions")
        ]

    if not lane_specs:
        return

    lane_chunks = [
        _chunk_local_inversion_lane(
            spec["positions"],
            records_per_worker,
        )
        for spec in lane_specs
    ]
    total_chunks = sum(len(chunks) for chunks in lane_chunks)

    num_processes = resolve_worker_count(
        params,
        "local_inversion_workers",
        len(lane_specs),
    )

    initializer_arguments = (
        paired_rows,
        observed_waveform_data,
        outfile_name,
        params,
        deepcopy(assembly_seed),
        simulation_seed,
        stf_handler,
    )

    # if num_processes <= 1:
    #     _initialize_local_inversion_worker(*initializer_arguments)

    #     completed_chunks = 0
    #     for state_index, chunks in enumerate(lane_chunks):
    #         spec = lane_specs[state_index]
    #         lane_id = int(spec["lane_id"])
    #         continuation_state = spec["continuation_state"]
    #         for chunk_index, chunk in enumerate(chunks):
    #             (
    #                 chunk_label,
    #                 chunk_output,
    #                 continuation_state,
    #             ) = _run_local_inversion_chunk(
    #                 chunk,
    #                 continuation_state,
    #             )
    #             completed_chunks += 1
    #             print(
    #                 f"[parent] received chunk "
    #                 f"{completed_chunks}/{total_chunks}, "
    #                 f"lane={lane_id + 1}, part={chunk_index + 1}: "
    #                 f"{chunk_label}",
    #                 flush=True,
    #             )
    #             for item in chunk_output:
    #                 yield item

    #     return

    print(
        "Parallel local inversion: "
        f"{len(lane_specs)} active interlaced lanes, "
        f"stride={anchor_stride}, "
        f"workers={num_processes}, "
        f"max {records_per_worker} records/process, "
        f"chunks={total_chunks}",
        flush=True,
    )

    multiprocessing_context = get_context("fork")
    _initialize_local_inversion_worker(*initializer_arguments)
    result_queue = multiprocessing_context.Queue()

    lane_states = [
        {
            "lane_id": int(spec["lane_id"]),
            "chunks": chunks,
            "next_chunk": 0,
            "continuation_state": spec["continuation_state"],
        }
        for spec, chunks in zip(lane_specs, lane_chunks)
    ]
    ready_lanes = deque(range(len(lane_states)))
    active_tasks: Dict[int, Dict[str, Any]] = {}
    next_task_id = 0
    completed_chunks = 0

    def _start_ready_chunks() -> None:
        nonlocal next_task_id
        while ready_lanes and len(active_tasks) < num_processes:
            state_index = ready_lanes.popleft()
            lane_state = lane_states[state_index]
            lane_id = int(lane_state["lane_id"])
            chunk_index = int(lane_state["next_chunk"])
            chunk = lane_state["chunks"][chunk_index]
            task_id = next_task_id
            next_task_id += 1

            process = multiprocessing_context.Process(
                target=_run_local_inversion_chunk_process,
                args=(
                    result_queue,
                    task_id,
                    lane_id,
                    chunk_index,
                    chunk,
                    lane_state["continuation_state"],
                ),
                name=f"local-lane-{lane_id + 1}-part-{chunk_index + 1}",
            )
            process.start()
            active_tasks[task_id] = {
                "process": process,
                "state_index": state_index,
                "lane_id": lane_id,
                "chunk_index": chunk_index,
                "chunk": chunk,
            }

    try:
        _start_ready_chunks()
        while active_tasks:
            try:
                message = result_queue.get(timeout=1.0)
            except queue.Empty:
                failed_tasks = []
                for task_id, task in active_tasks.items():
                    process = task["process"]
                    if process.exitcode not in (None, 0):
                        failed_tasks.append((task_id, task, process.exitcode))
                if failed_tasks:
                    task_id, task, exitcode = failed_tasks[0]
                    raise RuntimeError(
                        "local inversion worker exited before returning its "
                        f"chunk: lane={task['lane_id'] + 1}, "
                        f"part={task['chunk_index'] + 1}, "
                        f"positions={task['chunk']}, exitcode={exitcode}. "
                        "A negative exit code usually identifies the signal "
                        "used by the OOM killer. Completed earlier chunks "
                        "have already been flushed to the result pickle."
                    )
                continue

            task_id = int(message["task_id"])
            task = active_tasks.pop(task_id, None)
            if task is None:
                raise RuntimeError(
                    f"received an unknown local-inversion task id {task_id}"
                )

            process = task["process"]
            process.join()
            if message["status"] != "ok":
                raise RuntimeError(
                    "local inversion worker failed in "
                    f"lane={message['lane_id'] + 1}, "
                    f"part={message['chunk_index'] + 1}, "
                    f"positions={message['chunk_label']}:\n"
                    f"{message['traceback']}"
                )
            if process.exitcode != 0:
                raise RuntimeError(
                    "local inversion worker returned data but exited with "
                    f"exitcode={process.exitcode}"
                )

            lane_id = int(message["lane_id"])
            state_index = int(task["state_index"])
            lane_state = lane_states[state_index]
            lane_state["continuation_state"] = message[
                "continuation_state"
            ]
            lane_state["next_chunk"] += 1
            completed_chunks += 1

            print(
                f"[parent] received chunk "
                f"{completed_chunks}/{total_chunks}, "
                f"lane={lane_id + 1}, "
                f"part={message['chunk_index'] + 1}: "
                f"{message['chunk_label']}",
                flush=True,
            )

            # The caller appends and flushes each of these rows immediately.
            for item in message["chunk_results"]:
                yield item

            if lane_state["next_chunk"] < len(lane_state["chunks"]):
                # Round-robin scheduling prevents one long logical lane from
                # monopolising all newly created processes.
                ready_lanes.append(state_index)
            _start_ready_chunks()
    finally:
        for task in active_tasks.values():
            process = task["process"]
            if process.is_alive():
                process.terminate()
            process.join()
        result_queue.close()
        result_queue.join_thread()


def _is_global_result_row(row: Dict[str, Any]) -> bool:
    return "global_best_misfit" in row


def _paired_position_lookup(paired_rows) -> Dict[int, int]:
    """Map the persisted waveform index to its current paired-row position."""

    lookup = {}
    for position, row in enumerate(paired_rows):
        if not np.isfinite(getattr(row, "Index", np.nan)):
            continue
        waveform_index = int(row.uw_idx)
        if waveform_index in lookup:
            raise RuntimeError(
                f"paired table contains duplicate uw_idx={waveform_index}"
            )
        lookup[waveform_index] = position
    return lookup


def _infer_resume_anchor_stride(
    local_rows_in_stream: list[Dict[str, Any]],
    position_by_waveform_index: Dict[int, int],
    configured_stride: int,
) -> int:
    """Recover the scientific lane stride used by a legacy pickle."""

    stored_values = {
        int(row["local_anchor_stride"])
        for row in local_rows_in_stream
        if row.get("local_anchor_stride") is not None
    }
    if len(stored_values) > 1:
        raise RuntimeError(
            "pickle contains incompatible local_anchor_stride values: "
            f"{sorted(stored_values)}"
        )
    if stored_values:
        return stored_values.pop()

    positions = []
    for row in local_rows_in_stream:
        waveform_index = int(row["idx_wave"])
        if waveform_index not in position_by_waveform_index:
            raise RuntimeError(
                "saved waveform is absent from the current paired table: "
                f"idx_wave={waveform_index}"
            )
        positions.append(position_by_waveform_index[waveform_index])

    positive_steps = [
        following - previous
        for previous, following in zip(positions, positions[1:])
        if following > previous
    ]
    if not positive_steps:
        return configured_stride

    counts = Counter(positive_steps)
    inferred_stride, occurrences = counts.most_common(1)[0]
    # A single positive jump may merely be the boundary between asynchronously
    # returned lanes.  Require repeated evidence before overriding the config.
    if occurrences < 2:
        return configured_stride
    return int(inferred_stride)


def _saved_layer_mean(
    row: Dict[str, Any],
    profile_key: str,
    fallback: float,
) -> float:
    values = row.get(profile_key)
    if values is None:
        return float(fallback)
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float(fallback)


def _interpolate_saved_model_profile(
    saved_axis: np.ndarray,
    saved_profile: np.ndarray,
    target_axis: np.ndarray,
    *,
    quantity: str,
) -> np.ndarray:
    """Transfer a saved nodal profile to an equivalent rebuilt grid."""

    saved_axis = np.asarray(saved_axis, dtype=float).reshape(-1)
    saved_profile = np.asarray(saved_profile, dtype=float).reshape(-1)
    target_axis = np.asarray(target_axis, dtype=float).reshape(-1)
    if saved_axis.size != saved_profile.size or saved_axis.size < 2:
        raise RuntimeError(
            f"saved {quantity} profile is inconsistent with its spatial axis"
        )
    if not np.all(np.diff(saved_axis) > 0.0):
        raise RuntimeError("saved spatial axis is not strictly increasing")

    span = max(float(saved_axis[-1] - saved_axis[0]), 1.0e-15)
    endpoint_error = max(
        abs(float(target_axis[0] - saved_axis[0])),
        abs(float(target_axis[-1] - saved_axis[-1])),
    )
    if endpoint_error > 0.02 * span:
        raise RuntimeError(
            f"cannot resume {quantity}: rebuilt and saved domains differ by "
            f"{endpoint_error:.6g} cm"
        )
    if (
        saved_axis.size == target_axis.size
        and np.allclose(saved_axis, target_axis, rtol=1.0e-9, atol=1.0e-12)
    ):
        return saved_profile.copy()
    return np.interp(target_axis, saved_axis, saved_profile)


def _resume_points_per_wavelength(
    saved_row: Dict[str, Any],
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
) -> int:
    saved_axis = np.asarray(saved_row["x_axis_cm"], dtype=float).reshape(-1)
    if saved_axis.size < 2:
        raise RuntimeError("saved spatial axis is too short to reconstruct grid")
    saved_dx = float(np.median(np.diff(saved_axis)))
    minimum_velocity, _ = _simulation_velocity_limits(params, assembly_dict)
    frequency_cutoff = float(params["frequency_cutoff"])
    inferred = int(round(minimum_velocity / (frequency_cutoff * saved_dx)))
    return max(2, inferred)


def _rebuild_simulation_from_saved_row(
    *,
    saved_row: Dict[str, Any],
    global_row: Dict[str, Any],
    paired_row,
    observed_waveform_data: np.ndarray,
    waveform_metadata: Dict[str, Any],
    stf_handler: UltrasonicDataHandler,
    params: Dict[str, Any],
    base_assembly: Dict[str, Any],
) -> tuple[Dict[str, Any], UltrasonicModeler]:
    """Rebuild a lightweight modeler and inject a legacy saved model."""

    assembly = deepcopy(base_assembly)
    update_assembly_dict_with_mech_data(assembly, paired_row)

    if _is_global_result_row(saved_row):
        h1 = saved_row.get(
            "best_gouge_thickness_1_cm",
            saved_row.get("gouge_thickness_1_cm"),
        )
        h2 = saved_row.get(
            "best_gouge_thickness_2_cm",
            saved_row.get("gouge_thickness_2_cm"),
        )
    else:
        h1 = saved_row.get("gouge_thickness_1_cm")
        h2 = saved_row.get("gouge_thickness_2_cm")
    if h1 is not None and h2 is not None:
        assembly.update(
            set_assembly_dict_guessed_with_gouge_thicknesses(
                assembly,
                float(h1),
                float(h2),
            )
        )

    fallback_c1 = float(global_row["best_gouge_velocity_1"])
    fallback_c2 = float(global_row["best_gouge_velocity_2"])
    fallback_damping = float(global_row["best_gouge_damping"])
    assembly["gouge_velocity_1"] = _saved_layer_mean(
        saved_row,
        "gouge_1_velocity",
        saved_row.get("avg_gouge_velocity", fallback_c1),
    )
    assembly["gouge_velocity_2"] = _saved_layer_mean(
        saved_row,
        "gouge_2_velocity",
        saved_row.get("avg_gouge_velocity", fallback_c2),
    )
    assembly["gouge_damping_1"] = _saved_layer_mean(
        saved_row,
        "gouge_1_damping",
        saved_row.get("avg_gouge_damping", fallback_damping),
    )
    assembly["gouge_damping_2"] = _saved_layer_mean(
        saved_row,
        "gouge_2_damping",
        saved_row.get("avg_gouge_damping", fallback_damping),
    )

    gouge_density = global_row.get("impedance_best_gouge_density_g_cm3")
    if gouge_density is None:
        gouge_density = global_row.get("best_gouge_density_g_cm3")
    if gouge_density is not None and np.isfinite(float(gouge_density)):
        assembly.update(
            _assembly_with_common_gouge_density(
                assembly,
                params,
                float(gouge_density),
            )
        )

    observed_time = np.asarray(
        waveform_metadata["time_ax_waveform"],
        dtype=float,
    )
    observed_waveform = observed_waveform_data[int(paired_row.uw_idx)]
    stf_time = np.asarray(
        stf_handler.metadata["time_ax_waveform"],
        dtype=float,
    )
    stf_duration = float(stf_time[-1] - stf_time[0])
    arrival_time = compute_dds_travel_time(
        assembly,
        assembly["gouge_velocity_1"],
        assembly["gouge_velocity_2"],
    )
    misfit_interval = direct_misfit_interval(
        observed_time,
        arrival_min_us=arrival_time,
        arrival_max_us=arrival_time,
        stf_duration_us=stf_duration,
        padding_us=params["direct_window_padding_us"],
    )
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(
        params,
        assembly,
    )
    points_per_wavelength = _resume_points_per_wavelength(
        saved_row,
        params,
        assembly,
    )

    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly,
        misfit_interval=misfit_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=max(
            float(assembly["gouge_damping_1"]),
            float(assembly["gouge_damping_2"]),
        ),
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=points_per_wavelength,
        store_wavefield=False,
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=False,
        enable_plotting=False,
        make_movie=False,
        plot_output_path=None,
    )

    saved_axis = np.asarray(saved_row["x_axis_cm"], dtype=float)
    target_axis = simulation.grid_handler.spatial_axis
    simulation.velocity_model_handler.velocity_array = (
        _interpolate_saved_model_profile(
            saved_axis,
            saved_row["velocity_profile"],
            target_axis,
            quantity="velocity",
        )
    )
    simulation.velocity_model_handler.damping_array = (
        _interpolate_saved_model_profile(
            saved_axis,
            saved_row["damping_profile"],
            target_axis,
            quantity="damping",
        )
    )
    if saved_row.get("density_profile") is not None:
        simulation.velocity_model_handler.density_array = (
            _interpolate_saved_model_profile(
                saved_axis,
                saved_row["density_profile"],
                target_axis,
                quantity="density",
            )
        )

    reference_scale = float(
        saved_row.get(
            "reference_trace_scale",
            global_row["reference_trace_scale"],
        )
    )
    simulation.reference_trace_scale = reference_scale
    simulation.trace_scale = reference_scale
    simulation.misfit = float(
        saved_row.get("misfit", saved_row.get("global_best_misfit", np.nan))
    )
    simulation.average_gouge_velocity = float(
        saved_row.get(
            "avg_gouge_velocity",
            0.5 * (
                assembly["gouge_velocity_1"]
                + assembly["gouge_velocity_2"]
            ),
        )
    )
    simulation.average_gouge_damping = float(
        saved_row.get(
            "avg_gouge_damping",
            0.5 * (
                assembly["gouge_damping_1"]
                + assembly["gouge_damping_2"]
            ),
        )
    )
    simulation.objective_components = {
        "relative_l2": saved_row.get("relative_l2_misfit"),
        "amplitude_misfit": saved_row.get("amplitude_misfit"),
        "spectral_shape_misfit": saved_row.get("spectral_shape_misfit"),
        "phase_misfit": saved_row.get("phase_misfit"),
        "amplitude_ratio": saved_row.get(
            "synthetic_to_recorded_amplitude_ratio"
        ),
        "correlation": saved_row.get("waveform_correlation"),
        "spectral_gamma": saved_row.get("spectral_gamma"),
    }
    simulation.local_update_strategy = saved_row.get(
        "local_update_strategy"
    )
    simulation.local_inversion_history = list(
        saved_row.get("local_iteration_history", [])
    )
    simulation.assembly_dict = assembly
    simulation.wavefield_forward = None
    return assembly, simulation


def _resume_checkpoint_state(
    *,
    saved_row: Dict[str, Any],
    global_row: Dict[str, Any],
    paired_row,
    observed_waveform_data: np.ndarray,
    waveform_metadata: Dict[str, Any],
    stf_handler: UltrasonicDataHandler,
    params: Dict[str, Any],
    base_assembly: Dict[str, Any],
) -> tuple[Dict[str, Any], UltrasonicModeler, Dict[str, Any]]:
    assembly, simulation = _rebuild_simulation_from_saved_row(
        saved_row=saved_row,
        global_row=global_row,
        paired_row=paired_row,
        observed_waveform_data=observed_waveform_data,
        waveform_metadata=waveform_metadata,
        stf_handler=stf_handler,
        params=params,
        base_assembly=base_assembly,
    )
    stored_state = saved_row.get("local_continuation_state")
    if stored_state is None:
        stored_state = saved_row.get("global_continuation_state")
    continuation_state = (
        deepcopy(stored_state)
        if stored_state is not None
        else _capture_local_continuation_state(assembly, simulation)
    )
    return assembly, simulation, continuation_state


def _resume_local_inversions_from_pickle(
    *,
    existing_rows: list[Dict[str, Any]],
    results_pkl: Path,
    paired_rows,
    observed_waveform_data: np.ndarray,
    waveform_metadata: Dict[str, Any],
    outfile_name: str,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    stf_handler: UltrasonicDataHandler,
) -> None:
    """Append only the local inversions missing from an existing stream."""

    global_rows = [row for row in existing_rows if _is_global_result_row(row)]
    if len(global_rows) != 1:
        raise RuntimeError(
            "resume requires exactly one global-search row; found "
            f"{len(global_rows)}"
        )
    global_row = global_rows[0]
    local_rows_in_stream = [
        row for row in existing_rows if not _is_global_result_row(row)
    ]
    position_by_waveform_index = _paired_position_lookup(paired_rows)

    global_waveform_index = int(global_row["idx_wave"])
    if global_waveform_index not in position_by_waveform_index:
        raise RuntimeError(
            "global pickle row does not belong to the current input data: "
            f"idx_wave={global_waveform_index}"
        )
    anchor_position = position_by_waveform_index[global_waveform_index]

    local_row_by_position = {}
    for row in local_rows_in_stream:
        waveform_index = int(row["idx_wave"])
        if waveform_index not in position_by_waveform_index:
            raise RuntimeError(
                "local pickle row does not belong to the current input data: "
                f"idx_wave={waveform_index}"
            )
        position = position_by_waveform_index[waveform_index]
        if position in local_row_by_position:
            raise RuntimeError(
                "pickle contains duplicate local inversion for paired "
                f"position {position}, idx_wave={waveform_index}"
            )
        local_row_by_position[position] = row

    configured_stride = int(params.get("local_anchor_stride", 10))
    resume_stride = _infer_resume_anchor_stride(
        local_rows_in_stream,
        position_by_waveform_index,
        configured_stride,
    )
    if resume_stride != configured_stride:
        print(
            "resume: existing pickle was generated with "
            f"local_anchor_stride={resume_stride}; overriding the current "
            f"value {configured_stride} only for this file",
            flush=True,
        )
    resume_params = dict(params)
    resume_params["local_anchor_stride"] = resume_stride

    anchor_saved_row = local_row_by_position.get(
        anchor_position,
        global_row,
    )
    (
        anchor_assembly,
        anchor_simulation,
        anchor_state,
    ) = _resume_checkpoint_state(
        saved_row=anchor_saved_row,
        global_row=global_row,
        paired_row=paired_rows[anchor_position],
        observed_waveform_data=observed_waveform_data,
        waveform_metadata=waveform_metadata,
        stf_handler=stf_handler,
        params=resume_params,
        base_assembly=assembly_dict,
    )

    # If the interruption happened after the global search but before the
    # anchor's local refinement, finish and checkpoint that record first.
    if anchor_position not in local_row_by_position:
        anchor_rec_n = int(anchor_assembly["rec_n"])
        result_local = local_refine_waveform(
            outfile_name=outfile_name + f"_rec_n_{anchor_rec_n}",
            params=resume_params,
            assembly_dict=anchor_assembly,
            simulation=anchor_simulation,
            stf_handler=stf_handler,
        )
        anchor_simulation = result_local.get("simulation_handler")
        if anchor_simulation is None:
            raise RuntimeError("resume could not refine the anchor waveform")
        anchor_local_row = _build_local_result_row(
            rec_n=anchor_rec_n,
            assembly_dict=anchor_assembly,
            simulation=anchor_simulation,
            result_local=result_local,
        )
        anchor_local_row["local_anchor_stride"] = resume_stride
        anchor_local_row["local_records_per_worker"] = int(
            resume_params.get("local_records_per_worker", 3)
        )
        with open(results_pkl, "ab") as stream:
            pickle.dump(
                anchor_local_row,
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
        local_row_by_position[anchor_position] = anchor_local_row
        anchor_state = deepcopy(
            anchor_local_row["local_continuation_state"]
        )

    all_lanes = _build_local_inversion_lanes(
        paired_rows,
        list(range(anchor_position + 1, len(paired_rows))),
        resume_stride,
    )
    lane_specs = []
    state_cache = {anchor_position: anchor_state}
    completed_positions = set(local_row_by_position)
    expected_positions = {anchor_position}

    for lane_id, lane in enumerate(all_lanes):
        expected_positions.update(lane)
        prefix_length = 0
        while (
            prefix_length < len(lane)
            and lane[prefix_length] in completed_positions
        ):
            prefix_length += 1
        completed_after_gap = [
            position
            for position in lane[prefix_length:]
            if position in completed_positions
        ]
        if completed_after_gap:
            raise RuntimeError(
                "pickle has a gap inside continuation lane "
                f"{lane_id + 1}: completed positions after the gap are "
                f"{completed_after_gap[:10]}"
            )
        if prefix_length == len(lane):
            continue

        checkpoint_position = (
            lane[prefix_length - 1]
            if prefix_length > 0
            else anchor_position
        )
        if checkpoint_position not in state_cache:
            checkpoint_row = local_row_by_position[checkpoint_position]
            _, _, checkpoint_state = _resume_checkpoint_state(
                saved_row=checkpoint_row,
                global_row=global_row,
                paired_row=paired_rows[checkpoint_position],
                observed_waveform_data=observed_waveform_data,
                waveform_metadata=waveform_metadata,
                stf_handler=stf_handler,
                params=resume_params,
                base_assembly=assembly_dict,
            )
            state_cache[checkpoint_position] = checkpoint_state

        lane_specs.append(
            {
                "lane_id": lane_id,
                "positions": lane[prefix_length:],
                "continuation_state": state_cache[checkpoint_position],
            }
        )

    unexpected = completed_positions - expected_positions
    if unexpected:
        raise RuntimeError(
            "pickle contains local rows outside the reconstructed lanes: "
            f"{sorted(unexpected)[:10]}"
        )

    completed_local = len(local_row_by_position)
    total_local = len(expected_positions)
    missing_local = total_local - completed_local
    print(
        "resume: "
        f"{completed_local}/{total_local} local inversions already saved; "
        f"continuing {missing_local} missing records in "
        f"{len(lane_specs)} active lanes",
        flush=True,
    )
    if not lane_specs:
        return

    parallel_rows = _run_parallel_local_inversions(
        paired_rows=paired_rows,
        remaining_positions=[],
        observed_waveform_data=observed_waveform_data,
        outfile_name=outfile_name,
        params=resume_params,
        assembly_seed=anchor_assembly,
        simulation_seed=anchor_simulation,
        stf_handler=stf_handler,
        lane_specs=lane_specs,
    )
    with open(results_pkl, "ab") as stream:
        for _, resumed_row in parallel_rows:
            pickle.dump(
                resumed_row,
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()


def _write_local_summary_plots(
    pkl_path: Path,
    params: Dict[str, Any],
) -> None:
    if not params.get("run_local_inversion", True):
        print("Local inversion disabled: skipped local profile-evolution plots.")
        return

    out_g1 = pkl_path.with_name(
        pkl_path.stem + "_gouge1_profiles_vs_time_with_shear.png"
    )
    plot_gouge_layer_colormap_with_shear(pkl_path, out_g1, layer=1)
    print(f"Saved: {out_g1}")

    out_g2 = pkl_path.with_name(
        pkl_path.stem + "_gouge2_profiles_vs_time_with_shear.png"
    )
    plot_gouge_layer_colormap_with_shear(pkl_path, out_g2, layer=2)
    print(f"Saved: {out_g2}")

    out1 = pkl_path.with_name(
        pkl_path.stem + "_shear_and_vavg_vs_time.png"
    )
    plot_shear_and_avg_velocity_from_pickle(pkl_path, out1)
    print(f"Saved: {out1}")

    out_vel = pkl_path.with_name(
        pkl_path.stem + "_velocity_profiles_vs_time.png"
    )
    out_dmp = pkl_path.with_name(
        pkl_path.stem + "_damping_profiles_vs_time.png"
    )
    plot_profile_evolution_from_pickle(pkl_path, out_vel, out_dmp)
    print(f"Saved: {out_vel}")
    print(f"Saved: {out_dmp}")

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
    paired_rows = list(paired.itertuples(index=False))


    # plt.plot(paired["uw_time_s"], paired['normal_stress_MPa'])
    # plt.show()
    # sys.exit()

    results_pkl = outfile_path.with_suffix(".pkl")
    results_summary = []      # list of dicts for pandas
    results_by_rec_n = {}     # rec_n -> dict

    if (
        params.get("resume_from_pickle", True)
        and params.get("run_local_inversion", True)
        and results_pkl.exists()
        and results_pkl.stat().st_size > 0
    ):
        existing_rows = load_pickle_stream(
            results_pkl,
            repair_incomplete_tail=True,
        )
        if existing_rows:
            print(
                f"resume: found {len(existing_rows)} complete rows in "
                f"{results_pkl}",
                flush=True,
            )
            _resume_local_inversions_from_pickle(
                existing_rows=existing_rows,
                results_pkl=results_pkl,
                paired_rows=paired_rows,
                observed_waveform_data=observed_waveform_data,
                waveform_metadata=waveform_metadata,
                outfile_name=outfile_name,
                params=params,
                assembly_dict=assembly_dict,
                stf_handler=stf_handler,
            )
            _write_local_summary_plots(results_pkl, params)
            print(
                f"--- {tm.time() - start_time:.2f} seconds for resuming "
                f"{infile_path.name} ---"
            )
            return

    with open(results_pkl, "ab") as f_results:

        for row_position, row in enumerate(paired_rows):
            
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

                # The global stage is allowed to profile one unknown scalar
                # amplitude.  Its final value calibrates the otherwise
                # arbitrary STF/source/receiver units once for the complete
                # experiment; local inversions must not refit it per record.
                sim_g = result_global["simulation_handler"]
                reference_trace_scale = float(sim_g.trace_scale)
                if not np.isfinite(reference_trace_scale) or reference_trace_scale <= 0.0:
                    raise RuntimeError(
                        "global amplitude calibration did not produce a positive "
                        "finite trace scale"
                    )
                sim_g.reference_trace_scale = reference_trace_scale
                result_global["reference_trace_scale"] = reference_trace_scale

                # Update state
                last_good_result = result_global

                # Save EXACT global-search outputs + best misfit from the final forward run
                contrast = result_global.get("velocity_contrast_result")
                thickness = result_global.get("gouge_thickness_result")
                joint = result_global.get("joint_kinematic_result")
                density = result_global.get("density_result")
                impedance = result_global.get("impedance_material_result")
                post_impedance_thickness = result_global.get(
                    "post_impedance_thickness_result"
                )
                spectral = result_global.get("spectral_diagnostics")
                global_row = {
                    # keys for recall
                    "rec_n": rec_n,
                    "idx_wave": int(assembly_dict["idx_wave"]),
                    "acquisition_time_s": float(assembly_dict["acquisition_time"]),
                    # mechanical state
                    "normal_stress_MPa": float(assembly_dict["normal_stress"]),
                    "shear_stress_MPa": float(assembly_dict["shear_stress"]),
                    "gouge_thickness_1_cm": float(
                        result_global["reference_gouge_thickness_1_cm"]
                    ),
                    "gouge_thickness_2_cm": float(
                        result_global["reference_gouge_thickness_2_cm"]
                    ),
                    "best_gouge_thickness_1_cm": float(
                        result_global["best_gouge_thickness_1_cm"]
                    ),
                    "best_gouge_thickness_2_cm": float(
                        result_global["best_gouge_thickness_2_cm"]
                    ),

                    # GLOBAL SEARCH arrays (as requested)
                    "x_axis_cm": sim_g.grid_handler.spatial_axis.copy(),
                    "gouge_velocity_list": result_global["gouge_velocity_list"],
                    "best_gouge_velocity": float(result_global["best_gouge_velocity"]),
                    "best_homogeneous_gouge_velocity": float(
                        result_global["best_homogeneous_gouge_velocity"]
                    ),
                    "best_gouge_velocity_1": float(
                        result_global["best_gouge_velocity_1"]
                    ),
                    "best_gouge_velocity_2": float(
                        result_global["best_gouge_velocity_2"]
                    ),
                    "gouge_damping_list": result_global["gouge_damping_list"],
                    "best_gouge_damping": float(result_global["best_gouge_damping"]),
                    "L2norm_waveform_list": result_global["L2norm_waveform_list"],

                    # CONSTANT-TIME c1/c2 CONTRAST SEARCH (None when disabled)
                    "velocity_contrast_log_list": (
                        None if contrast is None else contrast["log_contrast_list"]
                    ),
                    "velocity_contrast_ratio_list": (
                        None if contrast is None else contrast["velocity_ratio_list"]
                    ),
                    "velocity_contrast_c1_list": (
                        None if contrast is None else contrast["gouge_velocity_1_list"]
                    ),
                    "velocity_contrast_c2_list": (
                        None if contrast is None else contrast["gouge_velocity_2_list"]
                    ),
                    "velocity_contrast_misfit_list": (
                        None if contrast is None else contrast["misfit_list"]
                    ),
                    "velocity_contrast_stack_time_list_us": (
                        None
                        if contrast is None
                        else contrast["stack_travel_time_list_us"]
                    ),
                    "velocity_contrast_target_stack_time_us": (
                        None
                        if contrast is None
                        else contrast["target_stack_travel_time_us"]
                    ),
                    "best_velocity_log_contrast": (
                        None if contrast is None else contrast["best_log_contrast"]
                    ),

                    # CONSTANT-TOTAL-THICKNESS h1/h2 SEARCH (None when disabled)
                    "gouge_thickness_offset_mm_list": (
                        None
                        if thickness is None
                        else thickness["thickness_offset_mm_list"]
                    ),
                    "gouge_thickness_1_cm_list": (
                        None
                        if thickness is None
                        else thickness["gouge_thickness_1_cm_list"]
                    ),
                    "gouge_thickness_2_cm_list": (
                        None
                        if thickness is None
                        else thickness["gouge_thickness_2_cm_list"]
                    ),
                    "gouge_thickness_misfit_list": (
                        None if thickness is None else thickness["misfit_list"]
                    ),
                    "gouge_thickness_travel_time_us_list": (
                        None
                        if thickness is None
                        else thickness["gouge_travel_time_us_list"]
                    ),
                    "gouge_thickness_grid_dx_mm_list": (
                        None if thickness is None else thickness["grid_dx_mm_list"]
                    ),
                    "gouge_1_num_cells_list": (
                        None
                        if thickness is None
                        else thickness["gouge_1_num_cells_list"]
                    ),
                    "gouge_2_num_cells_list": (
                        None
                        if thickness is None
                        else thickness["gouge_2_num_cells_list"]
                    ),
                    "best_gouge_thickness_offset_mm": (
                        None
                        if thickness is None
                        else thickness["best_thickness_offset_mm"]
                    ),

                    # JOINT (c1/c2, h1/h2) SEARCH AT HOMOGENEOUS TRAVEL TIME
                    "joint_velocity_log_contrast_axis": (
                        None if joint is None else joint["log_contrast_axis"]
                    ),
                    "joint_thickness_offset_mm_axis": (
                        None
                        if joint is None
                        else joint["thickness_offset_mm_axis"]
                    ),
                    "joint_kinematic_misfit_grid": (
                        None if joint is None else joint["misfit_grid"]
                    ),
                    "joint_kinematic_c1_list": (
                        None if joint is None else joint["gouge_velocity_1_list"]
                    ),
                    "joint_kinematic_c2_list": (
                        None if joint is None else joint["gouge_velocity_2_list"]
                    ),
                    "joint_kinematic_h1_cm_list": (
                        None
                        if joint is None
                        else joint["gouge_thickness_1_cm_list"]
                    ),
                    "joint_kinematic_h2_cm_list": (
                        None
                        if joint is None
                        else joint["gouge_thickness_2_cm_list"]
                    ),
                    "joint_kinematic_stack_time_us_list": (
                        None
                        if joint is None
                        else joint["stack_travel_time_list_us"]
                    ),
                    "joint_kinematic_target_stack_time_us": (
                        None
                        if joint is None
                        else joint["target_stack_travel_time_us"]
                    ),
                    "joint_kinematic_best_log_contrast": (
                        None if joint is None else joint["best_log_contrast"]
                    ),
                    "joint_kinematic_best_thickness_offset_mm": (
                        None
                        if joint is None
                        else joint["best_thickness_offset_mm"]
                    ),
                    "joint_kinematic_best_misfit": (
                        None if joint is None else joint["best_misfit"]
                    ),

                    # COMMON GOUGE DENSITY SEARCH (None when disabled)
                    "gouge_density_list_g_cm3": (
                        None
                        if density is None
                        else density["gouge_density_list_g_cm3"]
                    ),
                    "density_misfit_list": (
                        None if density is None else density["misfit_list"]
                    ),
                    "density_nominal_echo_factor_list": (
                        None
                        if density is None
                        else density["nominal_echo_factor_list"]
                    ),
                    "best_gouge_density_g_cm3": (
                        None
                        if density is None
                        else density["best_gouge_density_g_cm3"]
                    ),
                    "density_best_misfit": (
                        None if density is None else density["best_misfit"]
                    ),

                    # JOINT c/damping/density IMPEDANCE SEARCH
                    "impedance_coarse_velocity_list": (
                        None if impedance is None
                        else impedance["coarse"]["gouge_velocity_list"]
                    ),
                    "impedance_coarse_damping_list": (
                        None if impedance is None
                        else impedance["coarse"]["gouge_damping_list"]
                    ),
                    "impedance_coarse_density_list_g_cm3": (
                        None if impedance is None
                        else impedance["coarse"]["gouge_density_list_g_cm3"]
                    ),
                    "impedance_coarse_shape_misfit_list": (
                        None if impedance is None
                        else impedance["coarse"]["waveform_shape_misfit_list"]
                    ),
                    "impedance_coarse_spectral_distance_list": (
                        None if impedance is None
                        else impedance["coarse"]["spectral_shape_distance_list"]
                    ),
                    "impedance_fine_velocity_list": (
                        None if impedance is None
                        else impedance["fine"]["gouge_velocity_list"]
                    ),
                    "impedance_fine_damping_list": (
                        None if impedance is None
                        else impedance["fine"]["gouge_damping_list"]
                    ),
                    "impedance_fine_density_list_g_cm3": (
                        None if impedance is None
                        else impedance["fine"]["gouge_density_list_g_cm3"]
                    ),
                    "impedance_fine_shape_misfit_list": (
                        None if impedance is None
                        else impedance["fine"]["waveform_shape_misfit_list"]
                    ),
                    "impedance_fine_spectral_distance_list": (
                        None if impedance is None
                        else impedance["fine"]["spectral_shape_distance_list"]
                    ),
                    "impedance_best_gouge_velocity": (
                        None if impedance is None else impedance["best_gouge_velocity"]
                    ),
                    "impedance_best_gouge_damping": (
                        None if impedance is None else impedance["best_gouge_damping"]
                    ),
                    "impedance_best_gouge_density_g_cm3": (
                        None if impedance is None
                        else impedance["best_gouge_density_g_cm3"]
                    ),
                    "impedance_best_shape_misfit": (
                        None if impedance is None
                        else impedance["best_waveform_shape_misfit"]
                    ),
                    "impedance_best_spectral_distance": (
                        None if impedance is None
                        else impedance["best_spectral_shape_distance"]
                    ),

                    # FIXED-TOTAL-THICKNESS SEARCH AFTER IMPEDANCE MATERIAL FIT
                    "post_impedance_thickness_offset_mm_list": (
                        None if post_impedance_thickness is None
                        else post_impedance_thickness["thickness_offset_mm_list"]
                    ),
                    "post_impedance_thickness_h1_cm_list": (
                        None if post_impedance_thickness is None
                        else post_impedance_thickness["gouge_thickness_1_cm_list"]
                    ),
                    "post_impedance_thickness_h2_cm_list": (
                        None if post_impedance_thickness is None
                        else post_impedance_thickness["gouge_thickness_2_cm_list"]
                    ),
                    "post_impedance_thickness_shape_misfit_list": (
                        None if post_impedance_thickness is None
                        else post_impedance_thickness["misfit_list"]
                    ),
                    "post_impedance_thickness_spectral_distance_list": (
                        None if post_impedance_thickness is None
                        else post_impedance_thickness[
                            "spectral_shape_distance_list"
                        ]
                    ),
                    "post_impedance_thickness_equivalent_best_offsets_mm": (
                        None if post_impedance_thickness is None
                        else post_impedance_thickness[
                            "equivalent_best_thickness_offsets_mm"
                        ]
                    ),
                    "post_impedance_best_thickness_offset_mm": (
                        None if post_impedance_thickness is None
                        else post_impedance_thickness["best_thickness_offset_mm"]
                    ),
                    "post_impedance_thickness_best_shape_misfit": (
                        None if post_impedance_thickness is None
                        else post_impedance_thickness["best_misfit"]
                    ),
                    "post_impedance_thickness_best_spectral_distance": (
                        None if post_impedance_thickness is None
                        else post_impedance_thickness[
                            "best_spectral_shape_distance"
                        ]
                    ),
                    "spectral_shape_distance": (
                        None
                        if spectral is None
                        else spectral["recorded_vs_simulated_distance"]
                    ),
                    "stf_recorded_spectral_shape_distance": (
                        None
                        if spectral is None
                        else spectral["stf_vs_recorded_distance"]
                    ),
                    "spectral_frequency_min_mhz": (
                        None if spectral is None else spectral["frequency_min_mhz"]
                    ),
                    "spectral_frequency_max_mhz": (
                        None if spectral is None else spectral["frequency_max_mhz"]
                    ),

                    # Best misfit from the FINAL forward run of global search
                    "global_best_misfit": float(sim_g.misfit),
                    "homogeneous_best_misfit": float(
                        result_global["homogeneous_best_misfit"]
                    ),
                    "reference_trace_scale": reference_trace_scale,

                    # Profiles at the global-search best model (useful baseline)
                    "velocity_profile": sim_g.velocity_model_handler.velocity_array.copy(),
                    "damping_profile": sim_g.velocity_model_handler.damping_array.copy() if hasattr(sim_g.velocity_model_handler, "damping_array") else None,
                    "density_profile": sim_g.velocity_model_handler.density_array.copy() if hasattr(sim_g.velocity_model_handler, "density_array") else None,
                }
                global_row["global_continuation_state"] = (
                    _capture_local_continuation_state(assembly_dict, sim_g)
                )
                global_row["local_anchor_stride"] = int(
                    params.get("local_anchor_stride", 10)
                )
                global_row["local_records_per_worker"] = int(
                    params.get("local_records_per_worker", 3)
                )

                results_summary.append(global_row)
                results_by_rec_n[rec_n] = global_row
                pickle.dump(global_row, f_results, protocol=pickle.HIGHEST_PROTOCOL)
                f_results.flush()

            # -----------------------------
            # 2) LOCAL refine on subsequent waveforms
            # -----------------------------
            if not params.get("run_local_inversion", True):
                continue

            last_good_result["simulation_handler"].observed_waveform = observed_waveform

            # IMPORTANT FIX: pass the prior simulation handler, not "result[...]"
            result_local = local_refine_waveform(
                outfile_name=outfile_name_wave,
                params=params,
                assembly_dict=assembly_dict,
                simulation=last_good_result["simulation_handler"],
                stf_handler=stf_handler,
            )

            if not result_local.get("simulation_handler", None):
                continue

            last_good_result = result_local
            sim = last_good_result["simulation_handler"]
            local_row = _build_local_result_row(
                rec_n=rec_n,
                assembly_dict=assembly_dict,
                simulation=sim,
                result_local=result_local,
            )
            local_row["local_anchor_stride"] = int(
                params.get("local_anchor_stride", 10)
            )
            local_row["local_records_per_worker"] = int(
                params.get("local_records_per_worker", 3)
            )

            results_summary.append(local_row)
            results_by_rec_n[rec_n] = local_row
            pickle.dump(local_row, f_results, protocol=pickle.HIGHEST_PROTOCOL)
            f_results.flush()

            # With one worker the loop above remains bit-for-bit the original
            # chronological continuation.  With more workers, the first
            # accepted local model seeds interlaced chronological chains.
            remaining_positions = list(
                range(row_position + 1, len(paired_rows))
            )
            if "local_inversion_workers" in params:
                local_worker_count = resolve_worker_count(
                    params,
                    "local_inversion_workers",
                    max(1, len(remaining_positions)),
                )
            else:
                # Backward-compatible default: the former code was serial.
                local_worker_count = 1
            if local_worker_count > 1 and remaining_positions:
                parallel_rows = _run_parallel_local_inversions(
                    paired_rows=paired_rows,
                    remaining_positions=remaining_positions,
                    observed_waveform_data=observed_waveform_data,
                    outfile_name=outfile_name,
                    params=params,
                    assembly_seed=assembly_dict,
                    simulation_seed=sim,
                    stf_handler=stf_handler,
                )
                for _, parallel_row in parallel_rows:
                    parallel_rec_n = int(parallel_row["rec_n"])
                    results_summary.append(parallel_row)
                    results_by_rec_n[parallel_rec_n] = parallel_row
                    pickle.dump(
                        parallel_row,
                        f_results,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                    f_results.flush()
                break

    _write_local_summary_plots(results_pkl, params)
    print(f"--- {tm.time() - start_time:.2f} seconds for processing {infile_path.name} ---")


def _simulation_velocity_limits(
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
) -> tuple[float, float]:
    """Return grid-design limits that always contain the actual model."""

    model_minimum = min_assembly_velocity(assembly_dict)
    model_maximum = max_assembly_velocity(assembly_dict)
    configured_minimum = params.get("min_velocity2simulate")
    configured_maximum = params.get("max_velocity2simulate")

    minimum = model_minimum
    if configured_minimum is not None and float(configured_minimum) > 0.0:
        minimum = min(float(configured_minimum), model_minimum)

    maximum = model_maximum
    if configured_maximum is not None and float(configured_maximum) > 0.0:
        maximum = max(float(configured_maximum), model_maximum)
    return minimum, maximum


def _legacy_simulation_velocity_limits(
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
) -> tuple[float, float]:
    """Reproduce the grid limits used before the optional second stages."""

    configured_minimum = params.get("min_velocity2simulate")
    configured_maximum = params.get("max_velocity2simulate")
    minimum = (
        float(configured_minimum)
        if configured_minimum
        else min_assembly_velocity(assembly_dict)
    )
    maximum = (
        float(configured_maximum)
        if configured_maximum
        else max_assembly_velocity(assembly_dict)
    )
    return minimum, maximum


def velocity_contrast_search_run(args):
    """Evaluate one constant-travel-time ``c1/c2`` contrast candidate."""

    (
        candidate,
        observed_waveform,
        observed_time,
        stf_handler,
        misfit_interval,
        params,
        assembly_dict,
        fixed_damping,
        plot_output_path,
        enable_plotting,
    ) = args

    assembly_candidate = set_assembly_dict_guessed_with_gouge_velocity_and_damping(
        assembly_dict,
        (
            candidate["gouge_velocity_1"],
            candidate["gouge_velocity_2"],
        ),
        (fixed_damping, fixed_damping),
    )
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(
        params,
        assembly_candidate,
    )

    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_candidate,
        misfit_interval=misfit_interval,
        normalization_interval=misfit_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=fixed_damping,
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=params.get(
            "velocity_contrast_points_per_wavelength",
            params.get("points_per_wavelength", 4),
        ),
        store_wavefield=False,
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=enable_plotting,
        make_movie=False,
        plot_output_path=plot_output_path,
    )

    result = dict(candidate)
    result["misfit"] = float(simulation.misfit)
    return result


def _plot_velocity_contrast_search(
    results: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Plot the contrast misfit and the corresponding constant-time pair."""

    ordered = sorted(results, key=lambda item: item["log_contrast"])
    delta = np.asarray([item["log_contrast"] for item in ordered])
    misfit = np.asarray([item["misfit"] for item in ordered])
    c1 = 1.0e4 * np.asarray([item["gouge_velocity_1"] for item in ordered])
    c2 = 1.0e4 * np.asarray([item["gouge_velocity_2"] for item in ordered])
    best = int(np.argmin(misfit))

    fig, (ax_misfit, ax_velocity) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        constrained_layout=True,
    )
    ax_misfit.plot(delta, misfit, marker="o")
    ax_misfit.scatter(delta[best], misfit[best], marker="*", s=140, zorder=3)
    ax_misfit.set_ylabel("Misfit (L2 norm)")
    ax_misfit.grid(alpha=0.25)

    ax_velocity.plot(delta, c1, marker="o", label="gouge 1")
    ax_velocity.plot(delta, c2, marker="s", label="gouge 2")
    ax_velocity.set_xlabel(r"Log-contrast $\delta$")
    ax_velocity.set_ylabel("Velocity [m/s]")
    ax_velocity.grid(alpha=0.25)
    ax_velocity.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def search_constant_time_velocity_contrast(
    *,
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    stf_handler: UltrasonicDataHandler,
    misfit_interval: np.ndarray,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    homogeneous_velocity: float,
    fixed_damping: float,
    plot_output_path: Path,
    save_plot: bool,
) -> Dict[str, Any]:
    """Search ``c1/c2`` after the homogeneous velocity/damping search.

    Only the velocity contrast changes.  A common scale is solved for every
    candidate so that the one-way travel time through the two gouges and the
    four groove ramps remains equal to the homogeneous reference value.
    """

    requested = np.asarray(
        params.get("velocity_contrast_log_values", [0.0]),
        dtype=float,
    ).reshape(-1)
    if requested.size == 0 or np.any(~np.isfinite(requested)):
        raise ValueError("velocity_contrast_log_values must be finite and non-empty")
    log_contrasts = np.unique(np.append(requested, 0.0))
    candidates = build_constant_time_contrast_candidates(
        assembly_dict,
        homogeneous_velocity,
        log_contrasts,
        wave_type=assembly_dict["wave_type"],
    )

    contrast_output_path = plot_output_path / "velocity_contrast_search"
    contrast_output_path.mkdir(parents=True, exist_ok=True)
    plot_all = bool(params.get("velocity_contrast_plot_all", True)) and save_plot

    args_list = [
        (
            candidate,
            observed_waveform,
            observed_time,
            stf_handler,
            misfit_interval,
            params,
            assembly_dict,
            fixed_damping,
            contrast_output_path,
            plot_all,
        )
        for candidate in candidates
    ]
    num_processes = resolve_worker_count(
        params, "global_search_workers", len(args_list)
    )

    if num_processes == 1:
        results = [velocity_contrast_search_run(args) for args in args_list]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(velocity_contrast_search_run, args_list)
    results = sorted(results, key=lambda item: item["log_contrast"])
    best = min(results, key=lambda item: item["misfit"])

    _plot_velocity_contrast_search(
        results,
        contrast_output_path / "velocity_contrast_misfit.png",
    )

    assembly_dict["gouge_velocity_1"] = best["gouge_velocity_1"]
    assembly_dict["gouge_velocity_2"] = best["gouge_velocity_2"]
    assembly_dict["gouge_damping_1"] = fixed_damping
    assembly_dict["gouge_damping_2"] = fixed_damping
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(
        params,
        assembly_dict,
    )

    best_output_path = contrast_output_path / "best"
    best_output_path.mkdir(parents=True, exist_ok=True)
    best_simulation = UltrasonicModeler()
    best_simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_dict,
        misfit_interval=misfit_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=fixed_damping,
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=params.get(
            "velocity_contrast_points_per_wavelength",
            params["points_per_wavelength"],
        ),
        store_wavefield=bool(params.get("run_local_inversion", True)),
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=save_plot,
        make_movie=False,
        plot_output_path=best_output_path,
    )

    print(
        "constant-time contrast search: "
        f"delta={best['log_contrast']:+.4f}, "
        f"c1={1.0e4 * best['gouge_velocity_1']:.0f} m/s, "
        f"c2={1.0e4 * best['gouge_velocity_2']:.0f} m/s, "
        f"misfit={best_simulation.misfit:.3f}"
    )

    return {
        "log_contrast_list": np.asarray(
            [item["log_contrast"] for item in results], dtype=float
        ),
        "velocity_ratio_list": np.asarray(
            [item["velocity_ratio"] for item in results], dtype=float
        ),
        "gouge_velocity_1_list": np.asarray(
            [item["gouge_velocity_1"] for item in results], dtype=float
        ),
        "gouge_velocity_2_list": np.asarray(
            [item["gouge_velocity_2"] for item in results], dtype=float
        ),
        "stack_travel_time_list_us": np.asarray(
            [item["stack_travel_time_us"] for item in results], dtype=float
        ),
        "misfit_list": np.asarray([item["misfit"] for item in results], dtype=float),
        "target_stack_travel_time_us": float(
            best["target_stack_travel_time_us"]
        ),
        "best_log_contrast": float(best["log_contrast"]),
        "best_gouge_velocity_1": float(best["gouge_velocity_1"]),
        "best_gouge_velocity_2": float(best["gouge_velocity_2"]),
        "best_misfit": float(best_simulation.misfit),
        "simulation_handler": best_simulation,
    }


def gouge_thickness_search_run(args):
    """Evaluate one fixed-total-thickness ``h1/h2`` candidate."""

    (
        candidate,
        observed_waveform,
        observed_time,
        stf_handler,
        misfit_interval,
        params,
        assembly_dict,
        fixed_velocity,
        fixed_damping,
        plot_output_path,
        enable_plotting,
    ) = args

    assembly_candidate = set_assembly_dict_guessed_with_gouge_velocity_and_damping(
        assembly_dict,
        (fixed_velocity, fixed_velocity),
        (fixed_damping, fixed_damping),
    )
    assembly_candidate = set_assembly_dict_guessed_with_gouge_thicknesses(
        assembly_candidate,
        candidate["gouge_thickness_1_cm"],
        candidate["gouge_thickness_2_cm"],
    )
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(
        params,
        assembly_candidate,
    )

    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_candidate,
        misfit_interval=misfit_interval,
        normalization_interval=misfit_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=fixed_damping,
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=params.get(
            "gouge_thickness_points_per_wavelength",
            params.get("points_per_wavelength", 4),
        ),
        store_wavefield=False,
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=enable_plotting,
        make_movie=False,
        plot_output_path=plot_output_path,
    )

    result = dict(candidate)
    result["misfit"] = float(simulation.misfit)
    spectral = simulation.write_direct_arrival_spectral_diagnostics(
        output_directory=None,
        frequency_min_mhz=float(params.get("frequency_min", 0.0)),
        frequency_max_mhz=float(params["frequency_cutoff"]),
    )
    result["spectral_shape_distance"] = float(
        spectral["recorded_vs_simulated_distance"]
    )
    result["grid_dx_mm"] = 10.0 * float(simulation.grid_handler.dx)
    result["gouge_1_num_cells"] = int(
        simulation.velocity_model_handler.idx_dict["gouge_1"].size
    )
    result["gouge_2_num_cells"] = int(
        simulation.velocity_model_handler.idx_dict["gouge_2"].size
    )
    return result


def _plot_gouge_thickness_search(
    results: list[dict[str, float]],
    output_path: Path,
) -> None:
    """Plot misfit and the two layer widths at fixed total thickness."""

    ordered = sorted(results, key=lambda item: item["thickness_offset_mm"])
    offsets = np.asarray([item["thickness_offset_mm"] for item in ordered])
    misfit = np.asarray([item["misfit"] for item in ordered])
    h1_mm = 10.0 * np.asarray(
        [item["gouge_thickness_1_cm"] for item in ordered]
    )
    h2_mm = 10.0 * np.asarray(
        [item["gouge_thickness_2_cm"] for item in ordered]
    )
    best = int(np.argmin(misfit))

    spectral_distance = np.asarray(
        [item["spectral_shape_distance"] for item in ordered]
    )
    best_spectral = int(np.argmin(spectral_distance))
    fig, (ax_misfit, ax_spectral, ax_thickness) = plt.subplots(
        3,
        1,
        figsize=(8, 10),
        sharex=True,
        constrained_layout=True,
    )
    ax_misfit.plot(offsets, misfit, marker="o")
    ax_misfit.scatter(offsets[best], misfit[best], marker="*", s=140, zorder=3)
    ax_misfit.set_ylabel("Misfit (L2 norm)")
    ax_misfit.grid(alpha=0.25)

    ax_spectral.plot(offsets, spectral_distance, marker="o", color="tab:orange")
    ax_spectral.scatter(
        offsets[best_spectral], spectral_distance[best_spectral], marker="*",
        s=140, color="tab:orange", zorder=3, label=r"minimum $D_f$"
    )
    ax_spectral.scatter(
        offsets[best], spectral_distance[best], marker="x", s=80,
        color="black", zorder=4, label=r"model selected by $J_{shape}$"
    )
    ax_spectral.set_ylabel(r"Spectral distance $D_f$")
    ax_spectral.grid(alpha=0.25)
    ax_spectral.legend()

    ax_thickness.plot(offsets, h1_mm, marker="o", label="gouge 1")
    ax_thickness.plot(offsets, h2_mm, marker="s", label="gouge 2")
    ax_thickness.set_xlabel(r"Thickness transfer $\Delta h$ [mm]")
    ax_thickness.set_ylabel("Gouge thickness [mm]")
    ax_thickness.grid(alpha=0.25)
    ax_thickness.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def search_constant_total_gouge_thickness(
    *,
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    stf_handler: UltrasonicDataHandler,
    misfit_interval: np.ndarray,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    fixed_velocity: float,
    fixed_damping: float,
    plot_output_path: Path,
    save_plot: bool,
    output_subdirectory: str = "gouge_thickness_search",
) -> Dict[str, Any]:
    """Redistribute the two gouge widths while keeping their sum fixed."""

    requested = np.asarray(
        params.get("gouge_thickness_offset_mm_values", [0.0]),
        dtype=float,
    ).reshape(-1)
    if requested.size == 0 or np.any(~np.isfinite(requested)):
        raise ValueError(
            "gouge_thickness_offset_mm_values must be finite and non-empty"
        )
    offsets_mm = np.unique(np.append(requested, 0.0))
    candidates = build_constant_total_thickness_candidates(
        assembly_dict,
        offsets_mm,
        common_velocity_cm_us=fixed_velocity,
    )

    thickness_output_path = plot_output_path / output_subdirectory
    thickness_output_path.mkdir(parents=True, exist_ok=True)
    plot_all = bool(params.get("gouge_thickness_plot_all", True)) and save_plot

    args_list = [
        (
            candidate,
            observed_waveform,
            observed_time,
            stf_handler,
            misfit_interval,
            params,
            assembly_dict,
            fixed_velocity,
            fixed_damping,
            thickness_output_path,
            plot_all,
        )
        for candidate in candidates
    ]
    num_processes = resolve_worker_count(
        params, "global_search_workers", len(args_list)
    )

    if num_processes == 1:
        results = [gouge_thickness_search_run(args) for args in args_list]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(gouge_thickness_search_run, args_list)
    results = sorted(results, key=lambda item: item["thickness_offset_mm"])
    best, equivalent_offsets = select_thickness_minimum(
        results,
        relative_tolerance=float(
            params.get("gouge_thickness_tie_relative_tolerance", 1.0e-6)
        ),
    )

    _plot_gouge_thickness_search(
        results,
        thickness_output_path / "gouge_thickness_misfit.png",
    )

    assembly_best = set_assembly_dict_guessed_with_gouge_thicknesses(
        assembly_dict,
        best["gouge_thickness_1_cm"],
        best["gouge_thickness_2_cm"],
    )
    assembly_dict.update(assembly_best)
    assembly_dict["gouge_velocity_1"] = fixed_velocity
    assembly_dict["gouge_velocity_2"] = fixed_velocity
    assembly_dict["gouge_damping_1"] = fixed_damping
    assembly_dict["gouge_damping_2"] = fixed_damping
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(
        params,
        assembly_dict,
    )

    best_output_path = thickness_output_path / "best"
    best_output_path.mkdir(parents=True, exist_ok=True)
    best_simulation = UltrasonicModeler()
    best_simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_dict,
        misfit_interval=misfit_interval,
        normalization_interval=misfit_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=fixed_damping,
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=params.get(
            "gouge_thickness_points_per_wavelength",
            params.get("points_per_wavelength", 4),
        ),
        # The local inversion recomputes the complete forward field.  Keeping
        # it here would only duplicate a potentially large wavefield.
        store_wavefield=False,
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=save_plot,
        make_movie=False,
        plot_output_path=best_output_path,
    )
    best_spectral = best_simulation.write_direct_arrival_spectral_diagnostics(
        output_directory=None,
        frequency_min_mhz=float(params.get("frequency_min", 0.0)),
        frequency_max_mhz=float(params["frequency_cutoff"]),
    )

    print(
        "constant-total-thickness search: "
        f"delta_h={best['thickness_offset_mm']:+.3f} mm, "
        f"h1={10.0 * best['gouge_thickness_1_cm']:.3f} mm, "
        f"h2={10.0 * best['gouge_thickness_2_cm']:.3f} mm, "
        f"dx={best['grid_dx_mm']:.4f} mm, "
        f"cells={best['gouge_1_num_cells']}+{best['gouge_2_num_cells']}, "
        f"misfit={best_simulation.misfit:.3f}, "
        f"Df={best_spectral['recorded_vs_simulated_distance']:.4f}"
    )
    if equivalent_offsets.size > 1:
        print(
            "reciprocal thickness minima within tolerance: "
            + ", ".join(f"{value:+.3f} mm" for value in equivalent_offsets)
        )

    return {
        "thickness_offset_mm_list": np.asarray(
            [item["thickness_offset_mm"] for item in results], dtype=float
        ),
        "gouge_thickness_1_cm_list": np.asarray(
            [item["gouge_thickness_1_cm"] for item in results], dtype=float
        ),
        "gouge_thickness_2_cm_list": np.asarray(
            [item["gouge_thickness_2_cm"] for item in results], dtype=float
        ),
        "total_gouge_thickness_cm_list": np.asarray(
            [item["total_gouge_thickness_cm"] for item in results], dtype=float
        ),
        "gouge_travel_time_us_list": np.asarray(
            [item["gouge_travel_time_us"] for item in results], dtype=float
        ),
        "grid_dx_mm_list": np.asarray(
            [item["grid_dx_mm"] for item in results], dtype=float
        ),
        "gouge_1_num_cells_list": np.asarray(
            [item["gouge_1_num_cells"] for item in results], dtype=int
        ),
        "gouge_2_num_cells_list": np.asarray(
            [item["gouge_2_num_cells"] for item in results], dtype=int
        ),
        "misfit_list": np.asarray([item["misfit"] for item in results], dtype=float),
        "spectral_shape_distance_list": np.asarray(
            [item["spectral_shape_distance"] for item in results], dtype=float
        ),
        "target_gouge_travel_time_us": float(
            best["target_gouge_travel_time_us"]
        ),
        "best_thickness_offset_mm": float(best["thickness_offset_mm"]),
        "equivalent_best_thickness_offsets_mm": equivalent_offsets,
        "best_gouge_thickness_1_cm": float(best["gouge_thickness_1_cm"]),
        "best_gouge_thickness_2_cm": float(best["gouge_thickness_2_cm"]),
        "best_misfit": float(best_simulation.misfit),
        "best_spectral_shape_distance": float(
            best_spectral["recorded_vs_simulated_distance"]
        ),
        "simulation_handler": best_simulation,
    }


def joint_kinematic_search_run(args):
    """Evaluate one joint ``(velocity contrast, thickness offset)`` candidate."""

    (
        candidate,
        observed_waveform,
        observed_time,
        stf_handler,
        misfit_interval,
        params,
        assembly_dict,
        fixed_damping,
        plot_output_path,
        enable_plotting,
    ) = args

    assembly_candidate = set_assembly_dict_guessed_with_gouge_velocity_and_damping(
        assembly_dict,
        (candidate["gouge_velocity_1"], candidate["gouge_velocity_2"]),
        (fixed_damping, fixed_damping),
    )
    assembly_candidate = set_assembly_dict_guessed_with_gouge_thicknesses(
        assembly_candidate,
        candidate["gouge_thickness_1_cm"],
        candidate["gouge_thickness_2_cm"],
    )
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(
        params,
        assembly_candidate,
    )

    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_candidate,
        misfit_interval=misfit_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=fixed_damping,
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=params.get(
            "joint_kinematic_points_per_wavelength",
            params.get(
                "gouge_thickness_points_per_wavelength",
                params.get("points_per_wavelength", 4),
            ),
        ),
        store_wavefield=False,
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=enable_plotting,
        make_movie=False,
        plot_output_path=plot_output_path,
    )

    result = dict(candidate)
    result["misfit"] = float(simulation.misfit)
    result["grid_dx_mm"] = 10.0 * float(simulation.grid_handler.dx)
    result["gouge_1_num_cells"] = int(
        simulation.velocity_model_handler.idx_dict["gouge_1"].size
    )
    result["gouge_2_num_cells"] = int(
        simulation.velocity_model_handler.idx_dict["gouge_2"].size
    )
    return result


def _plot_joint_kinematic_search(
    results: list[dict[str, float]],
    output_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plot the two-dimensional constant-time misfit surface."""

    log_contrasts = np.unique(
        [item["log_contrast"] for item in results]
    ).astype(float)
    offsets_mm = np.unique(
        [item["thickness_offset_mm"] for item in results]
    ).astype(float)
    misfit_grid = np.full((offsets_mm.size, log_contrasts.size), np.nan)
    contrast_index = {value: index for index, value in enumerate(log_contrasts)}
    offset_index = {value: index for index, value in enumerate(offsets_mm)}
    for item in results:
        row = offset_index[item["thickness_offset_mm"]]
        column = contrast_index[item["log_contrast"]]
        misfit_grid[row, column] = item["misfit"]

    best = min(results, key=lambda item: item["misfit"])
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    image = ax.pcolormesh(
        log_contrasts,
        offsets_mm,
        misfit_grid,
        shading="auto",
        cmap="viridis",
    )
    ax.scatter(
        best["log_contrast"],
        best["thickness_offset_mm"],
        marker="*",
        s=180,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_xlabel(r"Velocity log-contrast $\delta_c$")
    ax.set_ylabel(r"Thickness transfer $\Delta h$ [mm]")
    ax.set_title("Joint kinematic search at constant travel time")
    fig.colorbar(image, ax=ax, label="Misfit (L2 norm)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return log_contrasts, offsets_mm, misfit_grid


def search_joint_constant_time_kinematics(
    *,
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    stf_handler: UltrasonicDataHandler,
    misfit_interval: np.ndarray,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    homogeneous_velocity: float,
    fixed_damping: float,
    plot_output_path: Path,
    save_plot: bool,
) -> Dict[str, Any]:
    """Search velocity contrast and thickness redistribution jointly.

    Every candidate preserves the complete gouge-plus-groove travel time of
    the homogeneous best model.  The two one-dimensional searches are the
    zero-offset row and zero-contrast column of this surface.
    """

    requested_contrasts = np.asarray(
        params.get(
            "joint_kinematic_log_values",
            params.get("velocity_contrast_log_values", [0.0]),
        ),
        dtype=float,
    ).reshape(-1)
    requested_offsets = np.asarray(
        params.get(
            "joint_kinematic_offset_mm_values",
            params.get("gouge_thickness_offset_mm_values", [0.0]),
        ),
        dtype=float,
    ).reshape(-1)
    if requested_contrasts.size == 0 or np.any(~np.isfinite(requested_contrasts)):
        raise ValueError("joint velocity contrasts must be finite and non-empty")
    if requested_offsets.size == 0 or np.any(~np.isfinite(requested_offsets)):
        raise ValueError("joint thickness offsets must be finite and non-empty")

    log_contrasts = np.unique(np.append(requested_contrasts, 0.0))
    offsets_mm = np.unique(np.append(requested_offsets, 0.0))
    candidates = build_joint_constant_time_candidates(
        assembly_dict,
        homogeneous_velocity,
        log_contrasts,
        offsets_mm,
        wave_type=assembly_dict["wave_type"],
    )

    output_dir = plot_output_path / "joint_kinematic_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_all = bool(params.get("joint_kinematic_plot_all", False)) and save_plot
    args_list = [
        (
            candidate,
            observed_waveform,
            observed_time,
            stf_handler,
            misfit_interval,
            params,
            assembly_dict,
            fixed_damping,
            output_dir,
            plot_all,
        )
        for candidate in candidates
    ]
    num_processes = resolve_worker_count(
        params, "global_search_workers", len(args_list)
    )
    if num_processes == 1:
        results = [joint_kinematic_search_run(args) for args in args_list]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(joint_kinematic_search_run, args_list)
    results = sorted(
        results,
        key=lambda item: (item["thickness_offset_mm"], item["log_contrast"]),
    )
    best = min(results, key=lambda item: item["misfit"])
    contrast_axis, offset_axis, misfit_grid = _plot_joint_kinematic_search(
        results,
        output_dir / "joint_kinematic_misfit.png",
    )

    assembly_best = set_assembly_dict_guessed_with_gouge_velocity_and_damping(
        assembly_dict,
        (best["gouge_velocity_1"], best["gouge_velocity_2"]),
        (fixed_damping, fixed_damping),
    )
    assembly_best = set_assembly_dict_guessed_with_gouge_thicknesses(
        assembly_best,
        best["gouge_thickness_1_cm"],
        best["gouge_thickness_2_cm"],
    )
    assembly_dict.update(assembly_best)
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(
        params,
        assembly_dict,
    )

    best_output_path = output_dir / "best"
    best_output_path.mkdir(parents=True, exist_ok=True)
    best_simulation = UltrasonicModeler()
    best_simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_dict,
        misfit_interval=misfit_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=fixed_damping,
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=params.get(
            "joint_kinematic_points_per_wavelength",
            params.get(
                "gouge_thickness_points_per_wavelength",
                params.get("points_per_wavelength", 4),
            ),
        ),
        store_wavefield=bool(params.get("run_local_inversion", True)),
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=save_plot,
        make_movie=False,
        plot_output_path=best_output_path,
    )

    print(
        "joint constant-time kinematic search: "
        f"delta_c={best['log_contrast']:+.4f}, "
        f"delta_h={best['thickness_offset_mm']:+.3f} mm, "
        f"c1={1.0e4 * best['gouge_velocity_1']:.0f} m/s, "
        f"c2={1.0e4 * best['gouge_velocity_2']:.0f} m/s, "
        f"h1={10.0 * best['gouge_thickness_1_cm']:.3f} mm, "
        f"h2={10.0 * best['gouge_thickness_2_cm']:.3f} mm, "
        f"misfit={best_simulation.misfit:.3f}"
    )

    return {
        "log_contrast_axis": contrast_axis,
        "thickness_offset_mm_axis": offset_axis,
        "misfit_grid": misfit_grid,
        "log_contrast_list": np.asarray(
            [item["log_contrast"] for item in results], dtype=float
        ),
        "thickness_offset_mm_list": np.asarray(
            [item["thickness_offset_mm"] for item in results], dtype=float
        ),
        "gouge_velocity_1_list": np.asarray(
            [item["gouge_velocity_1"] for item in results], dtype=float
        ),
        "gouge_velocity_2_list": np.asarray(
            [item["gouge_velocity_2"] for item in results], dtype=float
        ),
        "gouge_thickness_1_cm_list": np.asarray(
            [item["gouge_thickness_1_cm"] for item in results], dtype=float
        ),
        "gouge_thickness_2_cm_list": np.asarray(
            [item["gouge_thickness_2_cm"] for item in results], dtype=float
        ),
        "stack_travel_time_list_us": np.asarray(
            [item["stack_travel_time_us"] for item in results], dtype=float
        ),
        "grid_dx_mm_list": np.asarray(
            [item["grid_dx_mm"] for item in results], dtype=float
        ),
        "gouge_1_num_cells_list": np.asarray(
            [item["gouge_1_num_cells"] for item in results], dtype=int
        ),
        "gouge_2_num_cells_list": np.asarray(
            [item["gouge_2_num_cells"] for item in results], dtype=int
        ),
        "target_stack_travel_time_us": float(
            best["target_stack_travel_time_us"]
        ),
        "best_log_contrast": float(best["log_contrast"]),
        "best_thickness_offset_mm": float(best["thickness_offset_mm"]),
        "best_gouge_velocity_1": float(best["gouge_velocity_1"]),
        "best_gouge_velocity_2": float(best["gouge_velocity_2"]),
        "best_gouge_thickness_1_cm": float(best["gouge_thickness_1_cm"]),
        "best_gouge_thickness_2_cm": float(best["gouge_thickness_2_cm"]),
        "best_misfit": float(best_simulation.misfit),
        "simulation_handler": best_simulation,
    }


def _assembly_with_common_gouge_density(
    assembly_dict: Dict[str, Any],
    params: Dict[str, Any],
    gouge_density: float,
) -> Dict[str, Any]:
    """Return a copy with a physical density profile enabled.

    Density is expressed in any internally consistent unit; the supplied
    configuration uses g/cm^3.  Only impedance ratios enter the displacement
    equation, so a global change of density unit leaves the waveform intact.
    """
    rho_g = float(gouge_density)
    if not np.isfinite(rho_g) or rho_g <= 0.0:
        raise ValueError("gouge density must be finite and positive")
    rho_steel = float(params.get("steel_density_g_cm3", 7.8))
    candidate = deepcopy(assembly_dict)
    candidate.update(
        {
            # Grid-design helpers inspect this convention before the modeler
            # is called, so keep it explicit in every candidate assembly.
            "embedding_mode": params.get(
                "embedding_mode", candidate.get("embedding_mode", "legacy_stack")
            ),
            "density_model_enabled": True,
            "side1_density": float(
                params.get("side1_density_g_cm3", rho_steel)
            ),
            "central_density": float(
                params.get("central_density_g_cm3", rho_steel)
            ),
            "side2_density": float(
                params.get("side2_density_g_cm3", rho_steel)
            ),
            "gouge_density_1": rho_g,
            "gouge_density_2": rho_g,
        }
    )
    return candidate


def _pre_echo_normalization_interval(
    observed_time: np.ndarray,
    misfit_interval: np.ndarray,
    assembly_dict: Dict[str, Any],
) -> np.ndarray:
    """Fit amplitude before the earliest first internal gouge reverberation."""
    time = np.asarray(observed_time, dtype=float).reshape(-1)
    normalization_domain = sample_interval_mask(misfit_interval, time.size)

    c1 = float(assembly_dict["gouge_velocity_1"])
    c2 = float(assembly_dict["gouge_velocity_2"])
    h1 = float(assembly_dict["gouge_thickness_1"])
    h2 = float(assembly_dict["gouge_thickness_2"])
    arrival = compute_dds_travel_time(assembly_dict, c1, c2)
    first_echo_delay = min(2.0 * h1 / c1, 2.0 * h2 / c2)
    # Keep a small guard from the nominal echo onset.  The same mask is used
    # for every density candidate, so density changes cannot be absorbed into
    # a candidate-specific amplitude rescaling.
    stop = arrival + 0.90 * first_echo_delay
    interval = normalization_domain & (time < stop)
    if np.count_nonzero(interval) < 3:
        return normalization_domain
    return interval


def density_search_run(args):
    """Evaluate one common-density gouge candidate."""
    (
        gouge_density,
        observed_waveform,
        observed_time,
        stf_handler,
        misfit_interval,
        normalization_interval,
        params,
        assembly_dict,
        fixed_damping,
        plot_output_path,
        enable_plotting,
    ) = args
    assembly_candidate = _assembly_with_common_gouge_density(
        assembly_dict, params, gouge_density
    )
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(
        params, assembly_candidate
    )
    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_candidate,
        misfit_interval=misfit_interval,
        normalization_interval=normalization_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=fixed_damping,
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=params.get(
            "density_search_points_per_wavelength",
            params["points_per_wavelength"],
        ),
        store_wavefield=False,
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=enable_plotting,
        make_movie=False,
        plot_output_path=plot_output_path,
    )
    c1 = float(assembly_candidate["gouge_velocity_1"])
    c2 = float(assembly_candidate["gouge_velocity_2"])
    wave_type = assembly_candidate["wave_type"]
    z_side1 = float(assembly_candidate["side1_density"]) * float(
        assembly_candidate["side1_params"]["velocity" + wave_type]
    )
    z_central = float(assembly_candidate["central_density"]) * float(
        assembly_candidate["central_params"]["velocity" + wave_type]
    )
    z_side2 = float(assembly_candidate["side2_density"]) * float(
        assembly_candidate["side2_params"]["velocity" + wave_type]
    )
    z_gouge_1 = float(gouge_density) * c1
    z_gouge_2 = float(gouge_density) * c2
    echo_factor_1 = abs(
        (z_side1 - z_gouge_1) / (z_side1 + z_gouge_1)
        * (z_central - z_gouge_1) / (z_central + z_gouge_1)
    )
    echo_factor_2 = abs(
        (z_central - z_gouge_2) / (z_central + z_gouge_2)
        * (z_side2 - z_gouge_2) / (z_side2 + z_gouge_2)
    )
    return {
        "gouge_density": float(gouge_density),
        "misfit": float(simulation.misfit),
        "nominal_echo_factor": float(0.5 * (echo_factor_1 + echo_factor_2)),
    }


def _plot_density_search(results: list[dict[str, float]], output_path: Path) -> None:
    ordered = sorted(results, key=lambda item: item["gouge_density"])
    density = np.asarray([item["gouge_density"] for item in ordered])
    misfit = np.asarray([item["misfit"] for item in ordered])
    echo = np.asarray([item["nominal_echo_factor"] for item in ordered])
    best = int(np.argmin(misfit))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(density, misfit, marker="o", color="tab:blue")
    ax.scatter(density[best], misfit[best], marker="*", s=150, zorder=3)
    ax.set_xlabel(r"Common gouge density [g/cm$^3$]")
    ax.set_ylabel("Misfit (L2 norm)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(alpha=0.25)
    echo_axis = ax.twinx()
    echo_axis.plot(density, echo, linestyle="--", color="tab:orange")
    echo_axis.set_ylabel(r"Nominal internal echo factor $|R_1R_2|$", color="tab:orange")
    echo_axis.tick_params(axis="y", labelcolor="tab:orange")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def search_common_gouge_density(
    *,
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    stf_handler: UltrasonicDataHandler,
    misfit_interval: np.ndarray,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    fixed_damping: float,
    plot_output_path: Path,
    save_plot: bool,
) -> Dict[str, Any]:
    """Sweep one common gouge density after the kinematic stages."""
    requested = np.asarray(
        params.get("gouge_density_values_g_cm3", [1.9]), dtype=float
    ).reshape(-1)
    if requested.size == 0 or np.any(~np.isfinite(requested)) or np.any(requested <= 0):
        raise ValueError("gouge_density_values_g_cm3 must be finite, positive and non-empty")
    candidates = np.unique(requested)
    output_dir = plot_output_path / "density_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use the same complete direct-arrival interval used by the homogeneous
    # c/damping search.  A pre-echo amplitude fit would make the two objective
    # values incomparable and would over-weight later reverberations.
    normalization_interval = misfit_interval
    plot_all = bool(params.get("density_search_plot_all", False)) and save_plot
    args_list = [
        (
            density,
            observed_waveform,
            observed_time,
            stf_handler,
            misfit_interval,
            normalization_interval,
            params,
            assembly_dict,
            fixed_damping,
            output_dir,
            plot_all,
        )
        for density in candidates
    ]
    num_processes = resolve_worker_count(
        params, "global_search_workers", len(args_list)
    )
    if num_processes == 1:
        results = [density_search_run(args) for args in args_list]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(density_search_run, args_list)
    results = sorted(results, key=lambda item: item["gouge_density"])
    best = min(results, key=lambda item: item["misfit"])
    _plot_density_search(results, output_dir / "gouge_density_misfit.png")

    assembly_best = _assembly_with_common_gouge_density(
        assembly_dict, params, best["gouge_density"]
    )
    assembly_dict.update(assembly_best)
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(
        params, assembly_dict
    )
    best_output_path = output_dir / "best"
    best_output_path.mkdir(parents=True, exist_ok=True)
    best_simulation = UltrasonicModeler()
    best_simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_dict,
        misfit_interval=misfit_interval,
        normalization_interval=normalization_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=fixed_damping,
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=params.get(
            "density_search_points_per_wavelength", params["points_per_wavelength"]
        ),
        store_wavefield=False,
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=save_plot,
        make_movie=False,
        plot_output_path=best_output_path,
    )
    print(
        "common gouge density search: "
        f"rho_g={best['gouge_density']:.3f} g/cm^3, "
        f"nominal echo factor={best['nominal_echo_factor']:.3f}, "
        f"misfit={best_simulation.misfit:.3f}"
    )
    return {
        "gouge_density_list_g_cm3": np.asarray(
            [item["gouge_density"] for item in results], dtype=float
        ),
        "misfit_list": np.asarray([item["misfit"] for item in results], dtype=float),
        "nominal_echo_factor_list": np.asarray(
            [item["nominal_echo_factor"] for item in results], dtype=float
        ),
        "best_gouge_density_g_cm3": float(best["gouge_density"]),
        "best_misfit": float(best_simulation.misfit),
        "normalization_interval": normalization_interval,
        "simulation_handler": best_simulation,
    }


def _assembly_with_common_gouge_material(
    assembly_dict: Dict[str, Any],
    params: Dict[str, Any],
    *,
    gouge_velocity: float,
    gouge_damping: float,
    gouge_density: float,
) -> Dict[str, Any]:
    """Keep both measured layers but assign one common material to them."""

    candidate = set_assembly_dict_guessed_with_gouge_velocity_and_damping(
        assembly_dict,
        (float(gouge_velocity), float(gouge_velocity)),
        (float(gouge_damping), float(gouge_damping)),
    )
    return _assembly_with_common_gouge_density(candidate, params, gouge_density)


def impedance_material_search_run(args) -> Dict[str, float]:
    """Evaluate one ``(c_g, damping_g, rho_g)`` impedance-forward candidate."""

    (
        gouge_velocity,
        gouge_damping,
        gouge_density,
        observed_waveform,
        observed_time,
        stf_handler,
        misfit_interval,
        params,
        assembly_dict,
        points_per_wavelength,
    ) = args
    candidate = _assembly_with_common_gouge_material(
        assembly_dict,
        params,
        gouge_velocity=gouge_velocity,
        gouge_damping=gouge_damping,
        gouge_density=gouge_density,
    )
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(params, candidate)
    simulation = UltrasonicModeler()
    simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=candidate,
        misfit_interval=misfit_interval,
        # This is deliberately identical to the scoring interval: every
        # candidate receives one least-squares amplitude factor over the full
        # direct packet, including its internal reverberations.
        normalization_interval=misfit_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=float(gouge_damping),
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=int(points_per_wavelength),
        store_wavefield=False,
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=False,
        make_movie=False,
    )
    spectral = simulation.write_direct_arrival_spectral_diagnostics(
        output_directory=None,
        frequency_min_mhz=float(params.get("frequency_min", 0.0)),
        frequency_max_mhz=float(params["frequency_cutoff"]),
    )
    return {
        "gouge_velocity": float(gouge_velocity),
        "gouge_damping": float(gouge_damping),
        "gouge_density_g_cm3": float(gouge_density),
        "waveform_shape_misfit": float(simulation.misfit),
        "spectral_shape_distance": float(
            spectral["recorded_vs_simulated_distance"]
        ),
    }


def _evaluate_impedance_material_grid(
    *,
    candidates: list[tuple[float, float, float]],
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    stf_handler: UltrasonicDataHandler,
    misfit_interval: np.ndarray,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    points_per_wavelength: int,
    stage_name: str,
) -> list[Dict[str, float]]:
    args_list = [
        (
            velocity,
            damping,
            density,
            observed_waveform,
            observed_time,
            stf_handler,
            misfit_interval,
            params,
            assembly_dict,
            points_per_wavelength,
        )
        for velocity, damping, density in candidates
    ]
    num_processes = resolve_worker_count(
        params, "global_search_workers", len(args_list)
    )
    print(
        f"{stage_name}: {len(args_list)} receiver-only forward runs at "
        f"{points_per_wavelength} PPW using {num_processes} process(es)"
    )
    if num_processes == 1:
        results = []
        progress_stride = max(1, len(args_list) // 10)
        for index, args in enumerate(args_list, start=1):
            results.append(impedance_material_search_run(args))
            if index == len(args_list) or index % progress_stride == 0:
                print(f"  {stage_name}: {index}/{len(args_list)} candidates")
        return results
    with Pool(processes=num_processes) as pool:
        return pool.map(impedance_material_search_run, args_list)


def _material_result_arrays(results: list[Dict[str, float]]) -> Dict[str, np.ndarray]:
    return {
        "gouge_velocity_list": np.asarray(
            [item["gouge_velocity"] for item in results], dtype=float
        ),
        "gouge_damping_list": np.asarray(
            [item["gouge_damping"] for item in results], dtype=float
        ),
        "gouge_density_list_g_cm3": np.asarray(
            [item["gouge_density_g_cm3"] for item in results], dtype=float
        ),
        "waveform_shape_misfit_list": np.asarray(
            [item["waveform_shape_misfit"] for item in results], dtype=float
        ),
        "spectral_shape_distance_list": np.asarray(
            [item["spectral_shape_distance"] for item in results], dtype=float
        ),
    }


def _profiled_material_metric(
    results: list[Dict[str, float]],
    *,
    x_key: str,
    y_key: str,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.unique([item[x_key] for item in results])
    y_axis = np.unique([item[y_key] for item in results])
    grid = np.full((y_axis.size, x_axis.size), np.nan, dtype=float)
    for iy, y_value in enumerate(y_axis):
        for ix, x_value in enumerate(x_axis):
            values = [
                item[metric_key]
                for item in results
                if np.isclose(item[x_key], x_value)
                and np.isclose(item[y_key], y_value)
            ]
            if values:
                grid[iy, ix] = float(np.min(values))
    return x_axis, y_axis, grid


def _plot_profiled_material_results(
    results: list[Dict[str, float]],
    *,
    metric_key: str,
    metric_label: str,
    output_path: Path,
) -> None:
    projections = (
        ("gouge_velocity", "gouge_density_g_cm3", r"$c_g$ [m/s]", r"$\rho_g$ [g/cm$^3$]"),
        ("gouge_damping", "gouge_density_g_cm3", r"$\nu_g$", r"$\rho_g$ [g/cm$^3$]"),
        ("gouge_velocity", "gouge_damping", r"$c_g$ [m/s]", r"$\nu_g$"),
    )
    best = min(results, key=lambda item: item[metric_key])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, (x_key, y_key, x_label, y_label) in zip(axes, projections):
        x_values, y_values, grid = _profiled_material_metric(
            results, x_key=x_key, y_key=y_key, metric_key=metric_key
        )
        x_scale = 1.0e4 if x_key == "gouge_velocity" else 1.0
        image = axis.pcolormesh(x_scale * x_values, y_values, grid, shading="auto")
        axis.scatter(
            x_scale * best[x_key], best[y_key], marker="*", s=130, color="white",
            edgecolor="black", linewidth=0.7, zorder=3,
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.15)
        fig.colorbar(image, ax=axis, label=metric_label)
    fig.suptitle(f"Profiled joint material search: {metric_label}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _parameter_bounds(
    params: Dict[str, Any],
    key: str,
    *,
    default_lower: Optional[float],
    default_upper: Optional[float],
) -> tuple[Optional[float], Optional[float]]:
    values = params.get(key)
    if values is None:
        return default_lower, default_upper
    if len(values) != 2:
        raise ValueError(f"{key} must contain exactly [lower, upper]")
    lower = default_lower if values[0] is None else float(values[0])
    upper = default_upper if values[1] is None else float(values[1])
    if lower is not None and upper is not None and not lower < upper:
        raise ValueError(f"{key} lower bound must be smaller than upper bound")
    return lower, upper


def search_joint_impedance_material(
    *,
    observed_waveform: np.ndarray,
    observed_time: np.ndarray,
    stf_handler: UltrasonicDataHandler,
    misfit_interval: np.ndarray,
    params: Dict[str, Any],
    assembly_dict: Dict[str, Any],
    plot_output_path: Path,
    save_plot: bool,
) -> Dict[str, Any]:
    """Coarse then fine common-material search with the impedance forward."""

    coarse_velocity = np.asarray(
        params.get("impedance_coarse_velocity_values_cm_us", params["velocity_initial_list"]),
        dtype=float,
    )
    coarse_damping = np.asarray(
        params.get("impedance_coarse_damping_values", params["damping_initial_list"]),
        dtype=float,
    )
    coarse_density = np.asarray(
        params.get("impedance_coarse_density_values_g_cm3", np.linspace(1.8, 3.0, 7)),
        dtype=float,
    )
    coarse_candidates = material_candidates(
        coarse_velocity, coarse_damping, coarse_density
    )
    coarse_ppw = int(params.get("impedance_coarse_points_per_wavelength", 4))
    fine_ppw = int(params.get("impedance_fine_points_per_wavelength", 8))
    if coarse_ppw < 2 or fine_ppw < 2:
        raise ValueError("impedance-search PPW values must be at least two")

    output_dir = plot_output_path / "joint_impedance_material_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_results = _evaluate_impedance_material_grid(
        candidates=coarse_candidates,
        observed_waveform=observed_waveform,
        observed_time=observed_time,
        stf_handler=stf_handler,
        misfit_interval=misfit_interval,
        params=params,
        assembly_dict=assembly_dict,
        points_per_wavelength=coarse_ppw,
        stage_name="coarse impedance search",
    )
    coarse_best = min(coarse_results, key=lambda item: item["waveform_shape_misfit"])

    velocity_bounds = _parameter_bounds(
        params, "impedance_velocity_bounds_cm_us", default_lower=1.0e-9, default_upper=None
    )
    damping_bounds = _parameter_bounds(
        params, "impedance_damping_bounds", default_lower=0.0, default_upper=None
    )
    density_bounds = _parameter_bounds(
        params, "impedance_density_bounds_g_cm3", default_lower=1.5, default_upper=3.1
    )
    fine_count = int(params.get("impedance_fine_points_per_axis", 7))
    fine_velocity = centered_refined_axis(
        coarse_best["gouge_velocity"], coarse_velocity,
        number_of_points=fine_count,
        half_width=params.get("impedance_fine_velocity_half_width_cm_us"),
        lower_bound=velocity_bounds[0], upper_bound=velocity_bounds[1],
    )
    fine_damping = centered_refined_axis(
        coarse_best["gouge_damping"], coarse_damping,
        number_of_points=fine_count,
        half_width=params.get("impedance_fine_damping_half_width"),
        lower_bound=damping_bounds[0], upper_bound=damping_bounds[1],
    )
    fine_density = centered_refined_axis(
        coarse_best["gouge_density_g_cm3"], coarse_density,
        number_of_points=fine_count,
        half_width=params.get("impedance_fine_density_half_width_g_cm3"),
        lower_bound=density_bounds[0], upper_bound=density_bounds[1],
    )
    fine_candidates = material_candidates(fine_velocity, fine_damping, fine_density)
    fine_results = _evaluate_impedance_material_grid(
        candidates=fine_candidates,
        observed_waveform=observed_waveform,
        observed_time=observed_time,
        stf_handler=stf_handler,
        misfit_interval=misfit_interval,
        params=params,
        assembly_dict=assembly_dict,
        points_per_wavelength=fine_ppw,
        stage_name="fine impedance search",
    )
    fine_best = min(fine_results, key=lambda item: item["waveform_shape_misfit"])

    for stage, results in (("coarse", coarse_results), ("fine", fine_results)):
        _plot_profiled_material_results(
            results,
            metric_key="waveform_shape_misfit",
            metric_label=r"$J_{shape}$",
            output_path=output_dir / f"{stage}_waveform_shape_misfit.png",
        )
        _plot_profiled_material_results(
            results,
            metric_key="spectral_shape_distance",
            metric_label=r"$D_f$",
            output_path=output_dir / f"{stage}_spectral_shape_distance.png",
        )

    best_assembly = _assembly_with_common_gouge_material(
        assembly_dict,
        params,
        gouge_velocity=fine_best["gouge_velocity"],
        gouge_damping=fine_best["gouge_damping"],
        gouge_density=fine_best["gouge_density_g_cm3"],
    )
    assembly_dict.update(best_assembly)
    minimum_velocity, maximum_velocity = _simulation_velocity_limits(params, assembly_dict)
    best_output = output_dir / "best_fine_model"
    best_output.mkdir(parents=True, exist_ok=True)
    best_simulation = UltrasonicModeler()
    best_simulation.forward_simulation(
        absorbing=params["absorbing"],
        geometry_type="dds",
        observed_time=observed_time,
        observed_waveform=observed_waveform,
        stf_handler=stf_handler,
        frequency_cutoff=params["frequency_cutoff"],
        assembly_dict=assembly_dict,
        misfit_interval=misfit_interval,
        normalization_interval=misfit_interval,
        minimum_velocity=minimum_velocity,
        maximum_velocity=maximum_velocity,
        maximum_damping=float(fine_best["gouge_damping"]),
        embedding_mode=params["embedding_mode"],
        boundary_condition=params["boundary_condition"],
        source_injection=params["source_injection"],
        receiver_quantity=params["receiver_quantity"],
        points_per_wavelength=fine_ppw,
        store_wavefield=False,
        source_receiver_config=params["source_receiver_config"],
        normalize_waveform=True,
        enable_plotting=save_plot,
        make_movie=False,
        plot_output_path=best_output,
    )
    best_spectral = best_simulation.write_direct_arrival_spectral_diagnostics(
        output_directory=None,
        frequency_min_mhz=float(params.get("frequency_min", 0.0)),
        frequency_max_mhz=float(params["frequency_cutoff"]),
    )
    print(
        "joint impedance material minimum (selected only by J_shape): "
        f"c_g={1.0e4 * fine_best['gouge_velocity']:.0f} m/s, "
        f"damping={fine_best['gouge_damping']:.6f}, "
        f"rho_g={fine_best['gouge_density_g_cm3']:.3f} g/cm^3, "
        f"J_shape={best_simulation.misfit:.3f}, "
        f"D_f={best_spectral['recorded_vs_simulated_distance']:.4f}"
    )
    return {
        "coarse": _material_result_arrays(coarse_results),
        "fine": _material_result_arrays(fine_results),
        "coarse_best_gouge_velocity": float(coarse_best["gouge_velocity"]),
        "coarse_best_gouge_damping": float(coarse_best["gouge_damping"]),
        "coarse_best_gouge_density_g_cm3": float(coarse_best["gouge_density_g_cm3"]),
        "coarse_best_waveform_shape_misfit": float(coarse_best["waveform_shape_misfit"]),
        "coarse_best_spectral_shape_distance": float(coarse_best["spectral_shape_distance"]),
        "best_gouge_velocity": float(fine_best["gouge_velocity"]),
        "best_gouge_damping": float(fine_best["gouge_damping"]),
        "best_gouge_density_g_cm3": float(fine_best["gouge_density_g_cm3"]),
        "best_waveform_shape_misfit": float(best_simulation.misfit),
        "best_spectral_shape_distance": float(
            best_spectral["recorded_vs_simulated_distance"]
        ),
        "selection_metric": "waveform_shape_misfit",
        "simulation_handler": best_simulation,
    }


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

    velocity_contrast_enabled = bool(
        params.get("velocity_contrast_search_enabled", False)
    )
    thickness_search_enabled = bool(
        params.get("gouge_thickness_search_enabled", False)
    )
    joint_kinematic_enabled = bool(
        params.get("joint_kinematic_search_enabled", False)
    )
    density_search_enabled = bool(
        params.get("density_search_enabled", False)
    )
    impedance_material_enabled = bool(
        params.get("impedance_material_search_enabled", False)
    )
    post_impedance_thickness_enabled = bool(
        params.get("post_impedance_thickness_search_enabled", False)
    )
    if post_impedance_thickness_enabled and not impedance_material_enabled:
        raise ValueError(
            "post_impedance_thickness_search_enabled requires "
            "impedance_material_search_enabled=True"
        )
    # The homogeneous and kinematic stages deliberately retain the exact
    # historical density-normalised operator.  Physical density is enabled
    # only inside the optional fifth stage below.
    assembly_dict["density_model_enabled"] = False

    reference_gouge_thickness_1 = float(assembly_dict["gouge_thickness_1"])
    reference_gouge_thickness_2 = float(assembly_dict["gouge_thickness_2"])

    rec_n = assembly_dict["rec_n"]
              
    # Evaluate SNR    
    waveform_snr = evaluate_snr(
        observed_time,
        observed_waveform,
        assembly_dict["steel_only_time_wave"],
    )
    if  waveform_snr < minimum_SNR:
        normal_stress = assembly_dict["normal_stress"]
        shear_stress = assembly_dict["shear_stress"]
        print(f"SNR {waveform_snr:.2f} < {minimum_SNR}. Skipping rec n: {rec_n}, Normal Stress: {normal_stress:.2f}, Shear Stress: {shear_stress:.2f}.")
        return {"simulation_handler": None}

    # Compute windows
    windows = predict_event_windows(
        assembly_dict,
        params["velocity_initial_list"],
        wave_type=assembly_dict["wave_type"],
    )
    # plot_with_windows(observed_time, observed_waveform, windows, title=f"rec_n={assembly_dict['rec_n']} predicted events")
    # print(windows)

    # Generate velocity array
    gouge_velocity_list = params["velocity_initial_list"]
    gouge_damping_list  = params["damping_initial_list"]

    # Misfit_interval is choosen according to STF len
    stf_time = stf_handler.metadata["time_ax_waveform"]
    stf_duration = stf_time[-1]-stf_time[0]  
    misfit_interval = direct_misfit_interval(
        observed_time,
        arrival_min_us=windows["direct"][0],
        arrival_max_us=windows["direct"][1],
        stf_duration_us=stf_duration,
        padding_us=params["direct_window_padding_us"],
    )

    # Determine if we save plots and/or movies
    save_plot  = (assembly_dict['idx_wave'] % plot_save_interval == 0) 
    save_movie = (assembly_dict['idx_wave'] % movie_save_interval == 0) 

    # Construct output paths
    # if save_plot:
    plot_output_name = outfile_name
    plot_output_path = Path(outdir_path_image / plot_output_name)
    plot_output_path.mkdir(parents=True, exist_ok=True)
    plot_output_global = plot_output_path / "global_search_homogeneus"
    plot_output_global.mkdir(parents=True, exist_ok=True)
    # else:
    #     plot_output_path = None

    if save_movie:
        movie_output_name = f"{outfile_name}.mp4"
        movie_output_path = outdir_path_image / movie_output_name
    else:
        movie_output_path = None

    # Prepare arguments for multiprocessing.  A single process executes in
    # the parent process and avoids duplicating large numerical arrays.
    def _build_args(gouge_velocity: float, gouge_damping: float):

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
    num_processes = resolve_worker_count(
        params, "global_search_workers", len(args_list)
    )

    # Multiprocessing over velocity and damping candidates.
    if num_processes == 1:
        results = [global_search_run(args) for args in args_list]
    else:
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
        plot_output_dir = plot_output_global
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
    velocity_boundary = best_gouge_velocity in (
        np.min(gouge_velocity_list), np.max(gouge_velocity_list)
    )
    damping_boundary = best_gouge_damping in (
        np.min(gouge_damping_list), np.max(gouge_damping_list)
    )
    if velocity_boundary or damping_boundary:
        print(
            "Minimum misfit is on the search boundary: "
            f"velocity={velocity_boundary}, damping={damping_boundary}"
        )
        # params["velocity_range"] = 2*params["velocity_range"]
    
    minimum_velocity, maximum_velocity = _legacy_simulation_velocity_limits(
        params,
        assembly_dict,
    )
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
        embedding_mode          = params["embedding_mode"],
        boundary_condition      = params["boundary_condition"],
        source_injection        = params["source_injection"],
        receiver_quantity       = params["receiver_quantity"],
        points_per_wavelength   = params["points_per_wavelength"],
        store_wavefield         = bool(params.get("run_local_inversion", True))
                                  and not (
                                      velocity_contrast_enabled
                                      or thickness_search_enabled
                                      or joint_kinematic_enabled
                                      or density_search_enabled
                                      or impedance_material_enabled
                                      or post_impedance_thickness_enabled
                                  ),
        source_receiver_config  = params["source_receiver_config"],
        normalize_waveform      = True,
        enable_plotting         = save_plot,
        make_movie              = False,
        plot_output_path        = plot_output_global,
        plot_name_prefix        = "best_homogeneus_fit_",
        movie_output_path       = movie_output_path
    )

    homogeneous_best_misfit = float(simulation.misfit)
    homogeneous_assembly = deepcopy(assembly_dict)
    contrast_result = None
    thickness_result = None
    joint_result = None
    density_result = None
    impedance_material_result = None
    post_impedance_thickness_result = None

    # The one-dimensional stages are independent diagnostics around exactly
    # the same homogeneous optimum.  If a joint stage follows, they remain in
    # receiver-only mode; only the final joint model may retain a wavefield for
    # a later local inversion.
    diagnostic_params = dict(params)
    if joint_kinematic_enabled:
        diagnostic_params["run_local_inversion"] = False

    if velocity_contrast_enabled:
        contrast_assembly = deepcopy(homogeneous_assembly)
        contrast_result = search_constant_time_velocity_contrast(
            observed_waveform=observed_waveform,
            observed_time=observed_time,
            stf_handler=stf_handler,
            misfit_interval=misfit_interval,
            params=diagnostic_params,
            assembly_dict=contrast_assembly,
            homogeneous_velocity=float(best_gouge_velocity),
            fixed_damping=float(best_gouge_damping),
            plot_output_path=plot_output_path,
            save_plot=save_plot,
        )
    if thickness_search_enabled:
        thickness_assembly = deepcopy(homogeneous_assembly)
        thickness_result = search_constant_total_gouge_thickness(
            observed_waveform=observed_waveform,
            observed_time=observed_time,
            stf_handler=stf_handler,
            misfit_interval=misfit_interval,
            params=diagnostic_params,
            assembly_dict=thickness_assembly,
            fixed_velocity=float(best_gouge_velocity),
            fixed_damping=float(best_gouge_damping),
            plot_output_path=plot_output_path,
            save_plot=save_plot,
        )
    if joint_kinematic_enabled:
        joint_assembly = deepcopy(homogeneous_assembly)
        joint_result = search_joint_constant_time_kinematics(
            observed_waveform=observed_waveform,
            observed_time=observed_time,
            stf_handler=stf_handler,
            misfit_interval=misfit_interval,
            params=params,
            assembly_dict=joint_assembly,
            homogeneous_velocity=float(best_gouge_velocity),
            fixed_damping=float(best_gouge_damping),
            plot_output_path=plot_output_path,
            save_plot=save_plot,
        )
        assembly_dict.update(joint_assembly)
        simulation = joint_result["simulation_handler"]
    elif velocity_contrast_enabled and thickness_search_enabled:
        # This combination is retained for backward-compatible exploratory
        # runs.  The requested four-stage workflow should enable the joint
        # search, which supersedes this tie-breaking branch.
        if contrast_result["best_misfit"] <= thickness_result["best_misfit"]:
            assembly_dict.update(contrast_assembly)
            simulation = contrast_result["simulation_handler"]
        else:
            assembly_dict.update(thickness_assembly)
            simulation = thickness_result["simulation_handler"]
    elif velocity_contrast_enabled:
        assembly_dict.update(contrast_assembly)
        simulation = contrast_result["simulation_handler"]
    elif thickness_search_enabled:
        assembly_dict.update(thickness_assembly)
        simulation = thickness_result["simulation_handler"]

    if density_search_enabled:
        density_result = search_common_gouge_density(
            observed_waveform=observed_waveform,
            observed_time=observed_time,
            stf_handler=stf_handler,
            misfit_interval=misfit_interval,
            params=params,
            assembly_dict=assembly_dict,
            fixed_damping=float(best_gouge_damping),
            plot_output_path=plot_output_path,
            save_plot=save_plot,
        )
        simulation = density_result["simulation_handler"]

    if impedance_material_enabled:
        impedance_material_result = search_joint_impedance_material(
            observed_waveform=observed_waveform,
            observed_time=observed_time,
            stf_handler=stf_handler,
            misfit_interval=misfit_interval,
            params=params,
            assembly_dict=assembly_dict,
            plot_output_path=plot_output_path,
            save_plot=save_plot,
        )
        simulation = impedance_material_result["simulation_handler"]

    if post_impedance_thickness_enabled:
        post_impedance_thickness_result = search_constant_total_gouge_thickness(
            observed_waveform=observed_waveform,
            observed_time=observed_time,
            stf_handler=stf_handler,
            misfit_interval=misfit_interval,
            params=params,
            assembly_dict=assembly_dict,
            fixed_velocity=float(
                impedance_material_result["best_gouge_velocity"]
            ),
            fixed_damping=float(
                impedance_material_result["best_gouge_damping"]
            ),
            plot_output_path=plot_output_path,
            save_plot=save_plot,
            output_subdirectory="post_impedance_thickness_search",
        )
        simulation = post_impedance_thickness_result["simulation_handler"]

    spectral_result = None
    if params.get("spectral_diagnostics_enabled", True):
        spectral_result = simulation.write_direct_arrival_spectral_diagnostics(
            output_directory=plot_output_path / "spectral_diagnostics",
            frequency_min_mhz=float(params.get("frequency_min", 0.0)),
            frequency_max_mhz=float(params["frequency_cutoff"]),
        )
        print(
            "direct-arrival spectral shape distance: "
            f"Df={spectral_result['recorded_vs_simulated_distance']:.4f} "
            f"over {spectral_result['frequency_min_mhz']:.2f}-"
            f"{spectral_result['frequency_max_mhz']:.2f} MHz"
        )

    best_velocity_1 = float(assembly_dict["gouge_velocity_1"])
    best_velocity_2 = float(assembly_dict["gouge_velocity_2"])
    best_thickness_1 = float(assembly_dict["gouge_thickness_1"])
    best_thickness_2 = float(assembly_dict["gouge_thickness_2"])
    selected_common_velocity = (
        float(impedance_material_result["best_gouge_velocity"])
        if impedance_material_result is not None
        else float(best_gouge_velocity)
    )
    selected_common_damping = (
        float(impedance_material_result["best_gouge_damping"])
        if impedance_material_result is not None
        else float(best_gouge_damping)
    )

    output = {
        'gouge_velocity_list' : gouge_velocity_list,
        'best_gouge_velocity' : selected_common_velocity,
        'best_homogeneous_gouge_velocity': best_gouge_velocity,
        'best_gouge_velocity_1': best_velocity_1,
        'best_gouge_velocity_2': best_velocity_2,
        'reference_gouge_thickness_1_cm': reference_gouge_thickness_1,
        'reference_gouge_thickness_2_cm': reference_gouge_thickness_2,
        'best_gouge_thickness_1_cm': best_thickness_1,
        'best_gouge_thickness_2_cm': best_thickness_2,
        'gouge_damping_list'  : gouge_damping_list,
        'best_gouge_damping'  : selected_common_damping,
        'L2norm_waveform_list': L2norm_waveform_list,
        'homogeneous_best_misfit': homogeneous_best_misfit,
        'velocity_contrast_result': contrast_result,
        'gouge_thickness_result': thickness_result,
        'joint_kinematic_result': joint_result,
        'density_result': density_result,
        'impedance_material_result': impedance_material_result,
        'post_impedance_thickness_result': post_impedance_thickness_result,
        'spectral_diagnostics': spectral_result,
        'spectral_shape_distance': (
            None
            if spectral_result is None
            else spectral_result['recorded_vs_simulated_distance']
        ),
        'best_gouge_density_g_cm3': (
            impedance_material_result['best_gouge_density_g_cm3']
            if impedance_material_result is not None
            else (
                None if density_result is None
                else density_result['best_gouge_density_g_cm3']
            )
        ),
        "simulation_handler": simulation
    }
    return output

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
        embedding_mode          = params["embedding_mode"],
        boundary_condition      = params["boundary_condition"],
        source_injection        = params["source_injection"],
        receiver_quantity       = params["receiver_quantity"],
        points_per_wavelength   = params["points_per_wavelength"],
        store_wavefield         = False,
        source_receiver_config  = params["source_receiver_config"],
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
    simulation: UltrasonicModeler,
    stf_handler: UltrasonicDataHandler,
) -> Dict[str, Union[float, np.ndarray, None]]:
    """
    Local-only continuation step: forward_simulation + inject prior model + run_local_inversion.
    No global grid search.

    Does not change I/O of forward_simulation or run_local_inversion; it only sets initial model arrays.
    """

    plot_save_interval = params["plot_save_interval"]
    movie_save_interval = params["movie_save_interval"]
    outdir_path_image = params["outdir_path_image"]
    plot_sequence_index = int(
        assembly_dict.get(
            "_local_plot_counter",
            assembly_dict["idx_wave"],
        )
    )

    save_plot = (
        plot_sequence_index % plot_save_interval == 0
    )

    save_movie = (
        assembly_dict["idx_wave"] % movie_save_interval == 0
    )

    if save_plot:
        record_output_path = Path(outdir_path_image / f"{outfile_name}")
        record_output_path.mkdir(parents=True, exist_ok=True)
        local_output_path = record_output_path / "local_inversion"
        local_output_path.mkdir(parents=True, exist_ok=True)
    else:
        record_output_path = None
        local_output_path = None
    if save_movie:
        movie_output_path = outdir_path_image / f"{outfile_name}.mp4"
    else: movie_output_path = None

    guessed_arrival_time = compute_dds_travel_time(
        assembly_dict=assembly_dict,
        v_gouge_1=assembly_dict["gouge_velocity_1"],
        v_gouge_2=assembly_dict["gouge_velocity_2"],
    )
    # Misfit_interval is choosen according to STF len
    stf_time = stf_handler.metadata["time_ax_waveform"]
    stf_duration = stf_time[-1]-stf_time[0]  

    misfit_interval = direct_misfit_interval(
        simulation.observed_time,
        arrival_min_us=guessed_arrival_time,
        arrival_max_us=guessed_arrival_time,
        stf_duration_us=stf_duration,
        padding_us=params["direct_window_padding_us"],
    )
    simulation.misfit_interval = misfit_interval
    simulation.assembly_dict = assembly_dict

    # Local inversion hyperparameters
    dw_max_start = 0*np.amax(simulation.source_handler.time_function)
    dw_threshold = 0.01 * dw_max_start
    ds_max_start = 0*np.amax(simulation.source_handler.spatial_function)
    ds_threshold = 0.01 * ds_max_start

    dc_max_start = float(
        params.get("local_velocity_step_fraction", 0.02)
    ) * assembly_dict["gouge_velocity_1"]
    dc_threshold = float(
        params.get("local_velocity_step_threshold_fraction", 0.01)
    ) * dc_max_start

    da_max_start = float(
        params.get("local_damping_step_fraction", 0.05)
    ) * assembly_dict["gouge_damping_1"]
    da_threshold = float(
        params.get("local_damping_step_threshold_fraction", 0.01)
    ) * da_max_start

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
        misfit_threshold     = float(params.get("local_misfit_threshold", 1.0e-8)),
        normalize_waveform   = False,
        amplitude_scale_mode = "fixed",
        fixed_trace_scale    = float(simulation.reference_trace_scale),
        spectral_gamma       = float(params.get("local_spectral_gamma", 1.0)),
        objective_taper_fraction=float(
            params.get("local_objective_taper_fraction", 0.0)
        ),
        update_strategy      = str(
            params.get("local_update_strategy", "alternating")
        ),
        enable_plotting      = save_plot,
        make_movie           = False,
        plot_output_path     = local_output_path,
        plot_name_prefix     = outfile_name,
        movie_output_path    = movie_output_path,
        retain_wavefield     = bool(params.get("local_retain_wavefield", False)),
    )

    # Diagnose the model that was actually accepted by the local inversion.
    # Compute the scalar for every waveform, but write figures only for the
    # records selected by plot_save_interval.
    local_spectral_result = None
    if params.get("local_spectral_diagnostics_enabled", True):
        spectral_output_directory = (
            record_output_path / "local_spectral_diagnostics"
            if record_output_path is not None
            else None
        )
        local_spectral_result = simulation.write_direct_arrival_spectral_diagnostics(
            output_directory=spectral_output_directory,
            frequency_min_mhz=float(params.get("frequency_min", 0.0)),
            frequency_max_mhz=float(params["frequency_cutoff"]),
        )
        print(
            "post-local direct-arrival spectral shape distance: "
            f"Df={local_spectral_result['recorded_vs_simulated_distance']:.4f} "
            f"over {local_spectral_result['frequency_min_mhz']:.2f}-"
            f"{local_spectral_result['frequency_max_mhz']:.2f} MHz"
        )

    # Carry the layer means into the next waveform only as travel-time/window
    # predictors.  The full nodal profiles remain in the simulation handler
    # and are the actual starting model of the next local inversion.
    idx_dict = simulation.velocity_model_handler.idx_dict
    for layer_number in (1, 2):
        layer_indices = idx_dict[f"gouge_{layer_number}"]
        assembly_dict[f"gouge_velocity_{layer_number}"] = float(
            np.mean(simulation.velocity_model_handler.velocity_array[layer_indices])
        )
        assembly_dict[f"gouge_damping_{layer_number}"] = float(
            np.mean(simulation.velocity_model_handler.damping_array[layer_indices])
        )

    # Return in the same schema as global_search_homogeneus
    return {
        "gouge_velocity_list": None,
        "best_gouge_velocity": assembly_dict["gouge_velocity_1"],
        "gouge_damping_list": None,
        "best_gouge_damping": assembly_dict["gouge_damping_1"],
        "L2norm_waveform_list": None,
        "local_spectral_diagnostics": local_spectral_result,
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
    machine_name    = "big_biax"
    experiment_name = "r0072suw03anhdol_6035"
    wave_type       = "_s"    # that "_" is ugly, but needed
    data_type_uw    = "uw_data/data_tsv_files" + wave_type
    data_type_mech  = "mechanical_data"
    mech_file_name  = f"{experiment_name}_data_rp"
    outfolder_name  = "test_2026-10-08_s"
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
        "absorbing"                 : False,
        "embedding_mode"            : "effective_surface",
        "boundary_condition"        : "free",
        "source_injection"          : "body_force",
        "receiver_quantity"         : "velocity",
        # Keep the broad homogeneous search cheap; the small diagnostic
        # thickness search can use a finer grid in receiver-only mode.
        "points_per_wavelength"     : 4,
        "source_receiver_config"    : {
            "interpolation_radius_cells": 4,
            "source_extension_cm": 0.0,
            "receiver_extension_cm": 0.0,
            "source_normalization": "integral",
            "receiver_normalization": "sum",
            "spatial_boundary_mode": "truncate",
        },
        "frequency_min"             : 0.1,
        "direct_window_padding_us"  : 0.25,
        "diagnostic_plot_only"      : False,
        # The earlier kinematic diagnostics remain available, but are silent
        # during the common-material impedance search below.  Thus c1=c2,
        # rho1=rho2 and the two measured gouge thicknesses remain fixed.
        "velocity_contrast_search_enabled": False,
        "velocity_contrast_points_per_wavelength": 8,
        "velocity_contrast_log_values": np.linspace(-0.10, 0.10, 11),
        "velocity_contrast_plot_all": True,
        "gouge_thickness_search_enabled": False,
        "gouge_thickness_points_per_wavelength": 8,
        "gouge_thickness_offset_mm_values": np.linspace(-0.2, 0.2, 40),
        "gouge_thickness_plot_all": False,
        "gouge_thickness_tie_relative_tolerance": 1.0e-6,
        "joint_kinematic_search_enabled": False,
        "joint_kinematic_points_per_wavelength": 8,
        # If omitted, the joint search reuses the two arrays above.
        # "joint_kinematic_log_values": np.linspace(-0.10, 0.10, 11),
        # "joint_kinematic_offset_mm_values": np.linspace(-0.05, 0.05, 11),
        "joint_kinematic_plot_all": False,
        # Legacy one-dimensional density-only diagnostic (kept available).
        "density_search_enabled": False,
        "density_search_points_per_wavelength": 8,
        "steel_density_g_cm3": 7.8,
        "gouge_density_values_g_cm3": np.linspace(1.4, 2.5, 23),
        "density_search_plot_all": False,
        # Joint material workflow:
        # 1) the historical rho-normalised homogeneous c/damping grid above is
        #    retained as a baseline;
        # 2) coarse physical-impedance grid at 4 PPW;
        # 3) local Cartesian refinement around its minimum at 8 PPW;
        # 4) J_shape and D_f are stored separately, and only J_shape selects
        #    the model at this stage.
        "impedance_material_search_enabled": True,
        "impedance_coarse_points_per_wavelength": 4,
        "impedance_coarse_velocity_values_cm_us": np.linspace(0.1200, 0.1700, 10),
        "impedance_coarse_damping_values": np.linspace(0.0000, 0.0005, 5),
        "impedance_coarse_density_values_g_cm3": np.linspace(1.6, 2.9, 13),
        "impedance_fine_points_per_wavelength": 4,
        "impedance_fine_points_per_axis": 7,
        "impedance_density_bounds_g_cm3": (1.5, 3.1),
        # Once the common material is fixed, redistribute only the measured
        # total gouge thickness.  With c1=c2 this cannot alter the direct
        # travel time; it tests only the relative timing of internal echoes.
        "post_impedance_thickness_search_enabled": True,
        "spectral_diagnostics_enabled": True,
        "parallel_cpu_reserve": 1,
        "global_search_workers": 10,
        # Local records are split into three interlaced continuation chains:
        # record k starts from the accepted model at k-3 in its chain.  A
        # worker lives for at most three records, then a fresh process resumes
        # from the compact accepted-model checkpoint of the preceding chunk.
        "local_inversion_workers": 3,
        "local_anchor_stride": 3,
        "local_records_per_worker": 3,
        # Reuse the append-only result stream after an interruption. Legacy
        # pickles without explicit checkpoints are reconstructed from their
        # saved profiles; new rows contain an exact compact continuation state.
        "resume_from_pickle": True,
        # Local continuation: rho(x) remains fixed at the global-search value;
        # only c(x) and Kelvin-Voigt damping are updated in the gouge/groove
        # region.  Small initial steps are appropriate around the refined
        # global minimum.
        "run_local_inversion"      : True,
        "local_velocity_step_fraction": 0.02,
        "local_velocity_step_threshold_fraction": 0.01,
        "local_damping_step_fraction": 0.05,
        "local_damping_step_threshold_fraction": 0.01,
        # The final global trace scale is calibrated once on the first record
        # and then held fixed.  gamma=1 is exactly the relative L2 objective;
        # larger values explicitly upweight spectral-energy shape.
        "local_spectral_gamma": 2.0,
        "local_objective_taper_fraction": 0.0,
        "local_misfit_threshold": 1.0e-8,
        # "alternating" accepts velocity and damping in separate iterations.
        # Use "cross_talk" on a diagnostic subset to evaluate velocity-only,
        # damping-only and joint proposals at every iteration.  "joint"
        # reproduces the historical coupled acceptance rule.
        "local_update_strategy": "cross_talk",
        # D_f is computed for every local result.  The two spectral figures
        # follow plot_save_interval to avoid generating one folder per record.
        "local_spectral_diagnostics_enabled": True,
        # The next waveform reuses the accepted material profiles, not the
        # full space-time field.  Releasing it greatly lowers persistent RAM.
        "local_retain_wavefield": False,
        "num_waveform2process"      : 1,     # int, equespatially waveforms to sample for processing. Of "None", process all
        "maxtime2simulate"          : 25,           # [mus]
        "frequency_cutoff"          : 8,            # [MHz] low pass onserved data and simulate up to this frequency
        "minimum_SNR"               : 10,            # skip computation until time interval where signal should be is above SNR times surely-only-noise part 
        "velocity_initial_list"     : np.linspace(0.1200, 0.1700, 30),  # [cm/mus] first guess of best velocity. There is a visual tool for it, if needed
        "min_velocity2simulate"     : 0.10,         # [cm/mus] if not passed, computed by assembly and gouge velocity range
        "max_velocity2simulate"     : 0.33,         # [cm/mus]
        "damping_initial_list"      : np.linspace(0.000,0.0009, 10),
        "plot_save_interval"        : 1,
        "movie_save_interval"       : 15000,
        "l2norm_plot"               : True,
        "outdir_path_l2norm"        : outdir_path_l2norm[0],
        "outdir_path_image"         : outdir_path_image[0],
        "n_iterations"              : 20,
        "reduce_factor"             : 10/9,

        # densita', stimate
        "steel_density_g_cm3": 7.85,
        "side1_density_g_cm3": 7.85,
        "central_density_g_cm3": 4.50, # e' in titanio, diocane
        "side2_density_g_cm3": 7.85,
    }

    # Load the Source Time Function (HDF5)
    stf_h5_path = (
        dir_manager.paths.analysis_root("on_bench", "STF_pfake500_s250_multichannel")
        / f"source_time_functions{wave_type}" 
        / "stf.h5"
    )

    # read the STF saved under /stf/width250_volt70/cycle_000000/source_waveform
    stf_handler = UltrasonicDataHandler.read_stf_hdf5(
        stf_h5_path,
        name="s_70V_100nm",   # "width250_volt70_local_inversion" when available
        cycle_index=0,
    )

    ########### HO MESSO UN TRONCAMENTO ARBITRATIO A 4 MUS #####################
    stf_handler.waveform_data = stf_handler.truncate_time(4).waveform_data.squeeze(0)   # shape (L,) e c'e' anche un troncamento
    # plt.plot(stf_handler.metadata['time_ax_waveform'], stf_handler.waveform_data)
    # plt.show()
    # sys.exit()

    # Load Mechanical Data
    mech_folder = Path(dir_manager.base_dir) / f"experiments_{machine_name}" / experiment_name / data_type_mech
    mech_data, sync_data, sync_peaks = MechanicalDataHandler.find_and_load(mech_folder, mech_file_name)

    # print(mech_data.head(5))
    # sys.exit()
    # check sync with mech data
    # f, ax = plt.subplots()
    # ax.plot(mech_data.shear_stress_MPa, c='r')
    # ax.plot(mech_data.normal_stress_MPa, c='g')
    # ax1 = ax.twinx()
    # ax1.vlines(sync_peaks[2]+13000,ymin=0,ymax=5)
    # ax1.plot(mech_data.sync, c='k')
    # plt.show()
    # sys.exit()


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
        "embedding_mode"      : params["embedding_mode"],
    }
    assembly_dict["velocity" + wave_type] = side1_params["velocity" + wave_type]
    # Duct-taper to check mininimum possible arrival time. TO BE REMOVED!!!
    assembly_dict["steel_only_time_wave"] = compute_dds_travel_time(
        assembly_dict=assembly_dict,
        wave_type=wave_type,
    )
    
    # Make UW path list
    infile_path_list_uw = sorted( dir_manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type_uw))
 
    # Process each UW file
    for chosen_uw_file, infile_path in enumerate(infile_path_list_uw):
        infile_name = infile_path.name.split(".")[0]

        print(infile_name)
        if infile_name != "006_reload_ss_1_S":
            continue

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
                .zero_before(assembly_dict["steel_only_time_wave"] / 5) # zero early window [µs]
                .bandpass(
                    freqmin=params["frequency_min"],
                    freqmax=params["frequency_cutoff"],
                )
        )

        # import matplotlib.cm as cm
        # import matplotlib.colors as colors

        # # -----------------------------
        # # Global matplotlib parameters
        # # -----------------------------
        # plt.rcParams.update({
        #     "font.size": 12,
        #     "axes.titlesize": 16,
        #     "axes.titleweight": "bold",
        #     "axes.labelsize": 14,
        #     "axes.labelweight": "bold",
        #     "xtick.labelsize": 12,
        #     "ytick.labelsize": 12,
        #     "xtick.major.size": 6,
        #     "ytick.major.size": 6,
        #     "xtick.major.width": 1.5,
        #     "ytick.major.width": 1.5,
        #     "axes.linewidth": 1.5,
        # })

        # # -----------------------------
        # # Metadata
        # # -----------------------------
        # t_acq = uw_data_handler.metadata.get("time_ax_acquisition", None)

        # print(
        #     "mech time start/end =",
        #     mech_data_slice["time_s"].iloc[0],
        #     mech_data_slice["time_s"].iloc[-1],
        # )
        # print(f"num waves: {uw_data_handler.metadata.get('number_of_waveforms')}")

        # time = uw_data_handler.metadata["time_ax_waveform"]
        # data = uw_data_handler.waveform_data  # shape: (n_waveforms, n_samples)
        # n_waveforms = data.shape[0]

        # # -----------------------------
        # # Colormap
        # # -----------------------------
        # norm = colors.Normalize(vmin=0, vmax=n_waveforms - 1)
        # cmap = cm.winter

        # # -----------------------------
        # # Figure
        # # -----------------------------
        # fig, ax = plt.subplots(figsize=(9, 5.5))

        # for i in range(n_waveforms):
        #     ax.plot(
        #         time,
        #         data[i, :],
        #         color=cmap(norm(i)),
        #         alpha=0.25,
        #         linewidth=0.8,
        #     )

        # # -----------------------------
        # # Colorbar
        # # -----------------------------
        # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        # sm.set_array([])

        # cbar = fig.colorbar(
        #     sm,
        #     ax=ax,
        #     pad=0.02,
        #     aspect=30,
        # )
        # cbar.set_label(
        #     "Experiment time progression",
        #     fontsize=13,
        #     fontweight="bold",
        # )

        # # Map waveform index → acquisition time
        # cbar.set_ticks([0, n_waveforms // 2, n_waveforms - 1])
        # cbar.set_ticklabels([
        #     "Start",
        #     "Mid",
        #     "End",
        # ])

        # cbar.ax.tick_params(
        #     width=1.5,
        #     length=5,
        #     labelsize=11,
        # )

        # # -----------------------------
        # # Axes formatting
        # # -----------------------------
        # ax.set_xlabel("Waveform time [µs]")
        # ax.set_ylabel("Amplitude [a.u.]")
        # ax.set_title("Ultrasonic Waveform Evolution Over Experiment Time", pad=12)

        # ax.tick_params(
        #     direction="in",
        #     which="both",
        #     top=True,
        #     right=True,
        # )

        # ax.minorticks_on()

        # ax.grid(
        #     which="major",
        #     linestyle="--",
        #     linewidth=0.6,
        #     alpha=0.4,
        # )

        # plt.tight_layout()
        # plt.show()
        # sys.exit()

        if not params["diagnostic_plot_only"]:
            process_uw_file(
                infile_path=infile_path,
                uw_data_handler=uw_data_handler,
                mechanical_dataframe=mech_data_slice,
                stf_handler=stf_handler,
                params=params,
                assembly_dict=assembly_dict,
            )
            continue
