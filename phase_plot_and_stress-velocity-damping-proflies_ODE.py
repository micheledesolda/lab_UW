import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

# ============================================================
# USER PARAMETERS
# ============================================================

PKL_PATH = Path("/home/michele/Desktop/Dottorato/active_source_implementation/experiments_big_biax/r0072suw03anhdol_6035/data_analysis/test_2026-10-06_1/005_reload_ss_1_P.pkl")
OUTDIR = Path("J_operator_analysis")
OUTDIR.mkdir(exist_ok=True)

DPI = 300

# Choose which gouge to use for J0,J1,J2:
# options: "gouge_1", "gouge_2", "both_series"
GOUge_MODE = "both_series"

# If your inverted velocity is density-normalized wave speed:
# mu = velocity^2.
# Here velocity is assumed cm/us, damping cm^2/us, thickness cm.
USE_MU_FROM_VELOCITY_SQUARED = True

# Smoothing for dmu/dt used in J2.
# Use odd window; increase if J2 is noisy.
SMOOTH_WINDOW = 101
SMOOTH_POLY = 3

# Stress extrema detection
MIN_CYCLE_DISTANCE_S = 20.0
STRESS_PROMINENCE_MPA = 0.05

# ODE / loading options
# If True, use a displacement key from the pickle if available.
# If False, reconstruct delta_ext = delta0 + V_ext * (t-t0).
TRY_EXTRACT_DISPLACEMENT = True

# Candidate displacement keys; edit to match your data.
DELTA_EXT_KEY_CANDIDATES = [
    "delta_ext",
    "external_displacement",
    "actuator_displacement",
    "load_point_displacement",
    "lp_displacement",
    "shear_displacement",
    "shear_disp",
    "displacement_um",
    "displacement_mm",
    "slip_um",
    "slip_mm",
]

# Units conversion for displacement if extracted.
# We need delta_ext in cm to be consistent with J units.
DISPLACEMENT_UNIT = "auto"  # "cm", "mm", "um", or "auto"

# If no displacement is found, impose a linear ramp.
# Use cm/s. Example: 10 um/s = 1e-3 cm/s.
V_EXT_CM_PER_S = 1e-3
DELTA_EXT0_CM = 0.0

# External stiffness per area K_ext^tau.
# Units must be MPa/cm if tau is MPa and delta is cm.
# Set to None to fit it by least squares over a calibration window.
K_EXT_TAU_MPA_PER_CM = None

# Calibration window for fitting K_ext^tau if None.
# Options: "pre_peak", "all", or tuple (tmin, tmax)
K_FIT_WINDOW = "pre_peak"

# Output names
SAVE_PDF = True


# ============================================================
# UTILITIES
# ============================================================

def read_concatenated_pickle(path):
    out = []
    with path.open("rb") as f:
        while True:
            try:
                out.append(pickle.load(f))
            except EOFError:
                break
    return out


def savefig(fig, stem):
    png = OUTDIR / f"{stem}.png"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(OUTDIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")


def first_available_key(records, candidates):
    keys = set().union(*[set(r.keys()) for r in records[:min(len(records), 50)]])
    for k in candidates:
        if k in keys:
            return k
    return None


def infer_displacement_unit_from_key(key):
    lk = key.lower()
    if "um" in lk or "micron" in lk:
        return "um"
    if "mm" in lk:
        return "mm"
    if "cm" in lk:
        return "cm"
    return "cm"


def convert_disp_to_cm(x, unit):
    if unit == "auto":
        raise ValueError("auto should be resolved before conversion.")
    if unit == "cm":
        return x
    if unit == "mm":
        return x * 0.1
    if unit == "um":
        return x * 1e-4
    raise ValueError(f"Unknown displacement unit: {unit}")


def safe_gradient(y, t, smooth=True):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if smooth and len(y) >= SMOOTH_WINDOW:
        window = SMOOTH_WINDOW
        if window % 2 == 0:
            window += 1
        if window >= len(y):
            window = len(y) - 1 if (len(y) - 1) % 2 == 1 else len(y) - 2
        if window >= SMOOTH_POLY + 2:
            y = savgol_filter(y, window_length=window, polyorder=SMOOTH_POLY, axis=0)

    return np.gradient(y, t, axis=0)


def compute_J_for_one_gouge(t, velocity, damping, thickness_cm):
    """
    velocity: (nt, nz), cm/us
    damping:  (nt, nz), cm^2/us
    thickness_cm: (nt,), cm

    mu = velocity^2, units cm^2/us^2
    nu = damping, units cm^2/us
    z integration gives:
      J0 = ∫ dz/mu
      J1 = ∫ nu/mu^2 dz
      J2 = ∫ nu*dmu_dt/mu^3 dz

    Note: dmu_dt uses slow time in seconds. This creates mixed slow/fast units.
    That is fine diagnostically only if your derivation consistently treats t
    as the slow time. For strict dimensional consistency, convert microseconds
    to seconds in mu/nu definitions globally.
    """
    V_cm_per_us = np.asarray(velocity, dtype=float)
    nu_cm2_per_us = np.asarray(damping, dtype=float)

    V = V_cm_per_us * 1e6        # cm/s
    nu = nu_cm2_per_us * 1e6     # cm^2/s

    mu = V**2                    # cm^2/s^2
    h = np.asarray(thickness_cm, dtype=float)

    if USE_MU_FROM_VELOCITY_SQUARED:
        mu = V**2
    else:
        raise NotImplementedError("Provide mu directly if not using velocity^2.")

    # normalized coordinate eta, assumed uniform through gouge
    nz = V.shape[1]
    eta = np.linspace(0.0, 1.0, nz)

    dmu_dt = safe_gradient(mu, t, smooth=True)

    J0 = np.empty(len(t))
    J1 = np.empty(len(t))
    J2 = np.empty(len(t))

    for i in range(len(t)):
        z = eta * h[i]
        J0[i] = np.trapezoid(1.0 / mu[i], z)
        J1[i] = np.trapezoid(nu[i] / mu[i]**2, z)
        J2[i] = np.trapezoid(nu[i] * dmu_dt[i] / mu[i]**3, z)

    return J0, J1, J2


def get_thickness(records, gouge_name):
    """
    Tries the obvious keys. Falls back to constant from first record.
    """
    candidates = [
        f"{gouge_name}_thickness_cm",
        f"{gouge_name}_thickness",
        f"{gouge_name}_h_cm",
    ]

    key = first_available_key(records, candidates)
    if key is not None:
        return np.array([r[key] for r in records], dtype=float), key

    # Older/global keys
    if gouge_name == "gouge_1":
        candidates = ["gouge_thickness_1_cm", "gouge_1_thickness_cm"]
    else:
        candidates = ["gouge_thickness_2_cm", "gouge_2_thickness_cm"]

    key = first_available_key(records, candidates)
    if key is not None:
        return np.array([r[key] for r in records], dtype=float), key

    raise KeyError(f"No thickness key found for {gouge_name}.")


# ============================================================
# LOAD DATA
# ============================================================

records_all = read_concatenated_pickle(PKL_PATH)

records = [
    r for r in records_all
    if isinstance(r, dict)
    and all(k in r for k in [
        "acquisition_time_s",
        "shear_stress_MPa",
        "gouge_1_velocity",
        "gouge_1_damping",
        "gouge_2_velocity",
        "gouge_2_damping",
    ])
]

if not records:
    raise RuntimeError("No usable profile records found.")

t = np.array([r["acquisition_time_s"] for r in records], dtype=float)
tau_obs = np.array([r["shear_stress_MPa"] for r in records], dtype=float)

V1 = np.vstack([np.asarray(r["gouge_1_velocity"], dtype=float) for r in records])
D1 = np.vstack([np.asarray(r["gouge_1_damping"], dtype=float) for r in records])
V2 = np.vstack([np.asarray(r["gouge_2_velocity"], dtype=float) for r in records])
D2 = np.vstack([np.asarray(r["gouge_2_damping"], dtype=float) for r in records])

h1, h1_key = get_thickness(records, "gouge_1")
h2, h2_key = get_thickness(records, "gouge_2")

print(f"Loaded {len(records)} usable records")
print(f"time: {t.min():.3f}–{t.max():.3f} s")
print(f"stress: {tau_obs.min():.3f}–{tau_obs.max():.3f} MPa")
print(f"h1 from {h1_key}, h2 from {h2_key}")
print(f"V1 shape={V1.shape}, D1 shape={D1.shape}")
print(f"V2 shape={V2.shape}, D2 shape={D2.shape}")


# ============================================================
# COMPUTE J OPERATORS
# ============================================================

J0_1, J1_1, J2_1 = compute_J_for_one_gouge(t, V1, D1, h1)
J0_2, J1_2, J2_2 = compute_J_for_one_gouge(t, V2, D2, h2)

if GOUge_MODE == "gouge_1":
    J0, J1, J2 = J0_1, J1_1, J2_1
    mode_label = "Gouge 1"
elif GOUge_MODE == "gouge_2":
    J0, J1, J2 = J0_2, J1_2, J2_2
    mode_label = "Gouge 2"
elif GOUge_MODE == "both_series":
    # Double direct shear: two layers in series for displacement accommodation.
    # If both transmit the same macroscopic traction, compliances add.
    J0, J1, J2 = J0_1 + J0_2, J1_1 + J1_2, J2_1 + J2_2
    mode_label = "Gouge 1 + Gouge 2, series"
else:
    raise ValueError(GOUge_MODE)

# Remove pathological values if any
good = np.isfinite(J0) & np.isfinite(J1) & np.isfinite(J2) & (J1 != 0)
if not np.all(good):
    print(f"Warning: dropping {np.sum(~good)} bad records")
    t = t[good]
    tau_obs = tau_obs[good]
    J0, J1, J2 = J0[good], J1[good], J2[good]
    J0_1, J1_1, J2_1 = J0_1[good], J1_1[good], J2_1[good]
    J0_2, J1_2, J2_2 = J0_2[good], J1_2[good], J2_2[good]


# ============================================================
# STRESS MARKERS
# ============================================================

main_peak_idx = int(np.argmax(tau_obs))
dt_med = float(np.median(np.diff(t)))
distance_samples = max(20, int(MIN_CYCLE_DISTANCE_S / dt_med))

post = np.arange(main_peak_idx + 1, len(tau_obs))

mins_rel, _ = find_peaks(
    -tau_obs[post],
    prominence=STRESS_PROMINENCE_MPA,
    distance=distance_samples,
)
maxs_rel, _ = find_peaks(
    tau_obs[post],
    prominence=STRESS_PROMINENCE_MPA,
    distance=distance_samples,
)

mins = post[mins_rel]
maxs = post[maxs_rel]

first_min_idx = int(mins[0]) if len(mins) else min(len(t) - 1, main_peak_idx + 1)

pre = np.arange(0, main_peak_idx + 1)
transition = np.arange(main_peak_idx, first_min_idx + 1)
stickslip = np.arange(first_min_idx, len(t))


# ============================================================
# J0-J1 PHASE PLOT — RAW POSITIVE OPERATORS, NOT STANDARDIZED
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})

fig, ax = plt.subplots(figsize=(12.5, 9.5))

# Raw operator coordinates.
# J0 = ∫ dz / μ
# J1 = ∫ ν / μ² dz
xJ = J0
yJ = J1

ax.plot(xJ, yJ, color="0.72", lw=1.0, alpha=0.6, zorder=1)

ax.scatter(
    xJ[pre], yJ[pre],
    s=14, c="#2c7fb8", alpha=0.70,
    edgecolor="none",
    label="Elastic loading up to main peak",
    zorder=2,
)

ax.scatter(
    xJ[transition], yJ[transition],
    s=28, c="#fdae61", alpha=0.88,
    edgecolor="none",
    label="Post-peak transition",
    zorder=3,
)

ax.scatter(
    xJ[stickslip], yJ[stickslip],
    s=14, c="#1a9850", alpha=0.70,
    edgecolor="none",
    label="Stick-slip regime",
    zorder=2,
)

ax.scatter(
    xJ[main_peak_idx], yJ[main_peak_idx],
    s=420, marker="*",
    c="#d7191c", edgecolor="black", lw=1.0,
    label="Main stress peak",
    zorder=6,
)

ax.scatter(
    xJ[first_min_idx], yJ[first_min_idx],
    s=210, marker="D",
    c="#984ea3", edgecolor="black", lw=0.9,
    label="Onset of stick-slip",
    zorder=6,
)

if len(maxs):
    ax.scatter(
        xJ[maxs], yJ[maxs],
        s=140, marker="^",
        c="#e41a1c", edgecolor="black", lw=0.8,
        label="Stick-slip stress maxima",
        zorder=5,
    )

if len(mins):
    ax.scatter(
        xJ[mins], yJ[mins],
        s=140, marker="v",
        c="#377eb8", edgecolor="black", lw=0.8,
        label="Stick-slip stress minima",
        zorder=5,
    )

# Optional: if the dynamic range is too large, log axes preserve positivity
# without destroying physical meaning.
USE_LOG_AXES_FOR_J_PHASE = False

if USE_LOG_AXES_FOR_J_PHASE:
    ax.set_xscale("log")
    ax.set_yscale("log")

ax.set_xlabel(r"$J_0=\int dz/\mu$  [$\mu$s$^2$/cm]")
ax.set_ylabel(r"$J_1=\int \nu\,dz/\mu^2$  [$\mu$s$^3$/cm]")
ax.set_title(f"Reduced-operator phase trajectory: {mode_label}")

ax.grid(True, ls="--", alpha=0.35)
ax.legend(frameon=True, framealpha=0.94)

savefig(fig, "J0_J1_phase_trajectory_raw")

# ============================================================
# J0,J1,J2 TIME SERIES AGAINST STRESS
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True, constrained_layout=True)

for ax, Y, lab in zip(axes, [J0, J1, J2], ["$J_0$", "$J_1$", "$J_2$"]):
    ax.plot(t, Y, lw=1.5, label=lab)
    ax.axvline(t[main_peak_idx], color="k", ls="--", lw=1.0, alpha=0.7)
    ax.axvline(t[first_min_idx], color="k", ls=":", lw=1.2, alpha=0.7)
    ax.set_ylabel(lab)
    ax.grid(True, ls="--", alpha=0.35)

    ax2 = ax.twinx()
    ax2.plot(t, tau_obs, color="0.35", lw=1.0, alpha=0.55)
    ax2.set_ylabel("$\\tau_{obs}$ (MPa)")

axes[-1].set_xlabel("Acquisition time (s)")
fig.suptitle(f"Reduced operators versus observed shear stress: {mode_label}", y=1.02)

savefig(fig, "J_operators_vs_stress")

# ============================================================
# EXTRACT DELTA_EXT
# ============================================================

print("\nAvailable keys containing displacement-like words:")

disp_words = ["lp_disp",
    "disp", "delta", "slip", "lvdt", "actuator",
    "load", "point", "position", "stroke", "shear"
]

all_keys = sorted(set().union(*[set(r.keys()) for r in records[:min(len(records), 50)]]))

for k in all_keys:
    if any(w in k.lower() for w in disp_words):
        val = records[0].get(k)
        try:
            shape = np.asarray(val).shape
        except Exception:
            shape = None
        print(f"{k:40s} shape={shape} first={val}")


delta_key = None
delta_ext_cm = None

if TRY_EXTRACT_DISPLACEMENT:
    delta_key = first_available_key(records, DELTA_EXT_KEY_CANDIDATES)

if delta_key is not None:
    raw = np.array([r[delta_key] for r in records], dtype=float)

    unit = DISPLACEMENT_UNIT
    if unit == "auto":
        unit = infer_displacement_unit_from_key(delta_key)

    delta_ext_cm = convert_disp_to_cm(raw, unit)

    print(f"\nUsing displacement key {delta_key!r}")
    print(f"Interpreted displacement unit: {unit}")
    print("Converted displacement to cm")

else:
    # raise RuntimeError(
    #     "\nNo imposed-displacement key found.\n"
    #     "The ODE test needs delta_ext(t). I am not using a fake linear ramp, "
    #     "because it makes the stress prediction meaningless.\n"
    #     "Look at the printed displacement-like keys above, then add the correct "
    #     "one to DELTA_EXT_KEY_CANDIDATES."
    # )

    print(
        "\nNo imposed-displacement key found.\n"
        "The ODE test needs delta_ext(t). "
        "I am using a manually inserted constant loading velocity:\n"
    )

    V_EXT_CM_PER_S = 0.001  # cm/s
    DELTA_EXT0_CM = 0.0    # cm, arbitrary offset; fitted offset handles this later

    delta_ext_cm = DELTA_EXT0_CM + V_EXT_CM_PER_S * (t - t[0])

    print(f"Using V_EXT_CM_PER_S = {V_EXT_CM_PER_S:g} cm/s")
    print(f"delta_ext_cm shape = {delta_ext_cm.shape}")

delta_ext_cm_raw = delta_ext_cm.copy()

# ============================================================
# FIT K_EXT^TAU AND DELTA OFFSET IF NEEDED
# ============================================================

# The external relation implies:
#   tau = K * (delta_ext - delta)
# but delta is unobserved. The ODE also contains delta_ext.
# We can fit K and an additive delta offset by minimizing ODE residual:
#   d tau_obs/dt = A(t;K) tau_obs - delta_ext(t)/J1
# with delta_ext -> delta_ext + c.
#
# For fixed K, c enters linearly. Here we do a coarse log scan over K and solve c.
# This is only a calibration, not a proof.

tau_dot_obs = safe_gradient(tau_obs, t, smooth=True)

def window_mask_for_fit():
    if isinstance(K_FIT_WINDOW, tuple):
        tmin, tmax = K_FIT_WINDOW
        return (t >= tmin) & (t <= tmax)
    if K_FIT_WINDOW == "pre_peak":
        return np.arange(len(t)) <= main_peak_idx
    if K_FIT_WINDOW == "all":
        return np.ones(len(t), dtype=bool)
    raise ValueError(K_FIT_WINDOW)

fit_mask = window_mask_for_fit()
fit_mask &= np.isfinite(tau_dot_obs) & np.isfinite(J0) & np.isfinite(J1) & np.isfinite(J2)

def residual_for_K_and_offset(K):
    # tau_dot = [(J0+J2)/J1 + 1/(J1*K)] tau - (delta_ext + c)/J1
    A_no_offset = ((J0 + J2) / J1 + 1.0 / (J1 * K)) * tau_obs - delta_ext_cm_raw / J1
    # residual = tau_dot_obs - A_no_offset + c/J1
    # solve for c in least squares: tau_dot_obs - A_no_offset = -c/J1
    y = tau_dot_obs[fit_mask] - A_no_offset[fit_mask]
    x = -1.0 / J1[fit_mask]
    c = np.sum(x * y) / np.sum(x * x)
    pred = A_no_offset[fit_mask] - c / J1[fit_mask]
    rmse = np.sqrt(np.mean((tau_dot_obs[fit_mask] - pred)**2))
    return rmse, c

if K_EXT_TAU_MPA_PER_CM is None:
    # crude broad scan; adjust if needed
    K_grid = np.logspace(0, 8, 300)  # MPa/cm
    rmses = []
    offsets = []
    for K in K_grid:
        rmse, c = residual_for_K_and_offset(K)
        rmses.append(rmse)
        offsets.append(c)

    best = int(np.argmin(rmses))
    K_ext_tau = K_grid[best]
    delta_offset_cm = offsets[best]
    print(f"Fitted K_ext^tau = {K_ext_tau:.6g} MPa/cm")
    print(f"Fitted delta offset = {delta_offset_cm:.6g} cm")
else:
    K_ext_tau = float(K_EXT_TAU_MPA_PER_CM)
    _, delta_offset_cm = residual_for_K_and_offset(K_ext_tau)
    print(f"Using provided K_ext^tau = {K_ext_tau:.6g} MPa/cm")
    print(f"Fitted delta offset = {delta_offset_cm:.6g} cm")

delta_ext_cm = delta_ext_cm_raw + delta_offset_cm


# ============================================================
# ODE SOLVER FOR TAU
# ============================================================

# Smooth/interpolate coefficients for the ODE.
J0_fun = interp1d(t, J0, kind="linear", fill_value="extrapolate")
J1_fun = interp1d(t, J1, kind="linear", fill_value="extrapolate")
J2_fun = interp1d(t, J2, kind="linear", fill_value="extrapolate")
delta_fun = interp1d(t, delta_ext_cm, kind="linear", fill_value="extrapolate")

def rhs(tt, y):
    tau = y[0]
    j0 = float(J0_fun(tt))
    j1 = float(J1_fun(tt))
    j2 = float(J2_fun(tt))
    de = float(delta_fun(tt))

    return [
        ((j0 + j2) / j1 + 1.0 / (j1 * K_ext_tau)) * tau
        - de / j1
    ]

sol = solve_ivp(
    rhs,
    t_span=(t[0], t[-1]),
    y0=[tau_obs[0]],
    t_eval=t,
    method="RK45",
    rtol=1e-6,
    atol=1e-8,
)

tau_pred = np.full_like(t, np.nan, dtype=float)

if sol.t.size >= 2:
    tau_pred[:] = np.interp(t, sol.t, sol.y[0], left=np.nan, right=np.nan)

if not sol.success:
    print("ODE solver warning:", sol.message)
    print(f"Solver stopped at t={sol.t[-1]:.6f} s, requested final t={t[-1]:.6f} s")

finite = np.isfinite(tau_pred)
print(f"tau_pred finite fraction: {np.mean(finite):.3f}")

if np.any(finite):
    print(f"tau_pred finite range: {np.nanmin(tau_pred):.3g}–{np.nanmax(tau_pred):.3g} MPa")

# Avoid plotting numerical explosions.
tau_pred_plot = tau_pred.copy()
tau_pred_plot[np.abs(tau_pred_plot) > 1e4] = np.nan

# ============================================================
# COMPARE PREDICTED AND OBSERVED STRESS
# ============================================================

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(t, tau_obs, color="black", lw=2.2, label="observed shear stress")
ax.plot(t, tau_pred_plot, lw=1.8, label="ODE prediction from $J_0,J_1,J_2$")

ax.axvline(t[main_peak_idx], color="k", ls="--", lw=1.0, alpha=0.7, label="main peak")
ax.axvline(t[first_min_idx], color="k", ls=":", lw=1.2, alpha=0.7, label="onset marker")

ax.set_xlabel("Acquisition time (s)")
ax.set_ylabel("Shear stress (MPa)")
ax.set_title(f"Observed versus ODE-predicted shear stress: {mode_label}")
ax.grid(True, ls="--", alpha=0.35)
ax.legend(frameon=True)

savefig(fig, "tau_observed_vs_ODE_prediction")


# ============================================================
# ODE RESIDUAL DIAGNOSTIC USING OBSERVED TAU
# ============================================================

ode_rhs_obs = (
    ((J0 + J2) / J1 + 1.0 / (J1 * K_ext_tau)) * tau_obs
    - delta_ext_cm / J1
)

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(t, tau_dot_obs, color="black", lw=1.8, label=r"observed $\dot{\tau}$")
ax.plot(t, ode_rhs_obs, lw=1.6, label=r"ODE RHS using observed $\tau$")

ax.axvline(t[main_peak_idx], color="k", ls="--", lw=1.0, alpha=0.7)
ax.axvline(t[first_min_idx], color="k", ls=":", lw=1.2, alpha=0.7)

ax.set_xlabel("Acquisition time (s)")
ax.set_ylabel(r"$\dot{\tau}$ (MPa/s)")
ax.set_title("Differential-form test: observed stress derivative versus model RHS")
ax.grid(True, ls="--", alpha=0.35)
ax.legend(frameon=True)

savefig(fig, "tau_dot_observed_vs_ODE_rhs")


print()
print("Done.")
print(f"Outputs written in: {OUTDIR.resolve()}")