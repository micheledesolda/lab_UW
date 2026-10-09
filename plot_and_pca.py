import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks


# ============================================================
# USER PARAMETERS
# ============================================================

PKL_PATH = Path(
    "/home/michele/Desktop/Dottorato/active_source_implementation/"
    "experiments_Brava_2/s0273suw03anhdol_5050/data_analysis/"
    "test_2026-05-05_1/003_loading_spring_ss.pkl"
)

OUTDIR = Path("plots_dense")
OUTDIR.mkdir(exist_ok=True)

CMAP = "OrRd"
DPI = 300

# ============================================================
# TIME WINDOW / INITIAL CUT
# ============================================================

# Number of usable records to discard at the beginning.
# Use this to remove the unstable early inversion cloud during elastic loading.
N_SKIP_INITIAL = 0

# Optional: cut by acquisition time instead of by number of records.
# If not None, records with acquisition_time_s < T_START_S are discarded.
T_START_S = 11000

# Examples:
# N_SKIP_INITIAL = 2000
# T_START_S = 10600.0


# ============================================================
# STRESS-CYCLE DETECTION
# ============================================================

# If waveform sampling is dense, keep distance in physical seconds.
MIN_CYCLE_DISTANCE_S = 20.0
STRESS_PROMINENCE_MPA = 0.05

SAVE_PDF = True


# ============================================================
# LOAD CONCATENATED PICKLE
# ============================================================

records = []

with PKL_PATH.open("rb") as f:
    while True:
        try:
            records.append(pickle.load(f))
        except EOFError:
            break

usable = [
    r for r in records
    if all(k in r for k in [
        "gouge_1_velocity",
        "gouge_1_damping",
        "gouge_2_velocity",
        "gouge_2_damping",
        "gouge_1_eta",
        "gouge_2_eta",
        "acquisition_time_s",
        "shear_stress_MPa",
    ])
]

if len(usable) == 0:
    raise RuntimeError("No usable records found. Check the pickle keys.")


# ============================================================
# OPTIONAL INITIAL CUT
# ============================================================

n_usable_before_cut = len(usable)

if T_START_S is not None:
    usable = [
        r for r in usable
        if float(r["acquisition_time_s"]) >= float(T_START_S)
    ]

if N_SKIP_INITIAL > 0:
    usable = usable[N_SKIP_INITIAL:]

if len(usable) == 0:
    raise RuntimeError(
        "No usable records left after initial cut. "
        "Reduce N_SKIP_INITIAL or lower T_START_S."
    )

if T_START_S is None:
    CUT_TAG = f"skip{N_SKIP_INITIAL}"
else:
    CUT_TAG = f"from_{T_START_S:.1f}s_skip{N_SKIP_INITIAL}"

print()
print("Initial-cut settings:")
print(f"Usable records before cut: {n_usable_before_cut}")
print(f"N_SKIP_INITIAL = {N_SKIP_INITIAL}")
print(f"T_START_S      = {T_START_S}")
print(f"Remaining usable records after cut: {len(usable)}")
print(f"CUT_TAG        = {CUT_TAG}")


# ============================================================
# BUILD ARRAYS AFTER CUT
# ============================================================

t = np.array([r["acquisition_time_s"] for r in usable], dtype=float)
tau = np.array([r["shear_stress_MPa"] for r in usable], dtype=float)

V1 = np.vstack([np.asarray(r["gouge_1_velocity"], dtype=float) for r in usable])
D1 = np.vstack([np.asarray(r["gouge_1_damping"], dtype=float) for r in usable])
V2 = np.vstack([np.asarray(r["gouge_2_velocity"], dtype=float) for r in usable])
D2 = np.vstack([np.asarray(r["gouge_2_damping"], dtype=float) for r in usable])

eta1 = np.asarray(usable[0]["gouge_1_eta"], dtype=float)
eta2 = np.asarray(usable[0]["gouge_2_eta"], dtype=float)

print()
print(f"Total pickle objects: {len(records)}")
print(f"Usable profile records after cut: {len(usable)}")
print(f"Gouge 1 velocity shape: {V1.shape}")
print(f"Gouge 1 damping shape:  {D1.shape}")
print(f"Gouge 2 velocity shape: {V2.shape}")
print(f"Gouge 2 damping shape:  {D2.shape}")
print(f"Time range after cut:   {t.min():.3f}–{t.max():.3f} s")
print(f"Stress range after cut: {tau.min():.3f}–{tau.max():.3f} MPa")


# ============================================================
# PLOTTING STYLE
# ============================================================

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 24,
})


# ============================================================
# FIGURE SAVING
# ============================================================

def savefig(fig, stem):
    png_path = OUTDIR / f"{stem}.png"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")

    if SAVE_PDF:
        pdf_path = OUTDIR / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved {png_path}")


# ============================================================
# VELOCITY / DAMPING PROFILES VS TIME + STRESS
# ============================================================

def plot_profiles_for_gouge(gouge_label, eta, V, D, stem):
    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 11),
        sharex=True,
        constrained_layout=True,
    )

    panels = [
        (V, f"{gouge_label}: velocity evolution + shear stress", "Velocity (cm/µs)"),
        (D, f"{gouge_label}: damping evolution + shear stress", "Damping (cm²/µs)"),
    ]

    for ax, (arr, title, cbar_label) in zip(axes, panels):
        im = ax.pcolormesh(
            t,
            eta,
            arr.T,
            shading="auto",
            cmap=CMAP,
        )

        ax.set_title(title, pad=12)
        ax.set_ylabel("Normalized gouge coordinate η")
        ax.grid(False, linestyle="--", linewidth=0.8, alpha=0.35)

        cbar = fig.colorbar(im, ax=ax, pad=0.012)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=14)

        ax2 = ax.twinx()
        ax2.plot(t, tau, color="black", linewidth=2.2, label="Shear stress")
        ax2.set_ylabel("Shear stress (MPa)")
        ax2.set_ylim(tau.min() - 0.2, tau.max() + 0.2)
        ax2.tick_params(axis="y", labelsize=15)

    axes[-1].set_xlabel("Acquisition time (s)")

    savefig(fig, stem)


plot_profiles_for_gouge(
    "Gouge 1",
    eta1,
    V1,
    D1,
    f"gouge1_velocity_damping_vs_stress_OrRd_{CUT_TAG}",
)

plot_profiles_for_gouge(
    "Gouge 2",
    eta2,
    V2,
    D2,
    f"gouge2_velocity_damping_vs_stress_OrRd_{CUT_TAG}",
)


# ============================================================
# PCA OF FULL VISCOELASTIC STATE
# ============================================================

# Full state vector:
# [G1 velocity, G1 damping, G2 velocity, G2 damping]
X = np.hstack([V1, D1, V2, D2])

scaler = StandardScaler()
Xz = scaler.fit_transform(X)

pca = PCA(n_components=3)
PC = pca.fit_transform(Xz)

print()
print("PCA explained variance after cut:")
print(f"PC1: {pca.explained_variance_ratio_[0] * 100:.2f} %")
print(f"PC2: {pca.explained_variance_ratio_[1] * 100:.2f} %")
print(f"PC3: {pca.explained_variance_ratio_[2] * 100:.2f} %")


# ============================================================
# STRESS MARKERS AFTER CUT
# ============================================================

main_peak_idx = int(np.argmax(tau))

dt_med = float(np.median(np.diff(t)))
distance_samples = max(20, int(MIN_CYCLE_DISTANCE_S / dt_med))

post = np.arange(main_peak_idx + 1, len(tau))

mins_rel, _ = find_peaks(
    -tau[post],
    prominence=STRESS_PROMINENCE_MPA,
    distance=distance_samples,
)

maxs_rel, _ = find_peaks(
    tau[post],
    prominence=STRESS_PROMINENCE_MPA,
    distance=distance_samples,
)

mins = post[mins_rel]
maxs = post[maxs_rel]

if len(mins) == 0:
    raise RuntimeError(
        "No post-peak stress minima found. "
        "Try lowering STRESS_PROMINENCE_MPA or MIN_CYCLE_DISTANCE_S."
    )

first_min_idx = int(mins[0])

pre = np.arange(0, main_peak_idx + 1)
transition = np.arange(main_peak_idx, first_min_idx + 1)
stickslip = np.arange(first_min_idx, len(t))

ss_mins = mins
ss_maxs = maxs[maxs > first_min_idx]

print()
print(f"Median dt after cut: {dt_med:.5f} s")
print(f"Cycle peak detection distance: {distance_samples} samples")
print(f"Main stress peak: t={t[main_peak_idx]:.3f} s, tau={tau[main_peak_idx]:.3f} MPa")
print(f"Onset marker / first post-peak minimum: t={t[first_min_idx]:.3f} s, tau={tau[first_min_idx]:.3f} MPa")
print(f"Stress maxima marked: {len(ss_maxs)}")
print(f"Stress minima marked: {len(ss_mins)}")


# ============================================================
# CONFERENCE-STYLE PCA PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(12.5, 9.5))

# Continuity line
ax.plot(
    PC[:, 0],
    PC[:, 1],
    color="0.72",
    linewidth=1.1,
    alpha=0.60,
    zorder=1,
)

# Phase clouds
ax.scatter(
    PC[pre, 0],
    PC[pre, 1],
    s=12,
    c="#2c7fb8",
    edgecolor="none",
    alpha=0.70,
    label="Elastic loading up to main peak",
    zorder=2,
)

ax.scatter(
    PC[transition, 0],
    PC[transition, 1],
    s=22,
    c="#fdae61",
    edgecolor="none",
    alpha=0.85,
    label="Post-peak transition",
    zorder=3,
)

ax.scatter(
    PC[stickslip, 0],
    PC[stickslip, 1],
    s=12,
    c="#1a9850",
    edgecolor="none",
    alpha=0.70,
    label="Stick-slip regime",
    zorder=2,
)

# Main stress peak
ax.scatter(
    PC[main_peak_idx, 0],
    PC[main_peak_idx, 1],
    s=420,
    marker="*",
    c="#d7191c",
    edgecolor="black",
    linewidth=1.0,
    label="Main stress peak",
    zorder=6,
)

# Onset marker
ax.scatter(
    PC[first_min_idx, 0],
    PC[first_min_idx, 1],
    s=210,
    marker="D",
    c="#984ea3",
    edgecolor="black",
    linewidth=0.9,
    label="Onset of stick-slip",
    zorder=6,
)

# Stick-slip stress maxima
if len(ss_maxs) > 0:
    ax.scatter(
        PC[ss_maxs, 0],
        PC[ss_maxs, 1],
        s=140,
        marker="^",
        c="#e41a1c",
        edgecolor="black",
        linewidth=0.8,
        label="Stick-slip stress maxima",
        zorder=5,
    )

# Stick-slip stress minima
if len(ss_mins) > 0:
    ax.scatter(
        PC[ss_mins, 0],
        PC[ss_mins, 1],
        s=140,
        marker="v",
        c="#377eb8",
        edgecolor="black",
        linewidth=0.8,
        label="Stick-slip stress minima",
        zorder=5,
    )

# Direction arrows in post-peak transition
step = max(1, len(transition) // 7)

for j in transition[::step]:
    if j + 1 <= transition[-1]:
        ax.annotate(
            "",
            xy=(PC[j + 1, 0], PC[j + 1, 1]),
            xytext=(PC[j, 0], PC[j, 1]),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                lw=1.6,
                alpha=0.75,
            ),
            zorder=7,
        )

# Direct labels
ax.text(
    np.median(PC[pre, 0]),
    np.median(PC[pre, 1]),
    "elastic\nloading",
    fontsize=18,
    weight="bold",
    color="#1f5f8b",
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#2c7fb8",
        alpha=0.85,
    ),
    zorder=8,
)

ax.text(
    np.mean(PC[transition, 0]),
    np.mean(PC[transition, 1]),
    "post-peak\ntransition",
    fontsize=18,
    weight="bold",
    color="#9b4d00",
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#fdae61",
        alpha=0.88,
    ),
    zorder=8,
)

ax.text(
    np.median(PC[stickslip, 0]),
    np.median(PC[stickslip, 1]),
    "stick-slip\ncycles",
    fontsize=18,
    weight="bold",
    color="#136d37",
    ha="center",
    va="center",
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#1a9850",
        alpha=0.86,
    ),
    zorder=8,
)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
ax.set_title("Viscoelastic-state trajectory from velocity and damping profiles")

ax.grid(False, linestyle="--", linewidth=0.8, alpha=0.35)

for spine in ax.spines.values():
    spine.set_linewidth(1.4)

legend = ax.legend(
    loc="best",
    frameon=True,
    framealpha=0.94,
    borderpad=0.8,
    labelspacing=0.6,
    handletextpad=0.6,
)

legend.get_frame().set_edgecolor("0.25")
legend.get_frame().set_linewidth(1.0)

fig.tight_layout()
savefig(fig, f"pca_viscoelastic_state_trajectory_{CUT_TAG}")


print()
print("Done.")
print(f"Output directory: {OUTDIR.resolve()}")