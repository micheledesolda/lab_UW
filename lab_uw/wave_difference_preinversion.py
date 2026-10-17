"""Compare selected successive recorded waves before any parameter search."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np


def _noise_window(time, signal_mask, settings):
    if not settings.get("noise_enabled", True):
        return None, "disabled"
    explicit = settings.get("noise_window_us")
    if explicit is None:
        end = float(settings.get("noise_end_us", 10.))
        if not np.isfinite(end) or end < time[0] or end > time[-1]:
            raise ValueError("noise_end_us esterno all'asse temporale.")
        # Equal sample count on the same uniform grid gives equal duration.
        if not np.allclose(np.diff(time), np.diff(time)[0], rtol=1e-7, atol=1e-12):
            raise ValueError("Per un asse non uniforme specifica noise_window_us.")
        last = int(np.searchsorted(time, end, side="right")-1)
        first = last-int(np.count_nonzero(signal_mask))+1
        if first < 0:
            raise ValueError("Non c'è spazio prima di noise_end_us per una finestra della stessa durata del segnale.")
        mask = np.zeros(len(time), dtype=bool); mask[first:last+1] = True
        origin = "same number of samples as signal window, ending at noise_end_us"
    else:
        start, stop = map(float, explicit)
        if not np.isfinite(start+stop) or start >= stop or start < time[0] or stop > time[-1]:
            raise ValueError("noise_window_us: estremi non validi o esterni al segnale.")
        mask = (time >= start) & (time <= stop)
        origin = "explicit noise_window_us"
    if np.count_nonzero(mask) < 3 or time[mask][-1] >= time[signal_mask][0]:
        raise ValueError("La finestra di rumore deve avere almeno 3 campioni e precedere interamente l'arrivo diretto.")
    return mask, origin


def noise_metrics(a, b, signal_mask, noise_mask, signal_weights, noise_weights, objective, gamma, taper):
    names = ("J_noise_relative", "noise_relative_l2", "signal_delta_energy", "signal_energy_previous",
             "noise_delta_energy", "noise_energy_previous", "noise_energy_current",
             "J_noise_equivalent_signal_L2", "signal_minus_noise_L2", "difference_power_ratio")
    result = dict.fromkeys(names)
    result["noise_status"] = "disabled"
    if noise_mask is None:
        return result
    if not all(np.all(np.isfinite(x[m])) for x in (a,b) for m in (signal_mask,noise_mask)):
        result["noise_status"] = "nonfinite_signal_or_noise"
        return result
    s0, s1 = signal_weights*a[signal_mask], signal_weights*b[signal_mask]
    n0, n1 = noise_weights*a[noise_mask], noise_weights*b[noise_mask]
    es, en0, en1 = float(s0@s0), float(n0@n0), float(n1@n1)
    ds, dn = s1-s0, n1-n0
    es_delta, en_delta = float(ds@ds), float(dn@dn)
    result.update(signal_delta_energy=es_delta,signal_energy_previous=es,
                  noise_delta_energy=en_delta,noise_energy_previous=en0,noise_energy_current=en1)
    if es <= 1e-30 or en0 <= 1e-30 or en1 <= 1e-30:
        result["noise_status"] = "zero_signal_or_noise_energy"
        return result
    raw = objective(a,b,noise_mask,spectral_gamma=gamma,taper_fraction=taper)
    factor = float(signal_weights@signal_weights)/float(noise_weights@noise_weights)
    equivalent = factor*en_delta/es
    result.update(J_noise_relative=float(raw["total"]),noise_relative_l2=float(raw["relative_l2"]),
                  J_noise_equivalent_signal_L2=equivalent,signal_minus_noise_L2=es_delta/es-equivalent,
                  difference_power_ratio=None if en_delta <= 1e-30 else es_delta/(factor*en_delta),
                  noise_status="ok")
    return result


def _window(functions, rows, time, stf, params, assembly, settings):
    explicit = settings.get("window_us")
    if explicit is not None:
        start, stop = map(float, explicit)
        if not np.isfinite(start+stop) or start >= stop or start < time[0] or stop > time[-1]:
            raise ValueError("wave_difference_diagnostic['window_us']: estremi non validi o esterni al segnale.")
        mask = (time >= start) & (time <= stop)
        origin = "explicit window_us"
    else:
        # This is the same initial direct-arrival window used by the original
        # global search. Geometry is taken from the first selected acquisition.
        reference_assembly = deepcopy(assembly)
        functions["update_assembly_dict_with_mech_data"](reference_assembly, rows[0])
        windows = functions["predict_event_windows"](
            reference_assembly, params["velocity_initial_list"],
            wave_type=reference_assembly["wave_type"],
        )
        stf_time = np.asarray(stf.metadata["time_ax_waveform"], dtype=float)
        mask = functions["direct_misfit_interval"](
            time, arrival_min_us=windows["direct"][0], arrival_max_us=windows["direct"][1],
            stf_duration_us=float(stf_time[-1]-stf_time[0]),
            padding_us=params["direct_window_padding_us"],
        )
        mask = np.asarray(functions["sample_interval_mask"](mask, len(time)), dtype=bool)
        origin = "initial global-search direct-arrival window of first selected acquisition"
    if mask.shape != time.shape or np.count_nonzero(mask) < 3:
        raise ValueError("Finestra delle differenze troppo corta o incompatibile con il dato.")
    return mask, origin


def _record(row):
    for name in ("records_na", "Index"):
        value = getattr(row, name, np.nan)
        if np.isfinite(value):
            return int(value)
    return None


def compare_selected(functions, *, paired, observed_waveform_data, waveform_metadata,
                     stf_handler, params, assembly_dict, infile_path):
    """Return True only when diagnostic-only mode requests skipping inversion."""
    settings = params.get("wave_difference_diagnostic", {})
    if not settings.get("enabled", False):
        return False
    time = np.asarray(waveform_metadata["time_ax_waveform"], dtype=float)
    data = np.asarray(observed_waveform_data)
    if time.ndim != 1 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("Asse temporale delle onde non valido.")
    if data.ndim != 2 or data.shape[1] != len(time):
        raise ValueError("Le onde devono avere forma (numero_onde, numero_campioni).")
    if not {"uw_idx", "uw_time_s"}.issubset(paired.columns):
        raise ValueError("Manca l'associazione temporale fra onde e dato meccanico.")
    ordered = paired.sort_values(["uw_time_s", "uw_idx"], kind="stable")
    rows = list(ordered.itertuples(index=False))
    if not rows:
        print("Differenze fra onde: nessuna onda selezionata.", flush=True)
        return bool(settings.get("only", False))
    indices = [int(row.uw_idx) for row in rows]
    if len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= len(data):
        raise ValueError("Indici delle onde duplicati o esterni ai dati.")
    acquisition_times = np.asarray([row.uw_time_s for row in rows], dtype=float)
    if not np.all(np.isfinite(acquisition_times)) or np.any(np.diff(acquisition_times) <= 0):
        raise ValueError("Tempi di acquisizione duplicati o non crescenti.")
    mask, origin = _window(functions, rows, time, stf_handler, params, assembly_dict, settings)
    noise_mask, noise_origin = _noise_window(time, mask, settings)
    gamma = settings.get("spectral_gamma")
    gamma = float(params.get("local_spectral_gamma", 1.) if gamma is None else gamma)
    taper = settings.get("taper_fraction")
    taper = float(params.get("local_objective_taper_fraction", 0.) if taper is None else taper)
    if not np.isfinite(gamma) or gamma < 0 or not np.isfinite(taper) or not 0 <= taper <= 1:
        raise ValueError("Gamma o taper non validi per il confronto fra onde.")
    fields = ["previous_rec_n", "current_rec_n", "previous_uw_idx", "current_uw_idx",
              "previous_time_s", "current_time_s", "delta_time_s", "J_between_successive",
              "relative_l2", "amplitude_misfit", "spectral_shape_misfit", "phase_misfit", "status",
              "J_noise_relative", "noise_relative_l2", "signal_delta_energy", "signal_energy_previous",
              "noise_delta_energy", "noise_energy_previous", "noise_energy_current",
              "J_noise_equivalent_signal_L2", "signal_minus_noise_L2", "difference_power_ratio", "noise_status"]
    results = []
    from scipy.signal.windows import tukey
    weights = tukey(np.count_nonzero(mask), alpha=taper) if taper else np.ones(np.count_nonzero(mask))
    noise_weights = None if noise_mask is None else (tukey(np.count_nonzero(noise_mask), alpha=taper) if taper else np.ones(np.count_nonzero(noise_mask)))
    # Reuse the user's objective, with no fitted amplitude scale and no alignment.
    objective = functions["relative_waveform_objective"]
    for position in range(1, len(rows)):
        previous, current = rows[position-1], rows[position]
        a = np.asarray(data[int(previous.uw_idx)], dtype=float)
        b = np.asarray(data[int(current.uw_idx)], dtype=float)
        status = "ok"
        if not np.all(np.isfinite(a[mask])) or not np.all(np.isfinite(b[mask])):
            values = {}; status = "nonfinite_data_in_window"
        else:
            # The forward function falls back to absolute energy for zero data.
            # Such a value would not be a relative distance between observations.
            if np.linalg.norm(weights*a[mask]) <= 1e-15:
                values = {}; status = "zero_previous_wave_energy"
            else:
                values = objective(a, b, mask, spectral_gamma=gamma, taper_fraction=taper)
        noise = noise_metrics(a,b,mask,noise_mask,weights,noise_weights,objective,gamma,taper)
        results.append({"previous_rec_n": _record(previous), "current_rec_n": _record(current),
            "previous_uw_idx": int(previous.uw_idx), "current_uw_idx": int(current.uw_idx),
            "previous_time_s": float(previous.uw_time_s), "current_time_s": float(current.uw_time_s),
            "delta_time_s": float(current.uw_time_s-previous.uw_time_s),
            "J_between_successive": values.get("total"), "relative_l2": values.get("relative_l2"),
            "amplitude_misfit": values.get("amplitude_misfit"), "spectral_shape_misfit": values.get("spectral_shape_misfit"),
            "phase_misfit": values.get("phase_misfit"), "status": status, **noise})
        if position % 2000 == 0:
            print(f"Differenze fra onde: {position}/{len(rows)-1} coppie.", flush=True)
    output = Path(params["outdir_path_l2norm"])/"wave_differences"
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(infile_path).name.split(".")[0]
    # Keep earlier diagnostic results when parameters or selections change.
    prefix = output/stem
    counter = 1
    while prefix.with_name(prefix.name+"_successive.png").exists() or prefix.with_suffix(".csv").exists() or prefix.with_suffix(".json").exists():
        prefix = output/f"{stem}_{counter:03d}"
        counter += 1
    with prefix.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    summary = {"schema": 2, "input_name": Path(infile_path).name, "wave_type": assembly_dict.get("wave_type"),
        "selected_waveforms": len(rows), "compared_pairs": len(results),
        "valid_pairs": sum(r["status"] == "ok" for r in results),
        "time_interval_s": [float(acquisition_times[0]), float(acquisition_times[-1])],
        "window_us": [float(time[mask][0]), float(time[mask][-1])], "window_origin": origin,
        "window_indices": np.flatnonzero(mask).tolist(), "spectral_gamma": gamma, "taper_fraction": taper,
        "formula": "J_gamma(d_previous, d_current), using the original relative_waveform_objective",
        "normalization": "previous waveform energy; original gamma-dependent objective; no amplitude refit",
        "time_coordinate": "uw_time_s synchronized with mechanical experiment; value assigned to the later waveform",
        "selection": "all and only selected records, before SNR filtering and before worker chains",
        "diagnostic_only": bool(settings.get("only", False)),
        "adaptive_inversion_implemented": bool(params.get("adaptive_inversion", {}).get("enabled", False))}
    summary["noise"] = {"window_us": None if noise_mask is None else [float(time[noise_mask][0]),float(time[noise_mask][-1])],
        "window_origin": noise_origin,"window_indices": [] if noise_mask is None else np.flatnonzero(noise_mask).tolist(),
        "signal_samples": int(np.count_nonzero(mask)),"noise_samples": 0 if noise_mask is None else int(np.count_nonzero(noise_mask)),
        "squared_weight_ratio": None if noise_mask is None else float(weights@weights)/float(noise_weights@noise_weights),
        "valid_pairs": sum(r["noise_status"]=="ok" for r in results),
        "equivalent_noise_formula": "(||wS||²/||wN||²) * ||wN*(n_current-n_previous)||² / ||wS*d_previous||²",
        "interpretation": "Estimated L2 contribution if noise difference power is comparable before and during the direct arrival. This is not a significance threshold or an exact noise subtraction. Filtering may correlate samples and waveforms.",
        "gamma_note": "Equivalent noise is an L2 energy diagnostic; it is directly comparable to J_gamma only for gamma=1."}
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    _plot(results, time, data[indices[0]], data[indices[-1]], mask, gamma, assembly_dict.get("wave_type"), prefix, noise_mask)
    print(f"Differenze fra onde: {len(rows)} onde, {len(results)} coppie; gamma={gamma:g}; "
          f"finestra [{time[mask][0]:g}, {time[mask][-1]:g}] µs.\nGrafico: {prefix.name}_successive.png\nCartella: {output}", flush=True)
    return bool(settings.get("only", False))


def _plot(results, time, first, last, mask, gamma, wave_type, prefix, noise_mask=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    ax.plot([r["current_time_s"] for r in results],
            [np.nan if r["J_between_successive"] is None else r["J_between_successive"] for r in results], lw=.9)
    if not results:
        ax.text(.5, .5, "Serve almeno una coppia di onde selezionate", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Tempo dell'esperimento [s]")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.set_ylabel(r"Misfit fra onde consecutive, $J_\gamma$")
    label = str(wave_type or "").replace("_", "").upper()
    ax.set_title(f"Onde {label}: differenze fra acquisizioni consecutive — γ={gamma:g}")
    ax.grid(alpha=.25)
    fig.savefig(prefix.with_name(prefix.name+"_successive.png"), dpi=180)
    plt.close(fig)
    if noise_mask is not None:
        fig, axes = plt.subplots(2,1,figsize=(12,7.5),sharex=True,constrained_layout=True)
        acquisition=[r["current_time_s"] for r in results]
        def series(key):
            return [np.nan if r[key] is None else r[key] for r in results]
        axes[0].plot(acquisition,series("relative_l2"),lw=.8,label="Differenza nell'arrivo diretto, L2")
        axes[0].plot(acquisition,series("J_noise_equivalent_signal_L2"),lw=.8,alpha=.85,label="Rumore riportato all'energia del segnale, L2")
        axes[0].set_ylabel("Differenza / energia del segnale"); axes[0].set_title("Confronto fra variazioni del segnale e del rumore")
        axes[1].plot(acquisition,series("J_noise_relative"),lw=.8,color="tab:orange",label=f"Norma relativa nella sola finestra di rumore, γ={gamma:g}")
        axes[1].set_ylabel("Differenza / energia del rumore");axes[1].set_xlabel("Tempo dell'esperimento [s]")
        for ax in axes:
            ax.ticklabel_format(axis="x",style="plain",useOffset=False);ax.grid(alpha=.25);ax.legend(fontsize=9)
        fig.savefig(prefix.with_name(prefix.name+"_noise_comparison.png"),dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    ax.plot(time, first, label="Prima onda selezionata", lw=1.)
    ax.plot(time, last, label="Ultima onda selezionata", lw=1., alpha=.8)
    ax.axvspan(time[mask][0], time[mask][-1], color="tab:green", alpha=.15, label="Finestra comune del confronto")
    if noise_mask is not None:
        ax.axvspan(time[noise_mask][0], time[noise_mask][-1], color="tab:orange", alpha=.15, label="Finestra di rumore")
    ax.set_xlabel("Tempo nella forma d'onda [µs]"); ax.set_ylabel("Ampiezza del dato preprocessato")
    ax.set_title("Controllo della finestra dell'arrivo diretto"); ax.grid(alpha=.25); ax.legend()
    fig.savefig(prefix.with_name(prefix.name+"_window.png"), dpi=160)
    plt.close(fig)
