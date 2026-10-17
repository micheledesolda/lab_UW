"""Selection of mechanical records; no forward or objective changes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def preprocessing_waveform_count(params):
    """With selection, sample only after matching the original acquisition times."""
    if params.get("local_restart", {}).get("enabled", False):
        return None
    if params.get("mechanical_selection", {}).get("enabled", False):
        return None
    return params.get("num_waveform2process")


def validate_polarization(wave_type, settings):
    if wave_type not in ("_s", "_p"):
        raise ValueError("wave_type deve essere '_s' oppure '_p'.")
    result = settings[wave_type]
    for key in ("velocity_values_cm_us", "impedance_velocity_values_cm_us"):
        values = np.asarray(result[key], dtype=float)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"Griglia {key} non valida per {wave_type}.")
    for key in ("damping_values_cm2_us", "impedance_damping_values_cm2_us"):
        values = np.asarray(result[key], dtype=float)
        if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"Griglia {key} non valida per {wave_type}.")
    return result


def _interval(value):
    if len(value) != 2:
        raise ValueError("L'intervallo meccanico deve contenere inizio e fine, in secondi.")
    start, end = map(float, value)
    if not np.isfinite(start) or not np.isfinite(end) or start >= end:
        raise ValueError("L'intervallo deve avere tempi finiti e inizio < fine.")
    return start, end


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def choose_interval_graphically(mechanical, paired):
    """Use an embedded Tk canvas, independently of the orchestration's Agg backend."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        root = tk.Tk()
    except Exception as exc:
        # Tk raises TclError when DISPLAY is absent. Do not silently select all data.
        raise RuntimeError(
            "Selezione grafica non disponibile. Installa python3-tk/usa un desktop, "
            "oppure imposta mechanical_selection['time_interval_s']=(inizio_s, fine_s)."
        ) from exc
    root.title("Intervallo meccanico da invertire")
    figure = Figure(figsize=(11, 6), constrained_layout=True)
    axis = figure.add_subplot(111)
    mechanical = mechanical.sort_values("time_s")
    times = mechanical["time_s"].to_numpy(dtype=float)
    if not len(times) or not np.all(np.isfinite(times)):
        root.destroy()
        raise ValueError("Il dato meccanico non contiene un asse time_s valido.")
    stresses = [name for name in ("shear_stress_MPa", "normal_stress_MPa") if name in mechanical]
    for name in stresses:
        axis.plot(times, mechanical[name], label=name, linewidth=1)
    axis.set(xlabel="Tempo dell'esperimento [s]", ylabel="Stress [MPa]",
             title="Due click: inizio e fine. Oppure scrivi i tempi e premi Conferma.")
    axis.grid(alpha=.25)
    if stresses:
        axis.legend(loc="upper left")
    displacement = next((name for name in ("lp_disp_mm", "ec_disp_mm", "disp_mm", "slip_mm") if name in mechanical), None)
    if displacement:
        other = axis.twinx()
        other.plot(times, mechanical[displacement], color="tab:green", alpha=.6)
        other.set_ylabel(displacement + " [mm]")
    matched_times = paired["uw_time_s"].to_numpy(dtype=float)
    axis.plot(matched_times, np.full(len(matched_times), .02), "|", color="black",
              transform=axis.get_xaxis_transform(), markersize=5)
    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    toolbar = NavigationToolbar2Tk(canvas, root)
    toolbar.update()
    controls = tk.Frame(root)
    controls.pack(fill=tk.X)
    tk.Label(controls, text="Inizio [s]").pack(side=tk.LEFT)
    start_var = tk.StringVar(value=f"{times[0]:.9g}")
    tk.Entry(controls, textvariable=start_var, width=18).pack(side=tk.LEFT)
    tk.Label(controls, text="Fine [s]").pack(side=tk.LEFT)
    end_var = tk.StringVar(value=f"{times[-1]:.9g}")
    tk.Entry(controls, textvariable=end_var, width=18).pack(side=tk.LEFT)
    status = tk.StringVar(value="La fascia selezionata comprende entrambi gli estremi.")
    tk.Label(root, textvariable=status).pack()
    state = {"result": None, "clicks": [], "patch": None}

    def highlight():
        try:
            a, b = sorted((float(start_var.get()), float(end_var.get())))
        except ValueError:
            return
        if state["patch"] is not None:
            state["patch"].remove()
        state["patch"] = axis.axvspan(a, b, color="orange", alpha=.2)
        canvas.draw_idle()

    def on_click(event):
        if event.xdata is None or event.inaxes is None or event.button != 1 or toolbar.mode:
            return
        if len(state["clicks"]) == 2:
            state["clicks"] = []
        state["clicks"].append(float(event.xdata))
        if len(state["clicks"]) == 1:
            start_var.set(f"{state['clicks'][0]:.9g}")
            status.set("Ora scegli la fine dell'intervallo.")
        else:
            a, b = sorted(state["clicks"])
            start_var.set(f"{a:.9g}"); end_var.set(f"{b:.9g}")
            status.set("Premi Conferma per salvare e avviare la selezione.")
            highlight()

    def confirm():
        try:
            value = _interval((start_var.get(), end_var.get()))
            if value[0] < times[0] or value[1] > times[-1]:
                raise ValueError("Scegli tempi compresi nel tratto meccanico visualizzato.")
            in_range = (matched_times >= value[0]) & (matched_times <= value[1])
            if not np.any(in_range & paired["time_s"].notna().to_numpy()):
                raise ValueError("Nell'intervallo non ci sono onde associate al dato meccanico.")
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Intervallo non valido", str(exc), parent=root)
            return
        state["result"] = value
        root.destroy()

    def reset():
        state["clicks"] = []
        start_var.set(f"{times[0]:.9g}"); end_var.set(f"{times[-1]:.9g}")
        highlight()

    tk.Button(controls, text="Mostra intervallo", command=highlight).pack(side=tk.LEFT)
    tk.Button(controls, text="Intero tratto", command=reset).pack(side=tk.LEFT)
    tk.Button(controls, text="Conferma", command=confirm).pack(side=tk.RIGHT)
    canvas.mpl_connect("button_press_event", on_click)
    highlight()
    root.mainloop()
    if state["result"] is None:
        raise RuntimeError("Selezione annullata: nessuna inversione avviata.")
    return state["result"]


def resolve_mechanical_interval(mechanical, paired, settings, chooser=None):
    """Resolve once per execution and persist absolute mechanical seconds for P and S."""
    if "_resolved_interval_s" in settings:
        return _interval(settings["_resolved_interval_s"])
    path = Path(settings["selection_file"])
    identity = settings["experiment_key"]
    explicit = settings.get("time_interval_s")
    if explicit is not None:
        interval = _interval(explicit)
    elif path.exists() and not settings.get("reselect", False):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != 1 or payload.get("experiment_key") != identity:
            raise ValueError(f"Selezione salvata incompatibile con questo esperimento: {path}")
        interval = _interval(payload["time_interval_s"])
    else:
        interval = _interval((chooser or choose_interval_graphically)(mechanical, paired))
    _write_json(path, {"schema": 1, "experiment_key": identity,
                       "time_interval_s": list(interval), "time_units": "s",
                       "bounds": "inclusive"})
    settings["_resolved_interval_s"] = list(interval)
    return interval


def select_mechanical_records(paired, mechanical, params, infile_path, chooser=None):
    """Filter already paired rows; keep original uw_idx, mechanical Index and times."""
    settings = params.get("mechanical_selection", {})
    if not settings.get("enabled", False):
        return paired
    start, end = resolve_mechanical_interval(mechanical, paired, settings, chooser=chooser)
    valid = (paired["uw_time_s"] >= start) & (paired["uw_time_s"] <= end) & paired["time_s"].notna()
    if "Index" in paired:
        valid &= paired["Index"].notna()
    selected = paired.loc[valid].copy()
    count = params.get("num_waveform2process")
    if count is not None:
        if isinstance(count, bool) or int(count) != count or int(count) < 1:
            raise ValueError("num_waveform2process deve essere None oppure un intero positivo.")
        if len(selected) > int(count):
            positions = np.linspace(0, len(selected) - 1, int(count), dtype=int)
            selected = selected.iloc[positions].copy()
    print(f"Intervallo meccanico [{start:g}, {end:g}] s: "
          f"{int(valid.sum())} onde disponibili, {len(selected)} selezionate in {Path(infile_path).name}.", flush=True)
    if selected.empty:
        return selected
    # Never renumber the source records: continuation/resume relies on them.
    selected = selected.reset_index(drop=True)
    values = selected[["uw_idx", "uw_time_s", "Index"]].to_numpy(dtype=float)
    digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()
    stem = Path(infile_path).name.split(".")[0]
    output = Path(params["outdir_path_l2norm"]) / stem
    manifest = output.with_suffix(".mechanical_selection.json")
    signature = {"schema": 1, "experiment_key": settings["experiment_key"],
                 "input_name": Path(infile_path).name, "time_interval_s": [start, end],
                 "selected_records_sha256": digest, "number_of_selected_waveforms": len(selected)}
    results = output.with_suffix(".pkl")
    if params.get("resume_from_pickle", False) and results.exists() and results.stat().st_size:
        if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != signature:
            raise ValueError("Il pickle esistente usa una selezione diversa o non documentata. "
                             "Per questa selezione usa una nuova cartella di output.")
    _write_json(manifest, signature)
    return selected
