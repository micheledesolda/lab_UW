#!/usr/bin/env python3
"""Mappe FWI e differenze consecutive, dai risultati locali senza nuove inversioni.

Dipendenze: numpy, matplotlib. Eseguire --help per gli esempi.
Le unita native dei pickle lab_uw sono cm/us e cm^2/us: l'uscita e in SI.
--interpolate-time aggiunge differenze stimate interpolando linearmente i profili
fra le valutazioni effettive, ai tempi originali delle acquisizioni.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gzip
from io import BytesIO, TextIOWrapper
import json
import math
from pathlib import Path
import pickle
import sys
import warnings
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter


# PARAMETRI GRAFICI ESPLICITI. Non entrano nelle inversioni o nelle sottrazioni.
# Limiti assoluti condivisi; differenze: None = scala propria per ogni pannello.
# Le scale automatiche comprendono tutti gli estremi, senza clipping o percentili.
COLORS = {
    "velocity_absolute": [300.0, 2600.0],
    "damping_absolute_P": [0.0, 0.12],
    "damping_absolute_S": [0.0, 0.024],
    "velocity_difference_limit": None,
    "velocity_consecutive_linear_core": None,
    "velocity_first_linear_core": None,
    "damping_difference_limit_P": None,
    "damping_difference_limit_S": None,
    "damping_consecutive_linear_core_P": None,
    "damping_consecutive_linear_core_S": None,
    "damping_first_linear_core_P": None,
    "damping_first_linear_core_S": None,
}
BLUE, ORANGE, TEXT = "#304fa6", "#d05b28", "#25334a"
MAP_NAMES = {
    "absolute": ("profili_assoluti", "Inverted profiles"),
    "first": ("variazioni_dal_primo", "Changes from the first profile"),
    "consecutive": ("differenze_onda_onda", "Differences between consecutive saved records"),
    "evaluated": ("differenze_inversioni", "Differences between consecutive evaluated profiles"),
    "interpolated": ("differenze_onda_onda_interpolate", "Differences from linearly interpolated profiles"),
}
DEFAULT_MAPS = ("absolute", "first", "consecutive", "evaluated")
DIFFERENCE_MAPS = ("consecutive", "evaluated", "interpolated")


def number(value, default=np.nan):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def save_npz(path, **arrays):
    """Completa l'archivio in memoria prima di scriverlo, anche su dischi sincronizzati."""
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    path.write_bytes(buffer.getvalue())


def stream_rows(path):
    """Legge uno alla volta i pickle concatenati, anche se compressi gzip."""
    with path.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        while stream.peek(1):
            try:
                obj = pickle.load(stream)
            except ModuleNotFoundError as exc:
                raise ValueError(
                    f"Il pickle richiede il modulo {exc.name!r}. Usa lo stesso ambiente "
                    "dell'inversione, oppure --project-root /percorso/lab_UW. "
                    "I file .analysis.pkl.gz e i cache NPZ normalmente non lo richiedono."
                ) from exc
            except (EOFError, pickle.UnpicklingError) as exc:
                raise ValueError(
                    f"Pickle incompleto o non leggibile: {path}. "
                    "Se la run e ancora attiva, usa una copia del file dopo un salvataggio completo."
                ) from exc
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, (list, tuple)):
                yield from (row for row in obj if isinstance(row, dict))


def native_profile(row, gouge):
    prefix = f"gouge_{gouge}"
    c = row.get(prefix + "_velocity")
    nu = row.get(prefix + "_damping")
    xi = row.get(prefix + "_eta")  # eta qui e la coordinata normalizzata!
    if c is None or nu is None:
        indices = row.get("idx_dict", {}).get(prefix)
        if indices is None:
            return None
        c = np.asarray(row["velocity_profile"])[indices]
        nu = np.asarray(row["damping_profile"])[indices]
        x = np.asarray(row["x_axis_cm"])[indices]
        xi = (x - x[0]) / (x[-1] - x[0])
    c, nu = np.asarray(c, float).reshape(-1), np.asarray(nu, float).reshape(-1)
    if xi is None:
        indices = row.get("idx_dict", {}).get(prefix)
        if indices is None or row.get("x_axis_cm") is None:
            raise ValueError(f"Manca la coordinata spaziale di {prefix}, rec={row.get('rec_n')}.")
        x = np.asarray(row["x_axis_cm"])[indices]
        xi = (x - x[0]) / (x[-1] - x[0])
    xi = np.asarray(xi, float).reshape(-1)
    if not (len(xi) == len(c) == len(nu) and len(xi) >= 2):
        raise ValueError(f"Dimensioni non valide per {prefix}, rec={row.get('rec_n')}.")
    if (not all(np.isfinite(a).all() for a in (xi, c, nu))
            or not np.all(np.diff(xi) > 0)
            or abs(xi[0]) > 1e-10 or abs(xi[-1] - 1) > 1e-10):
        raise ValueError(f"Coordinate o valori non validi per {prefix}, rec={row.get('rec_n')}.")
    return xi.copy(), c.copy() * 1e4, nu.copy() * 100.0


def read_results(path, gouges):
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as f:
            stored_gouge = int(f["gouge_index"]) if "gouge_index" in f else 1
            if gouges != [stored_gouge]:
                raise ValueError(f"Il cache {path.name} contiene soltanto gouge {stored_gouge}; "
                                 f"usa --gouge {stored_gouge}.")
            d = {k: f[k].copy() for k in f.files}
        if not {"record", "time_s", "xi", "velocity_m_s", "damping_m2_s"} <= d.keys():
            raise ValueError(f"Cache NPZ non riconosciuto: {path}")
        n = len(d["record"])
        d.setdefault("shear_stress_MPa", np.full(n, np.nan))
        d.setdefault("ob_disp_mm", np.full(n, np.nan))
        d.setdefault("thickness_mm", np.full(n, np.nan))
        d.setdefault("misfit", np.full(n, np.nan))
        d.setdefault("evaluation_known", np.full(n, "forward_evaluated" in d))
        d.setdefault("forward_evaluated", np.zeros(n, bool))
        # Non usare displacement_mm dei vecchi cache: conteneva ec_disp_mm.
        return {stored_gouge: d}

    rows = {g: {} for g in gouges}
    duplicates = {g: 0 for g in gouges}
    for count, row in enumerate(stream_rows(path), 1):
        if "global_best_misfit" in row or "rec_n" not in row:
            continue
        rec = int(row["rec_n"])
        for g in gouges:
            native = native_profile(row, g)
            if native is None:
                continue
            if rec in rows[g]:
                duplicates[g] += 1
            rows[g][rec] = {
                "record": rec,
                "time_s": number(row.get("acquisition_time_s")),
                "shear_stress_MPa": number(row.get("shear_stress_MPa")),
                "ob_disp_mm": number(row.get("ob_disp_mm", row.get("ob_disp"))),
                "thickness_mm": number(row.get(f"gouge_thickness_{g}_cm",
                                             row.get(f"gouge_{g}_thickness_cm"))) * 10,
                "misfit": number(row.get("misfit")),
                "evaluation_known": row.get("forward_evaluated") is not None,
                "forward_evaluated": bool(row.get("forward_evaluated", False)),
                "native": native,
            }
        if count % 5000 == 0:
            print(f"  {path.name}: letti {count} oggetti", flush=True)

    result = {}
    for g in gouges:
        ordered = sorted(rows[g].values(), key=lambda r: r["time_s"])
        if not ordered:
            raise ValueError(f"Nessun profilo del gouge {g} in {path}.")
        d = {key: np.asarray([r[key] for r in ordered])
             for key in ordered[0] if key != "native"}
        xi = ordered[0]["native"][0]
        c_rows, nu_rows, regridded = [], [], []
        native_offsets, native_x, native_c, native_nu = [0], [], [], []
        for r in ordered:
            x, c, nu = r["native"]
            native_x.append(x); native_c.append(c); native_nu.append(nu)
            native_offsets.append(native_offsets[-1] + len(x))
            same = len(x) == len(xi) and np.allclose(x, xi, rtol=0, atol=1e-12)
            c_rows.append(c if same else np.interp(xi, x, c))
            nu_rows.append(nu if same else np.interp(xi, x, nu))
            if not same:
                regridded.append(r["record"])
        d.update(xi=xi.copy(), velocity_m_s=np.stack(c_rows), damping_m2_s=np.stack(nu_rows),
                 gouge_index=np.array(g), native_offsets=np.asarray(native_offsets),
                 native_xi_flat=np.concatenate(native_x),
                 native_velocity_m_s_flat=np.concatenate(native_c),
                 native_damping_m2_s_flat=np.concatenate(native_nu))
        meta = {"source": str(path.resolve()), "source_size_bytes": path.stat().st_size,
                "duplicate_records_kept_last": duplicates[g],
                "comparison_grid": "normalized z/h nodes of the first saved profile",
                "regridded_records": regridded,
                "native_units": {"velocity": "cm/us", "damping": "cm^2/us"},
                "output_units": {"velocity": "m/s", "damping": "m^2/s"}}
        d["source_metadata_json"] = np.array(json.dumps(meta))
        if regridded:
            warnings.warn(f"Gouge {g}: {len(regridded)} profili con griglia diversa. "
                          "Confronto interpolato sullo z/h del primo profilo; nodi nativi conservati nel NPZ.")
        result[g] = d
    return result


def select_interval(d, tmin, tmax):
    n = len(d["record"])
    xi = np.asarray(d["xi"])
    if (xi.ndim != 1 or len(xi) < 2 or not np.isfinite(xi).all()
            or not np.all(np.diff(xi) > 0) or abs(xi[0]) > 1e-10 or abs(xi[-1]-1) > 1e-10):
        raise ValueError("La griglia di confronto deve crescere da z/h=0 a z/h=1.")
    for key in ("velocity_m_s", "damping_m2_s"):
        if np.shape(d[key]) != (n, len(xi)) or not np.isfinite(d[key]).all():
            raise ValueError(f"Forma o valori non validi per {key}.")
    if (not np.isfinite(d["time_s"]).all() or not np.all(np.diff(d["time_s"]) > 0)):
        raise ValueError("I tempi dei profili devono essere finiti e strettamente crescenti.")
    keep = np.ones(n, bool)
    if tmin is not None:
        keep &= d["time_s"] >= tmin
    if tmax is not None:
        keep &= d["time_s"] <= tmax
    if keep.sum() < 2:
        raise ValueError("Occorrono almeno due profili nell'intervallo richiesto.")
    out = dict(d)
    for key in ("record", "time_s", "shear_stress_MPa", "ob_disp_mm", "thickness_mm",
                "misfit", "evaluation_known", "forward_evaluated", "velocity_m_s", "damping_m2_s"):
        out[key] = np.asarray(d[key])[keep].copy()
    if "native_offsets" in d:
        offsets = d["native_offsets"]
        ids = np.flatnonzero(keep)
        out["native_offsets"] = np.r_[0, np.cumsum([offsets[i+1]-offsets[i] for i in ids])]
        for key in ("native_xi_flat", "native_velocity_m_s_flat", "native_damping_m2_s_flat"):
            out[key] = np.concatenate([d[key][offsets[i]:offsets[i+1]] for i in ids])
    return out


@contextmanager
def mechanical_stream(path, member=None):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if not n.endswith("/")
                     and not n.startswith("__MACOSX/")]
            if member is None:
                if len(names) != 1:
                    raise ValueError("ZIP meccanico con piu file: usa --mechanical-member. "
                                     f"Contenuto: {names}")
                member = names[0]
            with archive.open(member) as raw:
                with TextIOWrapper(raw, encoding="utf-8-sig") as stream:
                    yield stream
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            yield stream


def attach_mechanics(datasets, path, column, factor, member):
    wanted = {int(r) for d in datasets for r in d["record"]}
    found = {}
    with mechanical_stream(path, member) as stream:
        header_line = stream.readline()
        separator = "\t" if "\t" in header_line else (";" if ";" in header_line else ",")
        header = [s.strip() for s in next(csv.reader([header_line], delimiter=separator))]
        rec_name = next((s for s in ("records_na", "rec_n", "record") if s in header), None)
        time_name = next((s for s in ("time_s", "acquisition_time_s") if s in header), None)
        if rec_name is None or time_name is None or column not in header:
            raise ValueError(f"Colonne richieste nel meccanico: record, tempo e {column!r}. "
                             f"Colonne presenti: {header}")
        ir, it, idisp = [header.index(s) for s in (rec_name, time_name, column)]
        for line_no, values in enumerate(csv.reader(stream, delimiter=separator), 2):
            if not values or not any(s.strip() for s in values):
                continue
            try:
                rec_value = float(values[ir])
                rec = int(rec_value)
                if rec != rec_value:
                    raise ValueError("record non intero")
            except (ValueError, IndexError):
                if line_no == 2:  # riga delle unita nel file meccanico originale
                    continue
                raise ValueError(f"Record meccanico non valido alla riga {line_no}.")
            if rec not in wanted:
                continue
            value = (float(values[it]), float(values[idisp]) * factor)
            if rec in found and found[rec] != value:
                raise ValueError(f"Record meccanico duplicato con valori discordanti: {rec}")
            found[rec] = value
            if len(found) == len(wanted):
                break
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Mancano {len(missing)} record nel file meccanico; primi: {sorted(missing)[:8]}")
    for d in datasets:
        times = np.asarray([found[int(r)][0] for r in d["record"]])
        if not np.allclose(times, d["time_s"], rtol=0, atol=1e-8):
            raise ValueError("Tempi meccanici e tempi dei profili discordanti agli stessi record. "
                             "Non viene applicato alcuno shift automatico.")
        d["ob_disp_mm"] = np.asarray([found[int(r)][1] for r in d["record"]])
        d["displacement_source_json"] = np.array(json.dumps({
            "file": str(path.resolve()), "column": column, "factor_to_mm": factor,
            "matching": "exact record number; times checked with absolute tolerance 1e-8 s"}))


def changed_rows(d):
    return (np.any(np.diff(d["velocity_m_s"], axis=0) != 0, axis=1)
            | np.any(np.diff(d["damping_m2_s"], axis=0) != 0, axis=1))


def difference_data(d, kind):
    if kind == "interpolated":
        return interpolated_difference_data(d)
    ids = np.arange(len(d["record"]))
    if kind == "evaluated":
        if not np.all(d["evaluation_known"]):
            return None, "forward_evaluated manca in alcuni record"
        if np.any(changed_rows(d) & ~d["forward_evaluated"][1:]):
            return None, "esistono cambiamenti di modello in record marcati forward_evaluated=False"
        ids = np.flatnonzero(d["forward_evaluated"])
        ids = np.unique(np.r_[0, ids])  # includi il modello iniziale come riferimento
    if len(ids) < 2:
        return None, "meno di due profili valutati"
    c, nu = d["velocity_m_s"][ids], d["damping_m2_s"][ids]
    out = {"time_edges_s": d["time_s"][ids], "xi": d["xi"],
           "previous_record": d["record"][ids[:-1]], "current_record": d["record"][ids[1:]],
           "previous_time_s": d["time_s"][ids[:-1]], "current_time_s": d["time_s"][ids[1:]],
           "interval_duration_s": np.diff(d["time_s"][ids]),
           "delta_velocity_m_s": np.diff(c, axis=0), "delta_damping_m2_s": np.diff(nu, axis=0),
           "current_misfit": d["misfit"][ids[1:]],
           "current_forward_evaluated": d["forward_evaluated"][ids[1:]]}
    for key, original in (("delta_velocity_m_s", c), ("delta_damping_m2_s", nu)):
        if not np.allclose(out[key].sum(axis=0), original[-1]-original[0], rtol=1e-10, atol=1e-10):
            raise ValueError("Verifica della somma telescopica delle differenze fallita.")
    return out, None


def interpolated_difference_data(d):
    """Ipotesi lineare sui profili; differenze sulla cadenza originale, senza estrapolare.

    Non si interpolano gli impulsi Delta_i della mappa originale: si interpolano
    c(z/h,t) e nu(z/h,t) fra gli anchor valutati (piu il riferimento iniziale).
    La differenza di due tempi interni allo stesso intervallo vale quindi
    Delta_anchor * Delta_t_acquisizione / Delta_t_anchor.
    """
    evaluated, reason = difference_data(d, "evaluated")
    if evaluated is None:
        return None, reason
    anchor_time = evaluated["time_edges_s"]
    anchor_ids = np.searchsorted(d["time_s"], anchor_time)
    # L'ultimo anchor limita il dominio: nessun prolungamento artificiale in coda.
    interpolated = select_interval(d, float(anchor_time[0]), float(anchor_time[-1]))
    sample_time = interpolated["time_s"]
    for key in ("velocity_m_s", "damping_m2_s"):
        anchors = d[key][anchor_ids]
        values = np.column_stack([np.interp(sample_time, anchor_time, column)
                                  for column in anchors.T])
        positions = np.searchsorted(sample_time, anchor_time)
        if not np.array_equal(values[positions], anchors):
            raise ValueError("L'interpolazione non conserva esattamente i profili agli anchor.")
        interpolated[key] = values
    out, reason = difference_data(interpolated, "consecutive")
    if out is None:
        return None, reason
    # Il misfit originale non valuta i profili intermedi appena stimati.
    out["source_current_misfit"] = out["current_misfit"].copy()
    out["current_misfit"] = np.where(out["current_forward_evaluated"], out["current_misfit"], np.nan)
    out.update(anchor_record=d["record"][anchor_ids].copy(), anchor_time_s=anchor_time.copy(),
               anchor_velocity_m_s=d["velocity_m_s"][anchor_ids].copy(),
               anchor_damping_m2_s=d["damping_m2_s"][anchor_ids].copy(),
               interpolated_profile_time_s=sample_time.copy(),
               interpolated_velocity_m_s=interpolated["velocity_m_s"],
               interpolated_damping_m2_s=interpolated["damping_m2_s"],
               interpolation_method_json=np.array(json.dumps({
                   "method": "linear interpolation of profiles at each original normalized z/h node",
                   "anchors": "first retained profile plus forward_evaluated=True; no anchor removed for zero change",
                   "output_times": "original saved acquisition times within the anchor interval",
                   "differences": "interpolated profile at t_i minus interpolated profile at t_(i-1); not a time derivative",
                   "within_one_anchor_interval": "Delta_profile_anchor * Delta_t_acquisition / Delta_t_anchor",
                   "extrapolation": False, "new_forward_evaluations": False,
                   "interpretation": "assumed gradual evolution between anchors; not independently inverted intermediate profiles",
                   "source_records": int(len(d["record"])),
                   "trailing_records_outside_anchor_interval": int(np.sum(d["time_s"] > anchor_time[-1])),
               })))
    return out, None


def difference_summary(delta):
    result = {"pairs": len(delta["current_record"]),
              "dt_min_s": float(delta["interval_duration_s"].min()),
              "dt_max_s": float(delta["interval_duration_s"].max())}
    for key in ("delta_velocity_m_s", "delta_damping_m2_s"):
        a = delta[key]
        per_pair = np.max(np.abs(a), axis=1)
        result[key] = {"min": float(a.min()), "max": float(a.max()),
                       "max_abs": float(per_pair.max()),
                       "median_pair_max_abs": float(np.median(per_pair)),
                       "p95_pair_max_abs": float(np.quantile(per_pair, .95)),
                       "nonzero_pairs": int(np.count_nonzero(np.any(a != 0, axis=1)))}
    return result


def write_diagnostic_csv(path, delta):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["previous_record", "current_record", "previous_time_s", "current_time_s",
                         "dt_s", "max_abs_delta_velocity_m_s", "nodal_rms_delta_velocity_m_s",
                         "max_abs_delta_damping_m2_s", "nodal_rms_delta_damping_m2_s", "current_misfit"])
        for i in range(len(delta["current_record"])):
            dc, dn = delta["delta_velocity_m_s"][i], delta["delta_damping_m2_s"][i]
            writer.writerow([int(delta["previous_record"][i]), int(delta["current_record"][i]),
                             delta["previous_time_s"][i], delta["current_time_s"][i],
                             delta["interval_duration_s"][i], np.abs(dc).max(), np.sqrt(np.mean(dc**2)),
                             np.abs(dn).max(), np.sqrt(np.mean(dn**2)), delta["current_misfit"][i]])


def nice_up(x):
    if x == 0:
        return 0.0
    step = 10.0 ** (math.floor(math.log10(abs(x))) - 1)
    # Evita un intervallo in piu per rumore floating point su un limite gia tondo.
    rounded = math.ceil(x / step - 1e-12) * step
    return max(x, rounded)


def norm_for(values, wave, field, kind, settings, linear, difference_decades=4.0):
    minimum, maximum = float(values.min()), float(values.max())
    if kind == "absolute":
        key = "velocity_absolute" if field == "velocity" else "damping_absolute_" + wave
        low, high = settings[key]
        low, high = min(low, -nice_up(-minimum)), max(high, nice_up(maximum))
        if high <= low:
            high = low + 1
        return Normalize(low, high), {"norm": "linear", "limits": [low, high]}
    if field == "velocity":
        limit = settings["velocity_difference_limit"]
        core = settings["velocity_first_linear_core" if kind == "first" else "velocity_consecutive_linear_core"]
    else:
        limit = settings["damping_difference_limit_" + wave]
        core = settings[("damping_first_linear_core_" if kind == "first" else "damping_consecutive_linear_core_") + wave]
    configured_limit = limit
    actual_max = max(abs(minimum), abs(maximum))
    limit = max(0.0 if limit is None else float(limit), nice_up(actual_max))
    if actual_max == 0 and limit == 0:
        limit = 1.0  # soltanto per definire la barra di una mappa tutta nulla
    configured_core = core
    if core is None:
        core = limit / (10.0 ** difference_decades)
    if limit <= 0 or core <= 0:
        raise ValueError("Limiti e intervalli lineari delle scale colore devono essere positivi.")
    norm = Normalize(-limit, limit) if linear else SymLogNorm(
        linthresh=core, linscale=1, vmin=-limit, vmax=limit, base=10)
    return norm, {"norm": "linear" if linear else "symmetric_log_base10",
                  "limits": [-limit, limit], "linear_core": None if linear else [-core, core],
                  "limit_rule": "own panel maximum absolute difference, rounded upward" if configured_limit is None else "explicit colour configuration, expanded if needed",
                  "linear_core_rule": "limit / 10**difference_decades" if configured_core is None else "explicit colour configuration",
                  "difference_decades": difference_decades, "all_values_zero": actual_max == 0}


def common_colour_settings(models, settings):
    """Stessa scala di velocita per P/S e gouge diversi nella stessa esecuzione."""
    result = dict(settings)
    all_c = [d["velocity_m_s"] for d in models.values()]
    low, high = result["velocity_absolute"]
    result["velocity_absolute"] = [min(low, -nice_up(-min(a.min() for a in all_c))),
                                   max(high, nice_up(max(a.max() for a in all_c)))]
    for wave in ("P", "S"):
        arrays = [d["damping_m2_s"] for (w, g), d in models.items() if w == wave]
        if not arrays:
            continue
        low, high = result["damping_absolute_"+wave]
        result["damping_absolute_"+wave] = [min(low, -nice_up(-min(a.min() for a in arrays))),
                                           max(high, nice_up(max(a.max() for a in arrays)))]
    return result


def merge_equal_cells(edges, values):
    first = np.r_[True, np.any(values[1:] != values[:-1], axis=1)]
    starts = np.flatnonzero(first)
    return np.r_[edges[starts], edges[-1]], values[starts]


def mechanical_panel(ax, d):
    t = d["time_s"]
    handles = []
    if np.isfinite(d["shear_stress_MPa"]).any():
        handles += ax.plot(t, d["shear_stress_MPa"], color=BLUE, lw=1.25, label="Shear stress")
    ax.set_ylabel(r"Shear stress $\tau$ (MPa)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    if np.isfinite(d["ob_disp_mm"]).all():
        right = ax.twinx()
        handles += right.plot(t, d["ob_disp_mm"]-d["ob_disp_mm"][0], color=ORANGE,
                              lw=1.25, label="Shear displacement")
        right.set_ylabel("Displacement (mm)", color=ORANGE, labelpad=10)
        right.tick_params(axis="y", colors=ORANGE)
        right.margins(y=.08)
    ax.set_title("(a) Mechanical response", loc="left", pad=10)
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.035),
                  ncol=2, frameon=False, fontsize=10, borderaxespad=0)
    ax.set_xlim(t[0], t[-1])
    ax.grid(axis="x", color="#dfe5ed", lw=.6)
    ax.margins(y=.1)
    ax.tick_params(labelbottom=False)


def make_figure(d, wave, gouge, kind, delta, args, settings):
    signed = kind != "absolute"
    if kind in DIFFERENCE_MAPS:
        edges = delta["time_edges_s"]
        arrays = [delta["delta_velocity_m_s"], delta["delta_damping_m2_s"]]
        note = f"{len(arrays[0])} differences; no division by time interval"
        if kind == "interpolated":
            note = "Linear profile interpolation between evaluated records; changes per original acquisition interval"
    else:
        edges = d["time_s"]
        arrays = [d["velocity_m_s"], d["damping_m2_s"]]
        if kind == "first":
            arrays = [a-a[0] for a in arrays]
        note = (f"Reference: record {int(d['record'][0])}, t = {d['time_s'][0]:.3f} s"
                if kind == "first" else "Profiles held until the next acquisition; colours fixed in time")
    fig = plt.figure(figsize=(13.8, 10.2))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, .027], height_ratios=[.85, 1.25, 1.25],
                            left=.095, right=.86, top=.87, bottom=.10, hspace=.33, wspace=.20)
    top = fig.add_subplot(grid[0, 0])
    mechanical_panel(top, d)
    fig.text(.095, .975, f"Gouge {gouge} — {wave} waves: {MAP_NAMES[kind][1]}",
             fontsize=18, fontweight="bold", va="top", color=TEXT)
    fig.text(.095, .932, note, fontsize=12, color=TEXT)
    meta = {}
    for i, (field, values, unit) in enumerate(zip(
            ("velocity", "damping"), arrays, (r"m s$^{-1}$", r"m$^2$ s$^{-1}$")), 1):
        norm, norm_meta = norm_for(values, wave, field, kind, settings, args.linear_differences,
                                  args.difference_decades)
        # Nessuna durata inventata per l'ultimo profilo: e un valore di bordo.
        cell_values = values if kind in DIFFERENCE_MAPS else values[:-1]
        cell_edges, shown = merge_equal_cells(edges, cell_values)
        xi = d["xi"]
        if args.space_samples:
            display_xi = np.linspace(0, 1, args.space_samples)
            shown = np.stack([np.interp(display_xi, xi, a) for a in shown])
        else:
            display_xi = xi
        yedges = np.r_[0, (display_xi[:-1]+display_xi[1:])/2, 1]
        ax = fig.add_subplot(grid[i, 0], sharex=top)
        if kind == "interpolated":
            ax.set_facecolor("#e4e8ee")  # grigio fuori dal dominio degli anchor; bianco = zero
        cax = fig.add_subplot(grid[i, 1])
        mesh = ax.pcolormesh(cell_edges, yedges, shown.T,
                             cmap="RdBu_r" if signed else "turbo", norm=norm,
                             shading="flat", linewidth=0, edgecolors="none",
                             antialiased=False, rasterized=True)
        symbol = "c" if field == "velocity" else r"\nu"
        if kind == "interpolated":
            symbol = rf"\widetilde{{{symbol}}}"
        expression = (rf"${symbol}$" if not signed else
                      (rf"${symbol}(t)-{symbol}(t_0)$" if kind == "first"
                       else rf"${symbol}_i-{symbol}_{{i-1}}$"))
        cb = fig.colorbar(mesh, cax=cax)
        cb.set_label(expression + " (" + unit + ")", labelpad=12)
        if signed:
            limit = norm_meta["limits"][1]
            if norm_meta["norm"] == "linear":
                ticks = np.linspace(-limit, limit, 7)
            else:
                core = norm_meta["linear_core"][1]
                positive = [core]
                while positive[-1]*10 < limit:
                    positive.append(positive[-1]*10)
                if positive[-1] < limit:
                    positive.append(limit)
                ticks = [-x for x in positive[::-1]] + [0] + positive
                if len(ticks) > 13:
                    ticks = ticks[::2] + [0, -limit, limit]
                    ticks = sorted(set(ticks))
            cb.set_ticks(ticks)
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
            cb.ax.minorticks_off()
            if norm_meta.get("all_values_zero", False):
                cb.set_ticks([0])
        cb.ax.tick_params(labelsize=10)
        ax.set(ylim=(0, 1), ylabel=r"Position $z/h$")
        ax.set_yticks([0, .25, .5, .75, 1])
        title = f"({chr(97+i)}) {'Velocity' if field == 'velocity' else 'Damping'}"
        if signed and norm_meta.get("all_values_zero", False):
            title += " — all differences are zero"
        ax.set_title(title, loc="left", pad=10)
        ax.xaxis.set_major_locator(MaxNLocator(7))
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        if i == 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Acquisition time (s)", labelpad=9)
        meta[field] = {**norm_meta, "data_min": float(values.min()), "data_max": float(values.max()),
                       "rendered_time_cells": len(cell_values), "merged_time_cells": len(shown)}
    rendering = ("Original comparison nodes" if not args.space_samples
                 else f"Linear spatial display interpolation: {args.space_samples} positions")
    colour_note = ("Blue: decrease; white: zero; red: increase" if signed else "Values in physical units")
    footer = rendering + ". " + colour_note + "."
    if signed:
        footer += "\nEach difference panel has its own colour scale; exact zeros remain white."
    if kind == "interpolated":
        footer += " Grey: outside anchor coverage. Linear evolution assumed; no extrapolation."
    fig.text(.095, .022 if signed else .032, footer, fontsize=10, color=TEXT)
    name = f"gouge{gouge}_{wave}_{MAP_NAMES[kind][0]}"
    outputs = []
    for ext in args.formats:
        path = args.output / f"{name}.{ext}"
        buffer = BytesIO()
        fig.savefig(buffer, format=ext, dpi=args.dpi, bbox_inches="tight", pad_inches=.15)
        path.write_bytes(buffer.getvalue())
        outputs.append(path.name)
    plt.close(fig)
    return {"outputs": outputs, "colour_scales": meta,
            **({"interpolation": json.loads(str(delta["interpolation_method_json"]))}
               if kind == "interpolated" else {})}


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Esempio: python mappe_profili_fwi.py --p P.analysis.pkl.gz --s S.analysis.pkl.gz "
               "--mechanical dati_meccanici.zip --output mappe_nuove")
    p.add_argument("--p", type=Path, help="risultati P: .pkl, .pkl.gz oppure cache NPZ")
    p.add_argument("--s", type=Path, help="risultati S: .pkl, .pkl.gz oppure cache NPZ")
    p.add_argument("--gouge", choices=["1", "2", "both"], default="1")
    p.add_argument("--mechanical", type=Path, help="file meccanico originale TSV/CSV o ZIP; facoltativo")
    p.add_argument("--mechanical-member", help="nome del membro se lo ZIP contiene piu file")
    p.add_argument("--displacement-column", default="ob_disp")
    p.add_argument("--displacement-to-mm", type=float, default=1.0,
                   help="fattore del solo displacement verso mm; default 1, come concordato per ob_disp")
    p.add_argument("--output", type=Path, default=Path("mappe_FWI"))
    p.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf", "svg"])
    p.add_argument("--maps", nargs="+", choices=list(MAP_NAMES), default=list(DEFAULT_MAPS))
    p.add_argument("--interpolate-time", action="store_true",
                   help="aggiunge una mappa separata: profili interpolati linearmente fra valutazioni effettive, "
                        "poi differenze ai tempi originali di acquisizione; nessuna estrapolazione")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--space-samples", type=int, default=0,
                   help="0: celle sui nodi di confronto; >=2: interpolazione spaziale solo grafica")
    p.add_argument("--linear-differences", action="store_true", help="colori lineari anziche logaritmici simmetrici")
    p.add_argument("--difference-decades", type=float, default=4.,
                   help="solo colori: tratto lineare +/-L/10**n, con L=massimo della singola mappa; default n=4")
    p.add_argument("--color-config", type=Path, help="JSON che sovrascrive i parametri COLORS espliciti nello script")
    p.add_argument("--tmin", type=float, help="inizio facoltativo, secondi assoluti di acquisizione")
    p.add_argument("--tmax", type=float, help="fine facoltativa, secondi assoluti di acquisizione")
    p.add_argument("--project-root", type=Path, help="directory contenente lab_uw, per vecchi pickle con oggetti")
    p.add_argument("--no-plots", action="store_true", help="estrae solo NPZ, differenze, CSV e riepilogo JSON")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not (args.p or args.s):
        p.error("indica almeno --p oppure --s")
    if args.interpolate_time and "interpolated" not in args.maps:
        args.maps.append("interpolated")
    if args.space_samples == 1 or args.space_samples < 0 or args.dpi <= 0:
        p.error("--space-samples deve essere 0 oppure >=2; --dpi deve essere positivo")
    if not np.isfinite(args.difference_decades) or not 0 < args.difference_decades <= 12:
        p.error("--difference-decades deve essere finito e compreso fra 0 (escluso) e 12")
    if args.tmin is not None and args.tmax is not None and args.tmin >= args.tmax:
        p.error("--tmin deve essere minore di --tmax")
    if not np.isfinite(args.displacement_to_mm) or args.displacement_to_mm == 0:
        p.error("--displacement-to-mm deve essere finito e diverso da zero")
    if args.project_root:
        sys.path.insert(0, str(args.project_root.resolve()))
    settings = dict(COLORS)
    if args.color_config:
        settings.update(json.loads(args.color_config.read_text(encoding="utf-8")))
    gouges = [1, 2] if args.gouge == "both" else [int(args.gouge)]
    models = {}
    for wave, path in (("P", args.p), ("S", args.s)):
        if path is not None:
            print(f"Lettura {wave}: {path}", flush=True)
            for g, data in read_results(path, gouges).items():
                models[(wave, g)] = select_interval(data, args.tmin, args.tmax)
    if args.mechanical:
        attach_mechanics(list(models.values()), args.mechanical, args.displacement_column,
                         args.displacement_to_mm, args.mechanical_member)
    settings = common_colour_settings(models, settings)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 13,
                         "axes.titlesize": 14, "axes.titleweight": "bold", "axes.edgecolor": "#78869a",
                         "axes.linewidth": .9, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    report = {"method": {"units": "m/s, m^2/s, mm, MPa, s",
                          "differences": "current minus previous at equal normalized z/h; no division by dt",
                          "consecutive": "all consecutive saved records, including repeated models and zero changes",
                          "evaluated": "forward_evaluated=True plus first saved model; omitted if flags absent/inconsistent",
                          "interpolated": "optional separate map: linearly interpolate profiles at each z/h between evaluated anchors, "
                                          "then subtract at consecutive original acquisition times; assumed evolution, no extrapolation",
                          "time_display_absolute": "profile i on [t_i,t_(i+1)]; final profile is a zero-duration boundary",
                          "time_display_differences": "delta_i on [t_(i-1),t_i]; no extrapolation after last evaluated pair",
                          "reference_for_first": "first record in the requested interval",
                          "space_samples_display_only": args.space_samples,
                          "displacement": "ob_disp in mm minus first retained value; no fallback to ec_disp",
                          "colour_limits": "absolute maps share limits; each difference panel has its own symmetric range, including all extremes; no clipping",
                          "difference_colours": "SymLog, linear core +/-L/10**difference_decades; affects colours only",
                          "difference_decades": args.difference_decades},
              "requested_colour_settings": settings, "datasets": {}}
    for (wave, g), d in models.items():
        tag = f"gouge{g}_{wave}"
        if not np.isfinite(d["ob_disp_mm"]).all():
            print(f"  {tag}: LVDT non disponibile; grafico del solo stress. "
                  "Per aggiungerlo passa --mechanical.", flush=True)
        cache = args.output / f"{tag}_profili_compatti.npz"
        save_npz(cache, **d)
        info = {"records": len(d["record"]), "comparison_nodes": len(d["xi"]),
                "record_range": [int(d["record"][0]), int(d["record"][-1])],
                "time_range_s": [float(d["time_s"][0]), float(d["time_s"][-1])],
                "evaluation_flags_known": bool(np.all(d["evaluation_known"])),
                "evaluated_records": int(np.sum(d["forward_evaluated"])),
                "cache": cache.name, "cache_size_bytes": cache.stat().st_size,
                "lvdt_available": bool(np.isfinite(d["ob_disp_mm"]).all()), "maps": {}}
        if "source_metadata_json" in d:
            info["source"] = json.loads(str(d["source_metadata_json"]))
        for kind in args.maps:
            delta = None
            if kind in DIFFERENCE_MAPS:
                delta, reason = difference_data(d, kind)
                if delta is None:
                    info["maps"][kind] = {"omitted": reason}
                    print(f"  {tag}, {kind}: omessa ({reason}).", flush=True)
                    continue
                stem = f"{tag}_{MAP_NAMES[kind][0]}"
                save_npz(args.output / f"{stem}.npz", **delta)
                write_diagnostic_csv(args.output / f"{stem}_diagnostica.csv", delta)
                info["maps"][kind] = {"diagnostics": difference_summary(delta)}
                if kind == "interpolated":
                    info["maps"][kind]["interpolation"] = json.loads(str(delta["interpolation_method_json"]))
            else:
                info["maps"][kind] = {}
            if not args.no_plots:
                print(f"  Disegno {tag}: {kind}", flush=True)
                info["maps"][kind].update(make_figure(d, wave, g, kind, delta, args, settings))
        report["datasets"][tag] = info
        print(f"  {tag}: {len(d['record'])} record; cache {cache.stat().st_size/1e6:.2f} MB", flush=True)
    (args.output / "riepilogo_mappe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    (args.output / "colori_usati.json").write_text(json.dumps(settings, indent=2)+"\n", encoding="utf-8")
    print(f"Fatto: {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        raise SystemExit(2)
