#!/usr/bin/env python3
"""Confronta dati e simulazioni SALVATE e disegna il residuo simulazione-dato.

Estrazione nel progetto originale:
  python crea_mappa_simulazioni.py --main local_optimization_FWI_velocity_1D.py \
      --results /percorso/005_reload_ss_1_P.pkl --output mappe_P

Ridisegno senza progetto o pickle grande:
  python crea_mappa_simulazioni.py --input-npz tracce_per_confronto.npz --output mappe

Per iniziare le mappe da un tempo assoluto di acquisizione: --tmin 9299.095
Il ritaglio riguarda solo gli assi delle figure; cache, CSV e scale colore restano completi.

Nessun nuovo forward, inversione, fit di ampiezza o riallineamento delle onde.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import warnings

import numpy as np


class ExportCompleted(Exception):
    pass


def pickle_rows(path):
    import pickle
    with path.open('rb') as probe:
        compressed = probe.read(2) == b'\x1f\x8b'
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as stream:
        while stream.peek(1):
            position = stream.tell()
            try:
                row = pickle.load(stream)
            except Exception as exc:
                raise ValueError(f'Pickle incompleto/non leggibile al byte {position}: {exc}') from exc
            if isinstance(row, dict):
                yield row
            elif isinstance(row, (list, tuple)):
                yield from (r for r in row if isinstance(r, dict))
            else:
                raise ValueError(f'Oggetto non riconosciuto al byte {position}: {type(row).__name__}.')


def is_global(row):
    return ((row.get('_analysis_export') or {}).get('record_kind') == 'global_reference'
            or 'global_best_misfit' in row)


def is_inverted(row):
    """Una valutazione del forward basta; non occorre un aggiornamento accettato."""
    if is_global(row):
        return False
    if row.get('forward_evaluated') is not None:
        return bool(row['forward_evaluated'])
    decision = row.get('adaptive_inversion') or {}
    if decision.get('action') in ('reused', 'reuse', 'skipped', 'skip'):
        return False
    if 'optimized' in decision:
        return bool(decision['optimized'])
    return bool(row.get('local_iteration_history') or
                (row.get('local_iteration_history_summary') or {}).get('accepted_entries', 0))


def checkpoint_value(row, name):
    state = row.get('local_continuation_state') or {}
    attributes = state.get('simulation_attributes') or {}
    for mapping in (attributes, state, row):
        value = mapping.get(name)
        if value is not None:
            return value
    # Checkpoint estesi: gli input possono essere raggruppati in sottodizionari.
    for path, key, value in checkpoint_fields(row):
        if key == name and value is not None:
            return value
    return None


def checkpoint_fields(row):
    """Cerca metadati osservativi, escludendo configurazioni e sorgenti di riferimento."""
    excluded = {'source_calibration', '_kinematic_reference', '_kinematic_cache',
                'analysis_assembly', 'analysis_runtime', 'params', 'configuration',
                'source', 'stf', 'source_handler'}
    seen = set()
    def walk(mapping, path, depth):
        if not isinstance(mapping, dict) or id(mapping) in seen or depth > 8:
            return
        seen.add(id(mapping))
        for key, value in mapping.items():
            if key in excluded:
                continue
            name = f'{path}.{key}' if path else str(key)
            yield name, key, value
            if isinstance(value, dict):
                yield from walk(value, name, depth+1)
    yield from walk(row, '', 0)


def saved_window_specs(row):
    specs = []
    for path, key, value in checkpoint_fields(row):
        if value is None:
            continue
        if key in {'misfit_interval', 'misfit_mask', 'misfit_sample_indices',
                   'misfit_indices', 'misfit_interval_indices', 'misfit_interval_mask'}:
            specs.append(('samples', value, path))
        elif key in {'misfit_interval_us', 'misfit_window_us', 'misfit_window_bounds_us'}:
            specs.append(('microseconds', value, path))
    return specs


def mask_from_spec(value, size, acoustic_time, unit):
    if unit == 'microseconds':
        bounds = np.asarray(value, float).reshape(-1)
        if len(bounds) != 2:
            raise ValueError('Estremi della finestra del misfit non riconosciuti.')
        return window_mask(acoustic_time, bounds)
    if isinstance(value, dict):
        if 'indices' in value:
            return mask_from_spec(value['indices'], size, acoustic_time, 'samples')
        if 'mask' in value:
            return mask_from_spec(value['mask'], size, acoustic_time, 'samples')
        if 'start' in value or 'stop' in value:
            value = slice(value.get('start'), value.get('stop'), value.get('step'))
        else:
            raise ValueError('Codifica della finestra del misfit non riconosciuta.')
    mask = np.zeros(size, bool)
    if isinstance(value, slice):
        mask[value] = True
    else:
        values = np.asarray(value)
        if values.dtype.kind == 'b':
            if values.shape != (size,):
                raise ValueError('Maschera del misfit di lunghezza diversa dal dato.')
            mask = values.copy()
        else:
            flat = values.reshape(-1)
            if flat.size == 0 or not np.isfinite(flat).all() or np.any(flat != flat.astype(int)):
                raise ValueError('Indici del misfit non validi.')
            indices = flat.astype(int)
            if indices.min() < 0 or indices.max() >= size:
                raise ValueError('Indici del misfit fuori dalla traccia.')
            mask[indices] = True
    if not mask.any():
        raise ValueError('Finestra del misfit vuota.')
    return mask


def recover_misfit_masks(selected, acoustic_time):
    masks, paths, missing = [], [], []
    for row in selected:
        specs = row.get('misfit_window_specs', [])
        if not specs:
            missing.append(int(row['rec_n']))
            continue
        decoded = [(mask_from_spec(value, len(acoustic_time), acoustic_time, unit), path)
                   for unit, value, path in specs]
        reference = decoded[0][0]
        if any(not np.array_equal(reference, candidate) for candidate, path in decoded[1:]):
            raise ValueError(f"Finestre del misfit discordanti nel checkpoint rec={row['rec_n']}: "
                             f"{[path for candidate, path in decoded]}")
        masks.append(reference)
        paths.append([path for candidate, path in decoded])
    if missing:
        return None, {'missing_records': missing, 'recovered_records': len(masks)}
    return np.stack(masks), {'source': 'saved misfit sample masks/bounds', 'checkpoint_fields': paths}


def select_plot_window(acoustic_time, requested_us, misfit_mask, full_window=False):
    if requested_us is not None and full_window:
        raise ValueError('--window-us e --full-window sono alternativi.')
    if requested_us is not None:
        return window_mask(acoustic_time, requested_us), 'explicit --window-us'
    if full_window:
        return np.ones(len(acoustic_time), bool), 'explicit --full-window'
    if misfit_mask is None:
        raise ValueError('Finestra del misfit non salvata o non recuperabile. '
                         'Indica gli estremi effettivi con --window-us INIZIO FINE. '
                         'Non viene stimata una finestra dai picchi; --full-window mostra tutto solo se richiesto.')
    masks = np.asarray(misfit_mask, bool)
    if masks.ndim == 1:
        masks = masks[None, :]
    if masks.ndim != 2 or masks.shape[1] != len(acoustic_time) or not np.all(masks.any(axis=1)):
        raise ValueError('Maschere del misfit non valide per l’asse acustico.')
    union = np.any(masks, axis=0)
    indices = np.flatnonzero(union)
    keep = np.zeros(len(acoustic_time), bool)
    keep[indices[0]:indices[-1]+1] = True
    return keep, 'saved misfit: common span covering all compared windows'


def saved_synthetic(row):
    value = checkpoint_value(row, 'synthetic_waveform')
    if value is None:
        return None
    trace = np.asarray(value, dtype=float).reshape(-1)
    if not trace.size or not np.isfinite(trace).all():
        raise ValueError(f"Traccia sintetica non finita/vuota: rec_n={row.get('rec_n')}.")
    return trace.copy()  # La trace_scale e gia applicata. Non applicarla una seconda volta.


def read_traces(path):
    selected, metadata = {}, {}
    first_global, duplicates, fallback = None, 0, 0
    for n, row in enumerate(pickle_rows(path), 1):
        if is_global(row):
            if first_global is None:
                first_global = {k: row.get(k) for k in ('analysis_assembly', '_analysis_export', 'rec_n')}
            continue
        if not {'idx_wave', 'acquisition_time_s', 'rec_n'} <= row.keys():
            continue
        rec, idx = int(row['rec_n']), int(row['idx_wave'])
        if rec in metadata:
            duplicates += 1
        metadata[rec] = {'idx_wave': idx, 'time_saved_s': float(row['acquisition_time_s']), 'rec_n': rec}
        selected.pop(rec, None)  # nei restart prevale l'ultima occorrenza nel file
        if not is_inverted(row):
            continue
        trace = saved_synthetic(row)
        candidate = dict(metadata[rec], synthetic=trace,
                         misfit_window_specs=saved_window_specs(row),
                         saved_misfit=float(row.get('misfit', np.nan)),
                         saved_relative_l2=float(row.get('relative_l2_misfit')
                                                 if row.get('relative_l2_misfit') is not None else np.nan))
        for name in ('observed_time', 'observed_waveform'):
            value = checkpoint_value(row, name)
            if value is not None:
                candidate[name] = np.asarray(value, float).reshape(-1).copy()
        if row.get('forward_evaluated') is None:
            fallback += 1
        selected[rec] = candidate
        if n % 2000 == 0:
            print(f'Letti {n} record; {len(selected)} tracce valutate selezionate.', flush=True)
    rows = sorted(selected.values(), key=lambda r: r['idx_wave'])
    missing = [r['rec_n'] for r in rows if r['synthetic'] is None]
    if missing:
        raise ValueError(f'{len(missing)} valutazioni senza traccia sintetica salvata. '
                         'Serve il .pkl originale completo con checkpoint, non il .analysis.pkl.gz. '
                         f'Primi record: {missing[:5]}. Nessuna simulazione e stata ricostruita.')
    if len(rows) < 2:
        raise ValueError('Meno di due tracce sintetiche valutate disponibili.')
    ids = np.asarray([r['idx_wave'] for r in rows])
    times = np.asarray([r['time_saved_s'] for r in rows])
    if np.any(np.diff(ids) <= 0) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Indici UW o tempi duplicati/non crescenti fra le tracce selezionate.')
    if len({len(r['synthetic']) for r in rows}) != 1:
        raise ValueError('Tracce con campionamenti acustici diversi: non si assume un asse comune.')
    info = {'duplicate_records_kept_last': duplicates, 'legacy_selection_without_forward_flag': fallback}
    if duplicates or fallback:
        print('Selezione:', json.dumps(info), flush=True)
    if fallback:
        warnings.warn('Alcune righe legacy non hanno forward_evaluated: selezione da metadati adattivi/storia locale.')
    return rows, sorted(metadata.values(), key=lambda r: r['idx_wave']), first_global


def edges(centres):
    x = np.asarray(centres, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError('L’asse deve avere almeno due campioni finiti e crescenti.')
    return np.r_[x[0]-(x[1]-x[0])/2, (x[:-1]+x[1:])/2, x[-1]+(x[-1]-x[-2])/2]


def window_mask(acoustic_time, limits):
    if limits is None:
        return np.ones(len(acoustic_time), bool)
    low, high = map(float, limits)
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError('--window-us richiede due estremi finiti e crescenti.')
    if low < acoustic_time[0]-1e-9 or high > acoustic_time[-1]+np.median(np.diff(acoustic_time))+1e-9:
        raise ValueError(f'Finestra {low:g}–{high:g} us non disponibile; '
                         f'campioni {acoustic_time[0]:g}–{acoustic_time[-1]:g} us.')
    return (acoustic_time >= low-1e-12) & (acoustic_time <= high+1e-12)


def display_cells(times, values, display, end):
    """Solo geometria del disegno: nessuna sottrazione su onde mantenute."""
    if display == 'samples':
        return np.arange(len(times)+1)-.5, values
    # Il profilo acquisito al bordo finale ha durata nulla. Nessuna durata inventata.
    if end > times[-1]:
        return np.r_[times, end], values
    return np.asarray(times), values[:-1]


def validate_arrays(acoustic_time, anchor_time, observed, synthetic, rec_n):
    acoustic_time, anchor_time = np.asarray(acoustic_time, float), np.asarray(anchor_time, float)
    edges(acoustic_time); edges(anchor_time)
    observed, synthetic = np.asarray(observed, float), np.asarray(synthetic, float)
    shape = (len(anchor_time), len(acoustic_time))
    if observed.shape != shape or synthetic.shape != shape or len(rec_n) != len(anchor_time):
        raise ValueError('Dimensioni/assi di dati, simulazioni o record non corrispondono.')
    if not np.isfinite(observed).all() or not np.isfinite(synthetic).all():
        raise ValueError('Dati o simulazioni non finiti: nessuna sostituzione con zero.')
    if len(np.unique(rec_n)) != len(rec_n):
        raise ValueError('Record duplicati nel confronto.')
    return acoustic_time, anchor_time, observed, synthetic


def save_figure(fig, output, stem, formats, dpi):
    outputs = []
    for extension in formats:
        path = output / f'{stem}.{extension}'
        buffer = BytesIO()
        fig.savefig(buffer, format=extension, dpi=dpi, bbox_inches='tight', pad_inches=.14)
        path.write_bytes(buffer.getvalue())
        outputs.append(path.name)
    return outputs


def make_maps(output, *, acoustic_time, anchor_time, observed, synthetic,
              dense_time=None, dense_observed=None, rec_n, mech_time=None, shear=None,
              provenance=None, window_us=None, wave_label='', formats=('png', 'pdf', 'svg'),
              dpi=300, display='hold', include_dense=False, dense_acoustic_time=None,
              misfit_mask=None, full_window=False, include_common_residual=False,
              tmin=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator

    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    acoustic_time, anchor_time, observed, synthetic = validate_arrays(
        acoustic_time, anchor_time, observed, synthetic, rec_n)
    rec_n = np.asarray(rec_n, np.int64)
    if misfit_mask is not None:
        masks_shape = np.shape(misfit_mask)
        if len(masks_shape) == 2 and masks_shape[0] not in (1, len(anchor_time)):
            raise ValueError('Il numero di maschere del misfit non corrisponde ai record confrontati.')
    mask, window_source = select_plot_window(acoustic_time, window_us, misfit_mask, full_window)
    if mask.sum() < 3:
        raise ValueError('Meno di tre campioni nella finestra acustica richiesta.')
    y, d, s = acoustic_time[mask], observed[:, mask], synthetic[:, mask]
    residual = s-d  # definizione unica in figure, NPZ, CSV e documentazione
    residual_full = synthetic-observed
    base_limit = float(max(np.max(np.abs(d)), np.max(np.abs(s))))
    rmax = float(np.max(np.abs(residual)))
    common_limit = base_limit  # condiviso SOLO fra dati e simulazioni
    if common_limit == 0:
        common_limit = 1.0
        warnings.warn('Dati e simulazioni tutti nulli nella finestra: scala grafica convenzionale +/-1.')
    norm = Normalize(-common_limit, common_limit, clip=False)
    residual_norm = Normalize(-rmax, rmax, clip=False) if rmax > 0 else Normalize(-1., 1.)
    mech_time = np.asarray([] if mech_time is None else mech_time, float)
    shear = np.asarray([] if shear is None else shear, float)
    if mech_time.shape != shear.shape or (len(mech_time) and not np.isfinite([mech_time, shear]).all()):
        raise ValueError('Tempi e stress meccanici non corrispondono o non sono finiti.')
    if len(mech_time) > 1:
        edges(mech_time)
    dense_time = np.asarray([] if dense_time is None else dense_time, float)
    display_end = max(float(anchor_time[-1]), float(dense_time[-1]) if len(dense_time) else float(anchor_time[-1]))
    if display not in ('hold', 'samples'):
        raise ValueError('Rappresentazione temporale sconosciuta.')
    if tmin is not None and not np.isfinite(tmin):
        raise ValueError('--tmin deve essere un tempo finito in secondi di acquisizione.')
    display_start = float(anchor_time[0]) if tmin is None else max(float(anchor_time[0]), float(tmin))
    if display_start >= display_end:
        raise ValueError(f'--tmin lascia vuota la mappa: il confronto termina a {display_end:g} s.')
    first_shown = int(np.searchsorted(anchor_time, display_start, side='left'))
    if display == 'samples' and first_shown == len(anchor_time):
        raise ValueError('--tmin non lascia acquisizioni confrontate nella rappresentazione samples.')
    xlabel = ('Acquisition time (s)' if display == 'hold'
              else 'Compared acquisitions, equally spaced; labels show acquisition time (s)')
    description = ('Cells hold the SAVED sample until the next available one; no forward in the gaps.'
                   if display == 'hold' else 'One column per compared record; unequal time gaps are not drawn to scale.')
    if tmin is not None:
        description += '\nAcquisition-time crop affects plot axes only; cache, CSV and colour scales retain all records.'
    if 'legacy NPZ' in str((provenance or {}).get('clock', '')):
        description += '\nLegacy rescaled acquisition times preserved; no further clock transformation.'
    elif (provenance or {}).get('rescaling_factor') is not None:
        description += '\nAcquisition clock rescaling explicitly requested; acoustic waveform times unchanged.'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 14,
                         'axes.labelsize': 13, 'axes.titleweight': 'bold', 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    outputs = []

    def configure_x(ax):
        if display == 'samples':
            ids = np.unique(np.rint(np.linspace(first_shown, len(anchor_time)-1,
                                               min(7, len(anchor_time)-first_shown))).astype(int))
            ax.set_xticks(ids, [f'{anchor_time[i]:.2f}' for i in ids])
            ax.set_xlim(first_shown-.5, len(anchor_time)-.5)
        else:
            ax.set_xlim(display_start, display_end)
            ax.xaxis.set_major_locator(MaxNLocator(7))
            ax.ticklabel_format(axis='x', style='plain', useOffset=False)

    def panel(ax, values, title, own_norm=None):
        tx, shown = display_cells(anchor_time, values, display, display_end)
        mesh = ax.pcolormesh(tx, edges(y), shown.T, cmap='RdBu_r', norm=own_norm or norm,
                             shading='flat', linewidth=0, edgecolors='none', antialiased=False,
                             rasterized=True)
        ax.set(title=title, ylabel='Time from trigger (µs)', ylim=(y[0], y[-1]))
        configure_x(ax)
        marks = anchor_time if display == 'hold' else np.arange(len(anchor_time))
        ax.plot(marks, np.full(len(marks), y[-1]), '|', color='#222222', ms=4, alpha=.5)
        return mesh

    def stress_panel(ax):
        if len(mech_time):
            if display == 'hold':
                ax.plot(mech_time, shear, color='#304fa6', lw=1.)
            else:
                valid = (anchor_time >= mech_time[0]) & (anchor_time <= mech_time[-1])
                ax.plot(np.flatnonzero(valid), np.interp(anchor_time[valid], mech_time, shear),
                        color='#304fa6', lw=1., marker='.', ms=2)
        ax.set(ylabel='Shear stress (MPa)', xlabel=xlabel)
        ax.grid(axis='x', color='#dfe5ed', lw=.6)
        configure_x(ax)

    single_maps = [
            ('mappa_dati_stessi_record', d, 'Recorded data — same records as the simulations', None),
            ('mappa_simulazioni', s, 'Saved simulations', None),
            ('residuo_scala_propria', residual, 'Residual = simulation − data; separate amplitude scale',
             residual_norm)]
    if include_common_residual:
        # Diagnostica facoltativa: comprende l'intero residuo, anche se supera dato e sintetica.
        single_maps.append(('residuo_stessa_scala', residual, 'Residual — optional data/simulation amplitude reference',
                            Normalize(-max(common_limit, rmax), max(common_limit, rmax))))
    else:
        # Evita di lasciare una vecchia figura bianca accanto alle nuove uscite.
        for ext in formats:
            old = output / f'residuo_stessa_scala.{ext}'
            if old.exists():
                backup = output / 'figure_precedenti'
                backup.mkdir(exist_ok=True)
                old.replace(backup / old.name)
    for stem, values, title, own in single_maps:
        fig, ax = plt.subplots(figsize=(14, 5.5), layout='constrained')
        mesh = panel(ax, values, title, own)
        ax.set_xlabel(xlabel)
        fig.colorbar(mesh, ax=ax, label='Amplitude (preprocessed data units)')
        fig.suptitle(f'{wave_label + " waves — " if wave_label else ""}{len(anchor_time)} matched records', fontsize=15)
        fig.supxlabel(description, fontsize=10)
        outputs += save_figure(fig, output, stem, formats, dpi)
        plt.close(fig)

    fig = plt.figure(figsize=(14.5, 12.2), layout='constrained')
    grid = fig.add_gridspec(4, 2, width_ratios=[1, .035], height_ratios=[1.6, 1.6, 1.6, .8])
    axes = [fig.add_subplot(grid[0, 0])]
    axes += [fig.add_subplot(grid[i, 0], sharex=axes[0]) for i in (1, 2, 3)]
    panel(axes[0], d, '(a) Recorded data — identical compared records')
    data_mesh = panel(axes[1], s, '(b) Saved simulations — same scale as data')
    residual_mesh = panel(axes[2], residual, '(c) Simulation − data — separate residual scale', residual_norm)
    stress_panel(axes[3])
    fig.colorbar(data_mesh, cax=fig.add_subplot(grid[:2, 1]), label='Data / simulation amplitude (data units)')
    fig.colorbar(residual_mesh, cax=fig.add_subplot(grid[2, 1]), label='Residual amplitude (data units)')
    fig.suptitle(f'{wave_label + " waves: " if wave_label else ""}data, simulations and residual\n'
                 'Same samples; data/simulation share a scale, residual has its own scale', fontsize=16)
    fig.supxlabel(description, fontsize=10)
    outputs += save_figure(fig, output, 'confronto_stesse_onde', formats, dpi)
    plt.close(fig)

    dense_cache = {}
    if include_dense and dense_observed is not None and len(dense_time) >= 2:
        fast_dense = acoustic_time if dense_acoustic_time is None else np.asarray(dense_acoustic_time, float)
        dense_observed = np.asarray(dense_observed)
        if dense_observed.shape != (len(dense_time), len(fast_dense)):
            raise ValueError('Dimensioni della mappa di tutti i dati non compatibili con gli assi.')
        dense_mask = (fast_dense >= y[0]-1e-12) & (fast_dense <= y[-1]+1e-12)
        if dense_mask.sum() < 3:
            raise ValueError('La mappa densa non contiene la finestra acustica richiesta.')
        dense_values = dense_observed[:, dense_mask]
        if not np.isfinite(dense_values).all():
            raise ValueError('Valori non finiti nella mappa densa dei dati.')
        # La mappa densa e separata: nessun residuo fra dati nuovi e sintetiche vecchie.
        dense_limit = max(common_limit, float(np.max(np.abs(dense_values))))
        fig, ax = plt.subplots(figsize=(14, 5.5), layout='constrained')
        mesh = ax.pcolormesh(edges(dense_time), edges(fast_dense[dense_mask]), dense_values.T,
                             cmap='RdBu_r', vmin=-dense_limit, vmax=dense_limit, shading='flat', rasterized=True)
        ax.set(xlabel='Acquisition time (s)', ylabel='Time from trigger (µs)',
               title=f'All {len(dense_time)} recorded waves — separate context map')
        if tmin is not None:
            dense_start = max(float(dense_time[0]), float(tmin))
            if dense_start >= dense_time[-1]:
                plt.close(fig)
                raise ValueError('--tmin lascia vuota la mappa di tutti i dati registrati.')
            ax.set_xlim(dense_start, dense_time[-1])
        ax.ticklabel_format(axis='x', style='plain', useOffset=False)
        fig.colorbar(mesh, ax=ax, label='Amplitude (preprocessed data units)')
        outputs += save_figure(fig, output, 'mappa_dati_tutte_le_onde', formats, dpi)
        plt.close(fig)
        dense_cache = {'dense_time_s': dense_time, 'dense_window_time_us': fast_dense[dense_mask],
                       'dense_observed_window': dense_values}

    energy = np.sum(d*d, axis=1)
    J = np.divide(np.sum(residual*residual, axis=1), energy,
                  out=np.full(len(d), np.nan), where=energy > 0)
    with (output / 'scarti_finestra.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rec_n', 'acquisition_time_s', 'relative_squared_L2_selected_window',
                         'max_abs_residual', 'rms_residual'])
        writer.writerows(zip(rec_n, anchor_time, J, np.max(np.abs(residual), axis=1),
                             np.sqrt(np.mean(residual**2, axis=1))))
    info = dict(provenance or {})
    info.update(residual_definition='synthetic - observed', common_color_limit=common_limit, wave_label=wave_label,
                common_scale_applies_to='observed and synthetic only; residual has its own full-range scale',
                plot_window_source=window_source,
                data_simulation_max_abs=base_limit, residual_separate_color_limit=rmax,
                acoustic_window_requested_us=None if window_us is None else list(map(float, window_us)),
                acoustic_window_samples_us=[float(y[0]), float(y[-1])], compared_records=len(anchor_time),
                acquisition_tmin_requested_s=None if tmin is None else float(tmin),
                acquisition_display_bounds_s=[display_start if display == 'hold' else float(anchor_time[first_shown]),
                                              display_end if display == 'hold' else float(anchor_time[-1])],
                acquisition_crop_effect='plot axes only; complete arrays, CSV and colour scales retained',
                additional_trace_scaling=False, waveform_realignment=False,
                temporal_interpolation_of_synthetics=False, display=display, display_description=description,
                relative_squared_L2_window_median=float(np.nanmedian(J)) if np.isfinite(J).any() else None,
                relative_squared_L2_window_max=float(np.nanmax(J)) if np.isfinite(J).any() else None,
                outputs=outputs, formats=list(formats))
    buffer = BytesIO()
    saved_masks = {} if misfit_mask is None else {'misfit_mask': np.asarray(misfit_mask, bool)}
    np.savez_compressed(buffer, time_us=acoustic_time, acquisition_time_s=anchor_time, rec_n=rec_n,
                        observed=observed, synthetic=synthetic, residual=residual_full,
                        mechanical_time_s=mech_time, shear_stress_MPa=shear,
                        provenance_json=np.array(json.dumps(info, ensure_ascii=False)), **dense_cache, **saved_masks)
    (output / 'tracce_per_confronto.npz').write_bytes(buffer.getvalue())
    (output / 'provenienza.json').write_text(json.dumps(info, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'Confronto: {len(anchor_time)} record; L2 relativa mediana nella finestra = '
          f'{info["relative_squared_L2_window_median"]}; output {output}', flush=True)
    return info


def export_context(namespace, context, selected, metadata, args):
    handler = context['uw_data_handler']
    infile = Path(context['infile_path'])
    if infile.name.split('.')[0] != args.basename:
        raise ValueError('Il file UW caricato non corrisponde al pickle richiesto.')
    data = np.asarray(handler.waveform_data)
    acoustic_time = np.asarray(handler.metadata['time_ax_waveform'], float)
    uw_time = np.asarray(handler.metadata['time_ax_acquisition'], float)
    if data.shape != (len(uw_time), len(acoustic_time)):
        raise ValueError('Assi UW e dimensioni delle onde non compatibili.')
    edges(acoustic_time); edges(uw_time)
    idx = np.asarray([r['idx_wave'] for r in selected], int)
    if idx.min() < 0 or idx.max() >= len(data):
        raise ValueError('Indici salvati fuori dalle onde caricate: controllare il downsampling originale.')
    synthetic = np.stack([r['synthetic'] for r in selected])
    if synthetic.shape[1] != len(acoustic_time):
        raise ValueError('Campionamento/durata acustica diverso dalla run salvata: ripristinare il preprocessing.')
    observed = np.asarray(data[idx], float)
    misfit_masks, misfit_info = recover_misfit_masks(selected, acoustic_time)
    checked_time = checked_observed = 0
    for k, r in enumerate(selected):
        if 'observed_time' in r:
            if r['observed_time'].shape != acoustic_time.shape or not np.allclose(r['observed_time'], acoustic_time, rtol=0, atol=1e-10):
                raise ValueError(f"Asse acustico diverso dal checkpoint, rec={r['rec_n']}.")
            checked_time += 1
        if 'observed_waveform' in r:
            if r['observed_waveform'].shape != observed[k].shape or not np.allclose(r['observed_waveform'], observed[k], rtol=1e-10, atol=1e-12):
                raise ValueError(f"Preprocessing/dato diverso dal checkpoint, rec={r['rec_n']}.")
            checked_observed += 1
    saved_times = np.asarray([r['time_saved_s'] for r in selected])
    saved_step = float(np.median(np.diff(saved_times)/np.diff(idx)))
    current_step = float(np.median(np.diff(uw_time)))
    metadata = [m if isinstance(m, dict) else dict(zip(('idx_wave', 'time_saved_s', 'rec_n'), m)) for m in metadata]
    all_idx = np.asarray([m['idx_wave'] for m in metadata], int)
    dense_saved_time = np.asarray([m['time_saved_s'] for m in metadata], float)
    if all_idx.min() < 0 or all_idx.max() >= len(data) or np.any(np.diff(all_idx) <= 0):
        raise ValueError('Indici UW dei record salvati non validi/duplicati dopo la selezione.')
    edges(dense_saved_time)
    clock_info = {'clock': 'saved acquisition_time_s; no time transformation'}
    anchor_time, dense_time = saved_times, dense_saved_time
    mech = namespace.get('mech_data', context.get('mechanical_dataframe'))
    mt = np.asarray(mech['time_s'], float) if mech is not None else np.array([])
    shear = np.asarray(mech['shear_stress_MPa'], float) if mech is not None else np.array([])
    if args.clock == 'sync-affine':
        peaks = np.asarray(namespace['sync_peaks'], int)
        position = int(namespace['chosen_uw_file'])
        pair = peaks[2*position:2*position+2]
        if len(pair) != 2 or np.any(pair < 0) or np.any(pair >= len(mt)):
            raise ValueError('Coppia di sync non disponibile per il file UW.')
        start, stop = map(float, mt[pair])
        beta = (stop-start)/(uw_time[-1]-uw_time[0])
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError('Durate non valide per il riscalamento affine esplicitamente richiesto.')
        scaled = start+beta*(uw_time-uw_time[0])
        anchor_time, dense_time = scaled[idx], scaled[all_idx]
        clock_info = {'clock': 'sync-affine explicitly requested; acoustic waveform times unchanged',
                      'sync_start_s': start, 'sync_stop_s': stop, 'rescaling_factor': float(beta)}
    else:
        beta = 1.0
    if not (np.isclose(saved_step, current_step, rtol=1e-6, atol=1e-9)
            or (args.clock == 'sync-affine' and np.isclose(saved_step, current_step*beta, rtol=1e-6, atol=1e-9))):
        raise ValueError('Passo temporale UW diverso dalla run: non si associano automaticamente onde con indici ambigui.')
    keep_mech = (mt >= dense_time[0]) & (mt <= dense_time[-1])
    provenance = dict(results_path=str(args.results), results_bytes=args.results.stat().st_size,
                      main_path=str(args.main), main_sha256=hashlib.sha256(args.main.read_bytes()).hexdigest(),
                      input_uw_path=str(infile), source_of_synthetics='saved evaluated traces; no new forward',
                      **clock_info, saved_step_s_per_idx=saved_step, current_uw_step_s=current_step,
                      saved_anchor_time_s=saved_times.tolist(), anchor_idx_wave=idx.tolist(),
                      saved_misfit=[r['saved_misfit'] for r in selected],
                      saved_relative_l2=[r['saved_relative_l2'] for r in selected],
                      checkpoint_acoustic_axes_checked=checked_time,
                      checkpoint_observed_traces_checked=checked_observed,
                      misfit_window_recovery=misfit_info,
                      preprocessing_validation=('exact checkpoint trace/time checks where available; '
                                                'otherwise requires the original main preprocessing; length and UW cadence checked'),
                      note='Window L2 is recomputed for the requested display window; it may differ from the saved inversion objective.')
    if checked_observed < len(selected) or checked_time < len(selected):
        print('Nota: non tutti i checkpoint contengono dato e asse acustico. '
              'Usare il principale con il preprocessing originale; i controlli disponibili sono registrati.', flush=True)
    output = args.output or args.results.parent/(args.basename+'_mappe_simulazioni')
    make_maps(output, acoustic_time=acoustic_time, anchor_time=anchor_time,
              observed=observed, synthetic=synthetic, dense_time=dense_time,
              dense_observed=np.asarray(data[all_idx], float) if args.dense else None,
              rec_n=np.asarray([r['rec_n'] for r in selected]),
              mech_time=mt[keep_mech], shear=shear[keep_mech], provenance=provenance,
              window_us=args.window_us, wave_label=args.wave_label, formats=args.formats, dpi=args.dpi,
              display=args.display, include_dense=args.dense, misfit_mask=misfit_masks,
              full_window=getattr(args, 'full_window', False),
              include_common_residual=getattr(args, 'include_common_residual', False),
              tmin=getattr(args, 'tmin', None))
    raise ExportCompleted


def prepare_main(source, *, wave_type, basename):
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == 'process_uw_file']
    if len(functions) != 1 or isinstance(functions[0], ast.AsyncFunctionDef):
        raise ValueError('Interfaccia process_uw_file non riconosciuta; principale non eseguito.')
    functions[0].body = ast.parse('return _map_export_context(globals(), locals())').body
    blocks = [n for n in tree.body if isinstance(n, ast.If)
              and isinstance(n.test, ast.Compare) and ast.unparse(n.test) in
              ("__name__ == '__main__'", "'__main__' == __name__")]
    if len(blocks) != 1:
        raise ValueError('Blocco principale non riconosciuto; nessuna esecuzione.')
    changes, body = set(), []
    for node in blocks[0].body:
        names = [n.id for n in node.targets if isinstance(n, ast.Name)] if isinstance(node, ast.Assign) else []
        if 'wave_type' in names:
            node.value = ast.Constant(wave_type); changes.add('wave_type')
        if 'selected_uw_basename' in names:
            node.value = ast.Constant(basename); changes.add('selected_uw_basename')
        body.append(node)
        if 'params' in names:
            body.extend(ast.parse("params['diagnostic_plot_only'] = False\nparams['run_local_inversion'] = False").body)
            changes.add('params')
    if changes != {'wave_type', 'selected_uw_basename', 'params'}:
        raise ValueError(f'Configurazione principale non riconosciuta ({sorted(changes)}); nessuna esecuzione.')
    blocks[0].body = ast.parse('_map_disable_modeler(UltrasonicModeler)').body+body
    return ast.fix_missing_locations(tree)


def disable_modeler(modeler_class):
    missing = object()
    originals = {}
    def forbidden(*args, **kwargs):
        raise RuntimeError('Forward e inversioni sono disabilitati durante l’estrazione delle tracce salvate.')
    for name in dir(modeler_class):
        if name == 'forward_simulation' or name.startswith('run_local_'):
            originals[name] = modeler_class.__dict__.get(name, missing)
            setattr(modeler_class, name, forbidden)
    def restore():
        for name, value in originals.items():
            if value is missing:
                delattr(modeler_class, name)
            else:
                setattr(modeler_class, name, value)
    return restore


def from_npz(path, args):
    with np.load(path, allow_pickle=False) as f:
        required = {'time_us', 'observed', 'synthetic', 'rec_n'}
        if not required <= set(f.files):
            raise ValueError(f'NPZ incompleto: mancano {sorted(required-set(f.files))}.')
        arrays = {k: f[k].copy() for k in f.files}
    time_key = 'acquisition_time_s' if 'acquisition_time_s' in arrays else 'time_rescaled_s'
    if time_key not in arrays:
        raise ValueError('Nel NPZ manca l’asse temporale delle acquisizioni.')
    info = json.loads(str(arrays['provenance_json'])) if 'provenance_json' in arrays else {}
    if time_key == 'time_rescaled_s':
        info['clock'] = 'legacy NPZ time_rescaled_s preserved; original unscaled times are unavailable here'
        print('NPZ precedente: conservo i tempi gia riscalati, senza applicare altre trasformazioni.', flush=True)
    info['redrawn_from_npz'] = str(path.resolve())
    misfit_masks = arrays.get('misfit_mask')
    if misfit_masks is None and args.window_us is None and not args.full_window:
        original = args.window_results or info.get('results_path')
        if original and Path(original).expanduser().is_file():
            main_path = info.get('main_path')
            if main_path:
                sys.path.insert(0, str(Path(main_path).parent))
            print('Recupero le finestre del misfit dal pickle originale indicato nella provenienza...', flush=True)
            saved, _, _ = read_traces(Path(original).expanduser())
            by_record = {r['rec_n']: r for r in saved}
            missing = [int(rec) for rec in arrays['rec_n'] if int(rec) not in by_record]
            if missing:
                raise ValueError(f'Il pickle usato per la finestra non contiene {len(missing)} record del NPZ.')
            ordered = [by_record[int(rec)] for rec in arrays['rec_n']]
            for i, row in enumerate(ordered):
                if (row['synthetic'].shape != arrays['synthetic'][i].shape
                        or not np.array_equal(row['synthetic'], arrays['synthetic'][i])):
                    raise ValueError(f"NPZ e pickle originale hanno sintetiche diverse al rec={row['rec_n']}; "
                                     'non si trasferisce una finestra da un’altra run.')
                if 'observed_time' in row and not np.array_equal(row['observed_time'], arrays['time_us']):
                    raise ValueError('NPZ e pickle originale hanno assi acustici diversi.')
            misfit_masks, recovery = recover_misfit_masks(ordered, arrays['time_us'])
            info['misfit_window_recovery'] = recovery
    output = args.output or path.parent/'mappe_ridisegnate'
    make_maps(output, acoustic_time=arrays['time_us'], anchor_time=arrays[time_key],
              observed=arrays['observed'], synthetic=arrays['synthetic'], rec_n=arrays['rec_n'],
              dense_time=arrays.get('dense_time_s', arrays.get('dense_time_rescaled_s')),
              dense_observed=arrays.get('dense_observed_window'),
              dense_acoustic_time=arrays.get('dense_window_time_us'),
              mech_time=arrays.get('mechanical_time_s'), shear=arrays.get('shear_stress_MPa'), provenance=info,
              window_us=args.window_us, wave_label=args.wave_label or info.get('wave_label', ''), formats=args.formats, dpi=args.dpi,
              display=args.display, include_dense=args.dense, misfit_mask=misfit_masks,
              full_window=args.full_window, include_common_residual=args.include_common_residual,
              tmin=getattr(args, 'tmin', None))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--main', type=Path, default=Path('local_optimization_FWI_velocity_1D.py'),
                        help='principale con lo stesso preprocessing della run, usato solo con --results')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--results', type=Path, help='.pkl completo con tracce sintetiche nei checkpoint')
    source.add_argument('--input-npz', type=Path, help='ridisegna un tracce_per_confronto.npz gia esportato')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wave', choices=['P', 'S'], help='polarizzazione; altrimenti ricavata dal nome del pickle')
    parser.add_argument('--uw-basename', help='nome UW senza estensione, se il pickle e stato rinominato')
    window_options = parser.add_mutually_exclusive_group()
    window_options.add_argument('--window-us', nargs=2, type=float, metavar=('START', 'STOP'),
                        help='override esplicito; default intervallo che comprende le finestre del misfit salvate')
    window_options.add_argument('--full-window', action='store_true',
                        help='mostra tutti i campioni solo se richiesto esplicitamente')
    parser.add_argument('--window-results', type=Path,
                        help='per vecchi NPZ senza maschere: pickle originale da cui recuperare le finestre')
    parser.add_argument('--include-common-residual', action='store_true',
                        help='aggiunge facoltativamente la vecchia figura del residuo con riferimento alla scala del dato')
    parser.add_argument('--clock', choices=['saved', 'sync-affine'], default='saved',
                        help='default: tempi salvati; sync-affine richiede esplicitamente il vecchio riscalamento')
    parser.add_argument('--display', choices=['hold', 'samples'], default='hold',
                        help='hold: tempi reali e mantenimento SOLO grafico; samples: una colonna per record')
    parser.add_argument('--tmin', type=float,
                        help='inizio grafico in secondi di acquisizione; non modifica cache, CSV o scale colore; '
                             'in samples mostra le colonne con tempo >= tmin')
    parser.add_argument('--dense', action='store_true', help='aggiunge una mappa separata di tutti i dati registrati')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf', 'svg'])
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error('--dpi deve essere positivo')
    if args.tmin is not None and not np.isfinite(args.tmin):
        parser.error('--tmin deve essere un tempo finito in secondi di acquisizione')
    args.wave_label = args.wave or ''
    if args.output:
        args.output = args.output.expanduser().resolve()
    if args.input_npz:
        if args.clock != 'saved':
            parser.error('--clock sync-affine richiede i dati originali, non un NPZ gia esportato')
        from_npz(args.input_npz.expanduser().resolve(), args)
        return 0
    args.main, args.results = args.main.expanduser().resolve(), args.results.expanduser().resolve()
    if not args.main.is_file() or not args.results.is_file():
        parser.error('File principale o pickle non trovato.')
    args.basename = args.uw_basename or args.results.name.split('.')[0]
    match = re.search(r'_([PS])$', args.basename, re.I)
    wave = args.wave or (match.group(1).upper() if match else None)
    if wave is None:
        parser.error('Polarizzazione non riconoscibile: indica --wave P oppure --wave S e --uw-basename.')
    args.wave_label = wave
    sys.path.insert(0, str(args.main.parent))
    selected, metadata, _ = read_traces(args.results)
    tree = prepare_main(args.main.read_text(encoding='utf-8'), wave_type='_'+wave.lower(), basename=args.basename)
    restorers = []
    def guarded_modeler(modeler_class):
        restorers.append(disable_modeler(modeler_class))
    namespace = {'__name__': '__main__', '__file__': str(args.main), '__package__': None,
                 '_map_export_context': lambda ns, ctx: export_context(ns, ctx, selected, metadata, args),
                 '_map_disable_modeler': guarded_modeler}
    try:
        exec(compile(tree, str(args.main), 'exec'), namespace)
    except ExportCompleted:
        return 0
    finally:
        for restore in reversed(restorers):
            restore()
    raise RuntimeError('Nessun file ha raggiunto l’esportazione: verificare il principale indicato.')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print(f'ERRORE: {error}', file=sys.stderr)
        raise SystemExit(2)
