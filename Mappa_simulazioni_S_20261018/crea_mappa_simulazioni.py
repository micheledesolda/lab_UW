#!/usr/bin/env python3
"""Mappe delle tracce accettate nei checkpoint, senza eseguire inversioni.

Esempio, nell'ambiente Python della run:
  python crea_mappa_simulazioni.py --main local_optimization_FWI_velocity_1D.py \
      --results /percorso/006_reload_ss_1_S.pkl

Il principale viene eseguito in memoria con process_uw_file sostituita dalla
sola esportazione. Il file principale e il pickle non vengono modificati.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np


class ExportCompleted(Exception):
    pass


def pickle_rows(path):
    # Usare soltanto risultati prodotti dalla propria run: pickle esegue codice.
    import pickle
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        while True:
            position = stream.tell()
            # Distinguere un EOF pulito da un oggetto finale incompleto.
            first = stream.read(1)
            if not first:
                break
            stream.seek(position)
            try:
                row = pickle.load(stream)
            except Exception as exc:
                raise ValueError(f'Pickle incompleto/non leggibile al byte {position}; originale non modificato.') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Oggetto al byte {position} non riconosciuto: {type(row).__name__}.')
            yield row


def is_global(row):
    return (row.get('_analysis_export', {}).get('record_kind') == 'global_reference'
            or ('global_best_misfit' in row and 'relative_l2_misfit' not in row))


def is_inverted(row):
    if is_global(row):
        return False
    decision = row.get('adaptive_inversion') or {}
    if 'optimized' in decision:
        return bool(decision['optimized'])
    if decision.get('action') in ('reused', 'reuse', 'skipped', 'skip'):
        return False
    if row.get('forward_evaluated') is False:
        return False
    # Compatibilità con righe non adattive; non dedurre un'inversione dal
    # checkpoint, che può essere copiato anche nelle righe riutilizzate.
    return bool(row.get('local_iteration_history') or
                (row.get('local_iteration_history_summary') or {}).get('accepted_entries', 0))


def saved_synthetic(row):
    state = row.get('local_continuation_state') or {}
    attributes = state.get('simulation_attributes') or {}
    value = attributes.get('synthetic_waveform')
    if value is None:
        value = row.get('synthetic_waveform')
    if value is None:
        return None
    trace = np.asarray(value, dtype=float).reshape(-1)
    if not trace.size or not np.all(np.isfinite(trace)):
        raise ValueError(f"Traccia non finita/vuota: rec_n={row.get('rec_n')}.")
    return trace.copy()


def read_traces(path):
    selected = []
    metadata = []
    first_global = None
    missing = []
    for n, row in enumerate(pickle_rows(path), 1):
        if is_global(row):
            if first_global is None:
                first_global = {k: row.get(k) for k in ('analysis_assembly', '_analysis_export', 'rec_n')}
            continue
        if 'idx_wave' not in row or 'acquisition_time_s' not in row:
            continue
        metadata.append((int(row['idx_wave']), float(row['acquisition_time_s']), int(row['rec_n'])))
        if is_inverted(row):
            trace = saved_synthetic(row)
            if trace is None:
                missing.append(int(row['rec_n']))
            else:
                selected.append(dict(idx_wave=int(row['idx_wave']),
                                     rec_n=int(row['rec_n']),
                                     time_saved_s=float(row['acquisition_time_s']),
                                     saved_misfit=float(row.get('misfit', np.nan)),
                                     synthetic=trace))
        if n % 1000 == 0:
            print(f'Letti {n} record; {len(selected)} tracce accettate recuperate.', flush=True)
    if missing:
        raise ValueError(
            f'{len(missing)} inversioni non contengono la traccia sintetica. '
            'Serve il .pkl originale completo, non il .analysis.pkl.gz compatto. '
            'Nessun forward o inversione è stato eseguito. '
            f'Primi record senza traccia: {missing[:5]}')
    if len(selected) < 2:
        raise ValueError('Meno di due tracce di inversioni effettive disponibili.')
    selected.sort(key=lambda r: r['idx_wave'])
    ids = np.array([r['idx_wave'] for r in selected])
    if np.any(np.diff(ids) <= 0):
        raise ValueError('Indici UW duplicati/non crescenti fra le inversioni: risolvere prima la duplicazione.')
    lengths = {r['synthetic'].size for r in selected}
    if len(lengths) != 1:
        raise ValueError('Tracce salvate con lunghezze diverse: impossibile assumere lo stesso asse acustico.')
    return selected, metadata, first_global


def edges(centres):
    x = np.asarray(centres, float)
    if x.ndim != 1 or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError('L’asse della mappa deve avere almeno due campioni crescenti.')
    return np.r_[x[0] - (x[1] - x[0]) / 2,
                 (x[:-1] + x[1:]) / 2,
                 x[-1] + (x[-1] - x[-2]) / 2]


def make_maps(output, *, acoustic_time, anchor_time, observed, synthetic,
              dense_time, dense_observed, rec_n, mech_time, shear, provenance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    acoustic_time = np.asarray(acoustic_time, float)
    anchor_time = np.asarray(anchor_time, float)
    window = (acoustic_time >= 20.) & (acoustic_time <= 25.)
    if window.sum() < 3:
        raise ValueError('La finestra 20–25 µs non è disponibile. Usare la configurazione S della run.')
    if observed.shape != synthetic.shape or observed.shape != (len(anchor_time), len(acoustic_time)):
        raise ValueError('Assi/dimensioni delle onde registrate e simulate non corrispondono.')
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(synthetic)):
        raise ValueError('Dati o simulazioni non finiti: nessuna sostituzione automatica con zero.')
    y = acoustic_time[window]
    d = observed[:, window]
    s = synthetic[:, window]
    dense = dense_observed[:, window]
    residual = d - s
    # Un'unica scala di ampiezza per dati, simulazioni e residui. Nessuna
    # normalizzazione colonna per colonna e nessun riallineamento delle onde.
    limit = float(max(np.max(np.abs(dense)), np.max(np.abs(s))))
    if limit <= 0:
        raise ValueError('Ampiezze tutte nulle nella finestra del diretto.')
    norm = Normalize(-limit, limit)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'savefig.dpi': 180})
    xlabel = 'Tempo meccanico [s] — ipotesi di riscalamento fra i due sync'

    # Le celle delle tracce rade iniziano al loro istante: la convenzione
    # geometrica a metà fra campioni anticiperebbe visivamente le transizioni.
    # Il mantenimento è soltanto grafico; non calcola onde nei periodi saltati.
    display_end = max(float(dense_time[-1]), float(anchor_time[-1]))
    if display_end <= anchor_time[-1]:
        display_end = float(anchor_time[-1] + np.median(np.diff(dense_time))/2)
    anchor_edges = np.r_[anchor_time, display_end]

    def panel(ax, times, values, title, sparse=False):
        tx = anchor_edges if sparse else edges(times)
        mesh = ax.pcolormesh(tx, edges(y), values.T,
                             cmap='RdBu_r', norm=norm, shading='flat', rasterized=True)
        ax.set(title=title, ylabel='Tempo dal trigger [µs]')
        ax.set_ylim(20., 25.)
        ax.set_xlim(anchor_time[0], display_end)
        return mesh

    def stress_panel(ax):
        ax.plot(mech_time, shear, color='k', lw=.7)
        ax.set(ylabel='τ [MPa]', xlabel=xlabel)
        ax.grid(alpha=.2)

    fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
    mesh = panel(ax, anchor_time, s, f'Simulazioni accettate — {len(anchor_time)} inversioni effettive\nUltima traccia mantenuta graficamente fino alla successiva', sparse=True)
    ax.plot(anchor_time, np.full(len(anchor_time), 24.98), '|', color='k', ms=4, alpha=.5)
    ax.set_xlabel(xlabel)
    fig.colorbar(mesh, ax=ax, label='Ampiezza [unità del dato preprocessato]')
    fig.savefig(output / 'mappa_simulazioni.png')
    plt.close(fig)

    for filename, first_time, first_data, title in [
        ('confronto_stesse_onde.png', anchor_time, d, 'Dati: stessi istanti delle inversioni'),
        ('confronto_tutte_le_onde.png', dense_time, dense, f'Dati: tutte le {len(dense_time)} onde del tratto'),
    ]:
        fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True,
                               gridspec_kw={'height_ratios': [2, 2, 2, 1]}, layout='constrained')
        panel(ax[0], first_time, first_data, title, sparse=filename=='confronto_stesse_onde.png')
        mesh = panel(ax[1], anchor_time, s, 'Simulazioni: ultima traccia disponibile mantenuta fino alla successiva', sparse=True)
        ax[1].plot(anchor_time, np.full(len(anchor_time), 24.98), '|', color='k', ms=4, alpha=.5)
        panel(ax[2], anchor_time, residual, 'Residuo agli istanti invertiti, mantenuto graficamente; stessa scala di colore', sparse=True)
        stress_panel(ax[3])
        fig.colorbar(mesh, ax=list(ax[:3]), label='Ampiezza [unità del dato preprocessato]', shrink=.85)
        fig.suptitle('Arrivo diretto S — nessun riallineamento o riscalamento aggiuntivo delle tracce\n'
                     'Tacche nere = inversioni; nessun nuovo forward fra due tracce salvate')
        fig.savefig(output / filename)
        plt.close(fig)

    # Rende visibili anche residui piccoli rispetto all'onda, senza confondere
    # questa scala amplificata con quella delle tre mappe precedenti.
    rmax = float(np.max(np.abs(residual)))
    fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
    mesh = ax.pcolormesh(anchor_edges, edges(y), residual.T, cmap='RdBu_r',
                         vmin=-max(rmax, 1e-30), vmax=max(rmax, 1e-30), shading='flat', rasterized=True)
    ax.set(xlabel=xlabel, ylabel='Tempo dal trigger [µs]', ylim=(20., 25.),
           xlim=(anchor_time[0], display_end), title='Residuo agli istanti invertiti, mantenuto graficamente; scala di colore amplificata')
    fig.colorbar(mesh, ax=ax, label='Residuo [unità del dato preprocessato]')
    fig.savefig(output / 'residuo_scala_propria.png')
    plt.close(fig)

    denominator = np.sum(d*d, axis=1)
    J = np.divide(np.sum(residual*residual, axis=1), denominator,
                  out=np.full(len(d), np.nan), where=denominator > 0)
    with (output / 'scarti_finestra_20_25us.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rec_n', 'time_rescaled_s', 'relative_squared_L2_20_25us', 'max_abs_residual'])
        writer.writerows(zip(rec_n, anchor_time, J, np.max(np.abs(residual), axis=1)))
    np.savez_compressed(output / 'tracce_per_confronto.npz',
        time_us=acoustic_time, time_rescaled_s=anchor_time, rec_n=rec_n,
        observed=observed, synthetic=synthetic,
        dense_time_rescaled_s=dense_time, dense_window_time_us=y,
        dense_observed_window=dense.astype(np.float32),
        mechanical_time_s=mech_time, shear_stress_MPa=shear)
    provenance.update(common_color_limit=limit, residual_separate_color_limit=rmax,
                      acoustic_window_us=[20., 25.], actual_inversions=len(anchor_time),
                      dense_observed_waves=len(dense_time),
                      additional_trace_scaling=False, waveform_realignment=False,
                      temporal_interpolation_of_synthetics=False,
                      display='Ultima traccia mantenuta graficamente dal suo istante fino alla successiva; tacche nere = inversioni. Nessun nuovo forward entro le lacune.',
                      relative_squared_L2_window_median=float(np.nanmedian(J)))
    (output / 'provenienza.json').write_text(json.dumps(provenance, indent=2, ensure_ascii=False))


def export_context(namespace, context, selected, metadata, args):
    handler = context['uw_data_handler']
    infile = Path(context['infile_path'])
    if infile.name.split('.')[0] != args.basename:
        raise ValueError('Il file UW caricato non corrisponde al pickle richiesto.')
    data = np.asarray(handler.waveform_data)
    acoustic_time = np.asarray(handler.metadata['time_ax_waveform'], float)
    uw_time = np.asarray(handler.metadata['time_ax_acquisition'], float)
    if data.ndim != 2 or data.shape[1] != len(acoustic_time) or data.shape[0] != len(uw_time):
        raise ValueError('Assi UW e dimensioni delle onde non compatibili.')
    if np.any(np.diff(uw_time) <= 0) or np.any(np.diff(acoustic_time) <= 0):
        raise ValueError('Assi UW non crescenti.')
    idx = np.array([r['idx_wave'] for r in selected], int)
    if idx[-1] >= len(data) or idx[0] < 0:
        raise ValueError('Indici salvati fuori dalle onde caricate: controllare il downsampling della configurazione originale.')
    synthetic = np.stack([r['synthetic'] for r in selected])
    if synthetic.shape[1] != len(acoustic_time):
        raise ValueError('Durata/campionamento acustico diverso dalla run salvata. Ripristinare la sua configurazione.')
    # Recuperare i due sync del file selezionato dalla stessa configurazione.
    peaks = np.asarray(namespace['sync_peaks'], int)
    file_position = int(namespace['chosen_uw_file'])
    pair = peaks[2*file_position:2*file_position+2]
    mech = namespace['mech_data']
    mt = np.asarray(mech['time_s'], float)
    shear = np.asarray(mech['shear_stress_MPa'], float)
    if len(pair) != 2 or np.any(pair < 0) or np.any(pair >= len(mt)):
        raise ValueError('Coppia di sync non disponibile per il file UW.')
    start, stop = map(float, mt[pair])
    beta = (stop-start)/(uw_time[-1]-uw_time[0])
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError('Durate dei due orologi non valide.')
    scaled = start + beta*(uw_time-uw_time[0])
    # Gli indici sono indici UW; un downsampling cambiato può mantenere il
    # numero di campioni sufficiente ma associare onde sbagliate. Controllare
    # anche la pendenza del vecchio clock rispetto a idx_wave.
    saved_times = np.array([r['time_saved_s'] for r in selected])
    saved_step = np.median(np.diff(saved_times)/np.diff(idx))
    current_step = np.median(np.diff(uw_time))
    if not (np.isclose(saved_step, current_step, rtol=1e-6, atol=1e-9)
            or np.isclose(saved_step, current_step*beta, rtol=1e-6, atol=1e-9)):
        raise ValueError('Passo UW diverso da quello salvato: usare il preprocessing originale, senza cambiare downsampling.')
    all_idx = np.array([m[0] for m in metadata], int)
    dense_idx = np.arange(all_idx.min(), all_idx.max()+1)
    if dense_idx[-1] >= len(data):
        raise ValueError('La selezione salvata eccede le onde caricate.')
    output = args.output or args.results.parent / (args.basename + '_mappe_simulazioni')
    mask_mech = (mt >= scaled[dense_idx[0]]) & (mt <= scaled[dense_idx[-1]])
    source = Path(args.main)
    provenance = dict(results_path=str(args.results), results_bytes=args.results.stat().st_size,
                      main_path=str(source), main_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      input_uw_path=str(infile), source_of_synthetics='Tracce accettate già salvate nei checkpoint; nessun nuovo forward.',
                      clock='Ipotesi affine concordata: estremi UW sui due massimi dei sync meccanici; curva meccanica invariata.',
                      sync_start_s=start, sync_stop_s=stop, rescaling_factor=beta,
                      saved_step_s_per_idx=float(saved_step), current_uw_step_s=float(current_step),
                      acoustic_step_us=float(np.median(np.diff(acoustic_time))),
                      saved_anchor_time_s=saved_times.tolist(), anchor_idx_wave=idx.tolist(),
                      saved_misfit=[r['saved_misfit'] for r in selected],
                      note='La finestra diagnostica 20–25 µs può differire dalla finestra del misfit della run; il suo L2 viene ricalcolato e nominato separatamente.')
    print(f'Recuperate {len(selected)} tracce; scrittura mappe in {output}', flush=True)
    make_maps(output, acoustic_time=acoustic_time, anchor_time=scaled[idx],
              observed=np.asarray(data[idx], float), synthetic=synthetic,
              dense_time=scaled[dense_idx], dense_observed=np.asarray(data[dense_idx], float),
              rec_n=np.array([r['rec_n'] for r in selected]),
              mech_time=mt[mask_mech], shear=shear[mask_mech], provenance=provenance)
    print(f'Mappe pronte: {output}\nFile compatto da condividere: {output / "tracce_per_confronto.npz"}', flush=True)
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
    changes = set()
    body = []
    for node in blocks[0].body:
        names = [n.id for n in node.targets if isinstance(n, ast.Name)] if isinstance(node, ast.Assign) else []
        if 'wave_type' in names:
            node.value = ast.Constant(wave_type)
            changes.add('wave_type')
        if 'selected_uw_basename' in names:
            node.value = ast.Constant(basename)
            changes.add('selected_uw_basename')
        body.append(node)
        if 'params' in names:
            body.extend(ast.parse("params['diagnostic_plot_only'] = False\nparams['run_local_inversion'] = False").body)
            changes.add('params')
    if changes != {'wave_type', 'selected_uw_basename', 'params'}:
        raise ValueError(f'Configurazione principale non riconosciuta ({sorted(changes)}); nessuna esecuzione.')
    blocks[0].body = ast.parse('_map_disable_modeler(UltrasonicModeler)').body + body
    return ast.fix_missing_locations(tree)


def disable_modeler(modeler_class):
    def forbidden(*args, **kwargs):
        raise RuntimeError('Forward e inversioni sono disabilitati durante l’estrazione delle tracce salvate.')
    for name in dir(modeler_class):
        if name == 'forward_simulation' or name.startswith('run_local_'):
            setattr(modeler_class, name, forbidden)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--main', type=Path, default=Path('local_optimization_FWI_velocity_1D.py'),
                        help='Principale con le stesse impostazioni di preprocessing della run.')
    parser.add_argument('--results', type=Path, required=True,
                        help='Pickle originale completo, con i checkpoint; non .analysis.pkl.gz.')
    parser.add_argument('--output', type=Path, help='Cartella output; default accanto al pickle.')
    args = parser.parse_args(argv)
    args.main = args.main.expanduser().resolve()
    args.results = args.results.expanduser().resolve()
    if args.output:
        args.output = args.output.expanduser().resolve()
    if not args.main.is_file() or not args.results.is_file():
        parser.error('File principale o pickle non trovato.')
    args.basename = args.results.name.split('.')[0]
    match = re.search(r'_([PS])$', args.basename, re.I)
    if match is None or match.group(1).upper() != 'S':
        parser.error('Questa versione prepara la finestra S 20–25 µs; scegliere il risultato *_S.pkl.')
    sys.path.insert(0, str(args.main.parent))
    print('Lettura delle sole tracce accettate dal pickle completo...', flush=True)
    selected, metadata, _ = read_traces(args.results)
    tree = prepare_main(args.main.read_text(), wave_type='_s', basename=args.basename)
    namespace = {'__name__': '__main__', '__file__': str(args.main), '__package__': None,
                 '_map_export_context': lambda ns, ctx: export_context(ns, ctx, selected, metadata, args),
                 '_map_disable_modeler': disable_modeler}
    # Importa lab_uw dallo stesso progetto del principale. Le funzioni della
    # run restano disponibili ma il punto di avvio delle inversioni è sostituito.
    try:
        exec(compile(tree, str(args.main), 'exec'), namespace)
    except ExportCompleted:
        return 0
    raise RuntimeError('Nessun file ha raggiunto l’esportazione; verificare il principale indicato.')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        print(f'ERRORE: {error}', file=sys.stderr)
        raise SystemExit(2)
