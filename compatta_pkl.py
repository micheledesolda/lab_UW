#!/usr/bin/env python3
"""Esporta una copia per analisi dei risultati FWI, senza dati di ripresa.

Uso: python3 compatta_pkl.py risultati.pkl [copia.analysis.pkl.gz]
L'originale resta invariato. Il risultato NON permette di riprendere la FWI.
Il formato resta un flusso di dizionari pickle; il suffisso .gz attiva gzip.
Dipendenza: numpy. Usare su file di risultati fidati, a scrittura terminata.
"""
import argparse
import gzip
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ARRAY_KEYS = {
    'x_axis_cm', 'velocity_profile', 'damping_profile', 'density_profile',
    *(f'gouge_{i}_{field}' for i in (1, 2)
      for field in ('eta', 'velocity', 'damping', 'density')),
}
META_KEYS = {
    'adaptive_inversion', 'adaptive_independent', 'adaptive_state',
    'adaptive_configuration', 'adaptive_update_gain',
    'kinematic_diagnostics', 'local_stopping', 'objective_components',
    'acquisition_sync_calibration', 'analysis_assembly', 'analysis_runtime',
    'local_iteration_history_summary',
}
DROP_KEYS = {
    'local_continuation_state', 'global_continuation_state',
    'local_iteration_history', 'continuation', 'simulation_handler',
    'optimization_resume_state', 'scientific_stop_resume_state',
    '_kinematic_reference', '_kinematic_cache',
}
MISSING = object()


def small(value):
    """Conserva metadati piccoli, evitando stati e array numerici pesanti."""
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in DROP_KEYS:
                continue
            cleaned = small(item)
            if cleaned is not MISSING:
                result[key] = cleaned
        return result
    if isinstance(value, np.ndarray) and value.size <= 64:
        return value.copy()
    if isinstance(value, (list, tuple)) and len(value) <= 64:
        items = [small(item) for item in value]
        if all(item is not MISSING for item in items):
            return tuple(items) if isinstance(value, tuple) else items
    return MISSING


def compact_record(row):
    if not isinstance(row, dict):
        raise ValueError('Il flusso contiene un oggetto diverso da un dizionario.')
    # Mantieni anche scalari futuri non ancora noti a questo esportatore.
    out = {key: small(value) for key, value in row.items()
           if key not in DROP_KEYS and (value is None or isinstance(
               value, (bool, int, float, complex, str, np.generic, Path)))}
    for key in ARRAY_KEYS:
        if key in row and row[key] is not None:
            out[key] = np.asarray(row[key]).copy()
    for key in META_KEYS:
        if key in row:
            cleaned = small(row[key])
            if cleaned is not MISSING:
                out[key] = cleaned

    state = row.get('local_continuation_state')
    if not isinstance(state, dict):
        state = row.get('global_continuation_state')
    if isinstance(state, dict):
        # Estrai le informazioni materiali PRIMA di scartare il checkpoint.
        for old, new in [('grid_spatial_axis', 'x_axis_cm'),
                         ('velocity_array', 'velocity_profile'),
                         ('damping_array', 'damping_profile'),
                         ('density_array', 'density_profile')]:
            if out.get(new) is None and state.get(old) is not None:
                out[new] = np.asarray(state[old]).copy()
        indices = state.get('idx_dict', {})
        out['idx_dict'] = {name: np.asarray(ii).copy()
                           for name, ii in indices.items()}
        density = state.get('density_array')
        if density is not None:
            density = np.asarray(density)
            for i in (1, 2):
                name = f'gouge_{i}'
                if name in indices and f'{name}_velocity' in out:
                    ii = np.asarray(indices[name])
                    values = density[ii].copy()
                    if len(values) != len(out[f'{name}_velocity']):
                        raise ValueError(f'Densita e profilo incompatibili: {name}.')
                    out[f'{name}_density'] = values
        out['analysis_assembly'] = small(state.get('assembly_dict', {}))
        runtime = state.get('restart_runtime', {})
        # Conserva le opzioni numeriche scalari, non segnali e orologi completi.
        out['analysis_runtime'] = {
            group: {key: small(value) for key, value in runtime.get(group, {}).items()
                    if value is None or isinstance(value, (bool, int, float, str, np.generic))}
            for group in ('grid', 'simulation')
        }
    elif isinstance(row.get('idx_dict'), dict):
        out['idx_dict'] = {name: np.asarray(ii).copy()
                           for name, ii in row['idx_dict'].items()}

    history = row.get('local_iteration_history')
    if isinstance(history, (list, tuple)) and history:
        entries = [item for item in history if isinstance(item, dict)]
        if entries:
            out['local_iteration_history_summary'] = {
                'entries': len(entries),
                'first': small(entries[0]), 'last': small(entries[-1]),
                'accepted_entries': sum(bool(item.get('accepted')) for item in entries),
            }
    kind = ('global_reference' if 'global_best_misfit' in row else
            'profile' if 'gouge_1_velocity' in row else 'other')
    out['_analysis_export'] = {
        'schema': 1, 'record_kind': kind, 'resumable': False,
        'units': 'originali: nessuna conversione; densita nelle unita del modello',
    }
    return out


def export(source, destination=None):
    source = Path(source).expanduser().resolve()
    basename = source.name.removesuffix('.gz').removesuffix('.pkl')
    destination = (Path(destination).expanduser().resolve() if destination else
                   source.with_name(basename + '.analysis.pkl.gz'))
    if source == destination or destination.exists():
        raise ValueError(f'Non sovrascrivo file esistenti: {destination}')
    if not destination.parent.is_dir():
        raise ValueError(f'Cartella di destinazione assente: {destination.parent}')
    before = source.stat()
    fd, temporary = tempfile.mkstemp(prefix='.' + destination.name + '.', dir=destination.parent)
    counts = Counter()
    try:
        opener = gzip.open if source.suffix == '.gz' else open
        with os.fdopen(fd, 'wb') as raw, opener(source, 'rb') as reader:
            writer = (gzip.GzipFile(filename='', mode='wb', fileobj=raw,
                                   compresslevel=6, mtime=0)
                      if destination.suffix == '.gz' else raw)
            try:
                while reader.peek(1):
                    offset = reader.tell()
                    try:
                        row = pickle.load(reader)
                    except (EOFError, pickle.UnpicklingError) as error:
                        raise ValueError(f'Pickle incompleto o corrotto al byte {offset}; esportazione annullata.') from error
                    cleaned = compact_record(row)
                    pickle.dump(cleaned, writer, protocol=pickle.HIGHEST_PROTOCOL)
                    counts['records'] += 1
                    counts[cleaned['_analysis_export']['record_kind']] += 1
                    action = (row.get('adaptive_inversion') or {}).get('action', 'legacy')
                    counts[f'action:{action}'] += 1
                    if counts['records'] % 1000 == 0:
                        print(f"Esportati {counts['records']} record", flush=True)
                    del row, cleaned
            finally:
                if writer is not raw:
                    writer.close()
            raw.flush()
            os.fsync(raw.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Il file originale e cambiato durante la lettura. Riprova a scrittura terminata.')
        if not counts['records']:
            raise ValueError('Il file non contiene record.')
        # Pubblicazione senza sovrascrittura, anche se un altro processo ha
        # creato nel frattempo il file di destinazione.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    size = destination.stat().st_size
    print(f'Salvato: {destination}')
    print(f'Record: {dict(counts)}')
    print(f'Dimensione: {before.st_size / 2**20:.2f} -> {size / 2**20:.2f} MiB '
          f'({100 * size / before.st_size:.2f}% dell\'originale)')
    print('Originale conservato. La copia esportata serve per analisi, non per riprendere la FWI.')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path, nargs='?')
    args = parser.parse_args()
    try:
        export(args.source, args.destination)
    except (OSError, ValueError, TypeError, IndexError, KeyError, ImportError) as error:
        parser.exit(2, f'Esportazione interrotta: {error}\n')
