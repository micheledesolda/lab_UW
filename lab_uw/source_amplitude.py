"""Transfer the steel-reference gain once; never profile the DDS amplitude."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

PRODUCT = 'reference_calibrated_effective_source'


def wave_type(value):
    value = str(value).lower().lstrip('_')
    if value not in ('p', 's'):
        raise ValueError('La polarizzazione deve essere P oppure S.')
    return value


def arrays(time, waveform):
    time, waveform = (np.asarray(a, dtype='<f8').reshape(-1) for a in (time, waveform))
    if len(time) < 3 or time.shape != waveform.shape:
        raise ValueError('Asse temporale e sorgente devono avere almeno tre campioni e uguale lunghezza.')
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(waveform)):
        raise ValueError('La sorgente contiene campioni non finiti.')
    if np.any(np.diff(time) <= 0) or abs(time[0]) > 1.e-10 or not np.any(waveform):
        raise ValueError('La sorgente deve avere tempo relativo crescente da zero ed energia positiva.')
    return time, waveform


def source_id(time, waveform, metadata):
    values = {k: v for k, v in metadata.items() if k != 'id'}
    h = hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode())
    for a in arrays(time, waveform):
        h.update(a.tobytes())
    return h.hexdigest()


def validate_handler(handler):
    meta = getattr(handler, 'metadata', {}).get('source_calibration')
    if meta is None:
        return None
    if meta.get('schema') != 1 or meta.get('product_type') != PRODUCT:
        raise ValueError('Formato della sorgente calibrata non riconosciuto.')
    if source_id(handler.metadata['time_ax_waveform'], handler.waveform_data, meta) != meta.get('id'):
        raise ValueError('Sorgente calibrata modificata dopo il salvataggio: ricalibrare senza troncarla o riscalarla.')
    wave_type(meta['wave_type'])
    return meta


def load_source(path, *, expected_wave_type=None, frequency_min=None, frequency_max=None):
    with np.load(Path(path).expanduser(), allow_pickle=False) as data:
        time, waveform = arrays(data['time_us'], data['waveform'])
        meta = json.loads(str(data['metadata_json'].item()))
    handler = SimpleNamespace(waveform_data=waveform.copy(), metadata={
        'time_ax_waveform': time.copy(), 'number_of_samples': len(time),
        'number_of_waveforms': 1, 'product_type': PRODUCT, 'source_calibration': meta})
    validate_handler(handler)
    if expected_wave_type is not None and wave_type(expected_wave_type) != meta['wave_type']:
        raise ValueError('Sorgente P/S incompatibile con la polarizzazione richiesta.')
    for key, value in [('frequency_min_mhz', frequency_min), ('frequency_max_mhz', frequency_max)]:
        if value is not None and not np.isclose(float(value), float(meta[key]), rtol=1.e-10, atol=1.e-12):
            raise ValueError(f'Banda della sorgente incompatibile: {key}={meta[key]}, richiesto {value}.')
    return handler


def truncate_case(case, duration_us):
    """Crop only before calibration; DDS will load this exact waveform."""
    duration = float(duration_us)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('--source-duration-us deve essere positivo e finito.')
    mask = case.wavelet_time_relative_us <= duration + 1.e-12
    if np.count_nonzero(mask) < 3:
        raise ValueError('Durata troppo breve per calibrare la sorgente.')
    return replace(case, wavelet_time_absolute_us=case.wavelet_time_absolute_us[mask].copy(),
                   wavelet_waveform=case.wavelet_waveform[mask].copy())


def export_reference_source(path, *, case, geometry, config, score, provenance=None):
    """The sole fitted scalar is determined on the known steel direct packet."""
    expected = {'embedding_mode': 'free_steel', 'boundary_condition': 'free',
                'source_injection': 'body_force', 'receiver_quantity': 'velocity',
                'receiver_normalization': 'sum', 'spatial_boundary_mode': 'truncate',
                'source_inset_cm': 0., 'receiver_inset_cm': 0.,
                'source_extension_cm': 0., 'receiver_extension_cm': 0.}
    for key, value in expected.items():
        if getattr(config, key) != value:
            raise ValueError(f'Calibrazione trasferibile al DDS richiede {key}={value!r}.')
    if config.resolved_source_normalization() != 'integral':
        raise ValueError('La sorgente body_force deve avere normalizzazione spaziale integral.')
    gain = float(config.source_amplitude) * float(score.amplitude_scale)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Guadagno di riferimento non positivo: controllare polarità e arrivo diretto.')
    time, waveform = arrays(case.wavelet_time_relative_us, gain * case.wavelet_waveform)
    meta = {'schema': 1, 'product_type': PRODUCT, 'wave_type': wave_type(geometry.wave_type),
            'reference_case': case.stem, 'reference_gain': gain,
            'reference_trace_scale': float(score.amplitude_scale),
            'reference_source_amplitude': float(config.source_amplitude),
            'frequency_min_mhz': case.filter_freqmin_mhz, 'frequency_max_mhz': case.filter_freqmax_mhz,
            'duration_us': float(time[-1]), 'units': {'time': 'us', 'length': 'cm'},
            'geometry': asdict(geometry), 'reference_forward': asdict(config),
            'operator': {'source_injection': 'body_force', 'receiver_quantity': 'velocity',
                         'boundary_condition': 'free', 'source_normalization': 'integral',
                         'receiver_normalization': 'sum', 'spatial_boundary_mode': 'truncate',
                         'interpolation_radius_cells': int(config.interpolation_radius_cells)},
            'filter_order': (provenance or {}).get('filter_order'),
            'wavelet_filter_mode': case.wavelet_filter_mode,
            'reference_observed_sha256': hashlib.sha256(np.asarray(case.observed_waveform, dtype='<f8').tobytes()).hexdigest(),
            'provenance': provenance or {}}
    if meta['frequency_min_mhz'] is None or meta['frequency_max_mhz'] is None:
        raise ValueError('Per esportare specificare entrambi i limiti della banda filtrata.')
    meta['id'] = source_id(time, waveform, meta)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, time_us=time, waveform=waveform,
                                metadata_json=json.dumps(meta, sort_keys=True, allow_nan=False))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return meta


def bind_forward(model, arguments):
    """Called before any DDS or teeth-bank dispatch, including global workers."""
    meta = validate_handler(arguments['stf_handler'])
    model.source_calibration = deepcopy(meta)
    if meta is None:
        return False
    if arguments['embedding_mode'] != 'effective_surface':
        raise ValueError('La sorgente calibrata richiede embedding_mode=effective_surface.')
    assembly = arguments['assembly_dict']
    if wave_type(assembly['wave_type']) != meta['wave_type']:
        raise ValueError('Polarizzazione del DDS diversa dalla sorgente calibrata.')
    if not np.isclose(float(arguments['frequency_cutoff']), meta['frequency_max_mhz']):
        raise ValueError('Cutoff del forward diverso dalla calibrazione della sorgente.')
    for key in ('boundary_condition', 'source_injection', 'receiver_quantity'):
        if arguments.get(key) not in (None, meta['operator'][key]):
            raise ValueError(f'Operatore {key} incompatibile con la sorgente calibrata.')
    if arguments.get('absorbing') or arguments.get('boundary_impedance') is not None:
        raise ValueError('Condizioni al contorno incompatibili con la sorgente calibrata.')
    cfg = arguments.get('source_receiver_config') or {}
    mc = arguments.get('montecarlo') or {}
    if float(cfg.get('source_amplitude', mc.get('multiplier_STF', 1.))) != 1.:
        raise ValueError('Il guadagno è già nella sorgente calibrata: source_amplitude deve essere 1.')
    expected = {'source_normalization': cfg.get('spatial_normalization', 'integral'),
                'receiver_normalization': 'sum', 'spatial_boundary_mode': 'truncate',
                'interpolation_radius_cells': meta['operator']['interpolation_radius_cells']}
    for key, default in expected.items():
        if cfg.get(key, default) != meta['operator'][key]:
            raise ValueError(f'{key} diverso da quello della calibrazione.')
    radius = int(meta['operator']['interpolation_radius_cells'])
    for key in ('source_radius_cells', 'receiver_radius_cells'):
        if int(cfg.get(key, cfg.get('interpolation_radius_cells', 4))) != radius:
            raise ValueError(f'{key} diverso da quello della calibrazione.')
    for key in ('source_extension_cm', 'receiver_extension_cm'):
        if float(cfg.get(key, 0.)) != 0.:
            raise ValueError(f'{key} deve essere zero per la calibrazione sulle facce.')
    if float(cfg.get('source_position_cm', 0.)) != 0. or 'receiver_position_cm' in cfg:
        raise ValueError('La calibrazione usa le facce esterne: rimuovere gli override di posizione.')
    model.reference_trace_scale = 1.
    model.amplitude_scale_mode = 'fixed'
    return True


def require_unit_scale(model, scale):
    if getattr(model, 'source_calibration', None) is not None and scale is not None and float(scale) != 1.:
        raise ValueError('Sorgente già calibrata: un secondo trace_scale altererebbe la calibrazione.')


def annotate_source(row, handler):
    meta = validate_handler(handler)
    if meta is not None:
        row.update(source_calibration_id=meta['id'], source_reference_gain=meta['reference_gain'],
                   source_reference_case=meta['reference_case'],
                   amplitude_calibration_policy='fixed_steel_reference')


def require_saved_source(row, handler):
    meta = validate_handler(handler)
    current = meta['id'] if meta else None
    if row.get('source_calibration_id') != current:
        raise ValueError('La sorgente calibrata è cambiata rispetto ai risultati salvati: usare una nuova cartella di analisi.')


def require_checkpoint_source(model, state):
    saved = state['simulation_attributes'].get('source_calibration')
    current = getattr(model, 'source_calibration', None)
    if (saved or {}).get('id') != (current or {}).get('id'):
        raise ValueError('Checkpoint e sorgente calibrata non corrispondono.')


def preprocess_dds(handler, *, params, assembly, source):
    """Filter a full record once, before the final DDS time crop."""
    meta = validate_handler(source)
    if meta is None:
        raise ValueError('Questo preprocessing richiede una sorgente calibrata.')
    order = int(meta.get('filter_order') or 4)
    return (handler.bandpass(freqmin=params['frequency_min'], freqmax=params['frequency_cutoff'], order=order)
            .truncate_time(params['maxtime2simulate'])
            .zero_before(assembly['steel_only_time_wave'] / 5.))
