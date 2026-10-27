"""Run the existing chronological data/noise gate before any forward solve."""
from __future__ import annotations
from copy import deepcopy
import csv,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import adaptive_sequence as sequence
from .adaptive_independent import _plan,_record_id
from .wave_difference_preinversion import _window


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if isinstance(value,np.ndarray):return [_plain(v) for v in value.tolist()]
    if isinstance(value,np.generic):return _plain(value.item())
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    if value is None or isinstance(value,(str,bool,int,float)):return value
    raise TypeError(f'Cannot export value of type {type(value).__name__}')


def _json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(_plain(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)


def select_and_export(functions, *, paired, observed_waveform_data, waveform_metadata,
                      stf_handler, params, assembly_dict, infile_path):
    """Return matched selected rows; write a portable P/S acquisition package.

    Signal/noise masks, reference calibration and persistence confirmation use
    the current adaptive modules. Only the inputs needed by their planner are
    supplied; neither a model nor a source calibration is needed to select data.
    """
    o=params.get('adaptive_noise_preselection',{})
    if not o.get('enabled',False):return None
    if paired.empty:raise ValueError('Adaptive preselection: mechanical selection is empty.')
    if not sequence.enabled(params):
        raise ValueError('Adaptive preselection requires adaptive_inversion.enabled=True.')
    s=sequence.settings(params)
    if float(params.get('local_spectral_gamma',1.))!=1.:
        raise ValueError('Adaptive preselection requires local_spectral_gamma=1.')
    for key in ('gate_calibration_records','gate_confirm_records'):
        if isinstance(s[key],bool) or int(s[key])!=s[key] or s[key]<1:
            raise ValueError(f'{key} must be a positive integer.')
    if not 0<s['gate_noise_quantile']<1 or s['gate_noise_factor']<=0 or s['gate_strong_factor']<1:
        raise ValueError('Invalid adaptive gate calibration/confirmation settings.')
    taper=float(params.get('local_objective_taper_fraction',0.))
    if not np.isfinite(taper) or not 0<=taper<=1:raise ValueError('Invalid objective taper.')
    data=np.asarray(observed_waveform_data);time=np.asarray(waveform_metadata['time_ax_waveform'])
    if data.ndim!=2 or time.ndim!=1 or data.shape[1]!=len(time):
        raise ValueError('Adaptive preselection waveform/time shapes differ.')
    if not np.isfinite(time).all() or np.any(np.diff(time)<=0):
        raise ValueError('Waveform time must be finite and strictly increasing.')
    rows=list(paired.itertuples(index=False));valid_positions=[]
    for position,row in enumerate(rows):
        # Match the mechanical-match condition in the original processing loop.
        if not np.isfinite(getattr(row,'Index',np.nan)):continue
        idx=float(row.uw_idx)
        if not np.isfinite(idx) or int(idx)!=idx or not 0<=idx<len(data):
            raise ValueError(f'Invalid uw_idx at selected position {position}.')
        valid_positions.append(position)
    if not valid_positions:raise ValueError('No waveform has a valid mechanical match.')
    valid=paired.iloc[valid_positions].copy();valid_rows=list(valid.itertuples(index=False))
    ids=np.asarray([int(r.uw_idx) for r in valid_rows],dtype=np.int64)
    times=np.asarray([float(r.uw_time_s) for r in valid_rows])
    if np.any(np.diff(ids)<=0) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Adaptive preselection requires unique chronological waveform rows.')
    # Original global search starts at the first acquisition passing its SNR gate.
    anchor=None
    snr_by_position={}
    for i,row in enumerate(valid_rows):
        first=deepcopy(assembly_dict)
        functions['update_assembly_dict_with_mech_data'](assembly_dict=first,mech_data=row)
        snr=float(functions['evaluate_snr'](time,data[int(row.uw_idx)],first['steel_only_time_wave']))
        snr_by_position[i]=snr
        if not np.isnan(snr) and snr>=float(params.get('minimum_SNR',0.)):
            anchor=i;break
    if anchor is None:raise ValueError('No acquisition passes minimum_SNR for the adaptive reference.')
    # _window computes the same analytic direct window as the global search.
    # _masks applies the adaptive signal/noise overrides exactly as before.
    if s['signal_window_us'] is None:
        initial,origin=_window(functions,valid_rows[anchor:],time,stf_handler,params,assembly_dict,{})
    else:
        initial=np.ones(len(time),dtype=bool)
        origin='adaptive_inversion.signal_window_us'
    signal,noise=sequence._masks(SimpleNamespace(observed_time=time,misfit_interval=initial),s)
    manifest=dict(anchor=anchor,signal_mask=signal,noise_mask=noise,
                  global_seed={'assembly_dict':deepcopy(assembly_dict)})
    planning_params=dict(params);planning_params.pop('_adaptive_preselection',None)
    plan=_plan(functions,manifest,valid_rows,data,planning_params)
    selected_positions=[i for i,e in plan.items() if e['selected']]
    selected=valid.iloc[selected_positions].copy()
    selected_uw=np.asarray(selected['uw_idx'],dtype=np.int64)
    selected_original=np.asarray([valid_positions[i] for i in selected_positions],dtype=np.int64)
    selected_entries=[deepcopy(plan[i]) for i in selected_positions]
    selected_records=np.asarray([e['rec_n'] for e in selected_entries],dtype=np.int64)
    if len(np.unique(selected_records))!=len(selected_records):
        raise ValueError('Selected mechanical record identifiers are not unique.')
    assembly_by_record=[]
    for row in selected.itertuples(index=False):
        a=deepcopy(assembly_dict)
        functions['update_assembly_dict_with_mech_data'](assembly_dict=a,mech_data=row)
        assembly_by_record.append(a)
    from .source_amplitude import validate_handler
    calibration=validate_handler(stf_handler)
    metadata=dict(schema=1,product_type='adaptive_noise_preselection',
        input_file=str(infile_path),wave_type=str(assembly_dict['wave_type']).lstrip('_').upper(),
        original_mechanical_selection_rows=len(paired),valid_mechanical_rows=len(valid),
        selected_waveforms=len(selected),skipped_before_reference=anchor,
        first_reference_rec_n=int(plan[anchor]['rec_n']),
        adaptive_settings=s,objective_taper_fraction=taper,signal_window_origin=origin,
        signal_window_us=[float(time[signal][0]),float(time[signal][-1])],
        noise_window_us=[float(time[noise][0]),float(time[noise][-1])],
        source_calibrated=calibration is not None,
        source_calibration=calibration,source_metadata=stf_handler.metadata,
        processing=dict(additional_filter=False,amplitude_refit=False,time_shift=False,
                        forward_evaluations=0,inversions=0),
        planner='adaptive_independent._plan with the current adaptive settings',
        mapping='selected_waveforms[i] and selected_paired_rows_json[i] describe the same acquisition',
        params=params)
    # Avoid exporting transient references or recursively embedding an old plan.
    metadata['params']={k:v for k,v in params.items() if not str(k).startswith('_')}
    metadata=_plain(metadata)
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode())
    for arr in [time,stf_handler.waveform_data,np.asarray(stf_handler.metadata['time_ax_waveform'])]:
        digest.update(np.ascontiguousarray(arr).tobytes())
    for row in valid_rows:digest.update(np.ascontiguousarray(data[int(row.uw_idx)]).tobytes())
    digest.update(json.dumps(_plain([r._asdict() for r in valid_rows]),sort_keys=True).encode())
    root=Path(params['outdir_path_l2norm'])/'adaptive_noise_preselection'
    out=root/f'{Path(infile_path).stem}_{digest.hexdigest()[:12]}'
    out.mkdir(parents=True,exist_ok=True)
    prefix=str(assembly_dict['wave_type']).lstrip('_').upper()
    dest=out/f'{prefix}_adaptive_selected.npz';tmp=dest.with_suffix('.tmp')
    mechanical_json=np.asarray([json.dumps(_plain(row._asdict()),allow_nan=False)
                                for row in selected.itertuples(index=False)])
    with tmp.open('wb') as f:np.savez_compressed(f,selected_waveforms=data[selected_uw],
        time_us=time,original_uw_indices=selected_uw,original_paired_positions=selected_original,
        record_numbers=selected_records,selected_paired_columns=np.asarray(selected.columns,dtype=str),
        selected_paired_rows_json=mechanical_json,signal_mask=signal,noise_mask=noise,
        source_waveform=np.asarray(stf_handler.waveform_data),
        source_time_us=np.asarray(stf_handler.metadata['time_ax_waveform']),
        assembly_by_record_json=np.asarray([json.dumps(_plain(a),allow_nan=False) for a in assembly_by_record]),
        selected_gate_entries_json=np.asarray([json.dumps(_plain(e),allow_nan=False) for e in selected_entries]),
        metadata_json=np.array(json.dumps(metadata,allow_nan=False)))
    tmp.replace(dest)
    selected.to_csv(out/'selected_mechanical_data.csv',index=False,float_format='%.17g')
    # Include every decision, with an explicit reason for unmatched/SNR rows.
    columns=['original_paired_position','original_uw_idx','rec_n','selected','reference_rec_n',
             'gate_status','data_reference_misfit','noise_pair_misfit','calibrated_threshold',
             'effective_threshold','gate_confirmation_count']
    reverse_valid={v:i for i,v in enumerate(valid_positions)}
    with (out/'all_gate_decisions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for pos,row in enumerate(rows):
            i=reverse_valid.get(pos);e=plan.get(i,{}) if i is not None else {}
            decision=e.get('decision',{})
            status=('unmatched_mechanical_row' if i is None else 'below_SNR_before_reference' if i<anchor
                    else 'first_reference' if i==anchor else decision.get('gate_status','unknown'))
            record=e.get('rec_n')
            if record is None and i is not None:record=_record_id(functions,assembly_dict,valid_rows[i])
            item=dict(original_paired_position=pos,original_uw_idx=row.uw_idx,rec_n=record,
                      selected=bool(e.get('selected',False)),gate_status=status)
            item.update({k:decision.get(k) for k in columns if k not in item})
            w.writerow(item)
    _json(out/'selection_summary.json',metadata)
    frozen=dict(fingerprint=digest.hexdigest(),selected_records=selected_records.tolist(),selected_uw_indices=selected_uw.tolist(),
                signal_mask=signal.copy(),noise_mask=noise.copy(),entries=selected_entries)
    params['_adaptive_preselection']=frozen
    print(f'Adaptive noise preselection: {len(selected)}/{len(valid)} mechanically matched waves; '
          f'no forward calculation. Saved {dest}',flush=True)
    return dict(paired=selected,waveforms=data[selected_uw],path=dest,plan=plan,manifest=manifest,
                selected_positions=selected_positions,selected_original_positions=selected_original)


def frozen_plan(functions,manifest,paired_rows,params):
    """Consume the already-decided selection; do not gate the sparse list again."""
    f=params['_adaptive_preselection']
    records=[_record_id(functions,manifest['global_seed']['assembly_dict'],r) for r in paired_rows]
    if manifest['anchor']!=0 or records!=f['selected_records'] or [int(r.uw_idx) for r in paired_rows]!=f['selected_uw_indices']:
        raise ValueError('Preselected rows do not match the frozen adaptive plan.')
    for key in ('signal_mask','noise_mask'):
        if not np.array_equal(manifest[key],f[key]):raise ValueError('Adaptive preselection windows changed.')
    return {i:dict(position=i,selected=True,reference_position=max(0,i-1),
                   decision=deepcopy(e['decision']),rec_n=records[i],
                   original_preselection_position=int(e['position']))
            for i,e in enumerate(f['entries'])}
