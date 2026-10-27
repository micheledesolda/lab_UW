"""Independent adaptive FWI using the workflow's existing recycled workers.

Selection is chronological and data-only. Selected inversions are independent:
each starts from the SAME global/local reference snapshot, with new curvature
memory. A compact per-wave spool survives out-of-order completion and resumes.
Only the parent appends the chronological result stream and comparison CSV.
"""
from __future__ import annotations
from copy import deepcopy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np

from . import adaptive_sequence as sequence
from .adaptive_diagnostics import confirm_gate, record_gate
from .kinematic_velocity import ensure_reference, reference_from_state, result_fields
from .material_optimizer import settings as optimizer_settings, policy_digest, policy_changes
from .orchestration_restart import enrich_local_checkpoint, merge_iteration_history

SCHEMA=1
REGIONS=('groove_sb1','gouge_1','groove_cb1','groove_cb2','gouge_2','groove_sb2')

def is_independent(params):
    return sequence.enabled(params) and sequence.settings(params)['mode']=='independent'

def _atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            pickle.dump(value,f,protocol=pickle.HIGHEST_PROTOCOL);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()

def _load(path):
    with Path(path).open('rb') as f:return pickle.load(f)

def _directory(results_pkl):
    path=Path(results_pkl)
    return path.with_name(path.name+'.independent_jobs')

def _job_path(directory,position):
    return Path(directory)/f'wave_{int(position):08d}.pkl'

def _record_id(functions,assembly,pair):
    # The workflow supports both records_na and mechanical Index formats.
    value=deepcopy(assembly)
    functions['update_assembly_dict_with_mech_data'](assembly_dict=value,mech_data=pair)
    return int(value['rec_n'])

def _seed_hash(state):
    h=hashlib.sha256()
    for key in ('grid_spatial_axis','velocity_array','damping_array','density_array',
                'source_spatial_function','receiver_spatial_function'):
        a=np.ascontiguousarray(state[key]);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    rt=state['restart_runtime']
    for value in (rt['source_time_function'],rt['simulation_time']['simulation_time']):
        a=np.ascontiguousarray(value);h.update(a.tobytes())
    h.update(json.dumps({k:np.asarray(v).tolist() for k,v in state['idx_dict'].items()},sort_keys=True).encode())
    scale=state['simulation_attributes'].get('reference_trace_scale',1.)
    h.update(str(float(scale)).encode())
    return h.hexdigest()

def _fresh_state(state):
    state=deepcopy(state)
    attrs=state['simulation_attributes']
    # Seed profiles and global reference are common. Optimizer state is NOT.
    for name in ('local_stop_diagnostics','optimization_resume_state','scientific_stop_resume_state',
                 'optimization_policy','scientific_stop_policy'):
        attrs.pop(name,None)
    attrs['local_inversion_history']=[]
    attrs['_kinematic_cache']={}
    return state

def _restore(functions,simulation,state,observed,time,*,exact=False):
    """Restore the complete clock/source as well as the compact material state."""
    assembly=functions['_restore_local_continuation_state'](simulation,state)
    rt=state['restart_runtime'];saved=rt['simulation']
    if not np.array_equal(time,saved['observed_time']):
        raise ValueError('Independent waveform time axis differs from the reference checkpoint.')
    if exact and not np.array_equal(observed,saved['observed_waveform']):
        raise ValueError('Pending checkpoint does not contain this SAME waveform.')
    for k,v in rt['grid'].items():setattr(simulation.grid_handler,k,deepcopy(v))
    for k,v in rt['simulation_time'].items():setattr(simulation.sim_time_handler,k,deepcopy(v))
    for k,v in saved.items():
        if k!='observed_waveform':setattr(simulation,k,deepcopy(v))
    simulation.source_handler.time_function=rt['source_time_function'].copy()
    simulation.observed_waveform=np.asarray(observed).copy()
    simulation.wavefield_forward=None
    return assembly

def _clear_transients(simulation):
    # The old scheduler deepcopy's its seed in each fork. Do not copy a stale
    # O(nx*nt) forward/adjoint field into every worker.
    for name,value in list(vars(simulation).items()):
        if isinstance(value,np.ndarray) and value.ndim>=2:setattr(simulation,name,None)
    simulation.wavefield_forward=None
    return simulation

def _read_prefix(path,paired_rows,*,fingerprint=None):
    """Retain just the global row and last compact local row; validate ordering."""
    positions={int(r.uw_idx):i for i,r in enumerate(paired_rows)}
    global_row=None;last=None;completed=[];end=0;truncated=None
    with Path(path).open('rb') as stream:
        size=Path(path).stat().st_size
        while stream.tell()<size:
            try:row=pickle.load(stream)
            except EOFError:truncated=end;break
            except pickle.UnpicklingError as error:
                if 'truncated' in str(error).lower():truncated=end;break
                raise
            end=stream.tell()
            if 'global_best_misfit' in row:
                if global_row is not None or completed:raise ValueError('Independent stream has multiple/misordered global rows.')
                global_row=row
            else:
                meta=row.get('adaptive_independent')
                if not meta or meta.get('schema')!=SCHEMA:raise ValueError('Sequential/legacy results cannot be mixed with independent results.')
                if fingerprint is not None and meta['fingerprint']!=fingerprint:raise ValueError('Saved independent result fingerprint differs.')
                position=positions[int(row['idx_wave'])]
                if position!=int(meta['position']):raise ValueError('Saved independent record position differs.')
                completed.append(position);last=row
    if global_row is None:raise ValueError('No global reference checkpoint in the independent stream.')
    anchor=positions[int(global_row['idx_wave'])]
    if completed!=list(range(anchor,anchor+len(completed))):
        raise ValueError('Independent result stream must be a chronological prefix without duplicates or gaps.')
    return global_row,last,anchor,anchor+len(completed),truncated

def _repair_tail(path,truncate_to):
    if truncate_to is None:return
    path=Path(path)
    with path.open('rb') as f:f.seek(truncate_to);tail=f.read()
    backup=path.with_name(path.name+'.incomplete_tail_'+hashlib.sha256(tail).hexdigest()[:12]+'.bak')
    if not backup.exists():backup.write_bytes(tail)
    elif backup.read_bytes()!=tail:raise ValueError('Conflicting incomplete-tail backup.')
    with path.open('r+b') as f:f.truncate(truncate_to)
    print(f'Independent resume: incomplete tail retained at {backup}.',flush=True)

def _plan(functions,manifest,paired_rows,data,params):
    if params.get('_adaptive_preselection') is not None:
        from .adaptive_preselection import frozen_plan
        return frozen_plan(functions,manifest,paired_rows,params)
    s=sequence.settings(params);signal=manifest['signal_mask'];noise=manifest['noise_mask']
    taper=float(params.get('local_objective_taper_fraction',0.));anchor=manifest['anchor']
    ref_position=anchor;ref=np.asarray(data[int(paired_rows[anchor].uw_idx)])
    threshold,calibration=sequence._reference_threshold(ref,anchor,paired_rows,data,signal,noise,taper,s)
    gate_state={'gate_confirmation_count':0};entries={}
    rec_ids={i:_record_id(functions,manifest['global_seed']['assembly_dict'],paired_rows[i])
             for i in range(anchor,len(paired_rows))}
    for position in range(anchor,len(paired_rows)):
        row=paired_rows[position];observed=np.asarray(data[int(row.uw_idx)])
        if not np.all(np.isfinite(observed[signal])):
            raise ValueError(f'Nonfinite selected waveform at position {position}; no independent jobs created.')
        decision={};selected=True;old_ref=ref_position
        if position!=anchor:
            compatible,decision=sequence.gate_decision(ref,observed,signal,noise,taper,threshold,
                                                     functions['relative_waveform_objective'],s)
            if s['gate_enabled']:
                selected=not confirm_gate(gate_state,decision,compatible,s)
            else:
                selected=True
                decision.update(gate_status='gate_disabled',gate_confirmation_count=0,will_optimize=True)
            decision.update(reference_rec_n=rec_ids[old_ref],current_rec_n=rec_ids[position],
                reference_time_s=float(paired_rows[old_ref].time_s),current_time_s=float(row.time_s))
        entries[position]={'position':position,'selected':selected,'reference_position':old_ref,
                           'decision':decision,'rec_n':rec_ids[position]}
        if selected:
            ref_position=position;ref=observed;gate_state['gate_confirmation_count']=0
            threshold,calibration=sequence._reference_threshold(ref,position,paired_rows,data,signal,noise,taper,s)
    return entries

def _diagnostics(manifest,paired_rows,data,time,params,prefix):
    """Parent-only, repeatable diagnostics: one CSV row for each planned pair."""
    directory=Path(params['outdir_path_image'])/'reference_comparisons'
    directory.mkdir(parents=True,exist_ok=True);rows=[];s=sequence.settings(params)
    for position,entry in manifest['plan'].items():
        decision=entry['decision']
        if not decision:continue
        ref=entry['reference_position'];stem=f'{prefix}_ref_{decision["reference_rec_n"]}_rec_n_{decision["current_rec_n"]}_data_change'
        plot=(s['plot_gate_updates'] and entry['selected']) or s['plot_gate_reused_records'] or decision['gate_status']=='awaiting_confirmation'
        output=record_gate(time=time,reference=data[int(paired_rows[ref].uw_idx)],
            current=data[int(paired_rows[position].uw_idx)],signal_mask=manifest['signal_mask'],
            noise_mask=manifest['noise_mask'],decision=decision,output_directory=directory,prefix=prefix,
            write_plot=bool(plot and not (directory/(stem+'.png')).exists()),append_csv=False)
        rows.append(output)
    columns=['reference_rec_n','current_rec_n','reference_time_s','current_time_s',
             'data_reference_misfit','relative_L2_norm','noise_pair_misfit','effective_threshold',
             'gate_status','gate_confirmation_count']
    target=directory/f'{prefix}_gate_decisions.csv'
    fd,temporary=tempfile.mkstemp(prefix='.'+target.name+'.',dir=directory)
    try:
        with os.fdopen(fd,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
            f.flush();os.fsync(f.fileno())
        os.replace(temporary,target)
    finally:
        if Path(temporary).exists():Path(temporary).unlink()

def _check_job(row,manifest,position,*,seed_kind,seed_hash):
    meta=row.get('adaptive_independent') or {}
    expected={'schema':SCHEMA,'fingerprint':manifest['fingerprint'],'position':int(position),
              'initial_model_kind':seed_kind,'initial_model_hash':seed_hash}
    if any(meta.get(k)!=v for k,v in expected.items()):
        raise ValueError(f'Independent checkpoint belongs to a different waveform/seed: position {position}.')

def _run_one(functions,context,position,*,state,seed_kind,seed_hash,pending=None,role='wave'):
    manifest=context['params']['_adaptive_independent_context'];params=context['params']
    s=sequence.settings(params);pair=context['paired_rows'][position]
    observed=np.asarray(context['observed_waveform_data'][int(pair.uw_idx)])
    time=manifest['global_seed']['restart_runtime']['simulation']['observed_time']
    simulation=deepcopy(context['simulation_seed']);previous_history=[]
    local_params={k:v for k,v in params.items() if k!='_adaptive_independent_context' and not k.startswith('_adaptive_resume')}
    for k in ('_adaptive_reuse','_adaptive_first_reference'):local_params.pop(k,None)
    if pending is not None:
        _check_job(pending,manifest,position,seed_kind=seed_kind,seed_hash=seed_hash)
        assembly=_restore(functions,simulation,pending['local_continuation_state'],observed,time,exact=True)
        stopping=pending['local_stopping'];previous_history=deepcopy(pending['local_iteration_history'])
        local_params['_adaptive_resume_monitor']=deepcopy(stopping['continuation'])
        local_params['_adaptive_resume_limits']=deepcopy(stopping['step_limits'])
    else:
        assembly=_restore(functions,simulation,_fresh_state(state),observed,time)
        simulation.local_inversion_history=[]
        simulation.local_stop_diagnostics=None
    functions['update_assembly_dict_with_mech_data'](assembly_dict=assembly,mech_data=pair)
    assembly['_local_plot_counter']=position-manifest['anchor']
    simulation.adaptive_signal_mask=manifest['signal_mask'].copy()
    simulation.adaptive_noise_mask=manifest['noise_mask'].copy()
    simulation.independent_velocity_lower_bound_cm_us=manifest['velocity_lower_bound_cm_us']
    local_params['_adaptive_reuse']=False
    local_params['_adaptive_first_reference']=role=='reference'
    local_params['n_iterations']=int(s['first_max_iterations'] if role=='reference' else s['update_max_iterations'])
    grid_before=simulation.grid_handler.spatial_axis.copy()
    suffix='_reference_local' if role=='reference' else ''
    result=functions['local_refine_waveform'](outfile_name=f'{context["outfile_name"]}{suffix}_rec_n_{int(assembly["rec_n"])}',
        params=local_params,assembly_dict=assembly,simulation=simulation,stf_handler=context['stf_handler'])
    simulation=result.get('simulation_handler')
    if simulation is None:raise RuntimeError('Independent local inversion did not return its simulation.')
    if previous_history:
        simulation.local_inversion_history,_=merge_iteration_history(previous_history,simulation.local_inversion_history)
    stopping=simulation.local_stop_diagnostics or {}
    ready=bool(stopping.get('converged',stopping.get('reason')!='budget_exhausted'))
    complete=ready or not s['require_verified_reference']
    entry=manifest['plan'][position]
    row=functions['_build_local_result_row'](rec_n=int(assembly['rec_n']),assembly_dict=assembly,
        simulation=simulation,result_local=result)
    row['local_continuation_state']=enrich_local_checkpoint(row['local_continuation_state'],simulation)
    row.update(result_fields(simulation));row['local_anchor_stride']=1;row['local_records_per_worker']=1
    row['adaptive_state']={'schema':SCHEMA,'fingerprint':manifest['fingerprint'],'independent':True}
    row['adaptive_independent']={'schema':SCHEMA,'fingerprint':manifest['fingerprint'],'position':position,
        'role':role,'status':'complete' if complete else 'pending','initial_model_kind':seed_kind,
        'initial_model_hash':seed_hash,'initial_model_rec_n':manifest['reference_rec_n'],
        'velocity_lower_bound_m_s':manifest['velocity_lower_bound_cm_us']*1e4,
        'optimizer_policy_fingerprint':policy_digest(optimizer_settings(params)),
        'worker_pid':os.getpid(),'resumed':pending is not None,
        'initial_material_hash':_seed_hash(state)}
    row['adaptive_inversion']={'schema':SCHEMA,'mode':'independent',
        'action':('first_reference' if position==manifest['anchor'] else 'updated')+('' if complete else '_pending'),
        'model_origin_rec_n':int(assembly['rec_n']),'reference_idx_wave':int(pair.uw_idx),
        'reference_position':position,'reference_ready':ready,'reference_stop_reason':stopping.get('reason'),
        'stop_reason':stopping.get('reason','budget_exhausted'),'optimized':True,
        'geometry_grid_changed':not np.array_equal(grid_before,simulation.grid_handler.spatial_axis),
        'decision':deepcopy(entry['decision']),'local_stopping':deepcopy(stopping),
        'initial_model_kind':seed_kind,'initial_model_rec_n':manifest['reference_rec_n']}
    return row

def run_chunk(functions,context,row_positions,continuation_state=None):
    """Hook called by the EXISTING one-process-per-chunk scheduler."""
    if len(row_positions)!=1:raise ValueError('An independent worker must receive exactly one waveform.')
    manifest=context['params']['_adaptive_independent_context'];position=int(row_positions[0])
    path=_job_path(manifest['directory'],position);pending=None
    if path.exists():
        previous=_load(path)
        _check_job(previous,manifest,position,seed_kind=manifest['initial_model_kind'],seed_hash=manifest['seed_hash'])
        if previous['adaptive_independent']['status']=='complete':return row_positions,[(position,previous)],None
        pending=previous
        old=(previous.get('local_stopping') or {}).get('continuation',{}).get('policy')
        new=optimizer_settings(context['params'])
        if old is not None and old!=new:
            policy_changes(old,new)
            backup=path.with_name(path.name+'.before_policy_'+policy_digest(old)[:12]+'.bak')
            if not backup.exists():backup.write_bytes(path.read_bytes())
    print(f'[worker {os.getpid()}] independent rec_n={manifest["plan"][position]["rec_n"]}, '
          f'seed={manifest["initial_model_kind"]}, resume={pending is not None}',flush=True)
    row=_run_one(functions,context,position,state=manifest['seed'],seed_kind=manifest['initial_model_kind'],
                 seed_hash=manifest['seed_hash'],pending=pending)
    _atomic(path,row)
    print(f'[worker {os.getpid()}] independent rec_n={row["rec_n"]}: '
          f'{row["adaptive_independent"]["status"]}, J={row["misfit"]:.6g}',flush=True)
    return row_positions,[(position,row)],None

def _reuse_row(functions,context,manifest,position,reference):
    """Keep historical model-reuse behavior, including mechanical grid shrink."""
    pair=context['paired_rows'][position];entry=manifest['plan'][position]
    observed=np.asarray(context['observed_waveform_data'][int(pair.uw_idx)])
    simulation=deepcopy(context['simulation_seed'])
    assembly=_restore(functions,simulation,reference['local_continuation_state'],observed,
                      manifest['global_seed']['restart_runtime']['simulation']['observed_time'])
    functions['update_assembly_dict_with_mech_data'](assembly_dict=assembly,mech_data=pair)
    assembly['_local_plot_counter']=position-manifest['anchor']
    simulation.adaptive_signal_mask=manifest['signal_mask'].copy();simulation.adaptive_noise_mask=manifest['noise_mask'].copy()
    simulation.independent_velocity_lower_bound_cm_us=manifest['velocity_lower_bound_cm_us']
    p={k:v for k,v in context['params'].items() if k!='_adaptive_independent_context'}
    p.update(_adaptive_reuse=True,n_iterations=0,
             _adaptive_model_origin_rec_n=reference["adaptive_inversion"]["model_origin_rec_n"])
    p.pop('_adaptive_resume_monitor',None);p.pop('_adaptive_resume_limits',None)
    if not sequence.settings(p)['plot_reused_records']:
        p['plot_save_interval']=10**15;p['local_spectral_diagnostics_enabled']=False
    grid_before=simulation.grid_handler.spatial_axis.copy()
    result=functions['local_refine_waveform'](outfile_name=f'{context["outfile_name"]}_rec_n_{int(assembly["rec_n"])}',
        params=p,assembly_dict=assembly,simulation=simulation,stf_handler=context['stf_handler'])
    simulation=result['simulation_handler']
    row=functions['_build_local_result_row'](rec_n=int(assembly['rec_n']),assembly_dict=assembly,simulation=simulation,result_local=result)
    row['local_continuation_state']=enrich_local_checkpoint(row['local_continuation_state'],simulation);row.update(result_fields(simulation))
    row['local_anchor_stride']=1;row['local_records_per_worker']=1
    row['adaptive_state']={'schema':SCHEMA,'fingerprint':manifest['fingerprint'],'independent':True}
    row['adaptive_independent']={'schema':SCHEMA,'fingerprint':manifest['fingerprint'],'position':position,
        'role':'reuse','status':'complete','initial_model_kind':manifest['initial_model_kind'],
        'initial_model_hash':manifest['seed_hash'],'initial_model_rec_n':manifest['reference_rec_n'],
        'velocity_lower_bound_m_s':manifest['velocity_lower_bound_cm_us']*1e4}
    origin=reference['adaptive_inversion']['model_origin_rec_n']
    row['adaptive_inversion']={'schema':SCHEMA,'mode':'independent','action':'reused','model_origin_rec_n':origin,
        'reference_idx_wave':reference['adaptive_inversion']['reference_idx_wave'],
        'reference_position':reference['adaptive_inversion']['reference_position'],
        'reference_ready':True,'reference_stop_reason':reference['adaptive_inversion']['reference_stop_reason'],
        'stop_reason':'model_reused_pending_change_confirmation' if entry['decision']['gate_status']=='awaiting_confirmation' else 'model_reused_noise_compatible',
        'optimized':False,'geometry_grid_changed':not np.array_equal(grid_before,simulation.grid_handler.spatial_axis),
        'decision':deepcopy(entry['decision']),'local_stopping':deepcopy(getattr(simulation,'local_stop_diagnostics',None))}
    return row


def _misfit_progress(row):
    # ADAPTIVE_REUSE_WITHOUT_FORWARD_V1
    if row.get('model_misfit_evaluated', True):
        text=f'J_model={row["misfit"]:.6g}'
    else:
        text='J_model=not_evaluated, forward=skipped'
    decision=(row.get('adaptive_inversion') or {}).get('decision') or {}
    if decision:
        text+=(f', J_data={decision["data_reference_misfit"]:.6g}'
               f', J_noise={decision["noise_pair_misfit"]:.6g}'
               f', threshold={decision["effective_threshold"]:.6g}'
               f', gate={decision.get("gate_status", "unknown")}')
    return text

def _commit(functions,context,manifest,stream,next_position,last):
    """Commit only a chronological prefix; later finished jobs remain in spool."""
    while next_position<len(context['paired_rows']):
        entry=manifest['plan'][next_position]
        if entry['selected']:
            path=_job_path(manifest['directory'],next_position)
            if not path.exists():break
            row=_load(path)
            _check_job(row,manifest,next_position,seed_kind=manifest['initial_model_kind'],seed_hash=manifest['seed_hash'])
            if row['adaptive_independent']['status']!='complete':break
        else:
            if last is None:raise ValueError('A reused waveform has no completed reference.')
            row=_reuse_row(functions,context,manifest,next_position,last)
        pickle.dump(row,stream,protocol=pickle.HIGHEST_PROTOCOL);stream.flush()
        if hasattr(stream,'fileno'):os.fsync(stream.fileno())
        # The durable chronological stream now owns this checkpoint. Retain
        # only out-of-order/pending spools rather than doubling long run size.
        if entry['selected'] and path.exists():path.unlink()
        last=row;next_position+=1
        print(f'[parent] saved independent {next_position}/{len(context["paired_rows"])}: '
              f'rec_n={row["rec_n"]}, {row["adaptive_inversion"]["action"]}, '
              + _misfit_progress(row),flush=True)
    return next_position,last

def _adopt_optimizer(manifest,params,directory,completed_count):
    new=optimizer_settings(params);old=manifest['optimizer_policy']
    # LOCAL_MATERIAL_COORDINATES_V1
    if policy_digest(old)==policy_digest(new):return
    changes=policy_changes(old,new)
    complete=bool(completed_count)
    if not complete:
        for path in Path(directory).glob('wave_*.pkl'):
            if _load(path)['adaptive_independent']['status']=='complete':complete=True;break
    if complete:
        raise ValueError('Optimizer criteria changed after completed independent results. Use a new output folder to compare with new criteria.')
    manifest.setdefault('optimizer_policy_changes',[]).append({'old':old,'new':new,'changes':changes})
    manifest['optimizer_policy']=deepcopy(new)
    print('Independent optimizer change: pending waveforms retain their accepted models; '+json.dumps(changes,sort_keys=True),flush=True)

def _execute(functions,*,global_row,last,anchor,next_position,truncate_to,results_pkl,paired_rows,
             observed_waveform_data,waveform_metadata,outfile_name,params,assembly,simulation,stf_handler,stream=None):
    sequence.validate(params);s=sequence.settings(params)
    fingerprint=sequence._digest(functions,paired_rows,waveform_metadata,stf_handler,params,observed_waveform_data)
    if global_row.get('adaptive_configuration_fingerprint')!=fingerprint:
        raise ValueError('Independent reference/data/settings differ. Use its original settings or a new output folder.')
    directory=_directory(results_pkl);manifest_path=directory/'manifest.pkl'
    if manifest_path.exists():
        manifest=_load(manifest_path)
        if manifest.get('schema')!=SCHEMA or manifest['fingerprint']!=fingerprint or manifest['anchor']!=anchor:
            raise ValueError('Independent manifest belongs to a different selection/reference/settings.')
        manifest['directory']=str(directory.resolve())
        _adopt_optimizer(manifest,params,directory,next_position-anchor)
    else:
        if next_position!=anchor:raise ValueError('Independent manifest is missing; completed models cannot safely infer the original seed.')
        ensure_reference(simulation)
        signal,noise=sequence._masks(simulation,s)
        global_seed=functions['_capture_local_continuation_state'](assembly,simulation)
        global_seed=enrich_local_checkpoint(global_seed,simulation)
        ii=np.concatenate([global_seed['idx_dict'][name] for name in REGIONS])
        lower=s['velocity_lower_bound_m_s']
        lower=float(lower)/1e4 if lower is not None else .25*float(np.min(global_seed['velocity_array'][ii]))
        if lower>float(np.min(global_seed['velocity_array'][ii]))+1e-12:
            raise ValueError('The selected velocity lower bound excludes the homogeneous initial model.')
        manifest={'schema':SCHEMA,'fingerprint':fingerprint,'anchor':anchor,
            'reference_rec_n':_record_id(functions,assembly,paired_rows[anchor]),'directory':str(directory.resolve()),
            'initial_model_kind':s['initial_model'],'global_seed':global_seed,
            'velocity_lower_bound_cm_us':lower,'signal_mask':signal,'noise_mask':noise,
            'optimizer_policy':optimizer_settings(params),'seed':None,'seed_hash':None}
        manifest['plan']=_plan(functions,manifest,paired_rows,observed_waveform_data,params)
    directory.mkdir(parents=True,exist_ok=True)
    _repair_tail(results_pkl,truncate_to)
    _clear_transients(simulation)
    worker_params=dict(params);worker_params['local_anchor_stride']=1;worker_params['local_records_per_worker']=1
    worker_params['_adaptive_independent_context']=manifest
    context={'paired_rows':paired_rows,'observed_waveform_data':observed_waveform_data,
             'outfile_name':outfile_name,'params':worker_params,'assembly_seed':assembly,
             'simulation_seed':simulation,'stf_handler':stf_handler}
    if manifest['seed'] is None:
        if s['initial_model']=='homogeneous':
            manifest['seed']=_fresh_state(manifest['global_seed'])
        else:
            # Preliminary local fit of the SAME first waveform. Freeze it once
            # ready; ALL selected jobs, including that first waveform, then use
            # this fixed snapshot as their common initial model.
            reference_path=directory/'reference_local.pkl'
            pending=_load(reference_path) if reference_path.exists() else None
            h=_seed_hash(manifest['global_seed'])
            if pending is not None:
                _check_job(pending,manifest,anchor,seed_kind='homogeneous',seed_hash=h)
            if pending is not None and pending['adaptive_independent']['status']=='complete':
                row=pending
            else:
                row=_run_one(functions,context,anchor,state=manifest['global_seed'],seed_kind='homogeneous',
                             seed_hash=h,pending=pending,role='reference')
                _atomic(reference_path,row)
            if row['adaptive_independent']['status']!='complete':
                _atomic(manifest_path,manifest)
                print(f'Independent local reference pending: rec_n={row["rec_n"]}; relaunch to continue the SAME reference.',flush=True)
                return
            manifest['seed']=_fresh_state(row['local_continuation_state'])
        manifest['seed_hash']=_seed_hash(manifest['seed'])
    _atomic(manifest_path,manifest)
    _diagnostics(manifest,paired_rows,observed_waveform_data,np.asarray(waveform_metadata['time_ax_waveform']),params,outfile_name)
    own_stream=stream is None
    if own_stream:stream=Path(results_pkl).open('ab')
    try:
        next_position,last=_commit(functions,context,manifest,stream,next_position,last)
        specs=[]
        for position,entry in manifest['plan'].items():
            if position<next_position or not entry['selected']:continue
            path=_job_path(directory,position)
            if path.exists():
                row=_load(path)
                _check_job(row,manifest,position,seed_kind=manifest['initial_model_kind'],seed_hash=manifest['seed_hash'])
                if row['adaptive_independent']['status']=='complete':continue
            specs.append({'lane_id':position,'positions':[position],'continuation_state':None})
        print(f'Independent adaptive inversion: seed={s["initial_model"]}, selected={sum(e["selected"] for e in manifest["plan"].values())}, '
              f'jobs this pass={len(specs)}, requested workers={params.get("local_inversion_workers",1)}, '
              f'fixed lower bound={lower*1e4 if "lower" in locals() else manifest["velocity_lower_bound_cm_us"]*1e4:.6g} m/s.',flush=True)
        if specs:
            for _,row in functions['_run_parallel_local_inversions'](paired_rows=paired_rows,
                    remaining_positions=[s['positions'][0] for s in specs],observed_waveform_data=observed_waveform_data,
                    outfile_name=outfile_name,params=worker_params,assembly_seed=assembly,
                    simulation_seed=simulation,stf_handler=stf_handler,lane_specs=specs):
                next_position,last=_commit(functions,context,manifest,stream,next_position,last)
        next_position,last=_commit(functions,context,manifest,stream,next_position,last)
        pending=[]
        for position,entry in manifest['plan'].items():
            if position<next_position or not entry['selected']:continue
            path=_job_path(directory,position)
            if not path.exists() or _load(path)['adaptive_independent']['status']!='complete':pending.append(entry['rec_n'])
        if pending:
            print(f'Independent pending waveforms: {pending}. Finished later jobs are retained in {directory}; '
                  'relaunch to continue ONLY the pending/unstarted waveforms.',flush=True)
        elif next_position==len(paired_rows):print('Independent adaptive sequence completed.',flush=True)
    finally:
        if own_stream:stream.close()

def run_sequence(functions,*,paired_rows,start_position,observed_waveform_data,waveform_metadata,
                 outfile_name,params,assembly,simulation,stf_handler,stream):
    path=Path(stream.name)
    global_row,last,anchor,next_position,truncated=_read_prefix(path,paired_rows)
    if anchor!=start_position or next_position!=anchor:raise ValueError('A new independent run must start at its global reference waveform.')
    return _execute(functions,global_row=global_row,last=last,anchor=anchor,next_position=next_position,
        truncate_to=truncated,results_pkl=path,paired_rows=paired_rows,observed_waveform_data=observed_waveform_data,
        waveform_metadata=waveform_metadata,outfile_name=outfile_name,params=params,assembly=assembly,
        simulation=simulation,stf_handler=stf_handler,stream=stream)

def resume_file(functions,*,results_pkl,paired_rows,observed_waveform_data,waveform_metadata,
                outfile_name,params,assembly_dict,stf_handler):
    sequence.validate(params)
    fp=sequence._digest(functions,paired_rows,waveform_metadata,stf_handler,params,observed_waveform_data)
    global_row,last,anchor,next_position,truncated=_read_prefix(results_pkl,paired_rows,fingerprint=fp)
    if global_row.get('adaptive_configuration_fingerprint')!=fp:
        raise ValueError('Independent resume data/reference/settings changed; no files modified.')
    if next_position==len(paired_rows):
        manifest=_load(_directory(results_pkl)/'manifest.pkl')
        _adopt_optimizer(manifest,params,_directory(results_pkl),next_position-anchor)
        _repair_tail(results_pkl,truncated)
        print('Independent resume: all selected records already saved.',flush=True);return
    assembly,simulation=functions['_rebuild_simulation_from_saved_row'](saved_row=global_row,global_row=global_row,
        paired_row=paired_rows[anchor],observed_waveform_data=observed_waveform_data,waveform_metadata=waveform_metadata,
        stf_handler=stf_handler,params=params,base_assembly=assembly_dict)
    state=global_row['global_continuation_state']
    assembly=_restore(functions,simulation,state,np.asarray(observed_waveform_data[int(paired_rows[anchor].uw_idx)]),
                      np.asarray(waveform_metadata['time_ax_waveform']),exact=True)
    if not hasattr(simulation,'_kinematic_reference'):ensure_reference(simulation,reference_from_state(state))
    return _execute(functions,global_row=global_row,last=last,anchor=anchor,next_position=next_position,
        truncate_to=truncated,results_pkl=results_pkl,paired_rows=paired_rows,observed_waveform_data=observed_waveform_data,
        waveform_metadata=waveform_metadata,outfile_name=outfile_name,params=params,assembly=assembly,
        simulation=simulation,stf_handler=stf_handler)
