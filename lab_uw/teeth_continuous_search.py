"""Global DDS search on arbitrary parameter grids, using continuous teeth."""
from pathlib import Path
from types import SimpleNamespace
from collections import OrderedDict
import csv,json,time,gc
import multiprocessing as mp
import numpy as np
from .teeth_bank import METALS
from .teeth_bank_search import validate_configuration,candidate_assembly
from .teeth_continuous import ContinuousMatrixProvider,candidate_materials

_WORKER=None
_THREAD_LIMIT=None

def _limit_threads(options):
    global _THREAD_LIMIT
    threads=int(options.get('blas_threads',1))
    if threads<1:raise ValueError('blas_threads must be positive')
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:return
    _THREAD_LIMIT=threadpool_limits(limits=threads)

def _initialize(context):
    global _WORKER
    _limit_threads(context['options'])
    _WORKER=dict(context,provider=ContinuousMatrixProvider(context['options'],context['rows'],context['mode']))

def _model(context,provider,index,plot=False):
    from .forward_modeling import UltrasonicModeler
    row=context['rows'][index];mode=context['mode'];params=context['params']
    options=dict(context['options'],backend=provider.backend,_bank=provider,material_index=index)
    assembly=candidate_assembly(context['assembly'],row,mode,options)
    simulation=UltrasonicModeler()
    simulation.forward_simulation(geometry_type='dds',observed_time=context['time'],observed_waveform=context['observed'],
        stf_handler=context['stf'],frequency_cutoff=params['frequency_cutoff'],minimum_velocity=row['cp' if mode==0 else 'cs']/10,
        maximum_velocity=max(METALS['steel']['cp' if mode==0 else 'cs'],row['cp' if mode==0 else 'cs'])/10,
        assembly_dict=assembly,misfit_interval=context['interval'],embedding_mode='effective_surface',
        boundary_condition='free',source_injection='body_force',receiver_quantity='velocity',store_wavefield=False,
        points_per_wavelength=params.get('points_per_wavelength',8),normalize_waveform=True,
        normalization_interval=context['interval'],source_receiver_config=params.get('source_receiver_config',{}),
        enable_plotting=plot,plot_output_path=context['outdir']/'all_forward',
        spectral_gamma=context['gamma'],objective_taper_fraction=context['taper'],teeth_continuous_options=options)
    return simulation

def _result(context,simulation,index,stage):
    spectral=simulation.write_direct_arrival_spectral_diagnostics(output_directory=None,
        frequency_min_mhz=context['params'].get('frequency_min',0.),frequency_max_mhz=context['params']['frequency_cutoff'])
    return dict(context['rows'][index],misfit=float(simulation.misfit),
        spectral_distance=float(spectral['recorded_vs_simulated_distance']),trace_scale=float(simulation.trace_scale),
        mode=context['assembly']['wave_type'],evaluation_stage=stage)

def _group(indices):
    context=_WORKER;result=[]
    for index in indices:
        model=_model(context,context['provider'],index,context['options'].get('plot_all_candidates',False))
        result.append(_result(context,model,index,context['options'].get('backend','reduced')))
        del model
    gc.collect();return result

def _edges(values):
    values=np.asarray(values,float)
    if len(values)==1:
        half=max(abs(values[0])*.025,.001);return [values[0]-half,values[0]+half]
    mid=(values[1:]+values[:-1])/2
    return np.r_[values[0]-(mid[0]-values[0]),mid,values[-1]+values[-1]-mid[-1]]

def _maps(rows, outdir, mode):
    # PROFILED_CONSTITUTIVE_MAPS_V1
    from .material_search_maps import plot_teeth_maps
    return plot_teeth_maps(rows, outdir, mode)

def _return_result(context,results,best,selected,meta,spectral):
    mode=context['mode'];assembly=context['assembly']
    coarse=dict(material_index_list=np.array([r['index'] for r in results]),poisson_list=np.array([r['nu'] for r in results]),
       cp_m_s_list=np.array([r['cp']*1000 for r in results]),cs_m_s_list=np.array([r['cs']*1000 for r in results]),
       gouge_velocity_list=np.array([r['cp' if mode==0 else 'cs']/10 for r in results]),
       gouge_density_list_g_cm3=np.array([r['rho'] for r in results]),gouge_damping_list=np.array([r['damping_cm2_us'] for r in results]),
       waveform_shape_misfit_list=np.array([r['misfit'] for r in results]),spectral_shape_distance_list=np.array([r['spectral_distance'] for r in results]),
       trace_scale_list=np.array([r['trace_scale'] for r in results]))
    outdir=context['outdir']
    (outdir/'best_continuous_model.json').write_text(json.dumps(meta,indent=2))
    np.savez_compressed(outdir/'global_results.npz',**coarse,time_us=context['time'],recorded=context['observed'],
       best_raw=best.raw_synthetic_waveform,best_scaled=best.synthetic_waveform,misfit_mask=context['interval'],
       best_residual=context['observed']-best.synthetic_waveform,metadata_json=json.dumps(meta))
    best.plotter.plot_simulation_waveform(t=context['time'],sp_simulated=best.synthetic_waveform,sp_recorded=context['observed'],
       misfit_interval=context['interval'],outfile_path=outdir/'best_continuous_fit')
    common=selected['cp' if mode==0 else 'cs']/10;loss=selected['damping_cm2_us']
    material=dict(coarse=coarse,simulation_handler=best,elastic_best_gouge_velocity=common,elastic_best_gouge_damping=loss,
       elastic_best_gouge_density_g_cm3=selected['rho'],best_waveform_shape_misfit=best.misfit,
       best_spectral_shape_distance=spectral['recorded_vs_simulated_distance'],best_material=selected,
       forward_backend=meta['backend'],teeth_continuous_metadata=meta)
    return dict(simulation_handler=best,gouge_velocity_list=coarse['gouge_velocity_list'],best_gouge_velocity=common,
       best_homogeneous_gouge_velocity=common,best_gouge_velocity_1=common,best_gouge_velocity_2=common,
       reference_gouge_thickness_1_cm=assembly['gouge_thickness_1'],reference_gouge_thickness_2_cm=assembly['gouge_thickness_2'],
       best_gouge_thickness_1_cm=assembly['gouge_thickness_1'],best_gouge_thickness_2_cm=assembly['gouge_thickness_2'],
       gouge_damping_list=coarse['gouge_damping_list'],best_gouge_damping=loss,L2norm_waveform_list=coarse['waveform_shape_misfit_list'],
       complete_misfit_list=coarse['waveform_shape_misfit_list'],homogeneous_spectral_distance_list=coarse['spectral_shape_distance_list'],
       homogeneous_spectral_diagnostics=spectral,global_spectral_gamma=context['gamma'],homogeneous_best_misfit=best.misfit,
       velocity_contrast_result=None,gouge_thickness_result=None,joint_kinematic_result=None,density_result=None,
       impedance_material_result=None,elastic_material_result=material,absorption_comparison=None,
       selected_material_model='teeth_continuous_bulk_KV' if loss else 'teeth_continuous_elastic',
       post_impedance_thickness_result=None,spectral_diagnostics=spectral,spectral_shape_distance=spectral['recorded_vs_simulated_distance'],
       best_gouge_density_g_cm3=selected['rho'],teeth_continuous_result=material,
       teeth_continuous_metadata=meta,teeth_bank_metadata=meta)

def search_continuous_global(observed_waveform,waveform_metadata,outfile_name,stf_handler,params,assembly_dict,*,workflow=None):
    validate_configuration(params)
    # The caller may be a top-level script, outside the lab_uw package.
    # Its helpers are used only here in the parent process, never pickled
    # into the multiprocessing worker context.
    if workflow is None:
        from . import local_optimization_FWI_velocity_1D as workflow
    if assembly_dict['wave_type'] not in ('_p','_s'):raise ValueError('Only P and SV are supported')
    mode=0 if assembly_dict['wave_type']=='_p' else 1
    options=dict(params.get('teeth_continuous',{}));options['frequency_cutoff_mhz']=params['frequency_cutoff']
    rows=candidate_materials(options,mode)
    top=options.get('full_refinement_top_k',0)
    if not isinstance(top,(int,np.integer)) or top<0:raise ValueError('full_refinement_top_k must be a nonnegative integer')
    time_axis=np.asarray(waveform_metadata['time_ax_waveform'],float);observed=np.asarray(observed_waveform,float)
    snr=workflow.evaluate_snr(time_axis,observed,assembly_dict['steel_only_time_wave'])
    if snr<params['minimum_SNR']:
        print(f'Continuous teeth: SNR {snr:.2f} below minimum; skipping',flush=True);return {'simulation_handler':None}
    if 'misfit_interval_us' in options:
        lo,hi=options['misfit_interval_us'];interval=(time_axis>=lo)&(time_axis<=hi)
    else:
        bounds=[workflow.predict_event_windows(candidate_assembly(assembly_dict,r,mode,options),
              [r['cp' if mode==0 else 'cs']/10],wave_type=assembly_dict['wave_type'])['direct'] for r in rows]
        st=np.asarray(stf_handler.metadata['time_ax_waveform'])
        interval=workflow.direct_misfit_interval(time_axis,arrival_min_us=min(b[0] for b in bounds),
            arrival_max_us=max(b[1] for b in bounds),stf_duration_us=st[-1]-st[0],padding_us=params['direct_window_padding_us'])
    if np.count_nonzero(interval)<4:raise ValueError('Direct window contains fewer than four samples')
    outdir=Path(params['outdir_path_image'])/outfile_name/'continuous_teeth_search';outdir.mkdir(parents=True,exist_ok=True)
    context=dict(options=options,rows=rows,mode=mode,params=params,time=time_axis,observed=observed,interval=interval,
       stf=SimpleNamespace(metadata={'time_ax_waveform':np.asarray(stf_handler.metadata['time_ax_waveform'])},
                           waveform_data=np.asarray(stf_handler.waveform_data).reshape(-1)),
       assembly=assembly_dict,outdir=outdir,
       gamma=float(params.get('global_spectral_gamma',params.get('local_spectral_gamma',1.))),
       taper=float(params.get('global_objective_taper_fraction',params.get('local_objective_taper_fraction',0.))))
    grouped=OrderedDict()
    for r in rows:grouped.setdefault((r['rho'],r['cs'],r['nu']),[]).append(r['index'])
    worker_params=dict(params,global_search_workers=options.get('workers',params.get('global_search_workers',1)))
    workers=workflow.resolve_worker_count(worker_params,'global_search_workers',len(grouped))
    start=time.monotonic();results=[];pool=None
    try:
        if workers==1:_initialize(context);iterator=map(_group,grouped.values())
        else:
            pool=mp.get_context(options.get('multiprocessing_start_method','spawn')).Pool(workers,initializer=_initialize,initargs=(context,))
            iterator=pool.imap_unordered(_group,list(grouped.values()),chunksize=1)
        with (outdir/'all_candidates.csv').open('w',newline='') as f:
            writer=None
            for batch in iterator:
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(batch[0]));writer.writeheader()
                for r in batch:
                    writer.writerow(r);results.append(r)
                    print(f"Continuous teeth {len(results)}/{len(rows)}: rho={r['rho']:g}, "
                         f"velocity={r['cp' if mode==0 else 'cs']*1000:.1f} m/s, "
                         f"Poisson={r['nu']:g}, damping={r['damping_cm2_us']:g}, J={r['misfit']:.6g}",flush=True)
                f.flush()
        if pool is not None:pool.close();pool.join();pool=None
    finally:
        if pool is not None:pool.terminate();pool.join()
    results.sort(key=lambda r:r['index']);_maps(results,outdir,mode);_limit_threads(options)
    provider=_WORKER['provider'] if workers==1 else ContinuousMatrixProvider(options,rows,mode)
    rank=sorted(results,key=lambda r:(r['misfit'],r['index']));refinements=[]
    if top and options.get('backend','reduced')=='reduced':
        full=ContinuousMatrixProvider(dict(options,backend='full'),rows,mode)
        for r in rank[:top]:
            trial=_model(context,full,r['index']);refinements.append(_result(context,trial,r['index'],'full_refinement'))
            del trial;gc.collect()
        winner=min(refinements,key=lambda r:(r['misfit'],r['index']));provider=full
    else:winner=rank[0]
    best=_model(context,provider,winner['index']);selected=rows[winner['index']]
    assembly_dict.update(best.assembly_dict)
    spectral=best.write_direct_arrival_spectral_diagnostics(output_directory=outdir if params.get('spectral_diagnostics_enabled',True) else None,
       frequency_min_mhz=params.get('frequency_min',0.),frequency_max_mhz=params['frequency_cutoff'])
    meta=dict(best.teeth_continuous_metadata,material_count=len(results),worker_count=workers,
       selection_metric='existing_relative_waveform_objective',full_refinement_results=refinements,
       selected_from='full reevaluation of top reduced candidates' if refinements else 'continuous candidate grid',
       elapsed_seconds=time.monotonic()-start,window_us=[float(time_axis[interval][0]),float(time_axis[interval][-1])])
    best.teeth_continuous_metadata=meta;best.teeth_bank_metadata=meta
    return _return_result(context,results,best,selected,meta,spectral)
