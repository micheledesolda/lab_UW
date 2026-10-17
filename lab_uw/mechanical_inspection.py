"""Call visual_ispection_standard on the exact selected inversion input."""
from __future__ import annotations
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np


def _window(functions, paired, metadata, stf, params, assembly, options):
    explicit=options.get('window_us')
    if explicit is None:
        explicit=params.get('adaptive_inversion',{}).get('signal_window_us')
    if explicit is None:
        explicit=params.get('wave_difference_diagnostic',{}).get('window_us')
    time=np.asarray(metadata['time_ax_waveform'],dtype=float)
    if explicit is not None:
        lo,hi=map(float,explicit)
        if not np.isfinite([lo,hi]).all() or lo>=hi:raise ValueError('Invalid mechanical inspection window_us.')
        tolerance=float(np.median(np.diff(time)))*(1+1e-6)
        if lo<time[0]-tolerance or hi>time[-1]+tolerance:
            raise ValueError('Mechanical inspection window lies outside the samples available to the inversion.')
        # Match the inclusive sample mask used by the diagnostic and solver.
        mask=(time>=lo)&(time<=hi)
    else:
        first=deepcopy(assembly)
        functions['update_assembly_dict_with_mech_data'](assembly_dict=first,
            mech_data=next(paired.itertuples(index=False)))
        windows=functions['predict_event_windows'](first,params['velocity_initial_list'],wave_type=first['wave_type'])
        source_time=np.asarray(stf.metadata['time_ax_waveform'])
        mask=functions['direct_misfit_interval'](time,arrival_min_us=windows['direct'][0],
            arrival_max_us=windows['direct'][1],stf_duration_us=float(source_time[-1]-source_time[0]),
            padding_us=params['direct_window_padding_us'])
        if np.asarray(mask).dtype!=bool:
            selected=np.zeros(time.size,dtype=bool);selected[mask]=True;mask=selected
    selected=time[mask]
    if selected.size<4:raise ValueError('Mechanical inspection direct window has fewer than four samples.')
    return float(selected[0]),float(selected[-1])


def _load_script(functions,options):
    main=Path(functions.get('__file__','local_optimization_FWI_velocity_1D.py')).resolve()
    path=Path(options.get('script_path') or main.parent/'visual_ispection_standard.py').expanduser()
    if not path.is_absolute():path=main.parent/path
    if not path.is_file():raise FileNotFoundError(f'Visual inspection script not found: {path}')
    # Own module instance: the standalone user configuration is left intact.
    name='_mechanical_visual_'+hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not getattr(module,'MECHANICAL_INSPECTION_API',False):
        raise ValueError('visual_ispection_standard.py lacks the mechanical inspection patch.')
    return module,path.resolve()


def _overview(module,product,plotter,paired,output):
    time=product['metadata']['time_ax_acquisition']
    trace_time=product['metadata']['time_ax_waveform']
    values,xedges,yedges=plotter._bounded_map(product['filtered'],time,trace_time,signed=True)
    fig,(stress,map_ax)=module.plt.subplots(2,1,sharex=True,figsize=(13,9),
        gridspec_kw={'height_ratios':[1,2]},layout='constrained')
    stress_column=next((c for c in ('shear_stress_MPa','shear_stress','shear') if c in paired),None)
    if stress_column is None:
        stress.text(.5,.5,'Shear stress column unavailable',transform=stress.transAxes,ha='center')
    else:
        stress.plot(time,np.asarray(paired[stress_column],dtype=float),color='black',linewidth=1.5)
    stress.set_ylabel('Shear stress [MPa]',fontsize=18);stress.tick_params(labelsize=14);stress.grid(alpha=.2)
    stress.set_title('Intervallo selezionato: stress e dati in ingresso all\'inversione',fontsize=20)
    peak=float(np.max(np.abs(values))) or 1.
    image=map_ax.pcolormesh(xedges,yedges,values.T,cmap='RdBu_r',vmin=-peak,vmax=peak,
                            shading='flat',rasterized=True)
    map_ax.set_ylabel(r'Time after trigger [$\mu$s]',fontsize=18)
    map_ax.set_xlabel('Synchronized experiment time [s]',fontsize=18);map_ax.tick_params(labelsize=14)
    map_ax.ticklabel_format(axis='x',style='plain',useOffset=False)
    map_ax.set_ylim(trace_time[0],trace_time[-1])
    if len(time)>1:map_ax.set_xlim(time[0],time[-1])
    color=fig.colorbar(image,ax=map_ax,pad=.02)
    color.set_label(f"Amplitude [{product['metadata'].get('amplitude_unit','a.u.')}]",fontsize=16)
    color.ax.tick_params(labelsize=14)
    plotter.output_path_choice(fig=fig,outfile_path=str(output))


def inspect_selected(functions,*,paired,observed_waveform_data,waveform_metadata,
                     stf_handler,params,assembly_dict,infile_path):
    """Return True only when an explicitly requested plots-only run is complete.

    No file reading, SNR gate, signal filtering, time alignment or amplitude
    normalization occurs here. The main process has already done preprocessing.
    Gather selected rows in bounded blocks into a temporary disk array.
    """
    options=params.get('mechanical_selection_inspection',{})
    if not options.get('enabled',False):return False
    if paired.empty:return bool(options.get('only',False))
    data=np.asarray(observed_waveform_data)
    indices=np.asarray(paired['uw_idx'])
    if (data.ndim!=2 or indices.ndim!=1 or not np.issubdtype(indices.dtype,np.integer)
            or np.any(indices<0) or np.any(indices>=len(data)) or np.any(np.diff(indices)<=0)):
        raise ValueError('Mechanical inspection needs increasing, unique, valid uw_idx rows.')
    times=np.asarray(paired['uw_time_s'],dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Mechanical inspection synchronized UW times must strictly increase.')
    window=_window(functions,paired,waveform_metadata,stf_handler,params,assembly_dict,options)
    module,path=_load_script(functions,options)
    budget=float(options.get('processing_memory_mb',module.processing_memory_mb))
    if not np.isfinite(budget) or budget<=0:raise ValueError('processing_memory_mb must be positive.')
    # Explicit configuration for this isolated call; standalone defaults stay intact.
    module.processing_memory_mb=budget
    module.processing_block_rows=int(options.get('processing_block_rows',module.processing_block_rows))
    if module.processing_block_rows<1:raise ValueError('processing_block_rows must be positive.')
    module.number_of_example_waveforms=int(options.get('number_of_example_waveforms',module.number_of_example_waveforms))
    module.example_indices=None
    module.number_of_overlay_waveforms=int(options.get('number_of_overlay_waveforms',module.number_of_overlay_waveforms))
    module.show_filter_response=False;module.show_phase_spectrum=False
    module.spectrum_max_mhz=options.get('spectrum_max_mhz',params.get('frequency_cutoff'))
    module.cache_directory=None
    module.plot_settings=dict(module.plot_settings)
    module.plot_settings.update(processing_memory_mb=budget,
        additional_formats=('.pdf',) if options.get('save_pdf',False) else (),
        dpi=int(options.get('dpi',200)))
    for key in ('map_max_columns','map_max_rows'):
        module.plot_settings[key]=int(options.get(key,module.plot_settings[key]))
    subdirectory=options.get('output_subdirectory','mechanical_selection_inspection')
    if not subdirectory or Path(subdirectory).is_absolute() or len(Path(subdirectory).parts)!=1 or subdirectory in ('.','..'):
        raise ValueError('Inspection output_subdirectory must be one relative folder name.')
    base=Path(params['outdir_path_image'])/subdirectory/Path(infile_path).name.split('.')[0]
    base.mkdir(parents=True,exist_ok=True)
    output_dirs={key:base/key for key in module.image_types}
    for directory in output_dirs.values():directory.mkdir(exist_ok=True)
    channel=str(waveform_metadata.get('channel_name',waveform_metadata.get('channel','not_recorded')))
    if channel=='not_recorded':
        print('Mechanical inspection: input metadata does not identify the channel; using the SAME input matrix as inversion.',flush=True)
    columns=list(paired.columns)
    raw_times=np.asarray(waveform_metadata['time_ax_acquisition'],dtype=float)[indices]
    with (base/'selected_records.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['selection_position','input_acquisition_time_s',*columns])
        for position,row in enumerate(paired.itertuples(index=False,name=None)):
            writer.writerow([position,float(raw_times[position]),*row])
    record_column='records_na' if 'records_na' in paired else 'Index'
    summary={'schema':1,'input_file':str(Path(infile_path).resolve()),'visual_script':str(path),
        'channel':channel,'number_of_waveforms':len(indices),'direct_window_us':list(window),
        'first':{'uw_idx':int(indices[0]),'uw_time_s':float(times[0]),'mechanical_time_s':float(paired['time_s'].iloc[0])},
        'last':{'uw_idx':int(indices[-1]),'uw_time_s':float(times[-1]),'mechanical_time_s':float(paired['time_s'].iloc[-1])},
        'maximum_selected_index_gap':int(np.max(np.diff(indices))) if len(indices)>1 else 0,
        'maximum_selected_time_gap_s':float(np.max(np.diff(times))) if len(times)>1 else 0.,
        'frequency_min_mhz':params.get('frequency_min'),'frequency_cutoff_mhz':params.get('frequency_cutoff'),
        'minimum_SNR_configured':params.get('minimum_SNR'),'stage':'after_mechanical_selection_before_SNR',
        'input_already_preprocessed':True,'additional_signal_filter':False,'SNR_selection_applied':False,
        'amplitude_normalization':False,'time_alignment':False,
        'spectrum_remove_mean':bool(module.spectral_remove_mean),
        'spectrum_taper_fraction':float(module.spectral_taper_fraction),
        'processing_memory_mb':budget,'record_column':record_column}
    for which,index in [('first',0),('last',-1)]:
        if record_column in paired:
            value=paired[record_column].iloc[index]
            summary[which]['record']=float(value) if np.isfinite(value) else None
    (base/'selected_records.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'Mechanical inspection: {len(indices)} exact selected waveforms, channel={channel}, '
          f'UW indices [{indices[0]}, {indices[-1]}], synchronized times [{times[0]:.6f}, {times[-1]:.6f}] s; '
          f'window={window} us. No additional signal filtering. Output: {base}',flush=True)
    metadata=dict(waveform_metadata)
    metadata.update(number_of_waveforms=len(indices),time_ax_acquisition=times.copy(),time_ax_acquisition_unit='s')
    rows=max(1,min(module.processing_block_rows,int(budget*1024**2//max(1,data.shape[1]*data.dtype.itemsize*2))))
    if rows<1:raise ValueError('processing_block_rows must be positive.')
    with TemporaryDirectory(prefix='selected_input_',dir=base) as temporary:
        selected=np.lib.format.open_memmap(Path(temporary)/'input.npy',mode='w+',dtype=data.dtype,
                                           shape=(len(indices),data.shape[1]))
        handler=None
        try:
            pending_bytes=0
            for first in range(0,len(indices),rows):
                last=min(first+rows,len(indices));selected[first:last]=data[indices[first:last]]
                pending_bytes+=(last-first)*data.shape[1]*data.dtype.itemsize
                if pending_bytes>=budget*1024**2:
                    module._release_disk_arrays([selected])
                    module.Plotter._release_mapped_pages(data);pending_bytes=0
            module._release_disk_arrays([selected])
            handler=module.UltrasonicDataHandler(selected,metadata)
            # Keep GUI figures and rcParams owned by the caller untouched.
            with module.plt.rc_context():
                module.process_channel(handler,Path(infile_path),channel,output_dirs=output_dirs,
                    already_preprocessed=True,window_us=window,
                    on_products=lambda product,plotter:_overview(module,product,plotter,paired,base/'stress_and_direct_arrivals'))
        finally:
            handler=None
            selected._mmap.close()
    return bool(options.get('only',False))
