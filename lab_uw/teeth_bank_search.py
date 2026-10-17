"""Global elastic material search using the original P2 bank and waveform objective."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import time as clock
import numpy as np
from scipy.fft import next_fast_len
from .teeth_bank import P2MatrixBank, materials, METALS, ETA

def validate_configuration(params):
    continuous=bool(params.get('teeth_continuous_enabled',False))
    if continuous and params.get('teeth_bank_enabled',False):
        raise ValueError('Select only one of teeth_continuous_enabled and teeth_bank_enabled')
    if params.get('run_local_inversion', True):
        raise ValueError('P2 bank search requires run_local_inversion=False: multimodal adjoint is not implemented')
    if params.get('embedding_mode') != 'effective_surface' or params.get('absorbing', False):
        raise ValueError('P2 bank search requires effective_surface with absorbing=False')
    if params.get('impedance_material_search_enabled', False):
        raise ValueError('Disable the KV ramp search when selecting the bank-only global search')
    if not continuous and not params.get('teeth_bank', {}).get('directory'):
        raise ValueError('Set params[teeth_bank][directory] to the extracted original matrices directory')
    for key, expected in [('boundary_condition', 'free'), ('source_injection', 'body_force'), ('receiver_quantity', 'velocity')]:
        if params.get(key, expected) != expected:
            raise ValueError(f'P2 bank search requires {key}={expected}')

def candidate_assembly(assembly, row, mode, options):
    a = deepcopy(assembly); a['density_model_enabled'] = True
    for name, kind in [('side1', 'steel'), ('central', 'titanium'), ('side2', 'steel')]:
        m = METALS[kind]; block = a[name+'_params']
        if not options.get('use_bank_metal_properties', False):
            for key, v in [('velocity_p', m['cp']/10), ('velocity_s', m['cs']/10)]:
                if key in block and not np.isclose(float(block[key]), v, rtol=1e-9, atol=1e-12):
                    raise ValueError(f'{name}.{key} differs from the fixed bank metal; explicit use_bank_metal_properties is required')
            if name+'_density' in a and not np.isclose(float(a[name+'_density']), m['rho']):
                raise ValueError(f'{name} density differs from the bank metal')
        block['velocity_p'] = m['cp']/10; block['velocity_s'] = m['cs']/10
        a[name+'_density'] = m['rho']
        if not np.isclose(float(block['h_grooves'])*10, .5, rtol=0, atol=1e-10):
            raise ValueError('This bank contains 0.5 mm teeth only; acquisition groove heights must match')
    key = 'cp' if mode == 0 else 'cs'
    for j in (1, 2):
        a[f'gouge_velocity_{j}'] = row[key]/10
        a[f'gouge_damping_{j}'] = float(row.get('damping_cm2_us',0.))
        a[f'gouge_density_{j}'] = row['rho']
    a['gouge_cp_cm_us'] = row['cp']/10; a['gouge_cs_cm_us'] = row['cs']/10
    a['gouge_poisson'] = row['nu']; a['teeth_bank_material_index'] = row['index']
    return a

def mechanical_geometry(assembly):
    dims = np.asarray(assembly['sample_dimensions'], float)*10
    if dims.shape != (5,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        raise ValueError('DDS sample_dimensions must be five positive mechanical lengths in cm')
    for j, i in [(1, 1), (2, 3)]:
        if not np.isclose(dims[i], float(assembly[f'gouge_thickness_{j}'])*10, rtol=1e-10, atol=1e-10):
            raise ValueError('sample_dimensions and gouge thicknesses disagree')
    lengths = dims-np.array([.5, 0., 1., 0., .5])
    if np.any(lengths < 0): raise ValueError('Block thickness is smaller than its tooth zones')
    return lengths

def validate_operators(config, total_length_cm):
    c = {} if config is None else dict(config)
    requirements = {'source_position_cm': 0., 'receiver_position_cm': total_length_cm,
                    'source_extension_cm': 0., 'receiver_extension_cm': 0.}
    for k, v in requirements.items():
        if not np.isclose(float(c.get(k, v)), v, rtol=0, atol=1e-12):
            raise ValueError(f'P2 bank surface operator requires {k}={v}')
    if c.get('source_normalization', c.get('spatial_normalization', 'integral')) != 'integral':
        raise ValueError('Bank source convention requires integral normalization')
    if c.get('receiver_normalization', 'sum') != 'sum':
        raise ValueError('Bank receiver convention requires sum normalization')
    if c.get('spatial_boundary_mode', 'truncate') != 'truncate':
        raise ValueError('Bank surface operators require spatial_boundary_mode=truncate')

def temporal_trace(bank, index, observed_time, stf_time, stf_waveform, lengths, mode, options):
    t = np.asarray(observed_time, float).reshape(-1); st = np.asarray(stf_time, float).reshape(-1)
    sw = np.asarray(stf_waveform, float).reshape(-1)
    if (t.size < 3 or st.size < 2 or sw.size != st.size
            or not all(np.all(np.isfinite(a)) for a in (t, st, sw))):
        raise ValueError('Invalid observed/source time arrays')
    if np.any(np.diff(t) <= 0) or np.any(np.diff(st) <= 0) or not np.allclose(np.diff(t), t[1]-t[0]):
        raise ValueError('Observed time must be uniform; source time must increase')
    if not np.isclose(t[0], 0., atol=1e-12):
        raise ValueError('Bank temporal reconstruction requires observed_time starting at zero')
    cutoff = float(options['frequency_cutoff_mhz']); dt_obs = t[1]-t[0]
    if not 0 < cutoff <= 8: raise ValueError('The original bank is restricted to 0 < cutoff <= 8 MHz')
    if cutoff >= .5/dt_obs: raise ValueError('Observed Nyquist must exceed the cutoff')
    oversampling = int(options.get('time_oversampling', 1))
    if oversampling < 1 or oversampling != options.get('time_oversampling', 1):
        raise ValueError('time_oversampling must be a positive integer')
    dt = dt_obs/oversampling
    nt = (len(t)-1)*oversampling+1; tt = np.arange(nt)*dt
    source = np.interp(tt, st-st[0], sw, left=0., right=0.)
    padding = int(options.get('fft_padding_factor', 8))
    tolerance = float(options.get('periodic_tail_factor', 1e-4))
    if padding < 2 or not 0 < tolerance < 1: raise ValueError('Invalid FFT padding or periodic-tail factor')
    nfft = next_fast_len(max(padding*nt, int(np.ceil(-np.log(tolerance)/(ETA*dt)))))
    f = np.fft.rfftfreq(nfft, dt); active = np.flatnonzero(f <= cutoff+1e-12)
    response = np.zeros(len(f), complex)
    offsets = options.get('lateral_offsets_mm', [0., 0., 0., 0.])
    # For the existing surface acceleration source, integral-normalized in cm,
    # F=rho*s/2 and nodal velocity reception gives H*s/(2*c_steel_cm_us).
    normalization = 1/(2*(METALS['steel']['cp' if mode == 0 else 'cs']/10))
    for j in active:
        response[j] = bank.response(index, float(f[j]), lengths_mm=lengths, mode=mode,
                                    offsets_mm=offsets).conjugate()*normalization
    response[0] = response[0].real
    if nfft % 2 == 0: response[-1] = response[-1].real
    weight = np.exp(-ETA*tt)
    raw = np.fft.irfft(np.fft.rfft(source*weight, nfft)*response, nfft)[:nt]/weight
    metadata = dict(backend='p2_bank_direct', material_index=int(index), intrinsic_loss=0.,
                    lengths_mm=np.asarray(lengths).tolist(), tooth_heights_mm=[.5]*4, pitch_mm=1.,
                    lateral_offsets_mm=list(offsets), laplace_eta_per_us=ETA,
                    frequency_interpolation=bank.interpolation, bank_step_mhz=.025,
                    parity_reduction_above_mhz=.025,
                    fft_step_mhz=1/(nfft*dt), cutoff_mhz=cutoff, nfft=int(nfft), dt_us=float(dt),
                    fft_period_us=float(nfft*dt), periodic_contour_weight=float(np.exp(-ETA*nfft*dt)),
                    time_oversampling=oversampling, physical_source_amplitude_calibrated=False,
                    receiver_quantity='velocity', source_convention='surface integral-normalized body force',
                    numerical_limits='Frequency interpolation and cell accuracy require checks on the acquisition; periodic_contour_weight is not an error bound',
                    matrix_sha256=dict(bank.fingerprints))
    if hasattr(bank,'metadata_for_candidate'):
        metadata.update(bank.metadata_for_candidate(index))
        metadata.pop('matrix_sha256',None)
    return raw[::oversampling], source, tt, metadata

def forward_with_bank(model, args):
    from .simulation_setup import Grid1D, VelocityModel1D_DDS, Source1D, Receiver1D
    from .forward_modeling import fit_trace_scale, relative_waveform_objective
    args = dict(args)
    args['source_amplitude'] = float((args.get('source_receiver_config') or {}).get('source_amplitude', 1.))
    if not np.isfinite(args['source_amplitude']): raise ValueError('Invalid source amplitude')
    options = dict(args['teeth_bank_options']); bank = options.pop('_bank', None)
    if bank is None: bank = P2MatrixBank(options['directory'], frequency_interpolation=options.get('frequency_interpolation', 'linear'))
    mode = 0 if args['assembly_dict']['wave_type'] == '_p' else 1
    if args['assembly_dict']['wave_type'] not in ('_p', '_s'): raise ValueError('Only P and SV are supported')
    index = int(options['material_index']); row = bank.rows[index]
    assembly = candidate_assembly(args['assembly_dict'], row, mode, options)
    lengths = mechanical_geometry(assembly); total = float(np.sum(assembly['sample_dimensions']))
    validate_operators(args.get('source_receiver_config'), total)
    if (args['geometry_type'].lower() != 'dds' or args['embedding_mode'] != 'effective_surface'
            or args['absorbing'] or args['store_wavefield'] or args['make_movie']):
        raise ValueError('Bank forward requires DDS effective_surface, receiver-only mode, no absorbing boundary or movie')
    for k, expected in [('boundary_condition', 'free'), ('source_injection', 'body_force'), ('receiver_quantity', 'velocity')]:
        if args.get(k) not in (None, expected): raise ValueError(f'Bank forward requires {k}={expected}')
    if args.get('montecarlo'): raise ValueError('Monte Carlo source/receiver changes are not supported by this bank operator')
    options['frequency_cutoff_mhz'] = args['frequency_cutoff']
    stf = args['stf_handler']; t = np.asarray(args['observed_time'], float)
    raw, source, tt, meta = temporal_trace(bank, index, t, stf.metadata['time_ax_waveform'],
                                         np.asarray(stf.waveform_data).reshape(-1)*args['source_amplitude'], lengths, mode, options)
    scale_interval = args['misfit_interval'] if args['normalization_interval'] is None else args['normalization_interval']
    scale = fit_trace_scale(args['observed_waveform'], raw, scale_interval) if args['normalize_waveform'] else 1.
    predicted = raw*scale
    objective = relative_waveform_objective(args['observed_waveform'], predicted, args['misfit_interval'],
                 spectral_gamma=args['spectral_gamma'], taper_fraction=args['objective_taper_fraction'], eps=1e-12)
    # Compatibility objects describe the nominal z partition only. Their ramp
    # arrays are NOT the operator used to produce the multimodal bank trace.
    velocity = row['cp' if mode == 0 else 'cs']/10
    grid = Grid1D(velocity, args['frequency_cutoff'], total, args['points_per_wavelength'] or 8)
    vblocks = {k: METALS[m]['cp' if mode == 0 else 'cs']/10 for k, m in [('side1','steel'),('central','titanium'),('side2','steel')]}
    vm = VelocityModel1D_DDS(x=grid.spatial_axis, sample_dimensions=assembly['sample_dimensions'],
        x_transmitter=0., x_receiver=total, h_groove_side=.05, h_groove_central=.05,
        pzt_layer_width=0., pla_layer_width=0., block_velocity=vblocks, gouge_velocity=(velocity, velocity),
        gouge_damping=(float(row.get('damping_cm2_us',0.)),)*2, pzt_velocity=vblocks['side1'], pla_velocity=vblocks['side1'],
        block_density={'side1':7.85,'central':4.5,'side2':7.85}, gouge_density=(row['rho'],row['rho']),
        include_transducer_layers=False)
    vm.diagnostic_partition_only = True
    sh = Source1D(stf.metadata['time_ax_waveform'], np.asarray(stf.waveform_data).reshape(-1)*args['source_amplitude'],
                  position=0., pzt_layer_width=0., radius=4, extension=0., normalization='integral', boundary_mode='truncate')
    sh.time_function = source; sh.create_spatial_function(grid.spatial_axis, grid.dx)
    rh = Receiver1D(position=total, pzt_layer_width=0., radius=4, extension=0., normalization='sum', boundary_mode='truncate')
    rh.create_spatial_function(grid.spatial_axis, grid.dx)
    timing = SimpleNamespace(simulation_time=tt, dt=tt[1]-tt[0], num_t=len(tt))
    for k in ('geometry_type','observed_time','observed_waveform','stf_handler','frequency_cutoff','minimum_velocity',
              'maximum_velocity','misfit_interval','embedding_mode','normalize_waveform','source_amplitude',
              'source_receiver_config','objective_taper_fraction','spectral_gamma','normalization_interval',
              'absorbing','maximum_damping','montecarlo','enable_plotting','make_movie','plot_output_path'):
        setattr(model, k, args[k])
    model.assembly_dict = assembly; model.wave_type = assembly['wave_type']; model.sample_dimensions = assembly['sample_dimensions']
    model.boundary_condition = 'free'; model.source_injection = 'body_force'; model.receiver_quantity = 'velocity'
    model.boundary_impedance = None; model.store_wavefield = False; model.points_per_wavelength = grid.ppt
    model.synthetic_waveform = predicted; model.raw_synthetic_waveform = raw; model.trace_scale = float(scale)
    model.reference_trace_scale = float(scale); model.objective_components = objective; model.misfit = float(objective['total'])
    model.wavefield_forward = None; model.velocity_model_handler = vm; model.grid_handler = grid
    model.sim_time_handler = timing; model.source_handler = sh; model.receiver_handler = rh
    model.teeth_bank_options = {k:v for k,v in options.items() if k != '_bank'}
    model.teeth_bank_metadata = dict(meta, trace_scale=float(scale), material=row, diagnostic_profile_only=True)
    if args['enable_plotting']:
        path = Path(args['plot_output_path']); path.mkdir(parents=True, exist_ok=True)
        model.plotter.plot_simulation_waveform(t=t, sp_simulated=predicted, sp_recorded=args['observed_waveform'],
                  misfit_interval=args['misfit_interval'], outfile_path=path/f"bank_{index:03d}")
    return predicted, None, vm, timing, grid, sh, rh

def _maps(rows, outdir):
    import matplotlib.pyplot as plt
    for metric, filename in [('misfit', 'complete_misfit.png'), ('spectral_distance', 'spectral_shape_distance.png')]:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        densities = np.unique([r['rho'] for r in rows]); velocities = np.unique([r['cs'] for r in rows])
        for ax, nu in zip(axes.flat, np.linspace(.15, .45, 4)):
            grid = np.full((len(densities), len(velocities)), np.nan)
            for r in rows:
                if np.isclose(r['nu'], nu): grid[np.searchsorted(densities, r['rho']), np.searchsorted(velocities, r['cs'])] = r[metric]
            im = ax.imshow(grid, origin='lower', aspect='auto', extent=[velocities[0]*1000-.01, velocities[-1]*1000+.01, densities[0]-.01, densities[-1]+.01])
            ax.set(xlabel='cS [m/s]', ylabel='rho [g/cm3]', title=f'Poisson = {nu:.2f}')
            fig.colorbar(im, ax=ax, label=metric)
        fig.savefig(outdir/filename, dpi=160); plt.close(fig)

def search_bank_global(observed_waveform, waveform_metadata, outfile_name, stf_handler, params, assembly_dict):
    from .forward_modeling import UltrasonicModeler
    validate_configuration(params)
    options = dict(params['teeth_bank']); bank = P2MatrixBank(options['directory'],
                       frequency_interpolation=options.get('frequency_interpolation', 'linear'))
    indices = options.get('material_indices', list(range(200)))
    if not indices or len(set(indices)) != len(indices): raise ValueError('Material indices must be nonempty and unique')
    bank.preflight(indices); mode = 0 if assembly_dict['wave_type'] == '_p' else 1
    time = np.asarray(waveform_metadata['time_ax_waveform'], float)
    # Import the original functions lazily to retain the existing SNR/window policy.
    from . import local_optimization_FWI_velocity_1D as workflow
    snr = workflow.evaluate_snr(time, observed_waveform, assembly_dict['steel_only_time_wave'])
    if snr < params['minimum_SNR']:
        print(f'P2 bank: SNR {snr:.2f} below minimum; skipping', flush=True)
        return {'simulation_handler':None}
    if 'misfit_interval_us' in options:
        lo, hi = options['misfit_interval_us']; interval = (time >= lo)&(time <= hi)
    else:
        # Compute predicted-window bounds using ALL bank candidates, never the old grid.
        bounds = [workflow.predict_event_windows(candidate_assembly(assembly_dict, bank.rows[int(i)], mode, options),
                    [bank.rows[int(i)]['cp' if mode == 0 else 'cs']/10], wave_type=assembly_dict['wave_type'])['direct'] for i in indices]
        st = np.asarray(stf_handler.metadata['time_ax_waveform'])
        interval = workflow.direct_misfit_interval(time, arrival_min_us=min(b[0] for b in bounds),
                     arrival_max_us=max(b[1] for b in bounds), stf_duration_us=st[-1]-st[0],
                     padding_us=params['direct_window_padding_us'])
    if np.count_nonzero(interval) < 4: raise ValueError('Direct window contains fewer than four samples')
    outdir = Path(params['outdir_path_image'])/outfile_name/'elastic_teeth_bank_search'; outdir.mkdir(parents=True,exist_ok=True)
    gamma = float(params.get('global_spectral_gamma', params.get('local_spectral_gamma',1.)))
    taper = float(params.get('global_objective_taper_fraction', params.get('local_objective_taper_fraction',0.)))
    rows = []; best = None; tick = clock.monotonic()
    for k, index in enumerate(indices):
        row = bank.rows[int(index)]; a = candidate_assembly(assembly_dict, row, mode, options)
        simulation = UltrasonicModeler(); candidate_options = dict(options, material_index=int(index), _bank=bank)
        simulation.forward_simulation(geometry_type='dds', observed_time=time, observed_waveform=observed_waveform,
            stf_handler=stf_handler, frequency_cutoff=params['frequency_cutoff'], minimum_velocity=row['cp' if mode == 0 else 'cs']/10,
            maximum_velocity=max(METALS['steel']['cp' if mode == 0 else 'cs'],row['cp' if mode == 0 else 'cs'])/10,
            assembly_dict=a, misfit_interval=interval, embedding_mode='effective_surface', boundary_condition='free',
            source_injection='body_force', receiver_quantity='velocity', store_wavefield=False,
            points_per_wavelength=params.get('points_per_wavelength',8), normalize_waveform=True,
            normalization_interval=interval, source_receiver_config=params.get('source_receiver_config',{}),
            enable_plotting=False, spectral_gamma=gamma, objective_taper_fraction=taper, teeth_bank_options=candidate_options)
        spectral = simulation.write_direct_arrival_spectral_diagnostics(output_directory=None,
                       frequency_min_mhz=params.get('frequency_min',0.), frequency_max_mhz=params['frequency_cutoff'])
        result = dict(row, misfit=float(simulation.misfit), spectral_distance=float(spectral['recorded_vs_simulated_distance']),
                      trace_scale=float(simulation.trace_scale), mode=assembly_dict['wave_type'], damping=0.)
        rows.append(result)
        if best is None or simulation.misfit < best.misfit: best = simulation
        print(f"P2 bank {k+1}/{len(indices)}: index={index}, rho={row['rho']:g}, cS={row['cs']*1000:.1f}, "
              f"Poisson={row['nu']:.2f}, J={simulation.misfit:.6g}, scale={simulation.trace_scale:.6g}, "
              f"elapsed={clock.monotonic()-tick:.1f}s", flush=True)
    _maps(rows,outdir)
    with (outdir/'all_candidates.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    selected = bank.rows[best.teeth_bank_metadata['material_index']]; assembly_dict.update(best.assembly_dict)
    coarse = dict(material_index_list=np.array([r['index'] for r in rows]), poisson_list=np.array([r['nu'] for r in rows]),
                  cp_m_s_list=np.array([r['cp']*1000 for r in rows]), cs_m_s_list=np.array([r['cs']*1000 for r in rows]),
                  gouge_velocity_list=np.array([r['cp' if mode==0 else 'cs']/10 for r in rows]),
                  gouge_density_list_g_cm3=np.array([r['rho'] for r in rows]), gouge_damping_list=np.zeros(len(rows)),
                  waveform_shape_misfit_list=np.array([r['misfit'] for r in rows]),
                  spectral_shape_distance_list=np.array([r['spectral_distance'] for r in rows]),
                  trace_scale_list=np.array([r['trace_scale'] for r in rows]))
    spectral = best.write_direct_arrival_spectral_diagnostics(output_directory=outdir if params.get('spectral_diagnostics_enabled',True) else None,
                      frequency_min_mhz=params.get('frequency_min',0.),frequency_max_mhz=params['frequency_cutoff'])
    meta = dict(best.teeth_bank_metadata, material_count=len(rows), all_matrix_sha256=dict(bank.fingerprints),
                elapsed_seconds=clock.monotonic()-tick, window_us=[float(time[interval][0]),float(time[interval][-1])])
    (outdir/'best_bank_model.json').write_text(json.dumps(meta,indent=2))
    np.savez_compressed(outdir/'global_results.npz',**coarse,time_us=time,recorded=observed_waveform,
                        best_raw=best.raw_synthetic_waveform,best_scaled=best.synthetic_waveform,misfit_mask=interval,
                        best_residual=observed_waveform-best.synthetic_waveform,metadata_json=json.dumps(meta))
    best.plotter.plot_simulation_waveform(t=time,sp_simulated=best.synthetic_waveform,sp_recorded=observed_waveform,
                misfit_interval=interval,outfile_path=outdir/'best_bank_fit')
    common = selected['cp' if mode == 0 else 'cs']/10
    elastic = dict(coarse=coarse, simulation_handler=best, elastic_best_gouge_velocity=common, elastic_best_gouge_damping=0.,
                   elastic_best_gouge_density_g_cm3=selected['rho'],best_waveform_shape_misfit=best.misfit,
                   best_spectral_shape_distance=spectral['recorded_vs_simulated_distance'], best_material=selected,
                   forward_backend='p2_bank_direct', teeth_bank_metadata=meta)
    return dict(simulation_handler=best,gouge_velocity_list=coarse['gouge_velocity_list'],best_gouge_velocity=common,
        best_homogeneous_gouge_velocity=common,best_gouge_velocity_1=common,best_gouge_velocity_2=common,
        reference_gouge_thickness_1_cm=assembly_dict['gouge_thickness_1'],reference_gouge_thickness_2_cm=assembly_dict['gouge_thickness_2'],
        best_gouge_thickness_1_cm=assembly_dict['gouge_thickness_1'],best_gouge_thickness_2_cm=assembly_dict['gouge_thickness_2'],
        gouge_damping_list=coarse['gouge_damping_list'],best_gouge_damping=0.,L2norm_waveform_list=coarse['waveform_shape_misfit_list'],
        complete_misfit_list=coarse['waveform_shape_misfit_list'],homogeneous_spectral_distance_list=coarse['spectral_shape_distance_list'],
        homogeneous_spectral_diagnostics=spectral,global_spectral_gamma=gamma,homogeneous_best_misfit=best.misfit,
        velocity_contrast_result=None,gouge_thickness_result=None,joint_kinematic_result=None,density_result=None,
        impedance_material_result=None,elastic_material_result=elastic,absorption_comparison=None,selected_material_model='elastic_teeth_bank',
        post_impedance_thickness_result=None,spectral_diagnostics=spectral,spectral_shape_distance=spectral['recorded_vs_simulated_distance'],
        best_gouge_density_g_cm3=selected['rho'],teeth_bank_result=elastic,teeth_bank_metadata=meta)
