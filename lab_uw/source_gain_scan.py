"""Fixed source-gain scans on a coarse homogeneous DDS material grid.

For a linear forward operator, F[m, a*s] = a*F[m, s]. One unnormalised
forward per material is sufficient; every trial gain is then explicitly scored
with the production waveform objective. The calibrated source file is unchanged.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal.windows import tukey


def _axis(values, name, positive=True):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError(f'{name}: expected a finite, nonempty 1-D array.')
    if np.any(a <= 0 if positive else a < 0):
        raise ValueError(f'{name}: values must be {"positive" if positive else "nonnegative"}.')
    return np.unique(a)


def _coarse(values, count):
    a = _axis(values, 'material axis', positive=False)
    count = int(count)
    if count < 2:
        raise ValueError('Coarse-axis point counts must be at least two.')
    return a[np.unique(np.rint(np.linspace(0, len(a)-1, min(count, len(a)))).astype(int))]


def scan_axes(params):
    o = params['source_gain_scan']
    gain = _axis(o.get('gain_values', np.r_[np.geomspace(.1, 20., 41), 1.]), 'gain')
    c = _coarse(o.get('velocity_values_cm_us', params['impedance_velocity_values_cm_us']),
                o.get('velocity_points', 5))
    nu = _coarse(o.get('damping_values_cm2_us', params['impedance_damping_values']),
                 o.get('damping_points', 5))
    # Keep the same density nodes for P and S, with the original density spacing.
    rho = _axis(o.get('density_values_g_cm3', params['impedance_density_values_g_cm3']), 'density')
    return c, nu, rho, gain


def score_gains(observed, prediction, interval, gains, *, gamma=1., taper=0.):
    from .forward_modeling import relative_waveform_objective
    result = [relative_waveform_objective(observed, float(a)*prediction, interval,
              spectral_gamma=gamma, taper_fraction=taper) for a in gains]
    return {key: np.asarray([r[key] for r in result], dtype=float) for key in
            ['total', 'amplitude_ratio', 'amplitude_misfit', 'spectral_shape_misfit', 'phase_misfit']}


def _worker(task):
    from .forward_modeling import UltrasonicModeler
    i, arguments, gains, gamma, taper = task
    model = UltrasonicModeler()
    model.forward_simulation(**arguments)
    if float(model.trace_scale) != 1. or model.normalize_waveform:
        raise RuntimeError('The gain scan requires an unprofiled forward with trace_scale=1.')
    prediction = np.asarray(model.raw_synthetic_waveform, dtype=float)
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f'Nonfinite prediction at material candidate {i}.')
    scored = score_gains(arguments['observed_waveform'], prediction,
                         arguments['misfit_interval'], gains, gamma=gamma, taper=taper)
    if not np.isfinite(scored['total']).all():
        raise FloatingPointError(f'Nonfinite misfit at material candidate {i}.')
    return i, prediction, scored


def _write_json(path, content):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(content, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    return value


def export_input_case(path, *, observed, time, mask, source, params, assembly):
    """Small portable input: the exact preprocessed trace, source and geometry."""
    keys=['absorbing','embedding_mode','boundary_condition','source_injection','receiver_quantity',
          'frequency_cutoff','frequency_min','direct_window_padding_us','global_spectral_gamma',
          'global_objective_taper_fraction','local_spectral_gamma','local_objective_taper_fraction',
          'points_per_wavelength','impedance_points_per_wavelength','min_velocity2simulate',
          'max_velocity2simulate','source_receiver_config','side1_density_g_cm3',
          'central_density_g_cm3','side2_density_g_cm3','steel_density_g_cm3','global_search_workers',
          'parallel_cpu_reserve','impedance_velocity_values_cm_us','impedance_damping_values',
          'impedance_density_values_g_cm3','velocity_initial_list','source_gain_scan']
    meta=_plain(dict(schema=1,assembly=assembly,params={k:params[k] for k in keys if k in params},
                     source_calibration=source.metadata['source_calibration']))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,observed_waveform=observed,time_us=time,
        misfit_mask=mask,source_time_us=source.metadata['time_ax_waveform'],
        source_waveform=source.waveform_data,metadata_json=np.array(json.dumps(meta,allow_nan=False)))
    tmp.replace(path)


def _save_png(fig, path):
    import matplotlib.pyplot as plt
    tmp = path.with_suffix('.tmp')
    fig.savefig(tmp, format='png', dpi=200, facecolor='white')
    plt.close(fig)
    tmp.replace(path)


def profiled_density(J):
    """J axes are (c, nu, rho, a); retain density and gain sampled coordinates."""
    return np.min(J, axis=(0, 1))


def gain_minima(c, nu, rho, gains, J):
    table = []
    for ia, a in enumerate(gains):
        slab = J[..., ia]
        ic, inu, irho = np.unravel_index(np.argmin(slab), slab.shape)
        value = float(slab[ic, inu, irho])
        density_profile = np.min(slab, axis=(0, 1))
        ties = np.flatnonzero(np.isclose(density_profile, value, rtol=1e-10, atol=1e-12))
        table.append(dict(source_gain=float(a), misfit=value, velocity_cm_us=float(c[ic]),
            damping_cm2_us=float(nu[inu]), density_g_cm3=float(rho[irho]),
            tied_minimum_densities_g_cm3=';'.join(format(float(rho[i]), '.17g') for i in ties),
            velocity_on_boundary=ic in (0, len(c)-1),
            damping_on_boundary=inu in (0, len(nu)-1),
            density_on_boundary=irho in (0, len(rho)-1)))
    return table


def plot_maps(c, nu, rho, gains, J, output, context):
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm, LogNorm
    from matplotlib.ticker import MaxNLocator,LogLocator,NullFormatter
    # All maps use native candidate nodes and a COMMON colour scale.
    deltas = J - np.min(J, axis=(0, 1, 2), keepdims=True)
    positive = deltas[deltas > 1e-14]
    vmax = max(float(np.max(positive)) if positive.size else 1., 1e-12)
    width = max(float(np.percentile(positive, 2)) if positive.size else vmax, vmax*1e-8)
    norm = SymLogNorm(linthresh=width, vmin=0., vmax=vmax, base=10)
    maps = output/'maps_by_gain';maps.mkdir(exist_ok=True)
    minima = gain_minima(c, nu, rho, gains, J)
    for ia, a in enumerate(gains):
        slab = deltas[..., ia];best = minima[ia]
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), layout='constrained')
        panels = [
            (c*1e4, rho*1e3, slab.min(axis=1).T, 'Velocity [m/s]', r'Density [kg/m$^3$]',
             best['velocity_cm_us']*1e4, best['density_g_cm3']*1e3),
            (nu*100, rho*1e3, slab.min(axis=0).T, r'Kinematic damping [m$^2$/s]', r'Density [kg/m$^3$]',
             best['damping_cm2_us']*100, best['density_g_cm3']*1e3),
            (c*1e4, nu*100, slab.min(axis=2).T, 'Velocity [m/s]', r'Kinematic damping [m$^2$/s]',
             best['velocity_cm_us']*1e4, best['damping_cm2_us']*100)]
        for ax, (xx, yy, zz, xl, yl, bx, by) in zip(axes, panels):
            mesh = ax.pcolormesh(xx, yy, zz, shading='nearest', cmap='viridis_r', norm=norm)
            ax.scatter([bx], [by], marker='*', s=115, facecolor='#f7c95a', edgecolor='black', linewidth=.6)
            ax.set(xlabel=xl, ylabel=yl)
            ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(style='plain', useOffset=False)
        fig.colorbar(mesh, ax=axes, label=r'$\Delta J=J-J_{\min}(a)$; common scale')
        fig.suptitle(f'{context}   |   a={a:.6g}   |   Jmin={best["misfit"]:.6g}   |   '
                     f'rho at minimum={best["density_g_cm3"]*1000:.0f} kg/m³', fontsize=11)
        name=f'{ia:03d}_a_{a:.8g}'.replace('.', 'p')+'.png'
        _save_png(fig, maps/name)
    R = profiled_density(J)
    positive = R[R>0]
    low = max(float(positive.min()) if positive.size else 1e-12, 1e-12)
    high = max(float(R.max()), low*1.001)
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.1), layout='constrained')
    mesh = axes[0].pcolormesh(gains, rho*1e3, np.maximum(R,low), shading='nearest',
                              norm=LogNorm(low,high), cmap='viridis_r')
    axes[0].set(xscale='log', xlabel='Fixed source gain a', ylabel=r'Density [kg/m$^3$]')
    fig.colorbar(mesh, ax=axes[0], label=r'$\min_{c,\nu} J$')
    axes[1].plot(gains, [r['density_g_cm3']*1000 for r in minima], 'o-', ms=3)
    axes[1].set(xscale='log', xlabel='Fixed source gain a', ylabel=r'Density at minimum [kg/m$^3$]')
    for ax in axes:
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[1].grid(alpha=.2);fig.suptitle(context)
    _save_png(fig, output/'gain_density_summary.png')


def run_selected(functions, *, paired, observed_waveform_data, waveform_metadata,
                 stf_handler, params, assembly_dict, infile_path):
    """Dedicated diagnostic: no local inversion, restart or profiled amplitude."""
    from .source_amplitude import validate_handler
    o = params['source_gain_scan']
    source_meta = validate_handler(stf_handler)
    if source_meta is None:
        raise ValueError('source_gain_scan requires calibrated_source_path for this polarization: '
                         'load the steel-reference *.source.npz before running the scan.')
    if params.get('teeth_bank_enabled') or params.get('teeth_continuous_enabled'):
        raise ValueError('This scan uses the current DDS finite-difference/ramp operator; '
                         'teeth-bank and continuous-teeth modes must be disabled.')
    c, nu, rho, gains = scan_axes(params)
    if float(params.get('source_receiver_config', {}).get('source_amplitude',1.)) != 1.:
        raise ValueError('Keep source_amplitude=1; source_gain_scan.gain_values contains a.')
    time = np.asarray(waveform_metadata['time_ax_waveform'], dtype=float)
    records = o.get('record_numbers', [334906])
    wanted = None if records is None else {int(v) for v in records}
    if wanted is not None and not wanted:
        raise ValueError('record_numbers must contain at least one record or be None for the first.')
    chosen=[]
    for row in paired.itertuples(index=False):
        if not np.isfinite(getattr(row,'Index',np.nan)):
            continue
        assembly = deepcopy(assembly_dict)
        functions['update_assembly_dict_with_mech_data'](assembly_dict=assembly, mech_data=row)
        rec = int(assembly['rec_n'])
        if wanted is None or rec in wanted:
            chosen.append((row, assembly))
            if wanted is None:break
    found={int(a['rec_n']) for _,a in chosen}
    if wanted is not None and found != wanted:
        raise ValueError(f'Requested scan records not available after selection: {sorted(wanted-found)}')
    if not chosen:
        raise ValueError('No valid acquisition selected for the source-gain scan.')
    if len(found)!=len(chosen):
        raise ValueError('A requested mechanical record maps to more than one waveform.')
    gamma = float(params.get('global_spectral_gamma', params.get('local_spectral_gamma',1.)))
    taper = float(params.get('global_objective_taper_fraction',params.get('local_objective_taper_fraction',0.)))
    ppw = int(o.get('points_per_wavelength', params.get('impedance_points_per_wavelength',8)))
    if ppw < 2:raise ValueError('points_per_wavelength must be at least two.')
    output_root=Path(params['outdir_path_l2norm'])/'source_gain_scan'
    outputs=[]
    for row, assembly in chosen:
        observed=np.asarray(observed_waveform_data[int(row.uw_idx)], dtype=float)
        if time.shape != observed.shape or not np.isfinite(observed).all():
            raise ValueError('Observed trace and time must have the same finite 1-D shape.')
        explicit=o.get('window_us')
        if explicit is None:
            windows=functions['predict_event_windows'](assembly,params['velocity_initial_list'],wave_type=assembly['wave_type'])
            st=np.asarray(stf_handler.metadata['time_ax_waveform'])
            interval=functions['direct_misfit_interval'](time,
                arrival_min_us=windows['direct'][0],arrival_max_us=windows['direct'][1],
                stf_duration_us=float(st[-1]-st[0]),padding_us=params['direct_window_padding_us'])
            mask=functions['sample_interval_mask'](interval, len(time))
        else:
            lo,hi=map(float,explicit)
            if not np.isfinite([lo,hi]).all() or lo>=hi or lo<time[0] or hi>time[-1]:
                raise ValueError('window_us must be increasing and inside the observed samples.')
            mask=(time>=lo)&(time<=hi)
        if mask.sum()<4:raise ValueError('Source-gain scan window has fewer than four samples.')
        weights=tukey(int(mask.sum()),alpha=taper) if taper else np.ones(int(mask.sum()))
        if np.linalg.norm(observed[mask]*weights)<=1e-12:
            raise ValueError('Observed energy in scan window is too small for a relative objective.')
        case=output_root/'inputs'/f'{Path(infile_path).stem}_rec_{assembly["rec_n"]}_input.npz'
        export_input_case(case,observed=observed,time=time,mask=mask,source=stf_handler,
                          params=params,assembly=assembly)
        if o.get('export_only',False):
            outputs.append(case)
            print(f'Portable source-gain input saved: {case}',flush=True)
            continue
        base=functions['_global_objective_options'](params)
        args=[]
        for ic,inu,ir in product(range(len(c)),range(len(nu)),range(len(rho))):
            material=functions['_assembly_with_common_gouge_material'](deepcopy(assembly),params,
                gouge_velocity=c[ic],gouge_damping=nu[inu],gouge_density=rho[ir])
            minimum,maximum=functions['_simulation_velocity_limits'](params,material)
            forward=dict(geometry_type='dds',observed_time=time,observed_waveform=observed,
                stf_handler=stf_handler,frequency_cutoff=params['frequency_cutoff'],
                minimum_velocity=minimum,maximum_velocity=maximum,assembly_dict=material,
                misfit_interval=mask,absorbing=params['absorbing'],maximum_damping=float(nu[inu]),
                embedding_mode=params['embedding_mode'],boundary_condition=params['boundary_condition'],
                source_injection=params['source_injection'],receiver_quantity=params['receiver_quantity'],
                points_per_wavelength=ppw,store_wavefield=False,normalize_waveform=False,
                source_receiver_config=deepcopy(params['source_receiver_config']),enable_plotting=False,**base)
            args.append(((ic,inu,ir),forward,gains,gamma,taper))
        workers=functions['resolve_worker_count'](params,'global_search_workers',len(args))
        context=f'{str(assembly["wave_type"]).strip("_").upper()} waves - record {assembly["rec_n"]}'
        print(f'{context}: {len(args)} FD models, {len(gains)} gains each; {ppw} PPW; {workers} workers.',flush=True)
        J=np.empty((len(c),len(nu),len(rho),len(gains)))
        amplitude=np.empty_like(J);shape=np.empty_like(J)
        traces=np.empty((len(args),len(time)));indices=np.empty((len(args),3),dtype=int)
        def gather(results):
            for n,(idx,pred,scores) in enumerate(results):
                J[idx]=scores['total'];amplitude[idx]=scores['amplitude_ratio']
                shape[idx]=scores['spectral_shape_misfit'];traces[n]=pred;indices[n]=idx
                if (n+1)%max(1,len(args)//10)==0 or n+1==len(args):
                    print(f'  gain scan: {n+1}/{len(args)} material models',flush=True)
        if workers==1:gather(map(_worker,args))
        else:
            with Pool(processes=workers) as pool:gather(pool.imap(_worker,args,chunksize=1))
        digest=hashlib.sha256()
        for v in [time,observed,mask.astype(float),c,nu,rho,gains]:digest.update(np.ascontiguousarray(v).tobytes())
        digest.update(json.dumps(assembly,sort_keys=True,default=lambda v:np.asarray(v).tolist()).encode())
        digest.update(json.dumps(dict(source_id=source_meta['id'],gamma=gamma,taper=taper,ppw=ppw,
            receiver_config=params['source_receiver_config'],embedding=params['embedding_mode'],
            boundary=params['boundary_condition'],receiver=params['receiver_quantity'],
            source_injection=params['source_injection'],frequency_cutoff=params['frequency_cutoff']),sort_keys=True).encode())
        out=output_root/f'{Path(infile_path).stem}_rec_{assembly["rec_n"]}_{digest.hexdigest()[:12]}'
        out.mkdir(parents=True,exist_ok=True)
        minima=gain_minima(c,nu,rho,gains,J)
        best_index=tuple(map(int,np.unravel_index(np.argmin(J),J.shape)))
        meta=dict(schema=1,axis_order=['velocity_cm_us','damping_cm2_us','density_g_cm3','source_gain'],
            wave_type=str(assembly['wave_type']).strip('_').upper(),rec_n=int(assembly['rec_n']),
            acquisition_time_s=float(assembly['acquisition_time']),source_calibration_id=source_meta['id'],
            reference_gain=float(source_meta['reference_gain']),DDS_trace_scale=1.,amplitude_profiled=False,
            gain_applied='a times the unit-gain raw receiver trace; exact linear source scaling',
            spectral_gamma=gamma,objective_taper_fraction=taper,points_per_wavelength=ppw,
            frequency_min_mhz=float(params['frequency_min']),frequency_cutoff_mhz=float(params['frequency_cutoff']),
            source_receiver_config=params['source_receiver_config'],window_us=[float(time[mask][0]),float(time[mask][-1])],
            number_of_forwards=len(args),number_of_objective_evaluations=int(J.size),
            best_index=list(best_index),best_sampled=minimum_at_index(c,nu,rho,gains,J,best_index),
            density_tie_tolerance=dict(rtol=1e-10,atol=1e-12),
            maps='At fixed a, minimize only the unshown material parameter; no interpolation.',
            interpretation='Sampled grid minima; uniqueness between samples is not established.')
        dest=out/'source_gain_scan.npz';tmp=dest.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,velocity_cm_us=c,damping_cm2_us=nu,density_g_cm3=rho,
            source_gain=gains,misfit=J,amplitude_ratio=amplitude,spectral_shape_misfit=shape,
            gain_density_misfit=profiled_density(J),baseline_traces=traces,baseline_material_indices=indices,
            time_us=time,observed_waveform=observed,misfit_mask=mask,metadata_json=np.array(json.dumps(meta)))
        tmp.replace(dest)
        _write_json(out/'summary.json',meta)
        with (out/'minima_by_gain.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(minima[0]));w.writeheader();w.writerows(minima)
        with (out/'all_candidates.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['velocity_cm_us','damping_cm2_us','density_g_cm3','source_gain','misfit','amplitude_ratio'])
            for ic,inu,ir,ia in np.ndindex(J.shape):
                w.writerow([format(float(v),'.17g') for v in [c[ic],nu[inu],rho[ir],gains[ia],J[ic,inu,ir,ia],amplitude[ic,inu,ir,ia]]])
        if o.get('plot_maps',True):plot_maps(c,nu,rho,gains,J,out,context)
        outputs.append(dest)
        print(f'Source-gain scan saved: {dest}',flush=True)
    return outputs


def minimum_at_index(c,nu,rho,gains,J,index):
    ic,inu,ir,ia=index
    return dict(velocity_cm_us=float(c[ic]),damping_cm2_us=float(nu[inu]),
                density_g_cm3=float(rho[ir]),source_gain=float(gains[ia]),misfit=float(J[index]))
