"""Profile homogeneous search scores in rho, M and eta, without new forwards.

Input units follow lab_uw: rho g/cm3, v cm/us, nu cm2/us. Output is SI.
The interpolant is linear in the ORIGINAL search coordinates (rho,v,nu).
The nuisance parameter is minimized over its original sampled nodes. No
extrapolation, confidence intervals, or new amplitude calibration is used.
"""
from __future__ import annotations
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from scipy.interpolate import RegularGridInterpolator

REVISION = 1


def constitutive_rows(results):
    rows = []
    for source in results:
        row = dict(source)
        rho = float(row['gouge_density_g_cm3'])
        v = float(row['gouge_velocity'])
        nu = float(row['gouge_damping'])
        if not all(np.isfinite([rho, v, nu])) or rho <= 0 or v <= 0 or nu < 0:
            raise ValueError('Material search requires finite rho>0, v>0, nu>=0.')
        row.update(rho_kg_m3=rho*1000., velocity_m_s=v*1.e4, nu_m2_s=nu*100.,
                   modulus_Pa=rho*v*v*1.e11, viscosity_Pa_s=rho*nu*1.e5)
        rows.append(row)
    if not rows:
        raise ValueError('The material search table is empty.')
    return rows


class NativeGrid:
    """Strict multilinear interpolation; missing cells are not filled."""
    def __init__(self, rows, metric):
        self.axes = tuple(np.unique([r[k] for r in rows]) for k in
                          ['gouge_density_g_cm3', 'gouge_velocity', 'gouge_damping'])
        if int(np.prod([len(a) for a in self.axes])) > 2_000_000:
            raise ValueError('Search table is too sparse for a Cartesian profile map; use a regular native grid.')
        self.grid = np.full(tuple(len(a) for a in self.axes), np.nan)
        for row in rows:
            value = float(row[metric])
            if not np.isfinite(value):
                continue
            idx = tuple(int(np.searchsorted(a, row[k])) for a, k in zip(self.axes,
                ['gouge_density_g_cm3', 'gouge_velocity', 'gouge_damping']))
            old = self.grid[idx]
            self.grid[idx] = min(value, old) if np.isfinite(old) else value
        if not np.any(np.isfinite(self.grid)):
            raise ValueError(f'No finite scores for {metric}.')
        self.origin = np.array([a[0] for a in self.axes])
        self.scale = np.array([a[-1]-a[0] if len(a)>1 else max(abs(a[0]), 1.) for a in self.axes])
        normalized = tuple((a-o)/s for a, o, s in zip(self.axes, self.origin, self.scale))
        self.interpolator = RegularGridInterpolator(normalized, self.grid, method='linear',
                                                     bounds_error=False, fill_value=np.nan)

    def __call__(self, points):
        points = np.asarray(points, float)
        shape = points.shape[:-1]
        q = points.reshape(-1, 3).copy()
        # Only roundoff-sized boundary overshoots are snapped to the grid.
        nearest = []
        exact = np.ones(len(q), dtype=bool)
        for j, a in enumerate(self.axes):
            i = np.searchsorted(a, q[:, j]); i = np.clip(i, 0, len(a)-1)
            previous = np.maximum(i-1, 0)
            use_previous = abs(q[:, j]-a[previous]) < abs(q[:, j]-a[i])
            i[use_previous] = previous[use_previous]
            tol = 64*np.finfo(float).eps*np.maximum(abs(a[i]), np.finfo(float).tiny)
            on_node = abs(q[:, j]-a[i]) <= tol
            q[on_node, j] = a[i[on_node]]
            exact &= on_node; nearest.append(i)
        values = np.asarray(self.interpolator((q-self.origin)/self.scale))
        # Exact nodes remain valid next to missing cells (0*NaN must not erase them).
        if np.any(exact):
            values[exact] = self.grid[tuple(i[exact] for i in nearest)]
        return values.reshape(shape)


def _display_axis(values, best, count=64, zero_allowed=False):
    values = np.asarray(values, float)
    lo, hi = float(np.min(values)), float(np.max(values))
    if lo == hi:
        return np.array([lo])
    positive = values[values>0]
    if lo == 0 and zero_allowed:
        sampled = np.r_[0., np.geomspace(float(np.min(positive)), hi, count)]
    else:
        sampled = np.geomspace(lo, hi, count)
    return np.unique(np.r_[sampled, lo, hi, best])


def _reduce(values, nuisance):
    valid = np.isfinite(values)
    safe = np.where(valid, values, np.inf)
    which = np.argmin(safe, axis=-1)
    count = valid.sum(axis=-1)
    score = np.take_along_axis(safe, which[..., None], axis=-1)[..., 0]
    selected = np.asarray(nuisance)[which]
    score[count==0] = np.nan
    selected[count==0] = np.nan
    return score, selected, count


def make_profiles(results, metric='waveform_shape_misfit'):
    rows = constitutive_rows(results)
    finite = [r for r in rows if np.isfinite(float(r[metric]))]
    if not finite:
        raise ValueError(f'No finite candidate scores for {metric}.')
    best = min(finite, key=lambda r:float(r[metric]))
    native = NativeGrid(rows, metric)
    rho, v, nu = native.axes
    M = _display_axis([r['modulus_Pa'] for r in rows], best['modulus_Pa'])
    eta = _display_axis([r['viscosity_Pa_s'] for r in rows], best['viscosity_Pa_s'], zero_allowed=True)
    # Preserve exact per-density minima, rather than losing the sampled valley
    # to the arbitrary display mesh. Remaining pixels still use interpolation.
    density_minima=[min((r for r in finite if r['gouge_density_g_cm3']==density),
                        key=lambda r:float(r[metric])) for density in rho
                    if any(r['gouge_density_g_cm3']==density for r in finite)]
    M=np.unique(np.r_[M,[r['modulus_Pa'] for r in density_minima]])
    eta=np.unique(np.r_[eta,[r['viscosity_Pa_s'] for r in density_minima]])
    profiles = []
    # y=rho, x=M. For each rho, minimizing eta is equivalent to minimizing native nu.
    if len(rho)>1 and len(M)>1:
        xx, yy = np.meshgrid(M, rho)
        points = np.stack(np.broadcast_arrays(yy[...,None],
            np.sqrt(xx[...,None]/(yy[...,None]*1.e11)), nu[None,None,:]), axis=-1)
        score, argmin, support = _reduce(native(points), nu)
        profiles.append(dict(name='rho_M', x=M, y=rho*1000., score=score,
            nuisance_at_min=argmin*yy*1.e5, support_count=support, nuisance_total=len(nu),
            x_key='modulus_Pa', y_key='rho_kg_m3', nuisance_key='viscosity_Pa_s'))
    # y=rho, x=eta; minimize over the native velocity nodes.
    if len(rho)>1 and len(eta)>1:
        xx, yy = np.meshgrid(eta, rho)
        points = np.stack(np.broadcast_arrays(yy[...,None], v[None,None,:],
            xx[...,None]/(yy[...,None]*1.e5)), axis=-1)
        score, argmin, support = _reduce(native(points), v)
        profiles.append(dict(name='rho_eta', x=eta, y=rho*1000., score=score,
            nuisance_at_min=argmin**2*yy*1.e11, support_count=support, nuisance_total=len(v),
            x_key='viscosity_Pa_s', y_key='rho_kg_m3', nuisance_key='modulus_Pa'))
    # y=eta, x=M; minimize over the sampled density nodes, without extrapolation.
    if len(M)>1 and len(eta)>1:
        xx, yy = np.meshgrid(M, eta)
        points = np.stack(np.broadcast_arrays(rho[None,None,:],
            np.sqrt(xx[...,None]/(rho[None,None,:]*1.e11)),
            yy[...,None]/(rho[None,None,:]*1.e5)), axis=-1)
        score, argmin, support = _reduce(native(points), rho)
        profiles.append(dict(name='M_eta', x=M, y=eta, score=score,
            nuisance_at_min=argmin*1000., support_count=support, nuisance_total=len(rho),
            x_key='modulus_Pa', y_key='viscosity_Pa_s', nuisance_key='rho_kg_m3'))
    for p in profiles:
        p['delta'] = p['score']-float(best[metric])
        finite_delta = p['delta'][np.isfinite(p['delta'])]
        if len(finite_delta) and min(finite_delta) < -1.e-9*max(1., abs(float(best[metric]))):
            raise AssertionError('Interpolation created a score below every evaluated candidate.')
        p['delta'] = np.maximum(p['delta'], 0.)
    return rows, best, profiles, native


def _axis_scale(axis, values, which):
    setter = axis.set_xscale if which=='x' else axis.set_yscale
    if np.min(values)>0:
        setter('log')
    elif np.any(values>0):
        setter('symlog', linthresh=float(np.min(values[values>0]))/2, linscale=.6)


def _save_rows(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r if np.isscalar(r[k]) or r[k] is None))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:row.get(k) for k in keys})


def _save_figure(fig, path, **kwargs):
    """Publish a complete figure atomically, including during live inspection."""
    buffer=io.BytesIO()
    fig.savefig(buffer, format=path.suffix[1:], **kwargs)
    payload=buffer.getvalue()
    if path.suffix=='.png' and not payload.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82'):
        raise OSError('Incomplete PNG render.')
    descriptor,name=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally: Path(name).unlink(missing_ok=True)


def plot_material_maps(results, *, metric_key, metric_label, output_path,
                       elastic_only=False, scale_policy='not_recorded', context_label=''):
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    rows, best, profiles, native = make_profiles(results, metric_key)
    stem = output_path.with_suffix('')
    _save_rows(rows, stem.with_name(stem.name+'_candidates_moduli.csv'))
    if not profiles:
        # A diagnostic must not abort a valid one-candidate/one-variable search.
        fig,axis=plt.subplots(figsize=(7,4),constrained_layout=True)
        variables=[k for k in ['rho_kg_m3','modulus_Pa','viscosity_Pa_s'] if len({r[k] for r in rows})>1]
        if variables:
            key=variables[0]
            ordered=sorted(rows,key=lambda r:r[key])
            axis.plot([r[key] for r in ordered], [float(r[metric_key])-float(best[metric_key]) for r in ordered],'o-')
            _axis_scale(axis,np.array([r[key] for r in ordered]),'x')
            axis.set(xlabel=key,ylabel='Punteggio − minimo')
        else:
            axis.axis('off')
            axis.text(.5,.5,'Un solo candidato: nessun piano di ricerca 2D.\n'
                f"ρ={best['rho_kg_m3']:.6g} kg/m³; M={best['modulus_Pa']:.6g} Pa; η={best['viscosity_Pa_s']:.6g} Pa s\n"
                f"Punteggio={float(best[metric_key]):.6g}",ha='center',va='center',transform=axis.transAxes)
        axis.set_title(f'{context_label + ": " if context_label else ""}{metric_label}: ricerca con meno di due dimensioni libere')
        _save_figure(fig,output_path,dpi=180);_save_figure(fig,output_path.with_suffix('.pdf'));plt.close(fig)
        metadata=dict(revision=REVISION,metric=metric_key,amplitude_scale_policy=scale_policy,
            reference_minimum=float(best[metric_key]),profiles=[],status='fewer_than_two_varying_coordinates',
            native_grid_shape=list(native.grid.shape),confidence_intervals=False)
        stem.with_name(stem.name+'_moduli_profiles.json').write_text(json.dumps(metadata,indent=2))
        np.savez_compressed(stem.with_name(stem.name+'_moduli_profiles.npz'),
            candidate_score=np.array([float(r[metric_key]) for r in rows]))
        return metadata
    positive = np.concatenate([p['delta'][np.isfinite(p['delta']) & (p['delta']>1.e-14)] for p in profiles])
    vmax = float(np.max(positive)) if len(positive) else 1.
    linear_width = max(float(np.percentile(positive, 5)), vmax*1.e-6) if len(positive) else 1.
    norm = SymLogNorm(linthresh=linear_width, vmin=0., vmax=vmax, base=10)
    titles = {'rho_M':r'$\min_{\eta}\,J(\rho,M,\eta)$',
              'rho_eta':r'$\min_{M}\,J(\rho,M,\eta)$',
              'M_eta':r'$\min_{\rho}\,J(\rho,M,\eta)$'}
    labels = {'modulus_Pa':r'Modulo $M$ [Pa]', 'viscosity_Pa_s':r'Viscosità $\eta$ [Pa s]',
              'rho_kg_m3':r'Densità $\rho$ [kg/m³]'}
    fig, axes = plt.subplots(1, len(profiles), figsize=(5.2*len(profiles), 5.),
                             squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#d9d9d9')
    levels = np.geomspace(max(linear_width, vmax*1.e-6), vmax, 7) if vmax>linear_width else []
    for axis, p in zip(axes.flat, profiles):
        axis.set_facecolor('#d9d9d9')
        image = axis.pcolormesh(p['x'], p['y'], np.ma.masked_invalid(p['delta']),
                               shading='auto', cmap=cmap, norm=norm, rasterized=True)
        if len(levels)>1 and np.count_nonzero(np.isfinite(p['delta']))>4:
            axis.contour(p['x'], p['y'], np.ma.masked_invalid(p['delta']),
                         levels=levels, colors='white', linewidths=.45, alpha=.65)
        _axis_scale(axis, p['x'], 'x'); _axis_scale(axis, p['y'], 'y')
        axis.set_xlim(p['x'][0], p['x'][-1]); axis.set_ylim(p['y'][0], p['y'][-1])
        axis.scatter(best[p['x_key']], best[p['y_key']], marker='*', s=125,
                     c='white', edgecolors='black', linewidths=.8, zorder=5)
        minima=[min((r for r in rows if r['gouge_density_g_cm3']==density and np.isfinite(float(r[metric_key]))),
                    key=lambda r:float(r[metric_key])) for density in native.axes[0]
                if any(r['gouge_density_g_cm3']==density and np.isfinite(float(r[metric_key])) for r in rows)]
        axis.scatter([r[p['x_key']] for r in minima], [r[p['y_key']] for r in minima],
                     marker='o', s=14, c='none', edgecolors='black', linewidths=.6,
                     zorder=4, label='Minimi campionati a ρ fissata')
        # Reference paths preserve bulk k; they are not fitted Hessian eigenvectors.
        x = p['x']
        if p['name']=='rho_M': y=x/(best['velocity_m_s']**2)
        elif p['name']=='rho_eta':
            y=x/best['nu_m2_s'] if best['nu_m2_s']>0 else np.full_like(x, np.nan)
        else: y=x*(best['viscosity_Pa_s']/best['modulus_Pa'])
        keep = (y>=p['y'][0]) & (y<=p['y'][-1])
        if np.count_nonzero(keep)>1:
            axis.plot(x[keep], y[keep], '--', c='#e9553b', lw=1.5, label='Scala comune di ρ, M, η')
        axis.legend(fontsize=7, loc='best', framealpha=.85)
        axis.set(xlabel=labels[p['x_key']], ylabel=labels[p['y_key']], title=titles[p['name']])
        axis.grid(alpha=.12)
    diagnostic=metric_key=='spectral_shape_distance'
    color_label=(r'$\Delta D_f=D_f-D_{f,\min}$' if diagnostic else r'$\Delta J=J-J_{\min}$')
    fig.colorbar(image, ax=list(axes.flat), label=color_label+' (scala log oltre la zona lineare)', shrink=.88)
    label = 'Omogeneo elastico (η=0)' if elastic_only else 'Omogeneo Kelvin–Voigt'
    fig.suptitle(f'{context_label + ": " if context_label else ""}{label} · {metric_label}\n'
                 f'{"D_f" if diagnostic else "J"} minimo campionato = {float(best[metric_key]):.6g}; ampiezza: {scale_policy}', fontsize=12)
    fig.supxlabel('Minimo sui nodi del parametro escluso; interpolazione nella griglia originaria. Grigio: fuori supporto.', fontsize=9)
    _save_figure(fig, output_path, dpi=180)
    _save_figure(fig, output_path.with_suffix('.pdf'))
    plt.close(fig)
    arrays = {}
    for p in profiles:
        for key in ['x','y','score','delta','nuisance_at_min','support_count']:
            arrays[p['name']+'_'+key] = p[key]
    np.savez_compressed(stem.with_name(stem.name+'_moduli_profiles.npz'), **arrays)
    metadata = dict(revision=REVISION, metric=metric_key, metric_label=metric_label,
        amplitude_scale_policy=scale_policy, context_label=context_label,
        reference_minimum=float(best[metric_key]),
        best={k:float(best[k]) for k in ['rho_kg_m3','velocity_m_s','nu_m2_s','modulus_Pa','viscosity_Pa_s']},
        native_grid_shape=list(native.grid.shape), evaluated_nodes=int(np.count_nonzero(np.isfinite(native.grid))),
        interpolation='linear in native rho_g_cm3,v_cm_us,nu_cm2_us; no extrapolation; missing cells masked',
        profiling='min over original nuisance samples, not a continuous new forward optimization',
        display_axes='log positive coordinates; symlog includes actual eta=0 without replacing it',
        color_linear_width=linear_width,
        confidence_intervals=False,
        warning='Contour levels are score differences, not statistical confidence thresholds. Finite search bounds and changing nuisance coverage can truncate valleys.',
        profiles=[{k:v for k,v in p.items() if k in ['name','x_key','y_key','nuisance_key','nuisance_total']} for p in profiles])
    stem.with_name(stem.name+'_moduli_profiles.json').write_text(json.dumps(metadata, indent=2))
    return metadata


def plot_teeth_maps(rows, outdir, mode=None, context_label=''):
    """Keep Poisson fixed on each page: its effect in teeth is not a 1D modulus."""
    if mode is None:
        modes = {r.get('mode') for r in rows}
        if modes == {'_p'}: mode=0
        elif modes == {'_s'}: mode=1
        else: raise ValueError('Cannot infer P/S polarization from bank results.')
    poissons = sorted({float(r['nu']) for r in rows})
    outdir = Path(outdir)
    for index, poisson in enumerate(poissons):
        normalized = []
        for r in rows:
            if float(r['nu']) != poisson: continue
            normalized.append(dict(r, gouge_density_g_cm3=float(r['rho']),
                gouge_velocity=float(r['cp' if mode==0 else 'cs'])/10.,
                gouge_damping=float(r.get('damping_cm2_us', r.get('damping', 0.))),
                waveform_shape_misfit=float(r['misfit']), spectral_shape_distance=float(r['spectral_distance'])))
        suffix = '' if len(poissons)==1 else f'_poisson_{index+1:02d}'
        elastic = all(r['gouge_damping']==0 for r in normalized)
        for metric, filename, label in [('waveform_shape_misfit','complete_misfit','J globale'),
                                         ('spectral_shape_distance','spectral_shape_distance','Distanza spettrale')]:
            plot_material_maps(normalized, metric_key=metric, metric_label=label,
                output_path=outdir/(filename+suffix+'.png'), elastic_only=elastic,
                scale_policy='profilata per candidato',
                context_label=f'{context_label + "; " if context_label else ""}Poisson={poisson:g}')


def load_results(path):
    path = Path(path)
    if path.suffix.lower()=='.csv':
        with path.open(newline='') as stream: raw=list(csv.DictReader(stream))
        rows=[]
        for r in raw:
            row={}
            for k,v in r.items():
                try: row[k]=float(v)
                except (ValueError, TypeError): row[k]=v
            rows.append(row)
        return rows
    if path.suffix.lower()=='.npz':
        with np.load(path, allow_pickle=False) as a:
            aliases = {'gouge_velocity':'gouge_velocity_list','gouge_damping':'gouge_damping_list',
                'gouge_density_g_cm3':'gouge_density_list_g_cm3',
                'waveform_shape_misfit':'waveform_shape_misfit_list',
                'spectral_shape_distance':'spectral_shape_distance_list','trace_scale':'trace_scale_list'}
            columns={k:np.asarray(a[v]) for k,v in aliases.items() if v in a}
            if not all(k in columns for k in list(aliases)[:4]):
                raise ValueError('NPZ lacks the candidate density/velocity/damping/misfit vectors.')
            n=len(columns['gouge_velocity'])
            if 'poisson_list' in a:
                columns.update(nu=np.asarray(a['poisson_list']), cp=np.asarray(a['cp_m_s_list'])/1000.,
                               cs=np.asarray(a['cs_m_s_list'])/1000., rho=columns['gouge_density_g_cm3'],
                               damping_cm2_us=columns['gouge_damping'], misfit=columns['waveform_shape_misfit'])
                if 'spectral_shape_distance' in columns:
                    columns['spectral_distance']=columns['spectral_shape_distance']
            rows=[{k:float(v[i]) for k,v in columns.items()} for i in range(n)]
            if 'nu' in columns:
                is_p=np.allclose(columns['gouge_velocity']*10, columns['cp'], rtol=1.e-10, atol=0.)
                is_s=np.allclose(columns['gouge_velocity']*10, columns['cs'], rtol=1.e-10, atol=0.)
                if is_p != is_s:
                    for row in rows: row['mode']='_p' if is_p else '_s'
            return rows
    raise ValueError('Use the saved candidate CSV or global_results.npz.')


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wave', choices=['P','S'])
    args=parser.parse_args(argv)
    rows=load_results(args.input)
    out=args.output or args.input.parent/'moduli_maps'
    if 'gouge_velocity' not in rows[0] or all(k in rows[0] for k in ['cp','cs','nu','rho']):
        plot_teeth_maps(rows, out, mode={'P':0,'S':1}.get(args.wave))
    else:
        out.mkdir(parents=True, exist_ok=True)
        for metric, name in [('waveform_shape_misfit','complete_misfit'),('spectral_shape_distance','spectral_shape_distance')]:
            if metric not in rows[0]: continue
            plot_material_maps(rows, metric_key=metric, metric_label=metric,
                output_path=out/(name+'.png'), scale_policy='vedere metadati della ricerca originale',
                elastic_only=all(float(r['gouge_damping'])==0 for r in rows))
    print(f'Mappe salvate in {out.resolve()}')


if __name__=='__main__': main()
