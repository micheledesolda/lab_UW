"""Bounded local rho(x) coordinates for the matched DDS FD/adjoint.

Density uses the same native g/cm^3 convention as the existing model. Only
pure gouge nodes vary; metal and effective groove densities remain fixed.
There is no prior or smoothing penalty: bounds restrict admissible models,
but do not establish that a recovered density profile is uniquely identified.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import numpy as np

REVISION = 5
DENSITY_DEFAULTS = {'invert_density': False, 'density_bounds_g_cm3': None}


def configure_policy(policy, params):
    p = deepcopy(policy)
    if not isinstance(p.get('invert_density', False), (bool, np.bool_)):
        raise ValueError('local_optimizer.invert_density must be boolean')
    p['invert_density'] = bool(p.get('invert_density', False))
    if p['invert_density'] and not p['enabled']:
        raise ValueError('Density inversion requires local_optimizer.enabled=True')
    bounds = p.get('density_bounds_g_cm3')
    if p['invert_density'] and bounds is None:
        values = np.asarray(params.get('impedance_density_values_g_cm3',
                            params.get('gouge_density_values_g_cm3', [1.4, 3.])), dtype=float)
        bounds = [float(np.min(values)), float(np.max(values))]
    if bounds is not None:
        a = np.asarray(bounds, dtype=float)
        if a.shape != (2,) or not np.all(np.isfinite(a)) or not 0 < a[0] < a[1]:
            raise ValueError('density_bounds_g_cm3 must be [positive lower, larger upper]')
        p['density_bounds_g_cm3'] = a.tolist()
    return p


def canonical_policy(policy):
    p = deepcopy(policy)
    if not p.get('invert_density', False):
        p.pop('invert_density', None)
        p.pop('density_bounds_g_cm3', None)
    return p


def policy_changes(previous, current, defaults):
    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise ValueError('Cannot migrate an unknown material optimizer policy.')
    old = {**DENSITY_DEFAULTS, **previous}
    new = {**DENSITY_DEFAULTS, **current}
    old.setdefault('parameterization', 'velocity_damping')
    new.setdefault('parameterization', 'velocity_damping')
    if set(old) != set(defaults) or set(new) != set(defaults):
        raise ValueError('Cannot migrate an unknown material optimizer policy.')
    if not old['enabled'] or not new['enabled']:
        raise ValueError('Changing local_optimizer.enabled requires a separate result sequence.')
    if old['invert_density'] != new['invert_density']:
        raise ValueError('Density inversion mode changed; use a new output folder.')
    if old['invert_density'] and (old['density_bounds_g_cm3'] != new['density_bounds_g_cm3']
                                  or old['parameterization'] != new['parameterization']):
        raise ValueError('Density bounds/coordinates changed; use a new output folder.')
    return {k: {'old': old[k], 'new': new[k]} for k in defaults if old[k] != new[k]}


def pure_gouge_indices(idx):
    wanted = np.unique(np.concatenate([np.asarray(idx[k], dtype=int) for k in ('gouge_1', 'gouge_2')]))
    others = [np.asarray(v, dtype=int) for k, v in idx.items() if k not in ('gouge_1', 'gouge_2')]
    # Closed region conventions can share interface nodes. Those belong to
    # fixed mixed/metal regions as well and must not become free gouge rho.
    return np.setdiff1d(wanted, np.concatenate(others) if others else [], assume_unique=False)


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        b = np.ascontiguousarray(a)
        h.update(str((b.shape, b.dtype.str)).encode()); h.update(b.tobytes())
    return h.hexdigest()


def prepare(sim, policy):
    """Persist mode for zero-iteration reused records and compact restarts."""
    sim.local_density_inverted = bool(policy.get('invert_density', False))
    sim.local_density_bounds_g_cm3 = deepcopy(policy.get('density_bounds_g_cm3'))
    if sim.local_density_inverted:
        sim.optimization_policy = deepcopy(policy)


def fit_density_trace_scale(observed_waveform, synthetic_waveform, misfit_interval, *, taper_fraction=0.):
    """Fit the scalar against the SAME weighted L2 used by the adjoint."""
    from .forward_modeling import fit_trace_scale, _waveform_interval_indices
    if not taper_fraction:
        return fit_trace_scale(observed_waveform, synthetic_waveform, misfit_interval)
    from scipy.signal.windows import tukey
    ii=_waveform_interval_indices(misfit_interval,len(observed_waveform))
    weights=tukey(len(ii),alpha=float(taper_fraction))
    return fit_trace_scale(weights*np.asarray(observed_waveform)[ii],
                           weights*np.asarray(synthetic_waveform)[ii],slice(None))


def density_plot(sim, output_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    x = np.asarray(sim.grid_handler.spatial_axis) * 10.
    ax.plot(x, sim.velocity_model_handler.density_array, color='tab:purple', lw=1.5)
    for key in ('gouge_1', 'gouge_2'):
        ii = sim.velocity_model_handler.idx_dict[key]
        if len(ii): ax.axvspan(x[ii[0]], x[ii[-1]], color='tab:orange', alpha=.13)
    ax.set(xlabel='Position [mm]', ylabel=r'Density [g/cm$^3$]',
           title='Density profile: orange regions are the gouge layers')
    ax.grid(alpha=.2)
    fig.savefig(str(output_path), dpi=180); plt.close(fig)


def annotate_result(row, sim):
    active = bool(getattr(sim, 'local_density_inverted', False))
    row['local_material_density_fixed'] = not active
    row['local_density_inverted'] = active
    if active:
        m = sim.velocity_model_handler
        row['density_profile'] = m.density_array.copy()
        row['local_density_bounds_g_cm3'] = deepcopy(sim.local_density_bounds_g_cm3)
        for k in ('gouge_1', 'gouge_2'):
            row[k+'_density_g_cm3'] = m.density_array[m.idx_dict[k]].copy()
        row['density_inversion_interpretation'] = 'Admissible fitted profile; uniqueness is not implied.'
    return row


def make_density_optimizer(base_class, policy, velocity, damping, regions, v_lower, v_upper,
                           active_velocity, active_damping, *, density_model, density_regions,
                           resume=None):
    """Reuse the established two-loop L-BFGS and plateau observer.

    Only packing, projection, step evaluation and density provenance differ.
    Every trial enforces velocity limits jointly with the current trial rho,
    including when the optimizer coordinates are (M, eta, rho).
    """
    return DensityOptimizer(base_class, policy, velocity, damping, regions, v_lower, v_upper,
                            active_velocity, active_damping, density_model=density_model,
                            density_regions=density_regions, resume=resume)


class DensityOptimizer:
    def __init__(self, base_class, policy, velocity, damping, regions, v_lower, v_upper,
                 active_velocity, active_damping, *, density_model, density_regions, resume=None):
        self.base_class = base_class
        self.policy = deepcopy(policy)
        self.mode = policy['parameterization']
        self.ii = np.unique(np.asarray(regions, dtype=int))
        self.rr = np.unique(np.asarray(density_regions, dtype=int))
        if not len(self.rr) or not np.all(np.isin(self.rr, self.ii)):
            raise ValueError('Density inversion requires resolved pure gouge nodes within material regions.')
        self.active = (bool(active_velocity), bool(active_damping), True)
        self.rho = np.asarray(density_model, dtype=float).copy()
        self.frozen_v, self.frozen_a = velocity.copy(), damping.copy()
        self.v_lower, self.v_upper = v_lower.copy(), v_upper.copy()
        self.rho_bounds = np.asarray(policy['density_bounds_g_cm3'], dtype=float)
        lo, hi = self.rho_bounds
        if np.any(self.rho[self.rr] < lo-1.e-12) or np.any(self.rho[self.rr] > hi+1.e-12):
            raise ValueError('Initial gouge density is outside density_bounds_g_cm3; widen the bounds.')
        n, nr = len(self.ii), len(self.rr)
        a, b = self._coefficients(velocity, damping, self.rho)
        self.scales = np.r_[np.full(n, max(float(np.mean(a[self.ii])), 1.e-8)),
                            np.full(n, max(float(np.mean(b[self.ii])), 1.e-8)),
                            np.full(nr, float(np.mean(self.rho[self.rr])))]
        self.pairs=[]; self.previous_x=None; self.previous_g=None; self.weak=0; self.accepted=0
        self.last_iteration=0; self.reason=None; self.searches=[]; self.total_receiver_calls=0
        self.last_alpha=None; self.last_info={}; self.geometry_size=n; self.migration=None
        self.gradient_rho = None
        self.geometry_digest = _digest(self.ii, self.rr, self.rho[np.setdiff1d(np.arange(len(self.rho)),self.rr)])
        self.last_x = self._vector(velocity, damping)
        if resume:
            if resume.get('optimizer_revision') != REVISION:
                raise ValueError('Cannot reuse fixed-density optimizer memory; start density inversion in a new folder.')
            from .material_optimizer import policy_changes as changes
            changed = changes(resume['policy'], policy)
            if resume.get('reason') is not None:
                raise ValueError('Cannot resume an already converged density optimizer.')
            if resume.get('density_geometry_digest') != self.geometry_digest or tuple(resume['active']) != self.active:
                raise ValueError('Pending density optimizer geometry/active blocks changed.')
            # Same-wave restart retains the original bounds and scaling.
            for k in ('scales','pairs','previous_x','previous_g','weak','accepted','last_iteration',
                      'searches','total_receiver_calls','last_alpha','v_lower','v_upper','last_x'):
                setattr(self,k,deepcopy(resume[k]))
            if not np.allclose(self._vector(velocity,damping),self.last_x,rtol=2.e-13,atol=1.e-14):
                raise ValueError('Pending density optimizer does not match the accepted material profiles.')
            if changed:
                self.pairs=self.pairs[-policy['memory']:]; self.weak=0
                self.migration={'type':'numerical_policy_change','changes':changed,'curvature_retained':True}
        self.iteration_offset=self.last_iteration
        self.names = ('M','eta','rho') if self.mode=='modulus_viscosity' else ('velocity','damping','rho')
        print(f'Local density inversion: {self.names}; gouge rho in [{lo:g}, {hi:g}] g/cm^3; metal/mixed densities fixed.',flush=True)

    def _coefficients(self,v,a,r):
        return (r*v*v,r*a) if self.mode=='modulus_viscosity' else (v,a)

    def _vector(self,v,a):
        first,second=self._coefficients(v,a,self.rho)
        return np.r_[first[self.ii],second[self.ii],self.rho[self.rr]]/self.scales

    def _project(self,x):
        raw=x*self.scales; n=len(self.ii)
        r=self.rho.copy(); r[self.rr]=np.clip(raw[2*n:],*self.rho_bounds)
        first=raw[:n]; second=np.maximum(raw[n:2*n],0.)
        if self.mode=='modulus_viscosity':
            lower=r[self.ii]*self.v_lower[self.ii]**2
            upper=r[self.ii]*self.v_upper[self.ii]**2
            first=np.clip(first,lower,upper)
            if not self.active[0]: first=r[self.ii]*self.frozen_v[self.ii]**2
            if not self.active[1]: second=r[self.ii]*self.frozen_a[self.ii]
        else:
            first=np.clip(first,self.v_lower[self.ii],self.v_upper[self.ii])
            if not self.active[0]: first=self.frozen_v[self.ii]
            if not self.active[1]: second=self.frozen_a[self.ii]
        return np.r_[first,second,r[self.rr]]/self.scales

    def _project_direction(self,x,p):
        return self._project(x+p)-x

    def _decode(self,x,v,a):
        n=len(self.ii);raw=self._project(x)*self.scales
        r=self.rho.copy();r[self.rr]=raw[2*n:]
        vv,aa=v.copy(),a.copy()
        if self.mode=='modulus_viscosity':
            vv[self.ii]=np.sqrt(raw[:n]/r[self.ii]);aa[self.ii]=raw[n:2*n]/r[self.ii]
        else: vv[self.ii],aa[self.ii]=raw[:n],raw[n:2*n]
        if not self.active[0]:vv[self.ii]=self.frozen_v[self.ii]
        if not self.active[1]:aa[self.ii]=self.frozen_a[self.ii]
        return vv,aa,r

    def set_density_gradient(self,density,gradient):
        self.rho=np.asarray(density,dtype=float).copy()
        self.gradient_rho=np.asarray(gradient,dtype=float).copy()

    def _gradient(self,v,a,gv,ga):
        gr=self.gradient_rho.copy()
        if self.mode=='modulus_viscosity':
            gm=gv/(2*self.rho*v);ge=ga/self.rho
            if self.active[0]:gr-=v*gv/(2*self.rho)
            if self.active[1]:gr-=a*ga/self.rho
        else:gm,ge=gv.copy(),ga.copy()
        if not self.active[0]:gm=np.zeros_like(gm)
        if not self.active[1]:ge=np.zeros_like(ge)
        return np.r_[gm[self.ii],ge[self.ii],gr[self.rr]]*self.scales

    def _direction(self,x,g):
        return self.base_class._direction(self,x,g)

    def search(self,v,a,gradient_v,gradient_a,current_misfit,evaluate):
        x=self._vector(v,a);g=self._gradient(v,a,gradient_v,gradient_a)
        if not np.all(np.isfinite(g)):raise RuntimeError('Nonfinite three-parameter material gradient.')
        pg=-self._project_direction(x,-g)
        useful=max(self.policy['minimum_gain_absolute'],self.policy['minimum_gain_relative']*float(current_misfit))
        self.last_info={'receiver_calls':0,'trials':[],'useful_gain':useful,'rescue_checked':False,
                        'projected_gradient_inf':float(np.max(np.abs(pg))),'method':'projected_lbfgs',
                        'parameterization':self.mode,'optimization_variables':list(self.names)}
        base={'velocity':v,'damping':a,'density':self.rho.copy(),'misfit':float(current_misfit),
              'selected':'current','alpha':None}
        if self.last_info['projected_gradient_inf']<=self.policy['projected_gradient_tolerance']:
            self.last_info.update(gain=0.,stationary=True,rescue_checked=True);self.last_x=x.copy();return base
        cache={};n=len(self.ii)
        def trial(p,alpha,kind):
            xx=self._project(x+alpha*p);delta=xx-x;key=xx.tobytes()
            if key in cache:return cache[key]
            if not np.any(delta):return None
            vv,aa,rr=self._decode(xx,v,a)
            try:_,_,obj=evaluate(vv,aa,rr);j=float(obj['total'])
            except ValueError as error:
                if 'unstable explicit time step' not in str(error):raise
                j=np.inf
            self.last_info['receiver_calls']+=1
            self.last_info['trials'].append({'direction':kind,'alpha':float(alpha),'misfit':j if np.isfinite(j) else None})
            valid=np.isfinite(j) and j<current_misfit and j<=current_misfit+self.policy['armijo']*float(g@delta)
            result={'velocity':vv,'damping':aa,'density':rr,'misfit':j,'selected':kind,'alpha':float(alpha),'x':xx} if valid else None
            cache[key]=result;return result
        def line(p,kind,start=None):
            p=self._project_direction(x,p)
            if not np.any(p) or float(g@p)>=0:return None
            alpha=float(start if start is not None else self.policy['initial_fraction']/max(np.max(np.abs(p)),1.e-30))
            best=None
            for _ in range(self.policy['backtracks']):
                best=trial(p,alpha,kind)
                if best is not None:break
                alpha*=.5
            if best is None:return None
            for _ in range(self.policy['expansions']):
                larger=trial(p,alpha*2,kind)
                if larger is None or larger['misfit']>=best['misfit']:break
                best=larger;alpha*=2
            return best
        chosen=line(self._direction(x,g),'lbfgs',1. if self.pairs else None)
        if chosen is not None:base=chosen
        if float(current_misfit)-base['misfit']<useful:
            self.last_info['rescue_checked']=True
            directions=[(-g,'gradient_joint')]
            for block,name in enumerate(self.names):
                if not self.active[block]:continue
                p=np.zeros_like(g);sl=slice(block*n,(block+1)*n) if block<2 else slice(2*n,None)
                p[sl]=-g[sl];directions.append((p,'gradient_'+name))
            for p,kind in directions:
                candidate=line(p,kind)
                if candidate is not None and candidate['misfit']<base['misfit']:base=candidate
            if base['selected']!='lbfgs':self.pairs=[]
        self.total_receiver_calls+=self.last_info['receiver_calls']
        self.last_info.update(gain=float(current_misfit)-base['misfit'],selected=base['selected'],alpha=base['alpha'])
        self.last_alpha=base['alpha'];self.last_x=base.pop('x',x).copy();self.rho=base['density'].copy()
        return base

    def observe(self,**kwargs):
        return self.base_class.observe(self,**kwargs)

    def finish(self,*args,**kwargs):
        result=self.base_class.finish(self,*args,**kwargs)
        result.update(optimizer_revision=REVISION,parameterization=self.mode,optimization_variables=list(self.names),
                      density_inverted=True,density_bounds_g_cm3=self.rho_bounds.tolist())
        state=result['continuation']
        state.update(optimizer_revision=REVISION,density_geometry_digest=self.geometry_digest,
                     v_lower=self.v_lower.copy(),v_upper=self.v_upper.copy(),last_x=self.last_x.copy(),
                     parameterization=self.mode)
        # The base finish helper records lower/upper for its fixed boxes.
        # Here projection is coupled to rho, so retain physical limits above.
        return result

    @property
    def lower(self):
        return np.r_[self.v_lower[self.ii],np.zeros(len(self.ii)),np.full(len(self.rr),self.rho_bounds[0])]

    @property
    def upper(self):
        return np.r_[self.v_upper[self.ii],np.full(len(self.ii),np.inf),np.full(len(self.rr),self.rho_bounds[1])]
