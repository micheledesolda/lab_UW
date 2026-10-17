"""Projected limited-memory BFGS directions and receiver-only step searches.

Numerical convergence is independent of the observed noise energy. Trials
share the exact objective, bounds and forward operator of the local solver.
This is a safeguarded projected optimizer, not the L-BFGS-B algorithm.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib,json
import numpy as np

DEFAULTS={'enabled':True,'memory':8,'initial_fraction':.02,'armijo':1e-4,
          'backtracks':18,'expansions':6,'minimum_gain_absolute':1e-6,
          'minimum_gain_relative':1e-5,'weak_cycles':2,'projected_gradient_tolerance':1e-9}

def settings(params):
    p={**DEFAULTS,**params.get('local_optimizer',{})}
    for k in ('memory','backtracks','expansions','weak_cycles'):
        if isinstance(p[k],bool) or int(p[k])!=p[k] or p[k]<1:raise ValueError(f'local_optimizer.{k} must be positive integer')
    for k in ('initial_fraction','minimum_gain_absolute','minimum_gain_relative','projected_gradient_tolerance'):
        if not np.isfinite(p[k]) or p[k]<=0:raise ValueError(f'local_optimizer.{k} must be positive')
    if not 0<p['armijo']<.5:raise ValueError('Invalid Armijo coefficient')
    return p

def policy_digest(policy):
    return hashlib.sha256(json.dumps(policy,sort_keys=True).encode()).hexdigest()

def policy_changes(previous,current):
    """Only numerical controls of this same optimizer may change on resume."""
    if not isinstance(previous,dict) or set(previous)!=set(DEFAULTS) or set(current)!=set(DEFAULTS):
        raise ValueError('Cannot migrate an unknown material optimizer policy.')
    if not previous['enabled'] or not current['enabled']:
        raise ValueError('Changing local_optimizer.enabled requires a separate result sequence.')
    return {key:{'old':previous[key],'new':current[key]}
            for key in DEFAULTS if previous[key]!=current[key]}

class MaterialOptimizer:
    def __init__(self,policy,velocity,damping,regions,v_lower,v_upper,active_velocity,active_damping,resume=None):
        self.policy=deepcopy(policy);self.ii=np.asarray(regions,dtype=int)
        self.active=(bool(active_velocity),bool(active_damping));n=len(self.ii)
        self.scales=np.r_[np.full(n,max(float(np.mean(velocity[self.ii])),1e-8)),
                          np.full(n,max(float(np.mean(damping[self.ii])),1e-8))]
        self.lower=np.r_[v_lower[self.ii],np.zeros(n)]/self.scales
        self.upper=np.r_[v_upper[self.ii],np.full(n,np.inf)]/self.scales
        self.pairs=[];self.previous_x=None;self.previous_g=None;self.weak=0;self.accepted=0
        self.last_iteration=0;self.reason=None;self.searches=[];self.total_receiver_calls=0
        self.last_alpha=None;self.last_info={};self.geometry_size=n
        self.migration=None
        if resume:
            self.last_iteration=int(resume.get('last_iteration',0));self.accepted=int(resume.get('accepted',0))
            if resume.get('optimizer_revision')==3:
                changes=policy_changes(resume['policy'],self.policy)
                if resume.get('reason') is not None:raise ValueError('Cannot resume an optimizer that already converged.')
                if resume.get('geometry_size')!=n:raise ValueError('Pending optimizer geometry changed.')
                if tuple(resume['active'])!=self.active:raise ValueError('Pending optimizer active parameters changed.')
                for k in ('scales','pairs','previous_x','previous_g','weak','searches','total_receiver_calls','last_alpha','lower','upper'):
                    setattr(self,k,deepcopy(resume[k]))
                self.active=tuple(resume['active'])
                if changes:
                    # The objective, accepted model, scales and bounds are unchanged.
                    # Curvature remains valid; old weak-cycle counts do not certify
                    # convergence under the new tolerances/search controls.
                    self.pairs=self.pairs[-int(self.policy['memory']):]
                    self.weak=0
                    self.migration={'type':'numerical_policy_change','changes':changes,
                                    'accepted_model_retained':True,'curvature_retained':True,
                                    'weak_cycles_reset':True}
                    print('Optimizer numerical policy updated: accepted model, bounds and curvature retained; '
                          'weak cycles reset. '+json.dumps(changes,sort_keys=True),flush=True)
            else:
                self.migration='legacy_noise_monitor_to_material_optimizer: accepted model retained, curvature memory initialized'
                print('Optimizer migration: continuing the SAME acquisition and accepted model; legacy noise plateau rule replaced.',flush=True)
        self.iteration_offset=self.last_iteration
        print('Material optimizer: projected L-BFGS + searched steps; '
              f"useful gain=max({self.policy['minimum_gain_absolute']:.3g}, "
              f"{self.policy['minimum_gain_relative']:.3g}*J); noise is not a convergence test.",flush=True)

    def _vector(self,v,a):return np.r_[v[self.ii],a[self.ii]]/self.scales

    def _project_direction(self,x,p):
        p=p.copy();n=len(self.ii)
        if not self.active[0]:p[:n]=0
        if not self.active[1]:p[n:]=0
        p[(x<=self.lower+1e-14)&(p<0)]=0
        p[(x>=self.upper-1e-14)&(p>0)]=0
        return p

    def _direction(self,x,g):
        if self.previous_x is not None:
            s=x-self.previous_x;y=g-self.previous_g;sy=float(s@y)
            if sy>max(1e-14,1e-10*np.linalg.norm(s)*np.linalg.norm(y)):
                self.pairs.append((s.copy(),y.copy(),1/sy));self.pairs=self.pairs[-self.policy['memory']:]
        self.previous_x=x.copy();self.previous_g=g.copy()
        q=g.copy();alphas=[]
        for s,y,rho in reversed(self.pairs):
            alpha=rho*float(s@q);alphas.append(alpha);q-=alpha*y
        if self.pairs:
            s,y,rho=self.pairs[-1];q*=float(s@y)/float(y@y)
        for (s,y,rho),alpha in zip(self.pairs,reversed(alphas)):
            q+=s*(alpha-rho*float(y@q))
        p=self._project_direction(x,-q)
        if not np.all(np.isfinite(p)) or float(g@p)>=-1e-16:
            p=self._project_direction(x,-g);self.pairs=[]
        return p

    def search(self,v,a,gradient_v,gradient_a,current_misfit,evaluate):
        x=self._vector(v,a);g=np.r_[gradient_v[self.ii],gradient_a[self.ii]]*self.scales
        if not np.all(np.isfinite(g)):raise RuntimeError('Nonfinite gradient; optimizer cannot certify convergence.')
        pg=-self._project_direction(x,-g)
        useful=max(self.policy['minimum_gain_absolute'],self.policy['minimum_gain_relative']*float(current_misfit))
        self.last_info={'receiver_calls':0,'trials':[],'useful_gain':useful,'rescue_checked':False,
                        'projected_gradient_inf':float(np.max(np.abs(pg))), 'method':'projected_lbfgs'}
        base={'velocity':v,'damping':a,'misfit':float(current_misfit),'selected':'current','alpha':None}
        if self.last_info['projected_gradient_inf']<=self.policy['projected_gradient_tolerance']:
            self.last_info.update(gain=0.,stationary=True,rescue_checked=True)
            return base
        cache={};n=len(self.ii)
        def trial(p,alpha,kind):
            xx=np.clip(x+alpha*p,self.lower,self.upper);delta=xx-x
            key=xx.tobytes()
            if key in cache:return cache[key]
            if not np.any(delta):return None
            vv=v.copy();aa=a.copy();vv[self.ii]=xx[:n]*self.scales[:n];aa[self.ii]=xx[n:]*self.scales[n:]
            try:
                _,_,objective=evaluate(vv,aa);j=float(objective['total'])
            except ValueError as error:
                if 'unstable explicit time step' not in str(error):raise
                j=np.inf
            self.last_info['receiver_calls']+=1
            self.last_info['trials'].append({'direction':kind,'alpha':float(alpha),'misfit':j if np.isfinite(j) else None})
            valid=np.isfinite(j) and j<float(current_misfit) and j<=current_misfit+self.policy['armijo']*float(g@delta)
            r={'velocity':vv,'damping':aa,'misfit':j,'selected':kind,'alpha':float(alpha)} if valid else None
            cache[key]=r;return r
        def line(p,kind,start=None):
            p=self._project_direction(x,p)
            if not np.any(p) or float(g@p)>=0:return None
            # Newton-like directions use alpha=1. Fresh gradient directions
            # start at a dimensionless physical fraction, independent of
            # previously collapsed normalized-gradient step limits.
            alpha=float(start if start is not None else self.policy['initial_fraction']/max(np.max(np.abs(p)),1e-30))
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
        direction=self._direction(x,g)
        chosen=line(direction,'lbfgs',1. if self.pairs else None)
        if chosen is not None:base=chosen
        gain=float(current_misfit)-base['misfit']
        if gain<useful:
            # A tiny accepted increment does not end the search. Check a
            # fresh joint gradient and both parameter blocks on THIS model.
            self.last_info['rescue_checked']=True
            directions=[(-g,'gradient_joint')]
            if self.active[0] and self.active[1]:
                pv=-g.copy();pv[n:]=0;pa=-g.copy();pa[:n]=0
                directions.extend([(pv,'gradient_velocity'),(pa,'gradient_damping')])
            for p,kind in directions:
                candidate=line(p,kind)
                if candidate is not None and candidate['misfit']<base['misfit']:base=candidate
            if base['selected']!='lbfgs':self.pairs=[]
        self.total_receiver_calls+=self.last_info['receiver_calls']
        self.last_info.update(gain=float(current_misfit)-base['misfit'],selected=base['selected'],alpha=base['alpha'])
        self.last_alpha=base['alpha']
        return base

    def observe(self,*,accepted,best_misfit,iteration):
        self.last_iteration=int(iteration)
        if accepted:self.accepted+=1
        info=deepcopy(self.last_info);info.update(iteration=int(iteration),best_misfit=float(best_misfit),accepted=bool(accepted))
        self.searches.append(info)
        # Keep bounded diagnostics in compact checkpoints. The full per-cycle
        # report is also written into the local iteration history.
        self.searches=self.searches[-20:]
        self.weak=self.weak+1 if info.get('rescue_checked') and info['gain']<info['useful_gain'] else 0
        print(f"    Optimizer: gain={info['gain']:.6g}, useful={info['useful_gain']:.6g}, "
              f"direction={info.get('selected','stationary')}, receiver trials={info['receiver_calls']}, "
              f"rescue={info['rescue_checked']}, weak={self.weak}/{self.policy['weak_cycles']}",flush=True)
        if info.get('stationary'):
            self.reason='projected_gradient_converged';return True
        if self.weak>=self.policy['weak_cycles']:
            self.reason='optimizer_plateau_verified';return True
        return False

    def finish(self,fallback,dc,da,ct,at,noise_level=None):
        reason=self.reason or fallback
        if not any(self.active):reason='no_active_parameters'
        continuation={k:deepcopy(getattr(self,k)) for k in
                      ('policy','scales','pairs','previous_x','previous_g','weak','accepted','last_iteration',
                       'reason','searches','total_receiver_calls','last_alpha','geometry_size','active','lower','upper')}
        continuation['optimizer_revision']=3
        return {'reason':reason,'converged':reason in ('optimizer_plateau_verified','projected_gradient_converged','no_active_parameters'),
                'optimizer_revision':3,'initial_noise_level_single':noise_level,'noise_level_single':noise_level,
                'noise_target':None,'minimum_block_gain':None,'block_accepted_updates':None,'patience_blocks':None,
                'accepted_updates':self.accepted,'receiver_trial_calls':self.total_receiver_calls,
                'last_search':deepcopy(self.last_info),'migration':self.migration,
                'interpretation':'Numerical convergence after searched directions; not a global-minimum or noise-significance certificate.',
                'continuation':continuation,
                'step_limits':{'dc_max_start':max(dc,ct*(1+1e-6),1e-30) if self.active[0] else 0.,
                               'da_max_start':max(da,at*(1+1e-6),1e-30) if self.active[1] else 0.,
                               'dc_threshold':ct,'da_threshold':at}}

def make_optimizer(policy,*args,resume=None,**kwargs):
    if not policy or not policy.get('enabled'):return None
    return MaterialOptimizer(policy,*args,resume=resume,**kwargs)

def reuse_material_model(sim,*,scale,gamma,taper,strategy,dc,da,objective):
    """One receiver-only evaluation, no optimization and no stored wavefield."""
    from .fd1d import finite_difference_1d_receiver_trace
    from .kinematic_velocity import update_model_kinematics
    if scale is None:scale=getattr(sim,'reference_trace_scale',getattr(sim,'trace_scale',None))
    if scale is None or not np.isfinite(scale) or scale<=0:raise ValueError('fixed_trace_scale must be positive and finite')
    m=sim.velocity_model_handler;c=sim.sim_time_handler
    raw=finite_difference_1d_receiver_trace(num_x=len(m.velocity_array),delta_x=sim.grid_handler.dx,
        num_t=c.num_t,delta_t=c.dt,source_spatial_function=sim.source_handler.spatial_function,
        source_time_function=sim.source_handler.time_function,receiver_spatial_function=sim.receiver_handler.spatial_function,
        receiver_quantity=sim.receiver_quantity,velocity_model=m.velocity_array,damping_model=m.damping_array,
        density_model=m.density_array,boundary_condition=sim.boundary_condition,boundary_impedance=sim.boundary_impedance,
        source_injection=sim.source_injection)
    sim.synthetic_waveform=np.interp(sim.observed_time,c.simulation_time,raw)*float(scale)
    sim.objective_components=objective(sim.observed_waveform,sim.synthetic_waveform,sim.misfit_interval,
                                      spectral_gamma=gamma,taper_fraction=taper)
    sim.misfit=float(sim.objective_components['total']);sim.trace_scale=sim.reference_trace_scale=float(scale)
    sim.amplitude_scale_mode='fixed';sim.spectral_gamma=float(gamma);sim.objective_taper_fraction=float(taper)
    sim.local_update_strategy=strategy;sim.local_stop_diagnostics=None;sim.wavefield_forward=None
    sim.local_inversion_history=[{'iteration':0,'trial_misfit':sim.misfit,'best_misfit':sim.misfit,'accepted':True,
        'update_strategy':strategy,'parameter_block':'initial','velocity_only_misfit':None,'damping_only_misfit':None,
        'joint_misfit':None,'cross_talk_interaction':None,'velocity_step_limit':float(dc),'damping_step_limit':float(da)}]
    gouge=np.concatenate([m.idx_dict['gouge_1'],m.idx_dict['gouge_2']])
    sim.average_gouge_velocity=float(np.mean(m.velocity_array[gouge]));sim.average_gouge_damping=float(np.mean(m.damping_array[gouge]))
    update_model_kinematics(sim,native_trace=raw)
    print('Model reused: receiver-only evaluation, zero gradient updates.',flush=True)
