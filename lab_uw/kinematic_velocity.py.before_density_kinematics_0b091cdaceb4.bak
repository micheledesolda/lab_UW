"""Arrival-equivalent gouge speed; cm/us internally, m/s in saved results.

This diagnostic never changes FWI parameters, gradients or step limits.
The fixed positive lobe is a timing indicator, not a first-break estimator.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import numpy as np
from scipy.optimize import brentq
from .fd1d import finite_difference_1d_receiver_trace

GROOVES = {'groove_sb1': 'side_block_1', 'groove_cb1': 'central_block',
           'groove_cb2': 'central_block', 'groove_sb2': 'side_block_2'}
VARIABLE = ('gouge_1', 'gouge_2')
SPEED_ROOT_TOLERANCE_M_S = 1e-5
BOUNDARY_AWARE_KINEMATICS_V2 = True

_STATE_FIELDS=(('v','velocity_array'),('nu','damping_array'),('rho','density_array'),
               ('source','source_spatial_function'),('receiver','receiver_spatial_function'))

def _region_coverage(regions,size):
    """Allow shared interface nodes and explicit, fixed outer solver nodes.

    The forward uses closed intervals in legacy stacks and can leave its
    final numerical node outside a layer by floating-point roundoff.
    Interior holes and overlapping region interiors remain errors.
    """
    counts=np.zeros(size,dtype=int);interior=np.zeros(size,dtype=bool)
    for name,raw in regions.items():
        ii=np.asarray(raw)
        if ii.ndim!=1:raise ValueError(f'Material indices are not one-dimensional: {name}.')
        if not len(ii):continue
        if ii.dtype.kind not in 'iu' or np.any(ii<0) or np.any(ii>=size) or np.any(np.diff(ii)<=0):
            raise ValueError(f'Invalid, repeated or unordered material indices: {name}.')
        counts[ii]+=1;interior[ii[1:-1]]=True
    missing=np.flatnonzero(counts==0)
    holes=missing[(missing!=0)&(missing!=size-1)]
    if len(holes):raise ValueError(f'Material regions leave interior grid nodes uncovered: {holes[:12].tolist()}.')
    bad=np.flatnonzero((counts>1)&interior)
    if len(bad):raise ValueError(f'Material region interiors overlap at grid nodes: {bad[:12].tolist()}.')
    return counts,missing

def _boundary_values(state,indices):
    n=len(state['velocity_array'])
    return {('left' if i==0 else 'right'):
            {key:float(np.asarray(state[field])[i]) for key,field in _STATE_FIELDS}
            for i in indices if i in (0,n-1)}

def validate_arrival_branch(pairs):
    """Keep genuine branch reversals; ignore root-scale floating-point ties."""
    pairs=np.asarray(sorted(pairs),dtype=float)
    if len(pairs)<2:return
    dc=np.diff(pairs[:,0]);dt=np.diff(pairs[:,1])
    time_roundoff=64*np.finfo(float).eps*max(1.,float(np.max(np.abs(pairs[:,1]))))
    # Cached evaluations from successive Brent solves may differ by far less
    # than xtol in speed. Their arrival ordering is not numerically resolvable.
    bad=(dt>time_roundoff)|((dc>SPEED_ROOT_TOLERANCE_M_S)&(dt>=-time_roundoff))
    if np.any(bad):raise ValueError('Arrival calibration contains a nonmonotonic branch.')

def _hash(*values):
    h=hashlib.sha256()
    for value in values:
        if isinstance(value,np.ndarray):
            a=np.ascontiguousarray(value);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
        else:h.update(json.dumps(value,sort_keys=True,default=str).encode())
    return h.hexdigest()

def reference_from_state(state):
    """Capture the GLOBAL homogeneous reference, including fixed damping."""
    idx=state['idx_dict'];v=np.asarray(state['velocity_array']);a=np.asarray(state['damping_array'])
    _,missing=_region_coverage(idx,len(v))
    ii=np.concatenate([idx[k] for k in VARIABLE]);v0=float(np.mean(v[ii])) if len(ii) else np.nan
    if not np.isfinite(v0) or v0<=0 or not np.allclose(v[ii],v0,rtol=1e-10,atol=1e-13):
        raise ValueError('The global reference must have one common homogeneous gouge speed.')
    regions={}
    for name,indices in idx.items():
        indices=np.asarray(indices,dtype=int)
        regions[name]={key:np.asarray(state[field])[indices].copy() for key,field in
                      [('v','velocity_array'),('nu','damping_array'),('rho','density_array'),
                       ('source','source_spatial_function'),('receiver','receiver_spatial_function')]}
    for groove,block in GROOVES.items():
        cb=regions[block]['v']
        if not len(cb) or not np.allclose(cb,cb[0]):raise ValueError('Reference blocks must be homogeneous.')
        if np.isclose(cb[0],v0):raise ValueError('Cannot identify groove mixing when block and gouge speeds coincide.')
        f=(cb[0]**2-regions[groove]['v']**2)/(cb[0]**2-v0**2)
        if np.any(f < -1e-9) or np.any(f > 1+1e-9):raise ValueError('Invalid groove mixing fraction.')
        regions[groove]['fraction']=np.clip(f,0,1)
    result={'schema':1,'v0_m_s':v0*1e4,'regions':regions,
            'fixed_boundary_defaults':_boundary_values(state,missing),
            'source_time_hash':_hash(np.asarray(state['restart_runtime']['source_time_function'])),
            'elastic_path_time_us':float(np.sum(1/v)*state['restart_runtime']['grid']['dx']),
            'window_us':None}
    result['reference_hash']=_hash(v,a,np.asarray(state['density_array']),result['source_time_hash'])
    return result

def _simulation_state(sim):
    m=sim.velocity_model_handler
    return {'idx_dict':m.idx_dict,'velocity_array':m.velocity_array,'damping_array':m.damping_array,
            'density_array':m.density_array,'source_spatial_function':sim.source_handler.spatial_function,
            'receiver_spatial_function':sim.receiver_handler.spatial_function,
            'restart_runtime':{'source_time_function':sim.source_handler.time_function,
                               'grid':{'dx':sim.grid_handler.dx}}}

def ensure_reference(sim,reference=None):
    if reference is not None:sim._kinematic_reference=deepcopy(reference)
    if hasattr(sim,'_kinematic_reference'):return
    try:
        if getattr(sim,'geometry_type',None)!='dds':raise ValueError('Diagnostic currently defined for DDS geometry.')
        sim._kinematic_reference=reference_from_state(_simulation_state(sim))
    except ValueError as error:
        sim._kinematic_reference={'schema':1,'unavailable_reason':str(error)}

def _remap(values,count,name):
    values=np.asarray(values)
    if len(values)==count:return values.copy()
    if name not in VARIABLE or not len(values) or count==0:
        raise ValueError(f'Unsupported change in reference region {name}.')
    return np.interp(np.linspace(0,1,count),np.linspace(0,1,len(values)),values)

def arrival_time(time,waveform,window):
    t=np.asarray(time);y=np.asarray(waveform);ii=np.flatnonzero((t>=window[0])&(t<=window[1]))
    if len(ii)<3 or not np.all(np.isfinite(y[ii])):raise ValueError('Invalid arrival window or trace.')
    j=int(ii[np.argmax(y[ii])])
    if j in (ii[0],ii[-1]) or y[j]<=0:raise ValueError('Positive arrival lobe is absent or at the window boundary.')
    den=y[j-1]-2*y[j]+y[j+1]
    offset=.5*(y[j-1]-y[j+1])/den if den<0 else np.nan
    if not np.isfinite(offset) or abs(offset)>.5:raise ValueError('Ambiguous arrival interpolation.')
    return float(t[j]+offset*(t[1]-t[0]))

def _initial_window(sim,time,trace):
    mask=getattr(sim,'adaptive_signal_mask',None)
    if mask is None:mask=sim.misfit_interval
    mask=np.asarray(mask)
    if mask.dtype==bool:ii=np.flatnonzero(mask)
    else:ii=mask.astype(int)
    if len(ii)<3:raise ValueError('No direct-arrival window available for calibration.')
    low,high=np.asarray(sim.observed_time)[ii[[0,-1]]]
    selected=np.flatnonzero((time>=low)&(time<=high))
    peaks=selected[(selected>0)&(selected<len(trace)-1)]
    peaks=peaks[(trace[peaks]>trace[peaks-1])&(trace[peaks]>=trace[peaks+1])]
    peaks=peaks[trace[peaks]>=.5*np.max(trace[selected])]
    if not len(peaks):raise ValueError('No strong positive reference lobe in the direct-arrival window.')
    j=int(peaks[0]);left=np.flatnonzero(trace[:j]<=0);right=np.flatnonzero(trace[j+1:]<=0)
    if not len(left) or not len(right):raise ValueError('Reference lobe has no surrounding zero crossings.')
    # Freeze this isolated lobe for both the current profile and ALL reference speeds.
    half=max(time[j]-time[left[-1]],time[j+1+right[0]]-time[j])
    return (float(time[j]-1.05*half),float(time[j]+1.05*half))

class ArrivalFamily:
    def __init__(self,sim):
        self.sim=sim;self.ref=sim._kinematic_reference;m=sim.velocity_model_handler
        self.time=np.asarray(sim.sim_time_handler.simulation_time)
        if _hash(np.asarray(sim.source_handler.time_function))!=self.ref['source_time_hash']:
            raise ValueError('Temporal source changed from the homogeneous reference.')
        self.v=np.empty_like(m.velocity_array);self.nu=np.empty_like(m.damping_array)
        covered,missing=_region_coverage(m.idx_dict,len(self.v))
        assigned=np.zeros(len(self.v),dtype=bool)
        for name,ii in m.idx_dict.items():
            ii=np.asarray(ii,dtype=int);r=self.ref['regions'].get(name)
            if r is None:raise ValueError(f'Unknown material region {name}.')
            vv=_remap(r['v'],len(ii),name);aa=_remap(r['nu'],len(ii),name)
            common=assigned[ii]
            if (not np.allclose(self.v[ii[common]],vv[common],rtol=1e-10,atol=1e-12)
                    or not np.allclose(self.nu[ii[common]],aa[common],rtol=1e-10,atol=1e-12)):
                raise ValueError(f'Reference coefficients disagree at a shared interface of {name}.')
            self.v[ii]=vv;self.nu[ii]=aa;assigned[ii]=True
            for key,current in [('rho',m.density_array),('source',sim.source_handler.spatial_function),
                                ('receiver',sim.receiver_handler.spatial_function)]:
                expected=_remap(r[key],len(ii),name)
                if not np.allclose(expected,np.asarray(current)[ii],rtol=1e-10,atol=1e-12):
                    raise ValueError(f'{key} changed in region {name}; fixed reference is no longer valid.')
        if len(missing):
            if 'fixed_boundary_defaults' not in self.ref:
                # Old checkpoints did not store these coefficients. The
                # local optimizer updates material masks only; unassigned
                # outer nodes are fixed and are never removed by gouge shrink.
                self.ref['fixed_boundary_defaults']=_boundary_values(_simulation_state(sim),missing)
                self.ref['boundary_defaults_migration']='untouched outer nodes of legacy checkpoint'
            defaults=self.ref['fixed_boundary_defaults'];current=_simulation_state(sim)
            for i in missing:
                side='left' if i==0 else 'right';values=defaults.get(side)
                if values is None:raise ValueError(f'No fixed reference for unassigned {side} boundary node.')
                for key,field in _STATE_FIELDS:
                    if not np.isclose(values[key],np.asarray(current[field])[i],rtol=1e-10,atol=1e-12):
                        raise ValueError(f'{key} changed at the fixed {side} solver boundary.')
                self.v[i]=values['v'];self.nu[i]=values['nu']
        self.coverage_report={'shared_interface_nodes':np.flatnonzero(covered>1).tolist(),
                              'fixed_unassigned_outer_nodes':missing.tolist()}
        self.idx=m.idx_dict
        if not np.allclose(self.profile(self.ref['v0_m_s']),self.v,rtol=1e-10,atol=1e-12):
            raise ValueError('Homogeneous family does not reproduce its reference coefficients.')
        self.key=_hash('boundary-aware-v2',self.ref['reference_hash'],self.v,self.nu,
                       sim.grid_handler.spatial_axis,float(sim.grid_handler.dx),
                       self.time,m.density_array,sim.source_handler.spatial_function,
                       sim.receiver_handler.spatial_function,sim.source_handler.time_function,
                       [sim.receiver_quantity,sim.boundary_condition,sim.boundary_impedance,sim.source_injection],
                       {k:np.asarray(v).tolist() for k,v in self.idx.items()})
        allcache=getattr(sim,'_kinematic_cache',{})
        if self.key not in allcache:
            while len(allcache)>=4:allcache.pop(next(iter(allcache)))
            allcache[self.key]={'times':{},'window_us':None}
        self.entry=allcache[self.key];self.cache=self.entry['times']
        sim._kinematic_cache=allcache;self.forward_calls=0

    def profile(self,speed):
        v=self.v.copy();c=float(speed)/1e4
        for name in VARIABLE:v[self.idx[name]]=c
        for groove,block in GROOVES.items():
            f=self.ref['regions'][groove]['fraction'];cb=self.ref['regions'][block]['v'][0]
            v[self.idx[groove]]=np.sqrt((1-f)*cb**2+f*c**2)
        return v

    def trace(self,v,nu):
        s=self.sim;clock=s.sim_time_handler;self.forward_calls+=1
        return finite_difference_1d_receiver_trace(num_x=len(v),delta_x=s.grid_handler.dx,
            num_t=clock.num_t,delta_t=clock.dt,source_spatial_function=s.source_handler.spatial_function,
            source_time_function=s.source_handler.time_function,receiver_spatial_function=s.receiver_handler.spatial_function,
            receiver_quantity=s.receiver_quantity,velocity_model=v,damping_model=nu,
            density_model=s.velocity_model_handler.density_array,boundary_condition=s.boundary_condition,
            boundary_impedance=s.boundary_impedance,source_injection=s.source_injection)

    def evaluate(self,speed):
        speed=float(speed)
        if speed not in self.cache:
            y=self.trace(self.profile(speed),self.nu)
            if self.entry['window_us'] is None:
                if self.ref['window_us'] is None:
                    self.ref['window_us']=_initial_window(self.sim,self.time,y)
                # Track the same isolated lobe if layer thickness changes. This
                # shift only chooses a search window; the speed is still obtained
                # by exact homogeneous FD replay on the CURRENT geometry.
                shift=float(np.sum(1/self.v)*self.sim.grid_handler.dx)-self.ref['elastic_path_time_us']
                self.entry['window_us']=tuple(x+shift for x in self.ref['window_us'])
            self.cache[speed]=arrival_time(self.time,y,self.entry['window_us'])
        return self.cache[speed]

    def invert(self,target):
        v0=self.ref['v0_m_s'];t0=self.evaluate(v0)
        if abs(target-t0)<1e-10:return v0
        # A small local bracket usually needs only three or four new traces.
        half=max(10.,.01*v0)
        for _ in range(6):
            lo=max(.25*v0,v0-half);hi=min(2*v0,v0+half)
            tl,th=self.evaluate(lo),self.evaluate(hi)
            if not tl>t0>th:raise ValueError('Homogeneous arrival calibration is not monotonically decreasing.')
            if th<=target<=tl:
                speed=float(brentq(lambda c:self.evaluate(c)-target,lo,hi,xtol=SPEED_ROOT_TOLERANCE_M_S))
                # Check the selected branch at EVERY sampled calibration point.
                pairs=sorted(self.cache.items())
                validate_arrival_branch(pairs)
                return speed
            half*=2
        raise ValueError('Profile arrival is outside the calibrated homogeneous branch.')

def update_model_kinematics(sim,native_trace=None):
    ensure_reference(sim);ref=sim._kinematic_reference
    try:
        if 'unavailable_reason' in ref:raise ValueError(ref['unavailable_reason'])
        family=ArrivalFamily(sim);t0=family.evaluate(ref['v0_m_s'])
        if native_trace is None:
            m=sim.velocity_model_handler;native_trace=family.trace(m.velocity_array,m.damping_array)
        tp=arrival_time(family.time,native_trace,family.entry['window_us']);speed=family.invert(tp)
        result={'schema':1,'status':'ok','definition':'v_kin = inverse(T_hom)(T_profile)',
                'feature':'fixed_positive_lobe_quadratic_interpolation','time_grid':'native_solver',
                'reference_velocity_m_s':ref['v0_m_s'],'reference_arrival_us':t0,
                'arrival_us':tp,'arrival_change_ns':1000*(tp-t0),'velocity_m_s':speed,
                'arrival_window_us':tuple(family.entry['window_us']),'geometry_key':family.key,
                'homogeneous_replay_residual_ns':1000*(family.evaluate(speed)-tp),
                'calibration_forward_calls':family.forward_calls,
                'region_coverage':family.coverage_report,
                'arithmetic_gouge_velocity_m_s':float(np.mean(sim.velocity_model_handler.velocity_array[
                    np.concatenate([sim.velocity_model_handler.idx_dict[k] for k in VARIABLE])])*1e4)}
    except ValueError as error:
        result={'schema':1,'status':'unavailable','reason':str(error),'velocity_m_s':None,'arrival_us':None}
    sim.kinematic_velocity_m_s=result['velocity_m_s'];sim.kinematic_arrival_us=result['arrival_us']
    sim.kinematic_diagnostics=result
    print('Kinematic label: '+(f"{result['velocity_m_s']:.3f} m/s, arrival {result['arrival_us']:.9f} us"
                               if result['status']=='ok' else 'unavailable: '+result['reason']),flush=True)
    return result

def filename_label(sim):
    result=getattr(sim,'kinematic_diagnostics',{})
    if result.get('status')!='ok':return 'v_kin_unavailable'
    speed=str(round(result['velocity_m_s']));time=f"{result['arrival_us']:.6f}".replace('.',',')
    return f'v_kin_{speed}ms_t_arr_{time}us'

def result_fields(sim):
    return {name:deepcopy(getattr(sim,name,None)) for name in
            ('kinematic_velocity_m_s','kinematic_arrival_us','kinematic_diagnostics')}
