"""Continuous teeth provider compatible with the existing bank forward.

Parameter evaluation uses the affine elastic operator. Optional cubic sampling
is in frequency only. Teeth remain lossless; FD-unit damping applies to bulk.
"""
from collections import OrderedDict
from pathlib import Path
from itertools import product
import hashlib
import numpy as np
from .teeth_bank import P2MatrixBank,ETA
from .teeth_formula import ContinuousTeeth,FullTeeth
from .teeth_formula.continuous_teeth import gouge

DEFAULT_MODELS=Path(__file__).resolve().parent/'teeth_formula'/'models'

def _axis(options,name,default=None):
    if name not in options and default is None:raise ValueError(f'Set teeth_continuous[{name}]')
    values=np.asarray(options.get(name,default),float)
    if values.ndim!=1 or not len(values) or not np.all(np.isfinite(values)) or len(np.unique(values))!=len(values):
        raise ValueError(f'{name} must be a nonempty finite vector without duplicates')
    return values

def make_material(rho,cs,nu,damping=0.,mode=1,index=0):
    row=gouge(float(rho),float(cs),float(nu));damping=float(damping)
    if not np.isfinite(damping) or damping<0 or mode not in (0,1):raise ValueError('Invalid damping or polarization')
    velocity=row['cp' if mode==0 else 'cs']/10
    relaxation=damping/velocity**2
    row.update(index=int(index),G_GPa=row['rho']*row['cs']**2,
        K_GPa=row['rho']*(row['cp']**2-4*row['cs']**2/3),damping_cm2_us=damping,
        bulk_relaxation_time_us=relaxation,
        nu_s_cm2_us=relaxation*(row['cs']/10)**2,
        nu_p_cm2_us=relaxation*(row['cp']/10)**2)
    return row

def candidate_materials(options,mode):
    rows=[]
    if 'materials' in options:
        for p in options['materials']:
            row=make_material(p['rho'],p['cs'],p['nu'],p.get('damping_cm2_us',0.),mode,len(rows))
            if 'cp' in p and not np.isclose(p['cp'],row['cp']):raise ValueError('Material cp is inconsistent with cs and Poisson')
            rows.append(row)
    else:
        densities=_axis(options,'density_values_g_cm3')
        velocities=_axis(options,'velocity_values_cm_us')
        poisson=_axis(options,'poisson_values',[.25])
        damping=_axis(options,'bulk_damping_values_cm2_us',[0.])
        for rho,velocity,nu,loss in product(densities,velocities,poisson,damping):
            ratio=np.sqrt(2*(1-nu)/(1-2*nu)) if .15-1e-12<=nu<=.45+1e-12 else np.nan
            cs=10*velocity/(ratio if mode==0 else 1.)
            try:rows.append(make_material(rho,cs,nu,loss,mode,len(rows)))
            except ValueError as exc:
                raise ValueError(f'Candidate rho={rho:g}, velocity={velocity:g} cm/us, Poisson={nu:g}: {exc}') from exc
    keys=[(p['rho'],p['cs'],p['nu'],p['damping_cm2_us']) for p in rows]
    if not rows or len(set(keys))!=len(keys):raise ValueError('Continuous candidates must be nonempty and unique')
    return rows

class ContinuousMatrixProvider(P2MatrixBank):
    def __init__(self,options,rows,mode):
        self.options=dict(options);self.rows=list(rows);self.mode=int(mode)
        self.directory=Path(options.get('model_directory') or DEFAULT_MODELS).expanduser().resolve()
        self.backend=options.get('backend','reduced')
        self.frequency_evaluation=options.get('frequency_evaluation','cubic')
        if self.backend not in ('reduced','full'):raise ValueError('backend must be reduced or full')
        if self.frequency_evaluation not in ('cubic','direct'):raise ValueError('frequency_evaluation must be cubic or direct')
        self.interpolation='cubic_formula_frequency' if self.frequency_evaluation=='cubic' else 'direct_formula'
        self.cutoff=float(options.get('frequency_cutoff_mhz',8.))
        self.step=float(options.get('frequency_step_mhz',.025))
        if not 0<self.cutoff<=8 or not np.isfinite(self.step) or not 0<self.step<=.1:raise ValueError('Invalid frequency cutoff/step')
        self.damping_law=options.get('bulk_damping_law','constant_poisson_kelvin_voigt')
        if any(r['damping_cm2_us']>0 for r in rows) and self.damping_law!='constant_poisson_kelvin_voigt':
            raise ValueError('Nonzero bulk damping requires the explicit constant-Poisson Kelvin--Voigt law')
        self._spectra=OrderedDict();self.cache_size=int(options.get('cache_materials',1))
        if self.cache_size<1:raise ValueError('cache_materials must be positive')
        self.fingerprints={};self.coefficient_sha256={}
        self.models={}
        for metal in ['steel','titanium']:
            if self.backend=='reduced':
                path=self.directory/f'{metal}.npz'
                if not path.is_file():raise FileNotFoundError(f'Continuous coefficients missing: {path}; original bank NPZ files are not used')
                self.coefficient_sha256[metal]=hashlib.sha256(path.read_bytes()).hexdigest()
                self.models[metal]=ContinuousTeeth(metal,self.directory)
                md=self.models[metal].metadata
                if md['metal']!=metal or md['height_mm']!=.5 or md['pitch_mm']!=1. or md['coarse_nx']!=80:
                    raise ValueError(f'Continuous coefficient geometry/material mismatch: {path}')
            else:self.models[metal]=FullTeeth(metal,nx=80)

    def preflight(self,indices=None):
        for i in (range(len(self.rows)) if indices is None else indices):
            if not 0<=int(i)<len(self.rows):raise ValueError('Invalid continuous candidate index')

    def _material_spectra(self,index):
        r=self.rows[int(index)];key=(r['rho'],r['cs'],r['nu'])
        if key in self._spectra:self._spectra.move_to_end(key);return self._spectra[key]
        cells={m:model.compile(*key,eta=ETA) for m,model in self.models.items()}
        if self.frequency_evaluation=='cubic':
            n=int(np.floor(self.cutoff/self.step+1e-10));f=np.arange(n+1)*self.step
            if f[-1]<self.cutoff-1e-10:f=np.r_[f,self.cutoff]
            else:f[-1]=self.cutoff
            if len(f)<4:f=np.linspace(0.,self.cutoff,4)
            cells={m:cell.tabulate(f) for m,cell in cells.items()}
        self._spectra[key]=cells
        while len(self._spectra)>self.cache_size:self._spectra.popitem(last=False)
        return cells

    def sectors(self,index,metal,frequency):
        f=float(frequency)
        if f<0 or f>self.cutoff+1e-12:raise ValueError('Frequency outside the configured source band')
        return self._material_spectra(index)[metal].sector_at(min(f,self.cutoff))

    def response(self,index,frequency,*,lengths_mm,mode,offsets_mm=(0.,0.,0.,0.),reduced=None):
        row=self.rows[int(index)]
        if mode!=self.mode:raise ValueError('Provider damping convention belongs to another polarization')
        bulk=None
        if row['damping_cm2_us']:
            omega=2*np.pi*frequency+1j*ETA
            factor=np.sqrt(1-1j*omega*row['bulk_relaxation_time_us'])
            bulk=dict(rho=row['rho'],cp=row['cp']*factor,cs=row['cs']*factor)
        return super().response(index,frequency,lengths_mm=lengths_mm,mode=mode,
             offsets_mm=offsets_mm,reduced=reduced,bulk_material=bulk)

    def metadata_for_candidate(self,index):
        return dict(backend=f'p2_continuous_{self.backend}',candidate_index=int(index),
            parameter_evaluation='material_affine_elastic_operator',frequency_interpolation=self.interpolation,
            frequency_step_mhz=self.step if self.frequency_evaluation=='cubic' else None,
            bank_step_mhz=None,original_matrix_bank_required=False,coarse_nx=80,
            coefficient_sha256=self.coefficient_sha256,
            bulk_damping_cm2_us=self.rows[int(index)]['damping_cm2_us'],
            bulk_damping_law=self.damping_law,bulk_relaxation_time_us=self.rows[int(index)]['bulk_relaxation_time_us'],
            nu_s_cm2_us=self.rows[int(index)]['nu_s_cm2_us'],nu_p_cm2_us=self.rows[int(index)]['nu_p_cm2_us'],
            numerical_limits='Reduced operator and optional frequency spline require source-band checks; full backend is available. The local FD adjoint is not used.')

def forward_with_continuous(model,args):
    from .teeth_bank_search import forward_with_bank
    args=dict(args);options=dict(args['teeth_continuous_options'])
    mode=0 if args['assembly_dict']['wave_type']=='_p' else 1
    options['frequency_cutoff_mhz']=args['frequency_cutoff'];provider=options.get('_bank')
    if provider is None:
        if 'material' not in options:raise ValueError('Provide teeth_continuous_options[material] with rho,cs,nu')
        rows=candidate_materials(dict(options,materials=[options['material']]),mode)
        provider=ContinuousMatrixProvider(options,rows,mode);index=0
    else:index=int(options['material_index'])
    args['teeth_bank_options']=dict(options,_bank=provider,material_index=index)
    result=forward_with_bank(model,args)
    model.teeth_continuous_options={k:v for k,v in options.items() if not k.startswith('_')}
    model.teeth_continuous_metadata=dict(model.teeth_bank_metadata)
    # Keep the existing guard/checkpoint marker, preventing any accidental
    # use of the old FD adjoint on diagnostic ramp arrays.
    return result
