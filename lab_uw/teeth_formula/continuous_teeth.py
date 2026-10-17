"""Continuous matrix formulas for steel/gouge and titanium/gouge teeth.

Material dependence is explicit in rho, mu and lambda. The fixed real Galerkin
operators preserve the elastic operator's structure. Modal poles move with
the gouge. No frequency/material grid interpolation is required at runtime.
"""
from pathlib import Path
import json,time
from functools import lru_cache
from types import SimpleNamespace
import numpy as np
from scipy.linalg import eigh,solve
ROOT=Path(__file__).resolve().parent
from .reference_core.fast_solver import fast_boundary,waves,stable_dtn,modal_basis,reconstruct
from scipy.linalg import block_diag
from ..teeth_bank import METALS,ORDERS
from .port_formula import port

def gouge(rho,cs,nu,check=True):
    if check:
        for value,bounds in [(rho,(1.5,3.)),(cs,(.8,1.8)),(nu,(.15,.45))]:
            if not np.isfinite(value) or value<bounds[0]-1e-12 or value>bounds[1]+1e-12:
                raise ValueError('Material outside the verified bank parameter domain')
    return dict(rho=rho,cs=cs,cp=cs*np.sqrt(2*(1-nu)/(1-2*nu)),nu=nu)

class ContinuousTeeth:
    def __init__(self,metal,model_directory=None):
        if metal not in METALS:raise ValueError('metal must be steel or titanium')
        p=Path(model_directory or ROOT/'models')/f'{metal}.npz'
        with np.load(p,allow_pickle=False) as d:self.data={k:d[k].copy() for k in d.files}
        self.metadata=json.loads(str(self.data['metadata_json']));self.metal=metal
        md=self.metadata
        self.cell=SimpleNamespace(nx=md['nx'],pitch=1.,boundary_stride=2,
          orders=self.data['orders'],Q=self.data['Q'],hat_weights=self.data['hat_weights'])
        self.qboth=[block_diag(*([modal_basis(ORDERS,p)]*2)) for p in [1,-1]]
        self.retained_orders=ORDERS
        self.fourier_traces=[np.stack([self.cell.Q@self.data[f'trace_{pi}'][0::2],
           self.cell.Q@self.data[f'trace_{pi}'][1::2]],axis=1) for pi in range(2)]
        # Fourier traces are sparse up to roundoff. Assemble their exact
        # harmonic couplings once, avoiding dense alias-frequency products.
        self.port_projectors=[]
        for ft in self.fourier_traces:
            terms=[]
            for n,a in enumerate(ft):
                ci,bi=np.nonzero(abs(a)>1e-12)
                for i,b in zip(ci,bi):
                    for j,c in zip(ci,bi):terms.append((n,i,j,b,c,a[i,b].conjugate()*a[j,c]))
            ids=np.array([t[:5] for t in terms],int).T
            self.port_projectors.append((ids,np.array([t[5] for t in terms])))

    def compile(self,rho,cs,nu,*,eta=.02):
        material=gouge(rho,cs,nu);mu=rho*cs*cs;lam=2*mu*nu/(1-2*nu)
        if eta<0 or not np.isfinite(eta):raise ValueError('eta must be finite and nonnegative')
        started=time.monotonic();sectors=[]
        for pi in range(2):
            d=self.data;k=d[f'K0_{pi}']+mu*d[f'Kg_{pi}']+lam*d[f'Kl_{pi}']
            m=d[f'M0_{pi}']+rho*d[f'Mr_{pi}'];nb=self.metadata['sectors'][pi]['boundary_modes']
            eigenvalues,v=eigh(k[nb:,nb:],m[nb:,nb:],check_finite=False)
            sectors.append(dict(Kbb=k[:nb,:nb],Mbb=m[:nb,:nb],
              Kib=v.T@k[nb:,:nb],Mib=v.T@m[nb:,:nb],eigenvalues=eigenvalues,
              trace=d[f'trace_{pi}'],eigenvectors=v))
        result=CompiledTeeth(self,material,sectors,eta)
        result.compile_seconds=time.monotonic()-started
        return result

class CompiledTeeth:
    def __init__(self,model,material,sectors,eta):
        self.model=model;self.material=material;self.components=sectors;self.eta=eta
        self.sector_at=lru_cache(maxsize=512)(self._cached_sector_at)
    def _sector_at_original(self,frequency):
        f=float(frequency)
        if not np.isfinite(f) or f<0 or f>8:raise ValueError('Frequency outside 0--8 MHz')
        if f==0 and self.eta==0:raise ValueError('Use positive eta at zero frequency')
        omega=2*np.pi*f+1j*self.eta;c=self.model.cell;nch=50
        ports=[]
        for mat,direction in [(METALS[self.model.metal],-1),(self.material,1)]:
            op,a,u,t=fast_boundary(c,mat,omega,direction,ORDERS)
            inc,it=waves(mat,omega,ORDERS,1.,-direction,direction)
            residual=it-stable_dtn(mat,omega,ORDERS,1.,direction)@inc
            rhs=(a.conj()[:,:,None,None]*residual[:,None,:,:]).transpose(1,2,0,3).reshape(2*c.nx,nch)
            ports.append((op,a,u,inc,rhs))
        answers=[]
        for pi,comp in enumerate(self.components):
            trace=comp['trace'];nh=trace.shape[1];nb=2*nh
            coupling=comp['Kib']-omega**2*comp['Mib']
            a=comp['Kbb']-omega**2*comp['Mbb']-coupling.T@(coupling/(comp['eigenvalues']-omega**2)[:,None])
            a=a.astype(complex);rhs=np.zeros((nb,50),complex);q=self.model.qboth[pi]
            for side,(op,aa,u,inc,b) in enumerate(ports):
                sl=slice(side*nh,(side+1)*nh)
                a[sl,sl]-=trace.T@op@trace
                rhs[sl]=trace.T@b@q[side*50:(side+1)*50]
            displacement=solve(a,rhs,assume_a='sym',check_finite=False)
            result=np.zeros((100,50),complex)
            for side,(op,aa,u,inc,b) in enumerate(ports):
                boundary=trace@displacement[side*nh:(side+1)*nh]
                coeff=np.stack([aa@boundary[0::2],aa@boundary[1::2]],axis=1)
                cin=q[side*50:(side+1)*50].reshape(25,2,50)
                coeff-=inc@cin
                result[side*50:(side+1)*50]=np.linalg.solve(u,coeff).reshape(50,50)
            answers.append(q.T@result)
        return np.stack(answers)
    def _evaluate(self,frequency,jacobian=False):
        f=float(frequency)
        if not np.isfinite(f) or f<0 or f>8:raise ValueError('Frequency outside 0--8 MHz')
        if f==0 and self.eta==0:raise ValueError('Use positive eta at zero frequency')
        if jacobian and self.eta<=0:raise ValueError('Use positive eta for parameter derivatives at diffraction thresholds')
        omega=2*np.pi*f+1j*self.eta;answers=[];derivatives=[]
        for pi,comp in enumerate(self.components):
            nh=comp['trace'].shape[1];nb=2*nh
            coupling=comp['Kib']-omega**2*comp['Mib']
            divided=coupling/(comp['eigenvalues']-omega**2)[:,None]
            a=(comp['Kbb']-omega**2*comp['Mbb']-coupling.T@divided).astype(complex)
            b=np.zeros((nb,50),complex);c=np.zeros((50,nb),complex);d=np.zeros((50,50),complex)
            if jacobian:
                da=np.zeros((3,nb,nb),complex);db=np.zeros((3,nb,50),complex)
                dc=np.zeros((3,50,nb),complex);dd=np.zeros((3,50,50),complex)
                rho,cs,nu=(self.material[k] for k in ['rho','cs','nu'])
                mu=rho*cs**2;lam=2*mu*nu/(1-2*nu)
                dmu=[cs**2,2*rho*cs,0.];dlam=[lam/rho,2*lam/cs,2*mu/(1-2*nu)**2]
                x=comp['eigenvectors']@divided;ops=self.model.data
                for p in range(3):
                    derivative=dmu[p]*ops[f'Kg_{pi}']+dlam[p]*ops[f'Kl_{pi}']
                    if p==0:derivative=derivative-omega**2*ops[f'Mr_{pi}']
                    da[p]=derivative[:nb,:nb]-derivative[:nb,nb:]@x-x.T@derivative[nb:,:nb]+x.T@(derivative[nb:,nb:]@x)
            for side,(mat,direction) in enumerate([(METALS[self.model.metal],-1),(self.material,1)]):
                op,bp,cp,dp,dop,dbp,dcp,ddp=port(self.model,pi,mat,omega,direction,jacobian and side==1)
                sl=slice(side*nh,(side+1)*nh);sm=slice(side*25,(side+1)*25)
                a[sl,sl]-=op;b[sl,sm]=bp;c[sm,sl]=cp;d[sm,sm]=dp
                if jacobian:
                    da[:,sl,sl]-=dop;db[:,sl,sm]=dbp;dc[:,sm,sl]=dcp;dd[:,sm,sm]=ddp
            x=solve(a,b,assume_a='sym',check_finite=False);answers.append(c@x-d)
            if jacobian:
                rhs=np.concatenate([db[p]-da[p]@x for p in range(3)],axis=1)
                dx=solve(a,rhs,assume_a='sym',check_finite=False)
                derivatives.append(np.stack([dc[p]@x+c@dx[:,p*50:(p+1)*50]-dd[p] for p in range(3)]))
        if jacobian:return np.stack(answers),np.stack(derivatives,axis=1)
        return np.stack(answers)
    def _cached_sector_at(self,frequency):
        value=self._evaluate(frequency);value.setflags(write=False);return value
    def jacobian_at(self,frequency):
        """Return sectors (2,50,50) and analytic dS/d(rho,cS,nu) (3,2,50,50)."""
        return self._evaluate(frequency,True)
    def sectors(self,frequencies):
        return np.stack([self.sector_at(f) for f in np.asarray(frequencies).reshape(-1)])
    def scattering(self,frequency):return reconstruct(self.sector_at(frequency),ORDERS)
    def transmission(self,frequency):return self.scattering(frequency)[2]
    def tabulate(self,frequencies=None):
        """Optional spline acceleration; approximate frequency evaluation only.

        Material dependence comes from the analytic formula before sampling.
        Use the direct formula for verification or parameter derivatives.
        """
        return TabulatedTeeth(self,np.arange(321)*.025 if frequencies is None else frequencies)

class TabulatedTeeth:
    def __init__(self,compiled,frequencies):
        from scipy.interpolate import CubicSpline
        self.frequency=np.asarray(frequencies,float)
        if self.frequency.ndim!=1 or len(self.frequency)<4 or np.any(np.diff(self.frequency)<=0):
            raise ValueError('At least four ordered distinct frequencies required')
        self.spline=CubicSpline(self.frequency,compiled.sectors(self.frequency),axis=0,extrapolate=False)
        self.material=compiled.material;self.eta=compiled.eta
    def sector_at(self,frequency):
        f=float(frequency)
        if f<self.frequency[0] or f>self.frequency[-1]:raise ValueError('No extrapolation beyond the sampled frequency grid')
        return self.spline(f)
    def sectors(self,frequencies):return np.stack([self.sector_at(f) for f in np.asarray(frequencies).reshape(-1)])
    def scattering(self,frequency):return reconstruct(self.sector_at(frequency),ORDERS)
    def transmission(self,frequency):return self.scattering(frequency)[2]
