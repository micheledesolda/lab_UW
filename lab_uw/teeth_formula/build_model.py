"""Fixed, material-affine Galerkin model of the bank's original P2 cell.

Boundary Fourier traces and interior component modes are parameter independent.
The input bank is for validation, not a source of adjusted damping or amplitudes.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag,qr
from scipy.sparse.linalg import eigsh,splu
ROOT=Path(__file__).resolve().parent
from .reference_core.quadratic_cell import QuadraticCell
from .reference_core.fast_solver import FastCell,parity_basis
from ..teeth_bank import METALS,ORDERS

def affine_operators(nx,metal):
    # Only assembly is used for these basis materials, never a wave solve.
    basis=[dict(rho=0.,cs=1.,cp=1.),dict(rho=1.,cs=0.,cp=0.),
           dict(rho=1.,cs=1.,cp=np.sqrt(2)),dict(rho=1.,cs=0.,cp=1.)]
    cells=[FastCell(QuadraticCell(nx,METALS[metal],g),ORDERS) for g in basis]
    matrices=[]
    for pi in range(2):
        matrices.append(dict(K0=cells[0].K[pi],M0=cells[0].M[pi],
          Kg=cells[2].K[pi]-cells[0].K[pi],
          Kl=cells[3].K[pi]-cells[0].K[pi],
          Mr=cells[1].M[pi]-cells[0].M[pi]))
    return cells[0],matrices

def face_trace(nx,parity,ntrace):
    x=np.arange(nx)/nx;cols=[]
    # Reflection: ux odd and uz even in P parity, vice versa in SV parity.
    for n in range(ntrace+1):
        for component in [0,1]:
            sign=parity*(-1 if component==0 else 1)
            if n==0 and sign<0:continue
            v=np.zeros((nx,2));v[:,component]=(np.cos(2*np.pi*n*x) if sign>0 else np.sin(2*np.pi*n*x))
            v=v.ravel()
            if np.linalg.norm(v)<1e-10:continue
            v/=np.linalg.norm(v);cols.append(v)
    return np.column_stack(cols)

def build(nx,ntrace,nmodes,output,metals=('steel','titanium'),nstatic=0):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    report={}
    for metal in metals:
        fast,ops=affine_operators(nx,metal);cell=fast.cell;nface=cell.nx
        saved={};info=[]
        for pi,parity in enumerate([1,-1]):
            op=ops[pi];n=op['K0'].shape[0];ib=np.r_[np.arange(nface),np.arange(n-nface,n)];ii=np.arange(nface,n-nface)
            efull=face_trace(cell.nx,parity,ntrace)
            pb=parity_basis(cell.nx,0,parity)
            e=np.asarray(pb.T@efull);eb=block_diag(e,e);nb=eb.shape[1]
            gref=.8;rho=2.25
            snapshots=[]
            for nu in [.15,.45]:
                mu=rho*gref*gref;lam=2*mu*nu/(1-2*nu)
                k=op['K0']+mu*op['Kg']+lam*op['Kl'];m=op['M0']+rho*op['Mr']
                kii=k[ii,:][:,ii].tocsc();mii=m[ii,:][:,ii].tocsc()
                seed=20261012+(100 if metal=='titanium' else 0)+10*pi+int(100*nu)
                v0=np.random.default_rng(seed).normal(size=len(ii))
                eigenvalues,modes=eigsh(kii,k=nmodes,M=mii,sigma=0,which='LM',tol=1e-9,v0=v0)
                snapshots.append(modes)
                print(metal,parity,'nu',nu,'eigenmode ceiling MHz',np.sqrt(eigenvalues.max())/(2*np.pi),flush=True)
            # A fixed real basis, independent of every runtime material.
            phi,r=qr(np.column_stack(snapshots),mode='economic',check_finite=False)
            # Drop only numerically dependent columns through singular values.
            from scipy.linalg import svd
            ur,s,vr=svd(r,full_matrices=False,check_finite=False)
            rank=int(np.sum(s>s[0]*1e-10));phi=phi@ur[:,:rank]
            mu=rho*gref*gref;lam=2*mu*.3/(1-.6)
            k=op['K0']+mu*op['Kg']+lam*op['Kl'];kii=k[ii,:][:,ii].tocsc()
            w=-splu(kii,permc_spec='MMD_AT_PLUS_A').solve(np.asarray(k[ii,:][:,ib]@eb))
            if nstatic:
                # Static lifts at the stiffness corners add fields not well
                # represented by low cavity eigenmodes (especially evanescent
                # high-order boundary excitation). Randomized POD only builds
                # the fixed basis; it never randomizes a runtime evaluation.
                differences=[]
                for mu_s in [.96,9.72]:
                    for nu_s in [.15,.45]:
                        lam_s=2*mu_s*nu_s/(1-2*nu_s)
                        kk=op['K0']+mu_s*op['Kg']+lam_s*op['Kl']
                        ww=-splu(kk[ii,:][:,ii].tocsc(),permc_spec='MMD_AT_PLUS_A').solve(np.asarray(kk[ii,:][:,ib]@eb))
                        difference=ww-w;difference-=phi@(phi.T@difference)
                        differences.append(difference)
                residual=np.column_stack(differences);rng=np.random.default_rng(20261012+pi)
                sketch=residual@rng.normal(size=(residual.shape[1],nstatic+24))
                qq,_=qr(sketch,mode='economic',check_finite=False)
                uu,ss,vv=svd(qq.T@residual,full_matrices=False,check_finite=False)
                extra=qq@uu[:,:nstatic];extra-=phi@(phi.T@extra)
                extra,_=qr(extra,mode='economic',check_finite=False)
                phi=np.column_stack([phi,extra]);rank=phi.shape[1]
            v=np.zeros((n,nb+rank));v[ib,:nb]=eb;v[ii,:nb]=w;v[ii,nb:]=phi
            for name,a in op.items():
                reduced=v.T@(a@v);reduced=(reduced+reduced.T)/2
                saved[f'{name}_{pi}']=reduced
            saved[f'trace_{pi}']=efull
            info.append(dict(parity=parity,full_dof=n,boundary_modes=nb,interior_modes=rank,reduced_dof=nb+rank))
            print(metal,parity,'reduced',info[-1],'seconds',round(time.monotonic()-started,1),flush=True)
        saved['orders']=cell.orders;saved['Q']=cell.Q;saved['hat_weights']=cell.hat_weights
        meta=dict(metal=metal,coarse_nx=nx,nx=cell.nx,ntrace=ntrace,nmodes_per_reference=nmodes,
          sectors=info,nstatic=nstatic,height_mm=.5,pitch_mm=1.,eta_default_per_us=.02,
          density_domain_g_cm3=[1.5,3.],cs_domain_mm_us=[.8,1.8],nu_domain=[.15,.45],
          frequency_band_mhz=[0.,8.],physical_damping=0.,basis_fixed=True,
          note='Galerkin approximation of original P2 operator, not a statistical fit of bank entries')
        saved['metadata_json']=json.dumps(meta)
        np.savez_compressed(output/f'{metal}.npz',**saved);report[metal]=meta
    report['elapsed_seconds']=time.monotonic()-started
    (output/'build.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--nx',type=int,default=80)
    p.add_argument('--ntrace',type=int,default=80);p.add_argument('--nmodes',type=int,default=360)
    p.add_argument('--nstatic',type=int,default=80)
    p.add_argument('--output',default=str(ROOT/'models'));a=p.parse_args()
    build(a.nx,a.ntrace,a.nmodes,a.output,nstatic=a.nstatic)
