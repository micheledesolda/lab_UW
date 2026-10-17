"""Acceleration of the existing P1 periodic FEM, without changing its equations.

The circulant DtN is evaluated by FFT. Reflection symmetry splits the solve
into independent P-even and SV-even sectors. All ± diffraction orders and
both incidence ports can be reconstructed exactly from the stored sectors.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, block_diag
from scipy.sparse.linalg import splu

from .teeth_core.teeth_cell import Cell, moduli
from .teeth_core.teeth_forward import scattering_matrix as reference_scattering, cascade, propagation, mirrored


def waves(mat, omega, orders, pitch=1., direction=1, normal=1):
    q=2*np.pi*np.asarray(orders)/pitch
    kp,ks=omega/mat['cp'],omega/mat['cs']
    zp=direction*np.sqrt(np.asarray(kp*kp-q*q,complex))
    zs=direction*np.sqrt(np.asarray(ks*ks-q*q,complex))
    u=np.empty((len(q),2,2),complex)
    u[:,0,0]=q/kp;u[:,1,0]=zp/kp
    u[:,0,1]=zs/ks;u[:,1,1]=-q/ks
    lam,g=moduli(mat)
    z=np.column_stack([zp,zs])
    t=np.empty_like(u)
    t[:,0,:]=normal*1j*g*(z*u[:,0,:]+q[:,None]*u[:,1,:])
    t[:,1,:]=normal*1j*(lam*q[:,None]*u[:,0,:]+(lam+2*g)*z*u[:,1,:])
    return u,t


def stable_dtn(mat,omega,orders,pitch,direction):
    q=2*np.pi*np.asarray(orders)/pitch
    kp,ks=omega/mat['cp'],omega/mat['cs']
    zp=direction*np.sqrt(np.asarray(kp*kp-q*q,complex))
    zs=direction*np.sqrt(np.asarray(ks*ks-q*q,complex))
    denominator=q*q+zp*zs
    nonzero=q!=0
    denominator[nonzero]=(q[nonzero]**2*(kp*kp+ks*ks)-kp*kp*ks*ks)/(q[nonzero]**2-zp[nonzero]*zs[nonzero])
    ratio=mat['rho']*omega**2/denominator
    _,g=moduli(mat)
    out=np.empty((len(q),2,2),complex)
    out[:,0,0]=direction*1j*ratio*zp
    out[:,1,1]=direction*1j*ratio*zs
    out[:,0,1]=direction*1j*q*(2*g-ratio)
    out[:,1,0]=-out[:,0,1]
    return out


def fast_boundary(cell,mat,omega,direction,orders):
    alln=cell.orders
    u,t=waves(mat,omega,alln,cell.pitch,direction,direction)
    dt=stable_dtn(mat,omega,alln,cell.pitch,direction)
    stride=getattr(cell,'boundary_stride',1)
    nb=cell.nx//stride
    hats=getattr(cell,'hat_weights',np.sinc(alln/cell.nx)[:,None]**2/cell.nx)
    sums=np.zeros((nb,stride,2,stride,2),complex)
    for aa in range(stride):
        for bb in range(stride):
            w=cell.pitch*hats[:,aa]*hats[:,bb]*np.exp(2j*np.pi*alln*(aa-bb)/cell.nx)
            np.add.at(sums[:,aa,:,bb,:],np.mod(alln,nb),w[:,None,None]*dt)
    kernels=np.fft.ifft(sums,axis=0)*nb
    ii=np.arange(nb)
    op=kernels[(ii[:,None]-ii[None,:])%nb].transpose(0,2,3,1,4,5).reshape(2*cell.nx,2*cell.nx)
    un,tn=waves(mat,omega,orders,cell.pitch,direction,direction)
    a=cell.Q[np.searchsorted(alln,orders)]
    return op,a,un,tn


def parity_basis(nx,nz,parity):
    # Reflection x -> -x: ux changes sign and uz retains sign.
    rows=[];cols=[];data=[];column=0
    for z in range(nz+1):
        for j in range(nx//2+1):
            jj=(-j)%nx
            for component in range(2):
                sign=parity*(-1 if component==0 else 1)
                if jj==j:
                    if sign<0:continue
                    rows.append(2*(z*nx+j)+component);cols.append(column);data.append(1.)
                else:
                    rows.extend([2*(z*nx+j)+component,2*(z*nx+jj)+component])
                    cols.extend([column,column]);data.extend([2**-.5,sign*2**-.5])
                column+=1
    return coo_matrix((data,(rows,cols)),shape=(2*nx*(nz+1),column)).tocsc()


def modal_basis(orders,parity):
    nmax=max(orders);size=2*len(orders);cols=[]
    v=np.zeros(size);v[2*nmax+(0 if parity==1 else 1)]=1.;cols.append(v)
    for n in range(1,nmax+1):
        for mode in [0,1]:
            v=np.zeros(size);v[2*(nmax+n)+mode]=2**-.5
            v[2*(nmax-n)+mode]=parity*(1 if mode==0 else -1)*2**-.5
            cols.append(v)
    return np.column_stack(cols)


class FastCell:
    def __init__(self,cell,orders):
        self.cell=cell;self.orders=np.asarray(orders)
        self.P=[parity_basis(cell.nx,cell.nz,p) for p in [1,-1]]
        self.Q=[modal_basis(self.orders,p) for p in [1,-1]]
        self.Qboth=[block_diag([q,q]).toarray() for q in self.Q]
        self.K=[(p.T @ cell.K @ p).tocsc() for p in self.P]
        self.M=[(p.T @ cell.M @ p).tocsc() for p in self.P]
        self.boundary_P=[[p[ids,:].tocsr() for ids in cell.boundary] for p in self.P]

    def solve(self,frequency,eta=.02):
        c=self.cell;omega=2*np.pi*frequency+1j*eta
        nch=2*len(self.orders)
        rhsb=[];ops=[];basis=[];incidents=[]
        for side,(mat,d) in enumerate([(c.metal,-1),(c.gouge,1)]):
            op,a,u,t=fast_boundary(c,mat,omega,d,self.orders)
            inc,it=waves(mat,omega,self.orders,c.pitch,-d,d)
            residual=it-stable_dtn(mat,omega,self.orders,c.pitch,d) @ inc
            b=(c.pitch*a.conj()[:,:,None,None]*residual[:,None,:,:]).transpose(1,2,0,3).reshape(2*c.nx,nch)
            full=np.zeros((2*c.nx,2*nch),complex);full[:,side*nch:(side+1)*nch]=b
            rhsb.append(full);ops.append(op);basis.append((a,u));incidents.append(inc)
        sectors=[]
        for pi,p in enumerate(self.P):
            bp=self.boundary_P[pi];qb=self.Qboth[pi]
            op=coo_matrix((p.shape[1],p.shape[1]),dtype=complex).tocsc()
            rhs=np.zeros((p.shape[1],qb.shape[1]),complex)
            for side in range(2):
                # Restrict dense boundary operator before placing it in volume.
                active=np.unique(bp[side].indices)
                small=bp[side][:,active].toarray()
                dense=small.T @ ops[side] @ small
                rows=np.repeat(active,len(active));cols=np.tile(active,len(active))
                op+=coo_matrix((dense.ravel(),(rows,cols)),shape=op.shape).tocsc()
                rhs+=bp[side].T @ (rhsb[side] @ qb)
            A=self.K[pi]-omega**2*self.M[pi]-op
            # The FE/DtN sparsity is symmetric. Minimum-degree ordering reduces
            # fill while preserving the same complex linear system and pivoting.
            sol=splu(A,permc_spec='MMD_AT_PLUS_A').solve(rhs)
            result=np.zeros((2*nch,qb.shape[1]),complex)
            for side in range(2):
                a,u=basis[side]
                boundary=bp[side] @ sol
                coeff=np.stack([a @ boundary[0::2],a @ boundary[1::2]],axis=1)
                cin=qb[side*nch:(side+1)*nch].reshape(len(self.orders),2,-1)
                coeff-=incidents[side] @ cin
                result[side*nch:(side+1)*nch]=np.linalg.solve(u,coeff).reshape(nch,-1)
            sectors.append(qb.T @ result)
        return np.stack(sectors)

    def full(self,sectors):
        full=sum(q @ s @ q.T for q,s in zip(self.Qboth,sectors))
        n=full.shape[0]//2
        return full[:n,:n],full[n:,n:],full[n:,:n],full[:n,n:]


def reconstruct(sectors,orders):
    qs=[block_diag([modal_basis(orders,p)]*2).toarray() for p in [1,-1]]
    full=sum(q @ s @ q.T for q,s in zip(qs,sectors))
    n=full.shape[0]//2
    return full[:n,:n],full[n:,n:],full[n:,:n],full[:n,n:]


if __name__=='__main__':
    import time,json
    m=dict(rho=7.85,cp=5.9,cs=3.25);g=dict(rho=2.25,cp=2.,cs=1.)
    report=[]
    for nx in [64,96,128]:
        cell=Cell(nx,m,g);orders=np.arange(-12,13);fast=FastCell(cell,orders)
        tick=time.monotonic();s=fast.solve(4.3);elapsed=time.monotonic()-tick
        ref=reference_scattering(cell,4.3,orders,.02)
        full=fast.full(s)
        diff=max(np.max(abs(a-b)) for a,b in zip(ref,full))
        rel=np.linalg.norm(np.array(full)-np.array(ref))/np.linalg.norm(ref)
        report.append(dict(nx=nx,seconds=elapsed,max_absolute_difference=float(diff),relative_difference=float(rel)))
        print(report[-1],flush=True)
        assert rel<1e-9
    Path(__file__).with_name('fast_solver_verification.json').write_text(json.dumps(report,indent=2))
