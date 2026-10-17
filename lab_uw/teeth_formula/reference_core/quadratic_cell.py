"""Fitted quadratic (P2) triangles and exact Fourier boundary trace.

The same geometry, weak elastodynamic form and outgoing halfspaces as Cell.
Consistent mass, degree-4 exact triangle quadrature, plane strain.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from .teeth_core.teeth_cell import Cell,moduli
from .teeth_core.teeth_cell import wave_matrices


class QuadraticCell(Cell):
    def __init__(self,nx,metal,gouge,height=.5,pitch=1.,aliases=2,nz=None,
                 metal_classifier=None,mirror_upper=False,mirror_selector=None):
        assert nx%4==0
        coarse_nx=nx;coarse_nz=nx//2 if nz is None else int(nz)
        self.nx,self.nz=2*nx,2*coarse_nz
        self.height,self.pitch=height,pitch;self.metal,self.gouge=metal,gouge
        self.geometry='teeth';self.boundary_stride=2
        dx,dz=pitch/coarse_nx,height/coarse_nz
        nodes=np.array([[i*dx-pitch/2,j*dz] for j in range(coarse_nz+1) for i in range(coarse_nx+1)])
        triangles=[]
        for j in range(coarse_nz):
            for i in range(coarse_nx):
                a=j*(coarse_nx+1)+i;b,c,d=a+1,a+coarse_nx+1,a+coarse_nx+2
                flip=(mirror_upper and j>=coarse_nz//2) if mirror_selector is None else bool(mirror_selector((j+.5)*dz))
                if (i<coarse_nx//2)!=flip:
                    triangles.extend([[a,b,d],[a,d,c]])
                else:triangles.extend([[a,b,c],[b,d,c]])
        triangles=np.asarray(triangles);xy=nodes[triangles];centers=xy.mean(axis=1)
        mask=(centers[:,1]<height*(1-2*abs(centers[:,0])/pitch)) if metal_classifier is None else np.asarray(metal_classifier(centers),bool)
        area2=(xy[:,1,0]-xy[:,0,0])*(xy[:,2,1]-xy[:,0,1])-(xy[:,2,0]-xy[:,0,0])*(xy[:,1,1]-xy[:,0,1])
        area=area2/2
        grad=np.stack([np.stack([xy[:,1,1]-xy[:,2,1],xy[:,2,0]-xy[:,1,0]],axis=1),
                       np.stack([xy[:,2,1]-xy[:,0,1],xy[:,0,0]-xy[:,2,0]],axis=1),
                       np.stack([xy[:,0,1]-xy[:,1,1],xy[:,1,0]-xy[:,0,0]],axis=1)],axis=1)/area2[:,None,None]
        aa=.445948490915965;bb=.091576213509771
        bary=np.array([[aa,aa,1-2*aa],[aa,1-2*aa,aa],[1-2*aa,aa,aa],
                       [bb,bb,1-2*bb],[bb,1-2*bb,bb],[1-2*bb,bb,bb]])
        weights=np.array([.223381589678011]*3+[.109951743655322]*3)
        ne=len(triangles);ke=np.zeros((ne,12,12));me=np.zeros_like(ke)
        lm,gm=moduli(metal);lg,gg=moduli(gouge)
        # Optional per-element material arrays for a direct multi-metal DDS.
        element_material=getattr(metal_classifier,'element_material',None)
        if element_material is None:
            lam=np.where(mask,lm,lg);g=np.where(mask,gm,gg);rho=np.where(mask,metal['rho'],gouge['rho'])
        else:
            rho,lam,g=element_material(centers)
        D=np.zeros((ne,3,3));D[:,0,0]=D[:,1,1]=lam+2*g;D[:,0,1]=D[:,1,0]=lam;D[:,2,2]=g
        for L,w in zip(bary,weights):
            N=np.concatenate([L*(2*L-1),[4*L[0]*L[1],4*L[1]*L[2],4*L[2]*L[0]]])
            gradients=np.concatenate([(4*L-1)[None,:,None]*grad,
                np.stack([4*(L[0]*grad[:,1]+L[1]*grad[:,0]),4*(L[1]*grad[:,2]+L[2]*grad[:,1]),4*(L[2]*grad[:,0]+L[0]*grad[:,2])],axis=1)],axis=1)
            B=np.zeros((ne,3,12));B[:,0,0::2]=gradients[:,:,0];B[:,1,1::2]=gradients[:,:,1]
            B[:,2,0::2]=gradients[:,:,1];B[:,2,1::2]=gradients[:,:,0]
            ke+=np.einsum('eai,eab,ebj,e->eij',B,D,B,area*w)
            mass=rho[:,None,None]*area[:,None,None]*w*np.outer(N,N)[None,:,:]
            me[:,0::2,0::2]+=mass;me[:,1::2,1::2]+=mass
        ids=triangles;row0=(ids//(coarse_nx+1))*2;col0=(ids%(coarse_nx+1))*2
        cr=np.stack([col0[:,0],col0[:,1],col0[:,2],(col0[:,0]+col0[:,1])//2,(col0[:,1]+col0[:,2])//2,(col0[:,2]+col0[:,0])//2],axis=1)
        zr=np.stack([row0[:,0],row0[:,1],row0[:,2],(row0[:,0]+row0[:,1])//2,(row0[:,1]+row0[:,2])//2,(row0[:,2]+row0[:,0])//2],axis=1)
        periodic=zr*self.nx+(cr%self.nx)
        edof=np.stack([2*periodic,2*periodic+1],axis=2).reshape(ne,12)
        rows=np.repeat(edof,12,axis=1).ravel();cols=np.tile(edof,(1,12)).ravel()
        self.ndof=2*self.nx*(self.nz+1)
        self.K=coo_matrix((ke.ravel(),(rows,cols)),shape=(self.ndof,)*2).tocsc()
        self.M=coo_matrix((me.ravel(),(rows,cols)),shape=(self.ndof,)*2).tocsc()
        self.boundary=[np.arange(2*self.nx),np.arange(self.ndof-2*self.nx,self.ndof)]
        self.metal_fraction=float(area[mask].sum()/area.sum())
        self.orders=np.arange(-aliases*self.nx,aliases*self.nx+1)
        # Integrate the symmetric corner and midpoint P2 boundary hats.
        t,w=np.polynomial.legendre.leggauss(40);t=(t+1)/2;w=w/2
        phase=2*np.pi*self.orders[:,None]/coarse_nx
        corner=2/coarse_nx*np.sum(w*(1-3*t+2*t*t)*np.cos(phase*t),axis=1)
        mid=1/coarse_nx*np.sum(w*(1-t*t)*np.cos(phase*t/2),axis=1)
        self.hat_weights=np.column_stack([corner,mid])
        ii=np.arange(self.nx)
        self.Q=np.exp(-2j*np.pi*self.orders[:,None]*ii[None,:]/self.nx)*self.hat_weights[:,ii%2]

    def boundary_operator(self,mat,omega,direction):
        matrix=np.zeros((2*self.nx,2*self.nx),complex);basis=[]
        for n,a in zip(self.orders,self.Q):
            u,t=wave_matrices(mat,omega,2*np.pi*n/self.pitch,direction,direction)
            dt=t @ np.linalg.inv(u)
            matrix+=self.pitch*np.einsum('a,b,ij->aibj',a.conj(),a,dt).reshape(2*self.nx,2*self.nx)
            basis.append((int(n),a,u,t))
        return matrix,basis


if __name__=='__main__':
    import time,json
    from .fast_solver import FastCell,reference_scattering
    m=dict(rho=7.85,cp=5.9,cs=3.25);g=dict(rho=2.25,cp=2.,cs=.8)
    rows=[];orders=np.arange(-12,13)
    for nx in [24,32,48,64]:
        tick=time.monotonic();cell=QuadraticCell(nx,m,g);fast=FastCell(cell,orders);built=time.monotonic()-tick
        tick=time.monotonic();s=fast.solve(4.3);elapsed=time.monotonic()-tick
        ref=reference_scattering(cell,4.3,orders,.02)
        rel=np.linalg.norm(np.array(fast.full(s))-np.array(ref))/np.linalg.norm(ref)
        print(dict(nx=nx,build=built,seconds=elapsed,reference_error=float(rel)),flush=True)
        assert rel<1e-9
        n0=len(orders)//2;i=2*n0+1
        val=fast.full(s)[2][i,i]
        rows.append(dict(nx=nx,re=float(val.real),im=float(val.imag),seconds=elapsed))
    Path(__file__).with_name('quadratic_solver_verification.json').write_text(json.dumps(rows,indent=2))
