"""Shared modal algebra and independent P1/P2 scattering reference.

The sweep forward composes the local matrices in the layers. This module
contains no receiver-trace quotient correction or inversion integration.
Units: mm, microseconds, g/cm3; convention exp(-i omega t).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .teeth_cell import wave_matrices

def cascade(a, b):
    """Redheffer composition; ports retain evanescent as well as open modes."""
    al, ar, at, ab = a  # R_left, R_right, T_left_to_right, T_right_to_left
    bl, br, bt, bb = b
    eye = np.eye(len(al), dtype=complex)
    x = np.linalg.solve(eye-bl @ ar, np.column_stack([bl @ at, bb]))
    xx, xy = np.split(x, 2, axis=1)
    return al+ab @ xx, br+bt @ ar @ xy, bt @ (at+ar @ xx), ab @ xy

def propagation(mat, omega, length, orders, pitch):
    if length < -1e-10:
        raise ValueError('Negative bulk length: check mechanical dimensions and tooth heights')
    q=2*np.pi*np.asarray(orders)/pitch
    phase=[]
    for qq in q:
        for c in [mat['cp'],mat['cs']]:
            phase.append(np.exp(1j*np.sqrt(complex((omega/c)**2-qq**2))*max(length,0)))
    t=np.diag(phase);z=np.zeros_like(t)
    return z,z,t,t

def mirrored(s):
    """Reflect z; SV polarization changes sign in the modal convention."""
    l,r,t,b=s
    j=np.tile([1.,-1.],len(l)//2)
    def flip(a): return j[:,None]*a*j[None,:]
    return flip(r),flip(l),flip(b),flip(t)

def shifted(s, offset, orders, pitch):
    phase=np.repeat(np.exp(-2j*np.pi*np.asarray(orders)*offset/pitch),2)
    return tuple(phase[:,None]*a*phase.conj()[None,:] for a in s)

def scattering_matrix(cell, frequency, orders, eta):
    """One FEM factorization, all incident P/SV orders from both ports."""
    omega=2*np.pi*frequency+1j*eta
    ops=[];bases=[]
    for mat,d in [(cell.metal,-1),(cell.gouge,1)]:
        op,bas=cell.boundary_operator(mat,omega,d);ops.append(op)
        bases.append({n:(a,u,t) for n,a,u,t in bas})
    rows=[];cols=[];data=[]
    for ids,op in zip(cell.boundary,ops):
        rows.append(np.repeat(ids,len(ids)));cols.append(np.tile(ids,len(ids)));data.append(-op.ravel())
    A=cell.K-omega**2*cell.M+coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(cols))),shape=(cell.ndof,)*2).tocsc()
    nch=2*len(orders);rhs=np.zeros((cell.ndof,2*nch),complex)
    incidents={}
    for side,(mat,d,ids) in enumerate([(cell.metal,-1,cell.boundary[0]),(cell.gouge,1,cell.boundary[1])]):
        for k,n in enumerate(orders):
            q=2*np.pi*n/cell.pitch
            inc,it=wave_matrices(mat,omega,q,-d,d)
            a,out,ot=bases[side][int(n)]
            residual=it-ot @ np.linalg.solve(out,inc)
            incidents[side,int(n)]=inc
            for j in range(2):
                col=side*nch+2*k+j
                rhs[ids[0::2],col]=cell.pitch*a.conj()*residual[0,j]
                rhs[ids[1::2],col]=cell.pitch*a.conj()*residual[1,j]
    sol=splu(A).solve(rhs)
    result=np.zeros((2*nch,2*nch),complex)
    for side,ids in enumerate(cell.boundary):
        for k,n in enumerate(orders):
            a,u,t=bases[side][int(n)]
            coeff=np.stack([a @ sol[ids[0::2]],a @ sol[ids[1::2]]])
            coeff[:,side*nch+2*k:side*nch+2*k+2]-=incidents[side,int(n)]
            result[side*nch+2*k:side*nch+2*k+2]=np.linalg.solve(u,coeff)
    return result[:nch,:nch],result[nch:,nch:],result[nch:,:nch],result[:nch,nch:]

def free_reflection(mat,omega,orders,pitch,left):
    r=np.zeros((2*len(orders),)*2,complex)
    for k,n in enumerate(orders):
        q=2*np.pi*n/pitch
        _,tup=wave_matrices(mat,omega,q,1,1)
        _,tdn=wave_matrices(mat,omega,q,-1,1)
        r[2*k:2*k+2,2*k:2*k+2]=-np.linalg.solve(tup,tdn) if left else -np.linalg.solve(tdn,tup)
    return r

def surface_response(s,metal_left,metal_right,omega,orders,pitch,mode):
    rl,rr,tl,tr=s;m=len(rl);eye=np.eye(m,dtype=complex)
    fl=free_reflection(metal_left,omega,orders,pitch,True)
    fr=free_reflection(metal_right,omega,orders,pitch,False)
    back=np.linalg.solve(eye-fr @ rr,fr @ tl)
    drive=np.zeros(m,complex);n0=int(np.flatnonzero(np.asarray(orders)==0)[0]);drive[2*n0+mode]=1.
    a=np.linalg.solve(eye-fl @ (rl+tr @ back),drive)
    b=tl @ a+rr @ back @ a
    inc=back @ a
    up,_=wave_matrices(metal_right,omega,0,1,1)
    dn,_=wave_matrices(metal_right,omega,0,-1,1)
    u=up @ b[2*n0:2*n0+2]+dn @ inc[2*n0:2*n0+2]
    return u[1 if mode==0 else 0]

def scalar_segment(omega,rho,c,length):
    k=omega/c;z=rho*c;v=k*length
    return np.array([[np.cos(v),np.sin(v)/(omega*z)],[-omega*z*np.sin(v),np.cos(v)]],complex)

def ramp_scattering(omega,left,right,height,key,n=120):
    p=np.eye(2,dtype=complex)
    for a in (np.arange(n)+.5)/n:
        rho=(1-a)*left['rho']+a*right['rho'];c=np.sqrt((1-a)*left[key]**2+a*right[key]**2)
        p=scalar_segment(omega,rho,c,height/n) @ p
    zl=left['rho']*left[key];zr=right['rho']*right[key]
    l=np.array([[1,1],[1j*omega*zl,-1j*omega*zl]])
    r=np.array([[1,1],[1j*omega*zr,-1j*omega*zr]])
    w=np.linalg.solve(r,p @ l)
    return tuple(np.array([[x]]) for x in [-w[1,0]/w[1,1],w[0,1]/w[1,1],w[0,0]-w[0,1]*w[1,0]/w[1,1],1/w[1,1]])

def scalar_propagation(omega,mat,length,key):
    t=np.array([[np.exp(1j*omega*length/mat[key])]]);z=np.zeros((1,1),complex)
    return z,z,t,t

def dds_ramp_response(f,metal,gouge,lengths,heights,mode,eta):
    omega=2*np.pi*f+1j*eta;key='cp' if mode==0 else 'cs'
    mats=[metal[0],gouge[0],metal[1],gouge[1],metal[2]]
    s=scalar_propagation(omega,mats[0],lengths[0],key)
    for i in range(4):
        s=cascade(s,ramp_scattering(omega,mats[i],mats[i+1],heights[i],key))
        s=cascade(s,scalar_propagation(omega,mats[i+1],lengths[i+1],key))
    rl,rr,tl,tr=(x[0,0] for x in s)
    back=tl/(1-rr)
    incident=1/(1-rl-tr*back)
    return 2*back*incident
