"""Exact Fourier halfspace ports and analytic material derivatives.

Derivatives are ordered (rho[g/cm3], cS[mm/us], Poisson).
"""
import numpy as np

def material_differentials(mat,active):
    zero=np.zeros(3)
    if not active:return zero,zero,zero,zero,zero
    rho,cs,cp,nu=(mat[k] for k in ['rho','cs','cp','nu'])
    dr=np.array([1.,0.,0.]);ds=np.array([0.,1.,0.])
    dp=np.array([0.,cp/cs,cp/(2*(1-nu)*(1-2*nu))])
    dm=dr*cs**2+2*rho*cs*ds
    dl=dr*cp**2+2*rho*cp*dp-2*dm
    return dr,ds,dp,dm,dl

def wave_jac(mat,omega,q,direction,normal,active=False):
    q=np.asarray(q);rho,cs,cp=(mat[k] for k in ['rho','cs','cp'])
    dr,ds,dp,dm,dl=material_differentials(mat,active)
    kp,ks=omega/cp,omega/cs;dkp=-kp*dp/cp;dks=-ks*ds/cs
    zp=direction*np.sqrt(kp*kp-q*q+0j);zs=direction*np.sqrt(ks*ks-q*q+0j)
    dzp=dkp[:,None]*kp/zp[None,:];dzs=dks[:,None]*ks/zs[None,:]
    u=np.empty((len(q),2,2),complex);du=np.empty((3,len(q),2,2),complex)
    u[:,0,0]=q/kp;u[:,1,0]=zp/kp;u[:,0,1]=zs/ks;u[:,1,1]=-q/ks
    du[:,:,0,0]=-u[None,:,0,0]*dkp[:,None]/kp
    du[:,:,1,0]=dzp/kp-u[None,:,1,0]*dkp[:,None]/kp
    du[:,:,0,1]=dzs/ks-u[None,:,0,1]*dks[:,None]/ks
    du[:,:,1,1]=-u[None,:,1,1]*dks[:,None]/ks
    mu=rho*cs**2;lam=rho*cp**2-2*mu
    z=np.column_stack([zp,zs]);dz=np.stack([dzp,dzs],axis=-1)
    t=np.empty_like(u);dt=np.empty_like(du)
    for mode in [0,1]:
        ux,uz=u[:,0,mode],u[:,1,mode];zz=z[:,mode]
        t[:,0,mode]=normal*1j*mu*(zz*ux+q*uz)
        t[:,1,mode]=normal*1j*(lam*q*ux+(lam+2*mu)*zz*uz)
        for p in range(3):
            dx,dzdisp=du[p,:,0,mode],du[p,:,1,mode];dzz=dz[p,:,mode]
            dt[p,:,0,mode]=normal*1j*(dm[p]*(zz*ux+q*uz)+mu*(dzz*ux+zz*dx+q*dzdisp))
            dt[p,:,1,mode]=normal*1j*(dl[p]*q*ux+lam*q*dx+(dl[p]+2*dm[p])*zz*uz+(lam+2*mu)*(dzz*uz+zz*dzdisp))
    return u,t,du,dt

def dtn_jac(mat,omega,q,direction,active=False):
    q=np.asarray(q);rho,cs,cp=(mat[k] for k in ['rho','cs','cp'])
    dr,ds,dp,dm,dl=material_differentials(mat,active)
    kp,ks=omega/cp,omega/cs;dkp=-kp*dp/cp;dks=-ks*ds/cs
    zp=direction*np.sqrt(kp*kp-q*q+0j);zs=direction*np.sqrt(ks*ks-q*q+0j)
    dzp=dkp[:,None]*kp/zp[None,:];dzs=dks[:,None]*ks/zs[None,:]
    den=q*q+zp*zs;dden=dzp*zs[None,:]+zp[None,:]*dzs
    nz=q!=0;qq=q[nz]**2
    numerator=qq*(kp*kp+ks*ks)-kp*kp*ks*ks
    denominator=qq-zp[nz]*zs[nz]
    den[nz]=numerator/denominator
    dn=qq[None,:]*(2*kp*dkp[:,None]+2*ks*dks[:,None])-2*kp*dkp[:,None]*ks**2-2*ks*dks[:,None]*kp**2
    dd=-dzp[:,nz]*zs[None,nz]-zp[None,nz]*dzs[:,nz]
    dden[:,nz]=(dn-den[None,nz]*dd)/denominator[None,:]
    ratio=rho*omega**2/den;d_ratio=omega**2*dr[:,None]/den[None,:]-ratio[None,:]*dden/den[None,:]
    mu=rho*cs**2;out=np.empty((len(q),2,2),complex);dout=np.empty((3,len(q),2,2),complex)
    out[:,0,0]=direction*1j*ratio*zp;out[:,1,1]=direction*1j*ratio*zs
    out[:,0,1]=direction*1j*q*(2*mu-ratio);out[:,1,0]=-out[:,0,1]
    dout[:,:,0,0]=direction*1j*(d_ratio*zp[None,:]+ratio[None,:]*dzp)
    dout[:,:,1,1]=direction*1j*(d_ratio*zs[None,:]+ratio[None,:]*dzs)
    dout[:,:,0,1]=direction*1j*q[None,:]*(2*dm[:,None]-d_ratio);dout[:,:,1,0]=-dout[:,:,0,1]
    return out,dout

def port(model,pi,mat,omega,direction,active=False):
    """Return boundary DtN O, drive B, output C, incident D and derivatives."""
    cell=model.cell;ft=model.fourier_traces[pi]
    qr=2*np.pi*cell.orders;dt,ddt=dtn_jac(mat,omega,qr,direction,active)
    (n,i,j,brow,bcol),weight=model.port_projectors[pi]
    op=np.zeros((ft.shape[2],ft.shape[2]),complex)
    np.add.at(op,(brow,bcol),weight*dt[n,i,j])
    dop=np.zeros((3,*op.shape),complex)
    if active:
        for p in range(3):np.add.at(dop[p],(brow,bcol),weight*ddt[p,n,i,j])
    ids=np.searchsorted(cell.orders,model.retained_orders);fr=ft[ids]
    qo=2*np.pi*model.retained_orders
    u,t,du,dtout=wave_jac(mat,omega,qo,direction,direction,active)
    inc,it,dinc,dit=wave_jac(mat,omega,qo,-direction,direction,active)
    dtn,d_dtn=dtn_jac(mat,omega,qo,direction,active)
    res=it-dtn@inc
    b=np.einsum('nib,nij->bnj',fr.conj(),res).reshape(fr.shape[2],-1)
    cin=np.linalg.solve(u,fr);di=np.linalg.solve(u,inc)
    qb=model.qboth[pi][:50,:25]
    bm=b@qb;cm=qb.T@cin.reshape(50,-1)
    dim=qb.T@(di@qb.reshape(25,2,25)).reshape(50,25)
    if not active:
        return op,bm,cm,dim,np.zeros((3,*op.shape),complex),np.zeros((3,*bm.shape),complex),np.zeros((3,*cm.shape),complex),np.zeros((3,*dim.shape),complex)
    db=[];dc=[];ddi=[]
    for p in range(3):
        dres=dit[p]-d_dtn[p]@inc-dtn@dinc[p]
        db.append(np.einsum('nib,nij->bnj',fr.conj(),dres).reshape(fr.shape[2],-1)@qb)
        dc.append(qb.T@(-np.linalg.solve(u,du[p]@cin)).reshape(50,-1))
        ddi.append(qb.T@(np.linalg.solve(u,dinc[p]-du[p]@di)@qb.reshape(25,2,25)).reshape(50,25))
    return op,bm,cm,dim,dop,np.stack(db),np.stack(dc),np.stack(ddi)
