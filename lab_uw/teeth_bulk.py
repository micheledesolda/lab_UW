"""Uniform complex bulk propagation with explicit matching to teeth ports."""
import numpy as np
from .teeth_bank import ORDERS,modal_basis,cascade,propagation
from .teeth_formula.reference_core.fast_solver import waves

def planar_interface(left,right,omega,mode=None):
    up,tp=waves(left,omega,ORDERS,direction=1,normal=1)
    um,tm=waves(left,omega,ORDERS,direction=-1,normal=1)
    vp,sp=waves(right,omega,ORDERS,direction=1,normal=1)
    vm,sm=waves(right,omega,ORDERS,direction=-1,normal=1)
    a=np.concatenate([np.concatenate([um,-vp],axis=2),np.concatenate([tm,-sp],axis=2)],axis=1)
    b=np.concatenate([np.concatenate([-up,vm],axis=2),np.concatenate([-tp,sm],axis=2)],axis=1)
    sol=np.linalg.solve(a,b);blocks=[]
    q=modal_basis(ORDERS,1 if mode==0 else -1) if mode is not None else None
    for rs,cs in [(slice(0,2),slice(0,2)),(slice(2,4),slice(2,4)),
                  (slice(2,4),slice(0,2)),(slice(0,2),slice(2,4))]:
        block=np.zeros((50,50),complex)
        for n in range(25):block[2*n:2*n+2,2*n:2*n+2]=sol[n,rs,cs]
        blocks.append(q.T@block@q if q is not None else block)
    return tuple(blocks)

def bulk_segment(port_material,bulk_material,omega,length,mode=None):
    return cascade(cascade(planar_interface(port_material,bulk_material,omega,mode),
        propagation(bulk_material,omega,length,mode)),planar_interface(bulk_material,port_material,omega,mode))
