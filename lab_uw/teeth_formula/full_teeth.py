"""Exact evaluation of the material-affine P2 matrix formula, no reduction.

This is a slower reference/refinement backend. It uses the same geometry and
ports as the original bank with a fixed mesh, avoiding grid interpolation and
the reduced-model error. It does not remove FEM discretization error.
"""
from copy import copy
from functools import lru_cache
import numpy as np
from .continuous_teeth import gouge,TabulatedTeeth
from .build_model import affine_operators
from ..teeth_bank import METALS,ORDERS
from .reference_core.fast_solver import reconstruct

class FullTeeth:
    def __init__(self,metal,nx=80):
        if metal not in METALS:raise ValueError('steel or titanium required')
        self.base,self.operators=affine_operators(nx,metal);self.metal=metal;self.nx=nx
    def compile(self,rho,cs,nu,*,eta=.02):
        g=gouge(rho,cs,nu);mu=rho*cs*cs;lam=2*mu*nu/(1-2*nu)
        if eta<0 or not np.isfinite(eta):raise ValueError('eta must be finite and nonnegative')
        fast=copy(self.base);fast.cell=copy(self.base.cell);fast.cell.gouge=g
        fast.K=[(a['K0']+mu*a['Kg']+lam*a['Kl']).tocsc() for a in self.operators]
        fast.M=[(a['M0']+rho*a['Mr']).tocsc() for a in self.operators]
        return FullCompiled(fast,g,eta)

class FullCompiled:
    def __init__(self,fast,material,eta):
        self.fast=fast;self.material=material;self.eta=eta
        self.sector_at=lru_cache(maxsize=512)(self._sector_at)
    def _sector_at(self,frequency):
        f=float(frequency)
        if not np.isfinite(f) or f<0 or f>8 or (f==0 and self.eta==0):raise ValueError('Invalid frequency/contour')
        s=self.fast.solve(f,self.eta);s.setflags(write=False);return s
    def sectors(self,frequency):return np.stack([self.sector_at(f) for f in np.asarray(frequency).reshape(-1)])
    def scattering(self,frequency):return reconstruct(self.sector_at(frequency),ORDERS)
    def transmission(self,frequency):return self.scattering(frequency)[2]
    def tabulate(self,frequencies=None):return TabulatedTeeth(self,np.arange(321)*.025 if frequencies is None else frequencies)
