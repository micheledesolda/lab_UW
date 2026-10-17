"""Lossless 2-D P/SV scattering by a periodic, bonded triangular interface.

P1 conforming finite elements, fitted 45-degree interface, periodic x.
Exact continuum half-space Dirichlet-to-Neumann operators in Fourier space.
No damping, PML, source fitting or changes to the user's inversion software.
Numerical units: mm, microseconds, g/cm^3; stiffness values are GPa.
Time convention exp(-i omega t). Normal incidence, infinite periodic array.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parent
MATERIALS = {
    "steel": {"rho": 7.85, "cp": 5.9, "cs": 3.25},
    "titanium": {"rho": 4.5, "cp": 6.1, "cs": 3.1},
}

def moduli(mat):
    g = mat["rho"] * mat["cs"]**2
    lam = mat["rho"] * mat["cp"]**2 - 2*g
    return lam, g

def wave_matrices(mat, omega, q, direction, normal):
    """Columns are P and SV displacement, and outward boundary traction."""
    kp, ks = omega / mat["cp"], omega / mat["cs"]
    zp = direction * np.sqrt(complex(kp*kp-q*q))
    zs = direction * np.sqrt(complex(ks*ks-q*q))
    u = np.array([[q/kp, zs/ks], [zp/kp, -q/ks]], complex)
    lam, g = moduli(mat)
    t = np.empty((2, 2), complex)
    for j, z in enumerate([zp, zs]):
        ux, uz = u[:, j]
        t[:, j] = normal * 1j * np.array([
            g * (z*ux+q*uz), lam*q*ux+(lam+2*g)*z*uz])
    return u, t

class Cell:
    def __init__(self, nx, metal, gouge, geometry="teeth", height=.5, pitch=1., aliases=2,
                 nz=None, metal_classifier=None, mirror_upper=False):
        assert nx % 4 == 0
        self.nx, self.nz = nx, nx//2 if nz is None else int(nz)
        self.height, self.pitch = height, pitch
        self.metal, self.gouge = metal, gouge
        dx, dz = pitch/nx, height/self.nz
        nodes = np.array([[i*dx-pitch/2, j*dz]
                          for j in range(self.nz+1) for i in range(nx+1)])
        triangles = []
        for j in range(self.nz):
            for i in range(nx):
                a = j*(nx+1)+i
                b, c, d = a+1, a+nx+1, a+nx+2
                if (i < nx//2) != (mirror_upper and j >= self.nz//2):
                    triangles += [[a, b, d], [a, d, c]]
                else:
                    triangles += [[a, b, c], [b, d, c]]
        triangles = np.array(triangles)
        xy = nodes[triangles]
        centers = xy.mean(axis=1)
        if metal_classifier is not None:
            ismetal = np.asarray(metal_classifier(centers), bool)
        elif geometry == "teeth":
            ismetal = centers[:, 1] < height*(1-2*np.abs(centers[:, 0])/pitch)
        elif geometry == "flat":
            ismetal = centers[:, 1] < height/2
        elif geometry == "homogeneous":
            ismetal = np.ones(len(triangles), bool)
        else:
            raise ValueError(geometry)
        self.geometry = geometry
        area2 = ((xy[:, 1, 0]-xy[:, 0, 0])*(xy[:, 2, 1]-xy[:, 0, 1])
                 -(xy[:, 2, 0]-xy[:, 0, 0])*(xy[:, 1, 1]-xy[:, 0, 1]))
        area = area2/2
        grad = np.stack([
            np.stack([xy[:, 1, 1]-xy[:, 2, 1], xy[:, 2, 0]-xy[:, 1, 0]], axis=1),
            np.stack([xy[:, 2, 1]-xy[:, 0, 1], xy[:, 0, 0]-xy[:, 2, 0]], axis=1),
            np.stack([xy[:, 0, 1]-xy[:, 1, 1], xy[:, 1, 0]-xy[:, 0, 0]], axis=1),
        ], axis=1) / area2[:, None, None]
        ne = len(triangles)
        B = np.zeros((ne, 3, 6))
        B[:, 0, 0::2] = grad[:, :, 0]
        B[:, 1, 1::2] = grad[:, :, 1]
        B[:, 2, 0::2] = grad[:, :, 1]
        B[:, 2, 1::2] = grad[:, :, 0]
        lm, gm = moduli(metal)
        lg, gg = moduli(gouge)
        lam = np.where(ismetal, lm, lg)
        g = np.where(ismetal, gm, gg)
        D = np.zeros((ne, 3, 3))
        D[:, 0, 0] = D[:, 1, 1] = lam+2*g
        D[:, 0, 1] = D[:, 1, 0] = lam
        D[:, 2, 2] = g
        ke = np.einsum('eai,eab,ebj,e->eij', B, D, B, area)
        rho = np.where(ismetal, metal["rho"], gouge["rho"])
        mass0 = (np.ones((3, 3))+np.eye(3))/12
        me = np.zeros((ne, 6, 6))
        me[:, 0::2, 0::2] = rho[:, None, None]*area[:, None, None]*mass0
        me[:, 1::2, 1::2] = me[:, 0::2, 0::2]
        # Identify right and left periodic nodes, sharing both displacement DOFs.
        full_ids = np.arange(len(nodes))
        periodic_ids = (full_ids//(nx+1))*nx + full_ids % (nx+1) % nx
        tnodes = periodic_ids[triangles]
        edof = np.stack([2*tnodes, 2*tnodes+1], axis=2).reshape(ne, 6)
        row = np.repeat(edof, 6, axis=1).ravel()
        col = np.tile(edof, (1, 6)).ravel()
        self.ndof = 2*nx*(self.nz+1)
        self.K = coo_matrix((ke.ravel(), (row, col)), shape=(self.ndof,)*2).tocsc()
        self.M = coo_matrix((me.ravel(), (row, col)), shape=(self.ndof,)*2).tocsc()
        self.boundary = [np.arange(2*nx), np.arange(self.ndof-2*nx, self.ndof)]
        self.metal_fraction = float(area[ismetal].sum()/area.sum())
        # Fourier coefficients of exact periodic P1 hat basis functions.
        # Modes through Nyquist plus their aliases resolve the boundary trace.
        self.orders = np.arange(-aliases*nx, aliases*nx+1)
        self.Q = np.exp(-2j*np.pi*self.orders[:, None]*np.arange(nx)[None, :]/nx)
        self.Q *= (np.sinc(self.orders/nx)**2/nx)[:, None]

    def boundary_operator(self, mat, omega, direction):
        nx = self.nx
        boundary_matrix = np.zeros((2*nx, 2*nx), complex)
        basis = []
        for n, a in zip(self.orders, self.Q):
            if abs(n) % nx == 0 and n != 0:
                continue
            q = 2*np.pi*n/self.pitch
            u, t = wave_matrices(mat, omega, q, direction, direction)
            dt = t @ np.linalg.inv(u)
            weight = self.pitch*np.outer(a.conj(), a)
            for i in range(2):
                for j in range(2):
                    boundary_matrix[i::2, j::2] += dt[i, j]*weight
            basis.append((int(n), a, u, t))
        return boundary_matrix, basis

    def solve(self, frequency, incidence="metal"):
        omega = 2*np.pi*frequency
        topmat = self.metal if self.geometry == "homogeneous" else self.gouge
        bottom, bmodes = self.boundary_operator(self.metal, omega, -1)
        top, tmodes = self.boundary_operator(topmat, omega, +1)
        rows, cols, data = [], [], []
        for ids, op in zip(self.boundary, [bottom, top]):
            rows.append(np.repeat(ids, len(ids)))
            cols.append(np.tile(ids, len(ids)))
            data.append(-op.ravel())
        A = self.K-omega**2*self.M + coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.ndof,)*2).tocsc()
        if incidence == "metal":
            incmat, incdir, incnormal, incids = self.metal, +1, -1, self.boundary[0]
            sides = [("R", bmodes, self.boundary[0], self.metal),
                     ("T", tmodes, self.boundary[1], topmat)]
        elif incidence == "gouge":
            incmat, incdir, incnormal, incids = topmat, -1, +1, self.boundary[1]
            sides = [("R", tmodes, self.boundary[1], topmat),
                     ("T", bmodes, self.boundary[0], self.metal)]
        else:
            raise ValueError(incidence)
        uinc, tinc = wave_matrices(incmat, omega, 0., incdir, incnormal)
        uout, tout = wave_matrices(incmat, omega, 0., -incdir, incnormal)
        residual = tinc-tout @ np.linalg.solve(uout, uinc)
        rhs = np.zeros((self.ndof, 2), complex)
        for j in range(2):
            rhs[incids[0::2], j] = self.pitch/self.nx*residual[0, j]
            rhs[incids[1::2], j] = self.pitch/self.nx*residual[1, j]
        sol = splu(A).solve(rhs)
        results = []
        for incident, name in enumerate(["P", "SV"]):
            incflux = .5*incmat["rho"]*omega**2*incmat[
                "cp" if incident == 0 else "cs"]
            modes = []
            for side, bas, ids, mat in sides:
                ux, uz = sol[ids[0::2], incident], sol[ids[1::2], incident]
                for n, a, u, t in bas:
                    coeff = np.array([a @ ux, a @ uz])
                    if side == "R" and n == 0:
                        coeff -= uinc[:, incident]
                    amps = np.linalg.solve(u, coeff)
                    q = 2*np.pi*n/self.pitch
                    for mode, speed, amp, j in zip(["P", "SV"], [mat["cp"], mat["cs"]], amps, [0, 1]):
                        if abs(q) < omega/speed*(1-1e-10):
                            e = .5*omega*np.imag(np.vdot(u[:, j]*amp, t[:, j]*amp))/incflux
                            modes.append({"side": side, "mode": mode, "order": n,
                                          "energy": float(e), "amplitude_re": float(amp.real),
                                          "amplitude_im": float(amp.imag)})
            R = sum(m["energy"] for m in modes if m["side"] == "R")
            T = sum(m["energy"] for m in modes if m["side"] == "T")
            direct = next(m for m in modes if m["side"] == "T" and m["mode"] == name and m["order"] == 0)
            results.append({"incident": name, "incidence_from": incidence, "f_MHz": frequency,
                            "T_direct": direct["energy"], "T_total": T, "R_total": R,
                            "energy_error": R+T-1,
                            "T_converted": sum(m["energy"] for m in modes if m["side"] == "T" and m["mode"] != name),
                            "T_lateral": sum(m["energy"] for m in modes if m["side"] == "T" and m["order"] != 0),
                            "amplitude_re": direct["amplitude_re"],
                            "amplitude_im": direct["amplitude_im"], "modes": modes})
        return results

def run(args):
    frequencies = [float(x) for x in args.frequencies.split(',')]
    gouge = {"rho": args.rho, "cp": 2., "cs": 1.267}
    mat = MATERIALS[args.metal]
    cell = Cell(args.nx, mat, gouge, args.geometry, aliases=args.aliases)
    results = []
    for f in frequencies:
        tick = time.monotonic()
        r = cell.solve(f, args.incidence)
        for item in r:
            item.update(metal=args.metal, rho_g=args.rho, nx=args.nx, geometry=args.geometry)
            print(json.dumps({k: v for k, v in item.items() if k != "modes"}), flush=True)
        results += r
        print(f"frequency {f:g}: {time.monotonic()-tick:.2f} s", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"parameters": {"metal": mat, "gouge": gouge,
                         "height_mm": .5, "pitch_mm": 1., "nx": args.nx,
                         "geometry": args.geometry, "aliases": args.aliases,
                         "metal_fraction": cell.metal_fraction},
                         "results": results}, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--metal', choices=list(MATERIALS), default='steel')
    parser.add_argument('--rho', type=float, default=3.2)
    parser.add_argument('--aliases', type=int, default=2)
    parser.add_argument('--nx', type=int, default=80)
    parser.add_argument('--geometry', choices=['teeth', 'flat', 'homogeneous'], default='teeth')
    parser.add_argument('--incidence', choices=['metal', 'gouge'], default='metal')
    parser.add_argument('--frequencies', default='1.2,4.3,5')
    parser.add_argument('--output', default=str(ROOT/'results.json'))
    run(parser.parse_args())
