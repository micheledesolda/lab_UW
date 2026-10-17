"""Read-only access to the original 200-material P2 teeth bank.

No FEM, matrix regeneration, material interpolation, or FD quotient is used.
Numerical units: mm, microseconds, g/cm3. exp(-i*omega*t), omega=2*pi*f+i*eta.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
from functools import lru_cache
import numpy as np

ORDERS = np.arange(-12, 13)
FREQUENCIES = np.arange(321, dtype=float) * .025
ETA = .02
METALS = {'steel': dict(rho=7.85, cp=5.9, cs=3.25),
          'titanium': dict(rho=4.5, cp=6.1, cs=3.1)}

def materials():
    rows = []
    for rho in np.linspace(1.5, 3., 5):
        for cs in np.linspace(.8, 1.8, 10):
            for nu in np.linspace(.15, .45, 4):
                cp = cs*np.sqrt(2*(1-nu)/(1-2*nu))
                rows.append(dict(index=len(rows), rho=float(rho), cs=float(cs),
                                 cp=float(cp), nu=float(nu), G_GPa=float(rho*cs*cs),
                                 K_GPa=float(rho*(cp*cp-4*cs*cs/3))))
    return rows

def modal_basis(orders, parity):
    nmax = int(max(orders)); size = 2*len(orders); cols = []
    v = np.zeros(size); v[2*nmax+(0 if parity == 1 else 1)] = 1.; cols.append(v)
    for n in range(1, nmax+1):
        for mode in (0, 1):
            v = np.zeros(size); v[2*(nmax+n)+mode] = 2**-.5
            v[2*(nmax-n)+mode] = parity*(1 if mode == 0 else -1)*2**-.5
            cols.append(v)
    return np.column_stack(cols)

def split_scattering(s):
    n = s.shape[0]//2
    return s[:n, :n], s[n:, n:], s[n:, :n], s[:n, n:]

def reconstruct(sectors):
    result = np.zeros((100, 100), complex)
    for sector, parity in zip(sectors, (1, -1)):
        q = modal_basis(ORDERS, parity); both = np.zeros((100, 50))
        both[:50, :25] = q; both[50:, 25:] = q
        result += both @ sector @ both.T
    return split_scattering(result)

def cascade(a, b):
    al, ar, at, ab = a; bl, br, bt, bb = b
    x = np.linalg.solve(np.eye(len(al))-bl@ar, np.column_stack((bl@at, bb)))
    xx, xy = np.split(x, 2, axis=1)
    return al+ab@xx, br+bt@ar@xy, bt@(at+ar@xx), ab@xy

def mirrored(s, signs):
    l, r, t, b = s
    def flip(a): return signs[:, None]*a*signs[None, :]
    return flip(r), flip(l), flip(b), flip(t)

def waves(mat, omega, q, direction):
    kp, ks = omega/mat['cp'], omega/mat['cs']
    zp = direction*np.sqrt(complex(kp*kp-q*q)); zs = direction*np.sqrt(complex(ks*ks-q*q))
    u = np.array([[q/kp, zs/ks], [zp/kp, -q/ks]], complex)
    g = mat['rho']*mat['cs']**2; lam = mat['rho']*mat['cp']**2-2*g
    t = np.empty((2, 2), complex)
    for j, z in enumerate((zp, zs)):
        ux, uz = u[:, j]
        t[:, j] = 1j*np.array([g*(z*ux+q*uz), lam*q*ux+(lam+2*g)*z*uz])
    return u, t

def propagation(mat, omega, length, mode=None):
    if length < 0: raise ValueError('Negative homogeneous-layer length')
    if mode is None:
        channels = [(int(n), j) for n in ORDERS for j in (0, 1)]
    else:
        channels = [(0, mode)]+[(n, j) for n in range(1, 13) for j in (0, 1)]
    phase = [np.exp(1j*np.sqrt(complex((omega/mat['cp' if j == 0 else 'cs'])**2
                                      -(2*np.pi*n)**2))*length) for n, j in channels]
    t = np.diag(phase); z = np.zeros_like(t)
    return z, z, t, t

def free_reflection(mat, omega, mode=None, left=False):
    r = np.zeros((50, 50), complex)
    for i, n in enumerate(ORDERS):
        _, up = waves(mat, omega, 2*np.pi*n, 1)
        _, dn = waves(mat, omega, 2*np.pi*n, -1)
        r[2*i:2*i+2, 2*i:2*i+2] = -np.linalg.solve(up, dn) if left else -np.linalg.solve(dn, up)
    if mode is not None:
        q = modal_basis(ORDERS, 1 if mode == 0 else -1)
        r = q.T@r@q
    return r

@lru_cache(maxsize=16384)
def steel_free_reflection(omega, mode, left):
    # Shared across candidates; returned matrices are read only in composition.
    r = free_reflection(METALS['steel'], omega, mode, left)
    r.setflags(write=False)
    return r

def surface_response(s, omega, mode, reduced=True):
    m = METALS['steel']; rl, rr, tl, tr = s
    chosen_mode = mode if reduced else None
    fl = steel_free_reflection(omega, chosen_mode, True)
    fr = steel_free_reflection(omega, chosen_mode, False)
    eye = np.eye(len(rl)); back = np.linalg.solve(eye-fr@rr, fr@tl)
    drive = np.zeros(len(rl), complex); drive[0 if reduced else 24+mode] = 1.
    a = np.linalg.solve(eye-fl@(rl+tr@back), drive)
    b = tl@a+rr@back@a; incoming = back@a
    if reduced:
        # n=0 displacement at the receiver, P vertical or SV horizontal.
        return b[0]-incoming[0]
    up, _ = waves(m, omega, 0., 1); dn, _ = waves(m, omega, 0., -1)
    return (up@b[24:26]+dn@incoming[24:26])[1 if mode == 0 else 0]

class P2MatrixBank:
    def __init__(self, directory, *, frequency_interpolation='linear', cache_size=2):
        self.directory = Path(directory).expanduser().resolve()
        self.interpolation = frequency_interpolation
        if self.interpolation not in ('linear', 'none'):
            raise ValueError("frequency_interpolation must be 'linear' or 'none'")
        self.cache_size = max(2, int(cache_size)); self.cache = OrderedDict(); self.rows = materials()
        self.fingerprints = {}; self.read_count = 0

    def path(self, index, metal):
        if not 0 <= int(index) < 200 or int(index) != index: raise ValueError('Invalid material index')
        if metal not in METALS: raise ValueError('Only steel and titanium bank ports exist')
        return self.directory/f'material_{int(index):03d}_{metal}.npz'

    def preflight(self, indices=range(200)):
        missing = [str(self.path(i, m)) for i in indices for m in METALS if not self.path(i, m).is_file()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} bank families missing; extract the original matrix archives into '
                                    f'{self.directory}. First missing: {missing[0]}. No matrices are generated.')

    def _read(self, index, metal):
        key = int(index), metal
        if key in self.cache: self.cache.move_to_end(key); return self.cache[key]
        path = self.path(index, metal)
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d['metadata'])); sectors = d['sectors'].copy()
            freq = d['frequency_mhz'].copy(); orders = d['orders'].copy()
        expected = self.rows[int(index)]
        checks = [sectors.shape == (321, 2, 50, 50), np.array_equal(orders, ORDERS),
                  freq.shape == (321,) and np.allclose(freq, FREQUENCIES, rtol=0, atol=1e-10),
                  meta.get('metal_name') == metal, meta.get('material', {}).get('index') == int(index),
                  meta.get('intrinsic_loss') == 0., meta.get('finite_element', '').startswith('P2'),
                  meta.get('height_mm') == .5, meta.get('pitch_mm') == 1.,
                  meta.get('laplace_eta_per_us') == ETA]
        checks += [np.isclose(meta['material'].get(k, np.nan), expected[k], rtol=1e-12, atol=1e-12)
                   for k in ('rho', 'cp', 'cs', 'nu')]
        checks += [np.isclose(meta['metal'].get(k, np.nan), v, rtol=1e-12, atol=1e-12)
                   for k, v in METALS[metal].items()]
        if not all(checks) or not np.all(np.isfinite(sectors)):
            raise ValueError(f'Bank metadata/format mismatch: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.fingerprints[path.name] = digest; self.read_count += 1
        self.cache[key] = (sectors, meta)
        while len(self.cache) > self.cache_size: self.cache.popitem(last=False)
        return self.cache[key]

    def sectors(self, index, metal, frequency):
        data, _ = self._read(index, metal); f = float(frequency)
        if not np.isfinite(f) or not 0 <= f <= 8.+1e-12: raise ValueError('Frequency outside [0,8] MHz')
        x = min(320., f/.025); j = int(round(x))
        if abs(x-j) <= 1e-8: return data[j]
        if self.interpolation == 'none': raise ValueError('FFT frequency is not a bank node; explicitly enable linear interpolation')
        lo = int(np.floor(x)); w = x-lo
        return (1-w)*data[lo]+w*data[lo+1]

    def response(self, index, frequency, *, lengths_mm, mode, offsets_mm=(0., 0., 0., 0.), reduced=None, bulk_material=None):
        lengths = np.asarray(lengths_mm, float); offsets = np.asarray(offsets_mm, float)
        if lengths.shape != (5,) or np.any(lengths < 0) or not np.all(np.isfinite(lengths)):
            raise ValueError('Five finite nonnegative bulk lengths required')
        if offsets.shape != (4,) or not np.all(np.isfinite(offsets)) or mode not in (0, 1):
            raise ValueError('Four finite lateral offsets and mode P=0/SV=1 required')
        aligned = np.allclose(offsets-offsets[0], 0., atol=1e-12, rtol=0)
        # Near the static limit the condensed Schur solves lose digits under
        # parity projection. Retain the original full-order assembly there.
        reduced = aligned and frequency > .025 if reduced is None else bool(reduced)
        if reduced and not aligned: raise ValueError('Parity reduction requires aligned teeth')
        steel = self.sectors(index, 'steel', frequency); titanium = self.sectors(index, 'titanium', frequency)
        if reduced:
            a, b = split_scattering(steel[mode]), split_scattering(titanium[mode])
            signs = np.r_[1. if mode == 0 else -1., np.tile([1., -1.], 12)]
        else:
            a, b = reconstruct(steel), reconstruct(titanium); signs = np.tile([1., -1.], 25)
        interfaces = [a, mirrored(b, signs), b, mirrored(a, signs)]
        if not reduced:
            for j, offset in enumerate(offsets):
                phase = np.repeat(np.exp(-2j*np.pi*ORDERS*offset), 2)
                interfaces[j] = tuple(phase[:, None]*m*phase.conj()[None, :] for m in interfaces[j])
        omega = 2*np.pi*frequency+1j*ETA
        g = self.rows[int(index)]
        mats = [METALS['steel'], g, METALS['titanium'], g, METALS['steel']]
        s = propagation(mats[0], omega, lengths[0], mode if reduced else None)
        for j in range(4):
            if bulk_material is not None and j+1 in (1,3):
                from .teeth_bulk import bulk_segment
                segment=bulk_segment(g,bulk_material,omega,lengths[j+1],mode if reduced else None)
            else:segment=propagation(mats[j+1], omega, lengths[j+1], mode if reduced else None)
            s = cascade(cascade(s, interfaces[j]), segment)
        return surface_response(s, omega, mode, reduced)
