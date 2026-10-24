"""Material coordinates for the existing fixed-density local optimizer.

The forward/receiver API keeps velocity and kinematic damping. Only the
optimizer coordinates, gradient chain rule and physical bounds are changed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import numpy as np

MODES = ('velocity_damping', 'modulus_viscosity')
REVISION = 4


def parameterization(policy):
    mode = policy.get('parameterization', 'velocity_damping')
    if mode not in MODES:
        raise ValueError('local_optimizer.parameterization must be velocity_damping or modulus_viscosity')
    if mode == 'modulus_viscosity' and not policy.get('enabled', True):
        raise ValueError('modulus_viscosity requires local_optimizer.enabled=True')
    return mode


def canonical_policy(policy):
    """Retain historical digests when the default coordinates are selected."""
    result = deepcopy(policy)
    if parameterization(result) == 'velocity_damping':
        result.pop('parameterization', None)
    return result


def policy_changes(previous, current, defaults):
    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise ValueError('Cannot migrate an unknown material optimizer policy.')
    old = {**previous, 'parameterization': parameterization(previous)}
    new = {**current, 'parameterization': parameterization(current)}
    if set(old) != set(defaults) or set(new) != set(defaults):
        raise ValueError('Cannot migrate an unknown material optimizer policy.')
    if not old['enabled'] or not new['enabled']:
        raise ValueError('Changing local_optimizer.enabled requires a separate result sequence.')
    return {key: {'old': old[key], 'new': new[key]} for key in defaults if old[key] != new[key]}


class MaterialCoordinates:
    """Invertible coordinates at fixed density; arrays use native model units."""

    def __init__(self, mode, density, regions, size):
        self.mode = parameterization({'parameterization': mode, 'enabled': True})
        self.regions = np.asarray(regions, dtype=int).copy()
        if self.regions.ndim != 1 or not self.regions.size:
            raise ValueError('Material update regions must be a nonempty index vector.')
        if np.any(self.regions < 0) or np.any(self.regions >= size):
            raise ValueError('Material update region is outside the model.')
        if density is None:
            if mode == 'modulus_viscosity':
                raise ValueError('modulus_viscosity requires the current density profile.')
            density = np.ones(size)
        self.density = np.asarray(density, dtype=float).copy()
        if self.density.shape != (size,) or np.any(~np.isfinite(self.density)) or np.any(self.density <= 0):
            raise ValueError('Material density must be a positive finite vector matching the model.')
        self.density.setflags(write=False)

    @property
    def density_digest(self):
        return hashlib.sha256(np.asarray(self.density, dtype='<f8').tobytes()).hexdigest()

    @property
    def region_digest(self):
        return hashlib.sha256(np.asarray(self.regions, dtype='<i8').tobytes()).hexdigest()

    def encode(self, velocity, damping):
        if self.mode == 'velocity_damping':
            return velocity, damping
        return self.density*velocity**2, self.density*damping

    def bounds(self, lower_velocity, upper_velocity):
        if self.mode == 'velocity_damping':
            return lower_velocity, upper_velocity
        ii = self.regions
        if np.any(lower_velocity[ii] <= 0) or np.any(upper_velocity[ii] < lower_velocity[ii]):
            raise ValueError('Modulus coordinates require positive ordered velocity bounds.')
        return self.density*lower_velocity**2, self.density*upper_velocity**2

    def gradients(self, velocity, gradient_velocity, gradient_damping):
        if self.mode == 'velocity_damping':
            return gradient_velocity, gradient_damping
        return gradient_velocity/(2*self.density*velocity), gradient_damping/self.density

    def decode(self, first, second, velocity, damping, active=(True, True)):
        if self.mode == 'velocity_damping':
            return first, second
        # Preserve every fixed node/block exactly, avoiding round-trip drift.
        v, a = velocity.copy(), damping.copy()
        ii = self.regions
        if active[0]:
            if np.any(~np.isfinite(first[ii])) or np.any(first[ii] <= 0):
                raise ValueError('Trial modulus must be positive and finite.')
            v[ii] = np.sqrt(first[ii]/self.density[ii])
        if active[1]:
            if np.any(~np.isfinite(second[ii])) or np.any(second[ii] < 0):
                raise ValueError('Trial viscosity must be non-negative and finite.')
            a[ii] = second[ii]/self.density[ii]
        return v, a


class ParameterizedOptimizer:
    """Adapter around the unchanged projected L-BFGS/search implementation."""

    def __init__(self, optimizer_class, policy, velocity, damping, regions,
                 v_lower, v_upper, active_velocity, active_damping, *,
                 density_model=None, resume=None):
        self.mode = parameterization(policy)
        self.coordinates = MaterialCoordinates(self.mode, density_model, regions, len(velocity))
        first, second = self.coordinates.encode(velocity, damping)
        lower, upper = self.coordinates.bounds(v_lower, v_upper)
        saved = deepcopy(resume)
        migration = None
        if saved and saved.get('optimizer_revision') in (3, 4):
            old_mode = saved.get('parameterization', saved.get('policy', {}).get('parameterization', 'velocity_damping'))
            if old_mode not in MODES:
                raise ValueError('Unknown saved material parameterization.')
            if saved.get('parameterization') is not None and old_mode != saved.get('policy', {}).get('parameterization', 'velocity_damping'):
                raise ValueError('Saved material coordinates and optimizer policy disagree.')
            if saved.get('density_digest') is not None and saved['density_digest'] != self.coordinates.density_digest:
                raise ValueError('Pending optimizer density changed; accepted coordinates cannot be reused.')
            if saved.get('region_digest') is not None and saved['region_digest'] != self.coordinates.region_digest:
                raise ValueError('Pending optimizer update regions changed.')
            if old_mode != self.mode:
                from .material_optimizer import policy_changes as check_policy
                check_policy(saved['policy'], policy)
                if saved.get('reason') is not None:
                    raise ValueError('Cannot change coordinates of an already converged optimizer.')
                if saved.get('geometry_size') != len(regions) or tuple(saved.get('active', ())) != (bool(active_velocity), bool(active_damping)):
                    raise ValueError('Pending optimizer geometry or active parameters changed.')
                # A sequential continuation may recompute a smaller vmin from
                # the accepted model. Retain the original physical bounds.
                n = len(regions)
                old_scales = np.asarray(saved['scales'], dtype=float)
                old_lower = np.asarray(saved['lower'], dtype=float)*old_scales
                old_upper = np.asarray(saved['upper'], dtype=float)*old_scales
                if old_scales.shape != (2*n,) or np.any(~np.isfinite(old_scales)) or np.any(old_scales <= 0):
                    raise ValueError('Invalid saved material coordinate scales.')
                if old_lower.shape != (2*n,) or old_upper.shape != (2*n,):
                    raise ValueError('Invalid saved material coordinate bounds.')
                ii = self.coordinates.regions
                physical_lower, physical_upper = v_lower.copy(), v_upper.copy()
                if old_mode == 'modulus_viscosity':
                    physical_lower[ii] = np.sqrt(old_lower[:n]/self.coordinates.density[ii])
                    physical_upper[ii] = np.sqrt(old_upper[:n]/self.coordinates.density[ii])
                else:
                    physical_lower[ii], physical_upper[ii] = old_lower[:n], old_upper[:n]
                if np.any(old_lower[n:] != 0) or not np.all(np.isposinf(old_upper[n:])):
                    raise ValueError('Unrecognized saved viscosity bounds; coordinate migration refused.')
                lower, upper = self.coordinates.bounds(physical_lower, physical_upper)
                migration = {'type': 'material_parameterization_change', 'from': old_mode, 'to': self.mode,
                             'accepted_model_retained': True, 'curvature_retained': False,
                             'scales_rebuilt': True, 'physical_bounds_retained': True, 'weak_cycles_reset': True}
                saved = None
            else:
                # The underlying class implements revision-3 curvature storage;
                # the wrapper owns revision-4 coordinate/density provenance.
                saved['optimizer_revision'] = 3
        self.inner = optimizer_class(policy, first, second, regions, lower, upper,
                                     active_velocity, active_damping, resume=saved)
        if migration is not None:
            self.inner.last_iteration = int(resume.get('last_iteration', 0))
            self.inner.iteration_offset = self.inner.last_iteration
            self.inner.accepted = int(resume.get('accepted', 0))
            self.inner.total_receiver_calls = int(resume.get('total_receiver_calls', 0))
            self.inner.migration = migration
            print('Material coordinates changed: accepted velocity/damping/density retained; '
                  'L-BFGS memory and scales initialized. '+json.dumps(migration, sort_keys=True), flush=True)
        names = ('M', 'eta') if self.mode == 'modulus_viscosity' else ('velocity', 'damping')
        self.names = names
        print(f'Local material parameterization: {self.mode}; optimization variables={names}; density fixed.', flush=True)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def search(self, velocity, damping, gradient_velocity, gradient_damping, current_misfit, evaluate):
        first, second = self.coordinates.encode(velocity, damping)
        g_first, g_second = self.coordinates.gradients(velocity, gradient_velocity, gradient_damping)
        def material_evaluate(a, b):
            v, nu = self.coordinates.decode(a, b, velocity, damping, self.inner.active)
            return evaluate(v, nu)
        result = self.inner.search(first, second, g_first, g_second, current_misfit, material_evaluate)
        if result['selected'] != 'current':
            result['velocity'], result['damping'] = self.coordinates.decode(
                result['velocity'], result['damping'], velocity, damping, self.inner.active)
        else:
            result['velocity'], result['damping'] = velocity, damping
        info = self.inner.last_info
        info['parameterization'] = self.mode
        info['optimization_variables'] = list(self.names)
        if self.mode == 'modulus_viscosity':
            labels = {'gradient_velocity': 'gradient_modulus', 'gradient_damping': 'gradient_viscosity'}
            result['selected'] = labels.get(result['selected'], result['selected'])
            if 'selected' in info:
                info['selected'] = labels.get(info['selected'], info['selected'])
            for trial in info.get('trials', []):
                trial['direction'] = labels.get(trial['direction'], trial['direction'])
        return result

    def observe(self, **kwargs):
        return self.inner.observe(**kwargs)

    def finish(self, *args, **kwargs):
        result = self.inner.finish(*args, **kwargs)
        result['optimizer_revision'] = REVISION
        result['parameterization'] = self.mode
        result['optimization_variables'] = list(self.names)
        state = result['continuation']
        state.update(optimizer_revision=REVISION, parameterization=self.mode,
                     density_digest=self.coordinates.density_digest,
                     region_digest=self.coordinates.region_digest)
        return result


def make_parameterized_optimizer(optimizer_class, policy, *args, **kwargs):
    return ParameterizedOptimizer(optimizer_class, policy, *args, **kwargs)


def annotate_result(row, simulation):
    """Record coordinates without duplicating large derived profile arrays."""
    policy = getattr(simulation, 'optimization_policy', None) or {}
    mode = getattr(simulation, 'local_material_parameterization', parameterization(policy))
    row['local_material_parameterization'] = mode
    row['local_material_parameterization_revision'] = 1
    row['local_material_density_fixed'] = True
    return row


# LOCAL_DENSITY_INVERSION_V1
_canonical_fixed_density = canonical_policy
_annotate_fixed_density = annotate_result

def canonical_policy(policy):
    from .density_inversion import canonical_policy as normalize_density
    return normalize_density(_canonical_fixed_density(policy))

def policy_changes(previous, current, defaults):
    from .density_inversion import policy_changes as density_policy_changes
    return density_policy_changes(previous, current, defaults)

def annotate_result(row, simulation):
    from .density_inversion import annotate_result as density_result
    return density_result(_annotate_fixed_density(row, simulation), simulation)
