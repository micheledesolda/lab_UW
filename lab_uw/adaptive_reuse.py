"""Carry accepted DDS profiles through data-gated records without propagation.

Only geometry bookkeeping and serialization are performed. In particular no
waveform fitting, spectral diagnostic or kinematic calibration is requested.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np


def reuse_without_forward(simulation, assembly_dict, params):
    from .forward_modeling import shrink_grid_if_gouge_shrinks
    from .material_optimizer import settings
    from .material_parameters import parameterization

    if not params.get('_adaptive_reuse', False):
        raise ValueError('No-forward reuse requires an explicit adaptive gate decision.')
    if simulation.geometry_type != 'dds':
        raise ValueError('No-forward adaptive reuse is currently defined for DDS.')
    origin = params.get('_adaptive_model_origin_rec_n')
    if origin is None:
        raise ValueError('No-forward reuse needs the originating accepted model record.')
    model = simulation.velocity_model_handler
    density = model.density_array.copy()
    old_kinematics = deepcopy(getattr(simulation, 'kinematic_diagnostics', None) or {})
    # Apply the same existing pure-array grid bookkeeping as the local solver.
    # This keeps density, operators, indices and future restart grids aligned.
    assembly, idx, x, v, nu, source, receiver, removed = shrink_grid_if_gouge_shrinks(
        assembly_dict=assembly_dict, idx_dict=model.idx_dict,
        spatial_axis=simulation.grid_handler.spatial_axis, dx=simulation.grid_handler.dx,
        velocity_model=model.velocity_array, damping_model=model.damping_array,
        source_spatial_function=simulation.source_handler.spatial_function,
        receiver_spatial_function=simulation.receiver_handler.spatial_function)
    simulation.assembly_dict = assembly
    if len(removed):
        simulation.grid_handler.spatial_axis = x
        simulation.grid_handler.total_grid_points = len(x)
        model.x = x
        model.idx_dict = idx
        model.velocity_array = v
        model.damping_array = nu
        model.density_array = np.delete(density, removed)
        simulation.source_handler.spatial_function = source
        simulation.receiver_handler.spatial_function = receiver

    policy = settings(params)
    simulation.local_material_parameterization = parameterization(policy)
    if 'invert_density' in policy:
        simulation.local_density_inverted = bool(policy['invert_density'])
        simulation.local_density_bounds_g_cm3 = deepcopy(policy.get('density_bounds_g_cm3'))
    gouge = np.concatenate([model.idx_dict[k] for k in ('gouge_1', 'gouge_2')])
    simulation.average_gouge_velocity = float(np.mean(model.velocity_array[gouge]))
    simulation.average_gouge_damping = float(np.mean(model.damping_array[gouge]))
    for i in (1, 2):
        ii = model.idx_dict[f'gouge_{i}']
        assembly_dict[f'gouge_velocity_{i}'] = float(np.mean(model.velocity_array[ii]))
        assembly_dict[f'gouge_damping_{i}'] = float(np.mean(model.damping_array[ii]))
    simulation.wavefield_forward = None
    simulation.misfit = float('nan')
    simulation.objective_components = {}
    simulation.local_inversion_history = []
    simulation.local_stop_diagnostics = {
        'reason': 'model_reused_without_forward', 'converged': True,
        'iterations': 0, 'accepted_updates': 0, 'forward_evaluations': 0}
    simulation.optimization_resume_state = None
    simulation.scientific_stop_resume_state = None
    simulation.model_reuse_diagnostics = {
        'schema': 1, 'forward_evaluated': False, 'model_misfit_evaluated': False,
        'source_scale_evaluated': False, 'kinematics_evaluated': False,
        'model_origin_rec_n': int(origin), 'current_rec_n': int(assembly_dict['rec_n']),
        'geometry_grid_changed': bool(len(removed)), 'removed_grid_nodes': int(len(removed)),
        'synthetic_waveform_status': 'retained_restart_placeholder_not_current_prediction',
        'source_scale_status': 'inherited_not_refitted'}
    # The cached synthetic is kept solely for legacy restart preconditions;
    # all current-data model metrics are invalidated and no synthetic is plotted.
    # A fresh selected inversion always recomputes its own initial forward.
    if len(removed):
        simulation.kinematic_velocity_m_s = None
        simulation.kinematic_arrival_us = None
        simulation.kinematic_diagnostics = {
            'schema': 1, 'status': 'unavailable', 'velocity_m_s': None, 'arrival_us': None,
            'reason': 'Geometry changed on a skipped record; no kinematic replay was performed.',
            'evaluation': 'skipped_without_forward', 'model_origin_rec_n': int(origin)}
    else:
        old_kinematics.update(evaluation='inherited_without_forward', model_origin_rec_n=int(origin),
                              calibration_forward_calls=0, profile_forward_calls=0)
        simulation.kinematic_diagnostics = old_kinematics
    return {'gouge_velocity_list': None, 'best_gouge_velocity': assembly_dict['gouge_velocity_1'],
            'gouge_damping_list': None, 'best_gouge_damping': assembly_dict['gouge_damping_1'],
            'L2norm_waveform_list': None, 'local_spectral_diagnostics': None,
            'simulation_handler': simulation}


def annotate_result(row, simulation):
    """Keep NaN model misfits separate from the already evaluated data gate."""
    diagnostics = getattr(simulation, 'model_reuse_diagnostics', None)
    row['forward_evaluated'] = diagnostics is None
    row['model_misfit_evaluated'] = diagnostics is None
    if diagnostics is not None:
        row['model_reuse_diagnostics'] = deepcopy(diagnostics)
    return row
