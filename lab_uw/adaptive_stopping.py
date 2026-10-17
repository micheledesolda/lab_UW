"""Noise discrepancy and plateau stopping for the unchanged FD local optimizer."""
from __future__ import annotations
from copy import deepcopy
import numpy as np


class LocalStopMonitor:
    def __init__(self, policy, initial_misfit):
        self.policy=deepcopy(policy)
        self.initial_misfit=float(initial_misfit)
        self.reason=None
        self.accepted=0
        self.block_start=float(initial_misfit)
        self.block_accepted=0
        self.weak_blocks=0
        self.block_gains=[]
        self.probes=[]
        self.probe_start=None
        self.probe_steps=None
        self.probe_number=0
        self.last_iteration=0

    def before_iteration(self, best_misfit):
        target=self.policy.get('noise_target')
        if target is not None and best_misfit <= target:
            self.reason='noise_level_reached'
            return True
        return False

    def observe(self, *, best_misfit, accepted, velocity_step, damping_step, iteration):
        self.last_iteration=int(iteration)
        if accepted:self.accepted+=1
        if self.before_iteration(best_misfit):
            return {'stop':True}
        epsilon=float(self.policy['minimum_block_gain'])
        if self.probe_start is not None:
            self.probes[-1].update(best_misfit=float(best_misfit),accepted=bool(accepted),iteration=int(iteration))
            gain=self.probe_start-float(best_misfit)
            if gain >= epsilon and gain > 0:
                self.probe_start=None
                self.probe_steps=None
                self.weak_blocks=0
                self.block_start=float(best_misfit)
                self.block_accepted=0
                return {}
            if self.probe_number < len(self.policy['probe_factors']):
                return self._next_probe()
            self.reason='plateau_after_step_probes'
            return {'stop':True}
        if accepted:self.block_accepted+=1
        if self.block_accepted < int(self.policy['block_accepted_updates']):
            return {}
        gain=self.block_start-float(best_misfit)
        self.block_gains.append({'iteration':int(iteration),'accepted_updates':self.block_accepted,'gain':float(gain)})
        self.weak_blocks=self.weak_blocks+1 if gain < epsilon else 0
        self.block_start=float(best_misfit)
        self.block_accepted=0
        if self.weak_blocks < int(self.policy['patience_blocks']):
            return {}
        factors=self.policy['probe_factors']
        if not factors:
            self.reason='plateau_without_step_probe'
            return {'stop':True}
        self.probe_start=float(best_misfit)
        self.probe_steps=(float(velocity_step),float(damping_step))
        self.probe_number=0
        return self._next_probe()

    def _next_probe(self):
        factor=float(self.policy['probe_factors'][self.probe_number])
        self.probe_number+=1
        self.probes.append({'factor':factor,'starting_misfit':self.probe_start})
        return {'velocity_step':self.probe_steps[0]*factor,'damping_step':self.probe_steps[1]*factor,
                'probe':True}

    def finish(self, fallback, velocity_step=None, damping_step=None, velocity_threshold=None, damping_threshold=None):
        result={'reason':self.reason or fallback,'noise_level_single':self.policy.get('noise_level_single'),
                'noise_target':self.policy.get('noise_target'),'minimum_block_gain':self.policy['minimum_block_gain'],
                'block_accepted_updates':self.policy['block_accepted_updates'],'patience_blocks':self.policy['patience_blocks'],
                'accepted_updates':self.accepted,'block_gains':self.block_gains,'step_probes':self.probes,
                'initial_misfit':self.initial_misfit,'interpretation':'Operational stop; plateau is not proof of global optimality.'}
        result['continuation']=deepcopy(self.__dict__)
        result['step_limits']={'dc_max_start':velocity_step,'da_max_start':damping_step,
                              'dc_threshold':velocity_threshold,'da_threshold':damping_threshold}
        return result


def make_monitor(policy, initial_misfit, gamma, resume_state=None):
    if not policy or not policy.get('enabled',False):
        return None
    if float(gamma)!=1.:
        raise ValueError('Scientific noise stopping requires local_spectral_gamma=1. Disable adaptive/local scientific stopping for other metrics.')
    for key in ('block_accepted_updates','patience_blocks'):
        value=policy[key]
        if isinstance(value,bool) or int(value)!=value or int(value)<1:
            raise ValueError(f'{key} must be a positive integer.')
    for key in ('minimum_block_gain','noise_level_single'):
        value=float(policy[key])
        if not np.isfinite(value) or value<0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    for value in policy['probe_factors']:
        if not np.isfinite(value) or value<=0:
            raise ValueError('Step probe factors must be positive and finite.')
    monitor=LocalStopMonitor(policy,initial_misfit)
    if resume_state:
        if resume_state['policy']!=policy:raise ValueError('Scientific stopping policy changed during continuation.')
        if resume_state.get('reason') is not None:raise ValueError('Cannot continue a completed scientific monitor.')
        monitor.__dict__.update(deepcopy(resume_state))
    monitor.iteration_offset=monitor.last_iteration
    return monitor
