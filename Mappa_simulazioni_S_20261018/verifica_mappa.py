#!/usr/bin/env python3
"""Verifiche su formati/estrazione/avvio isolato. Non ripete una FWI reale."""
import ast
import gzip
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from crea_mappa_simulazioni import (ExportCompleted, disable_modeler, edges,
                                    export_context, make_maps, prepare_main,
                                    read_traces, saved_synthetic)


def row(i, optimized=True):
    return dict(rec_n=100+i, idx_wave=10+i, acquisition_time_s=1000+.01*i,
                misfit=.02, adaptive_inversion={'optimized': optimized},
                local_continuation_state={'simulation_attributes': {
                    'synthetic_waveform': np.array([1., 2., 3.]) * (i+1),
                    'trace_scale': 7.}})


class Tests(unittest.TestCase):
    def test_selected_only_and_no_second_amplitude_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'x.pkl'
            with path.open('wb') as stream:
                for item in [dict(global_best_misfit=.4), row(0), row(1,False), row(2)]:
                    pickle.dump(item,stream)
            selected,metadata,_=read_traces(path)
            self.assertEqual([r['rec_n'] for r in selected],[100,102])
            self.assertEqual(len(metadata),3)
            np.testing.assert_equal(selected[0]['synthetic'],[1.,2.,3.])

    def test_compact_export_rejected_before_main(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'x.pkl.gz'
            with gzip.open(path,'wb') as stream:
                r=row(0);r.pop('local_continuation_state');pickle.dump(r,stream)
            with self.assertRaisesRegex(ValueError,'originale completo'):
                read_traces(path)

    def test_truncated_stream_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'x.pkl'
            path.write_bytes(pickle.dumps(row(0))+pickle.dumps(row(1))[:-12])
            with self.assertRaisesRegex(ValueError,'incompleto/non leggibile'):
                read_traces(path)

    def test_duplicate_inversions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'x.pkl'
            path.write_bytes(pickle.dumps(row(0))+pickle.dumps(row(0)))
            with self.assertRaisesRegex(ValueError,'duplicati'):
                read_traces(path)

    def test_nonfinite_trace_rejected(self):
        r=row(0);r['local_continuation_state']['simulation_attributes']['synthetic_waveform'][0]=np.nan
        with self.assertRaises(ValueError):saved_synthetic(r)

    def test_edges_use_actual_uneven_times(self):
        np.testing.assert_allclose(edges([0.,1.,4.]),[-.5,.5,2.5,5.5])

    def test_main_body_cannot_start_original_processing(self):
        code='''
class UltrasonicModeler:
    def forward_simulation(self): raise AssertionError('forward')
def process_uw_file(infile_path):
    raise AssertionError('Original inversion body executed')
if __name__ == '__main__':
    wave_type='_p'
    selected_uw_basename='wrong'
    params={'diagnostic_plot_only':True,'run_local_inversion':True}
    process_uw_file('example')
'''
        calls=[]
        tree=prepare_main(code,wave_type='_s',basename='006_reload_ss_1_S')
        namespace={'__name__':'__main__','_map_export_context':lambda ns,ctx:calls.append((ns['wave_type'],ns['selected_uw_basename'],ns['params'],ctx)),
                   '_map_disable_modeler':disable_modeler}
        exec(compile(tree,'test_main.py','exec'),namespace)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][:2],('_s','006_reload_ss_1_S'))
        self.assertFalse(calls[0][2]['run_local_inversion'])
        with self.assertRaisesRegex(RuntimeError,'disabilitati'):
            namespace['UltrasonicModeler']().forward_simulation()

    def test_plot_and_roundtrip_preserve_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            t=np.linspace(19.,26.,176)
            slow=np.array([100.,101.,105.,106.])
            sim=np.sin(2*np.pi*(t[None,:]-.02*np.arange(4)[:,None]))
            obs=sim+0.01*np.cos(t)[None,:]
            make_maps(directory,acoustic_time=t,anchor_time=slow,observed=obs,synthetic=sim,
                      dense_time=slow,dense_observed=obs,rec_n=np.arange(4),
                      mech_time=slow,shear=np.array([20.,21.,20.,19.]),provenance={})
            p=Path(directory)
            for name in ['mappa_simulazioni.png','confronto_stesse_onde.png',
                         'confronto_tutte_le_onde.png','residuo_scala_propria.png']:
                self.assertGreater((p/name).stat().st_size,10000)
            d=np.load(p/'tracce_per_confronto.npz',allow_pickle=False)
            np.testing.assert_equal(d['synthetic'],sim)
            meta=json.loads((p/'provenienza.json').read_text())
            self.assertFalse(meta['additional_trace_scaling'])
            self.assertFalse(meta['waveform_realignment'])
            self.assertFalse(meta['temporal_interpolation_of_synthetics'])

    def test_context_maps_indices_and_changes_only_slow_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);main=p/'main.py';main.write_text('# test configuration')
            results=p/'006_reload_ss_1_S.pkl';results.write_bytes(b'test')
            args=SimpleNamespace(main=main,results=results,basename='006_reload_ss_1_S',output=p/'output')
            fast=np.linspace(19.,26.,176);slow=np.arange(40)*.01
            observed=np.arange(40)[:,None]+np.sin(fast)[None,:]
            idx=np.array([10,12,17]);synthetic=observed[idx]*.98
            selected=[dict(idx_wave=int(j),rec_n=100+int(j),time_saved_s=1000.+.01*j,
                           saved_misfit=.02,synthetic=s) for j,s in zip(idx,synthetic)]
            metadata=[(i,1000+.01*i,100+i) for i in range(10,18)]
            mt=1000.+np.arange(1001)*.001
            ns=dict(sync_peaks=np.array([0,389]),chosen_uw_file=0,
                    mech_data={'time_s':mt,'shear_stress_MPa':20.+np.sin(mt)})
            handler=SimpleNamespace(waveform_data=observed,
                    metadata={'time_ax_waveform':fast,'time_ax_acquisition':slow})
            ctx=dict(uw_data_handler=handler,infile_path=p/'006_reload_ss_1_S.bscan.tsv')
            with patch('crea_mappa_simulazioni.make_maps') as plotting:
                with self.assertRaises(ExportCompleted):export_context(ns,ctx,selected,metadata,args)
            k=plotting.call_args.kwargs
            np.testing.assert_equal(k['synthetic'],synthetic)
            np.testing.assert_equal(k['observed'],observed[idx])
            np.testing.assert_equal(k['acoustic_time'],fast)
            np.testing.assert_allclose(k['anchor_time'],1000.+(.389/.39)*slow[idx])
            self.assertEqual(len(k['dense_time']),8)


if __name__=='__main__':unittest.main(verbosity=2)
