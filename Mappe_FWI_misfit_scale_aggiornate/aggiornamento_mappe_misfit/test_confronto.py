#!/usr/bin/env python3
"""Controlli di abbinamento, residui, selezione e isolamento dell'inversione."""
import ast
import gzip
import json
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import crea_mappa_simulazioni as module


def row(index, evaluated=True, trace=None):
    return dict(rec_n=100+index, idx_wave=10+index, acquisition_time_s=1000+.01*index,
                forward_evaluated=evaluated, misfit=.02,
                adaptive_inversion={'optimized': False},
                local_continuation_state={'simulation_attributes': {
                    'synthetic_waveform': np.array([1., 2., 3.])*(index+1) if trace is None else trace,
                    'trace_scale': 7.}})


class ComparisonTests(unittest.TestCase):
    def test_evaluated_even_without_accepted_update_and_no_second_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'waves.pkl'
            path.write_bytes(b''.join(pickle.dumps(r) for r in [row(0), row(1, False), row(2)]))
            selected, metadata, _ = module.read_traces(path)
            self.assertEqual([r['rec_n'] for r in selected], [100, 102])
            self.assertEqual(len(metadata), 3)
            np.testing.assert_array_equal(selected[0]['synthetic'], [1., 2., 3.])

    def test_gzip_detection_uses_contents_and_missing_synthetics_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'renamed_without_gz'
            path.write_bytes(gzip.compress(pickle.dumps(row(0))+pickle.dumps(row(1))))
            self.assertEqual(len(module.read_traces(path)[0]), 2)
            missing = row(1); missing.pop('local_continuation_state')
            path.write_bytes(pickle.dumps(row(0))+pickle.dumps(missing))
            with self.assertRaisesRegex(ValueError, 'originale completo'):
                module.read_traces(path)

    def test_truncated_stream_and_nonfinite_traces_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'waves.pkl'
            path.write_bytes(pickle.dumps(row(0))+pickle.dumps(row(1))[:-20])
            with self.assertRaisesRegex(ValueError, 'incompleto'):
                module.read_traces(path)
        with self.assertRaises(ValueError):
            module.saved_synthetic(row(0, trace=np.array([np.nan])))

    def test_restart_keeps_last_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'waves.pkl'
            path.write_bytes(b''.join(pickle.dumps(r) for r in [row(0), row(1), row(0, trace=np.array([4., 5., 6.]))]))
            selected, metadata, _ = module.read_traces(path)
            np.testing.assert_array_equal(selected[0]['synthetic'], [4., 5., 6.])
            self.assertEqual(len(metadata), 2)

    def test_no_hardcoded_s_window_and_no_residual_clipping(self):
        with tempfile.TemporaryDirectory() as folder:
            t = np.linspace(9., 13., 9)
            d = np.full((3, 9), -2.)
            s = np.full((3, 9), 3.)
            info = module.make_maps(folder, acoustic_time=t, anchor_time=np.array([100., 101., 105.]),
                                    observed=d, synthetic=s, rec_n=np.arange(3), formats=['png'], dpi=55,
                                    misfit_mask=np.ones((3, 9), bool))
            self.assertEqual(info['common_color_limit'], 3.)
            self.assertEqual(info['residual_separate_color_limit'], 5.)
            self.assertEqual(info['acoustic_window_samples_us'], [9., 13.])
            self.assertEqual(info['relative_squared_L2_window_median'], 6.25)
            with np.load(Path(folder)/'tracce_per_confronto.npz') as f:
                np.testing.assert_array_equal(f['observed'], d)
                np.testing.assert_array_equal(f['synthetic'], s)
                np.testing.assert_array_equal(f['residual'], s-d)

    def test_window_and_temporal_cell_boundaries(self):
        t = np.arange(5.)
        np.testing.assert_array_equal(module.window_mask(t, None), np.ones(5, bool))
        np.testing.assert_array_equal(module.window_mask(t, [1., 3.]), [False, True, True, True, False])
        with self.assertRaises(ValueError):
            module.window_mask(t, [20., 25.])
        times, values = np.array([0., 1., 4.]), np.arange(6).reshape(3, 2)
        e, v = module.display_cells(times, values, 'hold', 4.)
        np.testing.assert_array_equal(e, times)
        np.testing.assert_array_equal(v, values[:-1])
        e, v = module.display_cells(times, values, 'samples', 4.)
        np.testing.assert_array_equal(e, [-.5, .5, 1.5, 2.5])
        np.testing.assert_array_equal(v, values)

    def test_main_isolated_for_both_polarizations(self):
        source = '''
class UltrasonicModeler:
    def forward_simulation(self): raise AssertionError('forward')
    def run_local_inversion(self): raise AssertionError('inversion')
def process_uw_file(infile_path):
    raise AssertionError('Original processing body executed')
if __name__ == '__main__':
    wave_type='_s'
    selected_uw_basename='wrong'
    params={'diagnostic_plot_only':True,'run_local_inversion':True}
    process_uw_file('example')
'''
        for wave in ('_p', '_s'):
            calls = []
            tree = module.prepare_main(source, wave_type=wave, basename='file')
            namespace = {'__name__': '__main__', '_map_disable_modeler': module.disable_modeler,
                         '_map_export_context': lambda ns, ctx: calls.append((ns['wave_type'], ns['params']))}
            exec(compile(tree, 'stub.py', 'exec'), namespace)
            self.assertEqual(calls[0][0], wave)
            self.assertFalse(calls[0][1]['run_local_inversion'])
            for name in ('forward_simulation', 'run_local_inversion'):
                with self.assertRaisesRegex(RuntimeError, 'disabilitati'):
                    getattr(namespace['UltrasonicModeler'](), name)()

    def test_context_saved_clock_indices_and_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            main = root/'main.py'; main.write_text('# original preprocessing')
            results = root/'005_reload_ss_1_P.pkl'; results.write_bytes(b'fixture')
            args = SimpleNamespace(main=main, results=results, basename='005_reload_ss_1_P',
                                   output=root/'out', clock='saved', dense=False, window_us=None,
                                   wave_label='P', formats=['png'], dpi=55, display='hold')
            fast, slow = np.linspace(9., 13., 9), np.arange(30)*.01
            data = np.arange(270.).reshape(30, 9)
            idx = np.array([10, 12, 17])
            times = 1000.+slow[idx]
            selected = [dict(idx_wave=int(j), rec_n=100+int(j), time_saved_s=float(tt),
                             saved_misfit=.02, saved_relative_l2=.01, synthetic=data[j]*.98,
                             observed_time=fast, observed_waveform=data[j].copy(),
                             misfit_window_specs=[('samples', slice(1, 8), 'fixture.misfit_interval')]) for j, tt in zip(idx, times)]
            metadata = [dict(idx_wave=int(j), time_saved_s=float(1000.+slow[j]), rec_n=100+j)
                        for j in range(10, 18)]
            handler = SimpleNamespace(waveform_data=data,
                                      metadata={'time_ax_waveform': fast, 'time_ax_acquisition': slow})
            mech = {'time_s': 1000.+slow, 'shear_stress_MPa': np.ones(30)*20}
            context = dict(uw_data_handler=handler, infile_path=root/'005_reload_ss_1_P.bscan.tsv')
            namespace = dict(mech_data=mech, sync_peaks=np.array([0, 29]), chosen_uw_file=0)
            with patch.object(module, 'make_maps') as plotting:
                with self.assertRaises(module.ExportCompleted):
                    module.export_context(namespace, context, selected, metadata, args)
                kwargs = plotting.call_args.kwargs
                np.testing.assert_array_equal(kwargs['anchor_time'], times)
                np.testing.assert_array_equal(kwargs['observed'], data[idx])
                np.testing.assert_array_equal(kwargs['synthetic'], data[idx]*.98)
                self.assertEqual(kwargs['provenance']['checkpoint_observed_traces_checked'], 3)
            selected[0]['observed_waveform'][0] += 1
            with self.assertRaisesRegex(ValueError, 'Preprocessing/dato diverso'):
                module.export_context(namespace, context, selected, metadata, args)

    def test_modeler_guard_can_be_restored_in_interactive_python(self):
        class Modeler:
            def forward_simulation(self):
                return 'original forward'
        undo = module.disable_modeler(Modeler)
        with self.assertRaisesRegex(RuntimeError, 'disabilitati'):
            Modeler().forward_simulation()
        undo()
        self.assertEqual(Modeler().forward_simulation(), 'original forward')

    def test_real_main_ast_interface(self):
        path = Path(__file__).resolve().parents[1]/'upload/Pasted text(20261025-142926).txt'
        if path.exists():
            for wave in ('_p', '_s'):
                tree = module.prepare_main(path.read_text(), wave_type=wave, basename='example')
                compile(tree, 'real_main.py', 'exec')

    def test_misfit_window_union_and_strict_missing_window(self):
        time = np.arange(0., 16., .5)
        masks = np.zeros((2, len(time)), bool)
        masks[0, 20:28] = True
        masks[1, 21:30] = True
        selected, source = module.select_plot_window(time, None, masks)
        np.testing.assert_array_equal(time[selected], time[20:30])
        self.assertIn('saved misfit', source)
        with self.assertRaisesRegex(ValueError, 'non salvata'):
            module.select_plot_window(time, None, None)
        selected, source = module.select_plot_window(time, None, None, full_window=True)
        self.assertTrue(selected.all())

    def test_nested_checkpoint_masks_us_and_conflicts(self):
        time = np.arange(20.)*.5
        row = {'rec_n': 1, 'local_continuation_state': {'observation': {
            'misfit_interval': {'start': 10, 'stop': 18},
            'misfit_window_us': [5., 8.5]}},
            'source_calibration': {'misfit_interval': [0, 1]}}
        specs = module.saved_window_specs(row)
        self.assertEqual(len(specs), 2)
        masks, info = module.recover_misfit_masks([dict(rec_n=1, misfit_window_specs=specs)], time)
        expected = np.zeros((1,20), bool); expected[:,10:18] = True
        np.testing.assert_array_equal(masks, expected)
        bad = specs+[('samples', [0, 1], 'inconsistent_window')]
        with self.assertRaisesRegex(ValueError, 'discordanti'):
            module.recover_misfit_masks([dict(rec_n=1, misfit_window_specs=bad)], time)

    def test_old_npz_recovers_masks_only_from_the_same_saved_traces(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, cache = root/'source.pkl', root/'old.npz'
            fast = np.linspace(0., 16., 9)
            traces = np.arange(18.).reshape(2,9)
            rows = [row(i, trace=traces[i]) for i in range(2)]
            for r in rows:
                r['observations'] = {'misfit_interval': slice(5, 8), 'observed_time': fast}
            source.write_bytes(b''.join(pickle.dumps(r) for r in rows))
            np.savez_compressed(cache, time_us=fast, acquisition_time_s=np.array([1000.,1000.01]),
                                observed=traces*.99, synthetic=traces, rec_n=np.array([100,101]),
                                provenance_json=np.array(json.dumps({'results_path':str(source)})))
            args = SimpleNamespace(window_results=None, window_us=None, full_window=False,
                                   output=root/'out', wave_label='P', formats=['png'], dpi=55,
                                   display='hold', dense=False, include_common_residual=False)
            with patch.object(module, 'make_maps') as plotting:
                module.from_npz(cache, args)
                expected = np.zeros((2,9),bool);expected[:,5:8] = True
                np.testing.assert_array_equal(plotting.call_args.kwargs['misfit_mask'], expected)
            rows[0]['local_continuation_state']['simulation_attributes']['synthetic_waveform'] = traces[0]+1
            source.write_bytes(b''.join(pickle.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, 'sintetiche diverse'):
                module.from_npz(cache, args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
