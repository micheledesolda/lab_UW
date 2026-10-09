import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


# The distributed update is an overlay for the user's complete package.  Supply
# tiny import stubs only when this test is run against the overlay by itself.
if importlib.util.find_spec("lab_uw.plotting") is None:
    plotting = types.ModuleType("lab_uw.plotting")

    class Plotter:
        pass

    plotting.Plotter = Plotter
    sys.modules["lab_uw.plotting"] = plotting

if importlib.util.find_spec("lab_uw.data_io") is None:
    data_io_package = types.ModuleType("lab_uw.data_io")
    data_io_module = types.ModuleType("lab_uw.data_io.data_io")

    class UltrasonicDataHandler:
        pass

    data_io_module.UltrasonicDataHandler = UltrasonicDataHandler
    sys.modules["lab_uw.data_io"] = data_io_package
    sys.modules["lab_uw.data_io.data_io"] = data_io_module

from lab_uw.forward_modeling import (
    UltrasonicModeler,
    _filename_decimal,
    _local_inversion_plot_paths,
    fit_trace_scale,
)


class _STF:
    def __init__(self):
        time = np.arange(0.0, 1.5, 0.01)
        self.metadata = {"time_ax_waveform": time}
        self.waveform_data = np.sin(2.0 * np.pi * 2.0 * time) * np.exp(-3.0 * time)


def _assembly(gouge_velocity):
    side1 = {"z_pzt2grove": 0.5, "h_grooves": 0.05, "velocity_p": 0.57}
    side2 = {"z_pzt2grove": 0.5, "h_grooves": 0.05, "velocity_p": 0.55}
    central = {"z": 0.8, "h_grooves": 0.05, "velocity_p": 0.56}
    return {
        "side1_params": side1,
        "side2_params": side2,
        "central_params": central,
        "wave_type": "_p",
        "sample_dimensions": [0.5, 0.05, 0.8, 0.05, 0.5],
        "gouge_velocity_1": gouge_velocity,
        "gouge_velocity_2": gouge_velocity,
        "gouge_damping_1": 0.0002,
        "gouge_damping_2": 0.0002,
        "embedding_mode": "effective_surface",
    }


class EffectiveSurfaceModelerTests(unittest.TestCase):
    def test_local_plots_stay_inside_record_directory(self):
        record_directory = Path("images/005_reload_ss_1_P_rec_n_322414")
        local_directory = record_directory / "local_inversion"
        waveform, velocity, damping = _local_inversion_plot_paths(
            local_directory,
            "_vel_1744_damping_0,0008_FWI_misfit_0,1860",
            name_prefix=record_directory.name,
        )
        for output_path in (waveform, velocity, damping):
            self.assertEqual(output_path.parent, local_directory)
            self.assertTrue(output_path.name.startswith(record_directory.name))
        self.assertTrue(velocity.name.endswith("_velocity_model"))
        self.assertTrue(damping.name.endswith("_damping_model"))

    def test_fractional_misfit_is_not_interpreted_as_a_file_suffix(self):
        label = _filename_decimal(0.446, 4)
        self.assertEqual(label, "0,4460")
        output = Path(f"best_homogeneus_fit_misfit_{label}")
        self.assertEqual(output.suffix, "")

    def test_trace_scale_accepts_equivalent_boolean_mask_and_indices(self):
        synthetic = np.linspace(-1.0, 1.0, 50)
        observed = 3.25 * synthetic
        mask = np.zeros(50, dtype=bool)
        mask[12:31] = True
        from_mask = fit_trace_scale(observed, synthetic, mask)
        from_indices = fit_trace_scale(observed, synthetic, np.flatnonzero(mask))
        self.assertAlmostEqual(from_mask, 3.25, places=10)
        self.assertAlmostEqual(from_indices, 3.25, places=10)
        self.assertAlmostEqual(from_mask, from_indices, places=13)

    def test_final_forward_writes_two_spectral_diagnostics(self):
        observed_time = np.arange(0.0, 7.0, 0.02)
        model = UltrasonicModeler()
        model.forward_simulation(
            geometry_type="dds",
            observed_time=observed_time,
            observed_waveform=np.sin(2.0 * np.pi * observed_time),
            stf_handler=_STF(),
            frequency_cutoff=2.0,
            minimum_velocity=0.30,
            maximum_velocity=0.57,
            maximum_damping=0.0002,
            assembly_dict=_assembly(0.38),
            misfit_interval=np.flatnonzero(
                (observed_time >= 3.0) & (observed_time <= 6.0)
            ),
            embedding_mode="effective_surface",
            boundary_condition="free",
            source_injection="body_force",
            receiver_quantity="velocity",
            points_per_wavelength=12,
            source_receiver_config={"interpolation_radius_cells": 4},
            normalize_waveform=False,
            store_wavefield=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            diagnostics = model.write_direct_arrival_spectral_diagnostics(
                output_directory=Path(directory),
                frequency_min_mhz=0.2,
                frequency_max_mhz=2.0,
            )
            self.assertTrue(
                (Path(directory) / "spectrum_effective_wavelet_vs_recorded_direct.png").is_file()
            )
            self.assertTrue(
                (Path(directory) / "spectrum_recorded_vs_simulated_direct.png").is_file()
            )
            self.assertTrue(
                np.isfinite(diagnostics["recorded_vs_simulated_distance"])
            )
        # Grid-search candidates need the scalar without writing two figures
        # for every forward run.
        diagnostics_only = model.write_direct_arrival_spectral_diagnostics(
            output_directory=None,
            frequency_min_mhz=0.2,
            frequency_max_mhz=2.0,
        )
        self.assertTrue(
            np.isfinite(diagnostics_only["recorded_vs_simulated_distance"])
        )

    def test_receiver_only_mode_matches_stored_wavefield(self):
        observed_time = np.arange(0.0, 7.0, 0.02)
        common = dict(
            geometry_type="dds",
            observed_time=observed_time,
            observed_waveform=np.zeros_like(observed_time),
            stf_handler=_STF(),
            frequency_cutoff=2.0,
            minimum_velocity=0.30,
            maximum_velocity=0.57,
            maximum_damping=0.0002,
            assembly_dict=_assembly(0.38),
            misfit_interval=np.flatnonzero(
                (observed_time >= 3.0) & (observed_time <= 6.0)
            ),
            embedding_mode="effective_surface",
            boundary_condition="free",
            source_injection="body_force",
            receiver_quantity="velocity",
            points_per_wavelength=12,
            source_receiver_config={"interpolation_radius_cells": 4},
            normalize_waveform=False,
        )

        stored = UltrasonicModeler()
        stored.forward_simulation(store_wavefield=True, **common)
        streamed = UltrasonicModeler()
        streamed.forward_simulation(store_wavefield=False, **common)

        self.assertIsNotNone(stored.wavefield_forward)
        self.assertIsNone(streamed.wavefield_forward)
        np.testing.assert_allclose(
            streamed.synthetic_waveform,
            stored.synthetic_waveform,
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertAlmostEqual(streamed.misfit, stored.misfit, places=13)

    def test_forward_and_local_inversion_need_no_pzt_metadata(self):
        observed_time = np.arange(0.0, 7.0, 0.02)
        common = dict(
            geometry_type="dds",
            observed_time=observed_time,
            stf_handler=_STF(),
            frequency_cutoff=2.0,
            minimum_velocity=0.30,
            maximum_velocity=0.57,
            maximum_damping=0.0002,
            misfit_interval=np.flatnonzero((observed_time >= 3.0) & (observed_time <= 6.0)),
            embedding_mode="effective_surface",
            boundary_condition="free",
            source_injection="body_force",
            receiver_quantity="velocity",
            points_per_wavelength=12,
            source_receiver_config={"interpolation_radius_cells": 4},
        )
        truth = UltrasonicModeler()
        truth.forward_simulation(
            observed_waveform=np.zeros_like(observed_time),
            assembly_dict=_assembly(0.38),
            normalize_waveform=False,
            **common,
        )

        inversion = UltrasonicModeler()
        inversion.forward_simulation(
            observed_waveform=truth.synthetic_waveform.copy(),
            assembly_dict=_assembly(0.34),
            normalize_waveform=True,
            **common,
        )
        initial_misfit = inversion.misfit
        inversion.run_local_inversion(
            n_iterations=2,
            dc_max_start=0.002,
            dc_threshold=1e-6,
            da_max_start=0.0,
            da_threshold=0.0,
            reduce_factor=2.0,
            normalize_waveform=True,
            enable_plotting=False,
            retain_wavefield=False,
        )
        self.assertTrue(np.isfinite(inversion.misfit))
        self.assertLessEqual(inversion.misfit, initial_misfit + 1e-12)
        self.assertIsNone(inversion.wavefield_forward)
        self.assertAlmostEqual(
            inversion.grid_handler.spatial_axis[-1],
            sum(_assembly(0.34)["sample_dimensions"]),
        )

    def test_variable_density_forward_is_streamable_and_allows_fixed_density_local_adjoint(self):
        observed_time = np.arange(0.0, 7.0, 0.02)
        density_parameters = {
            "density_model_enabled": True,
            "side1_density": 7.8,
            "central_density": 7.8,
            "side2_density": 7.8,
            "gouge_density_1": 1.9,
            "gouge_density_2": 1.9,
        }
        truth_assembly = _assembly(0.40)
        truth_assembly.update(density_parameters)
        initial_assembly = _assembly(0.36)
        initial_assembly.update(density_parameters)
        common = dict(
            geometry_type="dds",
            observed_time=observed_time,
            stf_handler=_STF(),
            frequency_cutoff=2.0,
            minimum_velocity=0.30,
            maximum_velocity=0.57,
            maximum_damping=0.0002,
            misfit_interval=np.flatnonzero(
                (observed_time >= 3.0) & (observed_time <= 6.0)
            ),
            embedding_mode="effective_surface",
            boundary_condition="free",
            source_injection="body_force",
            receiver_quantity="velocity",
            points_per_wavelength=12,
            source_receiver_config={"interpolation_radius_cells": 4},
        )
        truth = UltrasonicModeler()
        truth.forward_simulation(
            observed_waveform=np.zeros_like(observed_time),
            assembly_dict=truth_assembly,
            normalize_waveform=False,
            store_wavefield=False,
            **common,
        )
        stored = UltrasonicModeler()
        stored.forward_simulation(
            observed_waveform=truth.synthetic_waveform.copy(),
            assembly_dict=initial_assembly,
            normalize_waveform=True,
            store_wavefield=True,
            **common,
        )
        streamed = UltrasonicModeler()
        streamed.forward_simulation(
            observed_waveform=truth.synthetic_waveform.copy(),
            assembly_dict=initial_assembly,
            normalize_waveform=True,
            store_wavefield=False,
            **common,
        )
        np.testing.assert_allclose(
            streamed.synthetic_waveform,
            stored.synthetic_waveform,
            rtol=1e-12,
            atol=1e-13,
        )
        initial_misfit = stored.misfit
        initial_density = stored.velocity_model_handler.density_array.copy()
        stored.run_local_inversion(
            n_iterations=2,
            dc_max_start=0.002,
            dc_threshold=1e-6,
            da_max_start=0.0,
            da_threshold=0.0,
            normalize_waveform=True,
            enable_plotting=False,
        )
        self.assertLessEqual(stored.misfit, initial_misfit + 1e-12)
        np.testing.assert_array_equal(
            stored.velocity_model_handler.density_array, initial_density
        )

    def test_alternating_and_cross_talk_local_schedules_run_and_record_diagnostics(self):
        observed_time = np.arange(0.0, 7.0, 0.02)
        common = dict(
            geometry_type="dds",
            observed_time=observed_time,
            stf_handler=_STF(),
            frequency_cutoff=2.0,
            minimum_velocity=0.30,
            maximum_velocity=0.57,
            maximum_damping=0.0004,
            misfit_interval=np.flatnonzero(
                (observed_time >= 3.0) & (observed_time <= 6.0)
            ),
            embedding_mode="effective_surface",
            boundary_condition="free",
            source_injection="body_force",
            receiver_quantity="velocity",
            points_per_wavelength=12,
            source_receiver_config={"interpolation_radius_cells": 4},
        )
        truth = UltrasonicModeler()
        truth.forward_simulation(
            observed_waveform=np.zeros_like(observed_time),
            assembly_dict=_assembly(0.38),
            normalize_waveform=False,
            **common,
        )

        for strategy in ("alternating", "cross_talk"):
            with self.subTest(strategy=strategy):
                inversion = UltrasonicModeler()
                inversion.forward_simulation(
                    observed_waveform=truth.synthetic_waveform.copy(),
                    assembly_dict=_assembly(0.35),
                    normalize_waveform=True,
                    **common,
                )
                initial_misfit = inversion.misfit
                inversion.run_local_inversion(
                    n_iterations=2,
                    dc_max_start=0.002,
                    dc_threshold=1e-6,
                    da_max_start=1e-5,
                    da_threshold=1e-8,
                    reduce_factor=2.0,
                    normalize_waveform=True,
                    update_strategy=strategy,
                    enable_plotting=False,
                    retain_wavefield=False,
                )
                self.assertEqual(inversion.local_update_strategy, strategy)
                self.assertLessEqual(inversion.misfit, initial_misfit + 1e-12)
                self.assertEqual(
                    inversion.local_inversion_history[-1]["update_strategy"],
                    strategy,
                )
                if strategy == "cross_talk":
                    row = inversion.local_inversion_history[1]
                    self.assertIsNotNone(row["velocity_only_misfit"])
                    self.assertIsNotNone(row["damping_only_misfit"])
                    self.assertIsNotNone(row["joint_misfit"])
                    self.assertIsNotNone(row["cross_talk_interaction"])

    def test_local_inversion_keeps_first_record_amplitude_calibration(self):
        observed_time = np.arange(0.0, 7.0, 0.02)
        interval = np.flatnonzero(
            (observed_time >= 3.0) & (observed_time <= 6.0)
        )
        common = dict(
            geometry_type="dds",
            observed_time=observed_time,
            stf_handler=_STF(),
            frequency_cutoff=2.0,
            minimum_velocity=0.30,
            maximum_velocity=0.57,
            maximum_damping=0.0002,
            misfit_interval=interval,
            assembly_dict=_assembly(0.38),
            embedding_mode="effective_surface",
            boundary_condition="free",
            source_injection="body_force",
            receiver_quantity="velocity",
            points_per_wavelength=12,
            source_receiver_config={"interpolation_radius_cells": 4},
        )
        truth = UltrasonicModeler()
        truth.forward_simulation(
            observed_waveform=np.zeros_like(observed_time),
            normalize_waveform=False,
            store_wavefield=False,
            **common,
        )

        inversion = UltrasonicModeler()
        inversion.forward_simulation(
            observed_waveform=2.0 * truth.synthetic_waveform,
            normalize_waveform=True,
            store_wavefield=True,
            **common,
        )
        calibration = inversion.trace_scale
        inversion.reference_trace_scale = calibration
        self.assertAlmostEqual(calibration, 2.0, places=8)

        # The next record has half the calibrated amplitude.  A per-record
        # fit would silently change the scale to one; fixed mode must retain
        # the factor two and therefore report an amplitude ratio of two.
        inversion.observed_waveform = truth.synthetic_waveform.copy()
        inversion.run_local_inversion(
            n_iterations=0,
            dc_max_start=0.0,
            da_max_start=0.0,
            normalize_waveform=False,
            amplitude_scale_mode="fixed",
            fixed_trace_scale=calibration,
            spectral_gamma=1.0,
            enable_plotting=False,
        )
        self.assertAlmostEqual(inversion.trace_scale, calibration, places=12)
        self.assertAlmostEqual(
            inversion.objective_components["amplitude_ratio"],
            2.0,
            places=8,
        )
        self.assertAlmostEqual(inversion.misfit, 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
