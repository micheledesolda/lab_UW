import unittest
import importlib.util
import sys
import types

import numpy as np


if (
    "lab_uw.plotting" not in sys.modules
    and importlib.util.find_spec("lab_uw.plotting") is None
):
    plotting = types.ModuleType("lab_uw.plotting")
    plotting.Plotter = type("Plotter", (), {})
    sys.modules["lab_uw.plotting"] = plotting

if (
    "lab_uw.data_io" not in sys.modules
    and importlib.util.find_spec("lab_uw.data_io") is None
):
    data_io_package = types.ModuleType("lab_uw.data_io")
    data_io_module = types.ModuleType("lab_uw.data_io.data_io")
    data_io_module.UltrasonicDataHandler = type(
        "UltrasonicDataHandler", (), {}
    )
    sys.modules["lab_uw.data_io"] = data_io_package
    sys.modules["lab_uw.data_io.data_io"] = data_io_module

from lab_uw.forward_modeling import (
    fit_trace_scale,
    relative_waveform_objective,
    select_cross_talk_candidate,
)


class RelativeWaveformObjectiveTests(unittest.TestCase):
    def test_profiled_scale_recovers_normalized_shape_objective(self):
        data = np.array([0.2, -0.7, 1.4, 0.8, -0.3, -0.9])
        synthetic = np.array([0.5, -0.2, 1.0, 0.1, -0.6, -0.4])
        interval = np.arange(data.size)
        scale = fit_trace_scale(data, synthetic, interval)
        result = relative_waveform_objective(
            data,
            scale * synthetic,
            interval,
        )
        correlation = np.dot(data, synthetic) / (
            np.linalg.norm(data) * np.linalg.norm(synthetic)
        )
        self.assertAlmostEqual(result["total"], 1.0 - correlation**2, places=12)

    def test_fixed_scale_preserves_amplitude_change(self):
        reference = np.sin(np.linspace(0.0, 3.0 * np.pi, 80))
        later_data = 0.5 * reference
        unchanged_prediction = reference.copy()
        interval = np.arange(reference.size)
        mismatch = relative_waveform_objective(
            later_data,
            unchanged_prediction,
            interval,
        )
        matched = relative_waveform_objective(
            later_data,
            0.5 * unchanged_prediction,
            interval,
        )
        self.assertAlmostEqual(mismatch["total"], 1.0, places=12)
        self.assertAlmostEqual(mismatch["amplitude_ratio"], 2.0, places=12)
        self.assertAlmostEqual(matched["total"], 0.0, places=12)

    def test_spectral_phase_decomposition_is_exact(self):
        time = np.linspace(0.0, 1.0, 96, endpoint=False)
        data = np.sin(2.0 * np.pi * 5.0 * time) + 0.4 * np.sin(
            2.0 * np.pi * 11.0 * time
        )
        predicted = 0.7 * (
            np.sin(2.0 * np.pi * 5.0 * time + 0.3)
            + 0.2 * np.sin(2.0 * np.pi * 11.0 * time - 0.2)
        )
        result = relative_waveform_objective(
            data,
            predicted,
            np.arange(time.size),
            spectral_gamma=1.8,
        )
        ratio = result["amplitude_ratio"]
        decomposed = result["amplitude_misfit"] + 2.0 * ratio * (
            1.8 * result["spectral_shape_misfit"]
            + result["phase_misfit"]
        )
        self.assertAlmostEqual(result["total"], decomposed, places=12)

    def test_returned_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(17)
        data = rng.normal(size=64)
        predicted = rng.normal(size=64)
        direction = rng.normal(size=64)
        interval = np.arange(5, 58)
        common = dict(
            interval=interval,
            spectral_gamma=1.7,
            taper_fraction=0.2,
        )
        result = relative_waveform_objective(
            data,
            predicted,
            return_gradient=True,
            **common,
        )
        step = 1.0e-6
        plus = relative_waveform_objective(
            data,
            predicted + step * direction,
            **common,
        )["total"]
        minus = relative_waveform_objective(
            data,
            predicted - step * direction,
            **common,
        )["total"]
        finite_difference = 0.5 * (plus - minus) / (2.0 * step)
        adjoint_directional = float(
            np.dot(result["gradient_predicted"], direction)
        )
        self.assertAlmostEqual(
            adjoint_directional,
            finite_difference,
            delta=2.0e-7,
        )


class CrossTalkCandidateTests(unittest.TestCase):
    def test_joint_candidate_is_selected_when_it_is_best(self):
        result = select_cross_talk_candidate(1.0, 0.8, 0.9, 0.7)
        self.assertEqual(result["selected"], "joint")
        self.assertAlmostEqual(result["interaction"], 0.0)

    def test_single_parameter_candidate_can_reject_worse_joint_change(self):
        result = select_cross_talk_candidate(1.0, 0.8, 1.1, 0.85)
        self.assertEqual(result["selected"], "velocity")
        self.assertAlmostEqual(result["selected_misfit"], 0.8)
        self.assertAlmostEqual(result["interaction"], -0.05)

    def test_current_model_is_kept_when_every_candidate_is_worse(self):
        result = select_cross_talk_candidate(1.0, 1.1, 1.2, 1.05)
        self.assertEqual(result["selected"], "current")
        self.assertAlmostEqual(result["selected_misfit"], 1.0)


if __name__ == "__main__":
    unittest.main()
