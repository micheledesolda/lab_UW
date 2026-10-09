import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from lab_uw.validation.synthetic.models import (
    apply_candidate,
    build_dds_assembly,
    cartesian_candidates,
    refined_axes,
)
from lab_uw.validation.synthetic.cli import load_configuration, run_validation
from lab_uw.validation.synthetic.metrics import (
    normalized_arrival_time,
    profile_relative_l2,
)


ROOT = Path(__file__).resolve().parents[1]


class SyntheticModelConfigurationTests(unittest.TestCase):
    def test_case_inherits_base_and_builds_physical_assembly(self):
        config, _ = load_configuration(
            ROOT / "configs" / "synthetic_validation" / "impedance_contrast.json"
        )
        assembly = build_dds_assembly(config, model_section="truth_model")
        self.assertTrue(assembly["density_model_enabled"])
        self.assertEqual(assembly["gouge_density_1"], 2.4)
        self.assertEqual(assembly["gouge_velocity_2"], 0.2)
        self.assertEqual(assembly["sample_dimensions"][1], 0.05)

        asymmetric, _ = load_configuration(
            ROOT / "configs" / "synthetic_validation" / "asymmetric_velocity.json"
        )
        self.assertEqual(
            set(asymmetric["inversion"]["coarse_axes"]),
            {"gouge_velocity_1", "gouge_velocity_2"},
        )

    def test_thickness_candidate_updates_both_public_representations(self):
        config, _ = load_configuration(
            ROOT / "configs" / "synthetic_validation" / "base.json"
        )
        assembly = build_dds_assembly(config, model_section="initial_model")
        candidate = apply_candidate(
            assembly,
            {"gouge_thickness_1": 0.04, "gouge_thickness_2": 0.06},
        )
        self.assertEqual(candidate["gouge_thickness_1"], 0.04)
        self.assertEqual(candidate["gouge_thickness_2"], 0.06)
        self.assertEqual(candidate["sample_dimensions"][1], 0.04)
        self.assertEqual(candidate["sample_dimensions"][3], 0.06)

    def test_cartesian_and_refined_axes_are_deterministic(self):
        coarse = {
            "gouge_velocity_common": [0.18, 0.20, 0.22],
            "gouge_density_common": [2.0, 2.4],
        }
        candidates = cartesian_candidates(coarse)
        self.assertEqual(len(candidates), 6)
        fine = refined_axes(
            coarse,
            {"gouge_velocity_common": 0.20, "gouge_density_common": 2.4},
            points_per_axis=3,
            bounds={"gouge_density_common": [1.5, 3.0]},
        )
        np.testing.assert_allclose(fine["gouge_velocity_common"], [0.18, 0.20, 0.22])
        np.testing.assert_allclose(fine["gouge_density_common"], [2.0, 2.4, 2.8])


class ValidationMetricTests(unittest.TestCase):
    def test_relative_arrival_threshold_is_amplitude_scale_invariant(self):
        time = np.arange(20, dtype=float) * 0.1
        waveform = np.zeros(20)
        waveform[7:] = np.linspace(0.1, 1.0, 13)
        self.assertEqual(
            normalized_arrival_time(time, waveform),
            normalized_arrival_time(time, 17.0 * waveform),
        )

    def test_profile_error_is_zero_across_different_grids(self):
        truth_x = np.linspace(0.0, 1.0, 101)
        recovered_x = np.linspace(0.0, 1.0, 21)
        self.assertAlmostEqual(
            profile_relative_l2(
                truth_x, 2.0 * truth_x + 1.0,
                recovered_x, 2.0 * recovered_x + 1.0,
            ),
            0.0,
            places=13,
        )


class SyntheticValidationEndToEndTests(unittest.TestCase):
    def test_homogeneous_case_recovers_known_common_velocity(self):
        config, base = load_configuration(
            ROOT / "configs" / "synthetic_validation" / "homogeneous.json"
        )
        config["numerics"] = {
            "truth_points_per_wavelength": 12,
            "inversion_points_per_wavelength": 8,
        }
        config["inversion"]["coarse_axes"] = {
            "gouge_velocity_common": [0.50, 0.57, 0.64],
        }
        config["inversion"]["fine_points_per_axis"] = 1
        config["inversion"]["local"]["enabled"] = False
        with tempfile.TemporaryDirectory() as directory:
            result = run_validation(
                config,
                base_directory=base,
                output_directory=Path(directory),
                disable_local=True,
            )
            self.assertEqual(result["name"], "homogeneous")
            self.assertAlmostEqual(
                result["recovered_assembly"]["gouge_velocity_1"],
                0.57,
            )
            self.assertTrue((Path(directory) / "validation_summary.json").is_file())
            stored = json.loads(
                (Path(directory) / "validation_summary.json").read_text()
            )
            self.assertEqual(stored["name"], "homogeneous")


if __name__ == "__main__":
    unittest.main()
