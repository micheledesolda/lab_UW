import unittest

import numpy as np

from lab_uw.velocity_contrast import (
    build_constant_time_contrast_candidates,
    constant_time_velocity_pair,
    dds_gouge_groove_travel_time,
    stiffness_ramp_travel_time,
)


def _assembly(*, h1=0.084, h2=0.079):
    return {
        "wave_type": "_p",
        "gouge_thickness_1": h1,
        "gouge_thickness_2": h2,
        "side1_params": {
            "velocity_p": 0.5751,
            "h_grooves": 0.05,
        },
        "side2_params": {
            "velocity_p": 0.5751,
            "h_grooves": 0.05,
        },
        "central_params": {
            "velocity_p": 0.61,
            "h_grooves": 0.05,
        },
    }


class TestVelocityContrast(unittest.TestCase):
    def test_stiffness_ramp_matches_constant_velocity_limit(self):
        self.assertAlmostEqual(
            stiffness_ramp_travel_time(0.05, 0.3, 0.3),
            0.05 / 0.3,
        )

    def test_zero_contrast_recovers_homogeneous_velocity(self):
        c1, c2, target, achieved = constant_time_velocity_pair(
            _assembly(),
            homogeneous_velocity=0.1933,
            log_contrast=0.0,
        )
        self.assertAlmostEqual(c1, 0.1933, places=11)
        self.assertAlmostEqual(c2, 0.1933, places=11)
        self.assertAlmostEqual(target, achieved, places=12)

    def test_nonzero_contrast_preserves_complete_stack_time(self):
        assembly = _assembly()
        c1, c2, target, achieved = constant_time_velocity_pair(
            assembly,
            homogeneous_velocity=0.1933,
            log_contrast=0.08,
        )
        self.assertGreater(c1, c2)
        self.assertAlmostEqual(c1 / c2, np.exp(0.16), places=11)
        self.assertAlmostEqual(target, achieved, places=11)
        self.assertAlmostEqual(
            achieved,
            dds_gouge_groove_travel_time(assembly, c1, c2),
            places=12,
        )

    def test_candidate_builder_always_reports_requested_pairs(self):
        candidates = build_constant_time_contrast_candidates(
            _assembly(),
            homogeneous_velocity=0.1933,
            log_contrasts=[-0.05, 0.0, 0.05],
        )
        self.assertEqual(len(candidates), 3)
        target = candidates[0]["target_stack_travel_time_us"]
        for candidate in candidates:
            self.assertAlmostEqual(
                candidate["stack_travel_time_us"],
                target,
                places=11,
            )

    def test_invalid_velocity_is_rejected(self):
        with self.assertRaises(ValueError):
            constant_time_velocity_pair(
                _assembly(),
                homogeneous_velocity=0.0,
                log_contrast=0.0,
            )


if __name__ == "__main__":
    unittest.main()
