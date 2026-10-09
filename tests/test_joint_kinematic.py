import unittest

import numpy as np

from lab_uw.joint_kinematic import build_joint_constant_time_candidates
from lab_uw.velocity_contrast import dds_gouge_groove_travel_time


def _assembly(*, h1=0.084, h2=0.079):
    return {
        "wave_type": "_p",
        "gouge_thickness_1": h1,
        "gouge_thickness_2": h2,
        "side1_params": {"velocity_p": 0.5751, "h_grooves": 0.05},
        "side2_params": {"velocity_p": 0.5751, "h_grooves": 0.05},
        "central_params": {"velocity_p": 0.61, "h_grooves": 0.05},
    }


class TestJointKinematicCandidates(unittest.TestCase):
    def test_cartesian_grid_size(self):
        candidates = build_joint_constant_time_candidates(
            _assembly(),
            0.1933,
            [-0.05, 0.0, 0.05],
            [-0.02, 0.0, 0.02],
        )
        self.assertEqual(len(candidates), 9)

    def test_every_candidate_preserves_reference_stack_time(self):
        assembly = _assembly()
        c0 = 0.1933
        target = dds_gouge_groove_travel_time(assembly, c0, c0)
        candidates = build_joint_constant_time_candidates(
            assembly,
            c0,
            np.linspace(-0.10, 0.10, 5),
            np.linspace(-0.05, 0.05, 5),
        )
        for candidate in candidates:
            self.assertAlmostEqual(
                candidate["stack_travel_time_us"],
                target,
                places=11,
            )
            self.assertAlmostEqual(
                candidate["gouge_thickness_1_cm"]
                + candidate["gouge_thickness_2_cm"],
                assembly["gouge_thickness_1"]
                + assembly["gouge_thickness_2"],
                places=14,
            )

    def test_origin_recovers_homogeneous_reference(self):
        assembly = _assembly()
        c0 = 0.1933
        candidate = build_joint_constant_time_candidates(
            assembly,
            c0,
            [0.0],
            [0.0],
        )[0]
        self.assertAlmostEqual(candidate["gouge_velocity_1"], c0)
        self.assertAlmostEqual(candidate["gouge_velocity_2"], c0)
        self.assertAlmostEqual(
            candidate["gouge_thickness_1_cm"],
            assembly["gouge_thickness_1"],
        )
        self.assertAlmostEqual(
            candidate["gouge_thickness_2_cm"],
            assembly["gouge_thickness_2"],
        )


if __name__ == "__main__":
    unittest.main()
