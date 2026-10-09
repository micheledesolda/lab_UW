import unittest

import numpy as np

from lab_uw.gouge_thickness import (
    build_constant_total_thickness_candidates,
    constant_total_thickness_pair,
    select_thickness_minimum,
)


def _assembly(*, h1=0.079, h2=0.079):
    return {
        "gouge_thickness_1": h1,
        "gouge_thickness_2": h2,
    }


class TestGougeThickness(unittest.TestCase):
    def test_zero_offset_recovers_reference_geometry(self):
        h1, h2 = constant_total_thickness_pair(0.079, 0.081, 0.0)
        self.assertAlmostEqual(h1, 0.079)
        self.assertAlmostEqual(h2, 0.081)

    def test_offset_is_converted_from_mm_to_cm(self):
        h1, h2 = constant_total_thickness_pair(0.079, 0.079, 0.05)
        self.assertAlmostEqual(h1, 0.084)
        self.assertAlmostEqual(h2, 0.074)

    def test_total_thickness_is_exactly_preserved(self):
        reference_total = 0.079 + 0.081
        for offset in np.linspace(-0.05, 0.05, 11):
            h1, h2 = constant_total_thickness_pair(0.079, 0.081, offset)
            self.assertAlmostEqual(h1 + h2, reference_total, places=14)

    def test_common_velocity_travel_time_is_preserved(self):
        candidates = build_constant_total_thickness_candidates(
            _assembly(),
            [-0.05, 0.0, 0.05],
            common_velocity_cm_us=0.1956,
        )
        target = candidates[0]["target_gouge_travel_time_us"]
        for candidate in candidates:
            self.assertAlmostEqual(
                candidate["gouge_travel_time_us"],
                target,
                places=14,
            )

    def test_offset_cannot_make_a_layer_non_positive(self):
        with self.assertRaises(ValueError):
            constant_total_thickness_pair(0.001, 0.001, 0.02)

    def test_reciprocal_minima_are_retained_and_sign_is_not_interpreted(self):
        results = [
            {"thickness_offset_mm": -0.03, "misfit": 10.0},
            {"thickness_offset_mm": 0.00, "misfit": 10.5},
            {"thickness_offset_mm": 0.03, "misfit": 10.0 + 1.0e-8},
        ]
        best, equivalent = select_thickness_minimum(
            results, relative_tolerance=1.0e-6
        )
        np.testing.assert_allclose(equivalent, [-0.03, 0.03])
        self.assertAlmostEqual(best["thickness_offset_mm"], 0.03)


if __name__ == "__main__":
    unittest.main()
