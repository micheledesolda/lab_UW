import unittest

import numpy as np

from lab_uw.impedance_search import centered_refined_axis, material_candidates


class ImpedanceSearchGridTests(unittest.TestCase):
    def test_cartesian_grid_contains_every_material_combination(self):
        candidates = material_candidates(
            [0.18, 0.20], [0.0004, 0.0008], [2.0, 2.5, 3.0]
        )
        self.assertEqual(len(candidates), 12)
        self.assertIn((0.18, 0.0004, 2.0), candidates)
        self.assertIn((0.20, 0.0008, 3.0), candidates)

    def test_refinement_uses_one_coarse_spacing_around_minimum(self):
        refined = centered_refined_axis(
            0.19, [0.17, 0.18, 0.19, 0.20, 0.21], number_of_points=5
        )
        np.testing.assert_allclose(
            refined, [0.18, 0.185, 0.19, 0.195, 0.20]
        )

    def test_refinement_shifts_interval_at_physical_bound(self):
        refined = centered_refined_axis(
            1.5,
            [1.5, 1.75, 2.0],
            number_of_points=3,
            lower_bound=1.5,
            upper_bound=3.1,
        )
        np.testing.assert_allclose(refined, [1.5, 1.625, 1.75])

    def test_negative_damping_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "damping"):
            material_candidates([0.19], [-0.001], [2.5])


if __name__ == "__main__":
    unittest.main()
