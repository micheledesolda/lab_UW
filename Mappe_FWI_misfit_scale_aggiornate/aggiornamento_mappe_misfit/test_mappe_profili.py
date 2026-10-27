import unittest
import numpy as np
import mappe_profili_fwi as module


class ProfileColoursTests(unittest.TestCase):
    def test_each_difference_panel_uses_own_extremes(self):
        small = np.array([[0., .002], [-.003, 0.]])
        large = np.array([[0., 300.], [-500., 0.]])
        norm, info = module.norm_for(small, 'P', 'velocity', 'consecutive', module.COLORS, False)
        other, second = module.norm_for(large, 'S', 'velocity', 'first', module.COLORS, False)
        self.assertAlmostEqual(info['limits'][1], .003)
        self.assertAlmostEqual(info['linear_core'][1], .003/10000)
        self.assertEqual(second['limits'][1], 500.)
        self.assertAlmostEqual(float(norm(0)), .5)
        self.assertLess(float(norm(-.003)), .01)
        self.assertGreater(float(norm(.002)), .9)

    def test_absolute_scales_shared_but_not_difference_scales(self):
        models = {('P',1): dict(velocity_m_s=np.array([[400., 5000.], [410., 5010.]]),
                                damping_m2_s=np.array([[.001,.002], [.002,.003]]))}
        settings = module.common_colour_settings(models, module.COLORS)
        self.assertGreaterEqual(settings['velocity_absolute'][1], 5010.)
        self.assertIsNone(settings['velocity_difference_limit'])
        self.assertIsNone(settings['damping_difference_limit_P'])

    def test_units_differences_and_zeros_are_not_modified(self):
        values = np.array([[0., 1e-5], [-2e-5, 0.]])
        before = values.copy()
        norm, info = module.norm_for(values, 'S', 'damping', 'evaluated', module.COLORS, False)
        np.testing.assert_array_equal(values, before)
        self.assertAlmostEqual(info['limits'][1], 2e-5)
        self.assertAlmostEqual(float(norm(0)), .5)
        norm, info = module.norm_for(np.zeros((2,3)), 'P', 'velocity', 'first', module.COLORS, False)
        self.assertTrue(info['all_values_zero'])
        self.assertAlmostEqual(float(norm(0)), .5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
