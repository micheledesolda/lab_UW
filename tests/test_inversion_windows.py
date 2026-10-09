import unittest

import numpy as np

from lab_uw.inversion_windows import direct_misfit_interval


class InversionWindowTests(unittest.TestCase):
    def test_window_covers_search_range_but_not_full_record(self):
        time = np.arange(0.0, 30.0, 0.01)
        indices = direct_misfit_interval(
            time,
            arrival_min_us=10.0,
            arrival_max_us=11.0,
            stf_duration_us=2.0,
            padding_us=0.25,
        )
        selected = time[indices]
        self.assertGreaterEqual(selected[0], 9.75)
        self.assertLessEqual(selected[-1], 13.25)
        self.assertLess(indices.size, time.size // 4)

    def test_candidate_velocity_does_not_change_window(self):
        time = np.arange(0.0, 20.0, 0.02)
        kwargs = dict(
            arrival_min_us=7.0,
            arrival_max_us=8.0,
            stf_duration_us=1.5,
            padding_us=0.2,
        )
        first = direct_misfit_interval(time, **kwargs)
        second = direct_misfit_interval(time, **kwargs)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
