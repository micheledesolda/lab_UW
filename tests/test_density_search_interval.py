import unittest

import numpy as np

from lab_uw.inversion_windows import sample_interval_mask


class DensityNormalizationIntervalTests(unittest.TestCase):
    def test_integer_indices_are_converted_to_record_length_mask(self):
        indices = np.arange(190, 306)
        result = sample_interval_mask(indices, 500)
        self.assertEqual(result.dtype, np.bool_)
        self.assertEqual(result.shape, (500,))
        np.testing.assert_array_equal(np.flatnonzero(result), indices)

    def test_boolean_mask_and_equivalent_indices_give_same_result(self):
        time = np.linspace(0.0, 10.0, 500)
        mask = (time >= 3.8) & (time <= 6.1)
        from_mask = sample_interval_mask(mask, time.size)
        from_indices = sample_interval_mask(np.flatnonzero(mask), time.size)
        np.testing.assert_array_equal(from_indices, from_mask)


if __name__ == "__main__":
    unittest.main()
