import unittest

import numpy as np

from lab_uw.spectral_diagnostics import compare_spectral_shapes


class SpectralDiagnosticsTests(unittest.TestCase):
    def test_distance_is_zero_for_a_pure_amplitude_change(self):
        time = np.arange(0.0, 8.0, 0.01)
        waveform = (
            np.sin(2.0 * np.pi * 2.0 * time)
            + 0.35 * np.sin(2.0 * np.pi * 4.5 * time)
        ) * np.exp(-0.2 * time)
        result = compare_spectral_shapes(
            time,
            waveform,
            time,
            7.5 * waveform,
            frequency_min_mhz=0.5,
            frequency_max_mhz=8.0,
        )
        self.assertAlmostEqual(result.shape_distance, 0.0, places=12)

    def test_distance_detects_a_frequency_change(self):
        time = np.arange(0.0, 8.0, 0.01)
        first = np.sin(2.0 * np.pi * 2.0 * time)
        second = np.sin(2.0 * np.pi * 5.0 * time)
        result = compare_spectral_shapes(
            time,
            first,
            time,
            second,
            frequency_min_mhz=0.5,
            frequency_max_mhz=8.0,
        )
        self.assertGreater(result.shape_distance, 0.9)
        self.assertLessEqual(result.shape_distance, 1.0)


if __name__ == "__main__":
    unittest.main()
