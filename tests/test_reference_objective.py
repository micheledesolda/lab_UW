import unittest

import numpy as np

from lab_uw.validation.reference.case import (
    ReferenceCase,
    _acquisition_stem,
    _calibrated_wavelet_time_axis,
)
from lab_uw.validation.reference.objective import (
    correlation_similarity,
    evaluate_reference_trace,
    reliable_packet_orders,
    signed_correlation,
)
from lab_uw.validation.reference.forward import (
    ReferenceForwardConfig,
    ReferenceGeometry,
    run_reference_forward,
)


def calibration(correlations=(0.9, -1.0)):
    return {
        "infile_stem": "synthetic",
        "refined_direct_arrival": {
            "best_t_start_us": 1.0,
            "best_t_end_us": 1.5,
            "effective_wavelet_start_us": 1.0,
            "direct_arrival_span_us": 0.5,
        },
        "zero_angle_packets": {
            "arrival_times_us": [1.0, 3.0, 5.0],
            "packet_window_span_us": 0.5,
        },
        "correlation": {
            "per_reflection": [
                {"arrival_time_us": 3.0, "corr_coeff": correlations[0]},
                {"arrival_time_us": 5.0, "corr_coeff": correlations[1]},
            ]
        },
        "reference_block": {
            "thickness_cm": 0.5,
            "estimated_velocity_cm_per_us": 0.5,
        },
    }


class ReferenceObjectiveTests(unittest.TestCase):
    def test_mistras_compound_suffix_maps_to_calibration_stem(self):
        self.assertEqual(
            _acquisition_stem("p_70V_100nm.bscan.tsv"),
            "p_70V_100nm",
        )

    def test_historical_zero_based_hdf5_axis_is_reconstructed_from_json(self):
        observed_time = np.arange(0.0, 6.0, 0.01)
        stored_time = np.arange(0.0, 0.5, 0.01)
        recovered, reconstructed = _calibrated_wavelet_time_axis(
            observed_time_us=observed_time,
            stored_wavelet_time_us=stored_time,
            wavelet_size=stored_time.size,
            calibration=calibration(),
        )
        self.assertTrue(reconstructed)
        np.testing.assert_allclose(recovered, np.arange(1.0, 1.5, 0.01))

    def test_inverse_polarity_is_not_a_good_match(self):
        a = np.array([0.0, 1.0, -0.5, 0.25])
        self.assertAlmostEqual(signed_correlation(a, -a), -1.0)
        self.assertEqual(correlation_similarity(-1.0), 0.0)

    def test_reliable_packets_use_signed_not_absolute_correlation(self):
        self.assertEqual(
            reliable_packet_orders(calibration(), minimum_signed_correlation=0.5),
            (0, 1),
        )

    def test_amplitude_is_fit_only_on_direct_packet(self):
        time = np.arange(0.0, 6.0, 0.01)
        observed = np.zeros_like(time)
        synthetic = np.zeros_like(time)
        pulse = np.sin(np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False))
        observed[100:150] = 2.0 * pulse
        synthetic[100:150] = pulse
        observed[300:350] = pulse
        synthetic[300:350] = 2.0 * pulse

        score = evaluate_reference_trace(
            observed_time_us=time,
            observed_waveform=observed,
            synthetic_waveform=synthetic,
            calibration=calibration(),
            calibration_orders=(0,),
            validation_orders=(0, 1),
        )
        self.assertAlmostEqual(score.amplitude_scale, 2.0)
        direct, reflection = score.packets
        self.assertLess(direct.normalized_l2, 1e-12)
        self.assertGreater(reflection.normalized_l2, 1.0)

    def test_reference_case_enforces_odd_multiple_arrivals(self):
        time = np.arange(0.0, 6.0, 0.01)
        case = ReferenceCase(
            stem="synthetic",
            observed_time_us=time,
            observed_waveform=np.zeros_like(time),
            wavelet_time_absolute_us=np.arange(1.0, 1.5, 0.01),
            wavelet_waveform=np.ones(50),
            calibration=calibration(),
            filter_freqmin_mhz=0.5,
            filter_freqmax_mhz=4.0,
            wavelet_filter_mode="already_filtered",
        )
        case.validate()
        self.assertEqual(case.wavelet_time_relative_us[0], 0.0)

    def test_free_steel_repeats_packet_at_three_t_without_polarity_flip(self):
        dt = 0.002
        time = np.arange(0.0, 7.5 + dt / 2.0, dt)
        wavelet_time = np.arange(2.0, 3.0 + dt / 2.0, dt)
        relative = wavelet_time - 2.0
        wavelet = (
            1.0 - 2.0 * (np.pi * 1.2 * (relative - 0.4)) ** 2
        ) * np.exp(-(np.pi * 1.2 * (relative - 0.4)) ** 2)
        case = ReferenceCase(
            stem="synthetic",
            observed_time_us=time,
            observed_waveform=np.zeros_like(time),
            wavelet_time_absolute_us=wavelet_time,
            wavelet_waveform=wavelet,
            calibration=calibration(correlations=(1.0, 1.0)) | {
                "refined_direct_arrival": {
                    "best_t_start_us": 2.0,
                    "effective_wavelet_start_us": 2.0,
                    "direct_arrival_span_us": 1.0,
                },
                "zero_angle_packets": {
                    "arrival_times_us": [2.0, 6.0],
                    "packet_window_span_us": 1.0,
                },
                "correlation": {
                    "per_reflection": [
                        {"arrival_time_us": 6.0, "corr_coeff": 1.0}
                    ]
                },
                "reference_block": {
                    "thickness_cm": 1.0,
                    "estimated_velocity_cm_per_us": 0.5,
                },
            },
            filter_freqmin_mhz=0.5,
            filter_freqmax_mhz=2.0,
            wavelet_filter_mode="already_filtered",
        )
        geometry = ReferenceGeometry(
            block_id="synthetic",
            wave_type="p",
            steel_thickness_cm=1.0,
            steel_velocity_cm_per_us=0.5,
            pzt_thickness_cm=0.05,
            pzt_velocity_cm_per_us=0.4,
        )
        result = run_reference_forward(
            case,
            geometry,
            ReferenceForwardConfig(
                frequency_cutoff_mhz=2.0,
                points_per_wavelength=30,
                boundary_condition="free",
                source_injection="body_force",
                receiver_quantity="velocity",
            ),
        )
        direct = (time >= 2.0) & (time < 3.0)
        first_multiple = (time >= 6.0) & (time < 7.0)
        self.assertGreater(
            signed_correlation(
                result.synthetic_waveform[direct],
                result.synthetic_waveform[first_multiple],
            ),
            0.98,
        )


if __name__ == "__main__":
    unittest.main()
