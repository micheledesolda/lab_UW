import unittest

import numpy as np

from lab_uw.fd1d import (
    finite_difference_1d_kelvin_voigt,
    finite_difference_1d_receiver_trace,
    sample_receiver,
)


class VariableDensityFiniteDifferenceTests(unittest.TestCase):
    @staticmethod
    def _small_problem():
        nt, nx = 100, 41
        source = np.zeros(nx)
        source[4] = 1.0
        receiver = np.zeros(nx)
        receiver[-5] = 1.0
        time = np.arange(nt) * 0.005
        source_time = np.sin(2.0 * np.pi * 2.0 * time) * np.exp(-2.5 * time)
        return dict(
            num_x=nx,
            delta_x=0.025,
            num_t=nt,
            delta_t=0.005,
            source_spatial_function=source,
            source_time_function=source_time,
            velocity_model=np.linspace(0.42, 0.48, nx),
            damping_model=np.full(nx, 0.0002),
            boundary_condition="free",
            source_injection="body_force",
        ), receiver

    def test_omitted_density_exactly_reproduces_unit_density(self):
        common, _ = self._small_problem()
        old_convention = finite_difference_1d_kelvin_voigt(**common)
        explicit_unit_density = finite_difference_1d_kelvin_voigt(
            **common, density_model=np.ones(common["num_x"])
        )
        np.testing.assert_array_equal(old_convention, explicit_unit_density)

    def test_uniform_density_scale_does_not_change_displacement(self):
        common, _ = self._small_problem()
        unit = finite_difference_1d_kelvin_voigt(
            **common, density_model=np.ones(common["num_x"])
        )
        rescaled = finite_difference_1d_kelvin_voigt(
            **common, density_model=np.full(common["num_x"], 7.8)
        )
        np.testing.assert_allclose(rescaled, unit, rtol=2e-14, atol=2e-16)

    def test_receiver_only_matches_full_wavefield_with_density_jump(self):
        common, receiver = self._small_problem()
        density = np.ones(common["num_x"])
        density[common["num_x"] // 2 :] = 3.0
        wavefield = finite_difference_1d_kelvin_voigt(
            **common, density_model=density
        )
        stored = sample_receiver(
            wavefield,
            receiver,
            quantity="velocity",
            delta_t=common["delta_t"],
            delta_x=common["delta_x"],
            velocity_model=common["velocity_model"],
            damping_model=common["damping_model"],
            density_model=density,
        )
        streamed = finite_difference_1d_receiver_trace(
            **common,
            density_model=density,
            receiver_spatial_function=receiver,
            receiver_quantity="velocity",
        )
        np.testing.assert_allclose(streamed, stored, rtol=1e-12, atol=1e-13)

    def test_density_interface_recovers_impedance_reflection(self):
        length = 8.0
        dx = 0.01
        x = np.linspace(0.0, length, int(length / dx) + 1)
        dt = 0.005
        time = np.arange(0.0, 14.0 + 0.5 * dt, dt)
        source = np.zeros(x.size)
        source[np.argmin(abs(x - 1.0))] = 1.0 / dx
        receiver = np.zeros(x.size)
        receiver[np.argmin(abs(x - 2.0))] = 1.0
        argument = np.pi * (time - 0.5)
        source_time = (1.0 - 2.0 * argument**2) * np.exp(-argument**2)
        common = dict(
            num_x=x.size,
            delta_x=dx,
            num_t=time.size,
            delta_t=dt,
            source_spatial_function=source,
            source_time_function=source_time,
            receiver_spatial_function=receiver,
            receiver_quantity="velocity",
            velocity_model=np.full(x.size, 0.5),
            damping_model=np.zeros(x.size),
            boundary_condition="absorbing",
            source_injection="body_force",
            n_pad=300,
        )
        homogeneous = finite_difference_1d_receiver_trace(
            **common, density_model=np.ones(x.size)
        )
        density = np.ones(x.size)
        density[x >= 4.0] = 4.0
        interface = finite_difference_1d_receiver_trace(
            **common, density_model=density
        )
        reflected = interface - homogeneous

        incident_window = (time >= 1.5) & (time <= 3.0)
        reflected_window = (time >= 9.0) & (time <= 11.0)
        incident_peak = np.max(np.abs(homogeneous[incident_window]))
        reflected_peak = np.max(np.abs(reflected[reflected_window]))
        numerical_ratio = reflected_peak / incident_peak
        theoretical_ratio = abs((1.0 * 0.5 - 4.0 * 0.5) / (1.0 * 0.5 + 4.0 * 0.5))
        self.assertAlmostEqual(numerical_ratio, theoretical_ratio, delta=0.03)

        incident_indices = np.flatnonzero(incident_window)
        reflected_indices = np.flatnonzero(reflected_window)
        incident_peak_time = time[
            incident_indices[np.argmax(np.abs(homogeneous[incident_window]))]
        ]
        reflected_peak_time = time[
            reflected_indices[np.argmax(np.abs(reflected[reflected_window]))]
        ]
        # Extra reflected path: receiver -> interface -> receiver = 4 cm.
        self.assertAlmostEqual(reflected_peak_time - incident_peak_time, 8.0, delta=0.04)


if __name__ == "__main__":
    unittest.main()
