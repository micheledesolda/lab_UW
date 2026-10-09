import unittest

import numpy as np

from lab_uw.fd1d import (
    finite_difference_1d_kelvin_voigt,
    finite_difference_1d_receiver_trace,
    receiver_field,
    sample_receiver,
)
from lab_uw.simulation_setup import Grid1D, Receiver1D, SimulationTime
from lab_uw.validation.reference.convergence import phase_velocity_ratio_at_fmax


class FiniteDifferenceCoreTests(unittest.TestCase):
    def test_body_force_has_consistent_physical_time_scaling(self):
        nx = 9
        velocity = np.ones(nx)
        source_x = np.ones(nx)

        def final_value(dt, nt):
            wave = finite_difference_1d_kelvin_voigt(
                num_x=nx,
                delta_x=1.0,
                num_t=nt,
                delta_t=dt,
                source_spatial_function=source_x,
                source_time_function=np.ones(nt),
                velocity_model=velocity,
                boundary_condition="free",
                source_injection="body_force",
            )
            return wave[-1, nx // 2]

        coarse = final_value(0.01, 100)
        fine = final_value(0.005, 200)
        self.assertLess(abs(coarse - fine), 0.01)
        self.assertAlmostEqual(fine, 0.5, delta=0.01)

    def test_fixed_boundary_nodes_remain_zero(self):
        nt, nx = 40, 21
        source = np.zeros(nx)
        source[nx // 2] = 1.0
        wave = finite_difference_1d_kelvin_voigt(
            num_x=nx,
            delta_x=0.1,
            num_t=nt,
            delta_t=0.02,
            source_spatial_function=source,
            source_time_function=np.r_[1.0, np.zeros(nt - 1)],
            velocity_model=np.ones(nx),
            boundary_condition="fixed",
            source_injection="body_force",
        )
        np.testing.assert_array_equal(wave[:, 0], 0.0)
        np.testing.assert_array_equal(wave[:, -1], 0.0)

    def test_streaming_receiver_matches_stored_wavefield(self):
        nt, nx = 80, 31
        source = np.zeros(nx)
        source[3] = 1.0
        receiver = np.zeros(nx)
        receiver[-4] = 1.0
        source_time = np.sin(np.linspace(0.0, 3.0 * np.pi, nt)) * np.exp(
            -np.linspace(0.0, 5.0, nt)
        )
        common = dict(
            num_x=nx,
            delta_x=0.05,
            num_t=nt,
            delta_t=0.01,
            source_spatial_function=source,
            source_time_function=source_time,
            velocity_model=np.full(nx, 0.5),
            damping_model=np.full(nx, 0.001),
            boundary_condition="free",
            source_injection="body_force",
        )
        wave = finite_difference_1d_kelvin_voigt(**common)
        for quantity in ("displacement", "velocity", "strain", "stress"):
            stored = sample_receiver(
                wave,
                receiver,
                quantity=quantity,
                delta_t=common["delta_t"],
                delta_x=common["delta_x"],
                velocity_model=common["velocity_model"],
                damping_model=common["damping_model"],
            )
            streamed = finite_difference_1d_receiver_trace(
                **common,
                receiver_spatial_function=receiver,
                receiver_quantity=quantity,
            )
            np.testing.assert_allclose(streamed, stored, atol=1e-13, rtol=1e-12)

    def test_receiver_observables_are_explicit(self):
        t = np.arange(6.0)
        x = np.arange(5.0)
        wave = t[:, None] * (2.0 * x[None, :])
        velocity = receiver_field(
            wave, quantity="velocity", delta_t=1.0, delta_x=1.0
        )
        strain = receiver_field(
            wave, quantity="strain", delta_t=1.0, delta_x=1.0
        )
        stress = receiver_field(
            wave,
            quantity="stress",
            delta_t=1.0,
            delta_x=1.0,
            velocity_model=np.full(x.size, 3.0),
        )
        np.testing.assert_allclose(
            velocity, np.broadcast_to(2.0 * x[None, :], wave.shape)
        )
        np.testing.assert_allclose(
            strain, np.broadcast_to(2.0 * t[:, None], wave.shape)
        )
        np.testing.assert_allclose(
            stress, np.broadcast_to(18.0 * t[:, None], wave.shape)
        )

        receiver = np.zeros(x.size)
        receiver[2] = 1.0
        sampled = sample_receiver(
            wave,
            receiver,
            quantity="velocity",
            delta_t=1.0,
            delta_x=1.0,
        )
        np.testing.assert_allclose(sampled, 4.0)

    def test_grid_ends_at_physical_length(self):
        grid = Grid1D(cmin=0.3, fmax=6.0, grid_len=5.0, ppt=10)
        self.assertEqual(grid.spatial_axis[0], 0.0)
        self.assertEqual(grid.spatial_axis[-1], 5.0)
        self.assertLessEqual(grid.dx, (0.3 / 6.0) / 10.0)

    def test_theoretical_dispersion_decreases_with_ppw(self):
        ratios = [phase_velocity_ratio_at_fmax(ppw, 0.4) for ppw in (5, 10, 20)]
        self.assertLess(ratios[0], ratios[1])
        self.assertLess(ratios[1], ratios[2])
        self.assertGreater(ratios[2], 0.996)

    def test_kelvin_voigt_clock_satisfies_combined_stability_bound(self):
        clock = SimulationTime(
            observed_time=np.arange(0.0, 2.0, 0.1),
            dx=0.1,
            max_velocity=0.5,
            max_alpha=0.01,
            cfl_factor=0.4,
        )
        number = (
            0.5**2 * clock.dt**2 + 2.0 * 0.01 * clock.dt
        ) / 0.1**2
        self.assertLessEqual(number, 0.4**2 + 1e-12)

    def test_truncated_hicks_operator_can_live_on_free_face(self):
        grid = Grid1D(cmin=0.5, fmax=2.0, grid_len=5.0, ppt=10)
        for position in (0.0, 5.0):
            receiver = Receiver1D(
                position=position,
                pzt_layer_width=0.05,
                radius=4,
                extension=0.0,
                normalization="sum",
                boundary_mode="truncate",
            )
            receiver.create_spatial_function(grid.spatial_axis, grid.dx)
            self.assertAlmostEqual(float(receiver.spatial_function.sum()), 1.0)
            self.assertGreater(np.max(np.abs(receiver.spatial_function)), 0.0)


if __name__ == "__main__":
    unittest.main()
