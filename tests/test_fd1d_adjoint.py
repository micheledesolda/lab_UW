import unittest

import numpy as np

from lab_uw.fd1d import (
    discrete_model_gradient_1d_kelvin_voigt,
    finite_difference_1d_kelvin_voigt,
    interpolation_adjoint,
    receiver_adjoint,
    sample_receiver,
)


class FiniteDifferenceAdjointTests(unittest.TestCase):
    def test_in_place_adjoint_workspace_matches_copying_mode(self):
        rng = np.random.default_rng(20261004)
        nt, nx = 23, 17
        wavefield = rng.normal(size=(nt, nx))
        wavefield_gradient = rng.normal(size=(nt, nx))
        velocity = np.linspace(0.32, 0.48, nx)
        damping = np.linspace(0.0001, 0.0007, nx)
        density = np.linspace(2.1, 7.8, nx)

        copied = discrete_model_gradient_1d_kelvin_voigt(
            wavefield=wavefield,
            wavefield_gradient=wavefield_gradient,
            velocity_model=velocity,
            damping_model=damping,
            density_model=density,
            delta_x=0.05,
            delta_t=0.01,
            boundary_condition="free",
        )
        workspace = wavefield_gradient.copy()
        consumed = discrete_model_gradient_1d_kelvin_voigt(
            wavefield=wavefield,
            wavefield_gradient=workspace,
            velocity_model=velocity,
            damping_model=damping,
            density_model=density,
            delta_x=0.05,
            delta_t=0.01,
            boundary_condition="free",
            consume_wavefield_gradient=True,
        )

        for expected, actual in zip(copied, consumed):
            np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.shares_memory(consumed[2], workspace))

    def test_interpolation_adjoint_dot_product(self):
        source_time = np.linspace(0.0, 1.0, 17)
        target_time = np.linspace(0.0, 1.0, 11)
        source_values = np.sin(3.0 * source_time)
        target_gradient = np.cos(2.0 * target_time)
        forward = np.interp(target_time, source_time, source_values)
        adjoint = interpolation_adjoint(source_time, target_time, target_gradient)
        self.assertAlmostEqual(
            float(np.dot(forward, target_gradient)),
            float(np.dot(source_values, adjoint)),
            places=12,
        )

    def test_free_velocity_receiver_model_gradient(self):
        nt, nx = 90, 25
        dx, dt = 0.05, 0.01
        source_x = np.zeros(nx)
        source_x[0] = 1.0
        receiver = np.zeros(nx)
        receiver[-1] = 1.0
        source_t = np.sin(np.linspace(0.0, 4.0 * np.pi, nt)) * np.exp(
            -np.linspace(0.0, 6.0, nt)
        )
        velocity = np.linspace(0.42, 0.48, nx)
        damping = np.full(nx, 0.001)
        direction_v = np.sin(np.linspace(0.0, np.pi, nx))
        direction_nu = 0.05 * np.cos(np.linspace(0.0, np.pi, nx))

        def objective(v, nu):
            wave = finite_difference_1d_kelvin_voigt(
                num_x=nx,
                delta_x=dx,
                num_t=nt,
                delta_t=dt,
                source_spatial_function=source_x,
                source_time_function=source_t,
                velocity_model=v,
                damping_model=nu,
                boundary_condition="free",
                source_injection="body_force",
            )
            trace = sample_receiver(
                wave,
                receiver,
                quantity="velocity",
                delta_t=dt,
                delta_x=dx,
                velocity_model=v,
                damping_model=nu,
            )
            return 0.5 * float(np.dot(trace, trace)), wave, trace

        _, wave, trace = objective(velocity, damping)
        wave_gradient = receiver_adjoint(
            trace, receiver, quantity="velocity", delta_t=dt
        )
        grad_v, grad_nu, _ = discrete_model_gradient_1d_kelvin_voigt(
            wavefield=wave,
            wavefield_gradient=wave_gradient,
            velocity_model=velocity,
            damping_model=damping,
            delta_x=dx,
            delta_t=dt,
            boundary_condition="free",
        )
        predicted = float(
            np.dot(grad_v, direction_v) + np.dot(grad_nu, direction_nu)
        )
        epsilon = 1e-5
        plus = objective(
            velocity + epsilon * direction_v,
            damping + epsilon * direction_nu,
        )[0]
        minus = objective(
            velocity - epsilon * direction_v,
            damping - epsilon * direction_nu,
        )[0]
        finite_difference = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(
            predicted,
            finite_difference,
            delta=1e-5 * max(1.0, abs(finite_difference)),
        )

    def test_free_velocity_receiver_gradient_with_fixed_density_contrast(self):
        nt, nx = 100, 31
        dx, dt = 0.04, 0.008
        source_x = np.zeros(nx)
        source_x[0] = 1.0
        receiver = np.zeros(nx)
        receiver[-1] = 1.0
        source_t = np.sin(np.linspace(0.0, 5.0 * np.pi, nt)) * np.exp(
            -np.linspace(0.0, 7.0, nt)
        )
        velocity = np.linspace(0.38, 0.50, nx)
        damping = np.linspace(0.0002, 0.0008, nx)
        density = np.full(nx, 7.8)
        density[nx // 3 : 2 * nx // 3] = 2.4
        direction_v = np.sin(np.linspace(0.0, np.pi, nx))
        direction_nu = 0.03 * np.cos(np.linspace(0.0, np.pi, nx))

        def objective(v, nu):
            wave = finite_difference_1d_kelvin_voigt(
                num_x=nx,
                delta_x=dx,
                num_t=nt,
                delta_t=dt,
                source_spatial_function=source_x,
                source_time_function=source_t,
                velocity_model=v,
                damping_model=nu,
                density_model=density,
                boundary_condition="free",
                source_injection="body_force",
            )
            trace = sample_receiver(
                wave,
                receiver,
                quantity="velocity",
                delta_t=dt,
                delta_x=dx,
                velocity_model=v,
                damping_model=nu,
                density_model=density,
            )
            return 0.5 * float(np.dot(trace, trace)), wave, trace

        _, wave, trace = objective(velocity, damping)
        wave_gradient = receiver_adjoint(
            trace, receiver, quantity="velocity", delta_t=dt
        )
        grad_v, grad_nu, _ = discrete_model_gradient_1d_kelvin_voigt(
            wavefield=wave,
            wavefield_gradient=wave_gradient,
            velocity_model=velocity,
            damping_model=damping,
            density_model=density,
            delta_x=dx,
            delta_t=dt,
            boundary_condition="free",
        )
        predicted = float(
            np.dot(grad_v, direction_v) + np.dot(grad_nu, direction_nu)
        )
        epsilon = 1e-5
        finite_difference = (
            objective(
                velocity + epsilon * direction_v,
                damping + epsilon * direction_nu,
            )[0]
            - objective(
                velocity - epsilon * direction_v,
                damping - epsilon * direction_nu,
            )[0]
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            predicted,
            finite_difference,
            delta=1e-5 * max(1.0, abs(finite_difference)),
        )


if __name__ == "__main__":
    unittest.main()
