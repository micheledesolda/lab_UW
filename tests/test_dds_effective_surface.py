import unittest

import numpy as np

from lab_uw.fd1d import finite_difference_1d_receiver_trace
from lab_uw.simulation_setup import (
    Grid1D,
    Receiver1D,
    Source1D,
    VelocityModel1D_DDS,
)


class DDSEffectiveSurfaceTests(unittest.TestCase):
    def _model_and_trace(self, pzt_width):
        dimensions = (1.0, 0.10, 2.0, 0.10, 1.0)
        total_length = float(sum(dimensions))
        grid = Grid1D(cmin=0.30, fmax=1.0, grid_len=total_length, ppt=10)
        model = VelocityModel1D_DDS(
            x=grid.spatial_axis,
            sample_dimensions=dimensions,
            x_transmitter=0.0,
            x_receiver=total_length,
            pzt_layer_width=pzt_width,
            pla_layer_width=10.0 * pzt_width,
            h_groove_side=0.05,
            h_groove_central=0.05,
            block_velocity={"side1": 0.57, "central": 0.56, "side2": 0.55},
            gouge_velocity=(0.35, 0.37),
            gouge_damping=(0.001, 0.002),
            pzt_velocity=9.0,
            pla_velocity=8.0,
            include_transducer_layers=False,
        )

        nt, dt = 1000, 0.01
        time = np.arange(nt) * dt
        source = Source1D(
            stf_time=time,
            stf_waveform=np.sin(2.0 * np.pi * 2.0 * time) * np.exp(-4.0 * time),
            position=0.0,
            pzt_layer_width=pzt_width,
            radius=4,
            extension=0.0,
            normalization="integral",
            boundary_mode="truncate",
        )
        source.interpolate_time_function(dt, time)
        source.create_spatial_function(grid.spatial_axis, grid.dx)
        receiver = Receiver1D(
            position=total_length,
            pzt_layer_width=pzt_width,
            radius=4,
            extension=0.0,
            normalization="sum",
            boundary_mode="truncate",
        )
        receiver.create_spatial_function(grid.spatial_axis, grid.dx)
        trace = finite_difference_1d_receiver_trace(
            num_x=grid.total_grid_points,
            delta_x=grid.dx,
            num_t=nt,
            delta_t=dt,
            source_spatial_function=source.spatial_function,
            source_time_function=source.time_function,
            receiver_spatial_function=receiver.spatial_function,
            receiver_quantity="velocity",
            velocity_model=model.velocity_array,
            damping_model=model.damping_array,
            boundary_condition="free",
            source_injection="body_force",
        )
        return model, trace

    def test_effective_surface_is_independent_of_pzt_dimensions(self):
        model_a, trace_a = self._model_and_trace(0.01)
        model_b, trace_b = self._model_and_trace(0.80)
        np.testing.assert_array_equal(model_a.velocity_array, model_b.velocity_array)
        np.testing.assert_array_equal(model_a.damping_array, model_b.damping_array)
        np.testing.assert_array_equal(trace_a, trace_b)
        self.assertEqual(model_a.idx_dict["pzt_1"].size, 0)
        self.assertEqual(model_a.idx_dict["pla_2"].size, 0)


if __name__ == "__main__":
    unittest.main()
