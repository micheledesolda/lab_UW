import unittest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from lab_uw.plotting import Plotter


class ModelPlottingTests(unittest.TestCase):
    @staticmethod
    def _geometry():
        x = np.linspace(0.0, 1.0, 101)
        dx = float(x[1] - x[0])
        boundaries = np.array([0, 22, 27, 32, 37, 63, 68, 73, 78, 101])
        keys = (
            "side_block_1",
            "groove_sb1",
            "gouge_1",
            "groove_cb1",
            "central_block",
            "groove_cb2",
            "gouge_2",
            "groove_sb2",
            "side_block_2",
        )
        idx_dict = {
            key: np.arange(boundaries[i], boundaries[i + 1], dtype=int)
            for i, key in enumerate(keys)
        }
        for key in ("pla_1", "pzt_1", "pzt_2", "pla_2"):
            idx_dict[key] = np.array([], dtype=int)
        return x, dx, idx_dict

    def test_damping_plot_has_physical_labels_and_compact_geometry(self):
        x, dx, idx_dict = self._geometry()
        damping = np.zeros_like(x)
        damping[idx_dict["gouge_1"]] = 0.0010
        damping[idx_dict["gouge_2"]] = 0.0015

        captured = {}
        plotter = Plotter(settings={"fontsize_ticks": 8})

        def capture(*, fig, outfile_path=None, format=None):
            captured["figure"] = fig

        plotter.output_path_choice = capture
        plotter.plot_velocity_model(
            x=x,
            c=damping,
            idx_dict=idx_dict,
            dx=dx,
            pzt_layer_width=0.0,
            pla_layer_width=0.0,
            quantity="damping",
        )

        figure = captured["figure"]
        axis = figure.axes[0]
        self.assertEqual(axis.get_title(), "Kelvin–Voigt damping model")
        self.assertIn(r"\nu(x)", axis.get_ylabel())
        self.assertIn(r"cm$^2$/$\mu$s", axis.get_ylabel())
        text_labels = [text.get_text() for text in axis.texts]
        self.assertIn("Central block", text_labels)
        self.assertIn("Gouge layer 1", text_labels)
        self.assertIn("Gouge layer 2", text_labels)
        self.assertEqual(text_labels.count("Grooves"), 2)
        central_indices = idx_dict["central_block"]
        central_start = x[central_indices[0]] - 0.5 * dx
        central_end = x[central_indices[-1]] + 0.5 * dx
        groove_label_positions = [
            text.get_position()[0]
            for text in axis.texts
            if text.get_text() == "Grooves"
        ]
        self.assertTrue(all(
            central_start < position < central_end
            for position in groove_label_positions
        ))
        self.assertIn("Source", text_labels)
        self.assertIn("Receiver", text_labels)
        self.assertGreaterEqual(
            sum(isinstance(patch, Rectangle) for patch in axis.patches),
            2,
        )
        legend_labels = [text.get_text() for text in axis.get_legend().texts]
        self.assertEqual(legend_labels, [r"$\nu(x)$"])
        plt.close(figure)

    def test_velocity_plot_retains_velocity_units(self):
        x, dx, idx_dict = self._geometry()
        velocity = np.full_like(x, 0.57)
        captured = {}
        plotter = Plotter(settings={"fontsize_ticks": 8})
        plotter.output_path_choice = (
            lambda *, fig, outfile_path=None, format=None:
            captured.setdefault("figure", fig)
        )
        plotter.plot_velocity_model(
            x=x,
            c=velocity,
            idx_dict=idx_dict,
            dx=dx,
            pzt_layer_width=0.0,
            pla_layer_width=0.0,
            quantity="velocity",
        )
        figure = captured["figure"]
        axis = figure.axes[0]
        self.assertEqual(axis.get_title(), "Velocity model")
        self.assertIn(r"c(x)", axis.get_ylabel())
        self.assertNotIn(r"cm$^2$", axis.get_ylabel())
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
