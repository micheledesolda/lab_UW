import json
from pathlib import Path
import tempfile
import unittest

from lab_uw.validation.reference.manifest import (
    load_reference_validation_manifest,
)
from lab_uw.validation.reference.cli import _manifest_namespace, _parser


class ReferenceValidationManifestTests(unittest.TestCase):
    def _tree(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        config_dir = root / "campaign" / "data_analysis" / "config"
        tsv_dir = root / "campaign" / "uw_data" / "only_p"
        calibration_dir = root / "campaign" / "data_analysis" / "calibration"
        output_dir = root / "campaign" / "data_analysis" / "outputs"
        metadata = root / "metadata" / "blocks_metadata.json"
        stf_h5 = root / "campaign" / "data_analysis" / "stf.h5"
        for directory in (config_dir, tsv_dir, calibration_dir, metadata.parent):
            directory.mkdir(parents=True, exist_ok=True)
        metadata.write_text("{}", encoding="utf-8")
        stf_h5.write_bytes(b"test")
        for name in ("p_a", "p_b"):
            (tsv_dir / f"{name}.bscan.tsv").write_text("test", encoding="utf-8")
            (calibration_dir / f"{name}_reference_calibration.json").write_text(
                "{}", encoding="utf-8"
            )
        manifest_path = config_dir / "reference_validation_p.json"
        manifest = {
            "schema_version": 1,
            "experiment": {
                "metadata": "../../../metadata/blocks_metadata.json",
                "block_id": "reference",
                "wave_type": "p",
            },
            "paths": {
                "tsv_directory": "../../uw_data/only_p",
                "stf_h5": "../stf.h5",
                "calibration_directory": "../calibration",
                "output_directory": "../outputs",
            },
            "cases": ["p_a", "p_b"],
            "forward": {"points_per_wavelength": 40},
            "convergence": {
                "cases": ["p_a"],
                "points_per_wavelength": [20, 40],
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return temporary, manifest_path, output_dir

    def test_paths_are_resolved_from_manifest_directory(self):
        temporary, manifest_path, output_dir = self._tree()
        self.addCleanup(temporary.cleanup)
        manifest = load_reference_validation_manifest(manifest_path)
        self.assertEqual(manifest.output_directory, output_dir.resolve())
        self.assertEqual(manifest.cases[0].name, "p_a")
        self.assertEqual(
            manifest.cases[0].tsv_path.name,
            "p_a.bscan.tsv",
        )
        self.assertEqual(manifest.cases[0].stf_name, "p_a")
        self.assertEqual(manifest.convergence_cases, ("p_a",))
        self.assertEqual(manifest.convergence_ppw, (20, 40))

    def test_single_case_selection_is_explicit(self):
        temporary, manifest_path, _ = self._tree()
        self.addCleanup(temporary.cleanup)
        manifest = load_reference_validation_manifest(manifest_path)
        self.assertEqual(manifest.select("p_b")[0].name, "p_b")
        with self.assertRaisesRegex(ValueError, "absent from the manifest"):
            manifest.select("missing")

    def test_unknown_keys_are_rejected(self):
        temporary, manifest_path, _ = self._tree()
        self.addCleanup(temporary.cleanup)
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["forward"]["point_per_wavelength"] = 40
        manifest_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unknown keys in forward"):
            load_reference_validation_manifest(manifest_path)

    def test_manifest_defaults_and_case_scoped_convergence_reach_runner(self):
        temporary, manifest_path, _ = self._tree()
        self.addCleanup(temporary.cleanup)
        manifest = load_reference_validation_manifest(manifest_path)
        cli = _parser().parse_args(["--config", str(manifest_path)])
        first = _manifest_namespace(manifest, manifest.cases[0], cli)
        second = _manifest_namespace(manifest, manifest.cases[1], cli)
        self.assertEqual(first.points_per_wavelength, 40)
        self.assertEqual(first.frequency_cutoff_mhz, 12.0)
        self.assertEqual(first.convergence_ppw, (20, 40))
        self.assertIsNone(second.convergence_ppw)

    def test_command_line_value_overrides_manifest_value(self):
        temporary, manifest_path, _ = self._tree()
        self.addCleanup(temporary.cleanup)
        manifest = load_reference_validation_manifest(manifest_path)
        cli = _parser().parse_args(
            [
                "--config",
                str(manifest_path),
                "--points-per-wavelength",
                "50",
                "--freqmax-mhz",
                "10",
            ]
        )
        values = _manifest_namespace(manifest, manifest.cases[0], cli)
        self.assertEqual(values.points_per_wavelength, 50)
        self.assertEqual(values.freqmax_mhz, 10.0)
        self.assertEqual(values.frequency_cutoff_mhz, 10.0)


if __name__ == "__main__":
    unittest.main()
