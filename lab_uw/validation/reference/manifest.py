"""Configuration loader for batch validation of homogeneous reference records.

All relative paths are resolved from the directory containing the manifest,
not from the process working directory.  One manifest represents one coherent
reference campaign: common geometry, P/S wave type, STF archive, filtering and
forward-model conventions, plus a list of acquisition stems.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Optional

from lab_uw.validation.reference.forward import ReferenceForwardConfig


TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment",
    "paths",
    "cases",
    "loading",
    "forward",
    "validation",
    "convergence",
}
EXPERIMENT_KEYS = {"metadata", "block_id", "wave_type"}
PATH_KEYS = {
    "tsv_directory",
    "tsv_suffix",
    "stf_h5",
    "calibration_directory",
    "calibration_suffix",
    "output_directory",
}
LOADING_KEYS = {
    "average_waveforms",
    "number_of_waveforms",
    "freqmin_mhz",
    "freqmax_mhz",
    "filter_order",
    "wavelet_filter_mode",
}
FORWARD_KEYS = {field.name for field in fields(ReferenceForwardConfig)}
VALIDATION_KEYS = {
    "minimum_packet_correlation",
    "max_packets",
    "max_diagnostic_lag_us",
}
CONVERGENCE_KEYS = {"cases", "points_per_wavelength"}
CASE_KEYS = {
    "name",
    "tsv",
    "calibration_json",
    "stf_name",
    "channel_name",
    "output_prefix",
}


def _reject_unknown(section: str, values: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {section}: {sorted(unknown)}")


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


@dataclass(frozen=True)
class ReferenceValidationCaseSpec:
    name: str
    tsv_path: Path
    calibration_path: Path
    stf_name: str
    channel_name: Optional[str]
    output_prefix: Path


@dataclass(frozen=True)
class ReferenceValidationManifest:
    manifest_path: Path
    metadata_path: Path
    block_id: str
    wave_type: str
    stf_h5_path: Path
    output_directory: Path
    loading: dict[str, Any]
    forward: dict[str, Any]
    validation: dict[str, Any]
    convergence_cases: tuple[str, ...]
    convergence_ppw: tuple[int, ...]
    cases: tuple[ReferenceValidationCaseSpec, ...]

    def select(self, case_name: Optional[str]) -> tuple[ReferenceValidationCaseSpec, ...]:
        if case_name is None:
            return self.cases
        selected = tuple(case for case in self.cases if case.name == case_name)
        if not selected:
            available = ", ".join(case.name for case in self.cases)
            raise ValueError(
                f"case {case_name!r} is absent from the manifest; available: {available}"
            )
        return selected


def load_reference_validation_manifest(
    manifest_path: Path | str,
    *,
    check_inputs: bool = True,
) -> ReferenceValidationManifest:
    """Read, validate and expand one reference-validation manifest."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("reference-validation manifest must contain a JSON object")
    _reject_unknown("manifest", raw, TOP_LEVEL_KEYS)
    if int(raw.get("schema_version", 1)) != 1:
        raise ValueError("unsupported reference-validation schema_version")

    base = manifest_path.parent
    experiment = dict(raw.get("experiment", {}))
    paths = dict(raw.get("paths", {}))
    loading = dict(raw.get("loading", {}))
    forward = dict(raw.get("forward", {}))
    validation = dict(raw.get("validation", {}))
    convergence = dict(raw.get("convergence", {}))
    _reject_unknown("experiment", experiment, EXPERIMENT_KEYS)
    _reject_unknown("paths", paths, PATH_KEYS)
    _reject_unknown("loading", loading, LOADING_KEYS)
    _reject_unknown("forward", forward, FORWARD_KEYS)
    _reject_unknown("validation", validation, VALIDATION_KEYS)
    _reject_unknown("convergence", convergence, CONVERGENCE_KEYS)

    for key in ("metadata", "block_id", "wave_type"):
        if key not in experiment:
            raise ValueError(f"experiment.{key} is required")
    for key in (
        "tsv_directory",
        "stf_h5",
        "calibration_directory",
        "output_directory",
    ):
        if key not in paths:
            raise ValueError(f"paths.{key} is required")

    wave_type = str(experiment["wave_type"]).lower().lstrip("_")
    if wave_type not in {"p", "s"}:
        raise ValueError("experiment.wave_type must be 'p' or 's'")

    metadata_path = _resolve(base, experiment["metadata"])
    tsv_directory = _resolve(base, paths["tsv_directory"])
    stf_h5_path = _resolve(base, paths["stf_h5"])
    calibration_directory = _resolve(base, paths["calibration_directory"])
    output_directory = _resolve(base, paths["output_directory"])
    tsv_suffix = str(paths.get("tsv_suffix", ".bscan.tsv"))
    calibration_suffix = str(
        paths.get("calibration_suffix", "_reference_calibration.json")
    )

    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty JSON list")

    cases: list[ReferenceValidationCaseSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_cases):
        if isinstance(item, str):
            values: dict[str, Any] = {"name": item}
        elif isinstance(item, dict):
            values = dict(item)
            _reject_unknown(f"cases[{index}]", values, CASE_KEYS)
        else:
            raise ValueError(f"cases[{index}] must be a string or JSON object")

        name = str(values.get("name", "")).strip()
        if not name:
            raise ValueError(f"cases[{index}].name is required")
        if name in seen:
            raise ValueError(f"duplicate case name: {name}")
        seen.add(name)

        tsv_path = (
            _resolve(base, values["tsv"])
            if "tsv" in values
            else (tsv_directory / f"{name}{tsv_suffix}").resolve()
        )
        calibration_path = (
            _resolve(base, values["calibration_json"])
            if "calibration_json" in values
            else (calibration_directory / f"{name}{calibration_suffix}").resolve()
        )
        output_prefix = (
            _resolve(base, values["output_prefix"])
            if "output_prefix" in values
            else (output_directory / name).resolve()
        )
        cases.append(
            ReferenceValidationCaseSpec(
                name=name,
                tsv_path=tsv_path,
                calibration_path=calibration_path,
                stf_name=str(values.get("stf_name", name)),
                channel_name=values.get("channel_name"),
                output_prefix=output_prefix,
            )
        )

    convergence_names = tuple(str(value) for value in convergence.get("cases", ()))
    unknown_convergence_cases = set(convergence_names) - seen
    if unknown_convergence_cases:
        raise ValueError(
            "convergence.cases contains unknown names: "
            f"{sorted(unknown_convergence_cases)}"
        )
    convergence_ppw = tuple(
        int(value) for value in convergence.get("points_per_wavelength", ())
    )
    if any(value < 2 for value in convergence_ppw):
        raise ValueError("convergence points_per_wavelength values must be >= 2")

    if check_inputs:
        _require_file(metadata_path, "metadata file")
        _require_directory(tsv_directory, "TSV directory")
        _require_file(stf_h5_path, "STF HDF5 file")
        _require_directory(calibration_directory, "calibration JSON directory")
        for case in cases:
            _require_file(case.tsv_path, f"TSV for {case.name}")
            _require_file(
                case.calibration_path,
                f"calibration JSON for {case.name}",
            )

    return ReferenceValidationManifest(
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        block_id=str(experiment["block_id"]),
        wave_type=wave_type,
        stf_h5_path=stf_h5_path,
        output_directory=output_directory,
        loading=loading,
        forward=forward,
        validation=validation,
        convergence_cases=convergence_names,
        convergence_ppw=convergence_ppw,
        cases=tuple(cases),
    )
