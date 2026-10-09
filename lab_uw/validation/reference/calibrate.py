"""Deterministic multi-record calibration for the 1D reference block.

The search is a Cartesian grid declared in a JSON manifest.  Only the direct
packet contributes to parameter ranking.  Later packets are evaluated once on
the best candidate of each structural model and are never used to re-fit its
parameters or amplitude.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from itertools import product
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from lab_uw.validation.reference.case import ReferenceCase, load_reference_case
from lab_uw.validation.reference.forward import (
    ReferenceForwardConfig,
    ReferenceGeometry,
    load_reference_geometry,
    run_reference_forward,
)
from lab_uw.validation.reference.objective import (
    ReferenceScore,
    evaluate_reference_trace,
    reliable_packet_orders,
)


GEOMETRY_GRID_FIELDS = {
    "steel_velocity_cm_per_us",
    "pzt_velocity_cm_per_us",
    "pla_velocity_cm_per_us",
}
CONFIG_FIELDS = {field.name for field in fields(ReferenceForwardConfig)}
GRID_ALIASES = {
    "pzt_depth_fraction": ("source_depth_fraction", "receiver_depth_fraction"),
    "operator_extension_cm": ("source_extension_cm", "receiver_extension_cm"),
    "operator_inset_cm": ("source_inset_cm", "receiver_inset_cm"),
}


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_cases(manifest: dict[str, Any], manifest_dir: Path) -> list[ReferenceCase]:
    loading = dict(manifest.get("loading", {}))
    cases = []
    for item in manifest["cases"]:
        cases.append(
            load_reference_case(
                tsv_path=_resolve(manifest_dir, item["tsv"]),
                stf_h5_path=_resolve(manifest_dir, item["stf_h5"]),
                calibration_path=_resolve(manifest_dir, item["calibration_json"]),
                stf_name=item.get("stf_name"),
                channel_name=item.get("channel_name"),
                average_waveforms=bool(loading.get("average_waveforms", True)),
                number_of_waveforms=loading.get("number_of_waveforms"),
                freqmin_mhz=loading.get("freqmin_mhz", 0.5),
                freqmax_mhz=loading.get("freqmax_mhz", 6.0),
                filter_order=int(loading.get("filter_order", 4)),
                wavelet_filter_mode=loading.get(
                    "wavelet_filter_mode", "already_filtered"
                ),
            )
        )
    if not cases:
        raise ValueError("manifest contains no reference cases")
    return cases


def _candidate_updates(grid: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if not grid:
        yield {}
        return
    unknown = set(grid) - CONFIG_FIELDS - GEOMETRY_GRID_FIELDS - set(GRID_ALIASES)
    if unknown:
        raise ValueError(f"unknown calibration-grid fields: {sorted(unknown)}")
    names = sorted(grid)
    value_lists = []
    for name in names:
        values = grid[name]
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"grid field {name!r} must be a non-empty JSON list")
        value_lists.append(values)
    for values in product(*value_lists):
        expanded: dict[str, Any] = {}
        for name, value in zip(names, values):
            targets = GRID_ALIASES.get(name, (name,))
            for target in targets:
                expanded[target] = value
        yield expanded


def _make_candidate(
    base_geometry: ReferenceGeometry,
    base_config: ReferenceForwardConfig,
    updates: dict[str, Any],
) -> tuple[ReferenceGeometry, ReferenceForwardConfig]:
    geometry_updates = {
        name: value for name, value in updates.items() if name in GEOMETRY_GRID_FIELDS
    }
    config_updates = {
        name: value for name, value in updates.items() if name in CONFIG_FIELDS
    }
    if "boundary_impedance" in config_updates and isinstance(
        config_updates["boundary_impedance"], list
    ):
        config_updates["boundary_impedance"] = tuple(
            config_updates["boundary_impedance"]
        )
    geometry = replace(base_geometry, **geometry_updates)
    config = replace(base_config, **config_updates)
    if config.embedding_mode == "free_steel":
        # PZT depth exists only in the layered hypothesis.  Canonicalising it
        # here prevents duplicate free-steel simulations in mixed grids.
        config = replace(
            config,
            source_depth_fraction=1.0,
            receiver_depth_fraction=1.0,
        )
    return geometry, config


def _candidate_identity(
    geometry: ReferenceGeometry,
    config: ReferenceForwardConfig,
) -> str:
    return json.dumps(
        {"geometry": asdict(geometry), "config": asdict(config)},
        sort_keys=True,
    )


def _structural_model(config: ReferenceForwardConfig) -> tuple[str, ...]:
    return (
        config.embedding_mode,
        config.source_injection,
        config.receiver_quantity,
        config.boundary_condition,
    )


def _direct_stop_time(case: ReferenceCase, margin_us: float) -> float:
    absolute_end = case.direct_start_us + case.packet_span_us + margin_us
    return float(absolute_end - case.observed_time_us[0])


def _score_direct(
    case: ReferenceCase,
    geometry: ReferenceGeometry,
    config: ReferenceForwardConfig,
    *,
    diagnostic_lag_us: float,
) -> tuple[ReferenceScore, dict[str, Any]]:
    result = run_reference_forward(
        case,
        geometry,
        config,
        stop_time_us=_direct_stop_time(case, diagnostic_lag_us),
    )
    score = evaluate_reference_trace(
        observed_time_us=result.observed_time_us,
        observed_waveform=case.observed_waveform[: result.observed_time_us.size],
        synthetic_waveform=result.synthetic_waveform,
        calibration=case.calibration,
        calibration_orders=(0,),
        validation_orders=(0,),
        max_diagnostic_lag_us=diagnostic_lag_us,
    )
    return score, result.metadata()


def _reflection_holdout_score(score: ReferenceScore) -> Optional[float]:
    terms = [
        packet.normalized_l2 + (1.0 - packet.correlation_similarity)
        for packet in score.packets
        if packet.order > 0
    ]
    return float(np.mean(terms)) if terms else None


def calibrate_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    manifest_dir = manifest_path.parent
    cases = _load_cases(manifest, manifest_dir)

    metadata_path = _resolve(manifest_dir, manifest["metadata"])
    base_geometry = load_reference_geometry(
        metadata_path,
        block_id=manifest["block_id"],
        wave_type=manifest["wave_type"],
        steel_thickness_cm=cases[0].block_thickness_cm,
    )
    for case in cases[1:]:
        if not np.isclose(
            case.block_thickness_cm,
            base_geometry.steel_thickness_cm,
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError("all calibration cases must use the same block thickness")

    base_config_values = dict(manifest.get("base_config", {}))
    unknown_base = set(base_config_values) - CONFIG_FIELDS
    if unknown_base:
        raise ValueError(f"unknown base_config fields: {sorted(unknown_base)}")
    if "boundary_impedance" in base_config_values and isinstance(
        base_config_values["boundary_impedance"], list
    ):
        base_config_values["boundary_impedance"] = tuple(
            base_config_values["boundary_impedance"]
        )
    base_config = ReferenceForwardConfig(**base_config_values)

    calibration = dict(manifest.get("calibration", {}))
    diagnostic_lag_us = float(calibration.get("max_diagnostic_lag_us", 0.25))
    max_candidates = int(calibration.get("max_candidates", 5000))
    updates = list(_candidate_updates(dict(manifest.get("grid", {}))))
    if len(updates) > max_candidates:
        raise ValueError(
            f"Cartesian grid has {len(updates)} candidates, above max_candidates={max_candidates}"
        )

    candidates = []
    seen = set()
    for candidate_index, update in enumerate(updates):
        geometry, config = _make_candidate(base_geometry, base_config, update)
        identity = _candidate_identity(geometry, config)
        if identity in seen:
            continue
        seen.add(identity)
        per_case = []
        for case in cases:
            score, numerical_metadata = _score_direct(
                case,
                geometry,
                config,
                diagnostic_lag_us=diagnostic_lag_us,
            )
            per_case.append(
                {
                    "stem": case.stem,
                    "calibration_score": score.calibration_score,
                    "amplitude_scale": score.amplitude_scale,
                    "direct_packet": score.to_dict()["packets"][0],
                    "numerical_grid": {
                        "realized_dx_cm": numerical_metadata["realized_dx_cm"],
                        "realized_dt_us": numerical_metadata["realized_dt_us"],
                    },
                }
            )
        candidates.append(
            {
                "candidate_index": candidate_index,
                "geometry": asdict(geometry),
                "config": asdict(config),
                "mean_calibration_score": float(
                    np.mean([entry["calibration_score"] for entry in per_case])
                ),
                "cases": per_case,
            }
        )

    candidates.sort(key=lambda item: item["mean_calibration_score"])
    for rank, item in enumerate(candidates, start=1):
        item["calibration_rank"] = rank

    # Validate only the best direct-packet calibration of each structural
    # forward model.  This prevents reflection data from tuning continuous
    # parameters while still letting it discriminate model hypotheses.
    validation = dict(manifest.get("validation", {}))
    min_corr = float(validation.get("minimum_packet_correlation", 0.5))
    max_packets = validation.get("max_packets")
    validated = []
    chosen_structures = set()
    for candidate in candidates:
        config = ReferenceForwardConfig(**candidate["config"])
        structure = _structural_model(config)
        if structure in chosen_structures:
            continue
        chosen_structures.add(structure)
        geometry = ReferenceGeometry(**candidate["geometry"])
        case_results = []
        for case in cases:
            result = run_reference_forward(case, geometry, config)
            orders = reliable_packet_orders(
                case.calibration,
                minimum_signed_correlation=min_corr,
                max_packets=max_packets,
            )
            score = evaluate_reference_trace(
                observed_time_us=result.observed_time_us,
                observed_waveform=case.observed_waveform,
                synthetic_waveform=result.synthetic_waveform,
                calibration=case.calibration,
                calibration_orders=(0,),
                validation_orders=orders,
                max_diagnostic_lag_us=diagnostic_lag_us,
            )
            case_results.append(
                {
                    "stem": case.stem,
                    "holdout_reflection_score": _reflection_holdout_score(score),
                    "score": score.to_dict(),
                }
            )
        finite_holdout = [
            item["holdout_reflection_score"]
            for item in case_results
            if item["holdout_reflection_score"] is not None
        ]
        validated.append(
            {
                "calibration_rank": candidate["calibration_rank"],
                "structural_model": {
                    "embedding_mode": structure[0],
                    "source_injection": structure[1],
                    "receiver_quantity": structure[2],
                    "boundary_condition": structure[3],
                },
                "mean_holdout_reflection_score": (
                    float(np.mean(finite_holdout)) if finite_holdout else None
                ),
                "cases": case_results,
            }
        )

    return {
        "manifest": str(manifest_path),
        "method": {
            "search": "deterministic Cartesian grid",
            "calibration_packets": [0],
            "amplitude": "one non-negative scale per record, fitted on direct packet",
            "reflection_policy": (
                "later packets validate the best direct-calibrated candidate of "
                "each structural model; they do not re-fit parameters"
            ),
            "negative_correlation": "zero similarity; absolute correlation is not used",
            "stf_deformation": "none",
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "holdout_validation": validated,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    result = calibrate_manifest(args.manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"evaluated {result['candidate_count']} deterministic candidates")
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
