"""Validate one effective wavelet against its homogeneous reference record.

Example
-------
python -m lab_uw.validation.reference \
  --tsv p_70V_500nm.tsv --stf-h5 stf.h5 \
  --calibration-json p_70V_500nm_reference_calibration.json \
  --metadata blocks_metadata.json --block-id on_bench_STF2 --wave-type p \
  --output-prefix results/p_70V_500nm_free_steel
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from lab_uw.validation.reference.case import load_reference_case
from lab_uw.validation.reference.convergence import check_reference_grid_convergence
from lab_uw.validation.reference.forward import (
    ReferenceForwardConfig,
    load_reference_geometry,
    run_reference_forward,
)
from lab_uw.validation.reference.objective import (
    evaluate_reference_trace,
    reliable_packet_orders,
)
from lab_uw.validation.reference.manifest import (
    ReferenceValidationCaseSpec,
    ReferenceValidationManifest,
    load_reference_validation_manifest,
)


DEFAULT_OPTIONS = {
    "source_duration_us": None,  # STEEL_REFERENCE_AMPLITUDE_V1
    "stf_name": None,
    "channel_name": None,
    "average_waveforms": True,
    "number_of_waveforms": None,
    "freqmin_mhz": 0.5,
    "freqmax_mhz": 12.0,
    "filter_order": 4,
    "wavelet_filter_mode": "already_filtered",
    "embedding_mode": "free_steel",
    "steel_velocity": None,
    "pzt_velocity": None,
    "frequency_cutoff_mhz": 12.0,
    "points_per_wavelength": 40,
    "cfl_factor": 0.4,
    "boundary_condition": "free",
    "boundary_impedance": None,
    "source_injection": "body_force",
    "receiver_quantity": "velocity",
    "source_depth_fraction": 1.0,
    "receiver_depth_fraction": 1.0,
    "source_inset_cm": 0.0,
    "receiver_inset_cm": 0.0,
    "source_extension_cm": 0.0,
    "receiver_extension_cm": 0.0,
    "interpolation_radius_cells": 4,
    "source_normalization": "auto",
    "receiver_normalization": "sum",
    "spatial_boundary_mode": "truncate",
    "source_amplitude": 1.0,
    "minimum_packet_correlation": 0.5,
    "max_packets": None,
    "max_diagnostic_lag_us": 0.25,
    "convergence_ppw": None,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 1D homogeneous reference model. Amplitude is fitted on "
            "the direct packet only; reflections remain validation data."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Batch manifest; paths inside it are relative to the manifest file.",
    )
    parser.add_argument(
        "--case",
        help="With --config, validate only this acquisition stem.",
    )
    parser.add_argument("--tsv", type=Path)
    parser.add_argument("--stf-h5", type=Path)
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--stf-name")
    parser.add_argument("--channel-name")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--block-id")
    parser.add_argument("--wave-type", choices=("p", "s", "_p", "_s"))
    parser.add_argument("--output-prefix", type=Path)

    parser.add_argument("--freqmin-mhz", type=float)
    parser.add_argument("--freqmax-mhz", type=float)
    parser.add_argument("--filter-order", type=int)
    parser.add_argument(
        "--wavelet-filter-mode",
        choices=("already_filtered", "apply", "from_tsv"),
    )
    parser.add_argument("--number-of-waveforms", type=int)
    parser.add_argument(
        "--average-waveforms",
        dest="average_waveforms",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-average-waveforms",
        dest="average_waveforms",
        action="store_false",
    )

    parser.add_argument(
        "--embedding-mode",
        choices=("free_steel", "full_area_pzt_stack"),
    )
    parser.add_argument("--steel-velocity", type=float)
    parser.add_argument("--pzt-velocity", type=float)
    parser.add_argument("--points-per-wavelength", type=int)
    parser.add_argument("--cfl-factor", type=float)
    parser.add_argument(
        "--boundary-condition",
        choices=("free", "fixed", "impedance", "absorbing"),
    )
    parser.add_argument("--boundary-impedance", type=float, nargs="+")
    parser.add_argument(
        "--source-injection",
        choices=("body_force", "displacement_increment"),
    )
    parser.add_argument(
        "--receiver-quantity",
        choices=("displacement", "velocity", "strain", "stress"),
    )
    parser.add_argument("--source-depth-fraction", type=float)
    parser.add_argument("--receiver-depth-fraction", type=float)
    parser.add_argument("--source-inset-cm", type=float)
    parser.add_argument("--receiver-inset-cm", type=float)
    parser.add_argument("--source-extension-cm", type=float)
    parser.add_argument("--receiver-extension-cm", type=float)
    parser.add_argument("--interpolation-radius-cells", type=int)
    parser.add_argument(
        "--source-normalization",
        choices=("auto", "sum", "l1", "l2", "integral", "legacy_peak"),
    )
    parser.add_argument(
        "--receiver-normalization",
        choices=("sum", "l1", "l2", "integral", "legacy_peak"),
    )
    parser.add_argument(
        "--spatial-boundary-mode",
        choices=("truncate", "legacy_odd"),
    )
    parser.add_argument("--source-amplitude", type=float)
    parser.add_argument("--minimum-packet-correlation", type=float)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--max-diagnostic-lag-us", type=float)
    parser.add_argument(
        "--convergence-ppw",
        type=int,
        nargs="+",
        help="Optional PPW levels, e.g. 10 15 20 30; compares the direct packet.",
    )
    parser.add_argument("--source-duration-us", type=float,
        help="Crop the source BEFORE fitting its steel-reference gain; export PREFIX.source.npz for the DDS.")
    return parser


def _boundary_impedance(values: Optional[Sequence[float]]):
    if values is None:
        return None
    if np.isscalar(values):
        return float(values)
    if len(values) == 1:
        return float(values[0])
    if len(values) == 2:
        return float(values[0]), float(values[1])
    raise ValueError("--boundary-impedance accepts one value or a left/right pair")


def _save_plot(
    path: Path,
    *,
    time_us: np.ndarray,
    observed: np.ndarray,
    synthetic: np.ndarray,
    scale: float,
    score,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_trace, ax_packets) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (2.0, 1.0)},
    )
    ax_trace.plot(time_us, observed, color="black", lw=1.0, label="observed")
    ax_trace.plot(
        time_us,
        scale * synthetic,
        color="tab:orange",
        lw=1.0,
        label="synthetic (direct-packet scale)",
    )
    for packet in score.packets:
        color = "tab:blue" if packet.used_for_calibration else "tab:green"
        ax_trace.axvspan(packet.start_us, packet.end_us, color=color, alpha=0.10)
    ax_trace.set_xlabel("Time (us)")
    ax_trace.set_ylabel("Amplitude")
    ax_trace.legend(loc="upper right")
    ax_trace.grid(alpha=0.2)

    orders = [packet.order for packet in score.packets]
    correlations = [packet.correlation for packet in score.packets]
    errors = [packet.normalized_l2 for packet in score.packets]
    width = 0.38
    x = np.arange(len(orders), dtype=float)
    ax_packets.bar(x - width / 2, correlations, width, label="signed correlation")
    ax_packets.bar(x + width / 2, errors, width, label="normalized L2")
    ax_packets.axhline(0.0, color="black", lw=0.7)
    ax_packets.set_xticks(x, [str(order) for order in orders])
    ax_packets.set_xlabel("Packet order (0 = direct)")
    ax_packets.set_ylabel("Metric")
    ax_packets.legend(loc="best")
    ax_packets.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_one(args: argparse.Namespace, *, provenance: Optional[dict] = None) -> dict:
    case = load_reference_case(
        tsv_path=args.tsv,
        stf_h5_path=args.stf_h5,
        calibration_path=args.calibration_json,
        stf_name=args.stf_name,
        channel_name=args.channel_name,
        average_waveforms=args.average_waveforms,
        number_of_waveforms=args.number_of_waveforms,
        freqmin_mhz=args.freqmin_mhz,
        freqmax_mhz=args.freqmax_mhz,
        filter_order=args.filter_order,
        wavelet_filter_mode=args.wavelet_filter_mode,
    )
    geometry = load_reference_geometry(
        args.metadata,
        block_id=args.block_id,
        wave_type=args.wave_type,
        steel_thickness_cm=case.block_thickness_cm,
    )
    if args.steel_velocity is not None:
        geometry = replace(
            geometry, steel_velocity_cm_per_us=float(args.steel_velocity)
        )
    if args.pzt_velocity is not None:
        geometry = replace(geometry, pzt_velocity_cm_per_us=float(args.pzt_velocity))

    config = ReferenceForwardConfig(
        embedding_mode=args.embedding_mode,
        frequency_cutoff_mhz=args.frequency_cutoff_mhz,
        points_per_wavelength=args.points_per_wavelength,
        cfl_factor=args.cfl_factor,
        boundary_condition=args.boundary_condition,
        boundary_impedance=_boundary_impedance(args.boundary_impedance),
        source_injection=args.source_injection,
        receiver_quantity=args.receiver_quantity,
        source_depth_fraction=args.source_depth_fraction,
        receiver_depth_fraction=args.receiver_depth_fraction,
        source_inset_cm=args.source_inset_cm,
        receiver_inset_cm=args.receiver_inset_cm,
        source_extension_cm=args.source_extension_cm,
        receiver_extension_cm=args.receiver_extension_cm,
        interpolation_radius_cells=args.interpolation_radius_cells,
        source_normalization=args.source_normalization,
        receiver_normalization=args.receiver_normalization,
        spatial_boundary_mode=args.spatial_boundary_mode,
        source_amplitude=args.source_amplitude,
    )
    # STEEL_REFERENCE_AMPLITUDE_V1
    if args.source_duration_us is not None:
        from lab_uw.source_amplitude import truncate_case
        case = truncate_case(case, args.source_duration_us)
    result = run_reference_forward(case, geometry, config)
    validation_orders = reliable_packet_orders(
        case.calibration,
        minimum_signed_correlation=args.minimum_packet_correlation,
        max_packets=args.max_packets,
    )
    score = evaluate_reference_trace(
        observed_time_us=result.observed_time_us,
        observed_waveform=case.observed_waveform[: result.observed_time_us.size],
        synthetic_waveform=result.synthetic_waveform,
        calibration=case.calibration,
        calibration_orders=(0,),
        validation_orders=validation_orders,
        max_diagnostic_lag_us=args.max_diagnostic_lag_us,
    )
    convergence = None
    if args.convergence_ppw:
        convergence = check_reference_grid_convergence(
            case=case,
            geometry=geometry,
            base_config=config,
            points_per_wavelength=args.convergence_ppw,
            packet_order=0,
            max_diagnostic_lag_us=args.max_diagnostic_lag_us,
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    png_path = args.output_prefix.with_suffix(".png")
    payload = {
        "reference_case": case.stem,
        "configuration": provenance,
        "model": result.metadata(),
        "filter": {
            "freqmin_mhz": case.filter_freqmin_mhz,
            "freqmax_mhz": case.filter_freqmax_mhz,
            "wavelet_filter_mode": case.wavelet_filter_mode,
            "wavelet_filter_provenance_verified": (
                case.wavelet_filter_provenance_verified
            ),
            "wavelet_filter_verification_method": (
                case.wavelet_filter_verification_method
            ),
            "wavelet_time_axis_reconstructed": (
                case.wavelet_time_axis_reconstructed
            ),
            "wavelet_observed_correlation": case.wavelet_observed_correlation,
            "wavelet_observed_scaled_l2": case.wavelet_observed_scaled_l2,
        },
        "score": score.to_dict(),
        "grid_convergence": convergence,
        "scope": {
            "dimensionality": "1D",
            "amplitude_fit": "one non-negative scalar fitted on direct packet only",
            "reflections": "withheld from amplitude/embedding calibration",
            "negative_correlation": "treated as zero similarity",
        },
    }
    # STEEL_REFERENCE_AMPLITUDE_V1
    if args.source_duration_us is not None:
        from lab_uw.source_amplitude import export_reference_source
        source_path = args.output_prefix.with_suffix(".source.npz")
        calibration = export_reference_source(source_path, case=case, geometry=geometry,
            config=config, score=score, provenance={
                "validation": provenance, "filter_order": args.filter_order,
                "tsv": str(args.tsv), "stf_h5": str(args.stf_h5),
                "calibration_json": str(args.calibration_json)})
        payload["exported_source"] = {"path": str(source_path), "id": calibration["id"],
            "reference_gain": calibration["reference_gain"],
            "duration_us": calibration["duration_us"], "dds_trace_scale": 1.0}
        print(f"calibrated source: {source_path} (gain={calibration['reference_gain']:.9g})")
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    _save_plot(
        png_path,
        time_us=result.observed_time_us,
        observed=case.observed_waveform[: result.observed_time_us.size],
        synthetic=result.synthetic_waveform,
        scale=score.amplitude_scale,
        score=score,
    )
    print(f"calibration score: {score.calibration_score:.6g}")
    print(f"wrote {json_path}")
    print(f"wrote {png_path}")
    return {
        "case": case.stem,
        "calibration_score": score.calibration_score,
        "output_json": str(json_path),
        "output_png": str(png_path),
        "grid_convergence_performed": convergence is not None,
    }


def _legacy_namespace(parser: argparse.ArgumentParser, args: argparse.Namespace):
    required = {
        "--tsv": args.tsv,
        "--stf-h5": args.stf_h5,
        "--calibration-json": args.calibration_json,
        "--metadata": args.metadata,
        "--block-id": args.block_id,
        "--wave-type": args.wave_type,
        "--output-prefix": args.output_prefix,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        parser.error(
            "without --config the following arguments are required: "
            + ", ".join(missing)
        )
    values = dict(DEFAULT_OPTIONS)
    for name, value in vars(args).items():
        if value is not None:
            values[name] = value
    # Historically the numerical cutoff followed the upper preprocessing
    # cutoff.  Preserve that behavior in the long-form legacy interface.
    values["frequency_cutoff_mhz"] = values["freqmax_mhz"]
    return argparse.Namespace(**values)


def _manifest_namespace(
    manifest: ReferenceValidationManifest,
    case: ReferenceValidationCaseSpec,
    cli: argparse.Namespace,
) -> argparse.Namespace:
    values = dict(DEFAULT_OPTIONS)
    values.update(manifest.loading)
    values.update(manifest.forward)
    values.update(manifest.validation)
    values.update(
        {
            "tsv": case.tsv_path,
            "stf_h5": manifest.stf_h5_path,
            "calibration_json": case.calibration_path,
            "stf_name": case.stf_name,
            "channel_name": case.channel_name,
            "metadata": manifest.metadata_path,
            "block_id": manifest.block_id,
            "wave_type": manifest.wave_type,
            "output_prefix": case.output_prefix,
            "steel_velocity": None,
            "pzt_velocity": None,
        }
    )

    override_names = set(DEFAULT_OPTIONS) - {"frequency_cutoff_mhz"}
    for name in override_names:
        value = getattr(cli, name, None)
        if value is not None:
            values[name] = value
    if cli.freqmax_mhz is not None:
        values["frequency_cutoff_mhz"] = cli.freqmax_mhz

    if cli.convergence_ppw is not None:
        values["convergence_ppw"] = cli.convergence_ppw
    elif case.name in manifest.convergence_cases:
        values["convergence_ppw"] = manifest.convergence_ppw
    else:
        values["convergence_ppw"] = None
    return argparse.Namespace(**values)


def _run_manifest(args: argparse.Namespace) -> int:
    forbidden = {
        "--tsv": args.tsv,
        "--stf-h5": args.stf_h5,
        "--calibration-json": args.calibration_json,
        "--stf-name": args.stf_name,
        "--channel-name": args.channel_name,
        "--metadata": args.metadata,
        "--block-id": args.block_id,
        "--wave-type": args.wave_type,
        "--output-prefix": args.output_prefix,
    }
    mixed = [flag for flag, value in forbidden.items() if value is not None]
    if mixed:
        raise ValueError(
            "with --config, paths and case identity come from the manifest; "
            f"remove: {', '.join(mixed)}"
        )

    manifest = load_reference_validation_manifest(args.config)
    selected = manifest.select(args.case)
    results = []
    for case in selected:
        print(f"validating {case.name}")
        run_args = _manifest_namespace(manifest, case, args)
        results.append(
            _run_one(
                run_args,
                provenance={
                    "mode": "manifest",
                    "manifest": str(manifest.manifest_path),
                    "case": case.name,
                    "resolved_inputs": {
                        "tsv": str(case.tsv_path),
                        "stf_h5": str(manifest.stf_h5_path),
                        "calibration_json": str(case.calibration_path),
                        "metadata": str(manifest.metadata_path),
                    },
                },
            )
        )

    manifest.output_directory.mkdir(parents=True, exist_ok=True)
    summary_name = (
        "reference_validation_summary.json"
        if args.case is None
        else f"{args.case}.validation_summary.json"
    )
    summary_path = manifest.output_directory / summary_name
    summary = {
        "manifest": str(manifest.manifest_path),
        "selected_case": args.case,
        "case_count": len(results),
        "results": results,
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"wrote {summary_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.config is not None:
        return _run_manifest(args)
    if args.case is not None:
        parser.error("--case requires --config")
    legacy_args = _legacy_namespace(parser, args)
    _run_one(legacy_args, provenance={"mode": "command_line"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
