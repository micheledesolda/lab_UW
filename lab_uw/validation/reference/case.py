"""Data loading and invariants for a homogeneous reference-block experiment.

The reference experiment is intentionally represented by three coordinated
files:

* the raw TSV acquisition;
* the HDF5 effective wavelet extracted from that acquisition;
* the JSON packet calibration produced by ``identify_effective_source_wavelet``.

This module checks that the three files refer to the same acquisition and
keeps absolute experimental time separate from source-relative time.  The
forward solver must receive the latter.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np


WaveletFilterMode = Literal["already_filtered", "apply", "from_tsv"]


def _acquisition_stem(path: Path | str) -> str:
    """Return the logical acquisition name after removing every suffix.

    Mistras exports commonly use names such as ``case.bscan.tsv`` whereas the
    STF writer and calibration JSON use ``case``.  ``Path.stem`` removes only
    ``.tsv`` and would therefore create a false TSV/JSON mismatch.
    """
    candidate = Path(path)
    while candidate.suffix:
        candidate = candidate.with_suffix("")
    return candidate.name


def _as_single_trace(data: np.ndarray, *, average: bool) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        return values.copy()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"expected one or more 1D traces, got shape {values.shape}")
    if values.shape[0] > 1 and not average:
        raise ValueError(
            "the TSV contains multiple traces; set average_waveforms=True or "
            "select/downsample them before loading the reference case"
        )
    return np.mean(values, axis=0) if values.shape[0] > 1 else values[0].copy()


def _single_channel(handler: Any, channel_name: Optional[str]) -> Any:
    """Resolve a multichannel facade without importing its concrete class."""
    channels = getattr(handler, "channels", None)
    if channels is None:
        return handler
    if channel_name is None:
        available = ", ".join(sorted(channels))
        raise ValueError(
            "the TSV contains multiple channels; pass channel_name explicitly "
            f"(available: {available})"
        )
    try:
        return channels[channel_name]
    except KeyError as exc:
        available = ", ".join(sorted(channels))
        raise ValueError(
            f"unknown channel_name={channel_name!r}; available: {available}"
        ) from exc


@dataclass(frozen=True)
class ReferenceCase:
    """One measured trace and its matched effective wavelet/calibration."""

    stem: str
    observed_time_us: np.ndarray
    observed_waveform: np.ndarray
    wavelet_time_absolute_us: np.ndarray
    wavelet_waveform: np.ndarray
    calibration: dict[str, Any]
    filter_freqmin_mhz: Optional[float]
    filter_freqmax_mhz: Optional[float]
    wavelet_filter_mode: WaveletFilterMode
    wavelet_filter_provenance_verified: bool = False
    wavelet_filter_verification_method: str = "unverified"
    wavelet_time_axis_reconstructed: bool = False
    wavelet_observed_correlation: float = float("nan")
    wavelet_observed_scaled_l2: float = float("nan")

    @property
    def wavelet_time_relative_us(self) -> np.ndarray:
        return self.wavelet_time_absolute_us - self.wavelet_time_absolute_us[0]

    @property
    def direct_start_us(self) -> float:
        return float(self.calibration["refined_direct_arrival"]["best_t_start_us"])

    @property
    def packet_span_us(self) -> float:
        return float(
            self.calibration["refined_direct_arrival"]["direct_arrival_span_us"]
        )

    @property
    def packet_arrivals_us(self) -> np.ndarray:
        return np.asarray(
            self.calibration["zero_angle_packets"]["arrival_times_us"],
            dtype=float,
        )

    @property
    def block_thickness_cm(self) -> float:
        return float(self.calibration["reference_block"]["thickness_cm"])

    @property
    def estimated_velocity_cm_per_us(self) -> float:
        return float(
            self.calibration["reference_block"]["estimated_velocity_cm_per_us"]
        )

    def source_handler(self) -> Any:
        """Return an ``UltrasonicDataHandler`` with a source-relative time axis."""
        from lab_uw.data_io.data_io import UltrasonicDataHandler

        metadata = {
            "time_ax_waveform": self.wavelet_time_relative_us.copy(),
            "number_of_samples": int(self.wavelet_waveform.size),
            "number_of_waveforms": 1,
            "product_type": "effective_source_wavelet",
            "experimental_wavelet_start_us": float(
                self.wavelet_time_absolute_us[0]
            ),
            "reference_infile_stem": self.stem,
        }
        return UltrasonicDataHandler(
            waveform_data=self.wavelet_waveform[None, :].copy(),
            metadata=metadata,
        )

    def validate(self) -> None:
        t = np.asarray(self.observed_time_us, dtype=float)
        d = np.asarray(self.observed_waveform, dtype=float)
        tw = np.asarray(self.wavelet_time_absolute_us, dtype=float)
        w = np.asarray(self.wavelet_waveform, dtype=float)

        if t.ndim != 1 or d.ndim != 1 or t.size != d.size or t.size < 2:
            raise ValueError("observed time and waveform must be matched 1D arrays")
        if tw.ndim != 1 or w.ndim != 1 or tw.size != w.size or tw.size < 2:
            raise ValueError("wavelet time and waveform must be matched 1D arrays")
        if np.any(np.diff(t) <= 0) or np.any(np.diff(tw) <= 0):
            raise ValueError("time axes must be strictly increasing")
        if not np.all(np.isfinite(d)) or not np.all(np.isfinite(w)):
            raise ValueError("waveforms contain NaN or infinite values")

        arrivals = self.packet_arrivals_us
        if arrivals.size == 0 or not np.isclose(
            arrivals[0], self.direct_start_us, atol=max(np.diff(t).min(), 1e-9)
        ):
            raise ValueError("the first packet arrival is not the refined direct start")

        # The calibration contract is the zero-offset odd-multiple rule:
        # direct=t0, first multiple=3t0, then 5t0, ... .
        expected = (2.0 * np.arange(arrivals.size) + 1.0) * self.direct_start_us
        tolerance = max(2.0 * np.diff(t).max(), 1e-6)
        if not np.allclose(arrivals, expected, atol=tolerance, rtol=0.0):
            raise ValueError(
                "packet arrivals do not satisfy t0, 3*t0, 5*t0, ... within "
                f"the sampling tolerance ({tolerance:g} us)"
            )

        expected_wavelet_start = float(
            self.calibration["refined_direct_arrival"][
                "effective_wavelet_start_us"
            ]
        )
        if not np.isclose(tw[0], expected_wavelet_start, atol=tolerance):
            raise ValueError(
                "the wavelet start does not match the calibration JSON: "
                f"{tw[0]:g} vs {expected_wavelet_start:g} us"
            )


def _calibrated_wavelet_time_axis(
    *,
    observed_time_us: np.ndarray,
    stored_wavelet_time_us: np.ndarray,
    wavelet_size: int,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    """Recover the absolute STF time axis from the matched TSV and JSON.

    The historical STF writer stores the waveform and sample rate but not the
    absolute experimental start time.  The latter is nevertheless known
    exactly: the extraction script used ``searchsorted`` on the matched TSV
    with the refined JSON bounds.  Rebuilding the axis from that same contract
    is therefore lossless and avoids treating a reader-generated 0-based HDF5
    axis as experimental time.
    """
    observed_time = np.asarray(observed_time_us, dtype=float).reshape(-1)
    stored_time = np.asarray(stored_wavelet_time_us, dtype=float).reshape(-1)
    if observed_time.size < 2 or np.any(np.diff(observed_time) <= 0):
        raise ValueError("observed time axis must be strictly increasing")
    if stored_time.size != int(wavelet_size):
        raise ValueError(
            "HDF5 wavelet time/waveform length mismatch: "
            f"{stored_time.size} vs {wavelet_size}"
        )
    if stored_time.size < 2 or np.any(np.diff(stored_time) <= 0):
        raise ValueError("HDF5 wavelet time axis must be strictly increasing")

    refined = calibration["refined_direct_arrival"]
    start_us = float(refined["effective_wavelet_start_us"])
    end_us = float(refined["best_t_end_us"])
    i0 = int(np.searchsorted(observed_time, start_us, side="left"))
    i1 = int(np.searchsorted(observed_time, end_us, side="left"))
    expected = observed_time[i0:i1]
    if expected.size != int(wavelet_size):
        raise ValueError(
            "TSV/JSON/HDF5 wavelet-length mismatch: the calibrated TSV slice "
            f"contains {expected.size} samples but HDF5 contains {wavelet_size}"
        )

    expected_dt = float(np.median(np.diff(expected)))
    stored_dt = float(np.median(np.diff(stored_time)))
    dt_tolerance = max(1e-12, 1e-6 * abs(expected_dt))
    if not np.isclose(stored_dt, expected_dt, rtol=1e-6, atol=dt_tolerance):
        raise ValueError(
            "HDF5/TSV sample intervals disagree: "
            f"{stored_dt:g} vs {expected_dt:g} us"
        )

    axis_tolerance = max(1e-12, 1e-6 * abs(expected_dt))
    already_absolute = bool(
        np.allclose(stored_time, expected, rtol=0.0, atol=axis_tolerance)
    )
    return expected.copy(), not already_absolute


def load_reference_case(
    *,
    tsv_path: Path | str,
    stf_h5_path: Path | str,
    calibration_path: Path | str,
    stf_name: Optional[str] = None,
    cycle_index: int = 0,
    channel_name: Optional[str] = None,
    average_waveforms: bool = True,
    number_of_waveforms: Optional[int] = None,
    freqmin_mhz: Optional[float] = 0.5,
    freqmax_mhz: Optional[float] = None,
    filter_order: int = 4,
    wavelet_filter_mode: WaveletFilterMode = "already_filtered",
) -> ReferenceCase:
    """Load and validate one matched TSV/HDF5/JSON reference case.

    ``identify_effective_source_wavelet.py`` writes a wavelet sliced from the
    already-filtered trace.  Therefore ``already_filtered`` is the safe
    default: the raw TSV is filtered once and the HDF5 wavelet is not filtered
    a second time.  Use ``apply`` only for an HDF5 wavelet known to be raw.
    """
    from lab_uw.data_io.data_io import UltrasonicDataHandler

    tsv_path = Path(tsv_path)
    stf_h5_path = Path(stf_h5_path)
    calibration_path = Path(calibration_path)

    with calibration_path.open("r", encoding="utf-8") as stream:
        calibration = json.load(stream)

    json_stem = str(calibration.get("infile_stem", ""))
    if not json_stem:
        raise ValueError("calibration JSON does not contain infile_stem")
    tsv_stem = _acquisition_stem(tsv_path)
    if tsv_stem != json_stem:
        raise ValueError(
            f"TSV/JSON mismatch: {tsv_stem!r} != {json_stem!r}"
        )
    group_name = json_stem if stf_name is None else str(stf_name)

    observed_handler = _single_channel(
        UltrasonicDataHandler.read_tsv(tsv_path), channel_name
    )
    if wavelet_filter_mode == "from_tsv":
        observed_handler.remove_mean()
    if number_of_waveforms is not None:
        observed_handler.downsample_waveforms(number_of_waveforms)
    if freqmin_mhz is not None or freqmax_mhz is not None:
        observed_handler.bandpass(
            freqmin=freqmin_mhz,
            freqmax=freqmax_mhz,
            order=filter_order,
        )

    wavelet_handler = UltrasonicDataHandler.read_stf_hdf5(
        stf_h5_path,
        name=group_name,
        cycle_index=cycle_index,
    )
    wavelet_filter_verified = False
    wavelet_filter_verification_method = "unverified"
    if wavelet_filter_mode == "already_filtered":
        stored_min = wavelet_handler.metadata.get("filter_freqmin_mhz")
        stored_max = wavelet_handler.metadata.get("filter_freqmax_mhz")
        if stored_min is not None or stored_max is not None:
            def _same_optional(left, right):
                if left is None or right is None:
                    return left is None and right is None
                return bool(np.isclose(float(left), float(right)))

            if not _same_optional(stored_min, freqmin_mhz) or not _same_optional(
                stored_max, freqmax_mhz
            ):
                raise ValueError(
                    "requested TSV filter does not match the filter stored with "
                    f"the effective wavelet: requested=({freqmin_mhz}, {freqmax_mhz}) "
                    f"MHz, stored=({stored_min}, {stored_max}) MHz"
                )
            wavelet_filter_verified = True
            wavelet_filter_verification_method = "metadata"
    if wavelet_filter_mode == "apply":
        if freqmin_mhz is not None or freqmax_mhz is not None:
            wavelet_handler.bandpass(
                freqmin=freqmin_mhz,
                freqmax=freqmax_mhz,
                order=filter_order,
            )
    elif wavelet_filter_mode not in {"already_filtered", "from_tsv"}:
        raise ValueError(f"unknown wavelet_filter_mode={wavelet_filter_mode!r}")

    observed_time = np.asarray(
        observed_handler.metadata["time_ax_waveform"], dtype=float
    ).reshape(-1)
    observed = _as_single_trace(
        observed_handler.waveform_data,
        average=average_waveforms,
    )
    stored_wavelet_time = np.asarray(
        wavelet_handler.metadata["time_ax_waveform"], dtype=float
    ).reshape(-1)
    wavelet = _as_single_trace(wavelet_handler.waveform_data, average=True)

    wavelet_time, wavelet_time_axis_reconstructed = _calibrated_wavelet_time_axis(
        observed_time_us=observed_time,
        stored_wavelet_time_us=stored_wavelet_time,
        wavelet_size=wavelet.size,
        calibration=calibration,
    )

    if wavelet_time[0] < observed_time[0] or wavelet_time[-1] > observed_time[-1]:
        raise ValueError("the HDF5 wavelet time axis lies outside the TSV record")
    observed_direct = np.interp(wavelet_time, observed_time, observed)
    # STEEL_REFERENCE_AMPLITUDE_V1
    # Re-extract from the newly filtered FULL record. The old HDF5 supplies
    # only the extraction axis, so there is no second filtering of a packet.
    if wavelet_filter_mode == "from_tsv":
        wavelet = observed_direct.copy()
        wavelet_filter_verified = True
        wavelet_filter_verification_method = "reextracted_from_filtered_full_tsv"
    direct_zero_mean = observed_direct - np.mean(observed_direct)
    wavelet_zero_mean = wavelet - np.mean(wavelet)
    denominator = np.linalg.norm(direct_zero_mean) * np.linalg.norm(wavelet_zero_mean)
    wavelet_observed_correlation = (
        float(np.dot(direct_zero_mean, wavelet_zero_mean) / denominator)
        if denominator > 1e-15
        else 0.0
    )
    scale_denominator = float(np.dot(wavelet, wavelet))
    consistency_scale = (
        max(0.0, float(np.dot(observed_direct, wavelet) / scale_denominator))
        if scale_denominator > 1e-15
        else 0.0
    )
    observed_norm = max(float(np.linalg.norm(observed_direct)), 1e-15)
    wavelet_observed_scaled_l2 = float(
        np.linalg.norm(observed_direct - consistency_scale * wavelet) / observed_norm
    )
    if wavelet_filter_mode == "already_filtered":
        # Under the extraction contract the HDF5 array is literally a slice
        # of the filtered, averaged TSV.  A merely high correlation is not
        # enough: it allowed an incorrect upper cutoff (for example 6 rather
        # than 12 MHz) to pass unnoticed.  Float32 HDF5 storage still leaves
        # several orders of magnitude of margin relative to these tolerances.
        identity_ok = (
            wavelet_observed_correlation >= 0.99999
            and wavelet_observed_scaled_l2 <= 1e-4
        )
        if not identity_ok:
            raise ValueError(
                "the HDF5 effective wavelet is not the calibrated direct TSV "
                "slice after the requested preprocessing: "
                f"correlation={wavelet_observed_correlation:.8f}, "
                f"scaled_l2={wavelet_observed_scaled_l2:.3e}; check the HDF5 "
                "group and the exact filter cutoffs/order"
            )
        wavelet_filter_verified = True
        if wavelet_filter_verification_method == "metadata":
            wavelet_filter_verification_method = "metadata_and_direct_packet_identity"
        else:
            wavelet_filter_verification_method = "direct_packet_identity"
    elif wavelet_observed_correlation < 0.9:
        raise ValueError(
            "the selected HDF5 wavelet does not match the direct packet in the "
            f"TSV after preprocessing (signed correlation="
            f"{wavelet_observed_correlation:.3f}); check group name, averaging "
            "and filter settings"
        )

    case = ReferenceCase(
        stem=json_stem,
        observed_time_us=observed_time,
        observed_waveform=observed,
        wavelet_time_absolute_us=wavelet_time,
        wavelet_waveform=wavelet,
        calibration=calibration,
        filter_freqmin_mhz=freqmin_mhz,
        filter_freqmax_mhz=freqmax_mhz,
        wavelet_filter_mode=wavelet_filter_mode,
        wavelet_filter_provenance_verified=wavelet_filter_verified,
        wavelet_filter_verification_method=wavelet_filter_verification_method,
        wavelet_time_axis_reconstructed=wavelet_time_axis_reconstructed,
        wavelet_observed_correlation=wavelet_observed_correlation,
        wavelet_observed_scaled_l2=wavelet_observed_scaled_l2,
    )
    case.validate()
    return case
