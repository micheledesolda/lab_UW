"""Metrics for calibrating and validating the 1D reference-block model.

Calibration and validation are deliberately different operations:

* the direct packet calibrates the small set of effective embedding choices;
* later zero-angle packets test those choices without being re-fit.

No absolute-value correlation is used.  A correlation of ``-1`` therefore has
zero similarity, as required for this experiment: the observed multiples do
not undergo a physical polarity reversal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PacketMetric:
    order: int
    start_us: float
    end_us: float
    correlation: float
    correlation_similarity: float
    normalized_l2: float
    amplitude_ratio: float
    shape_only_scale: float
    shape_only_normalized_l2: float
    best_lag_us: float
    best_lag_correlation: float
    used_for_calibration: bool
    stored_observed_packet_correlation: Optional[float]


@dataclass(frozen=True)
class ReferenceScore:
    amplitude_scale: float
    calibration_score: float
    calibration_orders: tuple[int, ...]
    validation_orders: tuple[int, ...]
    packets: tuple[PacketMetric, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["packets"] = [asdict(packet) for packet in self.packets]
        return result


def packet_indices(
    time_us: np.ndarray,
    *,
    start_us: float,
    span_us: float,
) -> np.ndarray:
    time = np.asarray(time_us, dtype=float).reshape(-1)
    i0 = int(np.searchsorted(time, float(start_us), side="left"))
    i1 = int(np.searchsorted(time, float(start_us + span_us), side="left"))
    i0 = max(0, min(i0, time.size))
    i1 = max(i0, min(i1, time.size))
    return np.arange(i0, i1, dtype=int)


def signed_correlation(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> float:
    """Pearson correlation with degenerate packets mapped to zero."""
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 2:
        raise ValueError("correlation inputs must have the same length >= 2")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator <= eps:
        return 0.0
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def correlation_similarity(correlation: float) -> float:
    """Map signed correlation to [0, 1], treating negative values as null."""
    return float(np.clip(correlation, 0.0, 1.0))


def fit_amplitude_scale(
    observed: np.ndarray,
    synthetic: np.ndarray,
    indices: np.ndarray,
    *,
    allow_polarity_flip: bool = False,
    eps: float = 1e-15,
) -> float:
    """Fit one global scalar; by default its sign cannot flip polarity."""
    d = np.asarray(observed, dtype=float)[indices]
    s = np.asarray(synthetic, dtype=float)[indices]
    denominator = float(np.dot(s, s))
    if denominator <= eps:
        return 0.0
    scale = float(np.dot(d, s) / denominator)
    return scale if allow_polarity_flip else max(0.0, scale)


def normalized_l2(observed: np.ndarray, synthetic: np.ndarray, eps: float = 1e-15) -> float:
    d = np.asarray(observed, dtype=float)
    s = np.asarray(synthetic, dtype=float)
    return float(np.linalg.norm(d - s) / max(np.linalg.norm(d), eps))


def rms_ratio(observed: np.ndarray, synthetic: np.ndarray, eps: float = 1e-15) -> float:
    d = np.asarray(observed, dtype=float)
    s = np.asarray(synthetic, dtype=float)
    return float(np.sqrt(np.mean(s * s)) / max(np.sqrt(np.mean(d * d)), eps))


def bounded_lag_correlation(
    observed: np.ndarray,
    synthetic: np.ndarray,
    *,
    max_lag_samples: int,
) -> tuple[int, float]:
    """Diagnostic best signed correlation without modifying the objective."""
    d = np.asarray(observed, dtype=float).reshape(-1)
    s = np.asarray(synthetic, dtype=float).reshape(-1)
    if d.size != s.size:
        raise ValueError("lag-correlation inputs must have the same length")
    max_lag = min(max(0, int(max_lag_samples)), max(0, d.size - 2))

    best_lag = 0
    best_corr = signed_correlation(d, s)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            d_part = d[-lag:]
            s_part = s[: lag or None]
        elif lag > 0:
            d_part = d[:-lag]
            s_part = s[lag:]
        else:
            d_part = d
            s_part = s
        corr = signed_correlation(d_part, s_part)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    # Positive lag means that the synthetic packet occurs later.
    return best_lag, best_corr


def stored_reflection_correlations(calibration: dict[str, Any]) -> dict[int, float]:
    values: dict[int, float] = {}
    entries = calibration.get("correlation", {}).get("per_reflection", [])
    for order, item in enumerate(entries, start=1):
        try:
            values[order] = float(item["corr_coeff"])
        except (KeyError, TypeError, ValueError):
            continue
    return values


def reliable_packet_orders(
    calibration: dict[str, Any],
    *,
    minimum_signed_correlation: float = 0.5,
    max_packets: Optional[int] = None,
) -> tuple[int, ...]:
    """Return direct plus reflections whose stored signed correlation passes."""
    arrivals = calibration["zero_angle_packets"]["arrival_times_us"]
    limit = len(arrivals) if max_packets is None else min(len(arrivals), max_packets)
    selected = [0]
    correlations = stored_reflection_correlations(calibration)
    if limit > 1 and not correlations:
        raise ValueError(
            "calibration JSON has no signed per-reflection correlations; use "
            "the latest JSON generated after enforcing the 3*t0 rule"
        )
    for order in range(1, limit):
        if correlations.get(order, -np.inf) >= minimum_signed_correlation:
            selected.append(order)
    return tuple(selected)


def evaluate_reference_trace(
    *,
    observed_time_us: np.ndarray,
    observed_waveform: np.ndarray,
    synthetic_waveform: np.ndarray,
    calibration: dict[str, Any],
    calibration_orders: Sequence[int] = (0,),
    validation_orders: Optional[Iterable[int]] = None,
    allow_polarity_flip: bool = False,
    max_diagnostic_lag_us: float = 0.25,
) -> ReferenceScore:
    """Fit amplitude on calibration packets and score all requested packets.

    The best-lag correlation is reported only as a diagnostic.  The actual
    score uses unshifted packets, so a wrong velocity or source position cannot
    be hidden by a correlation alignment.
    """
    time = np.asarray(observed_time_us, dtype=float).reshape(-1)
    observed = np.asarray(observed_waveform, dtype=float).reshape(-1)
    synthetic = np.asarray(synthetic_waveform, dtype=float).reshape(-1)
    if time.size != observed.size or observed.size != synthetic.size:
        raise ValueError("time, observed and synthetic arrays must have equal length")
    if time.size < 3 or np.any(np.diff(time) <= 0):
        raise ValueError("observed_time_us must be strictly increasing")

    arrivals = np.asarray(
        calibration["zero_angle_packets"]["arrival_times_us"], dtype=float
    )
    span = float(calibration["zero_angle_packets"]["packet_window_span_us"])
    calibration_orders = tuple(sorted(set(int(v) for v in calibration_orders)))
    if validation_orders is None:
        validation_orders = tuple(range(arrivals.size))
    else:
        validation_orders = tuple(sorted(set(int(v) for v in validation_orders)))
    requested_orders = tuple(sorted(set(calibration_orders + validation_orders)))
    if not requested_orders or requested_orders[0] < 0 or requested_orders[-1] >= arrivals.size:
        raise ValueError("packet order outside the available arrival list")

    calibration_idx_parts = [
        packet_indices(time, start_us=arrivals[order], span_us=span)
        for order in calibration_orders
    ]
    if any(part.size < 2 for part in calibration_idx_parts):
        raise ValueError("a calibration packet window contains fewer than two samples")
    calibration_idx = np.unique(np.concatenate(calibration_idx_parts))
    scale = fit_amplitude_scale(
        observed,
        synthetic,
        calibration_idx,
        allow_polarity_flip=allow_polarity_flip,
    )
    scaled = scale * synthetic

    dt = float(np.median(np.diff(time)))
    max_lag_samples = int(round(max_diagnostic_lag_us / dt))
    stored_corr = stored_reflection_correlations(calibration)
    packets: list[PacketMetric] = []
    calibration_terms: list[float] = []

    for order in requested_orders:
        idx = packet_indices(time, start_us=arrivals[order], span_us=span)
        if idx.size < 2:
            continue
        d = observed[idx]
        s = scaled[idx]
        corr = signed_correlation(d, s)
        similarity = correlation_similarity(corr)
        error = normalized_l2(d, s)
        shape_only_scale = fit_amplitude_scale(
            d,
            s,
            np.arange(idx.size),
            allow_polarity_flip=False,
        )
        shape_only_error = normalized_l2(d, shape_only_scale * s)
        lag_samples, lag_corr = bounded_lag_correlation(
            d,
            s,
            max_lag_samples=max_lag_samples,
        )
        used = order in calibration_orders
        if used:
            # Equal packet weighting.  Negative correlation contributes the
            # same shape penalty as zero correlation, never as a good match.
            calibration_terms.append(error + (1.0 - similarity))
        packets.append(
            PacketMetric(
                order=order,
                start_us=float(arrivals[order]),
                end_us=float(arrivals[order] + span),
                correlation=corr,
                correlation_similarity=similarity,
                normalized_l2=error,
                amplitude_ratio=rms_ratio(d, s),
                shape_only_scale=shape_only_scale,
                shape_only_normalized_l2=shape_only_error,
                best_lag_us=float(lag_samples * dt),
                best_lag_correlation=lag_corr,
                used_for_calibration=used,
                stored_observed_packet_correlation=stored_corr.get(order),
            )
        )

    if not calibration_terms:
        raise ValueError("no valid calibration packet was evaluated")
    return ReferenceScore(
        amplitude_scale=scale,
        calibration_score=float(np.mean(calibration_terms)),
        calibration_orders=calibration_orders,
        validation_orders=tuple(validation_orders),
        packets=tuple(packets),
    )
