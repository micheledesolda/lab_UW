"""Deterministic data windows used by the DDS inversion workflow."""

from __future__ import annotations

import numpy as np


def sample_interval_mask(interval: np.ndarray, num_samples: int) -> np.ndarray:
    """Return a record-length mask from either indices or a boolean mask."""
    size = int(num_samples)
    if size < 1:
        raise ValueError("num_samples must be positive")
    raw = np.asarray(interval)
    if raw.dtype == np.bool_:
        mask = raw.reshape(-1)
        if mask.size != size:
            raise ValueError(
                "a boolean interval must have exactly num_samples elements"
            )
        return mask.copy()

    indices = raw.reshape(-1)
    if np.any(~np.isfinite(indices)) or np.any(indices != indices.astype(int)):
        raise ValueError("interval indices must be finite integers")
    indices = indices.astype(int)
    if np.any(indices < 0) or np.any(indices >= size):
        raise IndexError("interval contains indices outside the record")
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return mask


def direct_misfit_interval(
    observed_time: np.ndarray,
    *,
    arrival_min_us: float,
    arrival_max_us: float,
    stf_duration_us: float,
    padding_us: float = 0.25,
) -> np.ndarray:
    """Return one fixed window containing all plausible direct packets.

    The bounds may span a homogeneous velocity search, but the returned data
    samples are fixed for every candidate.  This avoids changing the objective
    function merely because a trial model predicts a different travel time.
    """
    time = np.asarray(observed_time, dtype=float).reshape(-1)
    if time.size < 2 or np.any(np.diff(time) <= 0):
        raise ValueError("observed_time must be strictly increasing")
    if stf_duration_us <= 0:
        raise ValueError("stf_duration_us must be positive")
    if padding_us < 0:
        raise ValueError("padding_us cannot be negative")

    start = float(min(arrival_min_us, arrival_max_us) - padding_us)
    stop = float(max(arrival_min_us, arrival_max_us) + stf_duration_us + padding_us)
    indices = np.flatnonzero((time >= start) & (time <= stop))
    if indices.size == 0:
        raise ValueError(
            f"empty direct-arrival window [{start:.6g}, {stop:.6g}] us"
        )
    return indices
