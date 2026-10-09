"""Small, testable utilities for the joint gouge-material grid search."""

from __future__ import annotations

from itertools import product
from typing import Iterable, Optional

import numpy as np


def material_candidates(
    velocities_cm_us: Iterable[float],
    dampings: Iterable[float],
    densities_g_cm3: Iterable[float],
) -> list[tuple[float, float, float]]:
    """Return the Cartesian ``(velocity, damping, density)`` candidate list."""

    velocity = _validated_axis(velocities_cm_us, "velocity", strictly_positive=True)
    damping = _validated_axis(dampings, "damping", strictly_positive=False)
    density = _validated_axis(densities_g_cm3, "density", strictly_positive=True)
    return [tuple(map(float, item)) for item in product(velocity, damping, density)]


def centered_refined_axis(
    center: float,
    coarse_axis: Iterable[float],
    *,
    number_of_points: int = 7,
    half_width: Optional[float] = None,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
) -> np.ndarray:
    """Build a uniformly sampled fine axis around a coarse-grid minimum.

    Unless explicitly supplied, the half width is one coarse-grid spacing.
    Bounds clip the interval, producing a one-sided refinement when the coarse
    minimum lies close to a physical limit.
    """

    coarse = _validated_axis(coarse_axis, "coarse_axis", strictly_positive=False)
    npoints = int(number_of_points)
    if npoints < 2:
        raise ValueError("number_of_points must be at least two")
    center = float(center)
    if not np.isfinite(center):
        raise ValueError("center must be finite")

    if half_width is None:
        unique = np.unique(coarse)
        if unique.size < 2:
            raise ValueError("half_width is required for a one-point coarse axis")
        half_width = float(np.max(np.diff(unique)))
    half_width = float(half_width)
    if not np.isfinite(half_width) or half_width <= 0.0:
        raise ValueError("half_width must be finite and positive")

    low = center - half_width
    high = center + half_width
    if lower_bound is not None:
        low = max(low, float(lower_bound))
    if upper_bound is not None:
        high = min(high, float(upper_bound))
    if not low < high:
        raise ValueError("refined-axis bounds leave an empty interval")
    return np.linspace(low, high, npoints, dtype=float)


def _validated_axis(
    values: Iterable[float],
    name: str,
    *,
    strictly_positive: bool,
) -> np.ndarray:
    axis = np.asarray(list(values), dtype=float).reshape(-1)
    if axis.size == 0 or np.any(~np.isfinite(axis)):
        raise ValueError(f"{name} axis must be finite and non-empty")
    if strictly_positive and np.any(axis <= 0.0):
        raise ValueError(f"{name} axis must be strictly positive")
    if not strictly_positive and np.any(axis < 0.0):
        raise ValueError(f"{name} axis must be non-negative")
    return np.unique(axis)
