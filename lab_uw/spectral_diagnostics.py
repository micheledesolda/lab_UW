"""Shape-only spectral diagnostics for direct ultrasonic arrivals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal.windows import tukey


@dataclass(frozen=True)
class SpectralComparison:
    frequency_mhz: np.ndarray
    first_amplitude: np.ndarray
    second_amplitude: np.ndarray
    normalized_first_amplitude: np.ndarray
    normalized_second_amplitude: np.ndarray
    shape_distance: float


def _validate_trace(time_us: np.ndarray, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time_us, dtype=float).reshape(-1)
    signal = np.asarray(waveform, dtype=float).reshape(-1)
    if time.size != signal.size or time.size < 4:
        raise ValueError("time and waveform must have the same length >= 4")
    differences = np.diff(time)
    if np.any(~np.isfinite(time)) or np.any(~np.isfinite(signal)):
        raise ValueError("time and waveform must be finite")
    if np.any(differences <= 0.0):
        raise ValueError("time must be strictly increasing")
    dt = float(np.median(differences))
    if not np.allclose(differences, dt, rtol=1e-5, atol=1e-12):
        raise ValueError("spectral diagnostics require a uniformly sampled trace")
    return time, signal


def _amplitude_spectrum(
    time_us: np.ndarray,
    waveform: np.ndarray,
    frequency_mhz: np.ndarray,
    *,
    taper_fraction: float = 0.10,
) -> np.ndarray:
    time, signal = _validate_trace(time_us, waveform)
    signal = signal - np.mean(signal)
    window = tukey(signal.size, alpha=float(taper_fraction))
    tapered = signal * window
    nfft = max(256, 4 * 2 ** int(np.ceil(np.log2(signal.size))))
    native_frequency = np.fft.rfftfreq(nfft, d=float(time[1] - time[0]))
    native_amplitude = np.abs(np.fft.rfft(tapered, n=nfft))
    return np.interp(
        frequency_mhz,
        native_frequency,
        native_amplitude,
        left=0.0,
        right=0.0,
    )


def compare_spectral_shapes(
    first_time_us: np.ndarray,
    first_waveform: np.ndarray,
    second_time_us: np.ndarray,
    second_waveform: np.ndarray,
    *,
    frequency_min_mhz: float,
    frequency_max_mhz: float,
    number_of_frequencies: int = 1200,
    eps: float = 1e-15,
) -> SpectralComparison:
    r"""Compare amplitude-spectrum shape independently of trace amplitude.

    The reported distance is

    .. math::

        D_f = \frac{1}{\sqrt{2}}
              \left\|\frac{A_1}{\|A_1\|_2}
              -\frac{A_2}{\|A_2\|_2}\right\|_2,

    evaluated by frequency integration.  For non-negative spectra it lies in
    ``[0, 1]``.  It is zero for spectra with identical shape even when their
    absolute amplitudes differ.
    """
    fmin = float(frequency_min_mhz)
    fmax = float(frequency_max_mhz)
    if not 0.0 <= fmin < fmax:
        raise ValueError("require 0 <= frequency_min_mhz < frequency_max_mhz")
    if int(number_of_frequencies) < 16:
        raise ValueError("number_of_frequencies must be at least 16")
    frequency = np.linspace(fmin, fmax, int(number_of_frequencies))
    first = _amplitude_spectrum(first_time_us, first_waveform, frequency)
    second = _amplitude_spectrum(second_time_us, second_waveform, frequency)

    first_norm = float(np.sqrt(trapezoid(first * first, frequency)))
    second_norm = float(np.sqrt(trapezoid(second * second, frequency)))
    if first_norm <= eps or second_norm <= eps:
        distance = 1.0 if max(first_norm, second_norm) > eps else 0.0
        normalized_first = first / max(first_norm, eps)
        normalized_second = second / max(second_norm, eps)
    else:
        normalized_first = first / first_norm
        normalized_second = second / second_norm
        squared_difference = trapezoid(
            (normalized_first - normalized_second) ** 2,
            frequency,
        )
        distance = float(np.sqrt(max(0.0, squared_difference) / 2.0))
    distance = float(np.clip(distance, 0.0, 1.0))
    return SpectralComparison(
        frequency_mhz=frequency,
        first_amplitude=first,
        second_amplitude=second,
        normalized_first_amplitude=normalized_first,
        normalized_second_amplitude=normalized_second,
        shape_distance=distance,
    )


def plot_spectral_comparison(
    comparison: SpectralComparison,
    *,
    first_label: str,
    second_label: str,
    title: str,
    output_path: Path,
    show_distance: bool,
) -> None:
    """Plot peak-normalized amplitude spectra on the common frequency grid."""
    first_peak = max(float(np.max(comparison.first_amplitude)), 1e-15)
    second_peak = max(float(np.max(comparison.second_amplitude)), 1e-15)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(
        comparison.frequency_mhz,
        comparison.first_amplitude / first_peak,
        label=first_label,
        linewidth=1.7,
    )
    ax.plot(
        comparison.frequency_mhz,
        comparison.second_amplitude / second_peak,
        label=second_label,
        linewidth=1.7,
    )
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Peak-normalized amplitude")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    if show_distance:
        ax.text(
            0.98,
            0.94,
            rf"$D_f={comparison.shape_distance:.4f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
