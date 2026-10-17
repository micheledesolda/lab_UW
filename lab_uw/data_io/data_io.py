# lab_uw/data_io/data_io.py

import numpy as np
import json
import pandas as pd
from scipy.signal import find_peaks

import logging
from typing import Tuple, Dict, Iterator, Mapping, Optional, Union
from pathlib import Path

from lab_uw.data_io.schema import MultiChannelUWFrame, UWFrame
from lab_uw.data_io.readers.tsv_reader import read_tsv
from lab_uw.data_io.readers.hdf5_reader import read_hdf5, read_stf_hdf5
from lab_uw.data_io.writers.hdf5_writer import write_hdf5, write_stf_hdf5

logger = logging.getLogger(__name__)


class MultiChannelUltrasonicDataHandler:
    """Channel-indexed handlers belonging to one recorded acquisition."""

    def __init__(self, channels: Mapping[str, "UltrasonicDataHandler"], metadata: Dict = None):
        self.channels = dict(channels)
        self.metadata = {} if metadata is None else dict(metadata)
        if not self.channels:
            raise ValueError("At least one ultrasonic channel is required.")

    def __getitem__(self, channel_name: str) -> "UltrasonicDataHandler":
        return self.channels[channel_name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)

    def items(self):
        return self.channels.items()

class UltrasonicDataHandler:
    """
    Thin façade around UWFrame with:
      - symmetric readers: read_tsv(), read_hdf5()
      - open() dispatcher by file extension
      - STF helpers: read_stf_hdf5(), write_stf_hdf5()
      - small, chainable processors (µs everywhere)
    """
    def __init__(self, waveform_data: np.ndarray = None, metadata: Dict = None):
        if waveform_data is None: waveform_data = np.array([])
        if metadata is None: metadata = {}
        self.frame = UWFrame(waveform_data, metadata).ensure_2d()

    # --------- symmetric readers ----------
    @classmethod
    def read_tsv(cls, path: Path) -> Union["UltrasonicDataHandler", MultiChannelUltrasonicDataHandler]:
        f = read_tsv(path)
        if isinstance(f, MultiChannelUWFrame):
            channels = {
                name: cls(frame.waveform_data, frame.metadata)
                for name, frame in f.channels.items()
            }
            return MultiChannelUltrasonicDataHandler(channels, f.metadata)
        return cls(f.waveform_data, f.metadata)

    @classmethod
    def read_hdf5(cls, path: Path, **kwargs) -> "UltrasonicDataHandler":
        f = read_hdf5(path, **kwargs)
        return cls(f.waveform_data, f.metadata)

    # unified dispatcher (by suffix)
    @classmethod
    def open(cls, path: Path, **h5_options) -> Union["UltrasonicDataHandler", MultiChannelUltrasonicDataHandler]:
        ext = path.suffix.lower()
        if ext in {".tsv", ".txt"}:
            return cls.read_tsv(path)
        if ext in {".h5", ".hdf5"}:
            return cls.read_hdf5(path, **h5_options)
        raise ValueError(f"Unsupported file type: {path}")

    # --------- STF helpers (HDF5) ----------
    @classmethod
    def read_stf_hdf5(cls, h5_path: Path, *, name: str, cycle_index: int = 0) -> "UltrasonicDataHandler":
        """Read /stf/<name>/cycle_xxxxxx/source_waveform -> handler with (1, L)."""
        f = read_stf_hdf5(h5_path, name=name, cycle_index=cycle_index)
        return cls(f.waveform_data, f.metadata)

    # --------- (optional) acquisition writer convenience ----------
    def write_hdf5(
        self,
        out_path: Path,
        run_name: str,
        *,
        channel_count: int = 1,
        dataset: str = "active",
        add_timestamps: bool = False,
    ) -> None:
        """Thin wrapper around writers.write_hdf5 for acquisitions."""
        write_hdf5(
            frame=self.frame,
            out_path=out_path,
            run_name=run_name,
            channel_count=channel_count,
            dataset=dataset,
            add_timestamps=add_timestamps,
        )

    def write_stf_hdf5(
        self,
        out_path: Path,
        stf_name: str,
        *,
        cycle_index: int = 0,
        include_active_stub: bool = False,
        include_passive_stub: bool = False,
        t0_unix_s: Optional[float] = None,
    ) -> None:
        """Persist row 0 as /stf/<stf_name>/cycle_xxxxxx/source_waveform."""
        y = self.frame.waveform_data
        if y.ndim == 2:
            if y.shape[0] == 0:
                raise ValueError("No STF data to write.")
            if y.shape[0] > 1:
                logger.warning("STF writer: multiple rows detected; writing only row 0.")
            y = y[0]
        stf_frame = UWFrame(y[None, :], self.frame.metadata)
        write_stf_hdf5(
            stf=stf_frame,
            out_path=out_path,
            stf_name=stf_name,
            cycle_index=cycle_index,
            include_active_stub=include_active_stub,
            include_passive_stub=include_passive_stub,
            t0_unix_s=t0_unix_s,
        )

    # --------- properties ----------
    @property
    def waveform_data(self): return self.frame.waveform_data
    @waveform_data.setter
    def waveform_data(self, v): self.frame.waveform_data = v

    @property
    def metadata(self): return self.frame.metadata
    @metadata.setter
    def metadata(self, v): self.frame.metadata = v

    # --------- small processors (µs units) ----------
    def remove_mean(self, *, per_waveform: bool = False) -> "UltrasonicDataHandler":
        """Remove the global mean, or each trace's mean when requested."""
        axis = -1 if per_waveform else None
        self.frame.waveform_data = self.frame.waveform_data - np.mean(
            self.frame.waveform_data, axis=axis, keepdims=per_waveform
        )
        return self

    def select_waveform_window(
        self, t_min_us: float, t_max_us: float
    ) -> "UltrasonicDataHandler":
        """Keep samples in the inclusive interval; preserve absolute trace time.

        This selects samples *within each trace*, unlike
        select_by_time_ax_acquisition(), which selects entire acquisitions.
        Apply zero-phase filters to the full record before this operation.
        """
        t = np.asarray(self.metadata["time_ax_waveform"], dtype=float)
        lo, hi = float(t_min_us), float(t_max_us)
        if not np.isfinite([lo, hi]).all() or not lo < hi:
            raise ValueError("Require finite t_min_us < t_max_us.")
        if t.ndim != 1 or t.size != self.waveform_data.shape[-1] or t.size < 2:
            raise ValueError("time_ax_waveform must match the trace sample axis.")
        if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("time_ax_waveform must be finite and strictly increasing.")
        tolerance = 0.5 * float(np.median(np.diff(t)))
        if lo < t[0] - tolerance or hi > t[-1] + tolerance:
            raise ValueError(
                f"Requested window [{lo:g}, {hi:g}] us is outside the record "
                f"[{t[0]:g}, {t[-1]:g}] us. Check the direct-arrival limits."
            )
        start = int(np.searchsorted(t, lo, side="left"))
        stop = int(np.searchsorted(t, hi, side="right"))
        if stop - start < 4:
            raise ValueError("The selected trace window must contain at least 4 samples.")
        self.frame.waveform_data = self.waveform_data[:, start:stop]
        self.metadata["time_ax_waveform"] = t[start:stop]
        self.metadata["number_of_samples"] = stop - start
        self.metadata["waveform_window_min_us"] = lo
        self.metadata["waveform_window_max_us"] = hi
        return self

    def consecutive_differences(self) -> "UltrasonicDataHandler":
        """Return u[i+1] - u[i], without alignment or amplitude normalization.

        Subtraction is performed before any display thinning. Each difference
        is assigned to the midpoint of its two acquisition times. Acquisition
        time units are preserved, just as in the input metadata.
        """
        data = np.asarray(self.waveform_data, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2:
            raise ValueError("Consecutive differences require at least two traces.")
        times = np.asarray(self.metadata["time_ax_acquisition"], dtype=float)
        if times.shape != (data.shape[0],) or not np.isfinite(times).all():
            raise ValueError("Acquisition times must match waveform rows and be finite.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Acquisition times must be strictly increasing.")
        metadata = dict(self.metadata)
        metadata["number_of_waveforms"] = data.shape[0] - 1
        metadata["time_ax_acquisition"] = times[:-1] + 0.5 * np.diff(times)
        metadata["difference_definition"] = "next_minus_current"
        metadata["difference_time_reference"] = "pair_midpoint"
        return type(self)(np.diff(data, axis=0), metadata)

    def downsample_waveforms(self, number_of_waveforms2process: Optional[int]) -> "UltrasonicDataHandler":
        if not number_of_waveforms2process: return self
        n = self.metadata.get("number_of_waveforms", self.waveform_data.shape[0])
        step = max(1, round(n / number_of_waveforms2process))
        self.frame.waveform_data = self.frame.waveform_data[::step, :]
        if "time_ax_acquisition" in self.metadata:
            self.metadata["time_ax_acquisition"] = self.metadata["time_ax_acquisition"][::step]
        self.metadata["number_of_waveforms"] = int(self.frame.waveform_data.shape[0])
        return self

    def truncate_time(self, maxtime_us: float) -> "UltrasonicDataHandler":
        if maxtime_us and maxtime_us > 0:
            t = self.metadata["time_ax_waveform"]
            idx = int(np.searchsorted(t, maxtime_us))
            self.frame.waveform_data = self.frame.waveform_data[:, :idx]
            self.metadata["time_ax_waveform"] = t[:idx]
            self.metadata["number_of_samples"] = int(idx)
        return self

    def zero_before(self, t_us: float) -> "UltrasonicDataHandler":
        if t_us and t_us > 0:
            t = self.metadata["time_ax_waveform"]
            idx = int(np.searchsorted(t, t_us))
            self.frame.waveform_data[:, :idx] = 0.0
        return self

    def bandpass(
        self,
        freqmin: Optional[float] = None,
        freqmax: Optional[float] = None,
        order: int = 4,
    ) -> "UltrasonicDataHandler":
        """
        Zero-phase Butterworth band-pass / high-pass / low-pass filter
        applied along the waveform-sample axis.

        Parameters
        ----------
        freqmin : float, optional
            Lower cutoff frequency in MHz. If None, only the upper cutoff is used.
        freqmax : float, optional
            Upper cutoff frequency in MHz. If None, only the lower cutoff is used.
        order : int
            Filter order.

        Returns
        -------
        UltrasonicDataHandler
            self, for chaining.

        Notes
        -----
        - Uses scipy.signal.sosfiltfilt for zero-phase filtering.
        - metadata["sampling_rate"] is assumed to be the sampling interval in µs/sample.
        """
        if freqmin is None and freqmax is None:
            return self

        if "sampling_rate" not in self.metadata:
            raise KeyError("metadata does not contain 'sampling_rate' key.")

        dt_us = float(self.metadata["sampling_rate"])   # µs/sample
        if dt_us <= 0:
            raise ValueError("metadata['sampling_rate'] must be positive.")

        fs_mhz = 1.0 / dt_us
        nyq_mhz = 0.5 * fs_mhz

        if order <= 0 or not isinstance(order, int):
            raise ValueError("order must be a positive integer.")

        if freqmin is not None:
            freqmin = float(freqmin)
            if freqmin <= 0:
                raise ValueError("freqmin must be > 0.")
            if freqmin >= nyq_mhz:
                raise ValueError(f"freqmin must be < Nyquist ({nyq_mhz:.6g} MHz).")

        if freqmax is not None:
            freqmax = float(freqmax)
            if freqmax <= 0:
                raise ValueError("freqmax must be > 0.")
            if freqmax >= nyq_mhz:
                raise ValueError(f"freqmax must be < Nyquist ({nyq_mhz:.6g} MHz).")

        if freqmin is not None and freqmax is not None:
            if freqmin >= freqmax:
                raise ValueError("freqmin must be smaller than freqmax.")
            Wn = [freqmin, freqmax]
            btype = "bandpass"
        elif freqmin is not None:
            Wn = freqmin
            btype = "highpass"
        else:
            Wn = freqmax
            btype = "lowpass"

        from scipy.signal import butter, sosfiltfilt

        y = np.asarray(self.frame.waveform_data)
        input_was_1d = (y.ndim == 1)
        if input_was_1d:
            y = y[None, :]
        elif y.ndim != 2:
            raise ValueError(f"waveform_data must be 1D or 2D; got shape {y.shape}.")

        sos = butter(order, Wn, btype=btype, fs=fs_mhz, output="sos")
        y_filt = sosfiltfilt(sos, y, axis=-1)

        self.frame.waveform_data = y_filt[0] if input_was_1d else y_filt
        # Flat scalar provenance survives the existing HDF5 metadata writer
        # and lets downstream reference modeling verify that the raw record
        # and the already-filtered effective wavelet use the same pass band.
        self.metadata["filter_type"] = btype
        if freqmin is not None:
            self.metadata["filter_freqmin_mhz"] = freqmin
        else:
            self.metadata.pop("filter_freqmin_mhz", None)
        if freqmax is not None:
            self.metadata["filter_freqmax_mhz"] = freqmax
        else:
            self.metadata.pop("filter_freqmax_mhz", None)
        self.metadata["filter_order"] = int(order)
        self.metadata["filter_zero_phase"] = True
        return self

    def select_by_time_ax_acquisition(
        self,
        *,
        t_min_us: Optional[float] = None,
        t_max_us: Optional[float] = None,
        indices: Optional[np.ndarray] = None,
        reset_t0: bool = False,
    ) -> "UltrasonicDataHandler":
        """
        Select waveforms based on metadata['time_ax_acquisition'] (µs).

        Modes:
        1) Time window: keep rows whose acquisition time is within [t_min_us, t_max_us].
        2) Explicit indices: keep the specified row indices (applied after time filtering if both provided).

        Args:
            t_min_us: Minimum acquisition time (inclusive). If None, no lower bound.
            t_max_us: Maximum acquisition time (inclusive). If None, no upper bound.
            indices: Optional explicit row indices to keep (after time filtering).
            reset_t0: If True, subtract the first selected time from time_ax_acquisition.

        Returns:
            self (chainable)

        Raises:
            KeyError: if 'time_ax_acquisition' is missing from metadata.
            ValueError: if time_ax_acquisition length mismatches waveform rows, or selection is empty.
        """
        if "time_ax_acquisition" not in self.metadata:
            raise KeyError("metadata does not contain 'time_ax_acquisition' key.")

        t = np.asarray(self.metadata["time_ax_acquisition"])
        y = self.frame.waveform_data  # expected 2D (N, L)

        if y.ndim != 2:
            y = np.asarray(y)
            if y.ndim == 1:
                y = y[None, :]
            else:
                raise ValueError(f"waveform_data must be 1D or 2D; got shape {y.shape}.")

        if t.shape[0] != y.shape[0]:
            raise ValueError(
                f"Length mismatch: time_ax_acquisition has {t.shape[0]} entries "
                f"but waveform_data has {y.shape[0]} rows."
            )

        # --- build mask from time window ---
        mask = np.ones(t.shape[0], dtype=bool)
        if t_min_us is not None:
            mask &= (t >= float(t_min_us))
        if t_max_us is not None:
            mask &= (t <= float(t_max_us))

        sel = np.nonzero(mask)[0]

        # --- optionally apply explicit indices (relative to original rows) ---
        if indices is not None:
            idx = np.asarray(indices, dtype=int).ravel()
            # keep only those indices that also satisfy time mask
            sel_set = set(sel.tolist())
            sel = np.array([i for i in idx.tolist() if i in sel_set], dtype=int)

        if sel.size == 0:
            raise ValueError("Selection is empty: no waveforms match the requested criteria.")

        # --- apply selection ---
        self.frame.waveform_data = y[sel, :]
        t_sel = t[sel]
        if reset_t0:
            t_sel = t_sel - t_sel[0]
        self.metadata["time_ax_acquisition"] = t_sel
        self.metadata["number_of_waveforms"] = int(self.frame.waveform_data.shape[0])

        return self

    # --------- spectra ----------
    def compute_amplitude_phase_spectrum(
        self,
        *,
        remove_mean: bool = False,
        taper_fraction: float = 0.0,
        nfft: Optional[int] = None,
    ) -> tuple:
        """Return unnormalized |rFFT| and phase, with frequency in MHz.

        Defaults preserve the original transform. ``taper_fraction`` is the
        total Tukey fraction (alpha), shared by both ends, not per end.
        Zero padding interpolates the spectrum; it adds no spectral resolution.
        Neither peak normalization nor energy normalization is performed.

        Calls without processing options execute the original implementation,
        including its metadata handling, to preserve existing script behavior.
        """
        if not remove_mean and taper_fraction == 0.0 and nfft is None:
            # Keep the original no-argument implementation verbatim. Older
            # callers may crop data without updating time_ax_waveform; this
            # path historically needed only the sampling interval.
            if self.waveform_data.size == 0:
                raise ValueError("No waveform data is present to compute spectra.")
            data_2d = self.waveform_data
            n_samples = data_2d.shape[-1]
            if "sampling_rate" not in self.metadata:
                raise KeyError("metadata does not contain 'sampling_rate' key.")
            dt = float(self.metadata["sampling_rate"])
            self.frequencies = np.fft.rfftfreq(n_samples, d=dt)
            try:
                fft_data = np.fft.rfft(data_2d, axis=1)
            except Exception:
                fft_data = np.fft.rfft(data_2d)
            self.amplitude_spectrum = np.abs(fft_data)
            self.phase_spectrum = np.angle(fft_data)
            return self.frequencies, self.amplitude_spectrum, self.phase_spectrum

        if self.waveform_data.size == 0:
            raise ValueError("No waveform data is present to compute spectra.")
        data_2d = np.asarray(self.waveform_data, dtype=float)
        n_samples = data_2d.shape[-1]
        if "sampling_rate" not in self.metadata:
            raise KeyError("metadata does not contain 'sampling_rate' key.")
        dt = float(self.metadata["sampling_rate"])        # µs/sample
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("sampling_rate must be a finite positive interval in us.")
        if not 0.0 <= taper_fraction <= 1.0:
            raise ValueError("taper_fraction must lie in [0, 1].")
        if nfft is None:
            nfft = n_samples
        if not isinstance(nfft, (int, np.integer)) or nfft < n_samples:
            raise ValueError("nfft must be an integer >= the number of samples.")
        if "time_ax_waveform" in self.metadata:
            time = np.asarray(self.metadata["time_ax_waveform"], dtype=float)
            if time.shape != (n_samples,) or not np.isfinite(time).all():
                raise ValueError("time_ax_waveform must match the samples and be finite.")
            if not np.allclose(np.diff(time), dt, rtol=1e-5, atol=1e-10):
                raise ValueError("FFT requires uniform trace times consistent with sampling_rate.")
        if remove_mean:
            data_2d = data_2d - data_2d.mean(axis=-1, keepdims=True)
        if taper_fraction:
            from scipy.signal.windows import tukey
            data_2d = data_2d * tukey(n_samples, alpha=taper_fraction)
        self.frequencies = np.fft.rfftfreq(nfft, d=dt)
        fft_data = np.fft.rfft(data_2d, n=nfft, axis=-1)
        self.amplitude_spectrum = np.abs(fft_data)
        self.phase_spectrum = np.angle(fft_data)
        return self.frequencies, self.amplitude_spectrum, self.phase_spectrum

        
#########s######################################################################
# CLASS: MechanicalDataHandler
###############################################################################
class MechanicalDataHandler:
    """
    Handles mechanical data operations including preprocessing and synchronization signal extraction.
    """
    def __init__(self, mech_data: pd.DataFrame, metadata: dict):
        self.mech_data = mech_data
        self.metadata = metadata

    @classmethod
    def load_mechanical_data(cls, infile_path: Path) -> "MechanicalDataHandler":
        try:
            mech_data = pd.read_csv(infile_path, engine="python", sep=None, skiprows=[1])
            metadata = {"file_path": Path(infile_path)}
            logger.info(f"Mechanical data loaded successfully from {infile_path}.")
            return cls(mech_data=mech_data, metadata=metadata)
        except Exception as e:
            logger.error(f"Error loading mechanical data from {infile_path}: {e}")
            raise

    # NEW: simple resolver by folder + file name (no DirectoryManager)
    @classmethod
    def find_and_load(
        cls,
        folder: Path,
        filename: str,
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Load mechanical CSV from an explicit folder/filename and return
        (mech_data, sync_data, sync_peaks).
        """
        path = Path(folder) / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        handler = cls.load_mechanical_data(path)
        sync_data, sync_peaks = handler.find_sync_values()
        return handler.mech_data, sync_data, sync_peaks

    def find_sync_values(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            sync_data = self.mech_data['sync'].values
            sync_peaks, _ = find_peaks(sync_data, prominence=4.2, height=4)
            logger.info(f"Found {len(sync_peaks)} synchronization peaks.")
            return sync_data, sync_peaks
        except KeyError:
            logger.error("Synchronization column 'sync' is missing in the mechanical data.")
            return None, None

class BlockMetadataHandler:
    """
    Loads geometry and velocity metadata for blocks (side blocks, central block, etc.)
    from a JSON file (e.g., blocks_metadata.json).
    """

    def __init__(self, metadata_dict: dict):
        """
        Initialize the BlockMetadataHandler with a dictionary of block metadata.

        Parameters
        ----------
        metadata_dict : dict
            Dictionary loaded from JSON, containing multiple block entries
            (e.g., 'central_block1', 'mauro_side1', etc.).
        """
        self._metadata_dict = metadata_dict

    @classmethod
    def from_json(cls, config_path: Path) -> "BlockMetadataHandler":
        """
        Create a BlockMetadataHandler instance by loading from a JSON file.

        Parameters
        ----------
        config_path : Path
            Path to the blocks_metadata.json file.

        Returns
        -------
        BlockMetadataHandler
            A handler instance containing all blocks' metadata.
        """
        try:
            with config_path.open("r") as f:
                raw_data = json.load(f)
            return cls(raw_data)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load block metadata from {config_path}: {e}")
            raise

    def get_block_params(self, block_key: str) -> dict:
        """
        Retrieve parameters for a specific block (e.g., 'mauro_side1').

        Parameters
        ----------
        block_key : str
            Key identifying the block in the loaded dictionary (e.g., 'pignalberi_side1').

        Returns
        -------
        dict
            Dictionary of parameters for that block.

        Raises
        ------
        KeyError
            If the specified block_key is not found.
        """
        if block_key not in self._metadata_dict:
            raise KeyError(f"Block '{block_key}' not found in metadata.")
        return self._metadata_dict[block_key]

    def rename_block_key(self, old_key: str, new_key: str) -> None:
        """
        Rename a block key in the internal metadata dictionary.
        
        Parameters
        ----------
        old_key : str
            Existing block key to rename.
        new_key : str
            New block key name to use.

        Raises
        ------
        KeyError
            If old_key is not found.
        """
        if old_key not in self._metadata_dict:
            raise KeyError(f"Block '{old_key}' not found in metadata.")
        # Move the data under 'old_key' to 'new_key' and remove 'old_key'
        self._metadata_dict[new_key] = self._metadata_dict.pop(old_key)

    def update_block_params(self, block_key: str, updates: dict) -> None:
        """
        Update (or add) parameters for a specific block in the dictionary.
        
        Parameters
        ----------
        block_key : str
            The block to be updated (e.g., 'mauro_side1').
        updates : dict
            A dictionary of key-value pairs to be merged into that block's metadata.
            If a key doesn't exist, it will be created; if it does exist, it will be overwritten.
        """
        if block_key not in self._metadata_dict:
            # Optionally raise an error instead of creating a new block:
            # raise KeyError(f"Block '{block_key}' not found, cannot update.")
            logger.info(f"Block '{block_key}' not found; creating a new block entry.")
            self._metadata_dict[block_key] = {}

        for k, v in updates.items():
            self._metadata_dict[block_key][k] = v

    def save_blocks_metadata(self, config_path: Path) -> None:
        """
        Save the internal metadata dictionary to a JSON file.

        Parameters
        ----------
        config_path : Path
            File path at which to save the JSON data.
        """
        try:
            with config_path.open("w") as f:
                json.dump(self._metadata_dict, f, indent=2)
            logger.info(f"Block metadata successfully saved to {config_path}")
        except OSError as e:
            logger.error(f"Failed to write block metadata to {config_path}: {e}")
            raise

    @classmethod
    def load_blocks_metadata(
        cls,
        config_path: Path,
        block_keys: Tuple[str, ...]
    ) -> Tuple[dict, ...]:
        """
        Path-first loader: pass the full JSON path (e.g., base/metadata/blocks_metadata.json)
        and the block keys to extract.
        """
        handler = cls.from_json(Path(config_path))
        return tuple(handler.get_block_params(bk) for bk in block_keys) 
