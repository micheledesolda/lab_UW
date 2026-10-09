# lab_uw/data_io/readers/tsv_reader.py
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from lab_uw.data_io.schema import MultiChannelUWFrame, UWFrame

_CHANNEL_HEADER = re.compile(r"^Bscan Image from channel\s*:\s*([^\s(]+)", re.IGNORECASE)
_EMPTY_LINE = re.compile( r"empty\s+line\s+at\s+(?P<index>\d+)\s+index",re.IGNORECASE,)

def _extract_metadata_from_tsv_header(lines: List[str]) -> Tuple[list, list, str]:
    general, amplitude_scale, time_scale, acquisition_scale = lines[:4]
    match = _CHANNEL_HEADER.match(general)
    if match is None:
        raise ValueError(f"Invalid B-scan channel header: {general!r}")
    channel_name = match.group(1)
    acquisition_info = [float(e) for e in re.findall(r"\d+\.*\d*", acquisition_scale)]
    time_info       = [float(e) for e in re.findall(r"\d+\.*\d*", time_scale)]
    return acquisition_info, time_info, channel_name

def _frame_from_block(header: List[str], rows: List[List[float]],
                      empty_line_number: int = None) -> UWFrame:
    acquisition_info, time_info, channel_name = _extract_metadata_from_tsv_header(header)
    if len(time_info) < 4 or len(acquisition_info) < 3:
        raise ValueError(f"Incomplete metadata for channel {channel_name!r}.")

    n_samples = int(time_info[2])
    dt_us = float(time_info[3])
    t0 = float(time_info[0])
    acq_t0, acq_t1 = float(acquisition_info[0]), float(acquisition_info[1])
    acq_period_us = float(acquisition_info[2])

    bad_rows = [i for i, row in enumerate(rows) if len(row) != n_samples]
    if bad_rows:
        first = bad_rows[0]
        raise ValueError(
            f"Channel {channel_name!r}, waveform {first}: expected {n_samples} "
            f"samples, found {len(rows[first])}."
        )

    data = np.asarray(rows, dtype=float)
    if not rows:
        data = np.empty((0, n_samples), dtype=float)
    n_wave = data.shape[0]
    expected_n_wave = len(np.arange(acq_t0, acq_t1, acq_period_us))

    meta = {
        "number_of_samples": n_samples,
        "sampling_rate": dt_us,
        "time_ax_waveform": t0 + np.arange(n_samples) * dt_us,
        "acquisition_frequency": acq_period_us,
        "number_of_waveforms": n_wave,
        "expected_number_of_waveforms": expected_n_wave,
        "time_ax_acquisition": acq_t0 + np.arange(n_wave) * acq_period_us,
        "channel_name": channel_name,
        "is_complete": empty_line_number is None and n_wave == expected_n_wave,
        "empty_line_number": empty_line_number,
        "raw_header": tuple(header),
    }
    return UWFrame(data, meta).ensure_2d()


def read_tsv(infile_path: Path) -> Union[UWFrame, MultiChannelUWFrame]:
    """Read every channel block from a Mistras ``.bscan.tsv`` file.

    A one-channel file returns ``UWFrame`` for backward compatibility. A file
    with multiple channel headers returns ``MultiChannelUWFrame``.
    """
    frames: Dict[str, UWFrame] = {}
    header: List[str] = []
    rows: List[List[float]] = []
    empty_line_number = None

    def finish_block() -> None:
        nonlocal header, rows, empty_line_number
        if not header:
            return
        frame = _frame_from_block(header, rows, empty_line_number)
        name = frame.metadata["channel_name"]
        if name in frames:
            raise ValueError(f"Duplicate channel name {name!r} in {infile_path}.")
        frames[name] = frame
        header, rows, empty_line_number = [], [], None

    with open(infile_path, "r", encoding="iso8859") as f:
        for line_number, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue

            if _CHANNEL_HEADER.match(s):
                finish_block()
                header = [s]
                continue

            if header and len(header) < 4:
                header.append(s)
                continue

            marker = _EMPTY_LINE.search(s)
            if marker:
                if not header:
                    raise ValueError(f"Orphan empty-line marker at file line {line_number}.")
                empty_line_number = int(marker.group(1))
                continue

            if not header:
                raise ValueError(f"Expected a channel header at file line {line_number}: {s!r}")

            try:
                rows.append([float(v) for v in s.split()])
            except ValueError as exc:
                raise ValueError(
                    f"Unexpected text in channel data at file line {line_number}: {s!r}"
                ) from exc

    finish_block()
    if not frames:
        raise ValueError(f"No B-scan channel blocks found in {infile_path}.")
    if len(frames) == 1:
        return next(iter(frames.values()))
    return MultiChannelUWFrame(
        frames,
        metadata={"source_path": str(infile_path), "channel_names": tuple(frames)},
    )