from pathlib import Path

import numpy as np
import pytest

from lab_uw.data_io.readers.tsv_reader import read_tsv
from lab_uw.data_io.schema import MultiChannelUWFrame, UWFrame


def _header(channel, samples=3, end=0.03):
    return [
        f"Bscan Image from channel:{channel} (1 line => 1 Ascan), generated from EuroscanV",
        "[Amplitude Scale] | Gain=15dB | min:-100%=-2048 | max:100%=2047 | No value=-32640",
        f"[Time Scale] | offset=0µs | scale=100µs | sample={samples} | sampling=0.04MHz",
        f"[Axis Scale] | start=0 | end={end} | step=0.01",
    ]


def _write_tsv(tmp_path: Path, lines):
    path = tmp_path / "acquisition.bscan.tsv"
    path.write_text("\n".join(lines) + "\n", encoding="iso8859")
    return path


def test_single_channel_keeps_backward_compatible_frame(tmp_path):
    path = _write_tsv(tmp_path, _header("Ch:1") + ["1\t2\t3", "4\t5\t6", "7\t8\t9"])

    frame = read_tsv(path)

    assert isinstance(frame, UWFrame)
    assert frame.metadata["channel_name"] == "Ch:1"
    assert frame.metadata["is_complete"] is True
    np.testing.assert_array_equal(frame.waveform_data, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_multichannel_and_normal_early_stop_markers(tmp_path):
    lines = (
        _header("Ch:1")
        + ["1\t2\t3", "4\t5\t6", "Empty line at 2 index"]
        + [""]
        + _header("Ch:2")
        + ["7\t8\t9", "10\t11\t12", "13\t14\t15"]
    )
    frame = read_tsv(_write_tsv(tmp_path, lines))

    assert isinstance(frame, MultiChannelUWFrame)
    assert tuple(frame.channels) == ("Ch:1", "Ch:2")
    assert frame["Ch:1"].metadata["empty_line_indices"] == (2,)
    assert frame["Ch:1"].metadata["acquisition_stopped_early"] is True
    assert frame["Ch:1"].waveform_data.shape == (2, 3)
    assert frame["Ch:2"].metadata["is_complete"] is True


def test_rejects_waveform_with_wrong_sample_count(tmp_path):
    path = _write_tsv(tmp_path, _header("Ch:1") + ["1\t2"])

    with pytest.raises(ValueError, match="expected 3 samples"):
        read_tsv(path)
