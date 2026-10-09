# lab_uw/visual_ispection_standard.py

import sys
import time as tm
import numpy as np
import scipy.signal
from lab_uw.data_io.data_io import (
    MultiChannelUltrasonicDataHandler,
    UltrasonicDataHandler,
)
from lab_uw.directory_manager import DirectoryManager
from lab_uw.plotting import Plotter
import scipy

###### INPUT #######
base_dir = "/home/michele/Desktop/Dottorato/active_source_implementation"
machine_name = "big_biax"
experiment_name = "r0072suw03anhdol_6035"
data_type = "uw_data/data_tsv_files_p"

### The plots we want to see for a quick inspection of the experiment ###
image_types = (
    "stacked_waveforms",
    "amplitude_maps",
    "amplitude_spectrum_maps",
    "example_of_filtered_spectrum",
    "example_of_filtered_waveform",
)
remove_initial_samples = 0  # number of samples to be removed at the beginning, to get rid of the noise burst.
highlight_start = 0
highlight_end = 0

ticks_steps_waveforms = 5  # [microseconds] plot ticks
step_wf_to_plot = 100   # get one waveform each step_wf_to_plot
freq_min = None         #  [MHz] low and high frequency threshold
freq_max = 12.  
maxtime_us = 25
xlim_plot = maxtime_us
# Create instances of Plotter, DataHandler, and DirectoryManager
plotter = Plotter()
directory_manager = DirectoryManager(base_dir=base_dir)

infile_path_list = directory_manager.make_infile_path_list(
    machine_name, experiment_name, data_type=data_type
)
outdir_path_images = directory_manager.make_images_folders(
    machine_name, experiment_name, image_types
)

def iter_channels(uw):
    """Yield ``(channel_name, handler)`` for single- and multichannel input."""
    if isinstance(uw, MultiChannelUltrasonicDataHandler):
        yield from uw.items()
    else:
        yield uw.metadata.get("channel_name", "channel"), uw


def process_channel(uw, infile_path, channel_name):
    print(f"  processing channel {channel_name}")

    data, metadata = uw.waveform_data, uw.metadata
    if remove_initial_samples:
        data = data[:, remove_initial_samples:]
        metadata = dict(metadata)
        metadata['number_of_samples'] -= remove_initial_samples
        metadata['time_ax_waveform'] = metadata['time_ax_waveform'][remove_initial_samples:]

    taper_percent = 0.1
    window_scipy = scipy.signal.windows.tukey(
        metadata['number_of_samples'],
        alpha=2 * taper_percent
    )
    data = data * window_scipy

    uw = UltrasonicDataHandler(waveform_data=data, metadata=metadata).truncate_time(maxtime_us=maxtime_us)
    uw_filt = UltrasonicDataHandler(
        waveform_data=uw.waveform_data.copy(),
        metadata=dict(uw.metadata)
    ).bandpass(freqmin=freq_min,
               freqmax=freq_max
               )

    outfile_name = infile_path
    while outfile_name.suffix:
        outfile_name = outfile_name.with_suffix("")
    safe_channel_name = channel_name.replace(":", "_").replace(" ", "_")
    outfile_name = f"{outfile_name.name}_{safe_channel_name}"

    outfile_path_stacked_waveforms = outdir_path_images[0] / outfile_name
    outfile_path_amp_map = outdir_path_images[1] / outfile_name
    outfile_path_spectrum_map = outdir_path_images[2] / outfile_name

    start_time = tm.time()

    plotter.uw_all_plot(
        data=uw_filt.waveform_data,
        metadata=uw_filt.metadata,
        step_wf_to_plot=step_wf_to_plot,
        highlight_start=highlight_start,
        highlight_end=highlight_end,
        xlim_plot=xlim_plot,
        ticks_steps_waveforms=ticks_steps_waveforms,
        outfile_path=str(outfile_path_stacked_waveforms),
    )

    plotter.amplitude_map(
        data=uw_filt.waveform_data,
        metadata=uw_filt.metadata,
        outfile_path=str(outfile_path_amp_map),
    )

    freqs_filt, amp_filt, _ = uw_filt.compute_amplitude_phase_spectrum()

    plotter.amplitude_spectrum_map(
        signal_freqs=freqs_filt,
        amp_spectrum=amp_filt,
        metadata=uw_filt.metadata,
        freq_max=freq_max,
        outfile_path=str(outfile_path_spectrum_map),
    )

    plotter.amplitude_spectrum_distribution(
        signal_freqs=freqs_filt,
        amp_spectrum=amp_filt,
        metadata=uw_filt.metadata,
        freq_max=freq_max,
        outfile_path=str(outfile_path_spectrum_map) + "_distribution",
    )

    wave_chosed_list = range(0, uw.metadata['number_of_waveforms'], step_wf_to_plot)

    for wave_chosed in wave_chosed_list:
        outfile_path_example_filtered_spectrum = outdir_path_images[3] / f"{outfile_name}_{wave_chosed}"
        outfile_path_example_original_vs_filtered = outdir_path_images[4] / f"{outfile_name}_{wave_chosed}"

        print(f"wave_chosed: {wave_chosed}")
        print(f"Filtered spectrum path: {outfile_path_example_filtered_spectrum}")
        print(f"Original vs filtered path: {outfile_path_example_original_vs_filtered}")

        single_uw = UltrasonicDataHandler(
            waveform_data=uw.waveform_data[wave_chosed, :],
            metadata=dict(uw.metadata)
        )

        single_uw_filt = UltrasonicDataHandler(
            waveform_data=single_uw.waveform_data.copy(),
            metadata=dict(single_uw.metadata)
        ).bandpass(freqmin=freq_min,
               freqmax=freq_max
               )

        freqs_single, amp_single, phase_single = single_uw.compute_amplitude_phase_spectrum()
        _, amp_single_filt, _ = single_uw_filt.compute_amplitude_phase_spectrum()

        #################################
        # it is the exact filter used in compute_amplitude_phase_spectrum but I need it here for the plot. Ugly fix
        from scipy.signal import butter, sosfreqz
        dt_us = float(single_uw.metadata["sampling_rate"])
        fs_mhz = 1.0 / dt_us
        sos = butter(4, freq_max, btype="lowpass", fs=fs_mhz, output="sos")
        _, h = sosfreqz(sos, worN=freqs_single, fs=fs_mhz)
        filter_shape = np.abs(h)
        ############################

        amp_single = amp_single[0]
        phase_single = phase_single[0]
        amp_single_filt = amp_single_filt[0]

        phase_plot = phase_single.copy()
        phase_plot[1:][np.abs(np.diff(phase_plot)) > np.pi] = np.nan

        plotter.filtered_amp_and_phase_spectrum_plot(
            signal_freqs=freqs_single,
            amp_spectrum=amp_single,
            phase_spectrum=phase_plot,
            filtered_amp_spectrum=amp_single_filt,
            lowpass_filter=filter_shape,
            freq_max=freq_max,
            outfile_path=outfile_path_example_filtered_spectrum,
        )

        plotter.signal_vs_filtered_signal_plot(
            single_waveform=single_uw.waveform_data[0],
            single_waveform_filtered=single_uw_filt.waveform_data[0],
            metadata=single_uw.metadata,
            freq_max=freq_max,
            outfile_path=outfile_path_example_original_vs_filtered,
        )

    print(f"--- {tm.time() - start_time} seconds for processing {outfile_name} ---")


for infile_path in sorted(infile_path_list):
    try:
        print(f'PROCESSING UW DATA IN {infile_path}: ')

        acquisition = UltrasonicDataHandler.read_tsv(infile_path)
        for channel_name, uw in iter_channels(acquisition):
            process_channel(uw, infile_path, channel_name)

    except Exception as e:
        print(f"An error occurred while processing {infile_path}: {e}")
        pass