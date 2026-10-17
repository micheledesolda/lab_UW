"""Thesis figures: full-record examples and direct-arrival evolution.

Run with the existing project environment:
    poetry run python visual_ispection_standard.py
All trace times below are absolute times after the acquisition trigger, in us.
"""

from pathlib import Path
import traceback
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import gc
import mmap
import shutil

import numpy as np
from scipy.signal import butter, sosfreqz
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import tukey

# This batch script only exports files; avoid GUI/inline figure retention.
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lab_uw.data_io.data_io import MultiChannelUltrasonicDataHandler, UltrasonicDataHandler
from lab_uw.directory_manager import DirectoryManager
from lab_uw.plotting import Plotter

MECHANICAL_INSPECTION_API = 1


# ---------------- USER CONFIGURATION ----------------
base_dir = "/home/michele/Desktop/Dottorato/active_source_implementation"
machine_name = "big_biax"
experiment_name = "r0072suw03anhdol_6035"
data_type = "uw_data/data_tsv_files_s"

# IMPORTANT: replace these illustrative limits with those in the inversion.
# This single window controls every direct-arrival figure AND its FFT input.
direct_window_us = (20.0, 25.0)
direct_window_overrides_us = {
    # "001_runin_P.bscan.tsv": (10.0, 16.0),
    # ("001_runin_P.bscan.tsv", "ch2"): (10.5, 15.5),
}

number_of_example_waveforms = 3   # first, middle, last; per file and channel
example_indices = None           # optional explicit indices, e.g. [0, 1200, 2400]
number_of_overlay_waveforms = 12

freq_min = None                   # MHz; None for a low-pass filter
freq_max = 12.0                   # MHz; None for a high-pass filter
filter_order = 4
spectrum_max_mhz = 12.0           # plot limit; None displays the Nyquist range
show_filter_response = False      # optional dimensionless secondary axis
show_phase_spectrum = False
spectral_remove_mean = True
spectral_taper_fraction = 0.10     # total Tukey alpha, as in spectral_diagnostics.py
spectral_zero_padding_factor = 4  # interpolates FFT, without adding resolution

# Additional numeric workspace is bounded; the reader's input array is unchanged.
processing_memory_mb = 32
processing_block_rows = 128
map_max_columns = 1600
map_max_rows = 1024
cache_directory = None  # None uses a temporary folder beside the output images

# Unit of time_ax_acquisition as supplied by YOUR reader. Existing plots
# treated it as seconds. If the reader supplies microseconds, set "us" here.
# Explicit metadata['time_ax_acquisition_unit'], when present, takes priority.
acquisition_time_unit = "s"        # allowed values: "s", "ms", "us"

save_pdf = True
plot_settings = {
    "figure_size": (12, 7),
    "fontsize_title": 32,
    "fontsize_labels": 28,
    "fontsize_ticks": 22,
    "fontsize_legend": 20,
    "line_width": 1.8,
    "format": ".png",
    "additional_formats": (".pdf",) if save_pdf else (),
    "dpi": 300,
    "pdf_raster_dpi": 150,
    "spectrum_dynamic_range_decades": 4.0,
    "processing_memory_mb": processing_memory_mb,
    "map_max_columns": map_max_columns,
    "map_max_rows": map_max_rows,
}
# ---------------- END CONFIGURATION ----------------

image_types = (
    "full_record_examples",
    "direct_waveforms",
    "direct_amplitude_maps",
    "direct_spectrum_maps",
    "direct_spectrum_distributions",
    "direct_spectrum_examples",
    "direct_waveform_differences",
    "direct_spectral_differences",
    "direct_difference_spectra",
)


def iter_channels(acquisition):
    if isinstance(acquisition, MultiChannelUltrasonicDataHandler):
        yield from acquisition.items()
    else:
        yield acquisition.metadata.get("channel_name", "channel"), acquisition


def _output_stem(infile_path):
    stem = Path(infile_path)
    while stem.suffix:
        stem = stem.with_suffix("")
    return stem.name


def resolve_direct_window(infile_path, channel_name):
    path = Path(infile_path)
    for key in ((path.name, channel_name), (_output_stem(path), channel_name),
                path.name, _output_stem(path)):
        if key in direct_window_overrides_us:
            return tuple(direct_window_overrides_us[key])
    return tuple(direct_window_us)


def choose_examples(number_of_waveforms):
    if example_indices is None:
        if not isinstance(number_of_example_waveforms, int) or number_of_example_waveforms < 1:
            raise ValueError("number_of_example_waveforms must be a positive integer.")
        return np.unique(np.linspace(0, number_of_waveforms - 1,
                                     min(number_of_example_waveforms, number_of_waveforms), dtype=int))
    indices = np.asarray(example_indices)
    if indices.ndim != 1 or not indices.size or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("example_indices must be a non-empty list of integer indices.")
    if np.any(indices < 0) or np.any(indices >= number_of_waveforms):
        raise ValueError(f"example_indices must lie between 0 and {number_of_waveforms - 1}.")
    return np.unique(indices)


def prepare_full_record(uw):
    data = np.asarray(uw.waveform_data)
    metadata = dict(uw.metadata)
    time = np.asarray(metadata["time_ax_waveform"], dtype=float)
    if data.ndim != 2 or not data.size or time.shape != (data.shape[1],):
        raise ValueError("Expected non-empty (waveforms, samples) data with matching trace times.")
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Data must be finite and trace times strictly increasing.")
    dt_us = float(metadata["sampling_rate"])
    if not np.isfinite(dt_us) or dt_us <= 0 or not np.allclose(np.diff(time), dt_us, rtol=1e-5, atol=1e-10):
        raise ValueError("Trace times must be uniformly sampled at metadata['sampling_rate'] us.")
    times = np.asarray(metadata["time_ax_acquisition"], dtype=float)
    if times.shape != (data.shape[0],) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Acquisition times must match waveform rows and strictly increase.")
    unit = metadata.get("time_ax_acquisition_unit", acquisition_time_unit)
    scales = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
    if unit not in scales:
        raise ValueError("Specify acquisition_time_unit as 's', 'ms', or 'us'.")
    metadata.update(
        number_of_samples=data.shape[1], number_of_waveforms=data.shape[0],
        time_ax_waveform=time, time_ax_acquisition=times * scales[unit],
        time_ax_acquisition_unit="s",
    )
    return UltrasonicDataHandler(data, metadata)


def filter_response(frequency_mhz, dt_us):
    if freq_min is None and freq_max is None:
        return np.ones_like(frequency_mhz)
    if freq_min is not None and freq_max is not None:
        cutoff, kind = [freq_min, freq_max], "bandpass"
    elif freq_min is not None:
        cutoff, kind = freq_min, "highpass"
    else:
        cutoff, kind = freq_max, "lowpass"
    sos = butter(filter_order, cutoff, btype=kind, fs=1.0 / dt_us, output="sos")
    _, transfer = sosfreqz(sos, worN=frequency_mhz, fs=1.0 / dt_us)
    return np.abs(transfer) ** 2   # forward/backward zero-phase amplitude response


def _block_rows(full_samples, nfft):
    if not np.isfinite(processing_memory_mb) or processing_memory_mb <= 0:
        raise ValueError('processing_memory_mb must be positive.')
    if not isinstance(processing_block_rows, int) or processing_block_rows < 1:
        raise ValueError('processing_block_rows must be a positive integer.')
    per_row = max(full_samples * 8 * 8, nfft * 16 * 5)
    return max(1, min(processing_block_rows, int(processing_memory_mb * 1024**2 // per_row)))


def _release_disk_arrays(arrays, *, flush=True):
    for array in arrays:
        if flush:
            array.flush()
        mapping = getattr(array, '_mmap', None)
        if mapping is not None and hasattr(mapping, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
            try:
                mapping.madvise(mmap.MADV_DONTNEED)
            except (OSError, ValueError):
                pass


@contextmanager
def channel_products(full, window, examples, cache_parent, *, already_preprocessed=False):
    """Filter complete records in bounded blocks; retain derived arrays on disk.

    No full filtered record, complex FFT, phase matrix, or difference matrix
    spanning all acquisitions is allocated in RAM. Float64 disk arrays retain
    the original numerical precision. Only selected examples keep full records.
    """
    direct = UltrasonicDataHandler(full.waveform_data, dict(full.metadata)).select_waveform_window(*window)
    time = full.metadata['time_ax_waveform']
    start_sample = int(np.searchsorted(time, direct.metadata['time_ax_waveform'][0]))
    samples = direct.waveform_data.shape[1]
    sample_slice = slice(start_sample, start_sample + samples)
    count, full_samples = full.waveform_data.shape
    if not isinstance(spectral_zero_padding_factor, int) or spectral_zero_padding_factor < 1:
        raise ValueError('spectral_zero_padding_factor must be a positive integer.')
    if not 0 <= spectral_taper_fraction <= 1:
        raise ValueError('spectral_taper_fraction must lie in [0, 1].')
    nfft = max(256, spectral_zero_padding_factor * 2 ** int(np.ceil(np.log2(samples))))
    all_frequency = rfftfreq(nfft, d=float(full.metadata['sampling_rate']))
    frequency_limit = all_frequency[-1] if spectrum_max_mhz is None else min(float(spectrum_max_mhz), all_frequency[-1])
    if not frequency_limit > 0:
        raise ValueError('spectrum_max_mhz must be positive.')
    nfreq = int(np.searchsorted(all_frequency, frequency_limit, side='right'))
    frequency = all_frequency[:nfreq]
    rows = _block_rows(full_samples, nfft)
    taper = tukey(samples, alpha=spectral_taper_fraction) if spectral_taper_fraction else None

    def transform(block):
        work = np.array(block, dtype=np.float64, copy=True, order='C')
        if spectral_remove_mean:
            work -= work.mean(axis=-1, keepdims=True)
        if taper is not None:
            work *= taper
        return rfft(work, n=nfft, axis=-1, workers=1)[:, :nfreq]

    cache_parent = Path(cache_parent)
    cache_parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    with TemporaryDirectory(prefix='uw_plots_', dir=cache_parent) as temporary:
        def disk_array(name, shape):
            array = np.lib.format.open_memmap(Path(temporary) / (name + '.npy'),
                                              mode='w+', dtype=np.float64, shape=shape)
            arrays.append(array)
            return array

        try:
            filtered = disk_array('direct', (count, samples))
            amplitude = disk_array('amplitude', (count, nfreq))
            changes = disk_array('waveform_changes', (count - 1, samples)) if count > 1 else None
            amplitude_changes = disk_array('amplitude_changes', (count - 1, nfreq)) if count > 1 else None
            change_spectra = disk_array('change_spectra', (count - 1, nfreq)) if count > 1 else None
            example_filtered_full = np.empty((len(examples), full_samples), dtype=np.float64)
            example_rows = {int(index): row for row, index in enumerate(examples)}
            previous_direct = previous_transform = None
            spectral_peak = 0.0
            pending_bytes = 0
            cache_bytes = sum(array.nbytes for array in arrays)
            print(f'    bounded processing: <= {rows} records/block; '
                  f'{cache_bytes / 1024**2:.1f} MiB temporary disk cache', flush=True)
            for first in range(0, count, rows):
                last = min(first + rows, count)
                raw = full.waveform_data[first:last]
                if not np.isfinite(raw).all():
                    raise ValueError(f'Non-finite waveform samples in rows {first}:{last}.')
                block_metadata = dict(full.metadata)
                block_metadata.update(number_of_waveforms=last - first,
                                      time_ax_acquisition=full.metadata['time_ax_acquisition'][first:last])
                # bandpass() is unchanged: rows are independent, so filtering
                # blocks is numerically identical to filtering the whole matrix.
                if already_preprocessed:
                    # These are the exact samples entering the inversion.
                    # Do not filter, align or normalize them again.
                    handler = None
                    complete = raw
                else:
                    handler = UltrasonicDataHandler(raw, block_metadata).bandpass(
                        freqmin=freq_min, freqmax=freq_max, order=filter_order)
                    complete = handler.waveform_data
                block = complete[:, sample_slice]
                filtered[first:last] = block
                for index in examples[(examples >= first) & (examples < last)]:
                    example_filtered_full[example_rows[int(index)]] = complete[index - first]
                transformed = transform(block)
                amp = np.abs(transformed)
                amplitude[first:last] = amp
                spectral_peak = max(spectral_peak, float(amp.max()))
                if count > 1:
                    if previous_direct is None:
                        joined_direct = block
                        joined_transform = transformed
                        pair_first = first
                    else:
                        joined_direct = np.concatenate((previous_direct, block), axis=0)
                        joined_transform = np.concatenate((previous_transform, transformed), axis=0)
                        pair_first = first - 1
                    pairs = joined_direct.shape[0] - 1
                    if pairs:
                        pair_slice = slice(pair_first, pair_first + pairs)
                        changes[pair_slice] = np.diff(joined_direct, axis=0)
                        amplitude_changes[pair_slice] = np.diff(np.abs(joined_transform), axis=0)
                        # FFT, mean removal and taper are linear. This equals
                        # abs(FFT(u[i+1]-u[i])) without a second global FFT.
                        change_spectra[pair_slice] = np.abs(np.diff(joined_transform, axis=0))
                    previous_direct = np.array(block[-1:], copy=True)
                    previous_transform = np.array(transformed[-1:], copy=True)
                    del joined_direct, joined_transform
                pending_bytes += (last - first) * (2 * samples + 3 * nfreq) * 8
                if pending_bytes >= processing_memory_mb * 1024**2:
                    _release_disk_arrays(arrays)
                    Plotter._release_mapped_pages(full.waveform_data)
                    pending_bytes = 0
                del handler, complete, block, transformed, amp, raw
            _release_disk_arrays(arrays)
            metadata = dict(direct.metadata)
            metadata.update({key: value for key, value in block_metadata.items() if key.startswith('filter_')})
            difference_metadata = dict(metadata)
            times = metadata['time_ax_acquisition']
            difference_metadata.update(
                number_of_waveforms=max(0, count - 1),
                time_ax_acquisition=times[:-1] + .5 * np.diff(times),
                difference_definition='next_minus_current', difference_time_reference='pair_midpoint')
            example_original_full = np.array(full.waveform_data[examples], copy=True)
            example_direct = example_original_full[:, sample_slice]
            example_transform = transform(example_direct)
            example_amplitude = np.abs(example_transform)
            spectral_peak = max(spectral_peak, float(example_amplitude.max())) or 1.0
            yield dict(filtered=filtered, amplitude=amplitude, changes=changes,
                       amplitude_changes=amplitude_changes, change_spectra=change_spectra,
                       frequency=frequency, frequency_limit=frequency_limit,
                       metadata=metadata, difference_metadata=difference_metadata,
                       example_original_full=example_original_full,
                       example_filtered_full=example_filtered_full,
                       example_direct=example_direct, example_filtered_direct=example_filtered_full[:, sample_slice],
                       example_amplitude=example_amplitude,
                       example_phase=np.angle(example_transform) if show_phase_spectrum else None,
                       spectral_peak=spectral_peak, block_rows=rows, cache_bytes=cache_bytes,
                       cache_path=temporary)
        finally:
            for array in arrays:
                try:
                    array.flush()
                finally:
                    array._mmap.close()
            arrays.clear()


def _plot_inversion_waveform(plotter, single_waveform, single_waveform_filtered,
                             metadata, freq_max=None, outfile_path=None, *,
                             title='', highlight_window=None, amplitude_ylim=None):
    fig, ax = plt.subplots(figsize=plotter.settings['figure_size'], layout='constrained')
    time = metadata['time_ax_waveform']
    ax.plot(time, single_waveform, label="Input dell'inversione",
            linewidth=plotter.settings['line_width'])
    if highlight_window is not None:
        ax.axvspan(*highlight_window, alpha=.2, color=plotter.get_color('highlight'))
    plotter._style_data_axes(ax, title, r'Time after trigger [$\mu$s]',
                             f"Amplitude [{metadata.get('amplitude_unit', 'a.u.')}]")
    ax.set_xlim(time[0], time[-1])
    if amplitude_ylim is not None:ax.set_ylim(amplitude_ylim)
    ax.legend(fontsize=plotter.settings['fontsize_legend'])
    plotter.output_path_choice(fig=fig, outfile_path=outfile_path)


def _plot_inversion_spectrum(plotter, signal_freqs, amp_spectrum,
                             phase_spectrum, filtered_amp_spectrum, lowpass_filter,
                             freq_max, outfile_path=None, *, title='',
                             amplitude_ylim=None, amplitude_unit='a.u.', **kwargs):
    fig, ax = plt.subplots(figsize=plotter.settings['figure_size'], layout='constrained')
    ax.plot(signal_freqs, filtered_amp_spectrum, label="Input dell'inversione",
            linewidth=plotter.settings['line_width'])
    plotter._style_data_axes(ax, title, 'Frequency [MHz]', f'Amplitude [{amplitude_unit}]')
    ax.set_xlim(0, freq_max)
    if amplitude_ylim is not None:ax.set_ylim(amplitude_ylim)
    ax.legend(fontsize=plotter.settings['fontsize_legend'])
    plotter.output_path_choice(fig=fig, outfile_path=outfile_path)


def process_channel(uw, infile_path, channel_name, *, plotter=None, output_dirs=None,
                    already_preprocessed=False, window_us=None, on_products=None):
    # MECHANICAL_SELECTION_INSPECTION_V1: safe to call after a GUI selection.
    initial_figures = set(plt.get_fignums())
    if plotter is None:
        plotter = Plotter(settings=plot_settings)
    if output_dirs is None:
        raise ValueError('output_dirs must map figure categories to output folders.')
    full = prepare_full_record(uw)
    window = (resolve_direct_window(infile_path, channel_name) if window_us is None
              else tuple(window_us))
    count = full.waveform_data.shape[0]
    examples = choose_examples(count)
    safe_channel = str(channel_name).replace(':', '_').replace(' ', '_')
    name = f'{_output_stem(infile_path)}_{safe_channel}'
    window_tag = f'direct_{window[0]:g}-{window[1]:g}us'

    def output(category, suffix=''):
        tag = 'full' if category == 'full_record_examples' else window_tag
        return str(Path(output_dirs[category]) / f'{name}_{tag}{suffix}')

    if not isinstance(number_of_overlay_waveforms, int) or number_of_overlay_waveforms < 1:
        raise ValueError('number_of_overlay_waveforms must be a positive integer.')
    acquisition = full.metadata['time_ax_acquisition']
    print(f'  {channel_name}: {count} waveforms; direct window {window} us; '
          f'examples={examples.tolist()}', flush=True)
    cache_parent = (Path(cache_directory) if cache_directory is not None
                    else Path(output_dirs['direct_waveforms']).parent / 'processing_cache')
    def render(method, *args, **kwargs):
        existing = set(plt.get_fignums())
        try:
            return method(*args, **kwargs)
        finally:
            # Collect AFTER the plot method returns, when its local artist
            # and renderer references are no longer keeping buffers alive.
            for number in set(plt.get_fignums()) - existing:
                plt.close(number)
            gc.collect()

    waveform_plot = (lambda *a, **k: _plot_inversion_waveform(plotter, *a, **k)) if already_preprocessed else plotter.signal_vs_filtered_signal_plot
    spectrum_plot = (lambda *a, **k: _plot_inversion_spectrum(plotter, *a, **k)) if already_preprocessed else plotter.filtered_amp_and_phase_spectrum_plot

    product = None
    temporary_path = None
    try:
        with channel_products(full, window, examples, cache_parent,
                              already_preprocessed=already_preprocessed) as product:
            temporary_path = product['cache_path']
            full_peak = max(float(np.max(np.abs(product['example_original_full']))),
                            float(np.max(np.abs(product['example_filtered_full'])))) or 1.0
            direct_peak = max(float(np.max(np.abs(product['example_direct']))),
                              float(np.max(np.abs(product['example_filtered_direct'])))) or 1.0
            for row, index in enumerate(examples):
                suffix = f'_wave_{index:06d}'
                render(waveform_plot, 
                    single_waveform=product['example_original_full'][row],
                    single_waveform_filtered=product['example_filtered_full'][row], metadata=full.metadata,
                    freq_max=freq_max, outfile_path=output('full_record_examples', suffix),
                    title=f'Full waveform record: acquisition {index}\nExperiment time: {acquisition[index]:.3f} s',
                    highlight_window=window, amplitude_ylim=(-1.08 * full_peak, 1.08 * full_peak))
                render(waveform_plot, 
                    single_waveform=product['example_direct'][row],
                    single_waveform_filtered=product['example_filtered_direct'][row], metadata=product['metadata'],
                    freq_max=freq_max, outfile_path=output('direct_waveforms', suffix),
                    title=f'Direct arrival: acquisition {index}\nExperiment time: {acquisition[index]:.3f} s',
                    amplitude_ylim=(-1.08 * direct_peak, 1.08 * direct_peak))
            render(plotter.uw_all_plot, 
                data=product['filtered'], metadata=product['metadata'], step_wf_to_plot=1,
                highlight_start=0, highlight_end=0, max_waveforms=number_of_overlay_waveforms,
                xlim_plot=window[1], ticks_steps_waveforms=None,
                outfile_path=output('direct_waveforms', '_overlay'))
            render(plotter.amplitude_map, product['filtered'], product['metadata'],
                                  outfile_path=output('direct_amplitude_maps'))
            frequency, limit = product['frequency'], product['frequency_limit']
            render(plotter.amplitude_spectrum_map, frequency, product['amplitude'], product['metadata'], limit,
                                           outfile_path=output('direct_spectrum_maps'))
            render(plotter.amplitude_spectrum_distribution, frequency, product['amplitude'], product['metadata'], limit,
                                                    outfile_path=output('direct_spectrum_distributions'))
            gain = filter_response(frequency, float(full.metadata['sampling_rate'])) if show_filter_response else None
            for row, index in enumerate(examples):
                phase = None
                if show_phase_spectrum:
                    phase = product['example_phase'][row].copy()
                    phase[1:][np.abs(np.diff(phase)) > np.pi] = np.nan
                render(spectrum_plot, 
                    signal_freqs=frequency, amp_spectrum=product['example_amplitude'][row], phase_spectrum=phase,
                    filtered_amp_spectrum=product['amplitude'][index], lowpass_filter=gain, freq_max=limit,
                    show_phase=show_phase_spectrum, show_filter=show_filter_response,
                    amplitude_ylim=(0, 1.05 * product['spectral_peak']),
                    amplitude_unit=full.metadata.get('amplitude_unit', 'a.u.'),
                    title=f'Direct-arrival spectrum: acquisition {index}\nExperiment time: {acquisition[index]:.3f} s',
                    outfile_path=output('direct_spectrum_examples', f'_wave_{index:06d}'))
            if count > 1:
                render(plotter.amplitude_map, product['changes'], product['difference_metadata'], difference=True,
                                      outfile_path=output('direct_waveform_differences'))
                render(plotter.amplitude_spectrum_map, frequency, product['amplitude_changes'], product['difference_metadata'],
                                               limit, difference=True, outfile_path=output('direct_spectral_differences'))
                render(plotter.amplitude_spectrum_map, frequency, product['change_spectra'], product['difference_metadata'],
                                               limit, title='Spectrum of consecutive waveform differences',
                                               outfile_path=output('direct_difference_spectra'))
            else:
                print('    Single waveform: no consecutive-difference maps.')
            if on_products is not None:
                render(on_products, product, plotter)
    finally:
        for number in set(plt.get_fignums()) - initial_figures:
            plt.close(number)
        if product is not None:
            product.clear()
        gc.collect()
        # Perform the final cleanup only after renderer and memmap references
        # have been released. This path belongs exclusively to this channel.
        if temporary_path is not None:
            shutil.rmtree(temporary_path, ignore_errors=True)


def main():
    manager = DirectoryManager(base_dir=base_dir)
    inputs = manager.make_infile_path_list(machine_name, experiment_name, data_type=data_type)
    folders = manager.make_images_folders(machine_name, experiment_name, image_types)
    output_dirs = dict(zip(image_types, folders))
    plotter = Plotter(settings=plot_settings)
    failures = 0
    for infile_path in sorted(inputs):
        acquisition = uw = None
        print(f'PROCESSING UW DATA IN {infile_path}:', flush=True)
        try:
            acquisition = UltrasonicDataHandler.open(Path(infile_path))
            # Release each completed channel before moving to the next one.
            names = list(acquisition.channels) if isinstance(acquisition, MultiChannelUltrasonicDataHandler) else [None]
            for name in names:
                uw = acquisition.channels.pop(name) if name is not None else acquisition
                channel_name = name if name is not None else uw.metadata.get('channel_name', 'channel')
                try:
                    process_channel(uw, infile_path, channel_name, plotter=plotter, output_dirs=output_dirs)
                except Exception:
                    failures += 1
                    traceback.print_exc()
                finally:
                    uw = None
                    gc.collect()
        except Exception:
            failures += 1
            traceback.print_exc()
        finally:
            # Do not retain the previous input while the next reader allocates.
            acquisition = uw = None
            gc.collect()
    if failures:
        raise SystemExit(f'Processing completed with {failures} failed file(s)/channel(s). See tracebacks above.')


if __name__ == '__main__':
    main()
